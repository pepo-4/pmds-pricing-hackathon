"""Hidden market mechanics of the synthetic simulator: demand and auction clearing.

Nothing in this module is visible to agents. They only see what
`environment.Game` puts into their `Observation`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class DemandSpec:
    """Demand for one product. `M0` units per round are sold in total when every
    firm charges the reference price `P0`."""
    kind: str = "linear"            # linear | logit | segments | isoelastic
    P0: float = 50.0
    M0: float = 200.0
    # linear: firm demand = M/N * max(0, 1 - b*(p/P0 - 1) + g*(mean rival p - p)/P0)
    b: float = 1.5
    g: float = 1.5
    # logit: utility (quality - p)/mu, outside option utility 0
    quality: float = 60.0
    mu: float = 10.0
    # segments (after the 2017 Dynamic Pricing Challenge): bargain hunters buy from the
    # cheapest firm, loyal customers from "their" firm, the rest choose by logit
    beta: float = 30.0
    loyal_u: float = 1.7
    share_shop: float = 0.35
    share_loyal: float = 0.3
    # isoelastic: firm demand = M/N * (p/P0)^-eps * (p / mean market price)^-sigma
    eps: float = 1.8
    sigma: float = 2.5
    # randomness and dynamics
    noise: float = 0.10             # idiosyncratic multiplicative noise per firm
    common_noise: float = 0.05      # market-wide multiplicative noise per round
    trend: float = 0.0              # relative change of market size per round
    season_amp: float = 0.0
    season_period: int = 12
    shock_round: Optional[int] = None
    shock_mult: float = 1.0
    elast_round: Optional[int] = None
    elast_mult: float = 1.0
    spill: float = 0.3              # share of unserved demand that tries another firm

    def market_size(self, t: int) -> float:
        m = self.M0 * max(0.1, 1.0 + self.trend * t)
        if self.season_amp:
            m *= 1.0 + self.season_amp * math.sin(2.0 * math.pi * t / max(self.season_period, 2))
        if self.shock_round is not None and t >= self.shock_round:
            m *= self.shock_mult
        return max(m, 0.0)

    def sensitivity(self, t: int) -> float:
        if self.elast_round is not None and t >= self.elast_round:
            return self.elast_mult
        return 1.0


def expected_demand(ds: DemandSpec, t: int, prices: np.ndarray, n_total: int) -> np.ndarray:
    """Expected demand of each active firm (firms with stock on the shelf)."""
    n = len(prices)
    if n == 0:
        return np.zeros(0)
    M = ds.market_size(t)
    e = ds.sensitivity(t)
    p = np.asarray(prices, dtype=float)
    if ds.kind == "linear":
        rel = p / ds.P0
        others = (rel.sum() - rel) / (n - 1) if n > 1 else rel
        return (M / n_total) * np.maximum(0.0, 1.0 - ds.b * e * (rel - 1.0) + ds.g * e * (others - rel))
    if ds.kind == "logit":
        return M * _logit_share(ds, p, n_total, e)
    if ds.kind == "isoelastic":
        rel = np.maximum(p / ds.P0, 1e-6)
        pos = np.maximum(p / max(float(p.mean()), 1e-9), 1e-6)
        return (M / n_total) * rel ** (-ds.eps * e) * pos ** (-ds.sigma * e)
    if ds.kind == "segments":
        beta = ds.beta / e
        d = np.zeros(n)
        # bargain hunters: all go to the cheapest shelf
        pmin = p.min()
        cheapest = np.isclose(p, pmin, rtol=1e-9, atol=1e-12)
        d[cheapest] += M * ds.share_shop * math.exp(-(pmin - ds.P0) / beta) / cheapest.sum()
        # loyal customers: each firm has its own, with a higher willingness to pay
        d += (M * ds.share_loyal / n_total) * np.exp(-(p - ds.P0) / (ds.loyal_u * beta))
        # the rest compare offers
        rest = max(0.0, 1.0 - ds.share_shop - ds.share_loyal)
        d += M * rest * _logit_share(ds, p, n_total, e)
        return d
    raise ValueError(f"unknown demand kind {ds.kind}")


def _logit_share(ds: DemandSpec, p: np.ndarray, n_total: int, e: float) -> np.ndarray:
    mu = ds.mu / e
    ex = np.exp(np.clip((ds.quality - p) / mu, -60.0, 60.0))
    share = ex / (1.0 + ex.sum())
    # normalise so that total sales equal M when all n_total firms charge P0
    ref = math.exp(min(60.0, max(-60.0, (ds.quality - ds.P0) / ds.mu)))
    ref_total = n_total * ref / (1.0 + n_total * ref)
    return share / max(ref_total, 1e-9)


def realise_sales(ds: DemandSpec, t: int, prices: np.ndarray, stock: np.ndarray, n_total: int,
                  rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Return (sales, demand) per firm. Firms without stock or price are not on the shelf."""
    n = len(prices)
    demand = np.zeros(n)
    active = (stock > 1e-9) & np.isfinite(prices) & (prices > 0)
    if active.any():
        mean = expected_demand(ds, t, prices[active], n_total)
        common = math.exp(ds.common_noise * rng.standard_normal() - 0.5 * ds.common_noise ** 2)
        idio = np.exp(ds.noise * rng.standard_normal(int(active.sum())) - 0.5 * ds.noise ** 2)
        demand[active] = mean * common * idio
    sales = np.minimum(demand, stock)
    unmet = float((demand - sales).sum()) * ds.spill
    room = stock - sales
    if unmet > 1e-9 and room.sum() > 1e-9:
        w = demand * (room > 1e-9)
        if w.sum() > 1e-12:
            sales = sales + np.minimum(room, unmet * w / w.sum())
    return sales, demand


# --------------------------------------------------------------------------
# auctions
# --------------------------------------------------------------------------
@dataclass
class Clearing:
    won: List[float]
    paid: List[float]
    clearing_price: Optional[float]
    total_sold: float


def clear_auction(fmt: str, supply: float, reserve: float,
                  bids: List[Tuple[int, float, float]], n_agents: int,
                  rng: np.random.Generator) -> Clearing:
    """Clear one multi-unit sealed-bid auction.

    bids: (agent index, quantity, unit price). Formats:
      first_price  - discriminatory / pay-as-bid; reported price = lowest accepted bid
      uniform, second_price - every winner pays the highest rejected bid (the reserve
                     if supply is not exhausted); second_price is the same rule and
                     reduces to a Vickrey auction for a single unit
      posted       - `reserve` is the posted price; demand above supply is rationed pro rata
    """
    won = [0.0] * n_agents
    paid = [0.0] * n_agents
    valid = [(a, q, p) for a, q, p in bids if q > 1e-12 and p >= reserve - 1e-12 and p > 0]
    if supply <= 1e-12 or not valid:
        return Clearing(won, paid, None, 0.0)
    if fmt == "posted":
        asked = sum(q for _, q, _ in valid)
        scale = min(1.0, supply / asked)
        for a, q, _ in valid:
            won[a] += q * scale
            paid[a] += q * scale * reserve
        return Clearing(won, paid, reserve, min(asked, supply))
    order = list(rng.permutation(len(valid)))            # random tie-breaking
    order.sort(key=lambda i: -valid[i][2])
    left = supply
    accepted: List[Tuple[int, float, float]] = []
    first_rejected: Optional[float] = None
    for i in order:
        a, q, p = valid[i]
        if left <= 1e-12:
            first_rejected = p if first_rejected is None else first_rejected
            continue
        take = min(q, left)
        accepted.append((a, take, p))
        left -= take
        if take < q - 1e-12 and first_rejected is None:
            first_rejected = p                           # the unserved part of the marginal bid
    sold = supply - left
    if fmt == "first_price":
        for a, q, p in accepted:
            won[a] += q
            paid[a] += q * p
        price = min(p for _, _, p in accepted) if accepted else None
    else:
        price = max(reserve, first_rejected) if first_rejected is not None else reserve
        for a, q, p in accepted:
            won[a] += q
            paid[a] += q * price
    return Clearing(won, paid, price, sold)
