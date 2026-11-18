"""Procurement: turn resource values into bids.

The planner says what each extra unit of a resource is worth (a decreasing
schedule of tranches). This module decides what to bid for each tranche:

* second-price / uniform-price: bid the value (truthful), shading later
  tranches a little because our own marginal bid can set the price we pay;
* first-price / pay-as-bid: bid b* = argmax (v - b) * P(win | b), with the win
  probability estimated from observed clearing prices (Han et al. 2020: when
  the winning price is revealed, one observation tells you the outcome of
  every bid you could have placed);
* posted price: buy every tranche worth more than the price.

Winning is never the goal: no bid exceeds the value of the unit, and the
total commitment never exceeds the cash budget.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.schemas import AuctionResult, Bid


def _phi(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(z, dtype=float) / math.sqrt(2.0)))


@dataclass
class Tranche:
    qty: float
    value: float      # value per unit


class ClearingStats:
    """Exponentially weighted kernel estimate of the clearing-price distribution."""

    def __init__(self, alpha: float = 0.3, max_points: int = 40):
        self.alpha = alpha
        self.max_points = max_points
        self.points: List[float] = []
        self.weights: List[float] = []
        self.prior_mean: Optional[float] = None
        self.prior_weight = 1.0
        self.n = 0

    def set_prior(self, mean: float) -> None:
        if self.prior_mean is None and mean is not None and mean > 0:
            self.prior_mean = float(mean)

    def update(self, price: float, weight: float = 1.0) -> None:
        if price is None or not math.isfinite(price) or price < 0:
            return
        decay = 1.0 - self.alpha
        self.weights = [w * decay for w in self.weights]
        self.prior_weight *= decay
        self.points.append(float(price))
        self.weights.append(float(weight))
        if len(self.points) > self.max_points:
            self.points.pop(0)
            self.weights.pop(0)
        self.n += 1

    @property
    def mean(self) -> Optional[float]:
        tot = sum(self.weights)
        if self.prior_mean is not None:
            return (float(np.dot(self.points, self.weights)) + self.prior_weight * self.prior_mean) / (
                tot + self.prior_weight)
        if tot <= 0:
            return None
        return float(np.dot(self.points, self.weights)) / tot

    def win_prob(self, bids: np.ndarray) -> np.ndarray:
        """P(clearing price <= bid) for each bid."""
        bids = np.asarray(bids, dtype=float)
        m = self.mean
        if m is None or m <= 0:
            return np.ones_like(bids)
        pts = list(self.points)
        wts = list(self.weights)
        bws = [max(0.06 * m, 0.10 * abs(p)) for p in pts]
        if self.prior_mean is not None and self.prior_weight > 1e-3:
            pts.append(self.prior_mean)
            wts.append(self.prior_weight)
            bws.append(0.35 * self.prior_mean)
        pts_a, wts_a, bws_a = np.array(pts), np.array(wts), np.array(bws)
        z = (bids[:, None] - pts_a[None, :]) / bws_a[None, :]
        return (_phi(z) * wts_a[None, :]).sum(axis=1) / wts_a.sum()


class BiddingOptimizer:
    def __init__(self, shade_mode: str = "winprob", fixed_shade: float = 0.8,
                 win_bias: float = 1.0, alpha: float = 0.3, demand_reduction: float = 0.10):
        self.shade_mode = shade_mode
        self.fixed_shade = fixed_shade
        self.win_bias = max(0.1, win_bias)
        self.alpha = alpha
        self.demand_reduction = demand_reduction
        self.stats: Dict[str, ClearingStats] = {}
        self.last_bids: Dict[str, List[Tuple[float, float]]] = {}   # resource -> [(price, qty)]
        self.won_units: Dict[str, float] = {}
        self.bid_units: Dict[str, float] = {}
        self.spend: Dict[str, float] = {}

    def stat(self, resource: str) -> ClearingStats:
        if resource not in self.stats:
            self.stats[resource] = ClearingStats(alpha=self.alpha)
        return self.stats[resource]

    def expected_price(self, resource: str) -> Optional[float]:
        return self.stat(resource).mean

    # ------------------------------------------------------------------
    def observe(self, result: AuctionResult) -> None:
        st = self.stat(result.resource)
        mine = self.last_bids.get(result.resource, [])
        self.won_units[result.resource] = self.won_units.get(result.resource, 0.0) + result.won_qty
        self.spend[result.resource] = self.spend.get(result.resource, 0.0) + result.paid
        if result.clearing_price is not None:
            if result.total_sold is not None and result.total_sold <= 1e-9 and not mine:
                return  # nothing traded and we did not bid: no information
            st.update(result.clearing_price)
            return
        # clearing price not revealed: learn from our own outcome only
        if not mine:
            return
        asked = sum(q for _, q in mine)
        hi = max(p for p, _ in mine)
        lo = min(p for p, _ in mine)
        if result.won_qty <= 1e-9:
            st.update(hi * 1.08)              # the price to beat was above our best bid
        elif result.won_qty >= asked - 1e-9:
            st.update(lo * 0.95)              # even our lowest bid won
        else:
            cum = 0.0
            marginal = hi
            for p, q in sorted(mine, reverse=True):
                cum += q
                marginal = p
                if cum >= result.won_qty - 1e-9:
                    break
            st.update(marginal)

    # ------------------------------------------------------------------
    def shade(self, value: float, resource: str, fmt: str, reserve: float,
              tranche_index: int = 0, n_tranches: int = 1) -> Tuple[Optional[float], float]:
        """Return (bid price or None, estimated win probability)."""
        if value <= reserve or value <= 0:
            return None, 0.0
        st = self.stat(resource)
        if fmt == "posted":
            return float(reserve), 1.0
        if fmt in ("uniform", "second_price") or self.shade_mode == "truthful":
            frac = 1.0 - self.demand_reduction * (tranche_index / max(1, n_tranches))
            if fmt == "first_price":
                frac = 1.0
            b = max(reserve, value * frac)
            return float(b), float(st.win_prob(np.array([b]))[0])
        if self.shade_mode == "fixed":
            b = max(reserve, value * self.fixed_shade)
            return float(b), float(st.win_prob(np.array([b]))[0])
        # first price with a win-probability model
        lo = max(reserve, 1e-9)
        grid = np.linspace(lo, value, 41)
        g = st.win_prob(grid)
        util = (value - grid) * np.power(np.maximum(g, 1e-12), self.win_bias)
        k = int(np.argmax(util))
        if util[k] <= 0:
            return None, 0.0
        return float(grid[k]), float(g[k])

    # ------------------------------------------------------------------
    def make_bids(self, requests: List[Tuple[str, str, float, float, List[Tranche]]],
                  budget: float) -> Tuple[List[Bid], Dict[str, float]]:
        """requests: (resource, format, lot_quantity, reserve, tranches).

        Returns the bids and the expected quantity won per resource.
        """
        cands = []   # (ratio, resource, qty, price, pwin)
        for resource, fmt, lot_qty, reserve, tranches in requests:
            left = max(0.0, lot_qty)
            tr = [t for t in tranches if t.qty > 1e-9 and t.value > 0]
            tr.sort(key=lambda t: -t.value)
            for i, t in enumerate(tr):
                if left <= 1e-9:
                    break
                price, pwin = self.shade(t.value, resource, fmt, reserve, i, len(tr))
                if price is None or price <= 0:
                    continue
                price = min(price, t.value)          # never pay more than the unit is worth
                qty = min(t.qty, left)
                left -= qty
                cands.append((t.value / price, resource, qty, price, pwin))
        cands.sort(key=lambda c: -c[0])
        bids: List[Bid] = []
        expected: Dict[str, float] = {}
        left_budget = max(0.0, budget)
        self.last_bids = {}
        for _, resource, qty, price, pwin in cands:
            if left_budget <= 1e-9:
                break
            q = min(qty, left_budget / price)
            if q <= 1e-9:
                continue
            left_budget -= q * price
            bids.append(Bid(resource, float(q), float(price)))
            expected[resource] = expected.get(resource, 0.0) + q * pwin
            self.last_bids.setdefault(resource, []).append((float(price), float(q)))
            self.bid_units[resource] = self.bid_units.get(resource, 0.0) + q
        return bids, expected
