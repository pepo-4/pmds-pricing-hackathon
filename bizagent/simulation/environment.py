"""Synthetic competitive market.

Sequence of events in round t (documented because every agent's reasoning
depends on it):

  1. Raw material won `lead_time` rounds ago is delivered.
  2. The lots on auction this round are announced.
  3. BID   - every firm submits sealed bids. Auctions clear, winners pay at once.
             With lead_time = 0 the material is usable immediately.
  4. SELL  - every firm chooses production quantities and posts one price per
             product. Production is scaled down to what inputs, capacity and
             cash allow.
  5. Customers buy (hidden demand model). Unserved customers partly move to
     firms that still have stock.
  6. Holding and fixed costs are charged.

`two_stage=True` asks agents twice per round (phase "bid", then phase "sell"
after they have seen the auction result). `two_stage=False` asks once (phase
"all"): bids, production and prices are committed together.

What an agent sees: its own cash and stocks, the lots, its own sales and
auction outcomes, and - if the scenario reveals them - last round's rival
prices and the clearing price. It never sees the demand model, rival stocks,
rival cash or rival bids.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ..core.schemas import Action, AuctionLot, AuctionResult, GameSpec, Observation
from .market import DemandSpec, clear_auction, realise_sales


@dataclass
class Scenario:
    name: str
    seed: int
    n_rounds: int
    spec: GameSpec                       # public rules
    demand: Dict[str, DemandSpec]        # hidden
    supply: Dict[str, float]             # units on auction per round
    supply_noise: float = 0.10
    two_stage: bool = True
    reveal_prices: bool = True
    reveal_clearing: bool = True
    reveal_horizon: bool = True
    tags: Dict[str, Any] = field(default_factory=dict)
    scale: float = 1.0                   # fair-share revenue potential, to normalise scores

    def public_spec(self) -> GameSpec:
        spec = copy.deepcopy(self.spec)
        spec.horizon = self.n_rounds if self.reveal_horizon else None
        return spec


def score_of(spec: GameSpec, row: Dict[str, float]) -> float:
    o = spec.objective
    profit = row["final_cash"] - row["initial_cash"]
    if o.inventory_valued:
        profit += row["salvage_value"]
    return (o.w_profit * profit + o.w_revenue * row["revenue"] + o.w_units * row["units_sold"]
            + o.w_cash * row["final_cash"])


class Game:
    def __init__(self, scenario: Scenario, agents: List[Any], ids: Optional[List[str]] = None,
                 record: bool = False):
        self.sc = scenario
        self.agents = agents
        self.ids = ids or [f"F{i}" for i in range(len(agents))]
        self.record = record
        self.history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    def run(self) -> List[Dict[str, Any]]:
        sc, spec = self.sc, self.sc.spec
        n, T = len(self.agents), sc.n_rounds
        rng = np.random.default_rng(sc.seed)
        P, R = list(spec.products), list(spec.resources)
        J, K = len(P), len(R)
        A = np.array([[spec.products[p].recipe.get(r, 0.0) for r in R] for p in P]).reshape(J, K)
        c = np.array([spec.products[p].unit_cost for p in P])
        kap = np.array([spec.products[p].capacity_use for p in P])
        hold_f = np.array([spec.products[p].holding_cost for p in P])
        hold_r = np.array([spec.resources[r].holding_cost for r in R])
        salv_f = np.array([spec.products[p].salvage for p in P])
        salv_r = np.array([spec.resources[r].salvage for r in R])
        L = int(spec.lead_time)
        cash0 = float(spec.initial_cash or 0.0)

        cash = np.full(n, cash0)
        raw = np.zeros((n, K))
        fin = np.zeros((n, J))
        pending: List[List[float]] = []                  # [arrival round, agent, resource idx, qty]
        last_prices = [dict() for _ in range(n)]
        last_sales = [dict() for _ in range(n)]
        last_offered = [dict() for _ in range(n)]
        prev_results: List[List[AuctionResult]] = [[] for _ in range(n)]
        m = {k: np.zeros(n) for k in ("revenue", "units_sold", "spend", "units_bought", "units_bid",
                                      "produced", "prod_cost", "holding", "lost_sales", "errors",
                                      "ms_total", "ms_max", "calls")}
        min_cash = cash.copy()

        public = sc.public_spec()
        for i, ag in enumerate(self.agents):
            ag.reset(copy.deepcopy(public), self.ids[i], seed=int(rng.integers(0, 2 ** 31 - 1)))

        def call(i: int, obs: Observation) -> Action:
            t0 = time.perf_counter()
            try:
                act = self.agents[i].act(obs)
                if not isinstance(act, Action):
                    act = Action.from_dict(act)
            except Exception:
                m["errors"][i] += 1
                act = Action()
            ms = (time.perf_counter() - t0) * 1000.0
            m["ms_total"][i] += ms
            m["ms_max"][i] = max(m["ms_max"][i], ms)
            m["calls"][i] += 1
            return act

        def observe(i: int, t: int, phase: str, lots: List[AuctionLot],
                    results: List[AuctionResult]) -> Observation:
            pend: Dict[str, float] = {}
            for arr, a, ri, q in pending:
                if a == i:
                    pend[R[ri]] = pend.get(R[ri], 0.0) + q
            comp = {}
            if sc.reveal_prices:
                comp = {self.ids[k]: dict(last_prices[k]) for k in range(n) if k != i and last_prices[k]}
            return Observation(
                round=t, phase=phase, cash=float(cash[i]),
                raw_inventory={r: float(raw[i, ri]) for ri, r in enumerate(R)},
                fin_inventory={p: float(fin[i, j]) for j, p in enumerate(P)},
                pending_raw=pend,
                lots=[copy.copy(lot) for lot in lots] if phase != "sell" else [],
                last_prices=dict(last_prices[i]), last_sales=dict(last_sales[i]),
                last_offered=dict(last_offered[i]), competitor_prices=comp,
                auction_results=list(results),
                extra={"rounds_left": T - t} if sc.reveal_horizon else {})

        for t in range(T):
            # 1. deliveries
            keep = []
            for item in pending:
                if item[0] <= t:
                    raw[int(item[1]), int(item[2])] += item[3]
                else:
                    keep.append(item)
            pending = keep

            # 2. lots
            lots = []
            for ri, r in enumerate(R):
                q = sc.supply[r] * max(0.2, 1.0 + sc.supply_noise * rng.standard_normal())
                rs = spec.resources[r]
                lots.append(AuctionLot(r, float(q), float(rs.reserve_price), rs.auction_format))

            # 3. bids
            phase1 = "bid" if sc.two_stage else "all"
            acts = [call(i, observe(i, t, phase1, lots, [] if sc.two_stage else prev_results[i]))
                    for i in range(n)]
            book: Dict[str, List] = {r: [] for r in R}
            for i, act in enumerate(acts):
                budget = max(0.0, cash[i])
                for b in sorted(act.bids, key=lambda b: -b.price):
                    if b.resource not in book or not np.isfinite(b.price) or not np.isfinite(b.quantity):
                        continue
                    if b.price <= 0 or b.quantity <= 0:
                        continue
                    q = min(b.quantity, budget / b.price)
                    if q <= 1e-12:
                        continue
                    budget -= q * b.price
                    book[b.resource].append((i, float(q), float(b.price)))
                    m["units_bid"][i] += q
            results: List[List[AuctionResult]] = [[] for _ in range(n)]
            clearing_log = {}
            for ri, r in enumerate(R):
                lot = lots[ri]
                cl = clear_auction(lot.format, lot.quantity, lot.reserve_price, book[r], n, rng)
                clearing_log[r] = cl.clearing_price
                for i in range(n):
                    cash[i] -= cl.paid[i]
                    m["spend"][i] += cl.paid[i]
                    m["units_bought"][i] += cl.won[i]
                    if cl.won[i] > 0:
                        if L == 0:
                            raw[i, ri] += cl.won[i]
                        else:
                            pending.append([t + L, i, ri, cl.won[i]])
                    results[i].append(AuctionResult(
                        resource=r, clearing_price=cl.clearing_price if sc.reveal_clearing else None,
                        won_qty=float(cl.won[i]), paid=float(cl.paid[i]),
                        total_sold=float(cl.total_sold) if sc.reveal_clearing else None,
                        supply=float(lot.quantity), round=t))
            prev_results = results

            # 4. production and prices
            if sc.two_stage:
                acts = [call(i, observe(i, t, "sell", lots, results[i])) for i in range(n)]
            prices = np.full((n, J), np.nan)
            offered = np.zeros((n, J))
            for i, act in enumerate(acts):
                x = np.array([max(0.0, float(act.production.get(p, 0.0)))
                              if np.isfinite(act.production.get(p, 0.0)) else 0.0 for p in P])
                if x.sum() > 0:
                    s = 1.0
                    need = A.T @ x
                    for ri in range(K):
                        if need[ri] > raw[i, ri] + 1e-12:
                            s = min(s, raw[i, ri] / need[ri])
                    if spec.capacity is not None and kap @ x > spec.capacity + 1e-12:
                        s = min(s, spec.capacity / float(kap @ x))
                    if c @ x > max(cash[i], 0.0) + 1e-12:
                        s = min(s, max(cash[i], 0.0) / float(c @ x))
                    x = x * max(s, 0.0)
                    raw[i] = np.maximum(raw[i] - A.T @ x, 0.0)
                    fin[i] += x
                    cash[i] -= float(c @ x)
                    m["produced"][i] += x.sum()
                    m["prod_cost"][i] += float(c @ x)
                for j, p in enumerate(P):
                    pr = act.prices.get(p, last_prices[i].get(p))
                    if pr is None or not np.isfinite(pr) or pr <= 0:
                        continue
                    ps = spec.products[p]
                    pr = max(pr, ps.price_min)
                    if ps.price_max is not None:
                        pr = min(pr, ps.price_max)
                    prices[i, j] = pr
            min_cash = np.minimum(min_cash, cash)

            # 5. market
            round_sales = np.zeros((n, J))
            for j, p in enumerate(P):
                offered[:, j] = fin[:, j]
                sales, demand = realise_sales(sc.demand[p], t, prices[:, j], fin[:, j], n, rng)
                round_sales[:, j] = sales
                fin[:, j] -= sales
                rev = np.where(np.isfinite(prices[:, j]), prices[:, j], 0.0) * sales
                cash += rev
                m["revenue"] += rev
                m["units_sold"] += sales
                m["lost_sales"] += np.maximum(demand - sales, 0.0)
            fin = np.maximum(fin, 0.0)

            # 6. holding and fixed costs
            hold = fin @ hold_f + raw @ hold_r + spec.fixed_cost
            cash -= hold
            m["holding"] += hold
            min_cash = np.minimum(min_cash, cash)

            for i in range(n):
                last_prices[i] = {p: float(prices[i, j]) for j, p in enumerate(P) if np.isfinite(prices[i, j])}
                last_sales[i] = {p: float(round_sales[i, j]) for j, p in enumerate(P)}
                last_offered[i] = {p: float(offered[i, j]) for j, p in enumerate(P)}
            if self.record:
                self.history.append({"round": t, "prices": prices.copy(), "sales": round_sales.copy(),
                                     "offered": offered.copy(), "cash": cash.copy(),
                                     "clearing": dict(clearing_log),
                                     "won": [[res.won_qty for res in results[i]] for i in range(n)]})

        # undelivered material still belongs to the firm
        for arr, a, ri, q in pending:
            raw[int(a), int(ri)] += q
        rows = []
        for i, ag in enumerate(self.agents):
            row = {
                "id": self.ids[i], "agent": getattr(ag, "name", type(ag).__name__),
                "initial_cash": cash0, "final_cash": float(cash[i]),
                "salvage_value": float(fin[i] @ salv_f + raw[i] @ salv_r),
                "fin_left": float(fin[i].sum()), "raw_left": float(raw[i].sum()),
                "min_cash": float(min_cash[i]),
            }
            for k, v in m.items():
                row[k] = float(v[i])
            row["ms_mean"] = row["ms_total"] / max(row["calls"], 1.0)
            row["score"] = score_of(spec, row)
            row["score_norm"] = row["score"] / max(sc.scale, 1e-9)
            row["engine_errors"] = float(getattr(ag, "errors", 0) or 0)
            rows.append(row)
        order = np.argsort([-r["score"] for r in rows])
        for rank, i in enumerate(order, start=1):
            rows[int(i)]["rank"] = rank
        total_units = sum(r["units_sold"] for r in rows)
        for r in rows:
            r["market_share"] = r["units_sold"] / total_units if total_units > 0 else 0.0
            r["n_firms"] = n
        return rows
