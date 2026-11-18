"""Rule-based agents.

`Heuristic` is the plumbing shared by simple agents: it tracks sales, input
prices and rival prices, and turns three small decisions (how much to hold,
what to charge, how much of the margin to bid) into a full action.

`RuleBaselineAgent` is V1 of our own agent and the fallback policy the
strategic agent drops to if its optimiser ever fails: follow the cheapest
rival, never below a cost floor, never bid more than the inputs can earn.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ..core.schemas import Action, Bid, Observation
from .base import BaseAgent


class Heuristic(BaseAgent):
    name = "heuristic"
    bid_frac = 0.7        # share of the unit margin offered for inputs
    cover = 1.2           # rounds of expected sales to hold as stock + inputs
    blind_frac = 0.35     # first price as a share of the allowed range, when nothing is known
    spend_frac = 0.8      # share of cash that may be committed to bids in one round

    # ------------------------------------------------------------------
    def reset(self, spec, agent_id="me", seed=0):
        self.spec, self.agent_id = spec, agent_id
        self.rng = np.random.default_rng(seed)
        self.P = list(spec.products)
        self.R = list(spec.resources)
        self.n = spec.n_firms or 5
        self.raw_px: Dict[str, float] = {}
        self.sales_ew: Dict[str, float] = {}
        self.price: Dict[str, float] = {}
        self.hist: Dict[str, List[Tuple[float, float, bool]]] = {p: [] for p in self.P}
        self.lot_qty: Dict[str, float] = {}
        self._seen_round = -1
        self._seen_auc = set()
        self.setup()

    def setup(self) -> None:
        pass

    def on_sale(self, obs: Observation, product: str, price: float, sales: float, sold_out: bool) -> None:
        pass

    # ------------------------------------------------------------------
    def remaining(self, obs: Observation) -> int:
        if self.spec.horizon:
            return max(1, int(self.spec.horizon) - obs.round)
        return 99

    def learn(self, obs: Observation) -> None:
        if obs.lots:
            agg: Dict[str, float] = {}
            for lot in obs.lots:
                agg[lot.resource] = agg.get(lot.resource, 0.0) + lot.quantity
            self.lot_qty.update(agg)
        if obs.round > self._seen_round:
            self._seen_round = obs.round
            for p in self.P:
                if p not in obs.last_prices or p not in obs.last_sales:
                    continue
                s = obs.last_sales[p]
                offered = obs.last_offered.get(p, s + obs.fin_inventory.get(p, 0.0))
                if offered <= 1e-9:
                    continue
                sold_out = s >= offered - 1e-6 * max(1.0, offered)
                est = s * (1.15 if sold_out else 1.0)
                self.sales_ew[p] = est if p not in self.sales_ew else 0.6 * self.sales_ew[p] + 0.4 * est
                self.hist[p].append((obs.last_prices[p], s, sold_out))
                self.on_sale(obs, p, obs.last_prices[p], s, sold_out)
        for res in obs.auction_results:
            key = (res.round if res.round is not None else (obs.round, obs.phase), res.resource)
            if key in self._seen_auc:
                continue
            self._seen_auc.add(key)
            px = res.clearing_price
            if (px is None or px <= 0) and res.won_qty > 1e-9:
                px = res.paid / res.won_qty
            if px and px > 0:
                old = self.raw_px.get(res.resource)
                self.raw_px[res.resource] = px if old is None else 0.6 * old + 0.4 * px

    # ------------------------------------------------------------------
    def comp_prices(self, obs: Observation, p: str) -> List[float]:
        return [d[p] for d in obs.competitor_prices.values() if d.get(p, 0.0) > 0]

    def raw_cost(self, r: str) -> float:
        if r in self.raw_px:
            return self.raw_px[r]
        rs = self.spec.resources.get(r)
        if rs is None:
            return 0.0
        return rs.ref_price if rs.ref_price else rs.reserve_price * 1.3

    def unit_cost(self, p: str) -> float:
        ps = self.spec.products[p]
        return ps.unit_cost + sum(a * self.raw_cost(r) for r, a in ps.recipe.items())

    def clip_price(self, p: str, price: float) -> float:
        ps = self.spec.products[p]
        price = max(price, ps.price_min, 1e-6)
        if ps.price_max is not None:
            price = min(price, ps.price_max)
        return float(price)

    def blind_price(self, p: str) -> float:
        ps = self.spec.products[p]
        cost = self.unit_cost(p)
        if ps.price_max is not None:
            base = ps.price_min + self.blind_frac * (ps.price_max - ps.price_min)
        elif cost > 0:
            base = 2.0 * cost
        else:
            base = 1.0
        return max(base, 1.1 * cost)

    def ref_price(self, obs: Observation, p: str) -> float:
        cp = self.comp_prices(obs, p)
        if cp:
            return float(np.mean(cp))
        if p in self.price:
            return self.price[p]
        return self.blind_price(p)

    def fair_share(self, p: str) -> float:
        hint = self.spec.demand_hint.get(p)
        if hint and hint > 0:
            return hint
        ps = self.spec.products[p]
        vals = []
        for r, a in ps.recipe.items():
            if a > 0 and r in self.lot_qty:
                users = sum(1 for q in self.spec.products.values() if q.recipe.get(r, 0.0) > 0)
                vals.append(self.lot_qty[r] / self.n / a / max(users, 1))
        if vals:
            return min(vals)
        if self.spec.capacity:
            return 0.5 * self.spec.capacity / max(len(self.P), 1) / max(ps.capacity_use, 1e-9)
        return 10.0

    def exp_sales(self, p: str) -> float:
        return self.sales_ew.get(p, self.fair_share(p))

    def input_value(self, obs: Observation, r: str) -> float:
        """Rough value of one unit of resource r: the best product margin, split over its inputs."""
        best = 0.0
        for p in self.P:
            ps = self.spec.products[p]
            if ps.recipe.get(r, 0.0) <= 0:
                continue
            price = self.price.get(p) or self.ref_price(obs, p)
            best = max(best, (price - ps.unit_cost) / max(sum(ps.recipe.values()), 1e-9))
        return best

    # ---- the three decisions a subclass overrides ---------------------
    def target(self, obs: Observation, p: str) -> float:
        return self.cover * self.exp_sales(p)

    def price_for(self, obs: Observation, p: str) -> float:
        return self.unit_cost(p) * 1.35

    def bid_fraction(self, obs: Observation, r: str) -> float:
        return self.bid_frac

    def prod_target(self, obs: Observation, p: str) -> float:
        return 1.1 * self.exp_sales(p)

    # ------------------------------------------------------------------
    def bids(self, obs: Observation) -> List[Bid]:
        if self.spec.horizon and self.remaining(obs) - int(self.spec.lead_time) < 1:
            return []                      # inputs would arrive after the last round
        lots: Dict[str, List[float]] = {}
        fmts: Dict[str, str] = {}
        for lot in obs.lots:
            if lot.resource not in self.spec.resources:
                continue
            cur = lots.setdefault(lot.resource, [0.0, 0.0])
            cur[0] += lot.quantity
            cur[1] = max(cur[1], lot.reserve_price)
            fmts[lot.resource] = lot.format or self.spec.resources[lot.resource].auction_format
        out: List[Bid] = []
        budget = max(0.0, obs.cash) * self.spend_frac
        for r, (qty, reserve) in lots.items():
            need = -obs.raw_inventory.get(r, 0.0) - obs.pending_raw.get(r, 0.0)
            for p in self.P:
                a = self.spec.products[p].recipe.get(r, 0.0)
                if a > 0:
                    need += a * max(0.0, self.target(obs, p) - obs.fin_inventory.get(p, 0.0))
            if need <= 1e-9:
                continue
            v = self.input_value(obs, r)
            if fmts[r] == "posted":
                if v * 0.95 < reserve:
                    continue
                px = reserve
            else:
                px = v * self.bid_fraction(obs, r)
            if px <= 0 or px < reserve:
                continue
            q = min(need, qty, budget / px)
            if q > 1e-9:
                out.append(Bid(r, float(q), float(px)))
                budget -= q * px
        return out

    def sell(self, obs: Observation):
        prod, prices = {}, {}
        for p in self.P:
            want = max(0.0, self.prod_target(obs, p) - obs.fin_inventory.get(p, 0.0))
            if want > 1e-9:
                prod[p] = float(want)
            prices[p] = self.clip_price(p, self.price_for(obs, p))
        return prod, prices

    def act(self, obs: Observation) -> Action:
        self.learn(obs)
        a = Action()
        if obs.phase in ("all", "bid"):
            a.bids = self.bids(obs)
        if obs.phase in ("all", "sell"):
            a.production, a.prices = self.sell(obs)
            self.price.update(a.prices)
        return a


class RuleBaselineAgent(Heuristic):
    """V1: undercut the cheapest rival slightly, hold a cost floor, bid with discipline."""
    name = "v1_rule"
    bid_frac = 0.7
    cover = 1.3
    undercut = 0.02
    floor_margin = 0.08

    def setup(self):
        self.markup = 0.40

    def on_sale(self, obs, product, price, sales, sold_out):
        # only used when rival prices are hidden: move the mark-up with what the shelf says
        if sold_out:
            self.markup = min(1.5, self.markup + 0.05)
        elif sales < 0.7 * self.exp_sales(product):
            self.markup = max(0.05, self.markup - 0.05)

    def price_for(self, obs, p):
        cost = self.unit_cost(p)
        floor = cost * (1.0 + self.floor_margin)
        cp = self.comp_prices(obs, p)
        if cp:
            price = min(cp) * (1.0 - self.undercut)
        elif cost > 0 and self.hist[p]:
            price = cost * (1.0 + self.markup)
        else:
            price = self.blind_price(p)
        last_round = bool(self.spec.horizon) and self.remaining(obs) <= 1
        if last_round and self.spec.products[p].salvage <= 0:
            # stock is worthless tomorrow: clear it, but still not for free
            floor = max(self.spec.products[p].unit_cost, 0.3 * cost)
            if cp:
                price = min(price, min(cp) * 0.9)
        return max(floor, price)

    def target(self, obs, p):
        cover = 1.0 if (self.spec.horizon and self.remaining(obs) <= 2) else self.cover
        return cover * self.exp_sales(p)
