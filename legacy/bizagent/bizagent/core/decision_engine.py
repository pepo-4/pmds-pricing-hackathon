"""Strategic decision engine: one coordinated decision per call.

Per round the engine

1. updates beliefs (demand per product, rival prices, clearing prices);
2. prices the two things that link rounds together: what a unit of raw
   material kept in stock is worth (omega) and what an unsold finished unit is
   worth (mu);
3. solves one LP for production, sales volume and therefore price;
4. values extra raw material by re-solving that LP with more of it, and turns
   the value schedule into bids suited to the auction format;
5. passes everything through the risk manager.

No module optimises its own metric: procurement values inputs by what
production and sales can earn from them, and pricing charges for the inputs'
replacement cost.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..market.competitor_model import CompetitorModel
from ..market.demand_model import BayesianDemandModel
from ..pricing.bandit import ContextualPriceBandit, DiscountedTS
from ..procurement.bidding import BiddingOptimizer, Tranche
from ..production.planner import DemandCurve, PlanInput, Planner, PlanResult, revenue_segments
from ..risk.risk_manager import RiskContext, RiskManager
from .config import StrategyConfig
from .schemas import Action, GameSpec, Observation


class DecisionEngine:
    def __init__(self, spec: GameSpec, cfg: Optional[StrategyConfig] = None,
                 seed: Optional[int] = None):
        self.spec = spec
        self.cfg = cfg or StrategyConfig()
        self.rng = np.random.default_rng(seed)
        self.products = list(spec.products)
        self.resources = list(spec.resources)
        J, R = len(self.products), len(self.resources)
        self.A = np.zeros((J, R))
        for j, p in enumerate(self.products):
            for r, q in spec.products[p].recipe.items():
                if r in spec.resources:
                    self.A[j, self.resources.index(r)] = q
        self.c = np.array([spec.products[p].unit_cost for p in self.products])
        self.kappa = np.array([spec.products[p].capacity_use for p in self.products])
        valued = spec.objective.inventory_valued
        self.salv_fin = np.array([spec.products[p].salvage if valued else 0.0 for p in self.products])
        self.salv_raw = np.array([spec.resources[r].salvage if valued else 0.0 for r in self.resources])
        self.hold_fin = np.array([spec.products[p].holding_cost for p in self.products])
        self.hold_raw = np.array([spec.resources[r].holding_cost for r in self.resources])

        c = self.cfg
        self.models = {p: BayesianDemandModel(c.prior_elasticity, c.cross_frac, c.forgetting,
                                              min_elasticity_frac=c.min_elasticity_frac)
                       for p in self.products}
        self.comp = CompetitorModel(c.comp_feature, c.cv_max)
        self.bidder = BiddingOptimizer(c.shade_mode, c.fixed_shade, c.win_bias, c.clearing_alpha,
                                       c.demand_reduction)
        self.planner = Planner(c.solver)
        self.risk = RiskManager(spec)

        wp = max(spec.objective.w_profit + spec.objective.w_cash, 0.3)
        self.w_rev = c.w_revenue if c.w_revenue is not None else spec.objective.w_revenue / wp
        self.w_units = c.w_units if c.w_units is not None else spec.objective.w_units / wp

        self.last_prices: Dict[str, float] = {}
        self.price_hist: Dict[str, List[float]] = {p: [] for p in self.products}
        self.last_mc: Dict[str, float] = {}
        self.init_price: Dict[str, float] = {}
        self.initial_cash: Optional[float] = spec.initial_cash
        self.bandits: Dict[str, ContextualPriceBandit] = {}
        self.meta = DiscountedTS(max(1, len(c.meta_modes)), c.meta_discount)
        self.meta_arm: Optional[int] = None
        self.meta_start = 0
        self.prev_meta_arm: Optional[int] = None
        self._market_round = -1
        self._auction_sig = None
        self._auction_seen = set()
        self._theta_round = -1
        self._theta: Dict[str, Tuple[float, float, float]] = {}
        self._dither_round = -1
        self._dither: Dict[str, float] = {}
        self.decisions = 0

    # ==================================================================
    # beliefs
    # ==================================================================
    def _update_market(self, obs: Observation) -> None:
        if obs.round == self._market_round:
            return
        self._market_round = obs.round
        own_last = {p: obs.last_prices.get(p, self.last_prices.get(p)) for p in self.products}
        own_last = {p: v for p, v in own_last.items() if v}
        self.comp.update(obs.competitor_prices if self.cfg.use_comp_prices else {}, own_last)
        reward_total = 0.0
        for p in self.products:
            price = own_last.get(p)
            if price is None or p not in obs.last_sales:
                continue
            sales = obs.last_sales[p]
            offered = obs.last_offered.get(p, sales + obs.fin_inventory.get(p, 0.0))
            if offered <= 1e-9:
                continue                                   # nothing on the shelf: no information
            censored = sales >= offered - 1e-6 * max(1.0, offered)
            self.models[p].observe(price, self.comp.last(p), sales, censored)
            reward = sales * (price - self.last_mc.get(p, 0.0))
            reward_total += reward
            if p in self.bandits:
                self.bandits[p].update(reward)
        if self.prev_meta_arm is not None and own_last:
            self.meta.update(self.prev_meta_arm, reward_total)

    def _update_auctions(self, obs: Observation) -> None:
        if not obs.auction_results:
            return
        tagged = all(r.round is not None for r in obs.auction_results)
        if not tagged:
            # no round tag: assume each result is delivered once per (round, phase)
            sig = (obs.round, obs.phase, tuple((r.resource, r.clearing_price, r.won_qty, r.paid)
                                               for r in obs.auction_results))
            if sig == self._auction_sig:
                return
            self._auction_sig = sig
        for res in obs.auction_results:
            if res.resource not in self.spec.resources:
                continue
            if tagged:
                key = (res.round, res.resource)
                if key in self._auction_seen:
                    continue
                self._auction_seen.add(key)
            self.bidder.observe(res)

    # ==================================================================
    # economics helpers
    # ==================================================================
    def _remaining(self, obs: Observation) -> int:
        if self.spec.horizon:
            return max(1, int(self.spec.horizon) - obs.round)
        return max(2, int(self.cfg.assumed_remaining))

    def _rho(self) -> np.ndarray:
        """Expected acquisition price of each resource."""
        out = np.zeros(len(self.resources))
        for i, r in enumerate(self.resources):
            m = self.bidder.expected_price(r)
            rs = self.spec.resources[r]
            if m is None:
                m = rs.ref_price if rs.ref_price else rs.reserve_price * 1.3
            out[i] = max(m, 0.0)
        return out

    def _omega(self, rem: int) -> np.ndarray:
        """Value of a unit of raw material that is still in stock after this round."""
        if rem >= 2 and self.cfg.inventory_aware:
            return np.maximum(self.salv_raw, self._rho() - self.hold_raw)
        return self.salv_raw.copy()

    def _guess_demand(self, j: int, obs: Observation) -> float:
        name = self.products[j]
        hint = self.spec.demand_hint.get(name)
        if hint and hint > 0:
            return float(hint)
        if self.spec.capacity:
            return 0.5 * self.spec.capacity / max(self.kappa.sum(), 1e-9)
        feas = [obs.raw_inventory.get(r, 0.0) / a for r, a in self.spec.products[name].recipe.items()
                if a > 0]
        stock = obs.fin_inventory.get(name, 0.0)
        if feas and min(feas) + stock > 0:
            return min(feas) + stock
        mc = self.c[j] + float(self.A[j] @ self._rho())
        if mc > 0 and obs.cash > 0:
            return 0.25 * obs.cash / mc / max(1, len(self.products))
        return 1.0

    def _initial_price(self, j: int) -> float:
        name = self.products[j]
        if name in self.init_price:
            return self.init_price[name]
        ps = self.spec.products[name]
        mc0 = self.c[j] + float(self.A[j] @ self._rho())
        floor0 = max(ps.price_min, mc0 * (1.0 + self.cfg.min_margin))
        comp = self.comp.predict(name)
        if comp:
            p = max(floor0, comp)
        elif ps.price_max is not None:
            p = floor0 + self.cfg.init_price_frac * max(0.0, ps.price_max - floor0)
            if mc0 > 0:
                p = min(p, mc0 * (1.0 + 3.0 * self.cfg.init_markup))
        elif mc0 > 0:
            p = mc0 * (1.0 + self.cfg.init_markup)
        else:
            p = max(ps.price_min, 1.0)
        if ps.price_max is not None:
            p = min(p, ps.price_max)
        self.init_price[name] = float(max(p, 1e-9))
        return self.init_price[name]

    def _ensure_auction_priors(self) -> None:
        for i, r in enumerate(self.resources):
            st = self.bidder.stat(r)
            if st.prior_mean is not None:
                continue
            rs = self.spec.resources[r]
            if rs.ref_price:
                st.set_prior(rs.ref_price)
                continue
            top = 0.0
            for j in range(len(self.products)):
                if self.A[j, i] <= 0:
                    continue
                p = self.last_prices.get(self.products[j]) or self._initial_price(j)
                top = max(top, (p - self.c[j]) / max(self.A[j].sum(), 1e-9))
            prior = max(rs.reserve_price * 1.2, self.cfg.prior_clearing_frac * top)
            if prior > 0:
                st.set_prior(prior)

    def _params(self, name: str, obs: Observation) -> Optional[Tuple[float, float, float]]:
        if obs.round != self._theta_round:
            self._theta_round = obs.round
            self._theta = {}
        if name not in self._theta:
            m = self.models[name]
            if not m.ready:
                return None
            self._theta[name] = m.params(self.rng, self.cfg.ts_scale)
        return self._theta[name]

    def _curve(self, j: int, obs: Observation, lo: float, hi: float, n: int) -> DemandCurve:
        name = self.products[j]
        lo, hi = max(lo, 1e-9), max(hi, lo, 1e-9)
        grid = np.array([hi]) if hi - lo <= 1e-9 * hi else np.linspace(hi, lo, max(2, n))
        th = self._params(name, obs)
        if th is None:
            p0 = self._initial_price(j)
            q0 = self._guess_demand(j, obs)
            qty = q0 * np.maximum(0.0, 1.0 + self.cfg.prior_elasticity * (1.0 - grid / p0))
            return DemandCurve(grid, qty)
        a, b, c = th
        pc = self.comp.predict(name)
        if pc is None:
            qty = self.models[name].predict(grid, None, th)
        else:
            kappa = self.comp.conjectural_variation(name) if self.cfg.use_opponent_model else 0.0
            p_last = self.last_prices.get(name)
            pc_p = pc * (1.0 + kappa * (grid / p_last - 1.0)) if (p_last and kappa > 0) else pc
            qty = np.maximum(0.0, a + b * grid + c * pc_p)
        return DemandCurve(grid, qty)

    def _unit_bonus(self, j: int) -> float:
        ref = self.last_prices.get(self.products[j]) or self._initial_price(j)
        return self.w_units * ref

    def _leftover_value(self, j: int, rem: int, mc: float, stock: float, obs: Observation) -> float:
        """mu_j: what one finished unit is worth if it is not sold this round."""
        sal = float(self.salv_fin[j])
        if rem <= 1 or not self.cfg.inventory_aware:
            return sal
        cap = max(sal, 0.98 * mc - self.hold_fin[j])
        if stock <= 1e-9:
            return cap
        lo, hi = self._window(j, obs, sal, widen=3.0, fixed_ok=False)
        seg = revenue_segments(self._curve(j, obs, lo, hi, self.cfg.price_grid), self.w_rev,
                               self._unit_bonus(j))
        return float(max(sal, min(cap, seg.marginal_value_at(stock / rem))))

    # ==================================================================
    # pricing window
    # ==================================================================
    def _mode(self, obs: Observation) -> str:
        if self.cfg.pricing_mode != "meta":
            return self.cfg.pricing_mode
        modes = self.cfg.meta_modes or ["model"]
        if self.meta_arm is None or obs.round - self.meta_start >= self.cfg.meta_block:
            self.meta_arm = self.meta.select(self.rng) if len(modes) > 1 else 0
            self.meta_start = obs.round
        return modes[self.meta_arm]

    def _window(self, j: int, obs: Observation, floor: float, widen: float = 1.0,
                fixed_ok: bool = True) -> Tuple[float, float]:
        """Admissible price interval for product j this round."""
        name = self.products[j]
        ps = self.spec.products[name]
        cfg = self.cfg
        pmax = ps.price_max if ps.price_max is not None else np.inf
        p_last = self.last_prices.get(name)
        model = self.models[name]
        mode = self._mode(obs) if fixed_ok else "model"

        def clip(p):
            return float(min(max(p, floor, ps.price_min), pmax))

        if p_last is None:
            p = clip(self._initial_price(j))
            return (p, p) if fixed_ok else (clip(p / (1 + widen * cfg.early_step)),
                                           clip(p * (1 + widen * cfg.early_step)))
        if fixed_ok and not model.ready:
            if model.zero_demand_price is not None:
                p = clip(0.8 * min(model.zero_demand_price, p_last))     # nobody bought: come down
            else:
                p = clip(p_last)                                         # nothing was on sale: hold
            return p, p
        rng = self.comp.price_range(name)
        if fixed_ok and mode == "fixed":
            p = clip(self._initial_price(j))
            return p, p
        if fixed_ok and mode == "follower" and rng is not None:
            p = clip(rng[0] * (1.0 - cfg.undercut))
            return p, p
        if fixed_ok and mode == "bandit":
            if name not in self.bandits:
                lo_b = max(floor, ps.price_min, 1e-9) * 1.02
                hi_b = 0.9 * pmax if np.isfinite(pmax) else 4.0 * max(floor, p_last)
                self.bandits[name] = ContextualPriceBandit(list(np.geomspace(lo_b, max(hi_b, lo_b * 1.5), 9)))
            ctx_price = self.comp.predict(name)
            p = clip(self.bandits[name].select((ctx_price / p_last) if ctx_price else 1.0, self.rng))
            return p, p
        early = len(self.price_hist[name]) < cfg.early_rounds
        up = (cfg.early_step if early else cfg.max_price_step_up) * widen
        dn = (cfg.early_step if early else cfg.max_price_step_down) * widen
        lo, hi = p_last * max(0.05, 1.0 - dn), p_last * (1.0 + up)
        if cfg.comp_band and rng is not None:
            stats = self.comp.price_stats(name) or (rng[0], 0.5 * (rng[0] + rng[1]), rng[1])
            top_ref = stats[1] if cfg.band_ref == "mean" else stats[2]
            b_lo, b_hi = rng[0] * (1.0 - cfg.band_under), top_ref * (1.0 + cfg.band_over)
            if cfg.band_lift_when_short and fixed_ok and model.ready and b_hi < hi:
                # Short of stock, price is the only way to ration it, so the band must not
                # hold us down: lift it while we would still sell out at its top.
                avail = obs.fin_inventory.get(name, 0.0)
                feas = [obs.raw_inventory.get(r, 0.0) / a for r, a in ps.recipe.items() if a > 0]
                extra = min(feas) if feas else 0.0
                if self.spec.capacity is not None:
                    extra = min(extra, self.spec.capacity / max(ps.capacity_use, 1e-9))
                th = self._params(name, obs)
                d_top = float(model.predict(b_hi, self.comp.predict(name), th)) if th else 0.0
                if d_top > 1.05 * (avail + extra):
                    b_hi = np.inf
            if b_hi < lo:
                hi = lo            # rivals are far below us: move down as fast as allowed
            elif b_lo > hi:
                lo = hi            # rivals are far above us: move up as fast as allowed
            else:
                lo, hi = max(lo, b_lo), min(hi, b_hi)
        lo, hi = clip(lo), clip(hi)
        return (lo, max(lo, hi))

    # ==================================================================
    # planning
    # ==================================================================
    def _plan_input(self, obs: Observation, rem: int, fin: np.ndarray, raw: np.ndarray,
                    fixed_ok: bool, widen: float, cash: Optional[float],
                    price_override: Optional[Dict[str, float]] = None):
        omega = self._omega(rem)
        mc = self.c + self.A @ omega
        curves, segs, mu, floors = [], [], np.zeros(len(self.products)), np.zeros(len(self.products))
        keep_cap = np.zeros(len(self.products))
        for j in range(len(self.products)):
            mu[j] = self._leftover_value(j, rem, float(mc[j]), float(fin[j]), obs)
            floors[j] = max(mu[j], self.salv_fin[j]) * (1.0 + self.cfg.min_margin) if rem >= 2 else self.salv_fin[j]
            if price_override and self.products[j] in price_override:
                ps = self.spec.products[self.products[j]]
                p_fix = max(price_override[self.products[j]], floors[j], ps.price_min, 1e-9)
                lo = hi = min(p_fix, ps.price_max) if ps.price_max is not None else p_fix
            else:
                lo, hi = self._window(j, obs, floors[j], widen=widen, fixed_ok=fixed_ok)
            curve = self._curve(j, obs, lo, hi, self.cfg.price_grid)
            curves.append(curve)
            segs.append(revenue_segments(curve, self.w_rev, self._unit_bonus(j)))
            if rem >= 2:
                # stock we can expect to sell soon keeps its value; anything beyond is dead weight
                p_ref = self.last_prices.get(self.products[j]) or self._initial_price(j)
                keep_cap[j] = min(self.cfg.fin_cover_rounds, rem - 1) * curve.demand_at(
                    min(max(p_ref, curve.prices[-1]), curve.prices[0]))
        inp = PlanInput(products=self.products, resources=self.resources, recipe=self.A,
                        unit_cost=self.c, capacity_use=self.kappa, curves=curves, segments=segs,
                        fin_stock=np.maximum(fin, 0.0), raw_stock=np.maximum(raw, 0.0),
                        raw_use_cost=omega, leftover_value=mu, capacity=self.spec.capacity, cash=cash,
                        leftover_cap=keep_cap, leftover_low=self.salv_fin)
        return inp, mc, floors

    def _vec(self, d: Dict[str, float], names: List[str]) -> np.ndarray:
        return np.array([max(0.0, float(d.get(n, 0.0))) for n in names])

    def _cash_reserve(self, obs: Observation) -> float:
        if self.initial_cash is None:
            self.initial_cash = obs.cash
        return self.cfg.cash_reserve_frac * max(self.initial_cash or 0.0, 0.0)

    def _plan_current(self, obs: Observation, rem: int, extra_raw: Optional[np.ndarray] = None):
        fin = self._vec(obs.fin_inventory, self.products)
        raw = self._vec(obs.raw_inventory, self.resources)
        if extra_raw is not None:
            raw = raw + extra_raw
        cash = max(0.0, obs.cash - self._cash_reserve(obs))
        inp, mc, floors = self._plan_input(obs, rem, fin, raw, True, 1.0, cash)
        plan = self.planner.solve(inp)
        production = plan.production.copy()
        prices = plan.prices.copy()
        raw_left = raw - plan.raw_use
        cap_left = (self.spec.capacity - float(self.kappa @ production)
                    if self.spec.capacity is not None else np.inf)
        for j, name in enumerate(self.products):
            # nothing to sell: do not let the price drift
            if fin[j] + production[j] <= 1e-9 and name in self.last_prices:
                prices[j] = self.last_prices[name]
            # safety stock against demand noise (a stock-out also blinds the demand model)
            model = self.models[name]
            if rem >= 2 and model.ready and self.cfg.safety_z > 0 and plan.sales[j] > 0:
                # top the shelf up to planned sales plus a buffer; stock already there counts
                extra = (plan.sales[j] * (1.0 + self.cfg.safety_z * min(model.noise_cv, 0.5))
                         - fin[j] - production[j])
                extra = min(extra, cap_left / max(self.kappa[j], 1e-9))
                for r in range(len(self.resources)):
                    if self.A[j, r] > 0:
                        extra = min(extra, raw_left[r] / self.A[j, r])
                if self.c[j] > 0:
                    spent = float(self.c @ production)
                    extra = min(extra, max(0.0, cash - spent) / self.c[j])
                if extra > 1e-9:
                    production[j] += extra
                    raw_left -= self.A[j] * extra
                    cap_left -= self.kappa[j] * extra
        return plan, inp, production, prices, mc, floors, raw_left

    def _apply_dither(self, obs: Observation, prices: np.ndarray, inp: PlanInput) -> np.ndarray:
        cfg = self.cfg
        if cfg.dither <= 0 or self._mode(obs) != "model":
            return prices
        horizon = self.spec.horizon or (obs.round + cfg.assumed_remaining)
        if obs.round > max(4, cfg.explore_frac * horizon):
            return prices
        if obs.round != self._dither_round:
            self._dither_round = obs.round
            self._dither = {}
        out = prices.copy()
        for j, name in enumerate(self.products):
            hist = self.price_hist[name][-4:]
            if len(hist) < 2 or not self.models[name].ready:
                continue
            if np.std(hist) / max(np.mean(hist), 1e-9) > 0.01:
                continue
            if name not in self._dither:
                self._dither[name] = float(self.rng.choice([-1.0, 1.0]))
            p = out[j] * (1.0 + self._dither[name] * cfg.dither)
            lo, hi = inp.curves[j].prices[-1], inp.curves[j].prices[0]
            if hi - lo > 1e-9:
                p = min(max(p, lo), hi)
            out[j] = p
        return out

    # ==================================================================
    # procurement
    # ==================================================================
    def _value_tranches(self, r: int, base: PlanInput, lot_qty: float, rem_d: int,
                        use_rate_hint: float, reserve: float, fast: bool) -> List[Tranche]:
        cfg = self.cfg
        rho = self._rho()
        J = len(self.products)
        s_max = np.array([base.curves[j].qty[-1] for j in range(J)])
        if self.spec.capacity is not None:
            s_max = np.minimum(s_max, self.spec.capacity / np.maximum(self.kappa, 1e-9))
        q_useful = float(self.A[:, r] @ s_max) - base.raw_stock[r]
        q_cap = min(lot_qty, max(q_useful, 0.0))
        tranches: List[Tranche] = []
        if not cfg.coordinated_bids:
            # ablation: no joint valuation, bid a fixed share of the best unit margin
            top, need = 0.0, -base.raw_stock[r]
            for j in range(J):
                if self.A[j, r] <= 0:
                    continue
                p_ref = self.last_prices.get(self.products[j]) or self._initial_price(j)
                top = max(top, (p_ref - self.c[j]) / max(self.A[j].sum(), 1e-9))
                need += self.A[j, r] * base.curves[j].demand_at(p_ref) * (1.0 + cfg.coverage_rounds)
            need = min(lot_qty, need)
            if need > 1e-9 and top > 0:
                tranches.append(Tranche(float(need), float(cfg.simple_bid_frac * top)))
            return tranches
        if q_cap > 1e-9:
            n = 2 if fast else max(1, cfg.n_tranches)
            levels = np.linspace(0.0, q_cap, n + 1)
            use_cost = base.raw_use_cost.copy()
            use_cost[r] = 0.0
            pp = np.where(np.arange(len(self.resources)) == r, np.inf, np.maximum(rho, 1e-9))
            values = []
            for q in levels:
                raw = base.raw_stock.copy()
                raw[r] += q
                inp = PlanInput(products=base.products, resources=base.resources, recipe=base.recipe,
                                unit_cost=base.unit_cost, capacity_use=base.capacity_use,
                                curves=base.curves, segments=base.segments, fin_stock=base.fin_stock,
                                raw_stock=raw, raw_use_cost=use_cost,
                                leftover_value=base.leftover_value, capacity=base.capacity,
                                cash=None, purchase_price=pp, leftover_cap=base.leftover_cap,
                                leftover_low=base.leftover_low)
                values.append(self.planner.solve(inp).value)
            mv = np.diff(values) / np.maximum(np.diff(levels), 1e-12)
            mv = np.minimum.accumulate(np.maximum(mv, 0.0)) * (1.0 - cfg.value_haircut)
            for dq, v in zip(np.diff(levels), mv):
                if v > 1e-9:
                    tranches.append(Tranche(float(dq), float(v)))
        # extra cover: only worth buying below the typical price
        if rem_d >= 2 and rho[r] > 0 and cfg.coverage_rounds > 0 and cfg.inventory_aware:
            v_hold = cfg.stockahead_discount * rho[r] - self.hold_raw[r]
            if v_hold > reserve:
                at_typical = sum(t.qty for t in tranches if t.value >= rho[r])
                use_rate = max(use_rate_hint, at_typical)
                target = min(cfg.coverage_rounds, rem_d) * use_rate
                # finished goods on the shelf are cover too
                have = (base.raw_stock[r] + float(self.A[:, r] @ base.fin_stock)
                        + sum(t.qty for t in tranches if t.value >= v_hold))
                extra = target - have
                if extra > 1e-9:
                    tranches.append(Tranche(float(extra), float(v_hold)))
        return tranches

    def _bids(self, obs: Observation, rem_d: int, fin_proj: np.ndarray, raw_proj: np.ndarray,
              use_rate: np.ndarray, budget: float, t_start: float,
              current_prices: Optional[Dict[str, float]] = None):
        if rem_d < 1 or budget <= 0 or not obs.lots:
            return [], {}, {}
        override = None
        if self._mode(obs) != "model":
            # rule-driven prices: value inputs at the price we are actually charging
            override = {n: (current_prices or {}).get(n) or self.last_prices.get(n)
                        or self._initial_price(j) for j, n in enumerate(self.products)}
        base, _, _ = self._plan_input(obs, rem_d, fin_proj, raw_proj, False, 2.0, None, override)
        lots: Dict[str, List[float]] = {}
        fmts: Dict[str, str] = {}
        for lot in obs.lots:
            if lot.resource not in self.spec.resources or lot.quantity <= 0:
                continue
            cur = lots.setdefault(lot.resource, [0.0, 0.0])
            cur[0] += lot.quantity
            cur[1] = max(cur[1], lot.reserve_price)
            fmts[lot.resource] = lot.format or self.spec.resources[lot.resource].auction_format
        requests, caps = [], {}
        for res, (qty, reserve) in lots.items():
            r = self.resources.index(res)
            fast = (time.perf_counter() - t_start) * 1000.0 > self.cfg.time_budget_ms
            tr = self._value_tranches(r, base, qty, rem_d, float(use_rate[r]), reserve, fast)
            if tr:
                caps[res] = max(t.value for t in tr)
                requests.append((res, fmts[res], qty, reserve, tr))
        bids, expected = self.bidder.make_bids(requests, budget)
        return bids, expected, caps

    # ==================================================================
    # main entry
    # ==================================================================
    def decide(self, obs: Observation) -> Action:
        t0 = time.perf_counter()
        self.decisions += 1
        self._update_market(obs)
        self._update_auctions(obs)
        self._ensure_auction_priors()
        rem = self._remaining(obs)
        L = max(0, int(self.spec.lead_time))
        want_bids = obs.phase in ("all", "bid") and bool(obs.lots)
        want_sell = obs.phase in ("all", "sell")
        reserve = self._cash_reserve(obs)
        free_cash = max(0.0, obs.cash - reserve)
        R = len(self.resources)

        action = Action()
        ctx = RiskContext(integer=self.cfg.integer_quantities,
                          liquidating=bool(self.spec.horizon) and rem <= self.cfg.liquidate_last_rounds)
        expected_vec = np.zeros(R)

        if L == 0:
            if want_bids:
                fin = self._vec(obs.fin_inventory, self.products)
                raw = self._vec(obs.raw_inventory, self.resources)
                bids, expected, caps = self._bids(obs, rem, fin, raw, np.zeros(R),
                                                  self.cfg.max_spend_frac * free_cash, t0)
                action.bids, ctx.bid_cap = bids, caps
                expected_vec = np.array([expected.get(r, 0.0) for r in self.resources])
            plan_out = self._plan_current(obs, rem, expected_vec if obs.phase == "all" else None) \
                if want_sell else None
        else:
            plan_out = self._plan_current(obs, rem)
            if want_bids:
                plan, inp, production, prices, mc, floors, raw_left = plan_out
                pending = self._vec(obs.pending_raw, self.resources)
                fin_proj = np.maximum(0.0, plan.end_inventory + (production - plan.production))
                spend_now = float(self.c @ production) if want_sell else 0.0
                budget = self.cfg.max_spend_frac * max(0.0, free_cash - spend_now)
                bids, expected, caps = self._bids(obs, rem - L, fin_proj,
                                                  np.maximum(0.0, raw_left) + pending,
                                                  self.A.T @ production, budget, t0,
                                                  {n: float(prices[j]) for j, n in enumerate(self.products)})
                action.bids, ctx.bid_cap = bids, caps

        if plan_out is not None:
            plan, inp, production, prices, mc, floors, raw_left = plan_out
            if want_sell:
                prices = self._apply_dither(obs, prices, inp)
                for j, name in enumerate(self.products):
                    action.prices[name] = float(prices[j])
                    if production[j] > 1e-9:
                        action.production[name] = float(production[j])
                    ctx.price_floor[name] = float(floors[j])
                    ctx.fallback_price[name] = float(self.last_prices.get(name, self._initial_price(j)))
                    self.last_mc[name] = float(mc[j])
                ctx.production_cash = free_cash
                ctx.extra_raw = {r: float(expected_vec[i]) for i, r in enumerate(self.resources)}
            action.info.update(solver=plan.solver, plan_value=plan.value)

        ctx.bid_budget = free_cash
        action = self.risk.enforce(action, obs, ctx)
        if want_sell:
            self.prev_meta_arm = self.meta_arm if self.cfg.pricing_mode == "meta" else None
            for name, p in action.prices.items():
                self.last_prices[name] = p
                self.price_hist[name].append(p)
        action.info.update(mode=self._mode(obs), remaining=rem,
                           ms=(time.perf_counter() - t0) * 1000.0)
        return action
