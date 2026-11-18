"""Joint production / sales planner for one round.

Revenue is concave in volume once demand is sampled on a price grid and the
upper concave hull is taken, so the whole "how much of each product to make,
from which inputs, and at what price to sell it" question is a small linear
programme:

    max  sum_jk MR_jk * s_jk  - sum_j c_j x_j  - sum_r (w_r u_r + pi_r y_r)
         + sum_j (mu_j e_j + low_j f_j)
    s.t. sum_k s_jk + e_j + f_j - x_j = I_j           (finished goods balance)
         sum_j A_jr x_j - u_r - y_r <= 0              (raw material balance)
         sum_j kappa_j x_j <= K                       (capacity)
         sum_j c_j x_j + sum_r pi_r y_r <= cash       (liquidity)
         0 <= s_jk <= width_jk, 0 <= u_r <= R_r, 0 <= y_r <= ymax_r

s = sales by revenue segment, x = production, e = finished goods carried over
(worth mu each, at most `leftover_cap` units: the stock we can expect to sell
soon), f = stock carried beyond that (worth only `leftover_low`), u = raw
material taken from stock (opportunity cost w), y = raw material bought (at
price pi; only used when valuing resources).

HiGHS solves these in about a millisecond. A greedy solver with the same
interface is the fallback when the LP fails, is slow, or is switched off.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

try:
    from scipy.optimize import linprog
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False


@dataclass
class DemandCurve:
    """Expected demand sampled on a price grid (prices strictly descending)."""
    prices: np.ndarray
    qty: np.ndarray

    def __post_init__(self):
        p = np.asarray(self.prices, dtype=float)
        q = np.asarray(self.qty, dtype=float)
        order = np.argsort(-p)
        p, q = p[order], np.maximum(q[order], 0.0)
        q = np.maximum.accumulate(q)      # demand cannot fall as price falls
        self.prices, self.qty = p, q

    def price_for(self, volume: float) -> float:
        """Highest grid-interpolated price at which expected demand reaches `volume`."""
        q, p = self.qty, self.prices
        if volume <= q[0] or len(q) == 1:
            return float(p[0])
        if volume >= q[-1]:
            return float(p[-1])
        keep = np.concatenate(([True], np.diff(q) > 1e-12))
        return float(np.interp(volume, q[keep], p[keep]))

    def demand_at(self, price: float) -> float:
        return float(np.interp(price, self.prices[::-1], self.qty[::-1]))


@dataclass
class Segments:
    widths: np.ndarray       # units
    values: np.ndarray       # marginal objective value per unit, non-increasing

    def marginal_value_at(self, volume: float) -> float:
        """Marginal value of the next unit once `volume` units are being sold."""
        if len(self.widths) == 0:
            return 0.0
        cum = np.cumsum(self.widths)
        idx = int(np.searchsorted(cum, volume, side="right"))
        if idx >= len(self.values):
            return 0.0
        return float(self.values[idx])


def revenue_segments(curve: DemandCurve, w_revenue: float = 0.0,
                     unit_bonus: float = 0.0) -> Segments:
    """Upper concave hull of (volume, objective value) -> decreasing marginal-value segments."""
    q = np.concatenate(([0.0], curve.qty))
    v = np.concatenate(([0.0], (1.0 + w_revenue) * curve.prices * curve.qty + unit_bonus * curve.qty))
    hull: List[int] = [0]
    for i in range(1, len(q)):
        if q[i] <= q[hull[-1]] + 1e-12:
            if len(hull) > 1 and v[i] > v[hull[-1]]:
                hull[-1] = i
            continue
        while len(hull) >= 2:
            i1, i2 = hull[-2], hull[-1]
            if (v[i2] - v[i1]) * (q[i] - q[i2]) <= (v[i] - v[i2]) * (q[i2] - q[i1]):
                hull.pop()
            else:
                break
        hull.append(i)
    hq, hv = q[hull], v[hull]
    widths = np.diff(hq)
    values = np.diff(hv) / np.maximum(widths, 1e-12)
    keep = (widths > 1e-9) & (values > 1e-9)
    return Segments(widths=widths[keep], values=values[keep])


@dataclass
class PlanInput:
    products: List[str]
    resources: List[str]
    recipe: np.ndarray                    # (n_products, n_resources)
    unit_cost: np.ndarray                 # (n_products,)
    capacity_use: np.ndarray              # (n_products,)
    curves: List[DemandCurve]
    segments: List[Segments]
    fin_stock: np.ndarray                 # (n_products,)
    raw_stock: np.ndarray                 # (n_resources,)
    raw_use_cost: np.ndarray              # opportunity cost of using stocked raw material
    leftover_value: np.ndarray            # mu_j
    capacity: Optional[float] = None
    cash: Optional[float] = None
    purchase_price: Optional[np.ndarray] = None   # pi_r, np.inf where buying is not allowed
    purchase_max: Optional[np.ndarray] = None
    leftover_cap: Optional[np.ndarray] = None     # units per product that keep the value mu
    leftover_low: Optional[np.ndarray] = None     # value of carried stock beyond the cap


@dataclass
class PlanResult:
    production: np.ndarray
    sales: np.ndarray
    end_inventory: np.ndarray
    raw_use: np.ndarray
    purchases: np.ndarray
    prices: np.ndarray
    value: float
    solver: str = "lp"
    raw_dual: Optional[np.ndarray] = None
    info: Dict[str, float] = field(default_factory=dict)


class Planner:
    def __init__(self, solver: str = "lp", time_limit: float = 0.25):
        self.solver = solver if _HAVE_SCIPY else "greedy"
        self.time_limit = time_limit
        self.n_lp = 0
        self.n_fallback = 0

    # ------------------------------------------------------------------
    def solve(self, inp: PlanInput) -> PlanResult:
        if self.solver == "lp":
            try:
                res = self._solve_lp(inp)
                if res is not None:
                    self.n_lp += 1
                    return res
            except Exception:
                pass
            self.n_fallback += 1
        return self._solve_greedy(inp)

    # ------------------------------------------------------------------
    def _finish(self, inp: PlanInput, x, s_tot, e, u, y, value, solver, dual=None) -> PlanResult:
        prices = np.array([inp.curves[j].price_for(max(s_tot[j], 0.0))
                           for j in range(len(inp.products))])
        return PlanResult(production=np.maximum(x, 0.0), sales=np.maximum(s_tot, 0.0),
                          end_inventory=np.maximum(e, 0.0), raw_use=np.maximum(u, 0.0),
                          purchases=np.maximum(y, 0.0), prices=prices, value=float(value),
                          solver=solver, raw_dual=dual)

    def _solve_lp(self, inp: PlanInput) -> Optional[PlanResult]:
        J, R = len(inp.products), len(inp.resources)
        seg_n = [len(s.widths) for s in inp.segments]
        nS = int(sum(seg_n))
        n = nS + 3 * J + 2 * R
        ix_x, ix_e, ix_u, ix_y, ix_f = nS, nS + J, nS + 2 * J, nS + 2 * J + R, nS + 2 * J + 2 * R

        c = np.zeros(n)
        lo = np.zeros(n)
        hi = np.full(n, np.inf)
        A_eq = np.zeros((J, n))
        b_eq = np.asarray(inp.fin_stock, dtype=float).copy()
        k = 0
        for j in range(J):
            m = seg_n[j]
            c[k:k + m] = -inp.segments[j].values
            hi[k:k + m] = inp.segments[j].widths
            A_eq[j, k:k + m] = 1.0
            A_eq[j, ix_e + j] = 1.0
            A_eq[j, ix_f + j] = 1.0
            A_eq[j, ix_x + j] = -1.0
            k += m
        c[ix_x:ix_x + J] = inp.unit_cost
        c[ix_e:ix_e + J] = -inp.leftover_value
        if inp.leftover_cap is not None:
            hi[ix_e:ix_e + J] = np.maximum(inp.leftover_cap, 0.0)
            low = inp.leftover_low if inp.leftover_low is not None else np.zeros(J)
            c[ix_f:ix_f + J] = -np.minimum(low, inp.leftover_value)
        else:
            hi[ix_f:ix_f + J] = 0.0
        c[ix_u:ix_u + R] = inp.raw_use_cost
        hi[ix_u:ix_u + R] = np.maximum(inp.raw_stock, 0.0)
        if inp.purchase_price is not None:
            pp = np.asarray(inp.purchase_price, dtype=float)
            pm = (np.asarray(inp.purchase_max, dtype=float) if inp.purchase_max is not None
                  else np.full(R, np.inf))
            allowed = np.isfinite(pp)
            c[ix_y:ix_y + R] = np.where(allowed, pp, 0.0)
            hi[ix_y:ix_y + R] = np.where(allowed, pm, 0.0)
        else:
            hi[ix_y:ix_y + R] = 0.0

        rows, rhs = [], []
        for r in range(R):
            row = np.zeros(n)
            row[ix_x:ix_x + J] = inp.recipe[:, r]
            row[ix_u + r] = -1.0
            row[ix_y + r] = -1.0
            rows.append(row)
            rhs.append(0.0)
        if inp.capacity is not None and np.isfinite(inp.capacity):
            row = np.zeros(n)
            row[ix_x:ix_x + J] = inp.capacity_use
            rows.append(row)
            rhs.append(max(inp.capacity, 0.0))
        if inp.cash is not None and np.isfinite(inp.cash):
            row = np.zeros(n)
            row[ix_x:ix_x + J] = inp.unit_cost
            row[ix_y:ix_y + R] = c[ix_y:ix_y + R]
            if np.any(row > 0):
                rows.append(row)
                rhs.append(max(inp.cash, 0.0))
        # an unbounded hi on purchases with value above price would make the LP unbounded only if
        # demand were unbounded; segments are finite, so the LP is always bounded.
        res = linprog(c, A_ub=np.array(rows) if rows else None, b_ub=np.array(rhs) if rows else None,
                      A_eq=A_eq, b_eq=b_eq, bounds=np.column_stack([lo, hi]), method="highs",
                      options={"time_limit": self.time_limit, "presolve": True})
        if res.status != 0 or res.x is None or not np.all(np.isfinite(res.x)):
            return None
        z = res.x
        s_tot = np.zeros(J)
        k = 0
        for j in range(J):
            s_tot[j] = z[k:k + seg_n[j]].sum()
            k += seg_n[j]
        dual = None
        try:
            dual = -np.asarray(res.ineqlin.marginals[:R], dtype=float)
        except Exception:
            dual = None
        return self._finish(inp, z[ix_x:ix_x + J], s_tot, z[ix_e:ix_e + J] + z[ix_f:ix_f + J], z[ix_u:ix_u + R],
                            z[ix_y:ix_y + R], -res.fun, "lp", dual)

    # ------------------------------------------------------------------
    def _solve_greedy(self, inp: PlanInput) -> PlanResult:
        """Fill the most valuable (product, segment) first. Exact when products do not
        compete for the same input or capacity, near-optimal otherwise."""
        J, R = len(inp.products), len(inp.resources)
        x = np.zeros(J)
        s_tot = np.zeros(J)
        u = np.zeros(R)
        y = np.zeros(R)
        raw_left = np.maximum(np.asarray(inp.raw_stock, dtype=float).copy(), 0.0)
        stock_left = np.maximum(np.asarray(inp.fin_stock, dtype=float).copy(), 0.0)
        cap_left = inp.capacity if (inp.capacity is not None and np.isfinite(inp.capacity)) else np.inf
        cash_left = inp.cash if (inp.cash is not None and np.isfinite(inp.cash)) else np.inf
        pp = (np.asarray(inp.purchase_price, dtype=float) if inp.purchase_price is not None
              else np.full(R, np.inf))
        pm = (np.asarray(inp.purchase_max, dtype=float).copy() if inp.purchase_max is not None
              else np.full(R, np.inf))
        seg_pos = [0] * J
        seg_left = [s.widths.copy() for s in inp.segments]
        value = 0.0

        cap = (np.maximum(np.asarray(inp.leftover_cap, dtype=float), 0.0)
               if inp.leftover_cap is not None else np.full(J, np.inf))
        low = (np.minimum(np.asarray(inp.leftover_low, dtype=float), inp.leftover_value)
               if inp.leftover_low is not None else np.zeros(J))

        # 1) sell existing stock while a unit is worth more sold than kept
        for j in range(J):
            seg = inp.segments[j]
            while seg_pos[j] < len(seg.widths) and stock_left[j] > 1e-9:
                k = seg_pos[j]
                excess = stock_left[j] - cap[j]
                keep_value = low[j] if excess > 1e-9 else inp.leftover_value[j]
                if seg.values[k] <= keep_value:
                    break
                amt = min(seg_left[j][k], stock_left[j])
                if excess > 1e-9 and seg.values[k] <= inp.leftover_value[j]:
                    amt = min(amt, excess)       # only the excess is worth dumping at this value
                s_tot[j] += amt
                stock_left[j] -= amt
                seg_left[j][k] -= amt
                value += amt * seg.values[k]
                if seg_left[j][k] <= 1e-9:
                    seg_pos[j] += 1

        # 2) produce for the best remaining segment until nothing is profitable
        for _ in range(10 * (sum(len(s.widths) for s in inp.segments) + 1)):
            best, best_j, best_cost = 0.0, -1, 0.0
            for j in range(J):
                k = seg_pos[j]
                if k >= len(inp.segments[j].widths):
                    continue
                cost = inp.unit_cost[j]
                feasible = True
                for r in range(R):
                    a = inp.recipe[j, r]
                    if a <= 0:
                        continue
                    if raw_left[r] > 1e-9:
                        cost += a * inp.raw_use_cost[r]
                    elif np.isfinite(pp[r]) and pm[r] > 1e-9:
                        cost += a * pp[r]
                    else:
                        feasible = False
                        break
                if not feasible:
                    continue
                net = inp.segments[j].values[k] - cost
                score = net / max(inp.capacity_use[j], 1e-9) if np.isfinite(cap_left) else net
                if net > 1e-9 and score > best:
                    best, best_j, best_cost = score, j, cost
            if best_j < 0:
                break
            j, k = best_j, seg_pos[best_j]
            amt = seg_left[j][k]
            if np.isfinite(cap_left):
                amt = min(amt, cap_left / max(inp.capacity_use[j], 1e-9))
            cash_per_unit = inp.unit_cost[j]
            for r in range(R):
                a = inp.recipe[j, r]
                if a <= 0:
                    continue
                if raw_left[r] > 1e-9:
                    amt = min(amt, raw_left[r] / a)
                else:
                    amt = min(amt, pm[r] / a)
                    cash_per_unit += a * pp[r]
            if np.isfinite(cash_left) and cash_per_unit > 0:
                amt = min(amt, cash_left / cash_per_unit)
            if amt <= 1e-9:
                seg_pos[j] = len(inp.segments[j].widths)  # blocked: stop considering this product
                continue
            x[j] += amt
            s_tot[j] += amt
            seg_left[j][k] -= amt
            if np.isfinite(cap_left):
                cap_left -= amt * inp.capacity_use[j]
            if np.isfinite(cash_left):
                cash_left -= amt * cash_per_unit
            for r in range(R):
                a = inp.recipe[j, r]
                if a <= 0:
                    continue
                if raw_left[r] > 1e-9:
                    raw_left[r] -= amt * a
                    u[r] += amt * a
                else:
                    pm[r] -= amt * a
                    y[r] += amt * a
            value += amt * (inp.segments[j].values[k] - best_cost)
            if seg_left[j][k] <= 1e-9:
                seg_pos[j] += 1
        e = stock_left
        kept = np.minimum(e, cap)
        value += float(np.dot(kept, inp.leftover_value) + np.dot(e - kept, low))
        return self._finish(inp, x, s_tot, e, u, y, value, "greedy", None)
