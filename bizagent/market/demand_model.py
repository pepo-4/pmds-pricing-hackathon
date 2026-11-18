"""Bayesian linear demand model with forgetting.

    demand = a + b * own_price + c * competitor_price        (b < 0, c >= 0)

Why linear: with a handful of observations per game a two/three-parameter
local model is all the data can identify, and a linear model refitted with
forgetting tracks a non-linear or drifting curve locally (Besbes & Zeevi 2015;
den Boer 2015). The posterior gives Thompson samples for cheap, directed
exploration.

Everything is scale-free: prices and quantities are normalised by the first
informative observation, so the same priors work whether the market sells 5
units at 3 EUR or 50 000 units at 900 EUR.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


class BayesianDemandModel:
    def __init__(self, prior_elasticity: float = 2.0, cross_frac: float = 0.5,
                 forgetting: float = 0.95, censor_bump: float = 0.10,
                 min_elasticity_frac: float = 0.15):
        self.e0 = max(0.2, float(prior_elasticity))
        self.cross_frac = min(0.9, max(0.0, float(cross_frac)))
        self.rho = min(1.0, max(0.5, float(forgetting)))
        self.censor_bump = censor_bump
        self.min_el_frac = float(min_elasticity_frac)
        self.ready = False
        self.n_obs = 0
        self.p_ref = 1.0
        self.q_ref = 1.0
        self.zero_demand_price: Optional[float] = None   # lowest price at which we sold nothing
        self.censor_streak = 0                           # consecutive stock-outs
        # set in _init_prior
        self.theta0 = np.zeros(3)
        self.L0 = np.eye(3)
        self.L = np.eye(3)
        self.eta = np.zeros(3)
        self.sigma2 = 0.2 ** 2
        self._mx = np.array([1.0, 1.0, 1.0])   # EW mean of regressors
        self._my = 1.0                          # EW mean of response

    # ------------------------------------------------------------------
    def _init_prior(self, price: float, comp_price: Optional[float], sales: float) -> None:
        self.p_ref = max(price, 1e-9)
        self.q_ref = max(sales, 1e-9)
        pc = (comp_price / self.p_ref) if comp_price else 1.0
        b0 = -self.e0
        c0 = self.e0 * self.cross_frac
        a0 = 1.0 - b0 * 1.0 - c0 * pc
        self.theta0 = np.array([a0, b0, c0])
        sd = np.array([1.5, 1.0, 0.6]) * max(1.0, self.e0 / 2.0)
        self.L0 = np.diag(1.0 / sd ** 2)
        self.L = self.L0.copy()
        self.eta = self.L0 @ self.theta0
        self._mx = np.array([1.0, 1.0, pc])
        self._my = 1.0
        self.ready = True

    def _x(self, price: float, comp_price: Optional[float]) -> np.ndarray:
        pc = (comp_price / self.p_ref) if comp_price else self._mx[2]
        return np.array([1.0, price / self.p_ref, pc])

    # ------------------------------------------------------------------
    def observe(self, price: float, comp_price: Optional[float], sales: float,
                censored: bool = False) -> None:
        """Update with one round. `censored` means we sold out, so true demand >= sales."""
        if price is None or price <= 0 or sales is None or sales < 0:
            return
        if not self.ready:
            if sales <= 0:
                if not censored:
                    z = self.zero_demand_price
                    self.zero_demand_price = price if z is None else min(z, price)
                return
            # a first-round stock-out says little about the level of demand: start well above it
            self.censor_streak = 1 if censored else 0
            self._init_prior(price, comp_price, sales * (1.5 if censored else 1.0))
            self.n_obs = 1
            return
        x = self._x(price, comp_price)
        y = sales / self.q_ref
        pred = float(x @ self.mean())
        if censored:
            # A stock-out is only a lower bound on demand. Selling out again and again means
            # the estimate is far too low, and because we only stock what we expect to sell
            # it would never correct itself. Search upwards geometrically until we stop
            # selling out (bump 10%, 20%, 40%, ... capped at doubling).
            # The correction moves the level of the curve only: a stock-out says nothing
            # about the slope, and feeding it to the regression as a data point would teach
            # the model that demand rises with price whenever we sell out while raising it.
            self.censor_streak += 1
            bump = min(1.0, self.censor_bump * 2.0 ** (self.censor_streak - 1))
            target = max(y, pred) * (1.0 + bump)
            shift = np.array([target - pred, 0.0, 0.0])
            self.eta = self.eta + self.L @ shift
            self.theta0 = self.theta0 + shift
            self._mx = 0.8 * self._mx + 0.2 * x
            self._my = 0.8 * self._my + 0.2 * target
            self.n_obs += 1
            return
        self.censor_streak = 0
        err = y - pred
        self.sigma2 = max(0.05 ** 2, 0.85 * self.sigma2 + 0.15 * err * err)
        r = self.rho
        # stabilised forgetting: old evidence decays towards the prior instead of to nothing
        self.L = r * self.L + (1.0 - r) * self.L0 + np.outer(x, x) / self.sigma2
        self.eta = r * self.eta + (1.0 - r) * (self.L0 @ self.theta0) + x * y / self.sigma2
        self._mx = 0.8 * self._mx + 0.2 * x
        self._my = 0.8 * self._my + 0.2 * y
        self.n_obs += 1

    # ------------------------------------------------------------------
    def mean(self) -> np.ndarray:
        try:
            return np.linalg.solve(self.L, self.eta)
        except np.linalg.LinAlgError:
            return self.theta0.copy()

    def _project(self, th: np.ndarray) -> np.ndarray:
        """Enforce b < 0 and 0 <= c < |b|, keeping the fit at the recent operating point."""
        a, b, c = float(th[0]), float(th[1]), float(th[2])
        b_floor = self.min_el_frac * self.e0
        changed = False
        if not np.isfinite(b) or b > -b_floor:
            b, changed = -b_floor, True
        if not np.isfinite(c) or c < 0.0:
            c, changed = 0.0, True
        if c > 0.9 * abs(b):
            c, changed = 0.9 * abs(b), True
        if changed or not np.isfinite(a):
            a = self._my - b * self._mx[1] - c * self._mx[2]
        return np.array([a, b, c])

    def params(self, rng: Optional[np.random.Generator] = None,
               ts_scale: float = 0.0) -> Tuple[float, float, float]:
        """(a, b, c) in real units: demand = a + b*price + c*comp_price."""
        if not self.ready:
            raise RuntimeError("demand model has no informative observation yet")
        th = self.mean()
        if rng is not None and ts_scale > 0:
            try:
                cov = np.linalg.inv(self.L)
                cov = 0.5 * (cov + cov.T)
                th = rng.multivariate_normal(th, (ts_scale ** 2) * cov, method="cholesky")
            except (np.linalg.LinAlgError, ValueError):
                pass
        a, b, c = self._project(np.asarray(th, dtype=float))
        return a * self.q_ref, b * self.q_ref / self.p_ref, c * self.q_ref / self.p_ref

    def predict(self, price, comp_price: Optional[float] = None, params=None):
        a, b, c = params if params is not None else self.params()
        pc = comp_price if comp_price else self._mx[2] * self.p_ref
        return np.maximum(0.0, a + b * np.asarray(price, dtype=float) + c * pc)

    @property
    def noise_cv(self) -> float:
        """Residual standard deviation as a fraction of typical demand."""
        return float(np.sqrt(self.sigma2) / max(self._my, 0.2))

    @property
    def typical_demand(self) -> float:
        return float(self._my * self.q_ref)
