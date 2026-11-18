"""Bandit components.

* DiscountedTS: Gaussian Thompson sampling with discounting over a handful of
  arms. Used as the slow meta-selector over whole pricing policies.
* ContextualPriceBandit: linear Thompson sampling over a fixed price grid with
  the rival price as context. This is the "V4" pricing head. It learns each
  price separately, so it needs many more rounds than the parametric demand
  model; it is kept as a benchmark and for markets where the demand model is
  badly wrong.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


class DiscountedTS:
    def __init__(self, n_arms: int, discount: float = 0.9, prior_sd: float = 1.0):
        self.n = n_arms
        self.discount = discount
        self.prior_sd = prior_sd
        self.count = np.zeros(n_arms)
        self.total = np.zeros(n_arms)
        self.scale = 1.0   # running reward scale, so priors are scale-free

    def select(self, rng: np.random.Generator) -> int:
        untried = np.where(self.count < 1e-9)[0]
        if len(untried):
            return int(untried[0])
        mean = self.total / self.count
        sd = self.scale * self.prior_sd / np.sqrt(self.count)
        return int(np.argmax(rng.normal(mean, sd)))

    def update(self, arm: int, reward: float) -> None:
        if not np.isfinite(reward):
            return
        self.scale = max(1e-9, 0.9 * self.scale + 0.1 * abs(reward))
        self.count *= self.discount
        self.total *= self.discount
        self.count[arm] += 1.0
        self.total[arm] += reward

    def means(self) -> np.ndarray:
        return np.where(self.count > 0, self.total / np.maximum(self.count, 1e-9), np.nan)


class ContextualPriceBandit:
    """Per-arm Bayesian linear regression of profit on [1, context]."""

    def __init__(self, prices: List[float], discount: float = 0.97, noise: float = 1.0):
        self.prices = np.asarray(prices, dtype=float)
        self.k = len(self.prices)
        self.discount = discount
        self.noise = noise
        self.A = np.stack([np.eye(2) for _ in range(self.k)])
        self.b = np.zeros((self.k, 2))
        self.pulls = np.zeros(self.k)
        self.scale = 1.0
        self.last_arm: Optional[int] = None
        self.last_ctx: Optional[np.ndarray] = None

    def select(self, context: float, rng: np.random.Generator) -> float:
        x = np.array([1.0, context])
        untried = np.where(self.pulls < 0.5)[0]
        if len(untried):
            # visit the grid from the middle outwards: extreme prices are the costly ones
            mid = (self.k - 1) / 2.0
            arm = int(untried[np.argmin(np.abs(untried - mid))])
        else:
            best, arm = -np.inf, 0
            for a in range(self.k):
                cov = np.linalg.inv(self.A[a])
                mu = cov @ self.b[a]
                th = rng.multivariate_normal(mu, (self.noise ** 2) * cov)
                val = float(th @ x)
                if val > best:
                    best, arm = val, a
        self.last_arm, self.last_ctx = arm, x
        return float(self.prices[arm])

    def update(self, reward: float) -> None:
        if self.last_arm is None or not np.isfinite(reward):
            return
        self.scale = max(1e-9, 0.95 * self.scale + 0.05 * abs(reward))
        r = reward / self.scale
        a, x = self.last_arm, self.last_ctx
        self.A[a] = self.discount * self.A[a] + (1 - self.discount) * np.eye(2) + np.outer(x, x)
        self.b[a] = self.discount * self.b[a] + x * r
        self.pulls[a] += 1.0
