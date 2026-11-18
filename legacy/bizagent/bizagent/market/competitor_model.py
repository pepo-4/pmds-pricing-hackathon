"""Lightweight opponent tracking.

Two outputs matter for decisions:

* a forecast of the rival price aggregate for the coming round, per product;
* the conjectural variation kappa: how much of our own price move rivals
  matched one round later. If rivals follow us (kappa near 1) a price cut
  buys no share, so the planner should treat demand as less elastic.

A per-firm label (static / follower / discounter / premium / volatile) is
kept for diagnostics and for the meta-strategy.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


class CompetitorModel:
    def __init__(self, feature: str = "blend", cv_max: float = 0.8, discount: float = 0.9):
        self.feature = feature
        self.cv_max = cv_max
        self.discount = discount
        self.hist: Dict[str, Dict[str, List[float]]] = {}     # firm -> product -> prices
        self.agg_hist: Dict[str, List[float]] = {}            # product -> aggregate
        self.own_hist: Dict[str, List[float]] = {}            # product -> our price
        self._num: Dict[str, float] = {}
        self._den: Dict[str, float] = {}

    # ------------------------------------------------------------------
    def _aggregate(self, prices: List[float]) -> Optional[float]:
        vals = [p for p in prices if p is not None and np.isfinite(p) and p > 0]
        if not vals:
            return None
        if self.feature == "min":
            return float(min(vals))
        if self.feature == "mean":
            return float(np.mean(vals))
        return float(0.5 * min(vals) + 0.5 * np.mean(vals))

    def update(self, comp_prices: Dict[str, Dict[str, float]], own_prices: Dict[str, float]) -> None:
        """Both arguments refer to the same (previous) round."""
        by_product: Dict[str, List[float]] = {}
        for firm, prices in (comp_prices or {}).items():
            for prod, p in prices.items():
                self.hist.setdefault(firm, {}).setdefault(prod, []).append(p)
                by_product.setdefault(prod, []).append(p)
        for prod, vals in by_product.items():
            agg = self._aggregate(vals)
            if agg is None:
                continue
            a_hist = self.agg_hist.setdefault(prod, [])
            o_hist = self.own_hist.setdefault(prod, [])
            own = own_prices.get(prod)
            # rivals' move this round vs our move one round earlier
            if len(a_hist) >= 1 and len(o_hist) >= 2 and o_hist[-1] and o_hist[-2]:
                d_agg = agg / a_hist[-1] - 1.0
                d_own = o_hist[-1] / o_hist[-2] - 1.0
                self._num[prod] = self.discount * self._num.get(prod, 0.0) + d_agg * d_own
                self._den[prod] = self.discount * self._den.get(prod, 0.0) + d_own * d_own
            a_hist.append(agg)
            o_hist.append(own if own else (o_hist[-1] if o_hist else None))

    # ------------------------------------------------------------------
    def last(self, product: str) -> Optional[float]:
        h = self.agg_hist.get(product)
        return h[-1] if h else None

    def predict(self, product: str) -> Optional[float]:
        h = self.agg_hist.get(product)
        if not h:
            return None
        if len(h) == 1:
            return h[-1]
        ew = h[0]
        for v in h[1:]:
            ew = 0.5 * ew + 0.5 * v
        return 0.7 * h[-1] + 0.3 * ew

    def price_range(self, product: str) -> Optional[tuple]:
        vals = []
        for firm, prods in self.hist.items():
            if product in prods and prods[product]:
                vals.append(prods[product][-1])
        vals = [v for v in vals if v and v > 0]
        if not vals:
            return None
        return float(min(vals)), float(max(vals))

    def price_stats(self, product: str) -> Optional[tuple]:
        """(min, mean, max) of rivals' latest prices."""
        vals = [prods[product][-1] for prods in self.hist.values()
                if product in prods and prods[product]]
        vals = [v for v in vals if v and v > 0]
        if not vals:
            return None
        return float(min(vals)), float(np.mean(vals)), float(max(vals))

    def conjectural_variation(self, product: str) -> float:
        den = self._den.get(product, 0.0)
        if den <= 0:
            return 0.0
        kappa = self._num.get(product, 0.0) / (den + 1e-3)   # ridge: shrink to 0 with little data
        return float(min(self.cv_max, max(0.0, kappa)))

    def classify(self) -> Dict[str, str]:
        labels: Dict[str, str] = {}
        for firm, prods in self.hist.items():
            cvs, rel = [], []
            for prod, series in prods.items():
                s = np.array(series[-8:], dtype=float)
                if len(s) < 3 or s.mean() <= 0:
                    continue
                cvs.append(s.std() / s.mean())
                agg = self.agg_hist.get(prod, [])
                if agg:
                    rel.append(s[-1] / max(agg[-1], 1e-9))
            if not cvs:
                labels[firm] = "unknown"
                continue
            cv, r = float(np.mean(cvs)), float(np.mean(rel)) if rel else 1.0
            if cv < 0.01:
                labels[firm] = "static"
            elif cv > 0.15:
                labels[firm] = "volatile"
            elif r < 0.93:
                labels[firm] = "discounter"
            elif r > 1.10:
                labels[firm] = "premium"
            else:
                labels[firm] = "adaptive"
        return labels
