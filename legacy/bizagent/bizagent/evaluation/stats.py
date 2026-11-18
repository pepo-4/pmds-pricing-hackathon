"""Statistics for comparing agents honestly.

Comparisons are paired: two agents are always evaluated on the same scenario,
seed and opponents, and the statistic is the per-game difference. Intervals
are percentile bootstrap intervals over games; p-values are Wilcoxon
signed-rank tests with Holm correction across comparisons.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

try:
    from scipy.stats import wilcoxon
except Exception:  # pragma: no cover
    wilcoxon = None


def iqm(x: Sequence[float]) -> float:
    """Interquartile mean: the mean of the middle 50% (robust to a few extreme games)."""
    a = np.sort(np.asarray(x, dtype=float))
    n = len(a)
    if n == 0:
        return float("nan")
    lo, hi = int(np.floor(0.25 * n)), int(np.ceil(0.75 * n))
    return float(a[lo:hi].mean()) if hi > lo else float(a.mean())


def bootstrap_ci(x: Sequence[float], stat=np.mean, n_boot: int = 4000, alpha: float = 0.05,
                 seed: int = 0):
    a = np.asarray(x, dtype=float)
    if len(a) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    vals = np.array([stat(a[i]) for i in idx]) if stat is not np.mean else a[idx].mean(axis=1)
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def cvar_low(x: Sequence[float], q: float = 0.10) -> float:
    """Mean of the worst q share of games."""
    a = np.sort(np.asarray(x, dtype=float))
    k = max(1, int(np.ceil(q * len(a))))
    return float(a[:k].mean()) if len(a) else float("nan")


def holm(pvals: Sequence[float]) -> List[float]:
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def paired(a: Sequence[float], b: Sequence[float], seed: int = 0) -> Dict[str, float]:
    """Compare a against b on the same games (positive = a is better)."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    lo, hi = bootstrap_ci(d, seed=seed)
    p = float("nan")
    if wilcoxon is not None and len(d) >= 6 and np.any(np.abs(d) > 1e-12):
        try:
            p = float(wilcoxon(d, zero_method="wilcox").pvalue)
        except ValueError:
            p = float("nan")
    return {"n": int(len(d)), "mean_diff": float(d.mean()), "median_diff": float(np.median(d)),
            "ci_lo": lo, "ci_hi": hi,
            "p_wilcoxon": p, "p_improve": float((d > 0).mean() + 0.5 * (d == 0).mean())}


def summarise(rows: List[dict]) -> Dict[str, float]:
    s = np.array([r["score_norm"] for r in rows], dtype=float)
    rank = np.array([r["rank"] for r in rows], dtype=float)
    n = np.array([r["n_firms"] for r in rows], dtype=float)
    lo, hi = bootstrap_ci(s)
    return {
        "games": int(len(s)), "mean": float(s.mean()), "ci_lo": lo, "ci_hi": hi, "iqm": iqm(s),
        "median": float(np.median(s)), "sd": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
        "worst10": cvar_low(s), "min": float(s.min()),
        "win_rate": float((rank == 1).mean()), "top2_rate": float((rank <= 2).mean()),
        "rank_pct": float(((n - rank) / np.maximum(n - 1, 1)).mean()),
        "loss_rate": float((s < 0).mean()),
        "ms_mean": float(np.mean([r["ms_mean"] for r in rows])),
        "ms_max": float(np.max([r["ms_max"] for r in rows])),
        "errors": float(np.sum([r["errors"] + r.get("engine_errors", 0.0) for r in rows])),
    }
