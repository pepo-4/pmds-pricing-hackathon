"""Paired tournament runner.

A *game* is one scenario (drawn from a seed) with one candidate of ours and
n-1 opponents drawn from the opponent pool with the same seed. Every candidate
plays exactly the same games, so candidates are compared on per-game
differences.

Scenario seeds and opponent pools are split three ways so that tuning cannot
leak into the reported numbers:

  train  seeds 10000+  linear/logit/segments demand   7 opponents   (the tuner sees only this)
  valid  seeds 20000+  linear/logit/segments demand   all 10        (used to pick between versions)
  test   seeds 30000+  the same + isoelastic demand   all 10        (reported; never used for a choice)

The test split contains a demand family (isoelastic) that no development run
ever used, and three opponents that the tuner never sees.
"""
from __future__ import annotations

import csv
import json
import os
from multiprocessing import Pool
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..agents.registry import make_agent
from ..simulation.environment import Game
from ..simulation.opponents import HELDOUT_POOL, OPPONENTS, TRAIN_POOL
from ..simulation.scenarios import make_scenario
from .stats import holm, paired, summarise

SPLITS = {
    "train": dict(base=10000, demands=("linear", "logit", "segments"), pool=TRAIN_POOL),
    "valid": dict(base=20000, demands=("linear", "logit", "segments"), pool=list(OPPONENTS)),
    "test": dict(base=30000, demands=("linear", "logit", "segments", "isoelastic"),
                 pool=list(OPPONENTS)),
}

# a candidate is (label, registry name, overrides)
Candidate = Tuple[str, str, dict]


def as_candidate(x) -> Candidate:
    if isinstance(x, str):
        return (x, x, {})
    label, name, over = x
    return (str(label), str(name), dict(over or {}))


def lineup(seed: int, n_opponents: int, pool: Sequence[str]) -> List[str]:
    rng = np.random.default_rng(seed + 777)
    return [str(x) for x in rng.choice(list(pool), size=n_opponents, replace=True)]


def play(task) -> dict:
    cand, split, k, scen_kw = task
    label, name, over = cand
    cfg = SPLITS[split]
    seed = cfg["base"] + int(k)
    sc = make_scenario(seed, demands=cfg["demands"], **(scen_kw or {}))
    opp = lineup(seed, sc.tags["n_firms"] - 1, cfg["pool"])
    agents = [make_agent(name, label=label, **over)] + [make_agent(o) for o in opp]
    rows = Game(sc, agents).run()
    me = dict(rows[0])
    me.update(label=label, split=split, seed=seed, scenario=sc.name,
              best_opp=max(r["score_norm"] for r in rows[1:]),
              mean_opp=float(np.mean([r["score_norm"] for r in rows[1:]])),
              opponents="|".join(opp))
    me.update({f"tag_{k_}": v for k_, v in sc.tags.items()})
    return me


def run_tournament(candidates: Iterable, split: str = "valid", n_games: int = 100,
                   workers: int = 2, scen_kw: Optional[dict] = None, start: int = 0) -> List[dict]:
    cands = [as_candidate(c) for c in candidates]
    tasks = [(c, split, k, scen_kw) for k in range(start, start + n_games) for c in cands]
    if workers <= 1:
        return [play(t) for t in tasks]
    with Pool(workers) as pool:
        return pool.map(play, tasks, chunksize=max(1, len(tasks) // (workers * 8)))


def league(names: Sequence[str], split: str = "test", n_games: int = 50, start: int = 0,
           fill: int = 2) -> List[dict]:
    """Free-for-all: all listed agents in the same market, plus `fill` random opponents."""
    cfg = SPLITS[split]
    out = []
    for k in range(start, start + n_games):
        seed = cfg["base"] + 5000 + k
        sc = make_scenario(seed, demands=cfg["demands"], n_firms=len(names) + fill)
        opp = lineup(seed, fill, cfg["pool"])
        agents = [make_agent(n) for n in names] + [make_agent(o) for o in opp]
        for r in Game(sc, agents).run():
            r.update(label=r["agent"], seed=seed, split=split)
            out.append(r)
    return out


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def by_label(rows: List[dict]) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for r in rows:
        out.setdefault(r["label"], []).append(r)
    for v in out.values():
        v.sort(key=lambda r: r["seed"])
    return out


def compare(rows: List[dict], ref: str, metric: str = "score_norm") -> Dict[str, dict]:
    groups = by_label(rows)
    base = {r["seed"]: r[metric] for r in groups[ref]}
    out = {}
    for label, rs in groups.items():
        if label == ref:
            continue
        seeds = [r["seed"] for r in rs if r["seed"] in base]
        a = [r[metric] for r in rs if r["seed"] in base]
        b = [base[s] for s in seeds]
        out[label] = paired(a, b)
    adj = holm([v["p_wilcoxon"] if np.isfinite(v["p_wilcoxon"]) else 1.0 for v in out.values()])
    for v, p in zip(out.values(), adj):
        v["p_holm"] = p
    return out


def table(rows: List[dict], ref: Optional[str] = None, order: Optional[Sequence[str]] = None) -> str:
    groups = by_label(rows)
    labels = [l for l in (order or groups) if l in groups]
    cmp_ = compare(rows, ref) if ref and ref in groups else {}
    head = ("| agent | games | median score | IQM | mean score | 95% CI of mean | worst 10% | win rate "
            "| top-2 | loss rate | ms/decision (mean / max) | errors |")
    if cmp_:
        head += f" median diff vs {ref} | mean diff | 95% CI of mean diff | P(better) | p (Holm) |"
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for l in labels:
        s = summarise(groups[l])
        line = (f"| {l} | {s['games']} | {s['median']:.4f} | {s['iqm']:.4f} | {s['mean']:.4f} | "
                f"[{s['ci_lo']:.4f}, {s['ci_hi']:.4f}] | {s['worst10']:.4f} | {100 * s['win_rate']:.1f}% | "
                f"{100 * s['top2_rate']:.1f}% | {100 * s['loss_rate']:.1f}% | "
                f"{s['ms_mean']:.1f} / {s['ms_max']:.0f} | {s['errors']:.0f} |")
        if cmp_:
            if l in cmp_:
                c = cmp_[l]
                line += (f" {c['median_diff']:+.4f} | {c['mean_diff']:+.4f} | "
                         f"[{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}] | "
                         f"{100 * c['p_improve']:.0f}% | {c['p_holm']:.3g} |")
            else:
                line += " (reference) | | | | |"
        lines.append(line)
    return "\n".join(lines)


def breakdown(rows: List[dict], tag: str, labels: Sequence[str]) -> str:
    groups = by_label(rows)
    values = sorted({str(r.get(f"tag_{tag}")) for r in rows})
    lines = [f"| {tag} | games | " + " | ".join(labels) + " |", "|---|---|" + "---|" * len(labels)]
    for v in values:
        cells, n = [], 0
        for l in labels:
            rs = [r for r in groups.get(l, []) if str(r.get(f"tag_{tag}")) == v]
            n = max(n, len(rs))
            if rs:
                s = np.array([r["score_norm"] for r in rs])
                cells.append(f"{np.median(s):.3f} ({100 * np.mean([r['rank'] == 1 for r in rs]):.0f}%)")
            else:
                cells.append("-")
        lines.append(f"| {v} | {n} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def save(rows: List[dict], path_base: str, meta: Optional[dict] = None) -> None:
    os.makedirs(os.path.dirname(path_base) or ".", exist_ok=True)
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path_base + ".csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    if meta is not None:
        with open(path_base + ".json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, default=str)


def load(path_base: str) -> List[dict]:
    out = []
    with open(path_base + ".csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            row = {}
            for k, v in r.items():
                try:
                    row[k] = float(v)
                except (TypeError, ValueError):
                    row[k] = v
            out.append(row)
    return out
