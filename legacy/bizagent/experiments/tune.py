"""Random-search tuning of the strategic agent, with a guard against overfitting.

Stage 1 (train split): draw N random configurations, play the same games with each.
Stage 2 (valid split): the best K from stage 1 and the current default play fresh games
with opponents the tuner never saw. A configuration replaces the default only if the 95%
interval of its paired difference against the default lies above zero.

    python experiments/tune.py --configs 24 --games 60 --top 3 --valid-games 150
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bizagent.evaluation.stats import summarise  # noqa: E402
from bizagent.evaluation.tournament import by_label, compare, run_tournament, save, table  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def draw(rng: np.random.Generator) -> dict:
    step = float(rng.uniform(0.10, 0.35))
    return {
        "band_ref": str(rng.choice(["mean", "max"])),
        "band_over": round(float(rng.uniform(0.03, 0.40)), 3),
        "min_elasticity_frac": round(float(rng.uniform(0.10, 0.50)), 3),
        "prior_elasticity": round(float(rng.uniform(1.2, 3.0)), 2),
        "ts_scale": round(float(rng.uniform(0.0, 0.8)), 2),
        "forgetting": round(float(rng.uniform(0.88, 1.0)), 3),
        "coverage_rounds": round(float(rng.uniform(0.5, 2.5)), 2),
        "fin_cover_rounds": round(float(rng.uniform(0.2, 1.5)), 2),
        "win_bias": round(float(rng.uniform(0.8, 2.0)), 2),
        "safety_z": round(float(rng.uniform(0.0, 1.0)), 2),
        "max_price_step_up": round(step, 3),
        "max_price_step_down": round(step, 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", type=int, default=24)
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--valid-games", type=int, default=150)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "tuning"))
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    configs = {"default": {}}
    for i in range(args.configs):
        configs[f"c{i:02d}"] = draw(rng)
    cands = [(label, "v6_hybrid", over) for label, over in configs.items()]
    rows = run_tournament(cands, "train", args.games, args.workers)
    save(rows, args.out + "_train")
    groups = by_label(rows)
    # rank on the interquartile mean plus the worst decile: reward level and reliability
    def key(label):
        s = summarise(groups[label])
        return 0.75 * s["iqm"] + 0.25 * s["worst10"]
    ranked = sorted(configs, key=key, reverse=True)
    top = [l for l in ranked if l != "default"][: args.top]
    text = [f"# Tuning - stage 1: {args.configs} random configurations, {args.games} train games each",
            "", table(rows, "default", ranked), ""]

    vc = [("default", "v6_hybrid", {})] + [(l, "v6_hybrid", configs[l]) for l in top]
    vrows = run_tournament(vc, "valid", args.valid_games, args.workers)
    save(vrows, args.out + "_valid")
    cmp_ = compare(vrows, "default")
    text += [f"# Stage 2: top {args.top} against the default, {args.valid_games} validation games",
             "", table(vrows, "default", ["default"] + top), ""]
    accepted = [l for l in top if cmp_[l]["ci_lo"] > 0]
    best = max(accepted, key=lambda l: cmp_[l]["mean_diff"]) if accepted else None
    text += ["## Decision", "",
             (f"`{best}` beats the default on validation (95% interval of the paired difference "
              f"above zero) and is adopted: `{json.dumps(configs[best])}`") if best else
             "No configuration beat the default with a 95% interval above zero on validation. "
             "The default is kept.", ""]
    for l in top:
        text.append(f"- `{l}`: `{json.dumps(configs[l])}`")
    with open(args.out + ".md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(text) + "\n")
    with open(args.out + ".json", "w", encoding="utf-8") as fh:
        json.dump({"configs": configs, "ranked": ranked, "top": top, "accepted": accepted,
                   "best": best, "best_config": configs.get(best) if best else None,
                   "seconds": time.time() - t0}, fh, indent=2)
    print("\n".join(text))
    print(f"[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
