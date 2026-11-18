"""Small evolutionary search over pepo_agent parameters.

Each generation: 8 candidates play together (plus fixed rivals) on both market
scenarios across seeds; the best 3 by average rank survive and are mutated.
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from league import ROOT, play  # noqa: E402

SPACE = {
    "init_markup": [3.0, 4.0, 5.0, 6.0, 8.0, 10.0],
    "up": [1.03, 1.05, 1.08, 1.12],
    "hold_down": [0.95, 0.98, 1.0],
    "down": [0.9, 0.93, 0.96, 0.98],
    "min_markup": [1.25, 1.6, 2.0, 2.5],
    "target_min": [3, 5, 6, 8, 10, 14],
    "target_cover": [1.0, 2.0, 3.0, 4.0],
    "bid_premium": [0.0, 0.02, 0.1],
    "premium_up": [0.1, 0.25, 0.5],
    "explore_turns": [0, 2, 3, 4],
    "explore_keep": [0.85, 1.0, 1.1],
    "explore_down": [0.6, 0.7, 0.8],
    "streak_free": [0, 1, 2, 3],
    "streak_down": [0.85, 0.9, 0.95, 1.0],
    "start_stock": [1, 2, 3, 5],
    "ramp": [0.5, 1.0, 2.0],
    "ladder": [
        [1.0, 0.9, 1.12, 0.8, 1.25, 1.4],
        [1.0, 0.95, 1.05, 0.9, 1.1, 1.15],
        [1.0, 0.85, 1.2, 0.7, 1.45, 1.7],
        [1.0, 1.0, 1.0, 1.0],
    ],
}
FIXED = {
    "base": ("pepo_agent", {}),
    "B_old": ("pepo_agent", {"down": 0.93, "explore_turns": 0, "target_min": 14, "init_markup": 6.0,
                             "hold_down": 1.0, "bid_premium": 0.1, "streak_down": 1.0, "start_stock": 14}),
    "pepo_baseline": ("pepo_baseline", {}),
    "Dummy": ("DummyAgent", {}),
    "PW": ("PriceWarAgent", {}),
}


def mutate(cfg, rng, k=2):
    new = dict(cfg)
    for key in rng.sample(list(SPACE), k):
        new[key] = rng.choice(SPACE[key])
    return new


def main():
    rng = random.Random(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    gens = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    start = {"down": 0.93, "explore_turns": 0, "target_min": 14, "init_markup": 6.0,
             "hold_down": 1.0, "bid_premium": 0.1}
    pop = [start] + [mutate(start, rng, 3) for _ in range(7)]
    configs = ["config/default.yaml", "config/scaled_market.yaml", "tools/scenarios/costly.yaml",
               "tools/scenarios/cheap_wtp.yaml", "tools/scenarios/scarce.yaml",
               "tools/scenarios/beta_high.yaml", "config/tutorial_pricing.yaml"]
    with ProcessPoolExecutor() as pool:
        for gen in range(gens):
            roster = {f"c{i}": ("pepo_agent", cfg) for i, cfg in enumerate(pop)}
            roster.update(FIXED)
            jobs = [(str(ROOT / c), 5000 + 97 * gen + s, roster, None)
                    for c in configs for s in range(seeds)]
            results = list(pool.map(play, jobs))
            score = {t: statistics.mean(r[3][t] for r in results) for t in roster}
            ranked = sorted(roster, key=score.get)
            print(f"\n--- generation {gen}")
            for t in ranked:
                extra = json.dumps(roster[t][1]) if t.startswith("c") else ""
                print(f"{score[t]:5.2f} {t:<14} {extra}")
            sys.stdout.flush()
            elite = [roster[t][1] for t in ranked if t.startswith("c")][:3]
            pop = elite + [mutate(rng.choice(elite), rng) for _ in range(5)]


if __name__ == "__main__":
    main()
