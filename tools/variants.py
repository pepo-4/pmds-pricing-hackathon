"""Compare versions of our agent in a FIXED field (only our team changes).

Usage:
  uv run python tools/variants.py --field tools/opponents/real_field.json \
      --extra-dirs /path/to/other/teams/agents --seeds 6
Edit VARIANTS below to choose what to compare.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from league import ROOT, play  # noqa: E402

ME = "rotating_forks"

VARIANTS = {
    "current": {},
    "sp0.5": {"start_premium": 0.5},
    "sp1.0": {"start_premium": 1.0},
    "sp2.0": {"start_premium": 2.0},
    "sp1.0_cap3": {"start_premium": 1.0, "premium_cap": 3.0, "premium_up": 0.5},
}

CONFIGS = ["config/default.yaml", "config/scaled_market.yaml",
           "tools/scenarios/costly.yaml", "tools/scenarios/scarce.yaml",
           "tools/scenarios/cheap_wtp.yaml", "tools/scenarios/beta_high.yaml",
           "tools/scenarios/short.yaml"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default="tools/opponents/real_field.json")
    ap.add_argument("--extra-dirs", nargs="*", default=[])
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--seed0", type=int, default=2000)
    ap.add_argument("--configs", nargs="+", default=CONFIGS)
    ap.add_argument("--variants", help="JSON dict name -> params (overrides VARIANTS)")
    args = ap.parse_args()

    field = json.load(open(ROOT / args.field))
    variants = json.loads(args.variants) if args.variants else VARIANTS
    jobs, keys = [], []
    for name, params in variants.items():
        roster = {t: (b, dict(k)) for t, (b, k) in field.items()}
        roster[ME] = ("rotating_forks_agent", {**roster[ME][1], **params})
        for c in args.configs:
            for s in range(args.seeds):
                jobs.append((str(ROOT / c), args.seed0 + s, roster, None, args.extra_dirs))
                keys.append((name, c))
    with ProcessPoolExecutor() as pool:
        results = list(pool.map(play, jobs))

    rows = {}
    for (name, c), (_, _, capital, ranks) in zip(keys, results):
        best_other = max(v for t, v in capital.items() if t != ME)
        rows.setdefault(name, []).append((c, ranks[ME], capital[ME], capital[ME] - best_other))
    short = [Path(c).stem[:9] for c in args.configs]
    print(f"{'variant':<14}{'avg rank':>9}{'wins':>6}{'mean cap':>10}{'gap to best other':>19}  "
          + " ".join(f"{s:>9}" for s in short))
    for name, r in sorted(rows.items(), key=lambda kv: statistics.mean(x[1] for x in kv[1])):
        per = [statistics.mean(x[1] for x in r if x[0] == c) for c in args.configs]
        print(f"{name:<14}{statistics.mean(x[1] for x in r):>9.2f}{sum(x[1] == 1 for x in r):>6}"
              f"{statistics.mean(x[2] for x in r):>10,.0f}{statistics.mean(x[3] for x in r):>19,.0f}  "
              + " ".join(f"{v:>9.2f}" for v in per))


if __name__ == "__main__":
    main()
