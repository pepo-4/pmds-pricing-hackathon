"""Free-for-all league: all of our versions in the same market, plus two random opponents.

    python experiments/league.py --games 60
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bizagent.evaluation.stats import summarise  # noqa: E402
from bizagent.evaluation.tournament import by_label, league, save  # noqa: E402

NAMES = ["v1_rule", "v2_opt", "v3_bayes", "v4_bandit", "v5_rl", "v6_hybrid", "v7_meta"]


def _chunk(args):
    start, n, split = args
    return league(NAMES, split=split, n_games=n, start=start)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "results", "league"))
    args = ap.parse_args()
    t0 = time.time()
    per = max(1, args.games // max(args.workers, 1))
    chunks = [(s, min(per, args.games - s), args.split) for s in range(0, args.games, per)]
    with Pool(args.workers) as pool:
        rows = [r for part in pool.map(_chunk, chunks) for r in part]
    save(rows, args.out, meta=dict(split=args.split, games=args.games, names=NAMES))
    groups = by_label(rows)
    text = [f"# League - all versions in the same market, {args.games} games (split `{args.split}`)", "",
            "| agent | games | mean score | IQM | mean rank | win rate | top-2 |", "|---|---|---|---|---|---|---|"]
    for name in NAMES:
        s = summarise(groups[name])
        mean_rank = sum(r["rank"] for r in groups[name]) / len(groups[name])
        text.append(f"| {name} | {s['games']} | {s['mean']:.4f} | {s['iqm']:.4f} | {mean_rank:.2f} | "
                    f"{100 * s['win_rate']:.1f}% | {100 * s['top2_rate']:.1f}% |")
    with open(args.out + ".md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(text) + "\n")
    print("\n".join(text))
    print(f"[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
