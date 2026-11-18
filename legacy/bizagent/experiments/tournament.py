"""Paired tournament between agent versions.

    python experiments/tournament.py --split test --games 300 --ref v1_rule
    python experiments/tournament.py --split valid --games 100 --candidates v1_rule,v6_hybrid
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bizagent.evaluation.tournament import breakdown, run_tournament, save, table  # noqa: E402

DEFAULT = "v1_rule,v2_opt,v3_bayes,v4_bandit,v5_rl,v6_hybrid,v7_meta"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--candidates", default=DEFAULT)
    ap.add_argument("--ref", default="v1_rule")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--breakdowns", default="demand,fmt,scarcity,dynamics,two_stage,lead_time")
    args = ap.parse_args()
    cands = [c.strip() for c in args.candidates.split(",") if c.strip()]
    t0 = time.time()
    rows = run_tournament(cands, args.split, args.games, args.workers, start=args.start)
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                                   f"tournament_{args.split}")
    save(rows, out, meta=dict(split=args.split, games=args.games, start=args.start,
                              candidates=cands, seconds=time.time() - t0))
    text = [f"# Tournament - split `{args.split}`, {args.games} games per agent", "",
            "Score = objective value / fair-share revenue potential of the scenario. "
            "Simulated results only.", "", table(rows, args.ref, cands), ""]
    for tag in [t for t in args.breakdowns.split(",") if t]:
        text += [f"## Mean score (win rate) by {tag}", "", breakdown(rows, tag, cands), ""]
    with open(out + ".md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(text))
    print("\n".join(text))
    print(f"[{time.time() - t0:.0f}s] saved {out}.csv / .md")


if __name__ == "__main__":
    main()
