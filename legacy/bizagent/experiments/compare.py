"""Compare configuration variants of the strategic agent on one split (paired).

    python experiments/compare.py --split train --games 80 \
        --variant base: --variant tight:band_ref=mean,band_over=0.15
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bizagent.evaluation.tournament import breakdown, run_tournament, save, table  # noqa: E402


def parse_value(v: str):
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


def parse_variant(text: str):
    label, _, rest = text.partition(":")
    over = {}
    for item in [x for x in rest.split(",") if x]:
        k, _, v = item.partition("=")
        over[k.strip()] = parse_value(v.strip())
    return label.strip(), over


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "valid", "test"])
    ap.add_argument("--games", type=int, default=80)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--base", default="v6_hybrid")
    ap.add_argument("--variant", action="append", default=[])
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--breakdowns", default="")
    args = ap.parse_args()
    variants = [parse_variant(v) for v in args.variant] or [("base", {})]
    cands = [(label, args.base, over) for label, over in variants]
    t0 = time.time()
    rows = run_tournament(cands, args.split, args.games, args.workers, start=args.start)
    labels = [c[0] for c in cands]
    text = [f"# Variants of {args.base} - split `{args.split}`, {args.games} games each", "",
            table(rows, labels[0], labels), ""]
    for tag in [t for t in args.breakdowns.split(",") if t]:
        text += [f"## by {tag}", "", breakdown(rows, tag, labels), ""]
    if args.out:
        save(rows, args.out, meta=dict(split=args.split, games=args.games, variants=variants))
        with open(args.out + ".md", "w", encoding="utf-8") as fh:
            fh.write("\n".join(text))
    print("\n".join(text))
    print(f"[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
