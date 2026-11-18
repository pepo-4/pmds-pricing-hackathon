"""Ablation study: the full agent against copies with one module switched off (paired).

    python experiments/ablations.py --split valid --games 120
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bizagent.evaluation.tournament import run_tournament, save, table  # noqa: E402

ABLATIONS = {
    "full": {},
    "no_opponent_model": dict(use_comp_prices=False, use_opponent_model=False, comp_band=False),
    "no_bayesian_learning": dict(ts_scale=0.0, forgetting=1.0, dither=0.0),
    "bandit_instead_of_model": dict(pricing_mode="bandit"),
    "greedy_instead_of_lp": dict(solver="greedy"),
    "no_safety_stock": dict(safety_z=0.0),
    "no_dynamic_pricing": dict(pricing_mode="fixed"),
    "no_inventory_awareness": dict(inventory_aware=False),
    "no_coordinated_bidding": dict(coordinated_bids=False),
    "no_bid_learning": dict(shade_mode="fixed"),
    "no_risk_limits": dict(cash_reserve_frac=0.0, max_spend_frac=1.0, min_margin=0.0,
                           max_price_step_up=0.6, max_price_step_down=0.6, early_step=0.6),
    "meta_strategy_on": dict(pricing_mode="meta"),
}
WHAT = {
    "no_opponent_model": "rival prices ignored entirely",
    "no_bayesian_learning": "no forgetting, no Thompson sampling, no price variation",
    "bandit_instead_of_model": "contextual bandit over a price grid sets prices",
    "greedy_instead_of_lp": "greedy planner replaces the LP",
    "no_safety_stock": "no buffer above planned sales",
    "no_dynamic_pricing": "price fixed at the first-round rule",
    "no_inventory_awareness": "stock left after the round valued at salvage only",
    "no_coordinated_bidding": "bids = fixed share of unit margin, no LP valuation",
    "no_bid_learning": "fixed 20% bid shading, no clearing-price model",
    "no_risk_limits": "no cash reserve, no spend cap, no margin floor, wide price steps",
    "meta_strategy_on": "bandit switches between demand model and follower rule",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    ap.add_argument("--games", type=int, default=120)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "results", "ablations"))
    args = ap.parse_args()
    t0 = time.time()
    cands = [(label, "v6_hybrid", over) for label, over in ABLATIONS.items()]
    rows = run_tournament(cands, args.split, args.games, args.workers)
    save(rows, args.out, meta=dict(split=args.split, games=args.games, ablations=ABLATIONS))
    text = [f"# Ablations - split `{args.split}`, {args.games} games each", "",
            "Each row is the full agent with one change. `diff vs full` below zero means the module "
            "that was removed was helping.", "", table(rows, "full", list(ABLATIONS)), "",
            "| variant | what changes |", "|---|---|"]
    text += [f"| {k} | {v} |" for k, v in WHAT.items()]
    with open(args.out + ".md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(text) + "\n")
    print("\n".join(text))
    print(f"[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
