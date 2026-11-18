"""Re-render the Markdown tables from the saved per-game CSVs (no games are played).

    python experiments/render.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ablations import ABLATIONS, WHAT  # noqa: E402
from bizagent.evaluation.tournament import breakdown, by_label, load, table  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
VERSIONS = ["v1_rule", "v2_opt", "v3_bayes", "v4_bandit", "v5_rl", "v6_hybrid", "v7_meta"]


def tournament(name: str = "tournament_test") -> None:
    base = os.path.join(RES, name)
    if not os.path.exists(base + ".csv"):
        return
    rows = load(base)
    split = rows[0]["split"]
    n = len(by_label(rows)[VERSIONS[0]])
    big = [r for r in rows if r["score_norm"] > 5]
    fams = sorted({str(r["tag_demand"]) for r in big})
    text = [f"# Tournament - split `{split}`, {n} games per agent", "",
            "Score = objective value / fair-share revenue potential of the scenario. Simulated results "
            "only. Medians are per-game medians; 'median diff' is the median of paired differences.", ""]
    if big:
        text += [f"**Read the medians, not the means.** {len(big)} of {len(rows)} agent-games have a score "
                 f"above 5 (the largest is {max(r['score_norm'] for r in big):,.0f}), all in the demand "
                 f"famil{'y' if len(fams) == 1 else 'ies'} {', '.join(fams)}. They dominate every mean "
                 "in this table.", ""]
    text += ["## Against the rule baseline (V1)", "", table(rows, "v1_rule", VERSIONS), "",
             "## Against the hybrid (V6)", "", table(rows, "v6_hybrid", VERSIONS), ""]
    for tag in ("demand", "fmt", "scarcity", "dynamics", "two_stage", "lead_time", "reveal_prices",
                "reveal_clearing"):
        text += [f"## Median score (win rate) by {tag}", "", breakdown(rows, tag, VERSIONS), ""]
    with open(base + ".md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(text))
    print("rendered", base + ".md")


def ablations() -> None:
    base = os.path.join(RES, "ablations")
    if not os.path.exists(base + ".csv"):
        return
    rows = load(base)
    n = len(by_label(rows)["full"])
    text = [f"# Ablations - split `{rows[0]['split']}`, {n} games each", "",
            "Each row is the full agent with one change. A negative difference means the module that "
            "was removed was helping.", "", table(rows, "full", list(ABLATIONS)), "",
            "| variant | what changes |", "|---|---|"]
    text += [f"| {k} | {v} |" for k, v in WHAT.items()]
    with open(base + ".md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(text) + "\n")
    print("rendered", base + ".md")


if __name__ == "__main__":
    tournament()
    ablations()
