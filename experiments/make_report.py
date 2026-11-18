"""Assemble FINAL_REPORT.md from the template and the result files, so that no number in the
report is typed by hand.  python experiments/make_report.py"""
from __future__ import annotations

import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def include(match: re.Match) -> str:
    path = os.path.join(RES, match.group(1))
    if not os.path.exists(path):
        return f"_(missing: experiments/results/{match.group(1)} - run the experiment first)_"
    with open(path, encoding="utf-8") as fh:
        text = fh.read().strip()
    # demote headings so included files nest under the report's sections
    return re.sub(r"^(#+) ", lambda m: "#" * (len(m.group(1)) + 2) + " ", text, flags=re.M)


def main() -> None:
    with open(os.path.join(HERE, "report_template.md"), encoding="utf-8") as fh:
        text = fh.read()
    text = re.sub(r"\{\{include:([^}]+)\}\}", include, text)
    out = os.path.join(os.path.dirname(HERE), "FINAL_REPORT.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print("wrote", out)


if __name__ == "__main__":
    main()
