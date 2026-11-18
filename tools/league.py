"""Run many games with custom rosters and report average rank (the official metric).

Usage:
  uv run python tools/league.py --roster official --seeds 20
  uv run python tools/league.py --roster crowded --seeds 20 --configs config/default.yaml
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dme_simulator.agents.loader import discover_bundle_classes  # noqa: E402
from dme_simulator.config import load_config  # noqa: E402
from dme_simulator.runner import run_simulation  # noqa: E402

AGENTS_DIR = ROOT / "src" / "dme_simulator" / "agents"

# first version of pepo_agent, kept as a rival
V1 = {"init_markup": 5.0, "explore_turns": 3, "up": 1.10, "hold_down": 0.97, "down": 0.85,
      "min_markup": 1.25, "floor_patience": 999, "probes": [], "streak_free": 1,
      "streak_down": 0.92, "ramp": 1.0, "target_min": 3, "bid_premium": 0.02}

# team_id -> (bundle id, constructor kwargs)
ROSTERS = {
    "official": {
        "pepo_agent": ("pepo_agent", {}),
        "DummyAgent": ("DummyAgent", {}),
        "PatientAgent": ("PatientAgent", {}),
        "PriceWarAgent": ("PriceWarAgent", {}),
    },
    # a busier market: several strong-ish rivals
    "crowded": {
        "pepo_agent": ("pepo_agent", {}),
        "pepo_baseline": ("pepo_baseline", {}),
        "pepo_v1": ("pepo_agent", V1),
        "pepo_single": ("pepo_agent", {"split_listings": False}),
        "rival_c": ("pepo_baseline", {}),
        "DummyAgent": ("DummyAgent", {}),
        "PriceWarAgent": ("PriceWarAgent", {}),
        "PriceWarAgent2": ("PriceWarAgent", {}),
    },
}


def play(job):
    config_path, seed, roster, turns = job
    classes = discover_bundle_classes([AGENTS_DIR])
    agents = {}
    for team, (bundle, kwargs) in roster.items():
        cls = classes[bundle]
        agents[team] = cls(**kwargs) if kwargs else cls()
    config = load_config(config_path)
    sim = run_simulation(config, agents, seed_override=seed, quiet=True, turns_override=turns)
    capital = {t: s.capital for t, s in sim.teams_state.items()}
    ordered = sorted(capital.values(), reverse=True)
    ranks = {}
    for team, cap in capital.items():
        first = ordered.index(cap) + 1
        count = ordered.count(cap)
        ranks[team] = first + (count - 1) / 2
    return config_path, seed, capital, ranks


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--roster", default="official")
    ap.add_argument("--roster-json", help="JSON roster overriding --roster")
    ap.add_argument("--configs", nargs="+",
                    default=["config/default.yaml", "config/scaled_market.yaml"])
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--turns", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args(argv)

    roster = json.loads(args.roster_json) if args.roster_json else ROSTERS[args.roster]
    roster = {k: tuple(v) for k, v in roster.items()}
    jobs = [(str(ROOT / c), args.seed0 + s, roster, args.turns)
            for c in args.configs for s in range(args.seeds)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(play, jobs))

    for config in args.configs + ["ALL"]:
        rows = [r for r in results if config == "ALL" or r[0].endswith(config)]
        print(f"\n== {Path(config).name} ({len(rows)} games)")
        print(f"{'team':<18}{'avg rank':>9}{'wins':>6}{'mean capital':>14}{'min capital':>13}")
        stats = []
        for team in roster:
            ranks = [r[3][team] for r in rows]
            caps = [r[2][team] for r in rows]
            stats.append((statistics.mean(ranks), team, sum(x == 1 for x in ranks),
                          statistics.mean(caps), min(caps)))
        for avg, team, wins, mean_cap, min_cap in sorted(stats):
            print(f"{team:<18}{avg:>9.2f}{wins:>6}{mean_cap:>14,.0f}{min_cap:>13,.0f}")


if __name__ == "__main__":
    main()
