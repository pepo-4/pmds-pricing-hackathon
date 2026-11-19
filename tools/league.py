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

# first version of 444_agent, kept as a rival
V1 = {"init_markup": 5.0, "explore_turns": 3, "up": 1.10, "hold_down": 0.97, "down": 0.85,
      "min_markup": 1.25, "floor_patience": 999, "probes": [], "streak_free": 1,
      "streak_down": 0.92, "ramp": 1.0, "target_min": 3, "bid_premium": 0.02}

# team_id -> (bundle id, constructor kwargs)
ROSTERS = {
    "adaptive_v3": {
        "444_base": ("444_agent", {}),
        "adaptive_v3": ("adaptive_v3", {}),
        "tight": ("tight", {}),
        "tight_conservative": ("tight", {"budget_share": 0.5, "value_share": 0.6, "cover_turns": 2.0}),
        "g_open": ("tight", {"prior_markup": 3.5, "ladder_low": 0.50, "ladder_high": 1.20,
                             "open_turns": 3, "open_stock": 12, "open_bid": 2.5,
                             "cover_turns": 4.0, "max_stock": 80, "budget_share": 1.0,
                             "value_share": 1.0, "max_bid_mult": 10.0}),
        "g_stock": ("tight", {"prior_markup": 4.0, "ladder_low": 0.50, "ladder_high": 1.10,
                              "step_up": 0.08, "step_down": 0.14, "min_stock": 5,
                              "cover_turns": 6.0, "max_stock": 120, "bid_start": 1.5,
                              "bid_up": 1.45, "budget_share": 1.0, "value_share": 0.9,
                              "max_bid_mult": 10.0}),
        "DummyAgent": ("DummyAgent", {}),
        "PriceWarAgent": ("PriceWarAgent", {}),
    },
    "adaptive_tune": {
        "444_base": ("444_agent", {}),
        "ad_125": ("adaptive_v2", {"bid_start": 1.25, "short_budget": 0.75, "bid_cap": 6.0}),
        "ad_150": ("adaptive_v2", {"bid_start": 1.50, "short_budget": 0.75, "bid_cap": 6.0}),
        "ad_175": ("adaptive_v2", {"bid_start": 1.75, "short_budget": 0.80, "bid_cap": 7.0}),
        "ad_200": ("adaptive_v2", {"bid_start": 2.00, "short_budget": 0.85, "bid_cap": 8.0}),
        "ad_220": ("adaptive_v2", {"bid_start": 2.20, "short_budget": 0.90, "bid_cap": 10.0}),
        "tight": ("tight", {}),
        "tight_conservative": ("tight", {"budget_share": 0.5, "value_share": 0.6, "cover_turns": 2.0}),
    },
    "adaptive_single": {
        "444_base": ("444_agent", {}),
        "adaptive_v2": ("adaptive_v2", {}),
        "tight": ("tight", {}),
        "tight_conservative": ("tight", {"budget_share": 0.5, "value_share": 0.6, "cover_turns": 2.0}),
        "DummyAgent": ("DummyAgent", {}),
    },
    "adaptive_v2": {
        "444_base": ("444_agent", {}),
        "adaptive_v2": ("adaptive_v2", {}),
        "adaptive_v2_defensive": ("adaptive_v2", {"budget_share": 0.65, "protect_lead": 0.05}),
        "adaptive_v2_aggressive": ("adaptive_v2", {"bid_start": 1.5, "bid_up": 1.50, "budget_share": 1.0, "bid_cap": 8.0}),
        "tight": ("tight", {}),
        "tight_conservative": ("tight", {"budget_share": 0.5, "value_share": 0.6, "cover_turns": 2.0}),
        "DummyAgent": ("DummyAgent", {}),
        "PriceWarAgent": ("PriceWarAgent", {}),
    },
    "graph_tune": {
        "444_base": ("444_agent", {}),
        "tight_default": ("tight", {}),
        "g_volume": ("tight", {
            "prior_markup": 3.2, "ladder_low": 0.45, "ladder_high": 1.15,
            "step_up": 0.09, "step_down": 0.18, "min_stock": 3,
            "cover_turns": 4.0, "max_stock": 80, "bid_start": 1.3,
            "bid_up": 1.35, "budget_share": 1.0, "value_share": 1.0,
            "max_bid_mult": 8.0,
        }),
        "g_momentum": ("tight", {
            "prior_markup": 3.5, "ladder_low": 0.55, "ladder_high": 1.25,
            "step_up": 0.10, "step_down": 0.12, "min_stock": 4,
            "cover_turns": 5.0, "max_stock": 100, "bid_start": 1.4,
            "bid_up": 1.40, "budget_share": 1.0, "value_share": 1.0,
            "max_bid_mult": 10.0, "cap_grow": 1.30,
        }),
        "g_stock": ("tight", {
            "prior_markup": 4.0, "ladder_low": 0.50, "ladder_high": 1.10,
            "step_up": 0.08, "step_down": 0.14, "min_stock": 5,
            "cover_turns": 6.0, "max_stock": 120, "bid_start": 1.5,
            "bid_up": 1.45, "budget_share": 1.0, "value_share": 0.9,
            "max_bid_mult": 10.0,
        }),
        "g_conservative": ("tight", {
            "prior_markup": 4.5, "ladder_low": 0.65, "ladder_high": 1.0,
            "step_up": 0.06, "step_down": 0.12, "min_stock": 2,
            "cover_turns": 3.0, "max_stock": 40, "budget_share": 0.65,
            "value_share": 0.70, "max_bid_mult": 5.0,
        }),
        "g_open": ("tight", {
            "prior_markup": 3.5, "ladder_low": 0.50, "ladder_high": 1.20,
            "open_turns": 3, "open_stock": 12, "open_bid": 2.5,
            "cover_turns": 4.0, "max_stock": 80, "budget_share": 1.0,
            "value_share": 1.0, "max_bid_mult": 10.0,
        }),
        "g_hybrid6": ("tight", {
            "prior_markup": 3.5, "ladder_low": 0.50, "ladder_high": 1.20,
            "open_turns": 6, "open_stock": 12, "open_bid": 2.5,
            "cover_turns": 5.0, "min_stock": 5, "max_stock": 120,
            "budget_share": 1.0, "value_share": 0.95, "max_bid_mult": 12.0,
            "cap_grow": 1.30,
        }),
        "g_hybrid4": ("tight", {
            "prior_markup": 3.5, "ladder_low": 0.50, "ladder_high": 1.20,
            "open_turns": 4, "open_stock": 12, "open_bid": 2.5,
            "cover_turns": 5.0, "min_stock": 5, "max_stock": 100,
            "budget_share": 1.0, "value_share": 0.95, "max_bid_mult": 10.0,
        }),
        "DummyAgent": ("DummyAgent", {}),
        "PriceWarAgent": ("PriceWarAgent", {}),
    },
    "short_tune": {
        "444_base": ("444_agent", {}),
        "short_open2": ("tight", {"open_turns": 2, "open_stock": 8, "open_bid": 2.0,
                                  "budget_share": 0.9, "value_share": 0.9}),
        "short_open3": ("tight", {"open_turns": 3, "open_stock": 10, "open_bid": 3.0,
                                  "budget_share": 1.0, "value_share": 1.0}),
        "short_open5": ("tight", {"open_turns": 5, "open_stock": 12, "open_bid": 4.0,
                                  "budget_share": 1.0, "value_share": 1.0}),
        "short_low_markup": ("tight", {"prior_markup": 3.0, "ladder_low": 0.55,
                                       "ladder_high": 0.95, "step_down": 0.12,
                                       "cover_turns": 2.0, "budget_share": 0.9}),
        "short_high_markup": ("tight", {"prior_markup": 5.5, "ladder_low": 0.65,
                                        "ladder_high": 1.0, "step_up": 0.10,
                                        "cover_turns": 2.5, "budget_share": 0.8}),
        "short_big_stock": ("tight", {"min_stock": 4, "cover_turns": 4.0,
                                      "open_stock": 14, "open_turns": 3,
                                      "budget_share": 1.0, "value_share": 0.9}),
        "short_light_stock": ("tight", {"min_stock": 1, "cover_turns": 1.5,
                                        "budget_share": 0.85, "value_share": 0.8}),
        "short_bid2": ("tight", {"bid_start": 2.0, "bid_up": 1.25,
                                 "max_bid_mult": 8.0, "budget_share": 1.0}),
        "tight_conservative": ("tight", {"budget_share": 0.5, "value_share": 0.6,
                                         "cover_turns": 2.0}),
        "tight_aggressive": ("tight", {"budget_share": 1.0, "value_share": 1.0,
                                       "cover_turns": 4.0, "max_bid_mult": 8.0}),
        "tight_v6": ("tight", {"cap_mode": 0, "comp_aware": 0}),
        "short_hybrid2": ("tight", {
            "open_turns": 2, "open_stock": 8, "open_bid": 2.5,
            "min_stock": 1, "cover_turns": 1.5, "max_stock": 24,
            "budget_share": 0.85, "value_share": 0.8,
        }),
        "short_hybrid3": ("tight", {
            "open_turns": 3, "open_stock": 10, "open_bid": 3.0,
            "min_stock": 1, "cover_turns": 1.5, "max_stock": 24,
            "budget_share": 0.9, "value_share": 0.85,
        }),
        "short_hybrid5": ("tight", {
            "open_turns": 5, "open_stock": 12, "open_bid": 3.5,
            "min_stock": 1, "cover_turns": 1.5, "max_stock": 24,
            "budget_share": 0.9, "value_share": 0.85,
        }),
    },
    "tight": {
        "444_agent": ("444_agent", {}),
        "tight": ("tight", {}),
        "tight_v6": ("tight", {"cap_mode": 0, "comp_aware": 0}),
        "tight_conservative": ("tight", {"budget_share": 0.5, "value_share": 0.6, "cover_turns": 2.0}),
        "tight_aggressive": ("tight", {"budget_share": 1.0, "value_share": 1.0, "cover_turns": 4.0, "max_bid_mult": 8.0}),
        "DummyAgent": ("DummyAgent", {}),
        "PatientAgent": ("PatientAgent", {}),
        "PriceWarAgent": ("PriceWarAgent", {}),
    },
    "official": {
        "444_agent": ("444_agent", {}),
        "DummyAgent": ("DummyAgent", {}),
        "PatientAgent": ("PatientAgent", {}),
        "PriceWarAgent": ("PriceWarAgent", {}),
    },
    # a busier market: several strong-ish rivals
    "crowded": {
        "444_agent": ("444_agent", {}),
        "pepo_baseline": ("pepo_baseline", {}),
        "pepo_v1": ("444_agent", V1),
        "pepo_single": ("444_agent", {"split_listings": False}),
        "rival_c": ("pepo_baseline", {}),
        "DummyAgent": ("DummyAgent", {}),
        "PriceWarAgent": ("PriceWarAgent", {}),
        "PriceWarAgent2": ("PriceWarAgent", {}),
    },
    "aggressive": {
        "444_agent": ("444_agent", {}),
        "PriceWar1": ("PriceWarAgent", {}),
        "PriceWar2": ("PriceWarAgent", {}),
        "PriceWar3": ("PriceWarAgent", {}),
        "DummyAgent": ("DummyAgent", {}),
        "PatientAgent": ("PatientAgent", {}),
    },
    "passive": {
        "444_agent": ("444_agent", {}),
        "Dummy1": ("DummyAgent", {}),
        "Dummy2": ("DummyAgent", {}),
        "Patient1": ("PatientAgent", {}),
        "Patient2": ("PatientAgent", {}),
        "Baseline": ("pepo_baseline", {}),
    },
    "mirror": {
        "444_agent": ("444_agent", {}),
        "mirror_balanced": ("mirror_balanced", {}),
        "mirror_conservative": ("mirror_conservative", {}),
        "mirror_aggressive": ("mirror_aggressive", {}),
        "mirror_short": ("mirror_short", {}),
        "mirror_long": ("mirror_long", {}),
        "mirror_sniper": ("mirror_sniper", {}),
        "DummyAgent": ("DummyAgent", {}),
    },
    "mirror_tune": {
        "tune_base": ("444_agent", {}),
        "tune_low_markup": ("444_agent", {"init_markup": 5.0, "down": 0.88, "min_markup": 1.8}),
        "tune_high_markup": ("444_agent", {"init_markup": 10.0, "down": 0.94, "min_markup": 2.8}),
        "tune_light_stock": ("444_agent", {"target_min": 6, "target_cover": 1.0, "target_max": 20}),
        "tune_cash_guard": ("444_agent", {"budget_frac": 0.35, "leader_guard": True}),
        "tune_sniper": ("444_agent", {"bid_premium": 0.35, "premium_up": 0.5, "budget_frac": 0.85}),
        "tune_short": ("444_agent", {"horizon_adapt": True, "horizon_turn": 8, "target_min": 6}),
        "tune_aggressive": ("444_agent", {"down": 0.82, "hold_down": 0.90, "min_markup": 1.5}),
        "mirror_balanced": ("mirror_balanced", {}),
        "mirror_aggressive": ("mirror_aggressive", {}),
        "mirror_conservative": ("mirror_conservative", {}),
        "mirror_sniper": ("mirror_sniper", {}),
    },
    "mirror_counter": {
        "counter_sniper": ("444_agent", {
            "init_markup": 8.0,
            "ladder": [1.0, 0.86, 1.25, 0.72, 1.5, 1.8],
            "down": 0.90,
            "floor_patience": 2,
            "floor_relax": 0.95,
            "target_min": 8,
            "target_cover": 1.0,
            "target_max": 20,
            "bid_premium": 0.45,
            "premium_up": 0.40,
            "premium_cap": 1.8,
            "budget_frac": 0.90,
            "horizon_adapt": False,
            "leader_guard": False,
        }),
        "mirror_sniper": ("mirror_sniper", {}),
        "mirror_aggressive": ("mirror_aggressive", {}),
        "mirror_balanced": ("mirror_balanced", {}),
        "mirror_conservative": ("mirror_conservative", {}),
        "mirror_short": ("mirror_short", {}),
        "mirror_long": ("mirror_long", {}),
    },
    "diverse": {
        "444_agent": ("444_agent", {}),
        "under_cut": ("rival_under_cut", {}),
        "stockpile": ("rival_stockpile", {}),
        "chip": ("rival_chip", {}),
        "liquidity": ("rival_liquidity", {}),
        "adaptive": ("rival_adaptive", {}),
        "mature": ("rival_mature", {}),
        "mirror_aggressive": ("mirror_aggressive", {}),
    },
    "diverse_tune": {
        "444_adaptive": ("444_agent", {}),
        "444_mature_stock": ("444_agent", {"target_min": 24, "target_cover": 3.0, "target_max": 40}),
        "444_cost_guard": ("444_agent", {"budget_frac": 0.55, "maturity_adapt": True}),
        "444_chip": ("444_agent", {"bid_premium": 0.55, "budget_frac": 0.95}),
        "444_undercut": ("444_agent", {"down": 0.82, "min_markup": 1.3, "floor_relax": 0.92}),
        "under_cut": ("rival_under_cut", {}),
        "stockpile": ("rival_stockpile", {}),
        "chip": ("rival_chip", {}),
        "liquidity": ("rival_liquidity", {}),
        "adaptive": ("rival_adaptive", {}),
        "mature": ("rival_mature", {}),
    },
    "moderate_aggressive": {
        "444_base": ("444_agent", {}),
        "444_under1": ("444_agent", {
            "init_markup": 3.0, "ladder": [1.0, 0.82, 0.68, 1.08, 1.18],
            "down": 0.82, "hold_down": 0.97, "min_markup": 1.25,
            "floor_patience": 1, "floor_relax": 0.92, "floor_abs": 1.05,
            "target_min": 8, "target_cover": 1.2, "target_max": 30,
            "bid_premium": 0.20, "premium_up": 0.28, "premium_decay": 0.60,
            "budget_frac": 0.80, "leader_guard": False, "maturity_adapt": False,
        }),
        "444_under2": ("444_agent", {
            "init_markup": 3.5, "ladder": [1.0, 0.78, 0.62, 1.10, 1.22],
            "down": 0.85, "hold_down": 0.95, "min_markup": 1.35,
            "floor_patience": 1, "floor_relax": 0.92, "floor_abs": 1.10,
            "target_min": 10, "target_cover": 1.5, "target_max": 35,
            "bid_premium": 0.25, "premium_up": 0.30, "premium_decay": 0.65,
            "budget_frac": 0.85, "leader_guard": False, "maturity_adapt": False,
        }),
        "444_under3": ("444_agent", {
            "init_markup": 2.5, "ladder": [1.0, 0.88, 0.72, 1.05, 1.15],
            "down": 0.78, "hold_down": 0.92, "min_markup": 1.20,
            "floor_patience": 1, "floor_relax": 0.88, "floor_abs": 1.05,
            "target_min": 6, "target_cover": 1.0, "target_max": 24,
            "bid_premium": 0.18, "premium_up": 0.35, "premium_decay": 0.55,
            "budget_frac": 0.75, "leader_guard": False, "maturity_adapt": False,
        }),
        "under_cut": ("rival_under_cut", {}),
        "mirror_aggressive": ("mirror_aggressive", {}),
        "liquidity": ("rival_liquidity", {}),
    },
    "super_aggressive": {
        "444_base": ("444_agent", {}),
        "444_all_in": ("444_agent", {
            "init_markup": 2.8, "ladder": [1.0, 0.58, 1.12, 0.48, 1.28, 1.45],
            "down": 0.78, "hold_down": 0.88, "min_markup": 1.15,
            "floor_patience": 1, "floor_relax": 0.90, "floor_abs": 1.05,
            "target_min": 16, "target_cover": 4.0, "target_max": 60,
            "start_stock": 6, "ramp": 1.5, "bid_premium": 0.60,
            "premium_up": 0.50, "premium_decay": 0.75, "premium_cap": 2.5,
            "budget_frac": 1.0, "leader_guard": False, "maturity_adapt": False,
        }),
        "444_momentum": ("444_agent", {
            "init_markup": 3.5, "ladder": [1.0, 0.68, 1.18, 0.55, 1.35, 1.65],
            "down": 0.82, "hold_down": 0.92, "min_markup": 1.25,
            "floor_patience": 1, "floor_relax": 0.92,
            "target_min": 18, "target_cover": 3.5, "target_max": 50,
            "start_stock": 5, "ramp": 1.2, "bid_premium": 0.70,
            "premium_up": 0.55, "premium_decay": 0.70, "premium_cap": 2.8,
            "budget_frac": 1.0, "leader_guard": False, "maturity_adapt": False,
        }),
        "444_breaker": ("444_agent", {
            "init_markup": 2.5, "ladder": [1.0, 0.72, 0.92, 0.62, 1.15, 1.25],
            "down": 0.75, "hold_down": 0.85, "min_markup": 1.20,
            "floor_patience": 1, "floor_relax": 0.88,
            "target_min": 24, "target_cover": 5.0, "target_max": 60,
            "start_stock": 8, "ramp": 2.0, "bid_premium": 0.80,
            "premium_up": 0.60, "premium_decay": 0.80, "premium_cap": 3.0,
            "budget_frac": 1.0, "leader_guard": False, "maturity_adapt": False,
        }),
        "444_fast_floor": ("444_agent", {
            "init_markup": 3.0, "ladder": [1.0, 0.80, 1.08, 0.70, 1.20, 1.35],
            "down": 0.85, "hold_down": 0.95, "min_markup": 1.35,
            "floor_patience": 1, "floor_relax": 0.85,
            "target_min": 12, "target_cover": 2.5, "target_max": 40,
            "bid_premium": 0.50, "premium_up": 0.45, "budget_frac": 1.0,
            "leader_guard": False, "maturity_adapt": True,
        }),
        "mirror_aggressive": ("mirror_aggressive", {}),
        "under_cut": ("rival_under_cut", {}),
    },
    "ultralong_tune": {
        "444_ultra_adaptive": ("444_agent", {}),
        "444_ultra_balanced": ("444_agent", {
            "init_markup": 7.0, "ladder": [1.0, 0.88, 1.12, 0.78, 1.25],
            "down": 0.94, "hold_down": 0.97, "floor_patience": 3,
            "target_min": 24, "target_cover": 2.5, "target_max": 40,
            "bid_premium": 0.08, "premium_up": 0.25, "budget_frac": 0.55,
            "leader_guard": False, "maturity_adapt": True,
        }),
        "444_ultra_stock": ("444_agent", {
            "init_markup": 7.0, "ladder": [1.0, 0.88, 1.12, 0.78, 1.25],
            "down": 0.94, "hold_down": 0.97, "floor_patience": 4,
            "target_min": 24, "target_cover": 3.0, "target_max": 40,
            "bid_premium": 0.12, "premium_up": 0.30, "budget_frac": 0.75,
            "leader_guard": False, "maturity_adapt": True,
        }),
        "stockpile": ("rival_stockpile", {}),
        "mature": ("rival_mature", {}),
        "mirror_aggressive": ("mirror_aggressive", {}),
        "mirror_long": ("mirror_long", {}),
        "liquidity": ("rival_liquidity", {}),
    },
    "profiles": {
        "base": ("444_agent", {}),
        "stock_light": ("444_agent", {"target_min": 6, "target_cover": 1.0}),
        "stock_min": ("444_agent", {"target_min": 3, "target_cover": 1.0}),
        "cash_guard": ("444_agent", {"budget_frac": 0.35}),
        "balanced_guard": ("444_agent", {"target_min": 6, "target_cover": 1.0, "budget_frac": 0.35}),
        "patient_pricing": ("444_agent", {"down": 0.93, "hold_down": 1.0, "min_markup": 2.5}),
        "aggressive_pricing": ("444_agent", {"down": 0.85, "hold_down": 0.90, "min_markup": 1.6}),
        "DummyAgent": ("DummyAgent", {}),
    },
    "horizon": {
        "base": ("444_agent", {"horizon_adapt": False}),
        "leader_guard": ("444_agent", {"horizon_adapt": False, "leader_guard": True}),
        "horizon4": ("444_agent", {"horizon_adapt": True, "horizon_turn": 4}),
        "horizon6": ("444_agent", {"horizon_adapt": True, "horizon_turn": 6}),
        "horizon8": ("444_agent", {"horizon_adapt": True, "horizon_turn": 8}),
        "adaptive": ("444_agent", {"horizon_adapt": True, "horizon_turn": 8, "leader_guard": True}),
        "horizon12": ("444_agent", {"horizon_adapt": True, "horizon_turn": 12}),
        "stock_min": ("444_agent", {"horizon_adapt": False, "target_min": 3, "target_cover": 1.0}),
        "cash_guard": ("444_agent", {"horizon_adapt": False, "budget_frac": 0.35}),
        "DummyAgent": ("DummyAgent", {}),
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
        wanted = None if config == "ALL" else (ROOT / config).resolve()
        rows = [
            r for r in results
            if config == "ALL" or Path(r[0]).resolve() == wanted
        ]
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
