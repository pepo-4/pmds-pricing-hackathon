"""End-to-end: every agent version plays full games; adapters; objectives; speed."""
import json
import os

import numpy as np
import pytest

from bizagent.adapters.discovery import describe, suggest_mapping
from bizagent.adapters.generic import GenericAdapter
from bizagent.agents.registry import VERSION_NAMES, make_agent
from bizagent.api import CompetitionAgent
from bizagent.core.config import StrategyConfig, load_profile
from bizagent.core.schemas import Observation
from bizagent.evaluation.stats import bootstrap_ci, holm, iqm, paired
from bizagent.evaluation.tournament import play, run_tournament
from bizagent.simulation.environment import Game
from bizagent.simulation.opponents import OPPONENTS
from bizagent.simulation.scenarios import make_scenario

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.parametrize("name", VERSION_NAMES + list(OPPONENTS))
def test_every_agent_plays_a_game_without_errors(name):
    for seed, kw in ((101, dict(two_stage=True, lead_time=0)), (102, dict(two_stage=False, lead_time=1))):
        sc = make_scenario(seed, n_rounds=8, n_firms=3, **kw)
        agents = [make_agent(name), make_agent("opp_follower"), make_agent("opp_greedy")]
        row = Game(sc, agents).run()[0]
        assert row["errors"] == 0 and row["engine_errors"] == 0
        assert np.isfinite(row["score"])


@pytest.mark.parametrize("fmt", ["first_price", "uniform", "second_price", "posted"])
@pytest.mark.parametrize("two_stage,lead", [(True, 0), (True, 1), (False, 0), (False, 1)])
def test_hybrid_trades_in_every_format_and_timing(fmt, two_stage, lead):
    sc = make_scenario(7, n_rounds=12, n_firms=4, fmt=fmt, two_stage=two_stage, lead_time=lead,
                       scarcity="balanced")
    agents = [make_agent("v6_hybrid")] + [make_agent(o) for o in ("opp_follower", "opp_greedy", "opp_costplus")]
    row = Game(sc, agents).run()[0]
    assert row["errors"] == 0 and row["engine_errors"] == 0
    assert row["units_bought"] > 0 and row["units_sold"] > 0
    assert row["score"] > 0                      # it should at least not lose money here


def test_hybrid_beats_the_random_agent_head_to_head():
    wins = 0
    for seed in range(200, 208):
        sc = make_scenario(seed, n_firms=3, n_rounds=20)
        rows = Game(sc, [make_agent("v6_hybrid"), make_agent("opp_random"), make_agent("opp_random")]).run()
        wins += rows[0]["rank"] == 1
    assert wins >= 6


def test_objective_changes_behaviour():
    """Under a market-share objective the agent should sell more units than under profit."""
    units = {}
    for objective in ("profit", "market_share"):
        tot = 0.0
        for seed in (301, 302, 303, 304):
            sc = make_scenario(seed, n_firms=3, n_rounds=20, objective=objective, scarcity="abundant")
            rows = Game(sc, [make_agent("v6_hybrid"), make_agent("opp_costplus"), make_agent("opp_fixed")]).run()
            tot += rows[0]["market_share"]
        units[objective] = tot
    assert units["market_share"] > units["profit"]


def test_profiles_load_and_differ():
    base = load_profile("balanced")
    assert isinstance(base, StrategyConfig)
    for name in ("conservative", "aggressive", "profit_max", "market_share", "adaptive"):
        cfg = load_profile(name)
        assert cfg.name == name and cfg.to_dict() != base.to_dict()
    assert load_profile("balanced", {"ts_scale": 0.0}).ts_scale == 0.0
    with pytest.raises(FileNotFoundError):
        load_profile("does_not_exist")


def test_generic_adapter_maps_a_foreign_format():
    theirs = {"turn": 3, "company": {"money": 420.5,
                                     "materials": [{"name": "R", "quantity": 12}],
                                     "warehouse": {"A": 4}},
              "market": {"auctions": [{"resource": "R", "quantity": 30, "reserve_price": 2}],
                         "opponent_prices": {"teamX": {"A": 21.0}}}}
    mapping, missing, unused = suggest_mapping(theirs)
    assert mapping["round"] == "turn" and mapping["cash"] == "company.money"
    assert mapping["raw_inventory"] == "company.materials" and mapping["lots"] == "market.auctions"
    assert mapping["competitor_prices"] == "market.opponent_prices"
    ad = GenericAdapter(mapping, action_keys={"bids": "auction_bids", "quantity": "qty"},
                        integer_quantities=True)
    obs = ad.to_observation(theirs)
    assert (obs.round, obs.cash, obs.raw_inventory, obs.fin_inventory) == (3, 420.5, {"R": 12.0}, {"A": 4.0})
    assert obs.lots[0].quantity == 30 and obs.competitor_prices == {"teamX": {"A": 21.0}}
    spec = {"products": {"A": {"recipe": {"R": 1}, "unit_cost": 1}}, "horizon": 10, "lead_time": 0}
    agent = CompetitionAgent(spec=spec, adapter=ad)
    out = agent.act(theirs)
    assert "auction_bids" in out and all(float(b["qty"]).is_integer() for b in out["auction_bids"])
    assert all(float(x).is_integer() for x in out["production"].values())
    json.dumps(out)


def test_describe_reads_sdk_objects():
    info = describe(Observation)
    assert "cash" in info["fields"]
    assert "signature" in describe(make_scenario)


def test_stats_helpers():
    assert iqm([0, 1, 2, 3, 100]) == pytest.approx(2.0)
    lo, hi = bootstrap_ci(np.arange(100.0))
    assert lo < 49.5 < hi
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    res = paired([2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7])
    assert res["mean_diff"] == 1.0 and res["p_improve"] == 1.0 and res["p_wilcoxon"] < 0.05


def test_tournament_is_paired_and_reproducible():
    a = run_tournament(["v1_rule", "v6_hybrid"], "train", 3, workers=1)
    b = run_tournament(["v1_rule", "v6_hybrid"], "train", 3, workers=1)
    assert [r["scenario"] for r in a[::2]] == [r["scenario"] for r in a[1::2]]      # same games
    assert [r["opponents"] for r in a[::2]] == [r["opponents"] for r in a[1::2]]
    assert [round(r["score"], 6) for r in a if r["label"] == "v1_rule"] == \
           [round(r["score"], 6) for r in b if r["label"] == "v1_rule"]


def test_decision_time_is_within_budget():
    row = play((("v6_hybrid", "v6_hybrid", {}), "train", 4, {"n_products": 3, "n_resources": 3,
                                                               "n_rounds": 15}))
    assert row["ms_mean"] < 150 and row["ms_max"] < 1000
