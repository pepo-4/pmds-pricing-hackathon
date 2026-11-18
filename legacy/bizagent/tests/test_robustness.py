"""The agent must return a valid action whatever it is given."""
import json
import math
import subprocess
import sys
import os

import numpy as np
import pytest

from bizagent.agents.registry import make_agent
from bizagent.api import CompetitionAgent, decide
from bizagent.core.schemas import Action, GameSpec, Observation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = {"products": {"A": {"recipe": {"R": 1}, "unit_cost": 2.0, "price_max": 60},
                     "B": {"recipe": {"R": 2, "S": 1}, "unit_cost": 3.0}},
        "resources": {"R": {"auction_format": "first_price", "reserve_price": 1.0},
                      "S": {"auction_format": "uniform"}},
        "horizon": 10, "n_firms": 4, "initial_cash": 500, "lead_time": 0, "capacity": 40}


def valid(action: Action, spec: GameSpec, cash: float) -> None:
    for b in action.bids:
        assert b.resource in spec.resources
        assert math.isfinite(b.price) and math.isfinite(b.quantity) and b.price > 0 and b.quantity > 0
    assert sum(b.price * b.quantity for b in action.bids) <= max(cash, 0.0) + 1e-6
    for p, x in action.production.items():
        assert p in spec.products and math.isfinite(x) and x >= 0
    for p, x in action.prices.items():
        ps = spec.products[p]
        assert math.isfinite(x) and x >= ps.price_min
        assert ps.price_max is None or x <= ps.price_max + 1e-9


WEIRD = [
    {},                                                                  # empty observation
    {"round": 0, "cash": 500},                                           # no lots, no inventory
    {"round": 1, "cash": 0, "lots": [{"resource": "R", "quantity": 50}]},  # no cash
    {"round": 1, "cash": -100, "lots": [{"resource": "R", "quantity": 50}]},
    {"round": 2, "cash": 500, "raw_inventory": {"R": 0, "S": 0}, "fin_inventory": {"A": 0, "B": 0},
     "lots": [{"resource": "R", "quantity": 0}, {"resource": "ZZZ", "quantity": 10}]},
    {"round": 3, "cash": 1e12, "raw_inventory": {"R": 1e9}, "lots": [{"resource": "R", "quantity": 1e9}],
     "last_prices": {"A": 1e9, "B": 1e-9}, "last_sales": {"A": 0, "B": 1e9}},  # extreme values
    {"round": 4, "cash": "lots", "raw_inventory": "none", "fin_inventory": [1, 2], "lots": "x",
     "last_prices": {"A": None}, "last_sales": {"A": "NaN"}, "competitor_prices": {"f": None}},
    {"round": 5, "cash": 500, "last_prices": {"A": -5, "B": float("inf")}, "last_sales": {"A": -3},
     "auction_results": [{"resource": "R", "clearing_price": -1, "won_qty": 1e9, "paid": float("nan")},
                         {"resource": "nope"}, 17]},                    # invalid / unexpected results
    {"round": 6, "cash": 500, "unexpected": {"deep": [1, 2, 3]}, "phase": "weird",
     "lots": [{"resource": "R", "quantity": 20, "format": "dutch"}]},  # unknown fields and enums
    {"round": 9, "cash": 500, "fin_inventory": {"A": 30}, "raw_inventory": {"R": 10, "S": 4},
     "lots": [{"resource": "S", "quantity": 5, "reserve_price": 1e9}]},  # last round, absurd reserve
]


@pytest.mark.parametrize("name", ["v6_hybrid", "v2_opt", "v4_bandit", "v7_meta", "v1_rule"])
def test_valid_action_for_every_weird_input(name):
    spec = GameSpec.from_dict(SPEC)
    agent = make_agent(name)
    agent.reset(spec, "me", seed=0)
    for raw in WEIRD * 2:
        obs = Observation.from_dict(raw)
        action = agent.act(obs)
        valid(action, spec, obs.cash)
    assert getattr(agent, "errors", 0) == 0, getattr(agent, "last_error", None)


@pytest.mark.parametrize("spec_raw", [
    {},                                                                   # nothing known at all
    {"products": {"A": {}}},                                              # product with no recipe
    {"products": {"A": {"recipe": {"R": 1}}}, "horizon": 1},              # one-round game
    {"products": {f"P{i}": {"recipe": {f"R{i % 3}": 1 + i % 2}, "unit_cost": i}
                  for i in range(6)}, "lead_time": 2, "capacity": 5},     # 6 products, 3 resources
    {"products": {"A": {"recipe": {"R": 1}, "price_min": 5, "price_max": 5}}},   # no price freedom
    {"products": {"A": {"recipe": {"R": 1}}}, "objective": "market_share"},
    {"products": {"A": {"recipe": {"R": 1}}}, "objective": {"kind": "revenue"}},
])
def test_any_game_shape(spec_raw):
    spec = GameSpec.from_dict(spec_raw)
    agent = make_agent("v6_hybrid")
    agent.reset(spec, "me", seed=1)
    rng = np.random.default_rng(0)
    for t in range(6):
        obs = Observation(round=t, cash=300.0,
                          raw_inventory={r: float(rng.uniform(0, 20)) for r in spec.resources},
                          fin_inventory={p: float(rng.uniform(0, 10)) for p in spec.products},
                          last_prices={p: 10.0 + t for p in spec.products} if t else {},
                          last_sales={p: float(rng.uniform(0, 8)) for p in spec.products} if t else {})
        from bizagent.core.schemas import AuctionLot
        obs.lots = [AuctionLot(r, 30.0, 0.5) for r in spec.resources]
        valid(agent.act(obs), spec, obs.cash)
    assert agent.errors == 0, agent.last_error


def test_engine_failure_falls_back_to_rules(monkeypatch):
    spec = GameSpec.from_dict(SPEC)
    agent = make_agent("v6_hybrid")
    agent.reset(spec, "me")

    def boom(obs):
        raise ZeroDivisionError("bug")
    monkeypatch.setattr(agent.engine, "decide", boom)
    obs = Observation.from_dict({"round": 1, "cash": 500, "fin_inventory": {"A": 5},
                                 "lots": [{"resource": "R", "quantity": 40}]})
    action = agent.act(obs)
    assert agent.errors == 1 and "fallback" in action.info
    assert action.prices and action.bids          # the rule-based policy still plays the round
    valid(action, spec, 500.0)


def test_solver_failure_inside_a_game_is_survivable(monkeypatch):
    import bizagent.production.planner as planner_mod
    from bizagent.simulation.environment import Game
    from bizagent.simulation.scenarios import make_scenario
    monkeypatch.setattr(planner_mod, "linprog", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    sc = make_scenario(3, n_rounds=8)
    agents = [make_agent("v6_hybrid")] + [make_agent("opp_follower") for _ in range(sc.tags["n_firms"] - 1)]
    rows = Game(sc, agents).run()
    assert rows[0]["errors"] == 0 and rows[0]["engine_errors"] == 0
    assert agents[0].engine.planner.n_fallback > 0 and rows[0]["units_sold"] > 0


def test_function_style_entry_and_persistence(tmp_path):
    raw = {"round": 0, "cash": 500, "lots": [{"resource": "R", "quantity": 40, "reserve_price": 1}]}
    answer, memory = decide(raw, None, spec=SPEC)
    assert set(answer) == {"bids", "production", "prices"} and answer["bids"]
    answer2, memory2 = decide({"round": 1, "cash": 400}, memory)
    assert memory2 is memory
    path = str(tmp_path / "state.pkl")
    memory.save(path)
    again = CompetitionAgent.load(path)
    assert json.dumps(again.act({"round": 2, "cash": 400}))     # still JSON-serialisable


def test_cli_handles_malformed_json():
    out = subprocess.run([sys.executable, os.path.join(ROOT, "main.py"), "act"], input="{not json",
                         capture_output=True, text=True, timeout=60)
    assert json.loads(out.stdout)["bids"] == [] and out.returncode == 1
    good = subprocess.run([sys.executable, os.path.join(ROOT, "main.py"), "act", "--spec",
                           os.path.join(ROOT, "examples", "spec.json"), "--obs",
                           os.path.join(ROOT, "examples", "observation.json")],
                          capture_output=True, text=True, timeout=60)
    assert good.returncode == 0 and "prices" in json.loads(good.stdout)
