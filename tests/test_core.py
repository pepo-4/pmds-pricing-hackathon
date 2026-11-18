"""Unit tests for the decision modules."""
import math

import numpy as np
import pytest

from bizagent.core.schemas import (Action, AuctionLot, AuctionResult, Bid, GameSpec, Observation,
                                   ValidationError)
from bizagent.market.demand_model import BayesianDemandModel
from bizagent.procurement.bidding import BiddingOptimizer, ClearingStats, Tranche
from bizagent.production.planner import DemandCurve, PlanInput, Planner, revenue_segments
from bizagent.risk.risk_manager import RiskContext, RiskManager

SPEC = {"products": {"P": {"recipe": {"R": 1}, "unit_cost": 2.0, "price_max": 100}},
        "resources": {"R": {"auction_format": "first_price", "reserve_price": 1.0}},
        "horizon": 20, "n_firms": 4, "initial_cash": 1000, "lead_time": 0}


# ---------------------------------------------------------------- schemas
def test_spec_parsing_is_lenient_and_validation_is_strict():
    spec = GameSpec.from_dict({"products": [{"name": "A", "recipe": {"X": "2"}, "price_max": "nan"}],
                               "horizon": "abc", "capacity": None})
    assert spec.products["A"].recipe == {"X": 2.0}
    assert spec.products["A"].price_max is None and spec.horizon is None
    assert "X" in spec.resources                      # undeclared resource is added
    spec.validate()
    bad = GameSpec.from_dict(SPEC)
    bad.products["P"].price_min, bad.products["P"].price_max = 10.0, 5.0
    with pytest.raises(ValidationError):
        bad.validate()
    with pytest.raises(ValidationError):
        GameSpec.from_dict({}).validate()


def test_observation_parsing_survives_garbage():
    for raw in (None, [], "x", {}, {"cash": "NaN", "lots": [1, {"resource": "R", "quantity": -5}],
                                    "competitor_prices": {"f": {"P": "inf", "Q": 3}},
                                    "auction_results": [{"resource": "R", "all_bids": [[1], "x", [2, 3]]}],
                                    "unknown": 1}):
        obs = Observation.from_dict(raw)
        assert math.isfinite(obs.cash) and obs.round >= 0
    assert obs.lots[0].quantity == 0.0
    assert obs.competitor_prices == {"f": {"Q": 3.0}}
    assert obs.auction_results[0].all_bids == [[2.0, 3.0]]
    assert obs.extra["unknown"] == 1


def test_action_roundtrip():
    a = Action(bids=[Bid("R", 2, 3)], production={"P": 1.5}, prices={"P": 9})
    assert Action.from_dict(a.to_dict()).to_dict() == a.to_dict()
    assert Action.from_dict({"bids": [{"resource": "R", "quantity": -1, "price": 2}, 7]}).bids == []


# ---------------------------------------------------------------- demand
def _fit(model, a, b, c, n, rng, noise=0.03):
    for _ in range(n):
        p, pc = rng.uniform(8, 14), rng.uniform(9, 12)
        q = max(0.0, a + b * p + c * pc) * (1 + noise * rng.standard_normal())
        model.observe(p, pc, q)


def test_demand_model_recovers_a_linear_curve():
    rng = np.random.default_rng(0)
    m = BayesianDemandModel(forgetting=1.0)
    _fit(m, 100.0, -6.0, 2.0, 60, rng)
    a, b, c = m.params()
    assert b == pytest.approx(-6.0, rel=0.2)
    assert c == pytest.approx(2.0, abs=1.2)
    assert float(m.predict(10.0, 10.0)) == pytest.approx(60.0, rel=0.08)


def test_demand_model_tracks_a_shift_when_forgetting():
    rng = np.random.default_rng(1)
    keep, forget = BayesianDemandModel(forgetting=1.0), BayesianDemandModel(forgetting=0.9)
    for m in (keep, forget):
        _fit(m, 100.0, -6.0, 0.0, 40, rng)
        _fit(m, 160.0, -6.0, 0.0, 15, rng)          # demand jumps by 60 units
    truth = 160.0 - 60.0
    assert abs(float(forget.predict(10.0, 10.0)) - truth) < abs(float(keep.predict(10.0, 10.0)) - truth)


def test_stockouts_raise_the_level_but_not_the_slope():
    m = BayesianDemandModel()
    m.observe(10.0, None, 20.0)
    for p in (10.0, 10.5, 11.0, 11.5):            # prices rise while we keep selling out
        before = float(m.predict(p))
        m.observe(p, None, 20.0, censored=True)
        assert float(m.predict(p)) > before
    assert m.params()[1] < 0                       # still downward sloping
    assert m.censor_streak == 4
    m.observe(12.0, None, 25.0)
    assert m.censor_streak == 0


def test_demand_parameters_are_always_economically_sane():
    rng = np.random.default_rng(2)
    m = BayesianDemandModel()
    for _ in range(30):                            # perverse data: demand rising with price
        p = rng.uniform(5, 15)
        m.observe(p, rng.uniform(5, 15), 10 + 3 * p)
    for _ in range(20):
        a, b, c = m.params(rng, ts_scale=1.0)
        assert b < 0 and 0 <= c <= abs(b) and np.isfinite(a)


# ---------------------------------------------------------------- planner
def _plan_input(raw, fin=0.0, capacity=None, mu=0.0, cash=None, buy=None, cap=None):
    prices = np.linspace(20, 4, 33)
    curve = DemandCurve(prices, 100 - 5 * prices)          # revenue-max at p=10; with cost 4 at p=12
    return PlanInput(products=["P"], resources=["R"], recipe=np.array([[1.0]]), unit_cost=np.array([2.0]),
                     capacity_use=np.array([1.0]), curves=[curve], segments=[revenue_segments(curve)],
                     fin_stock=np.array([fin]), raw_stock=np.array([raw]), raw_use_cost=np.array([2.0]),
                     leftover_value=np.array([mu]), capacity=capacity, cash=cash,
                     purchase_price=buy, leftover_cap=cap)


@pytest.mark.parametrize("solver", ["lp", "greedy"])
def test_planner_finds_the_monopoly_price(solver):
    res = Planner(solver).solve(_plan_input(raw=1000.0))
    # marginal cost = 2 (production) + 2 (input) = 4 -> p* = (20 + 4) / 2 = 12, q* = 40
    assert res.prices[0] == pytest.approx(12.0, abs=0.6)
    assert res.production[0] == pytest.approx(40.0, abs=3.0)
    assert res.solver == solver


@pytest.mark.parametrize("solver", ["lp", "greedy"])
def test_planner_respects_inputs_capacity_and_cash(solver):
    res = Planner(solver).solve(_plan_input(raw=10.0))
    assert res.production[0] <= 10.0 + 1e-6 and res.prices[0] >= 17.5       # ration by price
    res = Planner(solver).solve(_plan_input(raw=1000.0, capacity=15.0))
    assert res.production[0] <= 15.0 + 1e-6
    res = Planner(solver).solve(_plan_input(raw=1000.0, cash=20.0))
    assert res.production[0] * 2.0 <= 20.0 + 1e-6


def test_lp_and_greedy_agree_on_value():
    for raw in (5.0, 25.0, 1000.0):
        lp, gr = Planner("lp").solve(_plan_input(raw)), Planner("greedy").solve(_plan_input(raw))
        assert gr.value == pytest.approx(lp.value, rel=0.03)


def test_carried_stock_is_capped():
    # 200 units on the shelf, each "worth" 9 if kept: without a cap nothing below 9 is sold
    free = Planner("lp").solve(_plan_input(raw=0.0, fin=200.0, mu=9.0))
    capped = Planner("lp").solve(_plan_input(raw=0.0, fin=200.0, mu=9.0, cap=np.array([10.0])))
    assert capped.sales[0] > free.sales[0]
    assert capped.prices[0] < free.prices[0]


def test_planner_falls_back_when_the_lp_breaks(monkeypatch):
    import bizagent.production.planner as planner_mod

    def boom(*a, **k):
        raise RuntimeError("solver down")
    monkeypatch.setattr(planner_mod, "linprog", boom)
    pl = Planner("lp")
    res = pl.solve(_plan_input(raw=1000.0))
    assert res.solver == "greedy" and pl.n_fallback == 1
    assert res.prices[0] == pytest.approx(12.0, abs=0.6)


def test_marginal_value_of_inputs_decreases():
    pl = Planner("lp")
    vals = [pl.solve(_plan_input(raw=q)).value for q in (0, 10, 20, 30, 40, 50)]
    marg = np.diff(vals)
    assert np.all(marg[:-1] >= marg[1:] - 1e-6) and marg[0] > marg[-1]


# ---------------------------------------------------------------- bidding
def test_first_price_bids_are_shaded_and_never_exceed_value():
    b = BiddingOptimizer(shade_mode="winprob")
    for px in (5.0, 5.2, 4.9, 5.1, 5.0):
        b.observe(AuctionResult("R", clearing_price=px, won_qty=0, paid=0))
    price, pwin = b.shade(10.0, "R", "first_price", reserve=1.0)
    assert 5.0 < price < 8.0 and pwin > 0.8          # just above the usual clearing price
    low, _ = b.shade(4.0, "R", "first_price", reserve=1.0)
    assert low is None or low <= 4.0
    assert b.shade(0.5, "R", "first_price", reserve=1.0)[0] is None       # value below reserve


def test_second_price_bids_are_truthful_and_posted_buys_at_the_price():
    b = BiddingOptimizer()
    assert b.shade(10.0, "R", "second_price", 1.0)[0] == pytest.approx(10.0)
    assert b.shade(10.0, "R", "uniform", 1.0, tranche_index=0, n_tranches=4)[0] == pytest.approx(10.0)
    assert b.shade(10.0, "R", "uniform", 1.0, tranche_index=3, n_tranches=4)[0] < 10.0
    assert b.shade(10.0, "R", "posted", 6.0)[0] == pytest.approx(6.0)
    assert b.shade(5.0, "R", "posted", 6.0)[0] is None


def test_bids_respect_budget_and_lot_size():
    b = BiddingOptimizer(shade_mode="truthful")
    bids, _ = b.make_bids([("R", "second_price", 30.0, 1.0,
                            [Tranche(20, 10.0), Tranche(20, 8.0), Tranche(20, 6.0)])], budget=150.0)
    assert sum(x.quantity for x in bids) <= 30.0 + 1e-9
    assert sum(x.quantity * x.price for x in bids) <= 150.0 + 1e-9
    assert all(x.price <= 10.0 for x in bids)


def test_clearing_price_learning_without_revealed_prices():
    b = BiddingOptimizer()
    b.last_bids = {"R": [(6.0, 10.0)]}
    b.observe(AuctionResult("R", clearing_price=None, won_qty=0.0))
    assert b.expected_price("R") > 6.0               # we lost: the price to beat is higher
    b2 = BiddingOptimizer()
    b2.last_bids = {"R": [(6.0, 10.0)]}
    b2.observe(AuctionResult("R", clearing_price=None, won_qty=10.0, paid=60.0))
    assert b2.expected_price("R") < 6.0              # we won everything: we could bid less


def test_win_probability_is_monotone():
    st = ClearingStats()
    for px in (4, 5, 6, 5, 5.5):
        st.update(px)
    g = st.win_prob(np.linspace(1, 10, 30))
    assert np.all(np.diff(g) >= -1e-12) and g[0] < 0.05 and g[-1] > 0.95


# ---------------------------------------------------------------- risk
def test_risk_manager_blocks_invalid_actions():
    spec = GameSpec.from_dict(SPEC)
    rm = RiskManager(spec)
    obs = Observation(cash=100.0, raw_inventory={"R": 5.0}, lots=[AuctionLot("R", 10.0, 1.0)])
    act = Action(bids=[Bid("R", 50, 20.0), Bid("R", 5, float("nan")), Bid("X", 1, 1), Bid("R", 5, 0.5)],
                 production={"P": 999.0}, prices={"P": 0.5})
    out = rm.enforce(act, obs, RiskContext(price_floor={"P": 4.0}, bid_cap={"R": 8.0}, bid_budget=40.0,
                                          production_cash=100.0))
    assert out.prices["P"] == 4.0                                  # floor
    assert out.production["P"] == pytest.approx(5.0)               # raw material on hand
    assert len(out.bids) == 1 and out.bids[0].price == 8.0         # capped at value, rest dropped
    assert out.bids[0].quantity * out.bids[0].price <= 40.0 + 1e-9
    out = rm.enforce(Action(prices={"P": 1e9}), obs, RiskContext(liquidating=True))
    assert out.prices["P"] == 100.0                                # price_max
