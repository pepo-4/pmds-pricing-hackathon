"""The simulator itself: auction rules, demand, and conservation checks on full games."""
import numpy as np
import pytest

from bizagent.agents.registry import make_agent
from bizagent.simulation.environment import Game
from bizagent.simulation.market import DemandSpec, clear_auction, expected_demand
from bizagent.simulation.opponents import OPPONENTS
from bizagent.simulation.scenarios import make_scenario

RNG = np.random.default_rng(0)
BIDS = [(0, 10.0, 9.0), (1, 10.0, 7.0), (2, 10.0, 5.0)]


def test_first_price_is_pay_as_bid():
    cl = clear_auction("first_price", 15.0, 1.0, BIDS, 3, RNG)
    assert cl.won == [10.0, 5.0, 0.0]
    assert cl.paid == [90.0, 35.0, 0.0]
    assert cl.clearing_price == 7.0 and cl.total_sold == 15.0


@pytest.mark.parametrize("fmt", ["uniform", "second_price"])
def test_uniform_price_is_highest_rejected_bid(fmt):
    cl = clear_auction(fmt, 15.0, 1.0, BIDS, 3, RNG)
    assert cl.won == [10.0, 5.0, 0.0]
    assert cl.clearing_price == 7.0                    # the marginal bid is partly rejected
    assert cl.paid == [70.0, 35.0, 0.0]
    cl = clear_auction(fmt, 20.0, 1.0, BIDS, 3, RNG)
    assert cl.clearing_price == 5.0 and cl.paid == [50.0, 50.0, 0.0]
    cl = clear_auction(fmt, 100.0, 1.0, BIDS, 3, RNG)  # supply not exhausted -> reserve
    assert cl.clearing_price == 1.0 and sum(cl.won) == 30.0


def test_single_unit_second_price_is_vickrey():
    cl = clear_auction("second_price", 1.0, 0.5, [(0, 1.0, 9.0), (1, 1.0, 7.0)], 2, RNG)
    assert cl.won == [1.0, 0.0] and cl.paid == [7.0, 0.0]


def test_posted_price_rations_pro_rata_and_reserve_filters():
    cl = clear_auction("posted", 10.0, 4.0, [(0, 10.0, 4.0), (1, 30.0, 6.0), (2, 5.0, 3.0)], 3, RNG)
    assert cl.won == pytest.approx([2.5, 7.5, 0.0]) and cl.paid == pytest.approx([10.0, 30.0, 0.0])
    cl = clear_auction("first_price", 10.0, 8.0, BIDS, 3, RNG)
    assert cl.won == [10.0, 0.0, 0.0]


@pytest.mark.parametrize("kind", ["linear", "logit", "segments", "isoelastic"])
def test_demand_falls_with_own_price_and_matches_market_size(kind):
    ds = DemandSpec(kind=kind, P0=50, M0=400, quality=60, mu=8, beta=25)
    base = expected_demand(ds, 0, np.full(4, 50.0), 4)
    assert base.sum() == pytest.approx(400.0, rel=1e-6)
    hi = expected_demand(ds, 0, np.array([60.0, 50, 50, 50]), 4)
    lo = expected_demand(ds, 0, np.array([40.0, 50, 50, 50]), 4)
    assert lo[0] > base[0] > hi[0]
    assert hi[1] >= base[1] - 1e-9                     # rivals do not lose when we raise our price


def test_scenarios_are_reproducible_and_overrides_do_not_reshuffle():
    a, b = make_scenario(5), make_scenario(5)
    assert a.name == b.name and a.supply == b.supply
    c = make_scenario(5, fmt="posted")
    assert c.tags["n_firms"] == a.tags["n_firms"] and c.tags["demand"] == a.tags["demand"]


@pytest.mark.parametrize("seed", [11, 12, 13, 14])
def test_game_conserves_material_and_money(seed):
    sc = make_scenario(seed, n_rounds=10)
    names = list(OPPONENTS)[: sc.tags["n_firms"]]
    agents = [make_agent(n) for n in names]
    rows = Game(sc, agents, record=True).run()
    recipe_units = {p: sum(ps.recipe.values()) for p, ps in sc.spec.products.items()}
    one_product = len(recipe_units) == 1
    for r in rows:
        assert r["errors"] == 0
        assert r["min_cash"] >= -1e-6 - r["holding"]            # only holding costs can overdraw
        assert r["units_sold"] <= r["produced"] + 1e-6
        assert r["units_bought"] <= r["units_bid"] + 1e-6
        # money: final = initial + revenue - inputs - production - holding
        assert r["final_cash"] == pytest.approx(
            r["initial_cash"] + r["revenue"] - r["spend"] - r["prod_cost"] - r["holding"], abs=1e-4)
        assert r["fin_left"] == pytest.approx(r["produced"] - r["units_sold"], abs=1e-5)
        if one_product:                                         # inputs: bought = used + left
            used = r["produced"] * list(recipe_units.values())[0]
            assert r["raw_left"] == pytest.approx(r["units_bought"] - used, abs=1e-5)
    assert sorted(r["rank"] for r in rows) == list(range(1, len(rows) + 1))
    assert sum(r["market_share"] for r in rows) == pytest.approx(1.0, abs=1e-6)


def test_agents_see_no_hidden_information():
    seen = []

    class Spy:
        name = "spy"

        def reset(self, spec, agent_id, seed=0):
            self.spec = spec

        def act(self, obs):
            seen.append(obs)
            from bizagent.core.schemas import Action
            return Action()

    sc = make_scenario(21, n_rounds=4, reveal_prices=False, reveal_clearing=False, reveal_horizon=False,
                       two_stage=False)
    spy = Spy()
    Game(sc, [spy, make_agent("opp_follower"), make_agent("opp_greedy")]).run()
    assert spy.spec.horizon is None
    for obs in seen:
        assert obs.competitor_prices == {} and "rounds_left" not in obs.extra
        assert all(r.clearing_price is None and r.total_sold is None for r in obs.auction_results)
    assert not hasattr(spy.spec, "demand")
