"""Randomised scenario generator.

Every structural choice the organisers could plausibly make is a random draw
here, so that tuning cannot latch on to one configuration: demand family,
auction format, input scarcity, market dynamics, number of firms, products and
resources, recipes, lead time, one or two decisions per round, what is
revealed, horizon, capacity, cash.

All parameter ranges are our own assumptions (see research/ASSUMPTIONS.md).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.schemas import GameSpec, ObjectiveSpec, ProductSpec, ResourceSpec
from .environment import Scenario
from .market import DemandSpec

DEMANDS = ("linear", "logit", "segments")
ALL_DEMANDS = DEMANDS + ("isoelastic",)
FORMATS = ("first_price", "uniform", "second_price", "posted")
SCARCITY = ("scarce", "balanced", "abundant")
DYNAMICS = ("stable", "trend", "seasonal", "shock", "elasticity")


def make_scenario(seed: int, demand: Optional[str] = None, fmt: Optional[str] = None,
                  scarcity: Optional[str] = None, dynamics: Optional[str] = None,
                  n_firms: Optional[int] = None, n_rounds: Optional[int] = None,
                  n_products: Optional[int] = None, n_resources: Optional[int] = None,
                  lead_time: Optional[int] = None, two_stage: Optional[bool] = None,
                  objective: str = "profit", reveal_prices: Optional[bool] = None,
                  reveal_clearing: Optional[bool] = None, reveal_horizon: Optional[bool] = None,
                  demands=DEMANDS) -> Scenario:
    rng = np.random.default_rng(seed)
    # draw every structural choice first, in a fixed order, so that an override
    # of one choice does not change the others for the same seed
    d_demand = str(rng.choice(list(demands)))
    d_fmt = str(rng.choice(FORMATS, p=[0.45, 0.30, 0.10, 0.15]))
    d_scar = str(rng.choice(SCARCITY))
    d_dyn = str(rng.choice(DYNAMICS, p=[0.36, 0.16, 0.16, 0.16, 0.16]))
    d_firms = int(rng.integers(3, 9))
    d_rounds = int(rng.choice([12, 20, 30, 45, 60]))
    d_prod = int(rng.choice([1, 1, 2, 3]))
    d_res = int(rng.choice([1, 1, 2, 2, 3]))
    d_lead = int(rng.choice([0, 1]))
    d_two = bool(rng.random() < 0.5)
    d_rp = bool(rng.random() < 0.85)
    d_rc = bool(rng.random() < 0.75)
    d_rh = bool(rng.random() < 0.85)
    demand = demand or d_demand
    fmt = fmt or d_fmt
    scarcity = scarcity or d_scar
    dynamics = dynamics or d_dyn
    n = n_firms or d_firms
    T = n_rounds or d_rounds
    J = n_products or d_prod
    K = n_resources or d_res
    lead = d_lead if lead_time is None else int(lead_time)
    two = d_two if two_stage is None else bool(two_stage)
    reveal_prices = d_rp if reveal_prices is None else reveal_prices
    reveal_clearing = d_rc if reveal_clearing is None else reveal_clearing
    reveal_horizon = d_rh if reveal_horizon is None else reveal_horizon

    res_names = [f"R{i + 1}" for i in range(K)]
    prod_names = [f"P{j + 1}" for j in range(J)]
    P0 = rng.uniform(20.0, 80.0, J)
    conv = rng.uniform(0.05, 0.25, J) * P0
    M0 = n * rng.uniform(20.0, 60.0, J)
    recipes = []
    for j in range(J):
        k = 2 if (K >= 2 and rng.random() < 0.5) else 1
        used = rng.choice(K, size=k, replace=False)
        recipes.append({int(r): float(rng.choice([1.0, 2.0], p=[0.7, 0.3])) for r in used})
    for r in range(K):                               # every resource is needed by someone
        if not any(r in rec for rec in recipes):
            recipes[int(rng.integers(0, J))][r] = 1.0
    margin_in = np.array([(P0[j] - conv[j]) / sum(recipes[j].values()) for j in range(J)])

    hints = rng.random() < 0.25
    products, dem, scale = {}, {}, 0.0
    for j, name in enumerate(prod_names):
        pmax = float(P0[j] * rng.uniform(1.6, 5.0)) if rng.random() < 0.85 else None
        products[name] = ProductSpec(
            name=name, recipe={res_names[r]: a for r, a in recipes[j].items()},
            unit_cost=float(conv[j]), capacity_use=1.0, price_min=0.0, price_max=pmax,
            salvage=float(P0[j] * rng.uniform(0.1, 0.4)) if rng.random() < 0.3 else 0.0,
            holding_cost=float(P0[j] * rng.uniform(0.0, 0.015)) if rng.random() < 0.5 else 0.0)
        ds = DemandSpec(kind=demand, P0=float(P0[j]), M0=float(M0[j]),
                        b=float(rng.uniform(1.0, 3.0)), g=float(rng.uniform(0.5, 3.0)),
                        quality=float(P0[j] * rng.uniform(1.05, 1.4)),
                        mu=float(P0[j] * rng.uniform(0.08, 0.25)),
                        beta=float(P0[j] * rng.uniform(0.3, 0.7)),
                        loyal_u=float(rng.uniform(1.5, 2.0)),
                        share_shop=float(rng.uniform(0.2, 0.5)),
                        share_loyal=float(rng.uniform(0.2, 0.4)),
                        eps=float(rng.uniform(1.2, 2.5)), sigma=float(rng.uniform(1.0, 4.0)),
                        noise=float(rng.uniform(0.05, 0.2)),
                        common_noise=float(rng.uniform(0.02, 0.08)),
                        spill=float(rng.uniform(0.1, 0.5)))
        if dynamics == "trend":
            ds.trend = float(rng.uniform(-0.015, 0.02))
        elif dynamics == "seasonal":
            ds.season_amp = float(rng.uniform(0.15, 0.4))
            ds.season_period = int(rng.choice([6, 10, 16]))
        elif dynamics == "shock":
            ds.shock_round = int(rng.uniform(0.3, 0.7) * T)
            ds.shock_mult = float(rng.uniform(0.5, 0.75) if rng.random() < 0.5 else rng.uniform(1.3, 1.7))
        elif dynamics == "elasticity":
            ds.elast_round = int(rng.uniform(0.3, 0.7) * T)
            ds.elast_mult = float(rng.uniform(0.5, 0.75) if rng.random() < 0.5 else rng.uniform(1.4, 2.0))
        dem[name] = ds
        scale += (M0[j] / n) * P0[j] * T

    mult = {"scarce": rng.uniform(0.45, 0.7), "balanced": rng.uniform(0.9, 1.15),
            "abundant": rng.uniform(1.5, 2.2)}[scarcity]
    resources, supply = {}, {}
    for r, name in enumerate(res_names):
        users = [j for j in range(J) if r in recipes[j]]
        mg = float(min(margin_in[j] for j in users))
        supply[name] = float(mult * sum(M0[j] * recipes[j][r] for j in users))
        reserve = mg * (rng.uniform(0.25, 0.5) if fmt == "posted" else rng.uniform(0.05, 0.2))
        resources[name] = ResourceSpec(
            name=name, auction_format=fmt, reserve_price=float(reserve),
            ref_price=float(mg * rng.uniform(0.3, 0.6)) if hints else None,
            salvage=float(0.5 * reserve) if rng.random() < 0.3 else 0.0,
            holding_cost=float(mg * rng.uniform(0.0, 0.01)) if rng.random() < 0.5 else 0.0)

    capacity = None
    if rng.random() < 0.5:
        capacity = float(rng.uniform(1.1, 2.0) * M0.sum() / n)
    cash0 = float(rng.uniform(2.0, 6.0) * sum((M0[j] / n) * P0[j] * 0.5 for j in range(J)))
    spec = GameSpec(products=products, resources=resources, horizon=T, n_firms=n, capacity=capacity,
                    lead_time=lead, initial_cash=cash0, fixed_cost=0.0,
                    demand_hint={p: float(M0[j] / n * rng.uniform(0.7, 1.3))
                                 for j, p in enumerate(prod_names)} if hints else {},
                    objective=ObjectiveSpec.from_dict(objective))
    tags = dict(demand=demand, fmt=fmt, scarcity=scarcity, dynamics=dynamics, n_firms=n, n_rounds=T,
                n_products=J, n_resources=K, lead_time=lead, two_stage=two, objective=objective,
                reveal_prices=reveal_prices, reveal_clearing=reveal_clearing,
                reveal_horizon=reveal_horizon, hints=bool(hints), capacity=capacity is not None)
    return Scenario(name=f"{demand}-{fmt}-{scarcity}-{dynamics}-s{seed}", seed=seed, n_rounds=T,
                    spec=spec, demand=dem, supply=supply, supply_noise=float(rng.uniform(0.0, 0.2)),
                    two_stage=two, reveal_prices=reveal_prices, reveal_clearing=reveal_clearing,
                    reveal_horizon=reveal_horizon, tags=tags, scale=float(scale))
