"""Last line of defence: every action passes through here before it leaves the agent.

The checks are independent of how the action was produced, so they also
protect against bugs in the strategy code:

* prices are finite, inside the allowed range and not below the cost floor
  (the floor is lifted only when liquidating at the end of the game);
* production is non-negative and fits capacity, raw material and cash;
* bids are positive, at or above the reserve, never above the unit's value,
  never for more than the lot, and their total never exceeds the cash budget.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

from ..core.schemas import Action, Bid, GameSpec, Observation


@dataclass
class RiskContext:
    price_floor: Dict[str, float] = field(default_factory=dict)
    bid_cap: Dict[str, float] = field(default_factory=dict)       # max unit price per resource
    bid_budget: Optional[float] = None
    production_cash: Optional[float] = None
    extra_raw: Dict[str, float] = field(default_factory=dict)     # raw expected to arrive before production
    strict_raw: bool = True
    liquidating: bool = False
    fallback_price: Dict[str, float] = field(default_factory=dict)
    integer: bool = False


def _ok(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


class RiskManager:
    def __init__(self, spec: GameSpec):
        self.spec = spec
        self.interventions = 0

    def enforce(self, action: Action, obs: Observation, ctx: RiskContext) -> Action:
        spec = self.spec
        out = Action(info=dict(action.info))

        # ---- prices
        for name, ps in spec.products.items():
            p = action.prices.get(name)
            if not _ok(p) or p <= 0:
                p = ctx.fallback_price.get(name)
                self.interventions += 1
            if not _ok(p) or p <= 0:
                continue
            floor = 0.0 if ctx.liquidating else ctx.price_floor.get(name, 0.0)
            floor = max(floor, ps.price_min)
            if p < floor:
                p = floor
                self.interventions += 1
            if ps.price_max is not None and p > ps.price_max:
                p = ps.price_max
            out.prices[name] = float(p)

        # ---- production
        raw = {r: max(0.0, obs.raw_inventory.get(r, 0.0)) + max(0.0, ctx.extra_raw.get(r, 0.0))
               for r in spec.resources}
        cap_left = spec.capacity if spec.capacity is not None else math.inf
        cash_left = ctx.production_cash if ctx.production_cash is not None else math.inf
        for name, ps in spec.products.items():
            x = action.production.get(name, 0.0)
            if not _ok(x) or x <= 0:
                continue
            if ps.capacity_use > 0:
                x = min(x, cap_left / ps.capacity_use)
            if ctx.strict_raw:
                for r, a in ps.recipe.items():
                    if a > 0:
                        x = min(x, raw.get(r, 0.0) / a)
            if ps.unit_cost > 0:
                x = min(x, cash_left / ps.unit_cost)
            if ctx.integer:
                x = math.floor(x + 1e-9)
            if x <= 1e-9:
                continue
            if x < action.production.get(name, 0.0) - 1e-6:
                self.interventions += 1
            cap_left -= x * ps.capacity_use
            cash_left -= x * ps.unit_cost
            for r, a in ps.recipe.items():
                raw[r] = raw.get(r, 0.0) - x * a
            out.production[name] = float(x)

        # ---- bids
        lots = {}
        for lot in obs.lots:
            lots[lot.resource] = lots.get(lot.resource, 0.0) + lot.quantity
        reserve = {lot.resource: lot.reserve_price for lot in obs.lots}
        budget = ctx.bid_budget if ctx.bid_budget is not None else max(0.0, obs.cash)
        budget = max(0.0, min(budget, max(0.0, obs.cash)))
        left = dict(lots)
        for b in sorted(action.bids, key=lambda b: -(b.price if _ok(b.price) else 0.0)):
            if not (_ok(b.price) and _ok(b.quantity)) or b.price <= 0 or b.quantity <= 0:
                self.interventions += 1
                continue
            if b.resource not in lots:
                self.interventions += 1
                continue
            price = b.price
            cap = ctx.bid_cap.get(b.resource)
            if cap is not None and price > cap:
                price = cap
                self.interventions += 1
            if price < reserve.get(b.resource, 0.0) or price <= 0:
                continue
            qty = min(b.quantity, left.get(b.resource, 0.0), budget / price)
            if ctx.integer:
                qty = math.floor(qty + 1e-9)
            if qty <= 1e-9:
                continue
            left[b.resource] -= qty
            budget -= qty * price
            # identical price for the same resource: one bid, not several
            same = next((o for o in out.bids if o.resource == b.resource
                         and abs(o.price - price) <= 1e-9 * max(1.0, price)), None)
            if same is not None:
                same.quantity += float(qty)
            else:
                out.bids.append(Bid(b.resource, float(qty), float(price)))
        return out
