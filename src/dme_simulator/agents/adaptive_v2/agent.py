from __future__ import annotations

import math

from dme_simulator.agents import (
    AssemblyOrder,
    BaseAgent,
    Bid,
    Decision,
    Listing,
    Observation,
)
from dme_simulator.config import GoodsConfig


class AdaptiveV2Agent(BaseAgent):
    """Demand-, rank-, and competition-aware adaptive agent."""

    def __init__(self, **overrides):
        defaults = {
            "rate_alpha": 0.35,
            "open_turns": 6,
            "budget_share": 0.90,
            "short_budget": 0.75,
            "long_budget": 0.95,
            "bid_start": 2.20,
            "bid_up": 1.30,
            "bid_down": 0.92,
            "bid_cap": 10.0,
            "protect_lead": 0.12,
            "attack_lag": 0.08,
        }
        self.p = {**defaults, **overrides}
        self.rate: dict[str, float] = {}
        self.total_sold: dict[str, int] = {}
        self.anchor: dict[str, float] = {}
        self.floor: dict[str, float] = {}
        self.bid_level: dict[str, float] = {}
        self.last_requested: dict[str, int] = {}
        self.sales_seen = 0
        self.auction_seen = 0
        self.last_public_capital: dict[str, float] = {}
        self.last_public_stock: dict[str, dict[str, int]] = {}

    def decide(self, state: Observation, goods_config: GoodsConfig) -> Decision:
        try:
            return self._decide(state, goods_config)
        except Exception:
            return self._fallback(state, goods_config)

    def _fallback(self, state: Observation, goods_config: GoodsConfig) -> Decision:
        listings = []
        for good in goods_config.finished_goods:
            held = int(state.inventory.get(good, 0))
            price = self.anchor.get(good)
            if held and price:
                listings.append(Listing(good, held, float(price)))
        return Decision(listings=listings)

    def _decide(self, state: Observation, goods_config: GoodsConfig) -> Decision:
        p = self.p
        turn = state.turn
        lots = {lot.good: lot for lot in state.auction_lots}

        sold: dict[str, list[float]] = {}
        for tx in state.sales_transactions[self.sales_seen:]:
            sold.setdefault(tx.good, []).extend([tx.price] * tx.quantity)
        self.sales_seen = len(state.sales_transactions)

        won: dict[str, int] = {}
        for tx in state.auction_transactions[self.auction_seen:]:
            won[tx.good] = won.get(tx.good, 0) + tx.quantity
        self.auction_seen = len(state.auction_transactions)

        recipe_for: dict[str, tuple[str, object, float]] = {}
        for name, recipe in goods_config.recipes.items():
            cost = sum(
                lots[m].reserve_price * quantity
                for m, quantity in recipe.inputs.items()
                if m in lots
            )
            if cost <= 0:
                continue
            current = recipe_for.get(recipe.output_good)
            if current is None or cost < current[2]:
                recipe_for[recipe.output_good] = (name, recipe, cost)

        good_cost = {good: value[2] for good, value in recipe_for.items()}

        for good in goods_config.finished_goods:
            units = len(sold.get(good, []))
            previous_rate = self.rate.get(good, 0.0)
            self.rate[good] = (
                (1.0 - p["rate_alpha"]) * previous_rate
                + p["rate_alpha"] * units
            )
            self.total_sold[good] = self.total_sold.get(good, 0) + units

        ordered_capital = sorted(state.public_capital.values(), reverse=True)
        rank = ordered_capital.index(state.capital) + 1
        leader_capital = ordered_capital[0]
        lead_ratio = (state.capital - ordered_capital[-1]) / max(
            1.0, state.capital
        )

        rival_stock = {good: 0 for good in goods_config.finished_goods}
        rival_capital_pressure = 0.0
        for team, inventory in state.public_inventories.items():
            if team not in state.public_capital:
                continue
            if state.public_capital[team] > state.capital:
                rival_capital_pressure += (
                    state.public_capital[team] - state.capital
                )
            previous = self.last_public_capital.get(team)
            if previous is not None:
                rival_capital_pressure += max(
                    0.0,
                    state.public_capital[team] - previous,
                )
            for good in goods_config.finished_goods:
                rival_stock[good] += int(inventory.get(good, 0))

        rival_capital_pressure = rival_capital_pressure / max(
            1.0, len(state.public_capital)
        )
        self.last_public_capital = dict(state.public_capital)
        self.last_public_stock = {
            team: dict(inventory)
            for team, inventory in state.public_inventories.items()
        }

        for good, cost in good_cost.items():
            if good not in self.anchor:
                markup = 3.2 if turn <= p["open_turns"] else 4.2
                self.anchor[good] = cost * markup
                self.floor[good] = cost * 1.30
                continue

            units = sold.get(good, [])
            ref = self.anchor[good]
            if units:
                high = max(units)
                if high >= ref:
                    ref *= 1.07
                elif self.rate[good] >= 3.0:
                    ref *= 1.03
                else:
                    ref *= 0.97
            else:
                ref *= 0.88 if turn <= p["open_turns"] else 0.92

            if turn <= p["open_turns"]:
                floor_multiplier = 1.25
            elif rival_stock[good] > 20:
                floor_multiplier = 1.18
            else:
                floor_multiplier = 1.35
            self.floor[good] = cost * floor_multiplier
            self.anchor[good] = max(self.floor[good], ref)

        market_saturation = sum(rival_stock.values()) / max(
            1.0, len(goods_config.finished_goods)
        )
        targets: dict[str, int] = {}
        for good in recipe_for:
            demand = self.rate.get(good, 0.0)
            if turn <= p["open_turns"]:
                target = 6 + 1.2 * demand
            elif turn <= 40:
                target = 10 + 2.0 * demand
            else:
                target = 18 + 3.0 * demand

            if rival_stock[good] > target * 2:
                target *= 0.75
            if rank == 1 and lead_ratio > p["protect_lead"]:
                target *= 0.85
            elif rank > 3 and rival_capital_pressure > 500:
                target *= 1.15

            if market_saturation > 60:
                target *= 0.90
            targets[good] = int(max(2, min(30, round(target))))

        order = []
        for good in recipe_for:
            _, recipe, cost = recipe_for[good]
            material_units = sum(recipe.inputs.values()) or 1
            margin = (self.anchor[good] - cost) / material_units
            demand = 1.0 + min(2.0, self.rate.get(good, 0.0) / 4.0)
            competition_penalty = 1.0 + 0.02 * rival_stock[good]
            order.append((good, margin * demand / competition_penalty))
        order.sort(key=lambda item: item[1], reverse=True)

        need = {
            good: max(0, targets[good] - state.inventory.get(good, 0))
            for good, _ in order
        }

        requested_before = dict(self.last_requested)
        new_level: dict[str, float] = {}
        for mat in lots:
            previous = requested_before.get(mat, 0)
            fill = won.get(mat, 0) / previous if previous > 0 else 1.0
            reserve = lots[mat].reserve_price
            level = self.bid_level.get(mat, reserve * p["bid_start"])
            if previous > 0 and fill < 0.8:
                level = level * p["bid_up"]
            elif previous > 0 and fill >= 0.999:
                level = level * p["bid_down"]
            new_level[mat] = level
        self.bid_level = new_level

        material_need: dict[str, int] = {}
        for good, _ in order:
            _, recipe, _ = recipe_for[good]
            for material, quantity in recipe.inputs.items():
                material_need[material] = (
                    material_need.get(material, 0)
                    + quantity * need[good]
                )

        budget_fraction = p["short_budget"] if turn <= 40 else p["long_budget"]
        if rank == 1 and lead_ratio > p["protect_lead"]:
            budget_fraction = min(budget_fraction, p["short_budget"])
        elif rank > 3:
            budget_fraction = max(budget_fraction, p["budget_share"])

        bids = []
        wanted: dict[str, tuple[int, float]] = {}
        planned_cost = 0.0
        for material, quantity in material_need.items():
            lot = lots.get(material)
            if lot is None:
                continue
            missing = quantity - state.inventory.get(material, 0)
            if missing <= 0:
                continue

            value_cap = 0.0
            for good, _ in order:
                _, recipe, cost = recipe_for[good]
                if material not in recipe.inputs:
                    continue
                input_units = sum(recipe.inputs.values()) or 1
                product_value = self.anchor[good]
                value_cap = max(
                    value_cap,
                    product_value / input_units,
                )

            reserve = lot.reserve_price
            level = self.bid_level.get(material, reserve * p["bid_start"])
            price = max(reserve * 1.001, level)
            price = min(price, reserve * p["bid_cap"])
            if value_cap > 0:
                price = min(price, value_cap)

            units = min(missing, lot.quantity)
            wanted[material] = (units, price)
            planned_cost += units * price

        budget = state.capital * budget_fraction
        scale = min(1.0, budget / planned_cost) if planned_cost > 0 else 1.0
        self.last_requested = {}
        for material, (quantity, price) in wanted.items():
            quantity = int(quantity * scale)
            if quantity > 0:
                bids.append(Bid(material, quantity, float(price)))
                self.last_requested[material] = quantity

        assembly = [
            AssemblyOrder(recipe_for[good][0], need[good])
            for good, _ in order
            if need[good] > 0
        ]

        listings = []
        for good, _ in order:
            held = state.inventory.get(good, 0) + need[good]
            if held <= 0:
                continue

            anchor = self.anchor[good]
            floor = self.floor[good]
            if turn <= p["open_turns"]:
                low, high = 0.70, 1.18
            else:
                low, high = 0.78, 1.28

            if rival_stock[good] == 0:
                high *= 1.12

            if held == 1:
                fractions = [0.5]
            else:
                fractions = [i / (held - 1) for i in range(held)]
            prices = [
                max(floor, anchor * (low + (high - low) * frac))
                for frac in fractions
            ]
            prices.sort()
            listings.extend(
                Listing(good, 1, round(float(price), 2))
                for price in prices
            )

        return Decision(bids=bids, assembly=assembly, listings=listings)
