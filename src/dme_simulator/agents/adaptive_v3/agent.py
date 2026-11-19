from __future__ import annotations

from dme_simulator.agents import BaseAgent
from dme_simulator.agents.tight import TightAgent


class AdaptiveV3Agent(TightAgent):
    def __init__(self) -> None:
        super().__init__()
        self._regime = ""

    def decide(self, state, goods_config):
        lots = {lot.good: lot for lot in state.auction_lots}
        reserve_avg = (
            sum(lot.reserve_price for lot in lots.values()) / len(lots)
            if lots
            else 0.0
        )

        own_inventory = sum(
            int(state.inventory.get(good, 0))
            for good in goods_config.finished_goods
        )
        initial_finished = own_inventory
        total_lot_qty = sum(lot.quantity for lot in lots.values())

        expensive = reserve_avg >= 25.0
        pricing_only = (
            initial_finished >= 10
            and total_lot_qty >= 200
            and reserve_avg <= 2.0
        )
        auction_only = (
            len(lots) == 1
            and total_lot_qty <= 50
            and reserve_avg >= 8.0
            and initial_finished == 0
        )

        ordered = sorted(state.public_capital.values(), reverse=True)
        rank = ordered.index(state.capital) + 1
        total_rival_stock = sum(
            int(inventory.get(good, 0))
            for team, inventory in state.public_inventories.items()
            if team != self.me
            for good in goods_config.finished_goods
        )

        if expensive:
            regime = "costly"
            self.prior_markup = 4.0
            self.ladder_low = 0.65
            self.ladder_high = 1.0
            self.cover_turns = 3.0
            self.min_stock = 2
            self.max_stock = 40
            self.budget_share = 0.65
            self.value_share = 0.70
            self.max_bid_mult = 5.0
            self.open_turns = 0
        elif pricing_only:
            regime = "pricing"
            self.prior_markup = 3.5
            self.ladder_low = 0.55
            self.ladder_high = 1.20
            self.cover_turns = 4.0
            self.min_stock = 3
            self.max_stock = 80
            self.budget_share = 1.0
            self.value_share = 1.0
            self.max_bid_mult = 10.0
            self.open_turns = 3
            self.open_stock = 12
            self.open_bid = 2.5
        elif auction_only:
            regime = "auction"
            self.prior_markup = 4.0
            self.ladder_low = 0.65
            self.ladder_high = 1.0
            self.cover_turns = 2.0
            self.min_stock = 1
            self.max_stock = 24
            self.budget_share = 0.5
            self.value_share = 0.6
            self.max_bid_mult = 4.0
            self.open_turns = 0
            self.bid_start = 2.0
            self.bid_up = 1.25
        elif state.turn <= 40:
            regime = "short"
            self.prior_markup = 3.5
            self.ladder_low = 0.50
            self.ladder_high = 1.20
            self.cover_turns = 4.0
            self.min_stock = 3
            self.max_stock = 80
            self.budget_share = 1.0
            self.value_share = 1.0
            self.max_bid_mult = 10.0
            self.open_turns = 3
            self.open_stock = 12
            self.open_bid = 2.5
        else:
            regime = "stock"
            self.prior_markup = 4.0
            self.ladder_low = 0.50
            self.ladder_high = 1.10
            self.cover_turns = 6.0
            self.min_stock = 5
            self.max_stock = 120
            self.budget_share = 1.0
            self.value_share = 0.9
            self.max_bid_mult = 10.0
            self.open_turns = 0

        if rank > 3:
            self.budget_share = min(1.0, self.budget_share + 0.10)
            self.max_bid_mult = min(12.0, self.max_bid_mult * 1.10)
        if rank == 1 and state.turn > 40:
            self.cover_turns = max(4.0, self.cover_turns * 0.85)
            self.budget_share = min(0.90, self.budget_share)

        if total_rival_stock > 100:
            self.max_stock = min(self.max_stock, 60)
            self.cover_turns = max(2.0, self.cover_turns * 0.75)

        self._regime = regime
        return super().decide(state, goods_config)
