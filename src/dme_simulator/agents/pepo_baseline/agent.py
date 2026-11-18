"""pepo_baseline: simple, safe fallback agent.

Keeps a few units of every good in stock, bids slightly above the reserve price,
and moves each good's price up when it sells out and down when nothing sells.
"""

from __future__ import annotations

from dme_simulator.agents import (
    AssemblyOrder,
    BaseAgent,
    Bid,
    Decision,
    Listing,
    Observation,
)
from dme_simulator.config import GoodsConfig

STOCK = 4
START_MARKUP = 4.0
MIN_MARKUP = 1.3


class PepoBaseline(BaseAgent):
    def __init__(self):
        self.price: dict[str, float] = {}
        self.listed: dict[str, int] = {}
        self.n_sales = 0

    def decide(self, state: Observation, goods_config: GoodsConfig) -> Decision:
        reserve = {lot.good: lot.reserve_price for lot in state.auction_lots}
        sold: dict[str, int] = {}
        for tx in state.sales_transactions[self.n_sales:]:
            sold[tx.good] = sold.get(tx.good, 0) + tx.quantity
        self.n_sales = len(state.sales_transactions)

        need_mat: dict[str, int] = {}
        assembly = []
        listings = []
        for name, recipe in goods_config.recipes.items():
            good = recipe.output_good
            cost = sum(reserve.get(m, 0.0) * q for m, q in recipe.inputs.items())
            if good not in self.price:
                self.price[good] = cost * START_MARKUP
            elif self.listed.get(good, 0) > 0:
                if sold.get(good, 0) >= self.listed[good]:
                    self.price[good] *= 1.10
                elif sold.get(good, 0) == 0:
                    self.price[good] *= 0.90
            self.price[good] = max(self.price[good], cost * MIN_MARKUP)

            have = state.inventory.get(good, 0)
            build = max(0, STOCK - have)
            if build:
                assembly.append(AssemblyOrder(name, build))
                for m, q in recipe.inputs.items():
                    need_mat[m] = need_mat.get(m, 0) + q * build
            listings.append(Listing(good, have + build, float(self.price[good])))
            self.listed[good] = have + build

        bids = []
        for mat, qty in need_mat.items():
            missing = qty - state.inventory.get(mat, 0)
            if mat in reserve and missing > 0:
                bids.append(Bid(mat, int(missing), float(reserve[mat] * 1.1)))
        return Decision(bids=bids, assembly=assembly, listings=listings)
