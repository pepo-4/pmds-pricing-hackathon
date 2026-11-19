from dme_simulator.agents import (
    AssemblyOrder,
    BaseAgent,
    Bid,
    Decision,
    Listing,
    Observation,
)
from dme_simulator.config import GoodsConfig


class CompetitiveRival(BaseAgent):
    def __init__(
        self,
        *,
        init_markup=6.0,
        up=1.04,
        down=0.92,
        floor=2.0,
        stock=12,
        stock_cover=2.0,
        bid_premium=0.12,
        budget_frac=0.55,
        ladder=(1.0, 0.88, 1.12, 0.78, 1.28, 1.45),
        demand_weight=0.3,
    ):
        self.p = {
            "init_markup": init_markup,
            "up": up,
            "down": down,
            "floor": floor,
            "stock": stock,
            "stock_cover": stock_cover,
            "bid_premium": bid_premium,
            "budget_frac": budget_frac,
            "ladder": ladder,
            "demand_weight": demand_weight,
        }
        self.anchor: dict[str, float] = {}
        self.rate: dict[str, float] = {}
        self.floor_mult: dict[str, float] = {}
        self.sales_seen = 0
        self.total_sold: dict[str, int] = {}
        self.last_bid_requested: dict[str, int] = {}
        self.last_auction_seen = 0

    def decide(self, state: Observation, goods_config: GoodsConfig) -> Decision:
        try:
            lots = {lot.good: lot for lot in state.auction_lots}
            sold: dict[str, list[float]] = {}
            for tx in state.sales_transactions[self.sales_seen:]:
                sold.setdefault(tx.good, []).extend([tx.price] * tx.quantity)
            self.sales_seen = len(state.sales_transactions)

            won: dict[str, int] = {}
            for tx in state.auction_transactions[self.last_auction_seen:]:
                won[tx.good] = won.get(tx.good, 0) + tx.quantity
            self.last_auction_seen = len(state.auction_transactions)

            recipe_for = {}
            for name, recipe in goods_config.recipes.items():
                current = recipe_for.get(recipe.output_good)
                cost = sum(lots[m].reserve_price * q for m, q in recipe.inputs.items() if m in lots)
                if cost <= 0:
                    continue
                if current is None or cost < current[0]:
                    recipe_for[recipe.output_good] = (cost, name, recipe)

            cost = {good: value[0] for good, value in recipe_for.items()}
            for good in goods_config.finished_goods:
                units = len(sold.get(good, []))
                alpha = self.p["demand_weight"]
                self.rate[good] = (1 - alpha) * self.rate.get(good, 0.0) + alpha * units
                self.total_sold[good] = self.total_sold.get(good, 0) + units

            for good, c in cost.items():
                if good not in self.anchor:
                    self.anchor[good] = c * self.p["init_markup"]
                    continue
                units = sold.get(good, [])
                if units:
                    if any(price >= self.anchor[good] for price in units):
                        self.anchor[good] *= self.p["up"]
                    else:
                        self.anchor[good] *= 0.97
                else:
                    self.anchor[good] *= self.p["down"]
                floor_mult = min(
                    self.p["floor"],
                    max(1.1, self.floor_mult.get(good, self.p["floor"]) * 0.98),
                )
                self.floor_mult[good] = floor_mult
                self.anchor[good] = max(c * floor_mult, self.anchor[good])

            target = {}
            for good in recipe_for:
                target[good] = int(min(
                    30,
                    max(
                        3,
                        self.p["stock"] + self.p["stock_cover"] * self.rate.get(good, 0.0),
                    ),
                ))

            inv = state.inventory
            def value(good):
                c = cost[good]
                units = sum(recipe_for[good][2].inputs.values()) or 1
                demand = 1.0 + min(1.5, self.rate.get(good, 0.0) / 5.0)
                return ((self.anchor[good] - c) / units) * demand

            order = sorted(recipe_for, key=value, reverse=True)
            need = {good: max(0, target[good] - inv.get(good, 0)) for good in order}

            material_need: dict[str, int] = {}
            for good in order:
                recipe = recipe_for[good][2]
                for material, quantity in recipe.inputs.items():
                    material_need[material] = material_need.get(material, 0) + quantity * need[good]

            bids = []
            planned_cost = 0.0
            wanted = {}
            for material, quantity in material_need.items():
                lot = lots.get(material)
                if lot is None:
                    continue
                missing = quantity - inv.get(material, 0)
                if missing <= 0:
                    continue
                premium = self.p["bid_premium"]
                requested = self.last_bid_requested.get(material, 0)
                if requested > 0 and won.get(material, 0) < requested:
                    premium = min(1.8, premium + 0.28)
                units = min(missing, lot.quantity)
                price = lot.reserve_price * (1 + premium)
                planned_cost += units * price
                wanted[material] = (units, price)

            scale = min(
                1.0,
                (state.capital * self.p["budget_frac"]) / planned_cost,
            ) if planned_cost > 0 else 1.0
            self.last_bid_requested = {}
            for material, (quantity, price) in wanted.items():
                quantity = int(quantity * scale)
                if quantity > 0:
                    bids.append(Bid(material, quantity, float(price)))
                    self.last_bid_requested[material] = quantity

            assembly = []
            for good in order:
                if need[good] > 0:
                    assembly.append(AssemblyOrder(recipe_for[good][1], int(need[good])))

            listings = []
            ladder = self.p["ladder"]
            for good in order:
                held = inv.get(good, 0) + need[good]
                if held <= 0:
                    continue
                floor = cost[good] * self.floor_mult.get(good, self.p["floor"])
                prices = [
                    max(floor, self.anchor[good] * ladder[i % len(ladder)])
                    for i in range(held)
                ]
                listings.extend(Listing(good, 1, float(price)) for price in prices)

            return Decision(bids=bids, assembly=assembly, listings=listings)
        except Exception:
            listings = []
            for good in goods_config.finished_goods:
                held = int(state.inventory.get(good, 0))
                if held > 0:
                    listings.append(Listing(good, held, float(self.anchor.get(good, 1.0))))
            return Decision(listings=listings)
