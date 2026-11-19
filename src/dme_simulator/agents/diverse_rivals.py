from dme_simulator.agents import AssemblyOrder, BaseAgent, Bid, Decision, Listing


class DiverseRival(BaseAgent):
    def __init__(self, strategy):
        self.strategy = strategy
        self.anchor = {}
        self.rate = {}
        self.sold_seen = 0
        self.premium = {}
        self.last_requested = {}
        self.auction_seen = 0
        self.prev_capital = {}
        self.prev_inventory = {}

    def decide(self, state, goods_config):
        lots = {lot.good: lot for lot in state.auction_lots}
        sold = {}
        for tx in state.sales_transactions[self.sold_seen:]:
            sold.setdefault(tx.good, []).extend([tx.price] * tx.quantity)
        self.sold_seen = len(state.sales_transactions)

        recipes = {}
        for name, recipe in goods_config.recipes.items():
            c = sum(lots[m].reserve_price * q for m, q in recipe.inputs.items() if m in lots)
            if c <= 0:
                continue
            if recipe.output_good not in recipes or c < recipes[recipe.output_good][0]:
                recipes[recipe.output_good] = (c, name, recipe)

        for good, (c, _, _) in recipes.items():
            units = len(sold.get(good, []))
            if good not in self.anchor:
                self.anchor[good] = c * {"undercut": 3.0, "stockpile": 7.0, "chip": 6.0,
                                          "liquidity": 5.0, "adaptive": 6.0, "mature": 8.0}[self.strategy]
            elif units:
                if any(price >= self.anchor[good] for price in sold[good]):
                    self.anchor[good] *= 1.05
                else:
                    self.anchor[good] *= 0.97
            else:
                down = {"undercut": 0.82, "stockpile": 0.94, "chip": 0.92,
                        "liquidity": 0.88, "adaptive": 0.91, "mature": 0.93}[self.strategy]
                self.anchor[good] *= down
            floor = c * {"undercut": 1.25, "stockpile": 2.5, "chip": 2.0,
                         "liquidity": 1.5, "adaptive": 2.2, "mature": 2.5}[self.strategy]
            self.anchor[good] = max(floor, self.anchor[good])
            alpha = 0.45 if self.strategy == "adaptive" else 0.30
            self.rate[good] = (1-alpha) * self.rate.get(good, 0.0) + alpha * units

        for team, capital in state.public_capital.items():
            self.prev_capital[team] = capital
        for team, inventory in state.public_inventories.items():
            self.prev_inventory[team] = dict(inventory)

        target = {}
        for good, (c, _, recipe) in recipes.items():
            if self.strategy == "undercut":
                target[good] = int(max(6, 8 + 1.2 * self.rate.get(good, 0.0)))
            elif self.strategy == "stockpile":
                target[good] = int(min(40, 24 + 2.0 * self.rate.get(good, 0.0)))
            elif self.strategy == "chip":
                target[good] = int(min(24, 10 + 1.5 * self.rate.get(good, 0.0)))
            elif self.strategy == "liquidity":
                target[good] = int(max(4, 5 + 0.8 * self.rate.get(good, 0.0)))
            elif self.strategy == "mature":
                target[good] = int(min(30, 12 + (2.0 if state.turn < 80 else 3.0) * self.rate.get(good, 0.0)))
            else:
                rival_stock = sum(
                    inv.get(good, 0) for inv in state.public_inventories.values()
                )
                target[good] = int(min(30, max(7, 12 + 1.5 * self.rate.get(good, 0.0) - rival_stock / 30)))

        def profit(good):
            c = recipes[good][0]
            used = sum(recipes[good][2].inputs.values()) or 1
            return (self.anchor[good] - c) / used

        order = sorted(recipes, key=profit, reverse=True)
        need = {good: max(0, target[good] - state.inventory.get(good, 0)) for good in order}

        material_need = {}
        for good in order:
            for material, quantity in recipes[good][2].inputs.items():
                material_need[material] = material_need.get(material, 0) + quantity * need[good]

        budget_frac = {
            "undercut": 0.80,
            "stockpile": 0.55,
            "chip": 0.95,
            "liquidity": 0.30,
            "adaptive": 0.65,
            "mature": 0.70,
        }[self.strategy]

        bids = []
        planned = {}
        planned_cost = 0.0
        for material, quantity in material_need.items():
            lot = lots.get(material)
            if lot is None:
                continue
            missing = quantity - state.inventory.get(material, 0)
            if missing <= 0:
                continue
            premium = {
                "undercut": 0.20,
                "stockpile": 0.08,
                "chip": 0.55 if material == "CHIP" else 0.18,
                "liquidity": 0.05,
                "adaptive": 0.18,
                "mature": 0.12,
            }[self.strategy]
            units = min(missing, lot.quantity)
            planned[material] = (units, lot.reserve_price * (1 + premium))
            planned_cost += units * planned[material][1]

        scale = min(1.0, state.capital * budget_frac / planned_cost) if planned_cost > 0 else 1.0
        self.last_requested = {}
        for material, (quantity, price) in planned.items():
            quantity = int(quantity * scale)
            if quantity > 0:
                bids.append(Bid(material, quantity, float(price)))
                self.last_requested[material] = quantity

        assembly = [
            AssemblyOrder(recipes[good][1], int(need[good]))
            for good in order if need[good] > 0
        ]

        listings = []
        for good in order:
            held = state.inventory.get(good, 0) + need[good]
            if held <= 0:
                continue
            c = recipes[good][0]
            floor = c * {"undercut": 1.25, "stockpile": 2.5, "chip": 2.0,
                          "liquidity": 1.5, "adaptive": 2.2, "mature": 2.5}[self.strategy]
            if self.strategy == "undercut":
                ladder = (1.0, 0.82, 0.68, 1.08, 1.18)
            elif self.strategy == "liquidity":
                ladder = (0.95, 0.85, 1.02, 0.75, 1.12)
            else:
                ladder = (1.0, 0.88, 1.12, 0.78, 1.25)
            for i in range(held):
                price = max(floor, self.anchor[good] * ladder[i % len(ladder)])
                listings.append(Listing(good, 1, float(price)))

        return Decision(bids=bids, assembly=assembly, listings=listings)
