"""pepo_agent: adaptive price-ladder agent for the Dynamic Market Exchange.

Each turn:
  1. read last turn's own sales and auction fills;
  2. move each good's price anchor (sold high -> up, sold nothing -> down);
  3. pick a stock target per good from its recent sales rate;
  4. plan production for the shortfall, best margin per unit of material first;
  5. bid for the missing materials just above the reserve, more if we were outbid;
  6. list every unit as its own listing on a ladder of prices around the anchor.
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

# Tuned with tools/evolve.py and checked on 12 scenarios (see SOLUZIONE_IT.md).
DEFAULTS = {
    # pricing
    "init_markup": 8.0,          # first anchor = material cost x this
    "explore_turns": 0,          # turns using the wide ladder (off: probes do it)
    "explore_ladder": (0.45, 0.65, 0.9, 1.25, 1.75, 2.5),
    "ladder": (1.0, 0.9, 1.12, 0.8, 1.25, 1.4),   # listing order = fill priority
    "up": 1.05,                  # a unit sold at or above the anchor
    "hold_down": 0.95,           # sold only below the anchor
    "down": 0.90,                # listed but sold nothing
    "min_markup": 2.5,           # never list below material cost x this...
    "floor_patience": 3,         # ...unless nothing sold at the floor this many turns
    "floor_relax": 0.85,         # then lower the floor by this factor
    "floor_abs": 1.1,            # but never below cost x this
    "split_listings": True,      # one listing per unit
    "probes": (0.6, 0.4),        # cheap test units while a good has never sold
    "explore_keep": 1.0,         # anchor = highest sold price x this on first sale
    "explore_down": 0.7,         # explore: nothing sold
    "streak_free": 0,            # no-sale turns before cuts accelerate
    "streak_down": 1.0,          # extra cut per further no-sale turn (1 = off)
    # stock
    "start_stock": 3,            # shelf before the first sale
    "ramp": 0.5,                 # extra shelf units per unit ever sold
    "target_min": 14,            # shelf size once the good sells
    "target_cover": 2.0,         # plus this many turns of recent sales
    "target_max": 30,
    "rate_alpha": 0.3,
    # auction
    "bid_premium": 0.1,          # bid reserve x (1 + premium)
    "premium_up": 0.25,          # raise the premium when we were outbid
    "premium_decay": 0.6,
    "premium_cap": 2.0,
    "budget_frac": 0.7,          # never commit more than this share of cash
    # end game: from this turn on, stop restocking and cut prices
    "endgame_turn": 0,           # 0 = off (the horizon is unknown)
    "endgame_cut": 0.9,
}


class PepoAgent(BaseAgent):
    """Price ladder + stock targets + adaptive auction premium."""

    def __init__(self, **params):
        self.p = {**DEFAULTS, **params}
        self.anchor: dict[str, float] = {}
        self.rate: dict[str, float] = {}
        self.premium: dict[str, float] = {}
        self.last_listed: dict[str, list[float]] = {}
        self.last_anchor: dict[str, float] = {}
        self.last_bids: dict[str, int] = {}
        self.n_sales = 0
        self.n_buys = 0
        self.mat_price: dict[str, float] = {}
        self.streak: dict[str, int] = {}
        self.floor_mult: dict[str, float] = {}
        self.total_sold: dict[str, int] = {}

    # ------------------------------------------------------------------
    def decide(self, state: Observation, goods_config: GoodsConfig) -> Decision:
        try:
            return self._decide(state, goods_config)
        except Exception:  # never lose a turn: still sell what we hold
            listings = []
            for good in goods_config.finished_goods:
                held = int(state.inventory.get(good, 0))
                price = self.anchor.get(good, 0.0)
                if held > 0 and price > 0:
                    listings.append(Listing(good, held, float(price)))
            return Decision(listings=listings)

    def _decide(self, state: Observation, goods_config: GoodsConfig) -> Decision:
        p = self.p
        turn = state.turn
        lots = {lot.good: lot for lot in state.auction_lots}
        for good, lot in lots.items():
            self.mat_price[good] = lot.reserve_price

        recipe_for = self._best_recipes(goods_config)
        cost = {g: max(1.0, self._recipe_cost(r)) for g, r in recipe_for.items()}

        # --- 1. feedback from last turn ---------------------------------
        sold: dict[str, list[float]] = {}
        for tx in state.sales_transactions[self.n_sales:]:
            sold.setdefault(tx.good, []).extend([tx.price] * tx.quantity)
        self.n_sales = len(state.sales_transactions)
        won: dict[str, int] = {}
        for tx in state.auction_transactions[self.n_buys:]:
            won[tx.good] = won.get(tx.good, 0) + tx.quantity
        self.n_buys = len(state.auction_transactions)

        for good in goods_config.finished_goods:
            units = sold.get(good, [])
            self.rate[good] = (1 - p["rate_alpha"]) * self.rate.get(good, 0.0) + p[
                "rate_alpha"
            ] * len(units)
            self.total_sold[good] = self.total_sold.get(good, 0) + len(units)

        # --- 2. prices ----------------------------------------------------
        for good in goods_config.finished_goods:
            c = cost.get(good)
            if c is None:
                continue
            # a floor that blocks every sale is relaxed step by step
            fm = self.floor_mult.get(good, p["min_markup"])
            if (
                self.streak.get(good, 0) >= p["floor_patience"]
                and self.last_listed.get(good)
                and good in self.anchor
                and self.anchor[good] <= c * fm * 1.01
                and not sold.get(good)
            ):
                fm = max(p["floor_abs"], fm * p["floor_relax"])
            self.floor_mult[good] = fm
            floor = c * fm
            if good not in self.anchor:
                self.anchor[good] = max(floor, c * p["init_markup"])
                continue
            listed = self.last_listed.get(good)
            if not listed:
                continue
            units = sold.get(good, [])
            ref = self.last_anchor.get(good, self.anchor[good])
            self.streak[good] = 0 if units else self.streak.get(good, 0) + 1
            first_sale = units and self.total_sold[good] == len(units)
            if first_sale and p["probes"]:
                # first evidence of what customers pay: jump there, and let the
                # floor follow if customers pay less than our usual minimum
                top = max(units)
                self.anchor[good] = max(top * p["explore_keep"], ref * p["down"]) \
                    if top >= ref * 0.999 else top * p["explore_keep"]
                self.floor_mult[good] = min(fm, max(p["floor_abs"], top / c * 0.95))
                self.anchor[good] = max(c * self.floor_mult[good], self.anchor[good])
                continue
            if turn <= p["explore_turns"] + 1:
                if units:
                    self.anchor[good] = max(units) * p["explore_keep"]
                else:
                    self.anchor[good] = ref * p["explore_down"]
            elif any(price >= ref * 0.999 for price in units):
                self.anchor[good] = ref * p["up"]
            elif units:
                self.anchor[good] = ref * p["hold_down"]
            else:
                # cut faster the longer nothing sells
                extra = max(0, self.streak[good] - p["streak_free"])
                self.anchor[good] = ref * p["down"] * p["streak_down"] ** extra
            self.anchor[good] = max(floor, self.anchor[good])

        # --- 3. stock targets ---------------------------------------------
        endgame = p["endgame_turn"] and turn >= p["endgame_turn"]
        target = {}
        for good in recipe_for:
            # grow the shelf only once the good has proven it sells
            floor_stock = min(
                p["target_min"],
                p["start_stock"] + p["ramp"] * self.total_sold.get(good, 0),
            )
            t = floor_stock + p["target_cover"] * self.rate.get(good, 0.0)
            target[good] = 0 if endgame else int(min(p["target_max"], round(t)))

        # --- 4. production plan ------------------------------------------
        inv = state.inventory
        def value(good: str) -> float:
            r = recipe_for[good]
            used = sum(r.inputs.values()) or 1
            return (self.anchor.get(good, 0.0) - cost[good]) / used

        order = sorted(recipe_for, key=value, reverse=True)
        need = {g: max(0, target[g] - inv.get(g, 0)) for g in order}

        # --- 5. bids --------------------------------------------------------
        for good in lots:
            requested = self.last_bids.get(good, 0)
            prem = self.premium.get(good, p["bid_premium"])
            if requested > 0 and won.get(good, 0) < requested:
                prem = min(p["premium_cap"], prem + p["premium_up"])
            else:
                prem = p["bid_premium"] + (prem - p["bid_premium"]) * p["premium_decay"]
            self.premium[good] = prem

        material_need: dict[str, int] = {}
        for good in order:
            for mat, qty in recipe_for[good].inputs.items():
                material_need[mat] = material_need.get(mat, 0) + qty * need[good]
        bids = []
        planned_cost = 0.0
        wanted = {}
        for mat, qty in material_need.items():
            lot = lots.get(mat)
            missing = qty - inv.get(mat, 0)
            if lot is None or missing <= 0:
                continue
            wanted[mat] = (min(missing, lot.quantity), lot.reserve_price * (1 + self.premium[mat]))
            planned_cost += wanted[mat][0] * wanted[mat][1]
        budget = max(0.0, state.capital * p["budget_frac"])
        scale = min(1.0, budget / planned_cost) if planned_cost > 0 else 1.0
        self.last_bids = {}
        for mat, (qty, price) in wanted.items():
            qty = int(qty * scale)
            if qty > 0:
                bids.append(Bid(mat, qty, float(price)))
                self.last_bids[mat] = qty

        assembly = [
            AssemblyOrder(recipe_for[g].name, int(need[g])) for g in order if need[g] > 0
        ]

        # --- 6. listings ----------------------------------------------------
        listings = []
        self.last_listed = {}
        self.last_anchor = {}
        ladder = p["explore_ladder"] if turn <= p["explore_turns"] else p["ladder"]
        for good in order:
            anchor = self.anchor[good]
            if endgame:
                anchor *= p["endgame_cut"]
                self.anchor[good] = max(cost[good] * self.floor_mult.get(good, p["min_markup"]), anchor)
            floor = cost[good] * self.floor_mult.get(good, p["min_markup"])
            n = inv.get(good, 0) + need[good]
            if n <= 0:
                continue
            prices = [max(floor, anchor * ladder[i % len(ladder)]) for i in range(n)]
            if p["probes"] and turn > 1 and self.total_sold.get(good, 0) == 0:
                # never sold yet: a few cheap probe units find the market fast
                probe_floor = cost[good] * p["floor_abs"]
                for i, mult in enumerate(p["probes"], start=1):
                    if i < n:
                        prices[i] = max(probe_floor, anchor * mult)
            if p["split_listings"]:
                listings.extend(Listing(good, 1, float(x)) for x in prices)
            else:
                listings.append(Listing(good, n, float(anchor)))
            self.last_listed[good] = prices
            self.last_anchor[good] = anchor

        return Decision(bids=bids, assembly=assembly, listings=listings)

    # ------------------------------------------------------------------
    def _recipe_cost(self, recipe) -> float:
        return sum(self.mat_price.get(m, 0.0) * q for m, q in recipe.inputs.items())

    def _best_recipes(self, goods_config: GoodsConfig):
        """Cheapest recipe per finished good (scenarios may offer several)."""
        best = {}
        for name, recipe in goods_config.recipes.items():
            recipe = _Named(name, recipe)
            current = best.get(recipe.output_good)
            if current is None or self._recipe_cost(recipe) < self._recipe_cost(current):
                best[recipe.output_good] = recipe
        return best


class _Named:
    """A recipe paired with its name (RecipeConfig does not carry one)."""

    __slots__ = ("name", "output_good", "inputs")

    def __init__(self, name, recipe):
        self.name = name
        self.output_good = recipe.output_good
        self.inputs = recipe.inputs
