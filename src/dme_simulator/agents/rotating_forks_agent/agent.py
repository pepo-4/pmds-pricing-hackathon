"""rotating_forks_agent: adaptive price-ladder agent for the Dynamic Market Exchange.

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

# Tuned with tools/evolve.py and tools/variants.py against a replica of the real
# field (README starter agents + stronger proxies), see SOLUZIONE_IT.md.
DEFAULTS = {
    # pricing
    "init_markup": 10.0,         # first anchor = material cost x this (early monopoly: price high)
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
    "stock_mode": "rate",        # "rate" (shelf follows sales) or "ramp" (old)
    "base_stock": 2,             # rate mode: shelf = base + cover_turns x sales rate
    "cover_turns": 5.0,
    "start_stock": 10,           # shelf before the first sale (be ready for early customers)
    "ramp": 0.5,                 # extra shelf units per unit ever sold
    "target_min": 14,            # shelf size once the good sells
    "target_cover": 2.0,         # plus this many turns of recent sales
    "target_max": 30,
    "rate_alpha": 0.3,
    # auction
    "bid_premium": 0.1,          # bid reserve x (1 + premium)
    "start_premium": 2.0,        # turn 1: bid 3x reserve (materials are scarcest early)
    "premium_up": 0.5,           # raise the premium when we were outbid
    "premium_decay": 0.6,
    "premium_cap": 3.0,
    "budget_frac": 0.5,          # never commit more than this share of cash
    # reading the rivals (public inventories, lots, full log when visible)
    "crowd_aware": True,         # use rivals' shelves for pricing and production
    "mono_up": 1.12,             # price step up when no rival holds the good
    "crowd_weight": 0.15,        # production priority / (1 + w x rival sellers)
    "rival_sales_aware": True,   # rivals sell, we don't -> cut harder
    "rival_down": 0.85,
    "lot_aware": True,           # supplier stock growing -> bid at the reserve
    "slack_premium": 0.01,
    "use_market_log": True,      # observe_all_transactions on: use rivals' prices
    "sellout_up": 1.0,           # all listed units sold -> price x this (1 = off)
    # end game: from this turn on, stop restocking and cut prices
    "endgame_turn": 0,           # 0 = off (the horizon is unknown)
    "endgame_cut": 0.9,
}


class RotatingForksAgent(BaseAgent):
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
        self.prev_public: dict[str, dict[str, int]] = {}
        self.prev_mine: dict[str, int] = {}
        self.prev_lot: dict[str, int] = {}
        self.me: str | None = None
        self.my_list_prices: set[float] = set()
        self.my_bid_prices: set[float] = set()
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
        # Scenarios may set observe_all_transactions: then these logs hold
        # EVERY team's trades. Keep ours apart and use the rest as market data.
        new_sales = state.sales_transactions[self.n_sales:]
        new_buys = state.auction_transactions[self.n_buys:]
        self.n_sales = len(state.sales_transactions)
        self.n_buys = len(state.auction_transactions)
        self._identify(state, new_sales, new_buys)
        sold: dict[str, list[float]] = {}
        rival_prices: dict[str, list[float]] = {}
        for tx in new_sales:
            if self._is_mine(tx.seller, tx.price, self.my_list_prices):
                sold.setdefault(tx.good, []).extend([tx.price] * tx.quantity)
            else:
                rival_prices.setdefault(tx.good, []).extend([tx.price] * tx.quantity)
        won: dict[str, int] = {}
        for tx in new_buys:
            if self._is_mine(tx.buyer, tx.price, self.my_bid_prices):
                won[tx.good] = won.get(tx.good, 0) + tx.quantity

        for good in goods_config.finished_goods:
            units = sold.get(good, [])
            self.rate[good] = (1 - p["rate_alpha"]) * self.rate.get(good, 0.0) + p[
                "rate_alpha"
            ] * len(units)
            self.total_sold[good] = self.total_sold.get(good, 0) + len(units)

        # --- 1b. read the rivals (public inventories) ----------------------
        # rival_stock: finished units other teams hold (their shelf, roughly)
        # rival_holders: how many other teams hold the good (competing offers)
        # rival_sold: units that left other teams' shelves since last turn
        finished = goods_config.finished_goods
        mine = state.inventory
        rival_stock, rival_holders, rival_sold = {}, {}, {}
        for good in finished:
            total = holders = drop = 0
            for team, held in state.public_inventories.items():
                q = held.get(good, 0)
                total += q
                holders += q > 0
                before = self.prev_public.get(team, {}).get(good, 0)
                drop += max(0, before - q)
            my_q = mine.get(good, 0)
            my_drop = max(0, self.prev_mine.get(good, 0) - my_q)
            rival_stock[good] = max(0, total - my_q)
            rival_holders[good] = max(0, holders - (my_q > 0))
            rival_sold[good] = max(0, drop - my_drop, len(rival_prices.get(good, [])))
        self.prev_public = {t: dict(h) for t, h in state.public_inventories.items()}
        self.prev_mine = dict(mine)

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
            elif len(units) >= len(listed) and p["sellout_up"] > 1.0:
                # everything we showed was bought: demand exceeds our supply,
                # we are underpriced -> climb fast
                self.anchor[good] = ref * p["sellout_up"]
            elif any(price >= ref * 0.999 for price in units):
                monopoly = p["crowd_aware"] and rival_stock.get(good, 0) == 0
                self.anchor[good] = ref * (p["mono_up"] if monopoly else p["up"])
            elif units:
                self.anchor[good] = ref * p["hold_down"]
            elif p["rival_sales_aware"] and rival_sold.get(good, 0) > 0:
                # customers are buying this good, just not from us: we are dear
                self.anchor[good] = ref * p["rival_down"]
            else:
                # cut faster the longer nothing sells
                extra = max(0, self.streak[good] - p["streak_free"])
                self.anchor[good] = ref * p["down"] * p["streak_down"] ** extra
            if p["use_market_log"] and not units and rival_prices.get(good):
                # full market log visible: rivals sold this good and we did not,
                # so go straight to the highest price that actually sold
                self.anchor[good] = min(self.anchor[good], max(rival_prices[good]))
            self.anchor[good] = max(floor, self.anchor[good])

        # --- 3. stock targets ---------------------------------------------
        endgame = p["endgame_turn"] and turn >= p["endgame_turn"]
        target = {}
        for good in recipe_for:
            if p["stock_mode"] == "rate":
                # shelf follows what we actually sell: a small base plus
                # a few turns of recent sales (crowded markets sell little)
                t = p["base_stock"] + p["cover_turns"] * self.rate.get(good, 0.0)
                if self.total_sold.get(good, 0) == 0:
                    t = p["start_stock"]
            else:
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
            v = (self.anchor.get(good, 0.0) - cost[good]) / used
            if p["crowd_aware"]:
                # scarce materials go first to goods with fewer rival sellers
                v /= 1.0 + p["crowd_weight"] * rival_holders.get(good, 0)
            return v

        order = sorted(recipe_for, key=value, reverse=True)
        need = {g: max(0, target[g] - inv.get(g, 0)) for g in order}

        # --- 5. bids --------------------------------------------------------
        for good in lots:
            requested = self.last_bids.get(good, 0)
            prem = self.premium.get(good, p["start_premium"])
            if good not in self.premium:
                pass  # first time we see this material: use start_premium as is
            elif requested > 0 and won.get(good, 0) < requested:
                prem = min(p["premium_cap"], prem + p["premium_up"])
            else:
                prem = p["bid_premium"] + (prem - p["bid_premium"]) * p["premium_decay"]
                # supplier stock is piling up: demand < supply, so every bid at
                # the reserve gets filled -> stop overpaying
                prev = self.prev_lot.get(good)
                if p["lot_aware"] and prev is not None and lots[good].quantity > prev:
                    prem = min(prem, p["slack_premium"])
            self.premium[good] = prem
            self.prev_lot[good] = lots[good].quantity

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

        self.my_list_prices = {round(float(x.price), 6) for x in listings}
        self.my_bid_prices = {round(float(b.price_per_unit), 6) for b in bids}
        return Decision(bids=bids, assembly=assembly, listings=listings)

    # ------------------------------------------------------------------
    def _identify(self, state: Observation, new_sales, new_buys) -> None:
        """Find our team id: the only team with our exact capital and inventory,
        or, among identical teams, the one trading at our (distinctive) prices."""
        if self.me is not None:
            return
        cands = [
            team for team, cap in state.public_capital.items()
            if abs(cap - state.capital) < 1e-6
            and state.public_inventories.get(team, {}) == state.inventory
        ]
        if len(cands) == 1:
            self.me = cands[0]
            return
        for tx in new_sales:
            if tx.seller in cands and _price_in(tx.price, self.my_list_prices):
                self.me = tx.seller
                return
        for tx in new_buys:
            if tx.buyer in cands and _price_in(tx.price, self.my_bid_prices):
                self.me = tx.buyer
                return

    def _is_mine(self, team: str, price: float, my_prices) -> bool:
        if self.me is not None:
            return team == self.me
        # not identified yet: our trades are the ones at our own prices
        return _price_in(price, my_prices)

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


def _price_in(price: float, prices) -> bool:
    return round(float(price), 6) in prices


class _Named:
    """A recipe paired with its name (RecipeConfig does not carry one)."""

    __slots__ = ("name", "output_good", "inputs")

    def __init__(self, name, recipe):
        self.name = name
        self.output_good = recipe.output_good
        self.inputs = recipe.inputs
