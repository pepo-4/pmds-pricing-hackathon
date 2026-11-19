"""Tight v7 = v6 + market reading from public information.

New in v7 (each one can be switched off with a class flag):
- comp_aware: `public_inventories` shows every team's stock. For each good we
  count the rival units and the rival teams that hold it (or could build it
  from their raw materials). A good nobody else can offer this turn gets a
  higher price ladder (mono_boost); scarce materials go first to goods with
  less rival stock.
- lot_signal: if a supplier's stock GREW since last turn, last turn's lot was
  not sold out, so every bid above the reserve was filled at its own price:
  the auction is uncontested and we bid just above the reserve instead of our
  learned level (pay-your-bid: anything more is pure loss).
  -> tested and switched OFF: suppliers also raise production after running dry,
  so the stock grows in contested big markets too and we starved.
- cap_mode 2 (replaces v6's sell-through rule): the bid cap (multiple of the
  reserve) grows x1.25 only when we lost material at auction AND at least half
  of the goods we listed sold out (demand is strong); it shrinks when finished
  stock covers more than 2 x cover_turns. v6's rule shrank the cap in healthy
  big markets (we always list ~3 turns of stock) and starved us there.
  The value cap also discounts goods that sit in stock (turnover factor).
Local results (crowded zoo, head-to-head): default 9.5 vs v6 13.2,
extended guess 4.3 vs 9.0, Davide's extended waves 2.0 vs 6.7.

Tight v6 = elia_fix (davide: v3 + price-up fix + faster bid decay, no opening)
+ an adaptive bid cap and larger stock, so it also scales to big markets.

Portal: elia_fix was 3rd on `default`; v4/v5 (aggressive opening) ~28th.
The portal added `default_extended` (240 turns, 40k capital, far more demand):
there materials are worth much more than 4x the reserve and 30 units of stock
is too little. The bid cap starts at 4x the reserve and grows while we sell
almost everything we list; it shrinks back when stock stays unsold.

elia_fix notes:

Tight v3 (= v1 + aggressive opening): adaptive bidding, margin-ranked production, per-unit price ladder.

Ideas:
- Pay-your-bid auction: bid just above what wins. Per material we keep a
  price level that rises when we were not filled and decays when we were.
- Production: keep a small stock of every finished good, giving scarce
  materials to the goods with the highest margin first.
- Sales: every unit is listed as its own 1-unit listing on a price ladder
  around a reference price. Each listing counts as a separate offer for the
  customers, and the ladder tells us whether the reference is too high or low.
"""

from __future__ import annotations

import math

from dme_simulator.agents import (
    AssemblyOrder, BaseAgent, Bid, Decision, Listing, Observation,
)
from dme_simulator.config import GoodsConfig


class Agent444(BaseAgent):
    # --- pricing ---
    prior_markup = 4.0          # first reference price = material cost * this
    ladder_low = 0.6           # ladder spans [low, high] * reference
    ladder_high = 1.0
    explore_turns = 6           # wider ladder and faster moves early on
    step_up = 0.07
    step_down = 0.15
    min_markup = 1.3            # never list below material cost * this
    # --- stock and production ---
    min_stock = 2
    max_stock = 200
    cover_turns = 3.0           # target stock = sales rate * cover + min_stock
    # --- bidding ---
    bid_start = 1.5            # first bid = reserve * this
    bid_up = 1.5
    bid_down = 0.94
    # --- fixes found with tools/analyze.py (40-turn behaviour) ---
    up_thresh = 0.9             # raise the reference when the top rungs sell (was 1.1: unreachable once
                                # the ladder tops out at 1.0 x ref, so prices could only fall)
    bid_down_full = 0.85        # fully filled -> cut the bid faster (pay-your-bid: overpaying is pure loss)
    value_share = 0.8          # max bid per material unit = share of product value
    budget_share = 0.8          # max share of capital committed per turn
    # --- opening: in the first turns few teams have goods, so be first to market ---
    max_bid_mult = 4.0          # starting bid cap, as a multiple of the reserve
    cap_grow = 1.25             # cap *= this after a turn where we sold almost all we listed
    cap_shrink = 0.9            # cap *= this (down to max_bid_mult) when much stays unsold
    cap_limit = 60.0            # absolute ceiling for the adaptive cap
    grow_at = 0.75               # sell-through (sold / listed) that counts as "demand absorbs it all"
    open_turns = 0               # opening disabled: aggressive first turns lost money in crowds (was 5)
    open_stock = 6              # minimum stock target per good during the opening
    open_bid = 3.0              # bid at least reserve * this during the opening
    early_low = 0.75            # price ladder during the exploration turns
    early_high = 1.6
    # --- v7: market reading from public information ---
    comp_aware = 1              # use rivals' public stock for pricing and production order
    mono_boost = 1.15           # ladder multiplier for a good no rival can offer this turn
    comp_weight = 2.0           # production order: score /= 1 + comp_weight * rival_units / our_target
    lot_signal = 0              # OFF: supplier stock also grows when the supplier ramps up production,
                                # so this signal starved us in big markets (tested: x9 vs x37 capital)
    uncontested_mult = 1.01     # bid = reserve * this when the auction looks uncontested
    cap_mode = 2                # 0 = v6 rule (sell-through); 1 = lost bids + low stock;
                                # 2 = lost bids + at least grow_sellout of our listed goods sold out
    turnover_floor = 0.25       # value cap uses ref * clamp(sales_rate * cover / stock, floor, 1)
    grow_needs_sales = 0        # 1 = the cap only grows once we have sold something recently
    grow_sellout = 0.5          # cap_mode 2: grow only if this share of our listed goods sold out

    def __init__(self, **overrides) -> None:
        # any class parameter can be overridden: TightAgent(max_bid_mult=6.0, ...)
        for k, v in overrides.items():
            if not hasattr(type(self), k):
                raise TypeError(f"unknown parameter {k}")
            setattr(self, k, v)
        self.me: str | None = None
        self.ref_price: dict[str, float] = {}
        self.sales_rate: dict[str, float] = {}
        self.bid_level: dict[str, float] = {}
        self.unit_cost: dict[str, float] = {}
        self.last_bids: dict[str, tuple[int, float]] = {}
        self.last_ladder: dict[str, list[float]] = {}
        self.seen_auction = 0
        self.seen_sales = 0
        self.cap_mult = self.max_bid_mult
        self.prev_lot_qty: dict[str, int] = {}

    # ------------------------------------------------------------------
    def decide(self, state: Observation, goods_config: GoodsConfig) -> Decision:
        try:
            return self._decide(state, goods_config)
        except Exception:
            return self._fallback(state, goods_config)

    def _fallback(self, state: Observation, goods_config: GoodsConfig) -> Decision:
        listings = []
        for good in goods_config.finished_goods:
            qty = int(state.inventory.get(good, 0))
            price = self.ref_price.get(good)
            if qty > 0 and price and math.isfinite(price) and price > 0:
                listings.append(Listing(good, qty, float(price)))
        return Decision(listings=listings)

    # ------------------------------------------------------------------
    def _identify(self, state: Observation) -> None:
        matches = [
            team
            for team, inv in state.public_inventories.items()
            if abs(state.public_capital.get(team, -1e18) - state.capital) < 1e-6
            and {k: v for k, v in inv.items() if v} == {k: v for k, v in state.inventory.items() if v}
        ]
        if len(matches) == 1:
            self.me = matches[0]

    def _mine(self, who: str) -> bool:
        return self.me is None or who == self.me

    def _decide(self, state: Observation, gc: GoodsConfig) -> Decision:
        self._identify(state)
        turn = state.turn
        inv = {k: int(v) for k, v in state.inventory.items()}
        lots = {lot.good: lot for lot in state.auction_lots}
        reserve_avg = (
            sum(lot.reserve_price for lot in lots.values()) / len(lots)
            if lots else 0.0
        )
        total_lot_qty = sum(lot.quantity for lot in lots.values())
        initial_finished = sum(inv.get(g, 0) for g in gc.finished_goods)

        if reserve_avg >= 25.0:
            self.prior_markup = 4.5
            self.ladder_low = 0.65
            self.ladder_high = 1.0
            self.cover_turns = 3.0
            self.min_stock = 2
            self.max_stock = 40
            self.budget_share = 0.65
            self.value_share = 0.70
            self.max_bid_mult = 5.0
            self.open_turns = 0
        elif initial_finished >= 10 and total_lot_qty >= 200 and reserve_avg <= 2.0:
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
        elif len(lots) == 1 and total_lot_qty <= 50 and reserve_avg >= 8.0:
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
        elif turn <= 40:
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

        # --- v7: rivals' public stock ----------------------------------------
        rival_units = {g: 0 for g in gc.finished_goods}
        rival_sellers = {g: 0 for g in gc.finished_goods}
        known_me = self.me is not None
        if self.comp_aware and known_me:
            for team, tinv in state.public_inventories.items():
                if team == self.me:
                    continue
                can_offer = set()
                for rec in gc.recipes.values():
                    if all(int(tinv.get(m, 0)) >= q for m, q in rec.inputs.items()):
                        can_offer.add(rec.output_good)        # could build it this turn
                for g in gc.finished_goods:
                    have = int(tinv.get(g, 0))
                    rival_units[g] += have
                    if have > 0 or g in can_offer:
                        rival_sellers[g] += 1

        # --- learn from last turn's auction --------------------------------
        new_auction = state.auction_transactions[self.seen_auction:]
        self.seen_auction = len(state.auction_transactions)
        won: dict[str, int] = {}
        for tx in new_auction:
            if not self._mine(tx.buyer):
                continue
            won[tx.good] = won.get(tx.good, 0) + tx.quantity
            old = self.unit_cost.get(tx.good, tx.price)
            self.unit_cost[tx.good] = 0.7 * old + 0.3 * tx.price

        # --- v7: was each supplier sold out last turn? -------------------------
        # Stock can also grow because the supplier raises production after running
        # dry, so the signal only counts when we ALSO got everything we asked for.
        uncontested = {}
        for m, lot in lots.items():
            prev = self.prev_lot_qty.get(m)
            asked = self.last_bids.get(m, (0, 0.0))[0]
            filled = asked > 0 and won.get(m, 0) >= asked
            uncontested[m] = bool(self.lot_signal and prev is not None and lot.quantity > prev and filled)
            self.prev_lot_qty[m] = lot.quantity

        # --- learn from last turn's sales ----------------------------------
        new_sales = state.sales_transactions[self.seen_sales:]
        self.seen_sales = len(state.sales_transactions)
        sold_prices: dict[str, list[float]] = {}
        for tx in new_sales:
            if not self._mine(tx.seller):
                continue
            sold_prices.setdefault(tx.good, []).extend([tx.price] * tx.quantity)

        # material cost estimates (what we pay, else the reserve)
        mat_cost = {
            m: self.unit_cost.get(m, lots[m].reserve_price if m in lots else 1.0)
            for m in gc.raw_materials
        }
        # cheapest recipe per finished good
        recipe_for: dict[str, str] = {}
        good_cost: dict[str, float] = {}
        for name, rec in gc.recipes.items():
            c = sum(mat_cost.get(m, 1.0) * q for m, q in rec.inputs.items())
            g = rec.output_good
            if g not in good_cost or c < good_cost[g]:
                good_cost[g], recipe_for[g] = c, name

        early = turn <= self.explore_turns
        tot_sold = tot_listed = 0
        goods_listed = goods_sold_out = 0
        for g in gc.finished_goods:
            cost = max(good_cost.get(g, 1.0), 1e-6)
            if g not in self.ref_price:
                self.ref_price[g] = cost * self.prior_markup
                self.sales_rate[g] = 1.0
            ladder = self.last_ladder.get(g, [])
            sold = sorted(sold_prices.get(g, []))
            n_sold = len(sold)
            tot_sold += n_sold
            tot_listed += min(len(self.last_ladder.get(g, [])), n_sold + inv.get(g, 0))
            self.sales_rate[g] = 0.7 * self.sales_rate[g] + 0.3 * n_sold
            # units really on sale last turn = sold + still in stock (we list all)
            listed = min(len(ladder), n_sold + inv.get(g, 0))
            if listed > 0:
                goods_listed += 1
                goods_sold_out += n_sold >= listed
                up = self.step_up * (2.0 if early else 1.0)
                down = self.step_down * (1.5 if early else 1.0)
                ref = self.ref_price[g]
                if n_sold >= listed:
                    ref *= 1 + 2 * up                       # sold out: clearly too cheap
                elif n_sold and sold[-1] >= ref * self.up_thresh:
                    ref *= 1 + up                           # upper rungs sell
                elif n_sold:
                    ref *= 1 - 0.3 * down                   # only cheap rungs sell
                else:
                    ref *= 1 - down                         # nothing sold
                self.ref_price[g] = max(ref, cost * self.min_markup)

        # --- adaptive bid cap ------------------------------------------------------
        if self.cap_mode == 0:
            # v6: grow while demand absorbs everything we list
            if tot_listed > 0:
                sell_through = tot_sold / tot_listed
                if sell_through >= self.grow_at:
                    self.cap_mult = min(self.cap_limit, self.cap_mult * self.cap_grow)
                elif sell_through < 0.5:
                    self.cap_mult = max(self.max_bid_mult, self.cap_mult * self.cap_shrink)
        else:
            # v7: grow when we lost materials AND we sell what we have (stock covers
            # fewer turns than planned); shrink when finished stock piles up.
            stock_now = sum(inv.get(g, 0) for g in gc.finished_goods)
            rate = sum(self.sales_rate.get(g, 0.0) for g in gc.finished_goods)
            cover_now = stock_now / max(0.3, rate)
            lost_bid = any(
                asked > 0 and won.get(m, 0) < 0.8 * asked for m, (asked, _) in self.last_bids.items()
            )
            strong = goods_listed > 0 and goods_sold_out >= self.grow_sellout * goods_listed
            if self.cap_mode == 2:
                grow = lost_bid and strong
            else:
                grow = lost_bid and cover_now <= self.cover_turns and (tot_sold > 0 or not self.grow_needs_sales)
            if grow:
                self.cap_mult = min(self.cap_limit, self.cap_mult * self.cap_grow)
            elif cover_now > 2 * self.cover_turns:
                self.cap_mult = max(self.max_bid_mult, self.cap_mult * self.cap_shrink)

        # --- production plan -------------------------------------------------
        target = {
            g: int(min(self.max_stock,
                       round(self.sales_rate[g] * self.cover_turns)
                       + (max(self.min_stock, self.open_stock) if turn <= self.open_turns else self.min_stock)))
            for g in gc.finished_goods
        }
        margin = {g: self.ref_price[g] - good_cost.get(g, 0.0) for g in gc.finished_goods}

        def score(g: str) -> float:
            s = margin[g] / max(1, sum(gc.recipes[recipe_for[g]].inputs.values()))
            if self.comp_aware and known_me:
                s /= 1.0 + self.comp_weight * rival_units[g] / max(1, target[g])
            return s

        ranked = sorted(
            (g for g in gc.finished_goods if g in recipe_for and margin[g] > 0),
            key=score,
            reverse=True,
        )
        need = {g: max(0, target[g] - inv.get(g, 0)) for g in ranked}

        # --- bid levels: raise only when a material is really starving us ----
        for mat, (asked, price) in self.last_bids.items():
            if asked <= 0:
                continue
            fill = won.get(mat, 0) / asked
            starving = any(
                mat in gc.recipes[recipe_for[g]].inputs
                and inv.get(g, 0) < max(1, target[g] // 2)
                for g in ranked
            )
            if fill < 0.8 and starving:
                self.bid_level[mat] = price * self.bid_up
            elif fill < 0.8:
                self.bid_level[mat] = price
            else:
                self.bid_level[mat] = price * (self.bid_down_full if fill >= 0.999 else self.bid_down)
        assembly = [AssemblyOrder(recipe_for[g], need[g]) for g in ranked if need[g] > 0]

        # materials required for the plan, minus what we hold
        want = {m: 0 for m in gc.raw_materials}
        for g in ranked:
            for m, q in gc.recipes[recipe_for[g]].inputs.items():
                want[m] = want.get(m, 0) + q * need[g]

        # --- bids --------------------------------------------------------------
        bids = []
        self.last_bids = {}
        budget = state.capital * self.budget_share
        for m in gc.raw_materials:
            lot = lots.get(m)
            if lot is None or lot.quantity <= 0:
                continue
            qty = min(max(0, want.get(m, 0) - inv.get(m, 0)), lot.quantity)
            if qty <= 0:
                continue
            # value cap: share of the best product's price per input unit
            cap = 0.0
            for g in ranked:
                rec = gc.recipes[recipe_for[g]]
                if m in rec.inputs:
                    value = self.ref_price[g]
                    if self.cap_mode == 1 and inv.get(g, 0) > 0:
                        # a unit that would sit in stock for many turns is worth less
                        turnover = self.sales_rate.get(g, 0.0) * self.cover_turns / inv[g]
                        value *= min(1.0, max(self.turnover_floor, turnover))
                    cap = max(cap, self.value_share * value / sum(rec.inputs.values()))
            level = self.bid_level.get(m, lot.reserve_price * self.bid_start)
            if turn <= self.open_turns:
                level = max(level, lot.reserve_price * self.open_bid)
            if uncontested.get(m):
                level = min(level, lot.reserve_price * self.uncontested_mult)
            price = max(lot.reserve_price * 1.001, level)
            price = min(price, lot.reserve_price * max(self.max_bid_mult, self.cap_mult))
            if cap > 0:
                price = min(price, max(cap, lot.reserve_price * 1.001))
            qty = min(qty, int(budget // price))
            if qty <= 0:
                continue
            budget -= qty * price
            bids.append(Bid(m, int(qty), float(price)))
            self.last_bids[m] = (int(qty), float(price))

        # --- listings: one unit per listing on a ladder ------------------------
        listings = []
        self.last_ladder = {}
        low, high = (self.early_low, self.early_high) if early else (self.ladder_low, self.ladder_high)
        for g in gc.finished_goods:
            planned = inv.get(g, 0) + need.get(g, 0)
            if planned <= 0:
                continue
            ref = self.ref_price[g]
            if self.comp_aware and known_me and turn > 1 and rival_sellers[g] == 0:
                ref *= self.mono_boost              # nobody else can offer it this turn
            floor = good_cost.get(g, 0.0) * self.min_markup
            n = planned
            prices = []
            for i in range(n):
                frac = 0.5 if n == 1 else i / (n - 1)
                p = ref * (low + (high - low) * frac)
                prices.append(round(max(p, floor), 2))
            # cheapest first: if production falls short, the expensive rungs drop
            ladder = sorted(prices)
            for p in ladder:
                listings.append(Listing(g, 1, float(p)))
            self.last_ladder[g] = ladder
        return Decision(bids=bids, assembly=assembly, listings=listings)