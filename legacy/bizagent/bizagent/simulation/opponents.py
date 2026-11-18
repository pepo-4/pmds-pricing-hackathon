"""Ten opponent archetypes for the internal tournament.

Each one is a distinct, plausible way a hackathon team might play. None is
deliberately handicapped: they all hold a cost floor, stop buying inputs that
would arrive after the last round, and respect their cash.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..agents.heuristics import Heuristic


class RandomAgent(Heuristic):
    """Random prices around the market, random bid aggressiveness."""
    name = "opp_random"

    def price_for(self, obs, p):
        return self.ref_price(obs, p) * self.rng.uniform(0.6, 1.4)

    def bid_fraction(self, obs, r):
        return self.rng.uniform(0.2, 0.9)

    def target(self, obs, p):
        return self.exp_sales(p) * self.rng.uniform(0.5, 1.6)


class FixedPriceAgent(Heuristic):
    """Picks a price once the market is visible and never moves it."""
    name = "opp_fixed"

    def setup(self):
        self.mult = self.rng.uniform(0.9, 1.2)
        self.anchor = {}
        self.anchored_on_market = set()

    def price_for(self, obs, p):
        cp = self.comp_prices(obs, p)
        if p not in self.anchor:
            self.anchor[p] = self.ref_price(obs, p)
        if cp and p not in self.anchored_on_market:
            self.anchor[p] = float(np.mean(cp))
            self.anchored_on_market.add(p)
        return max(self.anchor[p] * self.mult, self.spec.products[p].unit_cost * 1.05)


class GreedyProfitAgent(Heuristic):
    """Myopic profit maximiser: fits its own linear demand curve by least squares
    and charges this round's profit-maximising price. No view of rivals or of the future."""
    name = "opp_greedy"
    bid_frac = 0.85
    cover = 1.1

    def price_for(self, obs, p):
        cost = self.unit_cost(p)
        base = self.price.get(p) or self.ref_price(obs, p)
        h = [(pr, s) for pr, s, so in self.hist[p][-12:] if not so]
        if len(h) >= 4 and np.ptp([x[0] for x in h]) > 0.03 * base:
            X = np.array([[1.0, x[0]] for x in h])
            y = np.array([x[1] for x in h])
            (a, b), *_ = np.linalg.lstsq(X, y, rcond=None)
            if b < -1e-9 and a > 0:
                popt = 0.5 * (a / -b + cost)
                return max(cost * 1.05, min(max(popt, 0.75 * base), 1.25 * base))
        sign = 1.0 if obs.round % 2 == 0 else -1.0
        return max(cost * 1.05, base * (1.0 + 0.06 * sign))


class AggressiveBidderAgent(Heuristic):
    """Wants the inputs: bids close to the full margin for a lot of volume, then
    marks finished goods down until the stock moves."""
    name = "opp_aggr_bid"
    bid_frac = 0.95
    cover = 1.8

    def bid_fraction(self, obs, r):
        v = self.input_value(obs, r)
        if v > 0 and r in self.raw_px:
            return float(min(1.0, max(self.bid_frac, 1.1 * self.raw_px[r] / v)))
        return self.bid_frac

    def price_for(self, obs, p):
        base = self.ref_price(obs, p)
        over = max(0.0, obs.fin_inventory.get(p, 0.0) / max(self.exp_sales(p), 1e-9) - 1.0)
        return max(self.spec.products[p].unit_cost * 1.05, base * (1.0 - 0.08 * min(over, 4.0)))

    def prod_target(self, obs, p):
        return 1.5 * self.exp_sales(p)


class InventoryFirstAgent(Heuristic):
    """Builds a deep stock of inputs at moderate bids and sells at a premium."""
    name = "opp_inventory"
    bid_frac = 0.6
    cover = 3.0

    def price_for(self, obs, p):
        return max(self.unit_cost(p) * 1.25, self.ref_price(obs, p) * 1.03)

    def prod_target(self, obs, p):
        return 1.5 * self.exp_sales(p)


class CostPlusAgent(Heuristic):
    """Rule-based: cost plus a mark-up that rises after a stock-out and falls when stock piles up."""
    name = "opp_costplus"
    bid_frac = 0.65

    def setup(self):
        self.mk = defaultdict(lambda: 0.35)

    def on_sale(self, obs, product, price, sales, sold_out):
        if sold_out:
            self.mk[product] = min(2.0, self.mk[product] + 0.06)
        elif obs.fin_inventory.get(product, 0.0) > sales:
            self.mk[product] = max(0.05, self.mk[product] - 0.05)

    def price_for(self, obs, p):
        cost = self.unit_cost(p)
        if cost <= 0 or not self.raw_px:
            return self.ref_price(obs, p)
        return cost * (1.0 + self.mk[p])


class HillClimbAgent(Heuristic):
    """Adaptive pricing: keeps moving the price in the direction that raised profit last time."""
    name = "opp_hillclimb"

    def setup(self):
        self.dir = defaultdict(lambda: 1.0)
        self.last_profit = {}

    def on_sale(self, obs, product, price, sales, sold_out):
        prof = (price - self.unit_cost(product)) * sales
        if product in self.last_profit and prof < self.last_profit[product]:
            self.dir[product] *= -1.0
        self.last_profit[product] = prof

    def price_for(self, obs, p):
        base = self.price.get(p) or self.ref_price(obs, p)
        return max(self.unit_cost(p) * 1.02, base * (1.0 + 0.05 * self.dir[p]))


class ThompsonPricingAgent(Heuristic):
    """Bayesian pricing: Thompson sampling over a fixed grid of prices."""
    name = "opp_ts"

    def setup(self):
        self.mults = np.linspace(0.7, 1.5, 9)
        self.cnt = {p: np.zeros(9) for p in self.P}
        self.tot = {p: np.zeros(9) for p in self.P}
        self.arm = {}
        self.anchor = {}
        self.on_market = set()
        self.scale = 1.0

    def on_sale(self, obs, product, price, sales, sold_out):
        if product not in self.arm:
            return
        r = (price - self.unit_cost(product)) * sales
        self.scale = max(1e-9, 0.9 * self.scale + 0.1 * abs(r))
        a = self.arm[product]
        self.cnt[product] *= 0.97
        self.tot[product] *= 0.97
        self.cnt[product][a] += 1.0
        self.tot[product][a] += r / self.scale

    def price_for(self, obs, p):
        cp = self.comp_prices(obs, p)
        if p not in self.anchor:
            self.anchor[p] = self.ref_price(obs, p)
        if cp and p not in self.on_market:
            self.anchor[p] = float(np.mean(cp))
            self.on_market.add(p)
        mean = self.tot[p] / np.maximum(self.cnt[p], 1e-9)
        sd = 1.0 / np.sqrt(self.cnt[p] + 0.3)
        a = int(np.argmax(self.rng.normal(np.where(self.cnt[p] > 0, mean, 1.0), sd)))
        self.arm[p] = a
        return max(self.unit_cost(p) * 1.02, self.anchor[p] * self.mults[a])


class FollowerAgent(Heuristic):
    """Competitor-aware: undercut the cheapest rival, never below cost. This is the
    'greedy' rule that proved hard to beat in the 2017 Dynamic Pricing Challenge."""
    name = "opp_follower"
    bid_frac = 0.8

    def price_for(self, obs, p):
        cost = self.unit_cost(p)
        cp = self.comp_prices(obs, p)
        if cp:
            return max(cost * 1.05, min(cp) * 0.98)
        return max(cost * 1.3, self.blind_price(p)) if not self.hist[p] else max(cost * 1.3, self.price.get(p, 0.0))


class QLearningAgent(Heuristic):
    """Experimental learning-based agent: tabular Q-learning over relative price moves."""
    name = "opp_qlearn"
    ACTIONS = (0.90, 0.95, 1.0, 1.05, 1.10)

    def setup(self):
        self.Q = defaultdict(lambda: np.zeros(len(self.ACTIONS)))
        self.eps = 0.3
        self.pending = {}
        self.reward = {}
        self.scale = 1.0

    def _state(self, obs, p):
        cp = self.comp_prices(obs, p)
        own = self.price.get(p)
        rel = 1
        if cp and own:
            x = own / float(np.mean(cp))
            rel = 0 if x < 0.97 else (2 if x > 1.03 else 1)
        cov = obs.fin_inventory.get(p, 0.0) / max(self.exp_sales(p), 1e-9)
        return rel, (0 if cov < 0.5 else (2 if cov > 1.5 else 1))

    def on_sale(self, obs, product, price, sales, sold_out):
        r = (price - self.unit_cost(product)) * sales
        self.scale = max(1e-9, 0.9 * self.scale + 0.1 * abs(r))
        self.reward[product] = r / self.scale

    def price_for(self, obs, p):
        s = self._state(obs, p)
        if p in self.pending and p in self.reward:
            s0, a0 = self.pending[p]
            self.Q[s0][a0] += 0.3 * (self.reward[p] + 0.8 * self.Q[s].max() - self.Q[s0][a0])
        self.eps = max(0.05, self.eps * 0.97)
        a = int(self.rng.integers(len(self.ACTIONS))) if self.rng.random() < self.eps else int(np.argmax(self.Q[s]))
        self.pending[p] = (s, a)
        cp = self.comp_prices(obs, p)
        base = float(np.mean(cp)) if cp else (self.price.get(p) or self.ref_price(obs, p))
        return max(self.unit_cost(p) * 1.02, base * self.ACTIONS[a])


OPPONENTS = {
    "opp_random": RandomAgent,
    "opp_fixed": FixedPriceAgent,
    "opp_greedy": GreedyProfitAgent,
    "opp_aggr_bid": AggressiveBidderAgent,
    "opp_inventory": InventoryFirstAgent,
    "opp_costplus": CostPlusAgent,
    "opp_hillclimb": HillClimbAgent,
    "opp_ts": ThompsonPricingAgent,
    "opp_follower": FollowerAgent,
    "opp_qlearn": QLearningAgent,
}
# opponents the tuner is allowed to see; the other three appear only in validation/test
TRAIN_POOL = ["opp_random", "opp_fixed", "opp_greedy", "opp_aggr_bid", "opp_costplus",
              "opp_hillclimb", "opp_follower"]
HELDOUT_POOL = ["opp_inventory", "opp_ts", "opp_qlearn"]
