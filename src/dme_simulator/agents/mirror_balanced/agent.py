from dme_simulator.agents import BaseAgent
from dme_simulator.agents.competitive_rivals import CompetitiveRival


class MirrorBalanced(BaseAgent):
    def __init__(self):
        self.agent = CompetitiveRival(
            init_markup=6.0,
            up=1.04,
            down=0.92,
            floor=2.2,
            stock=14,
            stock_cover=2.0,
            bid_premium=0.12,
            budget_frac=0.65,
        )

    def decide(self, state, goods_config):
        return self.agent.decide(state, goods_config)
