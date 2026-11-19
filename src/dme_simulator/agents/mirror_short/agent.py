from dme_simulator.agents import BaseAgent
from dme_simulator.agents.competitive_rivals import CompetitiveRival


class MirrorShort(BaseAgent):
    def __init__(self):
        self.agent = CompetitiveRival(
            init_markup=5.5,
            up=1.05,
            down=0.88,
            floor=1.8,
            stock=6,
            stock_cover=1.0,
            bid_premium=0.18,
            budget_frac=0.75,
            ladder=(1.0, 0.82, 1.15, 0.72, 1.28, 1.4),
        )

    def decide(self, state, goods_config):
        return self.agent.decide(state, goods_config)
