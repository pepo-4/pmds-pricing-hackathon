from dme_simulator.agents import BaseAgent
from dme_simulator.agents.competitive_rivals import CompetitiveRival


class MirrorConservative(BaseAgent):
    def __init__(self):
        self.agent = CompetitiveRival(
            init_markup=7.0,
            up=1.03,
            down=0.96,
            floor=2.8,
            stock=10,
            stock_cover=1.2,
            bid_premium=0.05,
            budget_frac=0.35,
            ladder=(1.0, 0.95, 1.08, 0.88, 1.18, 1.3),
        )

    def decide(self, state, goods_config):
        return self.agent.decide(state, goods_config)
