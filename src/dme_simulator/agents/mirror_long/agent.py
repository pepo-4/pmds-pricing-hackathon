from dme_simulator.agents import BaseAgent
from dme_simulator.agents.competitive_rivals import CompetitiveRival


class MirrorLong(BaseAgent):
    def __init__(self):
        self.agent = CompetitiveRival(
            init_markup=7.0,
            up=1.06,
            down=0.94,
            floor=2.5,
            stock=16,
            stock_cover=2.5,
            bid_premium=0.10,
            budget_frac=0.60,
            ladder=(1.0, 0.9, 1.12, 0.82, 1.25, 1.42),
            demand_weight=0.25,
        )

    def decide(self, state, goods_config):
        return self.agent.decide(state, goods_config)
