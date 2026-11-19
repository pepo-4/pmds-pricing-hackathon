from dme_simulator.agents import BaseAgent
from dme_simulator.agents.competitive_rivals import CompetitiveRival


class MirrorAggressive(BaseAgent):
    def __init__(self):
        self.agent = CompetitiveRival(
            init_markup=4.5,
            up=1.07,
            down=0.84,
            floor=1.6,
            stock=18,
            stock_cover=3.0,
            bid_premium=0.28,
            budget_frac=0.85,
            ladder=(1.0, 0.78, 1.18, 0.65, 1.35, 1.55),
            demand_weight=0.45,
        )

    def decide(self, state, goods_config):
        return self.agent.decide(state, goods_config)
