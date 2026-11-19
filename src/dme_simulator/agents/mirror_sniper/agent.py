from dme_simulator.agents import BaseAgent
from dme_simulator.agents.competitive_rivals import CompetitiveRival


class MirrorSniper(BaseAgent):
    def __init__(self):
        self.agent = CompetitiveRival(
            init_markup=8.0,
            up=1.08,
            down=0.90,
            floor=2.5,
            stock=8,
            stock_cover=1.0,
            bid_premium=0.45,
            budget_frac=0.90,
            ladder=(1.0, 0.86, 1.25, 0.72, 1.5, 1.8),
            demand_weight=0.35,
        )

    def decide(self, state, goods_config):
        return self.agent.decide(state, goods_config)
