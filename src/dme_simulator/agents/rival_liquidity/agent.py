from dme_simulator.agents import BaseAgent
from dme_simulator.agents.diverse_rivals import DiverseRival

class RivalLiquidity(BaseAgent):
    def __init__(self):
        self.agent = DiverseRival("liquidity")
    def decide(self, state, goods_config):
        return self.agent.decide(state, goods_config)
