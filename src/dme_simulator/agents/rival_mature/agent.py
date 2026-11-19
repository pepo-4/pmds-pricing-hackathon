from dme_simulator.agents import BaseAgent
from dme_simulator.agents.diverse_rivals import DiverseRival

class RivalMature(BaseAgent):
    def __init__(self):
        self.agent = DiverseRival("mature")
    def decide(self, state, goods_config):
        return self.agent.decide(state, goods_config)
