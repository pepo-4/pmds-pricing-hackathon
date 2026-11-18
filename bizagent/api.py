"""The universal entry point.

    from bizagent.api import CompetitionAgent
    agent = CompetitionAgent(spec=rules_dict, profile="balanced")
    answer = agent.act(their_observation)          # their format in, their format out

or, for function-style interfaces that call you once per round and hand your
state back (the Dynamic Pricing Competition pattern):

    from bizagent.api import decide
    answer, memory = decide(their_observation, memory, spec=rules_dict)
"""
from __future__ import annotations

import pickle
from typing import Any, Optional, Tuple

from .adapters.base import BaseAdapter
from .adapters.competition_adapter import CompetitionAdapter
from .agents.registry import make_agent
from .core.schemas import Action, GameSpec


class CompetitionAgent:
    def __init__(self, spec: Any = None, profile: str = "balanced", agent: str = "v6_hybrid",
                 adapter: Optional[BaseAdapter] = None, seed: int = 0, agent_id: str = "me",
                 **overrides):
        self.adapter = adapter or CompetitionAdapter()
        self.spec: GameSpec = self.adapter.to_spec(spec if spec is not None else {})
        self.agent = make_agent(agent, profile=profile, **overrides)
        self.agent.reset(self.spec, agent_id, seed=seed)
        self.last_action: Optional[Action] = None

    def act_canonical(self, raw_obs: Any) -> Action:
        obs = self.adapter.to_observation(raw_obs)
        try:
            action = self.agent.act(obs)
        except Exception as exc:        # last resort: an empty but valid action
            action = Action(info={"error": f"{type(exc).__name__}: {exc}"})
        self.last_action = action
        return action

    def act(self, raw_obs: Any) -> Any:
        return self.adapter.from_action(self.act_canonical(raw_obs), raw_obs)

    # persistence, for interfaces that restart the process between rounds
    def save(self, path: str) -> None:
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: str) -> "CompetitionAgent":
        with open(path, "rb") as fh:
            return pickle.load(fh)


def decide(raw_obs: Any, memory: Optional[CompetitionAgent] = None, spec: Any = None,
           **kw) -> Tuple[Any, CompetitionAgent]:
    agent = memory if isinstance(memory, CompetitionAgent) else CompetitionAgent(spec=spec, **kw)
    return agent.act(raw_obs), agent
