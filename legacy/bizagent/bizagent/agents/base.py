"""Agent interface shared by our agent, its versions and every simulated opponent."""
from __future__ import annotations

from ..core.schemas import Action, GameSpec, Observation


class BaseAgent:
    name = "agent"

    def reset(self, spec: GameSpec, agent_id: str = "me", seed: int = 0) -> None:
        """Called once before the first round with the public rules of the game."""
        self.spec = spec
        self.agent_id = agent_id

    def act(self, obs: Observation) -> Action:  # pragma: no cover - interface
        raise NotImplementedError
