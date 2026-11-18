"""Adapter interface: the only code that knows the competition's data format."""
from __future__ import annotations

from typing import Any

from ..core.schemas import Action, GameSpec, Observation


class BaseAdapter:
    def to_spec(self, raw: Any) -> GameSpec:
        """Rules of the game (products, recipes, auction format, horizon, scoring)."""
        return GameSpec.from_dict(raw)

    def to_observation(self, raw: Any) -> Observation:
        """What the competition sends each round -> canonical Observation."""
        return Observation.from_dict(raw)

    def from_action(self, action: Action, raw_obs: Any = None) -> Any:
        """Canonical Action -> what the competition expects back."""
        return action.to_dict()
