"""The strategic agent: the decision engine behind the common agent interface,
with a rule-based fallback so that a valid action always comes back."""
from __future__ import annotations

import time
from typing import Optional

from ..core.config import StrategyConfig
from ..core.decision_engine import DecisionEngine
from ..core.schemas import Action, GameSpec, Observation
from .base import BaseAgent
from .heuristics import RuleBaselineAgent


class StrategicAgent(BaseAgent):
    def __init__(self, cfg: Optional[StrategyConfig] = None, name: Optional[str] = None):
        self.cfg = cfg or StrategyConfig()
        self.name = name or self.cfg.name
        self.engine: Optional[DecisionEngine] = None
        self.fallback = RuleBaselineAgent()
        self.errors = 0
        self.last_error: Optional[str] = None
        self.decisions = 0
        self.total_ms = 0.0
        self.max_ms = 0.0

    def reset(self, spec: GameSpec, agent_id: str = "me", seed: int = 0) -> None:
        self.spec, self.agent_id = spec, agent_id
        self.engine = DecisionEngine(spec, self.cfg, seed=seed)
        self.fallback.reset(spec, agent_id, seed)
        self.errors = 0
        self.last_error = None
        self.decisions = 0
        self.total_ms = 0.0
        self.max_ms = 0.0

    def act(self, obs: Observation) -> Action:
        t0 = time.perf_counter()
        # the fallback keeps learning every round so it is warm if it is ever needed
        try:
            fb = self.fallback.act(obs)
        except Exception:
            fb = Action()
        try:
            action = self.engine.decide(obs)
        except Exception as exc:  # never let a bug cost a round
            self.errors += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            action = fb
            action.info["fallback"] = self.last_error
        ms = (time.perf_counter() - t0) * 1000.0
        self.decisions += 1
        self.total_ms += ms
        self.max_ms = max(self.max_ms, ms)
        return action
