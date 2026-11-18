"""Every selectable agent by name: our versions V1-V7 and the opponent archetypes."""
from __future__ import annotations

from typing import Dict, Optional

from ..core.config import StrategyConfig, load_profile
from .base import BaseAgent
from .heuristics import RuleBaselineAgent
from .strategic import StrategicAgent


# Overrides on top of the "balanced" profile. V2 -> V3 -> V6 add one layer each, so the
# tournament shows what each layer is worth.
STRATEGIC: Dict[str, dict] = {
    # V2: LP planning on a demand curve fitted by plain least squares; no forgetting, no
    #     exploration, no rival model, fixed bid shading
    "v2_opt": dict(ts_scale=0.0, forgetting=1.0, dither=0.0, use_opponent_model=False,
                   comp_band=False, shade_mode="fixed"),
    # V3: V2 + Bayesian learning (forgetting, Thompson sampling, early price variation)
    "v3_bayes": dict(use_opponent_model=False, comp_band=False, shade_mode="fixed"),
    # V4: contextual bandit over a price grid in place of the demand model
    "v4_bandit": dict(pricing_mode="bandit"),
    # V6: the full hybrid (V3 + rival model + learned bid shading + competitor band)
    "v6_hybrid": dict(),
    # V7: V6 with a slow bandit choosing between the model and the follower rule
    "v7_meta": dict(pricing_mode="meta"),
}


def _qlearn() -> BaseAgent:
    from ..simulation.opponents import QLearningAgent
    ag = QLearningAgent()
    ag.name = "v5_rl"
    return ag


VERSION_NAMES = ["v1_rule", "v2_opt", "v3_bayes", "v4_bandit", "v5_rl", "v6_hybrid", "v7_meta"]


def available() -> list:
    from ..simulation.opponents import OPPONENTS
    return VERSION_NAMES + list(OPPONENTS)


def make_agent(name: str, label: Optional[str] = None, profile: str = "balanced",
               **overrides) -> BaseAgent:
    """`make_agent("v6_hybrid")`, `make_agent("opp_follower")`,
    `make_agent("v6_hybrid", ts_scale=0.0)` or a strategy profile such as
    `make_agent("aggressive")` (a name in config/profiles or a YAML path)."""
    from ..simulation.opponents import OPPONENTS
    if name in STRATEGIC:
        cfg = load_profile(profile, {**STRATEGIC[name], **overrides})
        return StrategicAgent(cfg, name=label or name)
    if name == "v1_rule":
        ag: BaseAgent = RuleBaselineAgent()
    elif name == "v5_rl":
        ag = _qlearn()
    elif name in OPPONENTS:
        ag = OPPONENTS[name]()
    else:
        return StrategicAgent(load_profile(name, overrides), name=label or name)
    if label:
        ag.name = label
    return ag
