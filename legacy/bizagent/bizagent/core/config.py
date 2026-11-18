"""Strategy configuration. Every behavioural knob lives here so that strategy
changes on the day are YAML edits, not code edits.

Twelve defaults (band_ref, band_over, min_elasticity_frac, prior_elasticity, ts_scale,
forgetting, coverage_rounds, fin_cover_rounds, win_bias, safety_z, max_price_step_up/down)
come from experiments/tune.py: random search on the train split, adopted after beating the
previous defaults on the validation split (experiments/results/tuning.md)."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional

try:  # yaml is optional: profiles can also be passed as dicts
    import yaml
except Exception:  # pragma: no cover
    yaml = None

_HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.normpath(os.path.join(_HERE, "..", "..", "config"))


@dataclass
class StrategyConfig:
    name: str = "balanced"

    # ---- objective shaping (set automatically from GameSpec.objective, can be overridden)
    w_revenue: Optional[float] = None    # extra objective weight on revenue
    w_units: Optional[float] = None      # value of one unit sold, as a fraction of its price

    # ---- demand model
    prior_elasticity: float = 2.85        # prior price elasticity at the first observed price
    cross_frac: float = 0.5              # prior cross effect as a fraction of the own effect
    min_elasticity_frac: float = 0.345    # the fitted elasticity is never below this share of the prior
    forgetting: float = 0.943             # 1.0 = never forget
    ts_scale: float = 0.03                # Thompson-sampling scale; 0 = use the posterior mean
    comp_feature: str = "blend"          # min | mean | blend

    # ---- pricing
    pricing_mode: str = "model"          # model | follower | bandit | meta | fixed
    price_grid: int = 25
    max_price_step_up: float = 0.256      # trust region around our last price
    max_price_step_down: float = 0.256
    early_step: float = 0.35             # wider trust region in the first rounds
    early_rounds: int = 3
    init_price_frac: float = 0.30        # first price = floor + frac * (price_max - floor)
    init_markup: float = 1.0             # used when price_max is unknown
    dither: float = 0.03                 # small price perturbation when prices have not varied
    explore_frac: float = 0.15           # share of the game in which dithering is allowed
    comp_band: bool = True               # keep price inside a band around competitor prices
    band_under: float = 0.25             # never more than this far below the cheapest rival
    band_over: float = 0.317              # never more than this far above the band reference ...
    band_ref: str = "mean"                # ... which is the mean or the max rival price
    band_lift_when_short: bool = True    # no upper band while we would sell out at its top
    undercut: float = 0.02               # follower mode: undercut the cheapest rival by this much
    min_margin: float = 0.02             # hard floor: price >= marginal cost * (1 + min_margin)
    use_opponent_model: bool = True      # conjectural variation from observed rival reactions
    cv_max: float = 0.8
    safety_z: float = 0.85                # safety stock in demand standard deviations

    # ---- procurement
    shade_mode: str = "winprob"          # winprob | truthful | fixed
    fixed_shade: float = 0.8
    win_bias: float = 1.57                # >1 favours winning over surplus when shading
    value_haircut: float = 0.0           # risk aversion: scale resource values down
    prior_clearing_frac: float = 0.5     # prior clearing price as a fraction of top value
    clearing_alpha: float = 0.3          # EW update rate for clearing price statistics
    n_tranches: int = 5
    coverage_rounds: float = 1.42         # rounds of input to hold after a successful auction
    fin_cover_rounds: float = 0.28        # rounds of sales we may carry as finished stock at full value
    stockahead_discount: float = 0.85    # bid for extra cover only below this * typical price
    demand_reduction: float = 0.10       # uniform-price: shade later tranches

    # ---- risk
    cash_reserve_frac: float = 0.05      # share of initial cash never committed
    max_spend_frac: float = 0.70         # max share of free cash committed to bids in one round
    liquidate_last_rounds: int = 1       # rounds at the end where the margin floor is lifted

    # ---- horizon
    assumed_remaining: int = 12          # used when the horizon is unknown

    # ---- meta-strategy
    meta_modes: List[str] = field(default_factory=lambda: ["model", "follower"])
    meta_block: int = 4
    meta_discount: float = 0.9

    # ---- module switches (used by the ablation study; leave on in competition)
    use_comp_prices: bool = True         # False: ignore rival prices entirely
    inventory_aware: bool = True         # False: stock left after this round is worth only salvage
    coordinated_bids: bool = True        # False: bid a fixed share of the unit margin, no LP valuation
    simple_bid_frac: float = 0.7

    # ---- execution
    solver: str = "lp"                   # lp | greedy
    integer_quantities: bool = False
    time_budget_ms: float = 150.0        # soft budget per decision; bid curves get coarser beyond it

    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "StrategyConfig":
        d = dict(d or {})
        valid = {f.name for f in fields(cls)}
        clean = {k: v for k, v in d.items() if k in valid}
        return cls(**clean)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def replace(self, **kw) -> "StrategyConfig":
        d = self.to_dict()
        d.update(kw)
        return StrategyConfig.from_dict(d)


def load_yaml(path: str) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is not installed; pass configuration as a dict instead")
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return data


def load_profile(name_or_path: str = "balanced",
                 overrides: Optional[Dict[str, Any]] = None) -> StrategyConfig:
    """Load config/default.yaml, then a profile on top, then explicit overrides."""
    base: Dict[str, Any] = {}
    default_path = os.path.join(CONFIG_DIR, "default.yaml")
    if yaml is not None and os.path.exists(default_path):
        base.update(load_yaml(default_path))
    path = name_or_path
    if not os.path.exists(path):
        path = os.path.join(CONFIG_DIR, "profiles", f"{name_or_path}.yaml")
    if yaml is not None and os.path.exists(path):
        base.update(load_yaml(path))
        base.setdefault("name", os.path.splitext(os.path.basename(path))[0])
    elif name_or_path not in ("balanced", "default"):
        raise FileNotFoundError(f"strategy profile not found: {name_or_path}")
    base.update(overrides or {})
    return StrategyConfig.from_dict(base)
