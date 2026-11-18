"""Canonical internal data structures.

The strategic core only ever sees these types. Adapters translate whatever the
competition provides into them, and translate `Action` back out.

Parsing is deliberately lenient (missing or malformed optional fields become
safe defaults) because a crash during the competition costs more than a
slightly wrong default. `GameSpec.validate()` is the strict check to run once,
at integration time, when a human can read the error.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

AUCTION_FORMATS = ("first_price", "uniform", "second_price", "posted")
PHASES = ("all", "bid", "sell")


class ValidationError(ValueError):
    """Raised by strict validation only."""


# --------------------------------------------------------------------------
# lenient coercion helpers
# --------------------------------------------------------------------------
def fnum(x: Any, default: float = 0.0, lo: Optional[float] = None,
         hi: Optional[float] = None) -> float:
    """Coerce to a finite float, falling back to `default`, then clamp."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    if math.isnan(v) or math.isinf(v):
        return default
    if lo is not None and v < lo:
        v = lo
    if hi is not None and v > hi:
        v = hi
    return v


def fopt(x: Any, lo: Optional[float] = None, hi: Optional[float] = None) -> Optional[float]:
    """Like fnum but returns None when the value is missing or unusable."""
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    if lo is not None and v < lo:
        v = lo
    if hi is not None and v > hi:
        v = hi
    return v


def numdict(d: Any, lo: Optional[float] = 0.0) -> Dict[str, float]:
    if not isinstance(d, dict):
        return {}
    out: Dict[str, float] = {}
    for k, v in d.items():
        val = fopt(v, lo=lo)
        if val is not None:
            out[str(k)] = val
    return out


def _as_items(x: Any, key: str = "name") -> List[dict]:
    """Accept {'A': {...}} or [{'name': 'A', ...}] and return a list of dicts."""
    if isinstance(x, dict):
        out = []
        for k, v in x.items():
            item = dict(v) if isinstance(v, dict) else {}
            item.setdefault(key, str(k))
            out.append(item)
        return out
    if isinstance(x, (list, tuple)):
        return [dict(i) for i in x if isinstance(i, dict)]
    return []


# --------------------------------------------------------------------------
# static game description
# --------------------------------------------------------------------------
@dataclass
class ProductSpec:
    name: str
    recipe: Dict[str, float] = field(default_factory=dict)  # resource -> units per unit of product
    unit_cost: float = 0.0          # production cost per unit, excluding raw materials
    capacity_use: float = 1.0       # capacity units consumed per unit produced
    price_min: float = 0.0
    price_max: Optional[float] = None
    salvage: float = 0.0            # value of one unsold unit at the end of the game
    holding_cost: float = 0.0       # per unit per round

    @classmethod
    def from_dict(cls, d: dict) -> "ProductSpec":
        return cls(
            name=str(d.get("name", "product")),
            recipe=numdict(d.get("recipe", {})),
            unit_cost=fnum(d.get("unit_cost", 0.0), 0.0, lo=0.0),
            capacity_use=fnum(d.get("capacity_use", 1.0), 1.0, lo=0.0),
            price_min=fnum(d.get("price_min", 0.0), 0.0, lo=0.0),
            price_max=fopt(d.get("price_max"), lo=0.0),
            salvage=fnum(d.get("salvage", 0.0), 0.0, lo=0.0),
            holding_cost=fnum(d.get("holding_cost", 0.0), 0.0, lo=0.0),
        )


@dataclass
class ResourceSpec:
    name: str
    auction_format: str = "first_price"
    reserve_price: float = 0.0
    ref_price: Optional[float] = None   # a hint of the typical price, if the organisers give one
    salvage: float = 0.0
    holding_cost: float = 0.0

    @classmethod
    def from_dict(cls, d: dict) -> "ResourceSpec":
        fmt = str(d.get("auction_format", d.get("format", "first_price"))).lower()
        if fmt not in AUCTION_FORMATS:
            fmt = "first_price"
        return cls(
            name=str(d.get("name", "resource")),
            auction_format=fmt,
            reserve_price=fnum(d.get("reserve_price", 0.0), 0.0, lo=0.0),
            ref_price=fopt(d.get("ref_price"), lo=0.0),
            salvage=fnum(d.get("salvage", 0.0), 0.0, lo=0.0),
            holding_cost=fnum(d.get("holding_cost", 0.0), 0.0, lo=0.0),
        )


@dataclass
class ObjectiveSpec:
    """How the competition scores a firm.

    score = w_profit*profit + w_revenue*revenue + w_units*units_sold
            (+ w_cash*final_cash), where profit already includes inventory at
    salvage value when `inventory_valued` is true.
    """
    kind: str = "profit"
    w_profit: float = 1.0
    w_revenue: float = 0.0
    w_units: float = 0.0
    w_cash: float = 0.0
    inventory_valued: bool = True

    @classmethod
    def from_dict(cls, d: Any) -> "ObjectiveSpec":
        if isinstance(d, str):
            d = {"kind": d}
        if not isinstance(d, dict):
            return cls()
        kind = str(d.get("kind", "profit")).lower()
        presets = {
            "profit": dict(w_profit=1.0),
            "cash": dict(w_profit=0.0, w_cash=1.0, inventory_valued=False),
            "valuation": dict(w_profit=1.0, inventory_valued=True),
            "revenue": dict(w_profit=0.0, w_revenue=1.0),
            "market_share": dict(w_profit=0.0, w_units=1.0),
        }
        base = dict(kind=kind, w_profit=1.0, w_revenue=0.0, w_units=0.0, w_cash=0.0,
                    inventory_valued=True)
        base.update(presets.get(kind, {}))
        for k in ("w_profit", "w_revenue", "w_units", "w_cash"):
            if k in d:
                base[k] = fnum(d[k], base[k])
        if "inventory_valued" in d:
            base["inventory_valued"] = bool(d["inventory_valued"])
        return cls(**base)


@dataclass
class GameSpec:
    products: Dict[str, ProductSpec] = field(default_factory=dict)
    resources: Dict[str, ResourceSpec] = field(default_factory=dict)
    horizon: Optional[int] = None          # number of rounds; None = unknown
    n_firms: Optional[int] = None
    capacity: Optional[float] = None       # capacity units per round; None = unlimited
    lead_time: int = 1                     # rounds between winning raw material and being able to use it
    initial_cash: Optional[float] = None
    fixed_cost: float = 0.0
    demand_hint: Dict[str, float] = field(default_factory=dict)  # product -> rough units per round, if known
    objective: ObjectiveSpec = field(default_factory=ObjectiveSpec)

    @classmethod
    def from_dict(cls, d: Any) -> "GameSpec":
        if not isinstance(d, dict):
            d = {}
        products = {p["name"]: ProductSpec.from_dict(p) for p in _as_items(d.get("products"))}
        resources = {r["name"]: ResourceSpec.from_dict(r) for r in _as_items(d.get("resources"))}
        # lenient: a recipe may mention a resource that was not declared
        for p in products.values():
            for r in p.recipe:
                resources.setdefault(r, ResourceSpec(name=r))
        horizon = fopt(d.get("horizon"), lo=1)
        n_firms = fopt(d.get("n_firms"), lo=1)
        return cls(
            products=products,
            resources=resources,
            horizon=int(horizon) if horizon is not None else None,
            n_firms=int(n_firms) if n_firms is not None else None,
            capacity=fopt(d.get("capacity"), lo=0.0),
            lead_time=int(fnum(d.get("lead_time", 1), 1, lo=0, hi=50)),
            initial_cash=fopt(d.get("initial_cash")),
            fixed_cost=fnum(d.get("fixed_cost", 0.0), 0.0, lo=0.0),
            demand_hint=numdict(d.get("demand_hint", {})),
            objective=ObjectiveSpec.from_dict(d.get("objective", "profit")),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        """Strict check. Run once at integration time, not in the decision loop."""
        if not self.products:
            raise ValidationError("GameSpec has no products")
        for p in self.products.values():
            if p.price_max is not None and p.price_max < p.price_min:
                raise ValidationError(f"{p.name}: price_max < price_min")
            for r, q in p.recipe.items():
                if r not in self.resources:
                    raise ValidationError(f"{p.name}: recipe uses undeclared resource {r}")
                if q < 0:
                    raise ValidationError(f"{p.name}: negative recipe quantity for {r}")
        for r in self.resources.values():
            if r.auction_format not in AUCTION_FORMATS:
                raise ValidationError(f"{r.name}: unknown auction format {r.auction_format}")
        if self.lead_time < 0:
            raise ValidationError("lead_time must be >= 0")


# --------------------------------------------------------------------------
# per-round data
# --------------------------------------------------------------------------
@dataclass
class AuctionLot:
    resource: str
    quantity: float
    reserve_price: float = 0.0
    format: Optional[str] = None        # overrides ResourceSpec.auction_format when given

    @classmethod
    def from_dict(cls, d: dict) -> "AuctionLot":
        fmt = d.get("format", d.get("auction_format"))
        fmt = str(fmt).lower() if fmt is not None else None
        if fmt not in AUCTION_FORMATS:
            fmt = None
        return cls(resource=str(d.get("resource", "")),
                   quantity=fnum(d.get("quantity", 0.0), 0.0, lo=0.0),
                   reserve_price=fnum(d.get("reserve_price", 0.0), 0.0, lo=0.0),
                   format=fmt)


@dataclass
class AuctionResult:
    resource: str
    clearing_price: Optional[float] = None   # lowest accepted price (pay-as-bid) or uniform price
    won_qty: float = 0.0
    paid: float = 0.0
    total_sold: Optional[float] = None
    supply: Optional[float] = None
    all_bids: Optional[List[List[float]]] = None  # [[price, qty], ...] if the organisers reveal them
    round: Optional[int] = None              # round the auction was held in (used to de-duplicate)

    @classmethod
    def from_dict(cls, d: dict) -> "AuctionResult":
        bids = d.get("all_bids")
        clean = None
        if isinstance(bids, (list, tuple)):
            clean = []
            for b in bids:
                try:
                    clean.append([float(b[0]), float(b[1])])
                except (TypeError, ValueError, IndexError):
                    continue
        return cls(resource=str(d.get("resource", "")),
                   clearing_price=fopt(d.get("clearing_price"), lo=0.0),
                   won_qty=fnum(d.get("won_qty", 0.0), 0.0, lo=0.0),
                   paid=fnum(d.get("paid", 0.0), 0.0, lo=0.0),
                   total_sold=fopt(d.get("total_sold"), lo=0.0),
                   supply=fopt(d.get("supply"), lo=0.0),
                   all_bids=clean,
                   round=int(fopt(d.get("round"), lo=0)) if fopt(d.get("round")) is not None else None)


@dataclass
class Observation:
    round: int = 0
    phase: str = "all"
    cash: float = 0.0
    raw_inventory: Dict[str, float] = field(default_factory=dict)
    fin_inventory: Dict[str, float] = field(default_factory=dict)
    pending_raw: Dict[str, float] = field(default_factory=dict)      # won, not yet delivered
    lots: List[AuctionLot] = field(default_factory=list)             # on offer this round
    last_prices: Dict[str, float] = field(default_factory=dict)      # our prices last round
    last_sales: Dict[str, float] = field(default_factory=dict)       # our unit sales last round
    last_offered: Dict[str, float] = field(default_factory=dict)     # units we had on sale last round
    competitor_prices: Dict[str, Dict[str, float]] = field(default_factory=dict)  # firm -> product -> price
    auction_results: List[AuctionResult] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Any) -> "Observation":
        if not isinstance(d, dict):
            d = {}
        phase = str(d.get("phase", "all")).lower()
        if phase not in PHASES:
            phase = "all"
        comp = {}
        cp = d.get("competitor_prices", {})
        if isinstance(cp, dict):
            for firm, prices in cp.items():
                pd_ = numdict(prices)
                if pd_:
                    comp[str(firm)] = pd_
        lots = [AuctionLot.from_dict(x) for x in d.get("lots", []) or [] if isinstance(x, dict)]
        res = [AuctionResult.from_dict(x) for x in d.get("auction_results", []) or []
               if isinstance(x, dict)]
        known = {"round", "phase", "cash", "raw_inventory", "fin_inventory", "pending_raw", "lots",
                 "last_prices", "last_sales", "last_offered", "competitor_prices",
                 "auction_results", "extra"}
        extra = dict(d.get("extra", {})) if isinstance(d.get("extra"), dict) else {}
        extra.update({k: v for k, v in d.items() if k not in known})
        return cls(
            round=int(fnum(d.get("round", 0), 0, lo=0)),
            phase=phase,
            cash=fnum(d.get("cash", 0.0), 0.0),
            raw_inventory=numdict(d.get("raw_inventory", {})),
            fin_inventory=numdict(d.get("fin_inventory", {})),
            pending_raw=numdict(d.get("pending_raw", {})),
            lots=[lot for lot in lots if lot.resource],
            last_prices=numdict(d.get("last_prices", {})),
            last_sales=numdict(d.get("last_sales", {})),
            last_offered=numdict(d.get("last_offered", {})),
            competitor_prices=comp,
            auction_results=[r for r in res if r.resource],
            extra=extra,
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Bid:
    resource: str
    quantity: float
    price: float            # per unit


@dataclass
class Action:
    bids: List[Bid] = field(default_factory=list)
    production: Dict[str, float] = field(default_factory=dict)
    prices: Dict[str, float] = field(default_factory=dict)
    info: Dict[str, Any] = field(default_factory=dict)   # diagnostics, never sent to the competition

    def to_dict(self) -> dict:
        return {
            "bids": [{"resource": b.resource, "quantity": b.quantity, "price": b.price}
                     for b in self.bids],
            "production": dict(self.production),
            "prices": dict(self.prices),
        }

    @classmethod
    def from_dict(cls, d: Any) -> "Action":
        if not isinstance(d, dict):
            return cls()
        bids = []
        for b in d.get("bids", []) or []:
            if not isinstance(b, dict):
                continue
            q = fnum(b.get("quantity", 0.0), 0.0, lo=0.0)
            p = fnum(b.get("price", 0.0), 0.0, lo=0.0)
            if q > 0 and b.get("resource") is not None:
                bids.append(Bid(str(b["resource"]), q, p))
        return cls(bids=bids, production=numdict(d.get("production", {})),
                   prices=numdict(d.get("prices", {})))
