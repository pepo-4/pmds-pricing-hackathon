"""Mapping-driven adapter. Most competition formats can be handled by filling in
`mapping` (their field paths) and `action_keys` (their names for our outputs),
without writing code."""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

from ..core.schemas import Action, GameSpec, Observation
from .base import BaseAdapter
from .discovery import SYNONYMS, suggest_mapping

_NAME_KEYS = ("name", "id", "resource", "product", "material", "item", "key")
_QTY_KEYS = ("quantity", "qty", "amount", "units", "value", "price", "stock")


def dig(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif hasattr(cur, part):
            cur = getattr(cur, part)
        else:
            return None
    return cur


def as_map(value: Any) -> Any:
    """[{name: A, quantity: 3}, ...] -> {A: 3}. Dicts and other values pass through."""
    if isinstance(value, (list, tuple)) and value and all(isinstance(v, dict) for v in value):
        out = {}
        for item in value:
            name = next((item[k] for k in _NAME_KEYS if k in item), None)
            qty = next((item[k] for k in _QTY_KEYS if k in item), None)
            if name is not None and qty is not None:
                out[str(name)] = qty
        if out:
            return out
    return value


class GenericAdapter(BaseAdapter):
    MAP_FIELDS = ("raw_inventory", "fin_inventory", "pending_raw", "last_prices", "last_sales",
                  "last_offered")

    def __init__(self, mapping: Optional[Dict[str, str]] = None,
                 action_keys: Optional[Dict[str, str]] = None, integer_quantities: bool = False,
                 spec_mapping: Optional[Dict[str, str]] = None):
        self.mapping = dict(mapping or {})
        self.action_keys = {"bids": "bids", "production": "production", "prices": "prices",
                            "resource": "resource", "quantity": "quantity", "price": "price"}
        self.action_keys.update(action_keys or {})
        self.integer = integer_quantities
        self.spec_mapping = dict(spec_mapping or {})

    @classmethod
    def from_sample(cls, sample: dict, **kw) -> "GenericAdapter":
        mapping, _, _ = suggest_mapping(sample)
        return cls(mapping=mapping, **kw)

    def to_spec(self, raw: Any) -> GameSpec:
        if isinstance(raw, GameSpec):
            return raw
        d = dict(raw) if isinstance(raw, dict) else {}
        for canon, path in self.spec_mapping.items():
            val = dig(raw, path)
            if val is not None:
                d[canon] = val
        return GameSpec.from_dict(d)

    def to_observation(self, raw: Any) -> Observation:
        if isinstance(raw, Observation):
            return raw
        d: Dict[str, Any] = dict(raw) if isinstance(raw, dict) else {}
        for canon in SYNONYMS:
            path = self.mapping.get(canon)
            if path:
                val = dig(raw, path)
                if val is not None:
                    d[canon] = val
        for f in self.MAP_FIELDS:
            if f in d:
                d[f] = as_map(d[f])
        return Observation.from_dict(d)

    def from_action(self, action: Action, raw_obs: Any = None) -> Any:
        k = self.action_keys

        def q(x: float) -> float:
            return float(math.floor(x + 1e-9)) if self.integer else float(x)

        bids = [{k["resource"]: b.resource, k["quantity"]: q(b.quantity), k["price"]: float(b.price)}
                for b in action.bids if q(b.quantity) > 0]
        return {k["bids"]: bids,
                k["production"]: {p: q(x) for p, x in action.production.items() if q(x) > 0},
                k["prices"]: {p: float(x) for p, x in action.prices.items()}}
