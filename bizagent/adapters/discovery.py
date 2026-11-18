"""Interface discovery: look at whatever the organisers hand out (a sample JSON
observation, a JSON schema, an SDK class or function) and propose how its
fields map onto our canonical observation.

    python main.py discover sample_observation.json
"""
from __future__ import annotations

import dataclasses
import difflib
import inspect
import re
from typing import Any, Dict, List, Tuple

SYNONYMS: Dict[str, List[str]] = {
    "round": ["round", "turn", "period", "step", "t", "tick", "day", "time", "round_number",
              "current_round", "round_idx", "iteration"],
    "phase": ["phase", "stage", "decision", "decision_type"],
    "cash": ["cash", "money", "budget", "balance", "funds", "capital", "wallet", "liquidity"],
    "raw_inventory": ["raw_inventory", "raw_materials", "materials", "raw", "resources",
                      "resource_inventory", "raw_stock", "inputs", "components"],
    "fin_inventory": ["fin_inventory", "finished_goods", "finished", "finished_inventory",
                      "product_inventory", "products", "inventory", "stock", "goods", "warehouse"],
    "pending_raw": ["pending_raw", "pending", "incoming", "in_transit", "deliveries", "on_order"],
    "lots": ["lots", "auctions", "auction", "offers", "available_lots", "market_lots", "supply",
             "open_auctions", "tenders"],
    "last_prices": ["last_prices", "my_prices", "own_prices", "previous_prices", "prices",
                    "posted_prices"],
    "last_sales": ["last_sales", "sales", "units_sold", "sold", "my_sales", "demand_served"],
    "last_offered": ["last_offered", "offered", "units_offered", "available_for_sale"],
    "competitor_prices": ["competitor_prices", "competitors", "rival_prices", "market_prices",
                          "opponent_prices", "others", "other_prices", "opponents"],
    "auction_results": ["auction_results", "auction_result", "results", "auction_outcome",
                        "last_auction", "bid_results", "allocations"],
}


def _norm(s: str) -> str:
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(s))      # camelCase -> snake_case
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def flatten(obj: Any, prefix: str = "", depth: int = 3) -> Dict[str, Any]:
    """Dotted paths of every key down to `depth` levels."""
    out: Dict[str, Any] = {}
    if isinstance(obj, dict) and depth > 0:
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else str(k)
            out[path] = v
            out.update(flatten(v, path, depth - 1))
    return out


def suggest_mapping(sample: Any) -> Tuple[Dict[str, str], List[str], List[str]]:
    """Return (canonical field -> path in the sample, canonical fields not found, unused paths)."""
    if "properties" in sample and isinstance(sample.get("properties"), dict):
        sample = sample["properties"]                        # a JSON schema
    paths = flatten(sample)
    mapping: Dict[str, str] = {}
    scores: Dict[str, float] = {}
    for canon, names in SYNONYMS.items():
        best, best_path = 0.0, None
        for path in paths:
            leaf = _norm(path.split(".")[-1])
            if leaf in names:
                score = 2.0 - 0.01 * names.index(leaf) - 0.1 * path.count(".")
            else:
                score = max(difflib.SequenceMatcher(None, leaf, n).ratio() for n in names)
                score = score if score >= 0.82 else 0.0
            if score > best:
                best, best_path = score, path
        if best_path is not None:
            mapping[canon], scores[canon] = best_path, best
    # one path cannot serve two fields: keep the stronger claim
    by_path: Dict[str, str] = {}
    for canon in sorted(mapping, key=lambda c: -scores[c]):
        if mapping[canon] in by_path:
            del mapping[canon]
        else:
            by_path[mapping[canon]] = canon
    missing = [c for c in SYNONYMS if c not in mapping]
    used = set(mapping.values())
    unused = [p for p in paths if p not in used and not any(p.startswith(u + ".") for u in used)
              and not any(u.startswith(p + ".") for u in used)]
    return mapping, missing, unused


def describe(obj: Any) -> Dict[str, Any]:
    """Summarise an SDK object: fields of a dataclass / class, or the signature of a function."""
    info: Dict[str, Any] = {"type": type(obj).__name__}
    target = obj if inspect.isclass(obj) or callable(obj) else type(obj)
    if dataclasses.is_dataclass(target):
        info["fields"] = {f.name: str(f.type) for f in dataclasses.fields(target)}
    elif inspect.isclass(target):
        info["annotations"] = {k: str(v) for k, v in getattr(target, "__annotations__", {}).items()}
        info["methods"] = {}
        for name, member in inspect.getmembers(target, predicate=callable):
            if name.startswith("_"):
                continue
            try:
                info["methods"][name] = str(inspect.signature(member))
            except (TypeError, ValueError):
                info["methods"][name] = "(?)"
    if callable(obj) and not inspect.isclass(obj):
        try:
            info["signature"] = str(inspect.signature(obj))
        except (TypeError, ValueError):
            pass
    if not inspect.isclass(obj) and not callable(obj) and hasattr(obj, "__dict__"):
        info["attributes"] = {k: type(v).__name__ for k, v in vars(obj).items() if not k.startswith("_")}
    doc = inspect.getdoc(obj)
    if doc:
        info["doc"] = doc.splitlines()[0]
    return info


def report(sample: Any) -> str:
    mapping, missing, unused = suggest_mapping(sample)
    lines = ["Suggested mapping (canonical field <- path in their data):"]
    lines += [f"  {c:18s} <- {p}" for c, p in mapping.items()]
    lines.append("Canonical fields with no match (optional unless marked *): " +
                 ", ".join(("*" + m) if m in ("cash", "lots") else m for m in missing))
    lines.append("Their fields we do not use yet: " + ", ".join(unused[:40]))
    return "\n".join(lines)
