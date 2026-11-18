"""THE FILE TO EDIT ON THE DAY.

Everything competition-specific belongs here and nowhere else. As shipped it
is the generic adapter, which already works for JSON that uses our field names
or common synonyms. Typical edits, in order of likelihood:

1. `MAPPING`     - where their observation keeps each thing (dotted paths).
                   `python main.py discover their_sample.json` prints a first draft.
2. `ACTION_KEYS` - what they call bids / production / prices in the answer.
3. `to_observation` - anything a path cannot express (units, nested lists, phases).
4. `from_action` - if they want a different shape (e.g. one call per bid).
"""
from __future__ import annotations

from typing import Any

from ..core.schemas import Action, Observation
from .generic import GenericAdapter

MAPPING: dict = {
    # "cash": "company.cash",
    # "raw_inventory": "company.materials",
    # "fin_inventory": "company.products",
    # "lots": "market.auctions",
    # "competitor_prices": "market.prices",
}
ACTION_KEYS: dict = {
    # "bids": "auction_bids", "production": "production_plan", "prices": "selling_prices",
}
INTEGER_QUANTITIES = False


class CompetitionAdapter(GenericAdapter):
    def __init__(self):
        super().__init__(mapping=MAPPING, action_keys=ACTION_KEYS,
                         integer_quantities=INTEGER_QUANTITIES)

    def to_observation(self, raw: Any) -> Observation:
        obs = super().to_observation(raw)
        # competition-specific fix-ups go here
        return obs

    def from_action(self, action: Action, raw_obs: Any = None) -> Any:
        return super().from_action(action, raw_obs)
