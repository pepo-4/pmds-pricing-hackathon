# Architecture

```
their data ──► adapters/ ──► core/schemas (Observation, GameSpec)
                                   │
                                   ▼
                     core/decision_engine.DecisionEngine
        ┌───────────────┬───────────────┬──────────────────┬──────────────┐
        ▼               ▼               ▼                  ▼              ▼
 market/demand_model  market/        production/planner  procurement/   pricing/bandit
 (Bayesian demand)    competitor_model (LP + greedy)     bidding        (meta, V4)
                                   │
                                   ▼
                        risk/risk_manager (every action passes through)
                                   │
                                   ▼
                core/schemas.Action ──► adapters/ ──► their format
```

The strategic core imports nothing from `adapters/` or `simulation/`. Only `adapters/competition_adapter.py` knows the competition's format.

## Modules

| Module | Role |
|---|---|
| `core/schemas.py` | Canonical `GameSpec` (rules), `Observation`, `Action`. Lenient parsing for the decision loop, strict `GameSpec.validate()` for integration time. |
| `core/config.py` | `StrategyConfig`: every behavioural knob. `config/default.yaml` + `config/profiles/*.yaml`. |
| `core/decision_engine.py` | The coordinator. One call per decision: update beliefs → value stock and inputs → solve the plan → value extra inputs → bids → risk check. |
| `market/demand_model.py` | Per product: demand = a + b·price + c·rival price, Bayesian with forgetting towards the prior; stock-outs move the level only; parameters projected to b < 0, 0 ≤ c < \|b\|. |
| `market/competitor_model.py` | Rival price aggregate and forecast, how much rivals follow our moves, per-firm labels. |
| `production/planner.py` | LP over sales volume (on the concave hull of revenue), production, input use, carried stock. Greedy solver with the same interface as fallback. |
| `procurement/bidding.py` | Clearing-price distribution per resource; bid rule per auction format; budget allocation by value per euro. |
| `pricing/bandit.py` | Discounted Thompson selector (meta-strategy) and a contextual price bandit (V4). |
| `risk/risk_manager.py` | Final feasibility and safety check, independent of how the action was produced. |
| `agents/` | Common agent interface; `StrategicAgent` (engine + rule-based fallback); `RuleBaselineAgent` (V1); registry of versions. |
| `adapters/` | `GenericAdapter` (mapping-driven), `discovery` (proposes a field mapping from a sample), `competition_adapter` (the file to edit on the day). |
| `api.py`, `main.py` | Entry points: object, function with memory, CLI (JSON in / out), HTTP. |
| `simulation/` | Synthetic market, auctions, four demand families, ten opponents, scenario generator. |
| `evaluation/` | Paired tournament runner and statistics. |

## One decision, step by step

1. **Beliefs.** Last round's sales update the demand model (a sell-out counts as a lower bound). Rival prices update the competitor model. Auction results update the clearing-price distribution; without a revealed price it is inferred from our own wins and losses.
2. **Link prices.** ω_r = expected replacement price of input r (salvage in the last round). μ_j = value of an unsold finished unit, capped at replacement cost and at what the remaining rounds can absorb. Only stock we can expect to sell soon keeps the value μ; the rest is worth salvage, which makes the plan clear excess stock.
3. **Price window.** Around our last price (trust region), inside a band around rival prices; the upper band is lifted while we would sell out at its top, because then price is the only way to ration stock.
4. **Plan.** LP → production, planned sales, price = the highest price at which expected demand reaches planned sales. A safety buffer is added when inputs allow.
5. **Input values.** Re-solve the LP with more of resource r (other inputs purchasable at expected prices, so complements are valued jointly). The differences are a decreasing schedule of tranche values. An extra tranche buys cover for later rounds only below the typical price.
6. **Bids.** Second / uniform price: bid the value, later tranches shaded slightly. First price: argmax (v − b)·P(win | b). Posted: buy while value exceeds the price. Total commitment within the cash budget.
7. **Risk.** Prices not below the floor (lifted only when liquidating at the end), production feasible, bids capped at value, at lot size and at the budget.

## Timing variants supported

- `phase = all`: bids, production and prices in one answer. `bid` / `sell`: two answers per round.
- `lead_time = 0`: inputs usable in the round they are won. `lead_time ≥ 1`: bids are valued against the projected state after delivery.
- Unknown horizon: an assumed number of remaining rounds (`assumed_remaining`); no end-game liquidation is triggered.

## Agent versions

| Version | What it is |
|---|---|
| V1 `v1_rule` | Rules only: undercut the cheapest rival, cost floor, bid a share of the margin |
| V2 `v2_opt` | LP planning with a plain least-squares demand fit; no forgetting, no exploration, no rival model, fixed bid shading |
| V3 `v3_bayes` | V2 + forgetting, Thompson sampling, early price variation |
| V4 `v4_bandit` | Contextual bandit over a price grid instead of the demand model |
| V5 `v5_rl` | Tabular Q-learning (the only RL variant; deep RL rejected, see `FINAL_REPORT.md`) |
| V6 `v6_hybrid` | V3 + rival model, competitor band, learned bid shading |
| V7 `v7_meta` | V6 + a slow bandit choosing between the demand model and the follower rule |
