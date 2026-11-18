# Hackathon adaptation guide — 18 November, 14:00–19:00

Goal: a working submission by 16:15, then improve it. Never trade a working submission for a feature.

## Before the day

```bash
pip install -r requirements.txt
python -m pytest -q                      # must be all green on your laptop
python main.py simulate --seed 3         # sanity check
```

Bring the repository offline (no internet may be available). Decide who edits the adapter and who reads the rules.

## 14:30 — ask these at the briefing, in this order

1. **Are pre-written code, AI assistants, internet and external APIs allowed?** If pre-written code is not allowed, stop here and rebuild the simplest part by hand (section "If pre-built code is not allowed").
2. What is scored? Profit, cash, valuation, revenue, share? One run, an average, a median, a rank?
3. Auction format (first price / second price / uniform / posted), reserve prices, what is revealed afterwards.
4. Is the number of rounds known? One game or several? Does our state survive between games?
5. Do we see rival prices, sales, stock?
6. Inventory: holding costs, perishability, value of leftovers at the end. Borrowing? Bankruptcy?
7. Time limit per decision, where the code runs, allowed libraries, how and when to submit.
8. Is there a local simulator or baseline bots to practise against?

## Minute plan

| Time | Step | Done when |
|---|---|---|
| 14:45–15:05 | Read the SDK and rules. Fill in the spec (step 1 below) | `spec.validate()` passes |
| 15:05–15:45 | Adapter (steps 2–3) | their sample observation → a sensible action |
| 15:45–16:15 | Integration test against their simulator or baseline (step 4). **Submit this version if submissions are open** | one full game without errors |
| 16:15–17:30 | Set the rules-dependent knobs (step 5), then tune at most 3 knobs | each change checked on ≥ 20 games, not on one |
| 17:30–18:15 | Compare the candidate with the 16:15 version; keep the old one unless the new one is clearly better | decision made |
| 18:15– | Freeze. Final submission with margin. Watch the logs | submitted |

If the final tournament runs before 19:00, move every step earlier by that amount.

## Step 1 — describe the game (`GameSpec`)

Write their rules as a dict (or JSON) in our terms:

```python
SPEC = {
  "products":  {"widget": {"recipe": {"steel": 2, "plastic": 1}, "unit_cost": 3.0,
                           "price_min": 0, "price_max": 200, "salvage": 0, "holding_cost": 0}},
  "resources": {"steel":   {"auction_format": "first_price", "reserve_price": 1.0},
                "plastic": {"auction_format": "first_price"}},
  "horizon": 50,            # omit if unknown
  "n_firms": 12,
  "capacity": 100,          # omit if unlimited
  "lead_time": 0,           # 0 = inputs usable in the round they are won, 1 = next round
  "initial_cash": 10000,
  "objective": "profit",    # profit | cash | valuation | revenue | market_share | {"w_profit":..,"w_units":..}
  "demand_hint": {"widget": 40},              # optional: rough units per round per firm
}
```

- `auction_format`: `first_price` (pay your bid), `second_price` / `uniform` (pay the highest losing bid), `posted` (fixed price in `reserve_price`).
- If they publish a typical input price, put it in `ref_price`. If they publish typical demand, put it in `demand_hint`. Both remove most of the first-round guessing.
- Run `GameSpec.from_dict(SPEC).validate()` once: it raises on a real inconsistency.

## Step 2 — map their observation

```bash
python main.py discover their_sample_observation.json
```

prints a proposed mapping and lists fields it could not place. Copy it into `MAPPING` in `bizagent/adapters/competition_adapter.py` and correct it. For an SDK object instead of JSON:

```python
from bizagent.adapters.discovery import describe
print(describe(their_observation_object))     # fields, methods, signatures
```

Canonical fields (only `cash` and, for bidding, `lots` are essential; everything else improves decisions):

| Field | Meaning |
|---|---|
| `round`, `phase` | round index from 0; `all`, `bid` or `sell` |
| `cash`, `raw_inventory`, `fin_inventory`, `pending_raw` | our state |
| `lots` | `[{resource, quantity, reserve_price, format}]` on auction now |
| `last_prices`, `last_sales`, `last_offered` | our previous round, per product. **`last_sales` is what the demand model learns from** |
| `competitor_prices` | `{firm: {product: price}}`, previous round |
| `auction_results` | `[{resource, clearing_price, won_qty, paid, round}]`. Include `round` so results are never counted twice |

Checks that catch most integration mistakes:
- units: is their price per unit or per lot? Is quantity in units of product or of input?
- timing: are `last_sales` from the previous round (correct) or cumulative (subtract)?
- phases: if they ask for bids and for prices at different moments, set `phase` to `bid` / `sell`.

## Step 3 — map our action

`ACTION_KEYS` renames the outputs. Set `INTEGER_QUANTITIES = True` if units must be whole. If their API takes one call per bid or per product, loop over `action.bids` / `action.prices` inside `from_action`.

Three ways to plug in:

```python
from bizagent.api import CompetitionAgent, decide
agent = CompetitionAgent(spec=SPEC, profile="balanced")
answer = agent.act(their_observation)                     # object style

answer, memory = decide(their_observation, memory, spec=SPEC)   # function style with returned state
```
```bash
python main.py serve --spec spec.json --port 8000         # HTTP: POST observation to /act
python main.py act --spec spec.json --state state.pkl < obs.json   # one process per round
```

## Step 4 — integration test

1. Feed 3 real observations by hand; read the actions. Are bids below what the product sells for? Is the price in their range?
2. Play one full game. Then check `agent.agent.errors` (must be 0; `agent.agent.last_error` says why if not) and `agent.agent.max_ms`.
3. Look at the first three rounds: did we win anything in the auction? Did we sell anything? If not, see "Symptoms" below.

## Step 5 — rules-dependent settings

| What you learned | Setting |
|---|---|
| Scoring is profit / cash / valuation | `objective` in the spec; salvage values for valuation |
| Scoring is revenue or share | `objective: revenue` / `market_share`, or profile `market_share` |
| Ranking by median or by many runs | profile `conservative` |
| Winner takes all on one run | profile `aggressive` is a gamble; decide deliberately |
| Horizon unknown | leave `horizon` out; set `assumed_remaining` |
| Few rounds (< 15) | `early_step: 0.5`, `ts_scale: 0.2`: learn fast, explore little |
| Rival prices hidden | nothing to do; the band and rival model switch off by themselves |
| Clearing price hidden | nothing to do; it is inferred from our wins and losses |
| Time limit under 50 ms | `solver: greedy`, `n_tranches: 2` |

Profiles: `python main.py simulate --profile aggressive`. A profile is a YAML file in `config/profiles/`; copy one, change values, pass its path.

## Symptoms and the knob to turn

| Symptom | Likely cause | Change |
|---|---|---|
| We win nothing in the auctions | bids shaded too far | `win_bias: 1.5`; first rounds: `prior_clearing_frac: 0.7` |
| We win everything and margins are thin | overbidding | `value_haircut: 0.15`, `win_bias: 0.8` |
| We sell out every round | prices too low or too little stock | check `last_sales` mapping; `coverage_rounds: 2.0` |
| Stock piles up | prices too high | `band_over` lower, `band_ref: mean`; check that `competitor_prices` is mapped |
| First-round price is absurd | no price information | give `ref_price`, `demand_hint`, or set `init_price_frac` |
| Prices jump around | exploration too strong | `ts_scale: 0.2`, `dither: 0` |
| Cash runs out | too much committed | `max_spend_frac: 0.5`, `cash_reserve_frac: 0.15` |
| Anything strange | — | switch to `python main.py ... --agent v1_rule`: the rule-based agent, same adapter |

## If pre-built code is not allowed

Rebuild in this order; each step is a complete agent:
1. Price = 2% below the cheapest rival, never below (production cost + what we paid for inputs) × 1.08.
2. Bid 70% of (price − production cost) / inputs per unit, for 1.3 rounds of expected sales. Stop bidding when inputs would arrive after the last round.
3. Produce expected sales × 1.1. In the last round, cut the price to clear stock.

That is `RuleBaselineAgent` in `bizagent/agents/heuristics.py` (about 40 lines of logic).

## Rule compliance

Use only what the organisers allow. The agent makes no network calls and uses only numpy, scipy and pyyaml at run time.
