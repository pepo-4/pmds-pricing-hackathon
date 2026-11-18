# Final report

{{include:findings.md}}

## 2. What was implemented

| Deliverable | Where |
|---|---|
| Autonomous agent (bids, production, prices, one coordinated decision) | `bizagent/core/decision_engine.py`, `bizagent/agents/strategic.py` |
| Demand estimation | `bizagent/market/demand_model.py` |
| Competitor model | `bizagent/market/competitor_model.py` |
| Production and price optimisation (LP + greedy fallback) | `bizagent/production/planner.py` |
| Procurement and auction strategy | `bizagent/procurement/bidding.py` |
| Financial and risk limits | `bizagent/risk/risk_manager.py`, cash rules in the engine |
| Agent versions V1–V7 | `bizagent/agents/registry.py` |
| Market simulator, four demand families, four auction formats | `bizagent/simulation/` |
| Ten opponent archetypes | `bizagent/simulation/opponents.py` |
| Tournament, statistics, tuning, ablations, league | `bizagent/evaluation/`, `experiments/` |
| Adapter, interface discovery, entry points (object, function, CLI, HTTP) | `bizagent/adapters/`, `bizagent/api.py`, `main.py` |
| Strategy profiles | `config/` |
| Tests (93) | `tests/` |
| Research, architecture, assumptions | `research/` |
| Adaptation guide | `HACKATHON_ADAPTATION_GUIDE.md` |
| Example input and output | `examples/` |
| Deployment | `Dockerfile` (written, not built here), `pyproject.toml`, `requirements.txt` |

Architecture: see `research/ARCHITECTURE.md`.

## 3. Methods selected, and alternatives rejected

Selected (reasons and sources in `research/LITERATURE_REVIEW.md`):

- Bayesian linear demand model with forgetting; stock-outs treated as lower bounds.
- One LP per decision for sales volume, production and price; resource and stock values link the rounds.
- Bids from LP marginal values: truthful under second / uniform price, shaded by a learned win-probability curve under first price.
- Rival price band and reaction estimate.
- Hard limits on every action; rule-based fallback.

Rejected:

| Alternative | Why |
|---|---|
| Deep RL (DQN, PPO, SAC), multi-agent RL | The real environment is unknown until the day and cannot be trained on; training on our own simulator would optimise our assumptions. No reviewed competition winner used it. Only tabular Q-learning is included (V5 and one opponent), as a benchmark |
| Gradient boosting / neural demand models | A game gives tens of observations per product; a three-parameter model is what that can identify |
| Contextual bandit as the pricing head | Implemented as V4 and measured below |
| Budget pacing with dual updates | Not implemented. With the per-round spend cap and reserve, the hybrid's cash never fell below zero in the 300 test games |
| CVXPY / Pyomo / OR-Tools | SciPy's HiGHS solves these LPs in about a millisecond with no extra dependency |

## 4. Evaluation protocol

- A game = one random scenario + our candidate + n−1 opponents drawn at random from the pool. Every candidate plays the same games (same scenario, seed, opponents), so comparisons are paired.
- **Score** = objective value (profit unless stated) divided by the scenario's fair-share revenue potential (market size per firm × reference price × rounds), so scenarios of different size are comparable. "Win" = highest score in the game.
- Three splits, disjoint seeds:

| Split | Demand families | Opponents | Used for |
|---|---|---|---|
| train | linear, logit, segments | 7 | random-search tuning |
| valid | linear, logit, segments | 10 | accepting a tuned configuration; ablations |
| test | + isoelastic | 10 | the numbers in section 5; no decision was taken from it |

- Intervals are 95% percentile bootstrap intervals over games. p-values are Wilcoxon signed-rank tests on paired differences with Holm correction.
- The isoelastic demand family was never used in any game before the test run. Three opponents (`opp_inventory`, `opp_ts`, `opp_qlearn`) were hidden from the tuner.

## 5. Results on the test split

{{include:tournament_test.md}}

### All versions in the same market

{{include:league.md}}

## 6. Tuning

{{include:tuning_decision.md}}

Full tables: `experiments/results/tuning.md`.

## 7. Ablations (validation split)

{{include:ablations.md}}

## 8. Robustness

Covered by `tests/test_robustness.py` and `tests/test_integration.py` (all passing):

- empty, partial and malformed observations (wrong types, NaN, infinities, negative values, unknown fields and enum values, unknown resources);
- zero and negative cash, zero inventory, absurd reserve prices, extreme magnitudes;
- game shapes from no information at all to six products and three resources, one-round games, no price freedom, lead time 2;
- solver failure (the LP raising an exception): the greedy planner takes over inside the same decision;
- engine failure (an exception anywhere in the strategy): the rule-based agent answers that round;
- malformed JSON on the command line: a valid empty action and a non-zero exit code;
- every agent version and opponent completes games in both timing modes without errors;
- conservation of money and material in the simulator, and no hidden information in observations.

Decision time and error counts for the tournament games are in the tables above (columns "ms/decision" and "errors").

## 9. Defects the simulator exposed during development

These were found on six development games (seeds 1–6, outside all splits) and fixed before tuning:

1. **Stock-out trap.** After selling out, the demand estimate stayed at the small quantity sold, so the agent kept buying little and kept selling out. Fix: repeated stock-outs raise the demand level geometrically.
2. **Slope contamination.** Stock-outs entered the regression as data points and taught the model that demand rises with price. Fix: a stock-out moves the level only.
3. **Unbounded carried stock.** The planner valued any amount of unsold stock at replacement cost and so kept converting cheap inputs into goods nobody bought. Fix: only stock that can be sold soon keeps that value.
4. **Safety stock ignored the shelf** and input cover ignored finished goods. Fixed.
5. **Pricing far above the market** when a few loyal customers kept buying. Mitigation: the upper price band refers to the mean rival price and is lifted only when we are short of stock.

A 60-game run on validation seeds before these fixes (linear and logit demand only at that time) already had the hybrid ahead of the rule baseline; those seeds were later reused for accepting the tuned configuration, which is why only the test split is reported as held-out.

## 10. Known limitations and remaining risks

- **Everything above is simulated.** The opponents are ours; the demand families are ours. Real teams and the real market may behave differently. The ranking of versions is evidence about the design, not a forecast of the result.
- The real rules are unknown. Negotiation, customer order auctions or combinatorial lots are not covered and would need new modules.
- The first-round price is a guess when nothing is revealed (no price ceiling, hint or rival price). A wrong first price costs the first two or three rounds.
- With one decision per round and immediate delivery (`phase = all`, `lead_time = 0`), production is planned on expected auction wins, so it can over- or under-shoot.
- The demand model is linear and local. Where demand has a cliff (bargain hunters who buy only from the cheapest firm), it can settle on a locally good price.
- Unknown horizon: the agent assumes a fixed number of remaining rounds and does not liquidate at the end.
- Multi-unit uniform-price auctions: later tranches are shaded by a fixed rule, not an equilibrium strategy.
- Whether pre-written code and AI assistance are allowed is unknown. Ask before using this repository.
- The Dockerfile was not built in this environment; the HTTP server was not load-tested.

## 11. Adaptation steps on the day

Full procedure with a minute plan: `HACKATHON_ADAPTATION_GUIDE.md`. In short: (1) write the rules as a `GameSpec` dict and validate it; (2) `python main.py discover sample.json`, paste the mapping into `bizagent/adapters/competition_adapter.py`; (3) rename the outputs in `ACTION_KEYS`; (4) play one game, check `errors == 0`; (5) set the objective and profile; (6) submit, then tune at most three knobs.

## 12. Reproduce

```bash
python -m pytest -q
python experiments/tune.py --configs 24 --games 60 --top 3 --valid-games 150
python experiments/tournament.py --split test --games 300
python experiments/ablations.py --split valid --games 100
python experiments/league.py --split test --games 40
python experiments/make_report.py
```

Seeds are fixed, so the tables regenerate exactly on the same library versions (numpy 2.5.3, scipy 1.18.1, Python 3.13 here). The tuning tables in `experiments/results/tuning.md` were produced with the pre-tuning defaults as "default"; re-running `tune.py` now would compare against the tuned defaults instead.
