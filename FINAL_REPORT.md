# Final report

All results below come from our own simulator, against opponents we wrote. They show that the agent works and how it compares with simpler designs under our assumptions. They are not a forecast for 18 November.

#### 1. Summary

**What exists.** A working agent that bids for inputs, plans production and sets prices in one coordinated decision; a market simulator with four auction formats and four demand families; ten opponents; a paired tournament with train / validation / test splits; 93 passing tests; an adapter layer so that only one file changes on the day.

**Headline (test split, 300 games per agent, 3–8 firms per game, so a random firm wins about 20% of the time).**

| | Hybrid agent (V6) | Rule baseline (V1) |
|---|---|---|
| Games won | 85.0% | 29.7% |
| Finished in the top two | 93.0% | 46.0% |
| Games ending with a loss | 0.7% | 1.7% |
| Median score | 0.72 | 0.08 |

- V6 scored higher than V1 in 94% of the paired games (Wilcoxon, Holm-corrected p ≈ 2e-47).
- No errors in any of the 2,100 agent-games. Mean decision time 13.9 ms, maximum 78 ms.
- V6's cash never fell below zero in the 300 games (lowest point: 148).

**What each layer is worth.**

- LP planning on a least-squares demand fit (V2) already wins 70.3% of games. Most of the gain over rules comes from here.
- Adding forgetting and Thompson sampling (V3) changes nothing measurable: 69.3% wins, same median.
- The full hybrid (V6) beats V2 in 66% of paired games and cuts the share of losing games from 11.0% to 0.7%. The gain is concentrated in first-price auctions (95% wins against 60%), where learned bid shading matters. Under uniform-price, second-price and posted-price rules V6 is no better than V2 (81% vs 81%, 76% vs 79%, 70% vs 73%).

**Rejected on the evidence.**

- The adaptive meta-strategy (V7) is worse than V6 in 82% of paired games. It stays available as profile `adaptive` and is off by default.
- The contextual bandit (V4) has a median score of 0.11 and loses money in 18% of games.
- Tabular Q-learning (V5) is indistinguishable from the rule baseline.

**Ablations (validation split, 100 games each).** Clearly helping: demand-model pricing instead of a bandit, dynamic pricing, LP-coordinated bidding, learned bid shading, inventory awareness. Not shown to help: the rival-price model, forgetting and Thompson sampling, the LP over the greedy planner, soft risk limits, safety stock (intervals include zero). These are left on by default because the tested configuration is the shipped one, and because rival-awareness and risk limits protect against conditions our simulator may not contain; profile `lean` switches three of them off.

**Where the hybrid is weakest.** Immediate delivery of inputs (78% wins against 91% with next-round delivery), posted-price supply (70%), and markets with loyal and bargain-hunting customer segments (lowest median score of the four families).

**Caveats that matter.**

- The means in the test tables are meaningless: 23 of 2,100 agent-games, all in the isoelastic demand family, have scores above 5 (the largest is 21,886). That family was added only for the test and never run before it; as specified it probably allows unbounded demand at prices far below the market, and unbounded prices where the scenario has no ceiling. It was not diagnosed further, so that the test split stays untouched. Read medians, IQM, win rates and paired win shares.
- Because of that flaw, the isoelastic family is a weak test of generalisation. The other three families were seen during development (on other seeds).
- The validation split was used to accept the tuned configuration and for the ablations, so only the test split is held out.
- The opponents are simple. A field of teams running model-based agents like this one would be a different market; in the league where all seven of our versions share a market, V6 wins 40% of 40 nine-firm games (mean rank 2.48).

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

### Tournament - split `test`, 300 games per agent

Score = objective value / fair-share revenue potential of the scenario. Simulated results only. Medians are per-game medians; 'median diff' is the median of paired differences.

**Read the medians, not the means.** 23 of 2100 agent-games have a score above 5 (the largest is 21,886), all in the demand family isoelastic. They dominate every mean in this table.

#### Against the rule baseline (V1)

| agent | games | median score | IQM | mean score | 95% CI of mean | worst 10% | win rate | top-2 | loss rate | ms/decision (mean / max) | errors | median diff vs v1_rule | mean diff | 95% CI of mean diff | P(better) | p (Holm) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1_rule | 300 | 0.0785 | 0.0894 | 0.1533 | [0.1321, 0.1760] | -0.0012 | 29.7% | 46.0% | 1.7% | 0.0 / 3 | 0 | (reference) | | | | |
| v2_opt | 300 | 0.5155 | 0.5347 | 1.6015 | [0.7235, 3.3044] | -0.0538 | 70.3% | 78.7% | 11.0% | 13.5 / 72 | 0 | +0.3359 | +1.4482 | [+0.5723, +3.1574] | 81% | 5.89e-35 |
| v3_bayes | 300 | 0.5051 | 0.5234 | 0.9905 | [0.7375, 1.3145] | -0.0554 | 69.3% | 78.7% | 10.3% | 13.5 / 89 | 0 | +0.3459 | +0.8372 | [+0.5853, +1.1620] | 81% | 4.68e-35 |
| v4_bandit | 300 | 0.1053 | 0.1435 | 0.4001 | [0.2990, 0.5216] | -0.0294 | 36.0% | 50.0% | 18.0% | 7.5 / 54 | 0 | +0.0006 | +0.2468 | [+0.1444, +0.3665] | 51% | 0.00345 |
| v5_rl | 300 | 0.0862 | 0.0937 | 0.1585 | [0.1368, 0.1813] | 0.0003 | 18.3% | 46.0% | 1.3% | 0.0 / 2 | 0 | +0.0036 | +0.0052 | [-0.0081, +0.0188] | 58% | 0.0343 |
| v6_hybrid | 300 | 0.7202 | 0.7371 | 73.9636 | [0.8859, 220.0394] | 0.1799 | 85.0% | 93.0% | 0.7% | 13.9 / 78 | 0 | +0.5750 | +73.8103 | [+0.7324, +219.8919] | 94% | 1.68e-47 |
| v7_meta | 300 | 0.5109 | 0.5211 | 0.9950 | [0.6312, 1.5996] | 0.0283 | 75.7% | 86.3% | 3.0% | 12.9 / 53 | 0 | +0.3421 | +0.8417 | [+0.4775, +1.4444] | 87% | 9.31e-42 |

#### Against the hybrid (V6)

| agent | games | median score | IQM | mean score | 95% CI of mean | worst 10% | win rate | top-2 | loss rate | ms/decision (mean / max) | errors | median diff vs v6_hybrid | mean diff | 95% CI of mean diff | P(better) | p (Holm) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1_rule | 300 | 0.0785 | 0.0894 | 0.1533 | [0.1321, 0.1760] | -0.0012 | 29.7% | 46.0% | 1.7% | 0.0 / 3 | 0 | -0.5750 | -73.8103 | [-219.8919, -0.7324] | 6% | 1.4e-47 |
| v2_opt | 300 | 0.5155 | 0.5347 | 1.6015 | [0.7235, 3.3044] | -0.0538 | 70.3% | 78.7% | 11.0% | 13.5 / 72 | 0 | -0.0476 | -72.3622 | [-216.9611, -0.0228] | 34% | 4.5e-14 |
| v3_bayes | 300 | 0.5051 | 0.5234 | 0.9905 | [0.7375, 1.3145] | -0.0554 | 69.3% | 78.7% | 10.3% | 13.5 / 89 | 0 | -0.0329 | -72.9732 | [-218.9171, +0.0612] | 34% | 2.07e-14 |
| v4_bandit | 300 | 0.1053 | 0.1435 | 0.4001 | [0.2990, 0.5216] | -0.0294 | 36.0% | 50.0% | 18.0% | 7.5 / 54 | 0 | -0.4741 | -73.5635 | [-219.5419, -0.5460] | 7% | 9.93e-40 |
| v5_rl | 300 | 0.0862 | 0.0937 | 0.1585 | [0.1368, 0.1813] | 0.0003 | 18.3% | 46.0% | 1.3% | 0.0 / 2 | 0 | -0.5929 | -73.8052 | [-219.8777, -0.7275] | 6% | 3.76e-48 |
| v6_hybrid | 300 | 0.7202 | 0.7371 | 73.9636 | [0.8859, 220.0394] | 0.1799 | 85.0% | 93.0% | 0.7% | 13.9 / 78 | 0 | (reference) | | | | |
| v7_meta | 300 | 0.5109 | 0.5211 | 0.9950 | [0.6312, 1.5996] | 0.0283 | 75.7% | 86.3% | 3.0% | 12.9 / 53 | 0 | -0.1313 | -72.9687 | [-219.0793, +0.3517] | 18% | 7.18e-30 |

#### Median score (win rate) by demand

| demand | games | v1_rule | v2_opt | v3_bayes | v4_bandit | v5_rl | v6_hybrid | v7_meta |
|---|---|---|---|---|---|---|---|---|
| isoelastic | 85 | 0.096 (18%) | 0.785 (78%) | 0.750 (74%) | 0.250 (47%) | 0.103 (18%) | 0.920 (92%) | 0.686 (78%) |
| linear | 80 | 0.058 (35%) | 0.500 (70%) | 0.493 (69%) | 0.036 (28%) | 0.055 (21%) | 0.758 (85%) | 0.529 (79%) |
| logit | 63 | 0.078 (41%) | 0.493 (70%) | 0.434 (68%) | 0.051 (32%) | 0.080 (22%) | 0.678 (79%) | 0.478 (73%) |
| segments | 72 | 0.079 (28%) | 0.445 (62%) | 0.437 (65%) | 0.214 (36%) | 0.094 (12%) | 0.616 (82%) | 0.373 (72%) |

#### Median score (win rate) by fmt

| fmt | games | v1_rule | v2_opt | v3_bayes | v4_bandit | v5_rl | v6_hybrid | v7_meta |
|---|---|---|---|---|---|---|---|---|
| first_price | 132 | 0.059 (40%) | 0.183 (60%) | 0.171 (58%) | 0.056 (38%) | 0.050 (20%) | 0.656 (95%) | 0.422 (82%) |
| posted | 44 | 0.075 (16%) | 0.644 (73%) | 0.620 (73%) | 0.353 (39%) | 0.090 (5%) | 0.686 (70%) | 0.529 (66%) |
| second_price | 33 | 0.081 (21%) | 0.792 (79%) | 0.746 (79%) | 0.107 (21%) | 0.146 (21%) | 0.732 (76%) | 0.612 (67%) |
| uniform | 91 | 0.101 (24%) | 0.851 (81%) | 0.842 (80%) | 0.137 (37%) | 0.127 (22%) | 0.836 (81%) | 0.634 (75%) |

#### Median score (win rate) by scarcity

| scarcity | games | v1_rule | v2_opt | v3_bayes | v4_bandit | v5_rl | v6_hybrid | v7_meta |
|---|---|---|---|---|---|---|---|---|
| abundant | 83 | 0.088 (31%) | 0.580 (71%) | 0.547 (71%) | 0.159 (41%) | 0.118 (22%) | 0.815 (87%) | 0.616 (80%) |
| balanced | 122 | 0.114 (31%) | 0.598 (73%) | 0.572 (70%) | 0.089 (33%) | 0.103 (19%) | 0.709 (84%) | 0.554 (79%) |
| scarce | 95 | 0.048 (26%) | 0.420 (66%) | 0.374 (66%) | 0.092 (36%) | 0.053 (15%) | 0.631 (84%) | 0.319 (68%) |

#### Median score (win rate) by dynamics

| dynamics | games | v1_rule | v2_opt | v3_bayes | v4_bandit | v5_rl | v6_hybrid | v7_meta |
|---|---|---|---|---|---|---|---|---|
| elasticity | 44 | 0.084 (27%) | 0.468 (75%) | 0.498 (75%) | 0.164 (41%) | 0.078 (20%) | 0.740 (84%) | 0.488 (75%) |
| seasonal | 43 | 0.111 (30%) | 0.482 (70%) | 0.462 (74%) | 0.105 (37%) | 0.119 (21%) | 0.648 (86%) | 0.484 (81%) |
| shock | 32 | 0.114 (44%) | 0.396 (69%) | 0.396 (62%) | 0.100 (38%) | 0.122 (22%) | 0.681 (81%) | 0.433 (66%) |
| stable | 125 | 0.069 (30%) | 0.581 (71%) | 0.562 (70%) | 0.102 (37%) | 0.063 (18%) | 0.741 (85%) | 0.612 (78%) |
| trend | 56 | 0.066 (21%) | 0.504 (66%) | 0.532 (64%) | 0.108 (29%) | 0.093 (12%) | 0.764 (88%) | 0.438 (71%) |

#### Median score (win rate) by two_stage

| two_stage | games | v1_rule | v2_opt | v3_bayes | v4_bandit | v5_rl | v6_hybrid | v7_meta |
|---|---|---|---|---|---|---|---|---|
| False | 138 | 0.077 (28%) | 0.515 (70%) | 0.514 (70%) | 0.104 (37%) | 0.080 (17%) | 0.756 (88%) | 0.515 (75%) |
| True | 162 | 0.082 (31%) | 0.527 (70%) | 0.495 (69%) | 0.106 (35%) | 0.093 (20%) | 0.709 (83%) | 0.496 (76%) |

#### Median score (win rate) by lead_time

| lead_time | games | v1_rule | v2_opt | v3_bayes | v4_bandit | v5_rl | v6_hybrid | v7_meta |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 134 | 0.212 (43%) | 0.428 (61%) | 0.426 (60%) | 0.080 (22%) | 0.218 (26%) | 0.550 (78%) | 0.438 (65%) |
| 1.0 | 166 | 0.039 (19%) | 0.644 (78%) | 0.601 (77%) | 0.134 (47%) | 0.048 (12%) | 0.890 (91%) | 0.597 (84%) |

#### Median score (win rate) by reveal_prices

| reveal_prices | games | v1_rule | v2_opt | v3_bayes | v4_bandit | v5_rl | v6_hybrid | v7_meta |
|---|---|---|---|---|---|---|---|---|
| False | 38 | 0.113 (21%) | 0.673 (79%) | 0.667 (82%) | 0.096 (34%) | 0.051 (5%) | 0.847 (95%) | 0.871 (92%) |
| True | 262 | 0.076 (31%) | 0.490 (69%) | 0.471 (68%) | 0.106 (36%) | 0.092 (20%) | 0.707 (84%) | 0.477 (73%) |

#### Median score (win rate) by reveal_clearing

| reveal_clearing | games | v1_rule | v2_opt | v3_bayes | v4_bandit | v5_rl | v6_hybrid | v7_meta |
|---|---|---|---|---|---|---|---|---|
| False | 87 | 0.096 (34%) | 0.580 (74%) | 0.562 (76%) | 0.089 (34%) | 0.092 (16%) | 0.757 (83%) | 0.613 (78%) |
| True | 213 | 0.075 (28%) | 0.486 (69%) | 0.482 (67%) | 0.112 (37%) | 0.086 (19%) | 0.714 (86%) | 0.484 (75%) |

### All versions in the same market

### League - all versions in the same market, 40 games (split `test`)

| agent | games | mean score | IQM | mean rank | win rate | top-2 |
|---|---|---|---|---|---|---|
| v1_rule | 40 | 0.1380 | 0.0687 | 6.47 | 2.5% | 7.5% |
| v2_opt | 40 | 0.5106 | 0.4111 | 4.12 | 7.5% | 22.5% |
| v3_bayes | 40 | 0.5339 | 0.4156 | 3.85 | 15.0% | 37.5% |
| v4_bandit | 40 | 0.2222 | 0.1111 | 6.30 | 2.5% | 10.0% |
| v5_rl | 40 | 0.1612 | 0.1162 | 5.60 | 0.0% | 2.5% |
| v6_hybrid | 40 | 0.6752 | 0.5540 | 2.48 | 40.0% | 60.0% |
| v7_meta | 40 | 0.4617 | 0.3692 | 4.10 | 22.5% | 42.5% |

## 6. Tuning

Random search over 12 parameters: 24 configurations × 60 train games, then the best three against the previous defaults on 150 validation games (opponents and seeds the search never saw).

| agent | games | mean score | 95% CI | IQM | worst 10% | win rate | top-2 | loss rate | ms/decision (mean / max) | errors | diff vs default | 95% CI | P(better) | p (Holm) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| default | 150 | 0.6460 | [0.5822, 0.7156] | 0.5770 | 0.1300 | 76.0% | 86.7% | 0.0% | 14.1 / 63 | 0 | (reference) | | | |
| c03 | 150 | 0.7290 | [0.6634, 0.7983] | 0.6580 | 0.1971 | 82.7% | 94.0% | 0.0% | 14.8 / 98 | 0 | +0.0829 | [+0.0465, +0.1207] | 76% | 4.2e-09 |
| c18 | 150 | 0.7000 | [0.6320, 0.7714] | 0.6147 | 0.1785 | 82.0% | 90.7% | 0.7% | 14.6 / 58 | 0 | +0.0540 | [+0.0218, +0.0884] | 66% | 0.000262 |
| c22 | 150 | 0.7094 | [0.6383, 0.7824] | 0.6452 | 0.1233 | 77.3% | 88.7% | 0.0% | 13.6 / 60 | 0 | +0.0633 | [+0.0352, +0.0920] | 72% | 2.82e-07 |

**Decision.**

`c03` beats the default on validation (95% interval of the paired difference above zero) and is adopted: `{"band_ref": "mean", "band_over": 0.317, "min_elasticity_frac": 0.345, "prior_elasticity": 2.85, "ts_scale": 0.03, "forgetting": 0.943, "coverage_rounds": 1.42, "fin_cover_rounds": 0.28, "win_bias": 1.57, "safety_z": 0.85, "max_price_step_up": 0.256, "max_price_step_down": 0.256}`

- `c03`: `{"band_ref": "mean", "band_over": 0.317, "min_elasticity_frac": 0.345, "prior_elasticity": 2.85, "ts_scale": 0.03, "forgetting": 0.943, "coverage_rounds": 1.42, "fin_cover_rounds": 0.28, "win_bias": 1.57, "safety_z": 0.85, "max_price_step_up": 0.256, "max_price_step_down": 0.256}`
- `c18`: `{"band_ref": "mean", "band_over": 0.263, "min_elasticity_frac": 0.245, "prior_elasticity": 2.57, "ts_scale": 0.02, "forgetting": 0.934, "coverage_rounds": 1.24, "fin_cover_rounds": 0.82, "win_bias": 0.95, "safety_z": 0.22, "max_price_step_up": 0.251, "max_price_step_down": 0.251}`
- `c22`: `{"band_ref": "mean", "band_over": 0.384, "min_elasticity_frac": 0.482, "prior_elasticity": 2.63, "ts_scale": 0.54, "forgetting": 0.981, "coverage_rounds": 2.38, "fin_cover_rounds": 0.23, "win_bias": 0.94, "safety_z": 0.36, "max_price_step_up": 0.234, "max_price_step_down": 0.234}`

Full tables: `experiments/results/tuning.md`.

## 7. Ablations (validation split)

### Ablations - split `valid`, 100 games each

Each row is the full agent with one change. A negative difference means the module that was removed was helping.

| agent | games | median score | IQM | mean score | 95% CI of mean | worst 10% | win rate | top-2 | loss rate | ms/decision (mean / max) | errors | median diff vs full | mean diff | 95% CI of mean diff | P(better) | p (Holm) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full | 100 | 0.6357 | 0.6592 | 0.7272 | [0.6437, 0.8161] | 0.1657 | 85.0% | 93.0% | 0.0% | 13.7 / 62 | 0 | (reference) | | | | |
| no_opponent_model | 100 | 0.6282 | 0.6718 | 0.7523 | [0.6553, 0.8578] | 0.1360 | 85.0% | 91.0% | 0.0% | 13.4 / 84 | 0 | +0.0000 | +0.0251 | [-0.0115, +0.0669] | 50% | 0.802 |
| no_bayesian_learning | 100 | 0.6422 | 0.6650 | 0.7328 | [0.6485, 0.8236] | 0.1677 | 86.0% | 92.0% | 0.0% | 13.4 / 77 | 0 | +0.0034 | +0.0056 | [-0.0173, +0.0258] | 57% | 0.464 |
| bandit_instead_of_model | 100 | 0.0650 | 0.1211 | 0.2318 | [0.1746, 0.2966] | -0.0170 | 33.0% | 44.0% | 18.0% | 7.4 / 56 | 0 | -0.4370 | -0.4954 | [-0.5949, -0.3978] | 9% | 2.15e-13 |
| greedy_instead_of_lp | 100 | 0.6626 | 0.6655 | 0.7213 | [0.6368, 0.8124] | 0.1332 | 83.0% | 91.0% | 0.0% | 2.0 / 13 | 0 | -0.0132 | -0.0059 | [-0.0303, +0.0208] | 37% | 0.393 |
| no_safety_stock | 100 | 0.6384 | 0.6529 | 0.7123 | [0.6309, 0.7993] | 0.1507 | 85.0% | 94.0% | 0.0% | 13.5 / 54 | 0 | -0.0032 | -0.0149 | [-0.0322, +0.0027] | 38% | 0.0404 |
| no_dynamic_pricing | 100 | 0.4313 | 0.4637 | 0.5321 | [0.4559, 0.6138] | 0.0424 | 69.0% | 79.0% | 3.0% | 11.6 / 50 | 0 | -0.1084 | -0.1951 | [-0.2715, -0.1215] | 19% | 4.1e-09 |
| no_inventory_awareness | 100 | 0.5413 | 0.5727 | 0.6471 | [0.5606, 0.7465] | 0.1235 | 81.0% | 91.0% | 0.0% | 13.6 / 90 | 0 | -0.0523 | -0.0801 | [-0.1415, -0.0049] | 23% | 2.09e-07 |
| no_coordinated_bidding | 100 | 0.4403 | 0.4259 | 0.5240 | [0.4439, 0.6135] | 0.0683 | 74.0% | 87.0% | 0.0% | 2.1 / 14 | 0 | -0.1021 | -0.2032 | [-0.2575, -0.1468] | 18% | 7.75e-10 |
| no_bid_learning | 100 | 0.4497 | 0.4566 | 0.5229 | [0.4410, 0.6108] | -0.0091 | 71.0% | 83.0% | 4.0% | 13.2 / 82 | 0 | +0.0000 | -0.2043 | [-0.2820, -0.1350] | 26% | 3.62e-08 |
| no_risk_limits | 100 | 0.5967 | 0.6246 | 0.7147 | [0.6254, 0.8151] | 0.1404 | 86.0% | 92.0% | 1.0% | 13.3 / 66 | 0 | +0.0005 | -0.0125 | [-0.0581, +0.0306] | 54% | 0.802 |
| meta_strategy_on | 100 | 0.5309 | 0.5295 | 0.5522 | [0.4800, 0.6289] | 0.0009 | 72.0% | 83.0% | 5.0% | 12.5 / 57 | 0 | -0.0765 | -0.1750 | [-0.2434, -0.1134] | 22% | 1.49e-07 |

| variant | what changes |
|---|---|
| no_opponent_model | rival prices ignored entirely |
| no_bayesian_learning | no forgetting, no Thompson sampling, no price variation |
| bandit_instead_of_model | contextual bandit over a price grid sets prices |
| greedy_instead_of_lp | greedy planner replaces the LP |
| no_safety_stock | no buffer above planned sales |
| no_dynamic_pricing | price fixed at the first-round rule |
| no_inventory_awareness | stock left after the round valued at salvage only |
| no_coordinated_bidding | bids = fixed share of unit margin, no LP valuation |
| no_bid_learning | fixed 20% bid shading, no clearing-price model |
| no_risk_limits | no cash reserve, no spend cap, no margin floor, wide price steps |
| meta_strategy_on | bandit switches between demand model and follower rule |

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
