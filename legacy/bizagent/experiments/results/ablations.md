# Ablations - split `valid`, 100 games each

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
