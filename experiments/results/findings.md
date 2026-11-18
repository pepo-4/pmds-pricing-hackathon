All results below come from our own simulator, against opponents we wrote. They show that the agent works and how it compares with simpler designs under our assumptions. They are not a forecast for 18 November.

## 1. Summary

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
