# Competition analysis — AIRIC × PMDS AI Dynamic Pricing Hackathon

Researched on 10 November 2026. Facts and hypotheses are kept apart. Confidence is High / Medium / Low.

## 1. Verified facts

| # | Fact | Source | Confidence |
|---|---|---|---|
| F1 | Title "AIRIC X PMDS – AI Dynamic Pricing Hackathon", organised by "AIRIC \| Polimi" | [Eventbrite](https://www.eventbrite.it/e/airic-x-pmds-ai-dynamic-pricing-hackathon-tickets-2003053526017) (page read) | High |
| F2 | Wednesday 18 Nov 2026, 14:00–19:00 CET, in person, doors 13:30 | Eventbrite | High |
| F3 | Venue: Sala conferenze Emilio Gatti, DEIB building 20, Via Ponzio 34/5, Milano | Eventbrite; [PMDS events](https://www.polimidatascientists.it/events) (page read) | High |
| F4 | Agenda: 14:00–14:30 welcome and team formation; 14:30–14:45 "Challenge Briefing – Introduction to the challenge, objectives and rules"; 14:45 hackathon begins. Rest of the agenda not readable | Eventbrite | High |
| F5 | "Bid for resources, set prices, and compete with other teams in a dynamic market simulation." | Eventbrite | High |
| F6 | "Build an AI agent that makes strategic decisions, bids for raw materials, plans production, sets prices, and adapts to competitors' moves." | PMDS events page | High |
| F7 | Open to students of Politecnico di Milano | Eventbrite FAQ (indexed text) | High |
| F8 | Teams recommended; you may join a team on the day or play alone. No team-size limit found | Eventbrite FAQ (indexed text) | High (size limit: Medium) |
| F9 | "We recommend bringing a computer with Python and an IDE installed"; GitHub Codespaces offered as an alternative | Eventbrite FAQ (indexed text) | High |
| F10 | Prizes are "a surprise for now"; optional aperitivo afterwards | PMDS LinkedIn post, seen only as search snippets | Medium |
| F11 | AIRIC earlier ran "Generative AI plays Taboo" with AIM: students built a Python bot for a special version of the game | LinkedIn snippets | Medium |
| F12 | PMDS is the PoliMi student AI / data-science association; AIRIC is PoliMi's AI research and innovation centre | [polimi.it](https://www.polimi.it/en/students/associazioni-degli-studenti/polimi-data-scientists), [polimi.it](https://www.polimi.it/en/research/polimi-ai/the-research) | High |

**Not published anywhere we could find:** scoring function, auction format, number of products or resources, horizon, SDK or API, runtime limits, submission format, policy on AI assistants / pre-written code / internet, prior editions, a public simulator. The PMDS GitHub organisation showed only unrelated repositories.

**Consequence:** real coding time is about 4h15m (14:45–19:00), less any final tournament run. Rules arrive at 14:30.

## 2. Hypotheses (not verified)

| # | Hypothesis | Why | Confidence |
|---|---|---|---|
| H1 | A Python starter kit or simulator client is supplied, probably a GitHub repo | Codespaces recommendation; the Taboo precedent | Medium–High |
| H2 | Round-based: auction → production → price posting → demand | Wording of F5/F6; design of TAC SCM, SCML, DPC | Medium |
| H3 | Ranking from a tournament of all teams' agents; score is cumulative profit, final cash or valuation | "compete with other teams"; analogous competitions | Medium (metric: Low–Medium) |
| H4 | Teams can iterate against organiser baseline bots before the final run | Common practice | Low–Medium |
| H5 | Sealed-bid auctions: first-price or simple multi-unit (uniform / pay-as-bid), not combinatorial | Easy to implement | Low–Medium |
| H6 | Synthetic parametric demand (linear, logit or segment mixture), possibly with shocks | All analogous simulators | Medium |
| H7 | Per-decision limits of seconds or less | DPC: 0.2 s mean, 5 s max; TAC SCM: 15 s | Low–Medium |
| H8 | First edition | Nothing earlier found | Medium |

## 3. Possible scoring systems → how the agent adapts

| Scoring | Setting |
|---|---|
| Cumulative profit / final cash (most likely) | `objective: profit` or `cash` in the spec |
| Final company valuation (cash + inventory at some value) | `objective: valuation` and `salvage` per product / resource |
| Revenue or revenue share (DPC 2017) | `objective: revenue` |
| Market share / units | `objective: market_share` or profile `market_share` |
| Median or rank over many runs (SCML) | profile `conservative` (lower variance) |
| Weighted mix, penalties | `objective: {w_profit, w_revenue, w_units, w_cash}`; holding costs for penalties |

## 4. Expected interface

The closest public precedent (Dynamic Pricing Competition) calls one Python function per period and hands the agent's state back: `p(...) -> (price, information_dump)`, under 0.2 s mean / 5 s max, 1 CPU, 512 MB. `bizagent.api.decide(obs, memory)` mirrors that pattern; `CompetitionAgent.act(obs)` covers object-style SDKs; `main.py serve` covers HTTP.

## 5. Mathematical formulation

Indices: products j, resources r, rounds t = 1..T, firms i (0 = us).

- **State** s_t = (cash C_t, raw stock I^raw_{t,r}, finished stock I^fin_{t,j}, history) plus beliefs over demand parameters and rival behaviour.
- **Decisions** bids {(r, q, b)}, production x_{t,j} ≥ 0, prices p_{t,j} ∈ [p_min, p_max].
- **Dynamics**
  - won quantity w_{t,r} = Auction_r(our bids, rival bids); payment Σ b·w (first price) or clearing price × w (uniform / second price);
  - Σ_j A_{jr} x_{t,j} ≤ I^raw_{t,r} (+ w_{t,r} if delivery is immediate); Σ_j κ_j x_{t,j} ≤ K;
  - sales S_{t,j} = min(I^fin_{t,j} + x_{t,j}, D_{t,j}(p_t, p^{rivals}_t; θ_t, ε_t));
  - C_{t+1} = C_t + Σ_j p_{t,j} S_{t,j} − payments − Σ_j c_j x_{t,j} − holding; C_t ≥ 0.
- **Objective** max E[Score], default Score = C_{T+1} − C_1 + terminal value of inventory.
- **Unknown to us** demand model and parameters (possibly drifting), rival costs, stocks, cash and bids.

**Decomposition used by the agent** (rolling horizon, certainty equivalent). Each round one LP chooses sales volume, production and input use on a concave revenue curve built from the demand estimate. Two prices link the rounds: ω_r, what a unit of raw material kept in stock is worth (its expected replacement price), and μ_j, what an unsold finished unit is worth. Extra raw material is valued by re-solving the LP with more of it; that value schedule is turned into bids according to the auction format.

## 6. Plausible scenarios and coverage

| Scenario | Covered by |
|---|---|
| One product, one resource, sealed first-price, price competition | default path |
| Several products with recipes, multi-unit uniform-price | LP valuation with other inputs bought at expected prices; later tranches shaded |
| Second-price / English | truthful bids |
| Posted supplier price, no auction | `auction_format: posted`: buy while value > price |
| Fixed stock, selling season | LP with stock, no lots |
| One decision per round or two (bid, then sell) | `phase`: `all` / `bid` / `sell` |
| Delivery next round | `lead_time` |
| Score = share or revenue | objective weights |
| Demand shocks, changing elasticity | forgetting in the demand model |

Not covered without new code: bilateral negotiation (SCML-style), reverse auctions for customer orders (TAC SCM-style), combinatorial lots. Each would need a new module, not a re-write of the core.

## 7. What won the closest competitions

- **2017 INFORMS Dynamic Pricing Challenge:** a logit demand model plus line search won overall; a greedy rule that follows the cheapest rival "proves very difficult to outperform"; bandits with heavy exploration finished at the bottom; no algorithm dominated in all settings.
- **ANAC Supply Chain Management League 2025:** risk-averse heuristics with cost floors won both tracks although an RL environment was provided; "robustness and cost-containment often outperform complex opponent modeling".
- **TAC SCM (TacTex-06):** prediction of demand and winning prices plus expected-profit optimisation; bids maximise P(win | price) × (price − cost).
- None of the reviewed winners used end-to-end deep RL.

## 8. Critical risks

| Risk | Mitigation in this repository |
|---|---|
| Only ~4 hours of coding | adapter-only integration, minute plan in `HACKATHON_ADAPTATION_GUIDE.md` |
| Rules differ from assumptions | format-agnostic core, configuration, three interface styles |
| Overbidding | bids never exceed LP value; cash reserve; spend cap |
| Price wars | price floor at replacement cost; follower mode has the same floor |
| Over-exploration | Thompson sampling only, small early price variation |
| Crashes / timeouts | rule-based fallback, greedy fallback for the LP, lenient parsing |
| Rule compliance (pre-built code, AI tools) | unknown: ask at the briefing, first question |
| Overfitting to our simulator | train / validation / test split, held-out demand family and opponents |

## 9. Questions for the briefing (14:30)

1. What exactly is scored, and is the ranking per match, aggregated, or a median?
2. Auction format, units, reserve prices; what is revealed afterwards?
3. Horizon: known? Several games? Does state persist between games?
4. What do we see of competitors (prices, sales, stock)?
5. Inventory rules: holding cost, perishability, value at the end; borrowing and bankruptcy.
6. Time limit per decision, where the code runs, allowed libraries, how and when to submit.
7. **Are pre-written code, AI assistants, internet access and external APIs allowed?**
8. Is there a local simulator, practice server or baseline bots?
