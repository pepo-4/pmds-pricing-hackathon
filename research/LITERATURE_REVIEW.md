# Literature review and method selection

Reading status is stated for every source: **[FULL]** full text read, **[PARTIAL]** a substantial part read, **[ABSTRACT]** abstract or search snippet only. Findings from [ABSTRACT] sources are the authors' claims as summarised there, not independently checked.

## 1. Analogous competitions (highest relevance)

### L1. Dynamic Pricing and Learning with Competition: Insights from the Dynamic Pricing Challenge at the 2017 INFORMS RM & Pricing Conference
van de Geer, den Boer, Bayliss, Currie, Ellina, Esders, Haensel, Lei, Maclean, Martinez-Sykora, Riseth, Ødegaard, Zachariades. J. Revenue & Pricing Management 18 (2019). https://arxiv.org/pdf/1804.03219 — **[PARTIAL: sections 1–3.3]**
- **Problem:** pricing with demand learning against unknown competitors.
- **Setup:** 8 algorithms; 5000 simulations of 28 duopolies plus one 8-firm oligopoly, 1000 periods each. Prices visible, own sales only, no inventory limits.
- **Hidden demand:** Poisson arrivals, λ ~ U(50,150); bargain hunters buy the lowest price if their willingness to pay (exponential, mean β ~ U(5,15)) exceeds it; loyal customers with mean u·β, u ~ U(1.5,2.0); the rest choose by a finite-mixture logit.
- **Scoring:** average of revenue share in the oligopoly and in the duopolies.
- **Findings:** a finite-mixture logit model with line search won overall; "None of the considered algorithms is able to dominate the others in all settings"; "A greedy algorithm that follows the lowest-priced competitor in a tit-for-tat fashion proves very difficult to outperform"; "Ignoring competition is increasingly harmful when competition is more fierce"; "Too much exploration hurts performance significantly" (ε = 0.2 bandits in the bottom four).
- **Limits:** stationary demand, no procurement, 8 entrants.
- **Used here:** the segment mixture is one simulator demand family; the greedy follower is an opponent, our V1 baseline and our fallback; exploration is small and early.

### L2. Dynamic Pricing Competition website
Haensel, Koole, van de Geer, Kleinschmidt, Winter. https://www.dynamic-pricing-competition.com/ — **[FULL]**
- Function interface `p(...) -> (price, information_dump)`; limits 0.2 s mean, 5 s max, 1 CPU, 512 MB; Docker with Python 3.12 and Anaconda.
- **Used here:** the function-style entry point and the runtime target.

### L3. An Autonomous Agent for Supply Chain Management (TacTex-06)
Pardoe & Stone, 2009. https://www.cs.utexas.edu/~TacTex/dpardoe/papers/TacTexChapter.pdf — **[PARTIAL: sections 1–4.3.3]**
- **Game:** 6 agents, 220 days, 15 s per day; procure components, assemble, win customer orders in a first-price reverse auction.
- **Method:** Bayesian demand tracking; a particle filter over winning prices; bids maximise expected profit = P(order | price) × (price − cost); a greedy scheduler chosen over an LP because "the greedy approach is nearly optimal".
- **Used here:** win-probability bidding; replacement-cost valuation; greedy fallback planner.

### L4. TacTex-05: A Champion Supply Chain Management Agent
Pardoe & Stone, AAAI 2006. https://cdn.aaai.org/AAAI/2006/AAAI06-234.pdf — **[ABSTRACT]**. Predictive planning and an explicit end-game strategy.

### L5. The Automated Negotiating Agents Competition (ANAC) 2025: Challenges and Results
Aydoğan, Baarslag, Florijn, Fujita, Jonker, Mohammad. arXiv 2604.13914 (2026). https://arxiv.org/html/2604.13914v1 — **[FULL]**
- SCML-Standard won by AS0, a "risk-balanced production-first strategy" with "price floors using expected manufacturing costs and storage fees"; OneShot track a three-way tie of cost-averse heuristics.
- "robustness and cost-containment often outperform complex opponent modeling"; winners "prioritized domain-specific heuristics over complex opponent modeling (for the second year in a row)", although a Gymnasium / PettingZoo RL environment was provided.
- **Used here:** cost floors and cash limits as first-class modules; no deep RL.

### L6. SCML specifications 2019–2025 — **[ABSTRACT]**
https://scml.cs.brown.edu/files/scml/y2025/overview2025.pdf. Scores by median (2020) or truncated mean: consistency is rewarded. **Used here:** worst-decile and IQM in every table.

### L7. Power TAC champions (TacTex'13; AgentUDE) — **[ABSTRACT]**
https://www.ifaamas.org/Proceedings/aamas2014/aamas/p1447.pdf. AgentUDE won 2014 by exploiting a fee rule. **Lesson:** read the scoring rules for quirks before tuning anything.

## 2. Dynamic pricing and demand modelling

### L8. On the (Surprising) Sufficiency of Linear Models for Dynamic Pricing with Demand Learning
Besbes & Zeevi, Management Science 61(4), 2015. https://pubsonline.informs.org/doi/10.1287/mnsc.2014.2031 — **[ABSTRACT]**
- The revenue loss from pricing with a misspecified linear demand model "can be made to be essentially negligible" under general conditions.
- **Used here:** the demand model is linear in own and rival price, refitted with forgetting.

### L9. Online Network Revenue Management Using Thompson Sampling
Ferreira, Simchi-Levi & Wang, Operations Research 66(6), 2018. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2588730 — **[ABSTRACT]**
- Sample demand parameters from the posterior, then solve an LP with inventory constraints.
- **Used here:** exactly this structure — a Thompson sample feeds the planning LP each round.

### L10. den Boer, Dynamic pricing and learning (survey, 2015); Tracking the market (EJOR 2015) — **[ABSTRACT]**
https://www.sciencedirect.com/science/article/abs/pii/S0377221715005925. Forgetting estimators for changing markets. **Used here:** forgetting factor 0.95, decaying towards the prior.

### L11. Artificial Intelligence, Algorithmic Pricing, and Collusion
Calvano, Calzolari, Denicolò & Pastorello, AER 110(10), 2020. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3304991 — **[ABSTRACT]**
- Q-learning in a logit Bertrand game learns supra-competitive prices, but "the number of repetitions required for completing the learning is typically high, on the order of hundreds of thousands".
- **Used here:** logit Bertrand is a simulator family; Q-learning is far too slow for a game of tens of rounds.

### L12. Bayesian Optimization for Dynamic Pricing and Learning
Anand, Agrawal & Bodas, arXiv 2510.12447 (DPC 2024 duopoly winners) — **[ABSTRACT]**. Bayesian optimisation reported to beat several RL algorithms on revenue. Not implemented here: one more model to tune, with no clear gain over the parametric model in a multi-input game.

## 3. Bandits and online learning

### L13. Pricing bandits from Politecnico di Milano
Trovò, Paladino, Restelli & Gatti (2018); Mussi, Genalti, Trovò, Nuara, Gatti & Restelli, KDD 2022. https://marcomussi.github.io/papers/kdd2022/paper.pdf — **[ABSTRACT]**
- Exploit the fact that demand decreases in price to tighten bandit confidence bounds.
- **Used here:** the fitted slope is always projected to be negative and the cross effect smaller than the own effect; demand curves are made monotone before planning.

### L14. Bandits with Knapsacks
Badanidiyuru, Kleinberg & Slivkins, J. ACM 2018. https://arxiv.org/pdf/1305.2545 — **[ABSTRACT]**. Pricing with limited supply as a bandit with resource constraints. **Used here:** the primal-dual idea only (resource prices ω and μ), not the algorithm.

### L15. On Upper-Confidence Bound Policies for Switching Bandit Problems
Garivier & Moulines, ALT 2011. https://link.springer.com/chapter/10.1007/978-3-642-24412-4_16 — **[ABSTRACT]**. Discounted and sliding-window UCB for changing rewards. **Used here:** discounting in the meta-selector and the contextual bandit.

## 4. Auctions and bidding

### L16. Learning to bid in repeated first-price auctions
Han, Zhou, Flores, Ordentlich & Weissman, arXiv 2007.04568; Han, Zhou & Weissman, arXiv 2003.09795 — **[ABSTRACT]**
- When the winning price is revealed, one observation tells you the outcome of every bid you could have placed.
- **Used here:** an exponentially weighted kernel estimate of the clearing-price distribution; bid b* = argmax (v − b) · P(clearing ≤ b).

### L17. Learning in Repeated Auctions with Budgets
Balseiro & Gur, Management Science 65(9), 2019 — **[ABSTRACT]**. Adaptive pacing of spend. **Used here:** a simpler per-round spend cap and cash reserve; full pacing was not implemented.

### L18. Bidding for Customer Orders in TAC SCM
Pardoe & Stone, AMEC 2004. https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/AMEC04-tac.pdf — **[ABSTRACT]**. Learned win-probability models beat fixed heuristics.

## 5. Evaluation

### L19. Deep RL at the Edge of the Statistical Precipice
Agarwal, Schwarzer, Castro, Courville & Bellemare, NeurIPS 2021. https://arxiv.org/abs/2108.13264 — **[ABSTRACT]**. Report interval estimates, the interquartile mean and performance profiles. **Used here:** bootstrap intervals, IQM, paired differences, Holm-corrected Wilcoxon tests.

## 6. Tools (general knowledge, not re-verified in this research)

SciPy `linprog` with the HiGHS backend solves these LPs (tens of variables) in about a millisecond and needs no extra dependency. OR-Tools is a sound alternative for integer recipes. CVXPY and Pyomo add modelling convenience and import time; they are not used.

## 7. Method comparison

Ratings are judgements from the evidence above for this hackathon, not measurements. ★ poor … ★★★★★ excellent; for implementation complexity ★ means hard.

| Method | Expected performance | Data needs | Sample efficiency | Compute | Implementation | Robustness | Generalisation | Adaptability | Fit for 5 hours |
|---|---|---|---|---|---|---|---|---|---|
| Rules (cost-plus, follow / undercut with floors) | ★★★ | ★★★★★ | ★★★★★ | ★★★★★ | ★★★★★ | ★★★★ | ★★★★ | ★★ | ★★★★★ |
| Parametric demand + optimisation | ★★★★ | ★★★ | ★★★★ | ★★★★★ | ★★★★ | ★★★ | ★★★★ | ★★★ | ★★★★★ |
| Bayesian / Thompson pricing with forgetting | ★★★★ | ★★★ | ★★★★★ | ★★★★★ | ★★★★ | ★★★★ | ★★★★ | ★★★★ | ★★★★ |
| Contextual bandit over a price grid | ★★★ | ★★ | ★★★ | ★★★★ | ★★★ | ★★★ | ★★★ | ★★★★ | ★★★ |
| LP / MPC planning | ★★★★ | ★★★ | n/a | ★★★★ | ★★★ | ★★★★ | ★★★★★ | ★★★★ | ★★★★ |
| Bid shading from a win-probability curve | ★★★★ | ★★★ | ★★★★ | ★★★★★ | ★★★★ | ★★★★ | ★★★★ | ★★★★ | ★★★★★ |
| Model-free deep RL | ★★ | ★ | ★ | ★★ | ★ | ★★ | ★ | ★★ | ★ |
| Multi-agent RL / self-play | ★★ | ★ | ★ | ★ | ★ | ★★ | ★★ | ★★★ | ★ |
| Random-search tuning on our simulator | ★★★★ | ★★★★ | ★★ | ★★★ | ★★★★ | ★★★ | ★★★ | ★★ | ★★★★ before, ★★ on the day |
| Meta-strategy over a few policies | ★★★★ | ★★★ | ★★★ | ★★★★★ | ★★★★ | ★★★★ | ★★★★ | ★★★★★ | ★★★★ |

## 8. Selected combination

1. Hard limits on every action (price floor at replacement cost, bids never above value, cash reserve, spend cap).
2. Bayesian linear demand model with forgetting; Thompson sampling for exploration.
3. One LP per round for sales volume, production and price, with resource and stock values that link rounds.
4. Bids from LP marginal values: truthful under second / uniform price, shaded by a learned win-probability curve under first price.
5. Light rival model (price aggregate forecast, reaction estimate, price band).
6. A rule-based follower as baseline, as fallback and as the alternative arm of an optional meta-selector.

**Why this over simpler rules:** the follower rule was second in the DPC oligopoly but seventh in the duopolies, so it is not robust alone; it also cannot value inputs. **Why not more:** deep RL has no access to the real environment before the day and no reviewed winner used it. Whether each layer earns its place in our own simulator is measured in `FINAL_REPORT.md`, not assumed here.
