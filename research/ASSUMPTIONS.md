# Assumptions register

Everything below is an assumption made because the real rules are unpublished. Each has a place where it is changed when the truth is known.

## About the competition

| # | Assumption | If wrong | Where to change |
|---|---|---|---|
| A1 | Discrete rounds; each round has an auction for inputs, production and price setting | A different loop (negotiation, customer RFQs) needs a new module | new adapter + module |
| A2 | Inputs are bought in sealed-bid auctions (first price, uniform / second price) or at a posted price | Other formats need a new rule in `procurement/bidding.py::shade` | `auction_format` per resource |
| A3 | Several units per lot; bids are (quantity, unit price) pairs, several per resource allowed | If one bid per resource: the adapter keeps only the best tranche | `from_action` |
| A4 | One price per product per round; demand falls with our price and rises with rivals' prices | A non-monotone demand rule would mislead the model | — |
| A5 | Score is profit-like by default | Set the objective | `GameSpec.objective` |
| A6 | Our own sales and auction outcomes are observed each round; rival prices and clearing prices may or may not be | Handled: both are optional | — |
| A7 | No borrowing; cash cannot go below zero | If borrowing is allowed the cash limits are merely conservative | `cash_reserve_frac`, `max_spend_frac` |
| A8 | Quantities may be fractional | Set integer rounding | `integer_quantities` / adapter |
| A9 | Decision time limit of at least ~0.2 s | Mean is ~10 ms here; `solver: greedy` and fewer tranches if it must be faster | config |
| A10 | Pre-written code is allowed | **Unknown. Ask first.** If not allowed, this repository is study material only | — |

## About the simulator (all parameter ranges are ours, not sourced)

| # | Assumption |
|---|---|
| S1 | 3–8 firms, 12–60 rounds, 1–3 products, 1–3 resources, recipes with 1–2 inputs of 1 or 2 units |
| S2 | Reference price P0 ~ U(20, 80); production cost 5–25% of P0; market size 20–60 units per firm per round at P0 |
| S3 | Four demand families: linear (own slope 1–3, cross 0.5–3 in relative terms), logit, customer segments (structure from the 2017 Dynamic Pricing Challenge, parameters ours), isoelastic (test only) |
| S4 | Input supply = 0.45–0.7 (scarce), 0.9–1.15 (balanced) or 1.5–2.2 (abundant) times what the market needs at P0 |
| S5 | Reserve price 5–20% of the unit margin per input (25–50% when the price is posted) |
| S6 | Dynamics: stable, trend, seasonal, a one-off demand shock, a one-off change in price sensitivity |
| S7 | Demand noise 5–20% per firm plus 2–8% market-wide; 10–50% of unserved demand moves to firms with stock |
| S8 | Starting cash 2–6 rounds of fair-share revenue; capacity limit in half of the scenarios |
| S9 | Price ceiling 1.6–5× P0 in 85% of scenarios, none otherwise. **The first-round price rule depends on this ceiling being meaningful**; set `init_price_frac` or give `ref_price` / `demand_hint` once the real numbers are known |
| S10 | Rival prices revealed in 85% of scenarios, clearing prices in 75%, horizon in 85%; hints (typical input price, rough demand) in 25% |
| S11 | Opponents are our own ten archetypes. Real teams may be stronger, weaker or simply different |

## What this means for the numbers in FINAL_REPORT.md

They measure the agent in a market we designed. They show that the code works, that it beats simpler rules under these assumptions, and which modules matter there. They do not predict the result on 18 November.
