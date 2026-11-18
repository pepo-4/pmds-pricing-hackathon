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
