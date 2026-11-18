# bizagent — autonomous business agent for the AIRIC × PMDS AI Dynamic Pricing Hackathon

An agent that bids for raw materials, plans production and sets prices in a competitive market, plus a simulator, ten opponents and a tournament to test it. The competition rules are unpublished, so the strategic core is format-agnostic and the competition-specific part is one adapter file.

**Read first on the day:** [`HACKATHON_ADAPTATION_GUIDE.md`](HACKATHON_ADAPTATION_GUIDE.md).
**Results and limits:** [`FINAL_REPORT.md`](FINAL_REPORT.md). All numbers are from our own simulator.

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q                                  # 93 tests
python main.py simulate --agent v6_hybrid --seed 3   # one game against random opponents
python main.py act --spec examples/spec.json < examples/observation.json
python main.py discover examples/observation.json    # propose a field mapping for foreign data
python main.py serve --spec examples/spec.json       # HTTP: POST an observation to /act
```

```python
from bizagent.api import CompetitionAgent
agent = CompetitionAgent(spec=rules_dict, profile="balanced")
answer = agent.act(their_observation)        # {"bids": [...], "production": {...}, "prices": {...}}
```

## Experiments (reproducible; seeds are fixed)

```bash
python experiments/tournament.py --split test --games 300    # versions V1–V7, paired, held-out scenarios
python experiments/ablations.py --split valid --games 120    # full agent vs one module switched off
python experiments/tune.py --configs 24 --games 60           # random search on train, checked on validation
python experiments/league.py --games 60                      # all versions in the same market
python experiments/compare.py --split valid --games 100 --variant "base:" --variant "x:win_bias=1.5"
```

Results are written to `experiments/results/` as CSV (one row per game) and Markdown tables.

## Layout

```
main.py                      CLI: act | serve | simulate | discover | dump-config
bizagent/api.py              CompetitionAgent, decide()
bizagent/core/               schemas, config, decision engine
bizagent/market/             demand model, competitor model
bizagent/production/         LP planner + greedy fallback
bizagent/procurement/        clearing-price model and bid rules
bizagent/pricing/            meta-selector and contextual bandit
bizagent/risk/               final safety check on every action
bizagent/agents/             agent interface, rule baseline, versions V1–V7
bizagent/adapters/           generic adapter, interface discovery, competition_adapter.py (edit on the day)
bizagent/simulation/         market, auctions, scenarios, ten opponents
bizagent/evaluation/         paired tournament and statistics
config/                      default.yaml and strategy profiles
experiments/                 tournament, ablations, tuning, league, compare
research/                    competition analysis, literature review, architecture, assumptions
tests/                       unit, simulator, robustness and integration tests
examples/                    a spec, an observation and the action the agent returned for it
```

## Strategy profiles

`balanced` (default), `conservative`, `aggressive`, `profit_max`, `market_share`, `adaptive`, `lean`. Select with `--profile NAME` or `CompetitionAgent(profile=NAME)`; a profile is a YAML file of overrides in `config/profiles/`.

## Docker

```bash
docker build -t bizagent . && docker run -p 8000:8000 bizagent
```

The Dockerfile was written but not built in the environment where this repository was produced.
