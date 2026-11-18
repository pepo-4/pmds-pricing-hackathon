#!/usr/bin/env python3
"""Single entry point.

  python main.py act --spec examples/spec.json --state state.pkl < examples/observation.json
  python main.py serve --spec examples/spec.json --port 8000        # POST /act with an observation
  python main.py simulate --agent v6_hybrid --seed 1
  python main.py discover examples/observation.json
  python main.py dump-config                                        # rewrite config/default.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def _load_json(path):
    if path in (None, "-"):
        return json.load(sys.stdin)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def cmd_act(args):
    from bizagent.api import CompetitionAgent
    if args.state and os.path.exists(args.state):
        agent = CompetitionAgent.load(args.state)
    else:
        agent = CompetitionAgent(spec=_load_json(args.spec) if args.spec else {},
                                 profile=args.profile, agent=args.agent, seed=args.seed)
    try:
        raw = _load_json(args.obs)
    except (json.JSONDecodeError, OSError) as exc:
        print(json.dumps({"bids": [], "production": {}, "prices": {}, "error": str(exc)}))
        return 1
    print(json.dumps(agent.act(raw)))
    if args.state:
        agent.save(args.state)
    return 0


def cmd_serve(args):
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from bizagent.api import CompetitionAgent
    agent = CompetitionAgent(spec=_load_json(args.spec) if args.spec else {}, profile=args.profile,
                             agent=args.agent, seed=args.seed)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                n = int(self.headers.get("Content-Length", "0"))
                raw = json.loads(self.rfile.read(n) or b"{}")
                body, code = json.dumps(agent.act(raw)).encode(), 200
            except Exception as exc:
                body = json.dumps({"bids": [], "production": {}, "prices": {}, "error": str(exc)}).encode()
                code = 400
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    print(f"listening on :{args.port} (POST an observation to /act)")
    HTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


def cmd_simulate(args):
    import numpy as np
    from bizagent.agents.registry import make_agent
    from bizagent.evaluation.tournament import lineup
    from bizagent.simulation.environment import Game
    from bizagent.simulation.opponents import OPPONENTS
    from bizagent.simulation.scenarios import make_scenario
    sc = make_scenario(args.seed)
    opp = lineup(args.seed, sc.tags["n_firms"] - 1, list(OPPONENTS))
    agents = [make_agent(args.agent, profile=args.profile)] + [make_agent(o) for o in opp]
    rows = Game(sc, agents).run()
    print(sc.name, json.dumps(sc.tags))
    for r in sorted(rows, key=lambda r: r["rank"]):
        print(f"  #{r['rank']} {r['agent']:14s} score={r['score']:10.0f} norm={r['score_norm']:+.3f} "
              f"revenue={r['revenue']:9.0f} input spend={r['spend']:8.0f} sold={r['units_sold']:7.0f} "
              f"unsold={r['fin_left']:5.0f} errors={r['errors'] + r['engine_errors']:.0f} "
              f"ms={r['ms_mean']:.1f}")
    return 0


def cmd_discover(args):
    from bizagent.adapters.discovery import report
    print(report(_load_json(args.sample)))
    return 0


def cmd_dump_config(args):
    import yaml
    from bizagent.core.config import CONFIG_DIR, StrategyConfig
    path = os.path.join(CONFIG_DIR, "default.yaml")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# Generated from bizagent/core/config.py (python main.py dump-config).\n"
                 "# Profiles in config/profiles/ override these values.\n")
        yaml.safe_dump(StrategyConfig().to_dict(), fh, sort_keys=False)
    print("wrote", path)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("act", "serve", "simulate"):
        p = sub.add_parser(name)
        p.add_argument("--agent", default="v6_hybrid")
        p.add_argument("--profile", default="balanced")
        p.add_argument("--seed", type=int, default=0)
        if name != "simulate":
            p.add_argument("--spec", default=None)
        if name == "act":
            p.add_argument("--obs", default="-")
            p.add_argument("--state", default=None)
        if name == "serve":
            p.add_argument("--port", type=int, default=8000)
    p = sub.add_parser("discover")
    p.add_argument("sample")
    sub.add_parser("dump-config")
    args = ap.parse_args()
    return {"act": cmd_act, "serve": cmd_serve, "simulate": cmd_simulate, "discover": cmd_discover,
            "dump-config": cmd_dump_config}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main() or 0)
