"""Scripted controllers on the exam seeds: is the task solvable, and how does it scale with latency?

    python -m ringsim.calibrate --episodes 100 --out outputs/ringsim/calibration.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path

LATENCIES = (0.0, 0.04, 0.08, 0.12, 0.16, 0.24, 0.5, 1.0, 4.0)


def run_one(args):
    name, seed = args
    os.environ.setdefault("MUJOCO_GL", "egl")
    from .baselines import Idle, Parked, Reactive, ScheduleOracle
    from .game import Game
    if name == "vision":
        game = Game(render=True)
        obs = game.reset(seed)
        pol = Reactive(game, latency=0.0, detector="vision")
        while not game.done:
            obs, _ = game.step(pol.act(game, obs))
        return name, game.result()
    game = Game(render=False)
    game.reset(seed)
    if name == "oracle":
        pol = ScheduleOracle(game)
    elif name.startswith("event@"):
        pol = Reactive(game, latency=float(name[6:]))
    elif name == "parked":
        pol = Parked(game)
    elif name == "parked_trap":
        pol = Parked(game)
        pol.trap = True
    else:
        pol = Idle(game)
    while not game.done:
        game.step(pol.act(game), observe=False)
    return name, game.result()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--start", type=int, default=50000)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    names = (["oracle", "vision", "parked", "parked_trap", "idle"]
             + [f"event@{L}" for L in LATENCIES])
    jobs = [(n, s) for n in names for s in range(args.start, args.start + args.episodes)]
    totals = {n: {"caught": 0, "drops": 0, "outcomes": {}} for n in names}
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for name, res in pool.imap_unordered(run_one, jobs, chunksize=2):
            t = totals[name]
            t["caught"] += res["caught"]
            t["drops"] += res["drops"]
            for e in res["events"]:
                t["outcomes"][e["outcome"]] = t["outcomes"].get(e["outcome"], 0) + 1
    for n, t in totals.items():
        t["catch_rate"] = round(t["caught"] / t["drops"], 4)
        print(f"{n:16s} {t['caught']:4d}/{t['drops']}  {t['catch_rate']:.1%}  {t['outcomes']}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"seeds": [args.start, args.start + args.episodes - 1],
                                          "controllers": totals}, indent=1))


if __name__ == "__main__":
    main()
