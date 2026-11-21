"""Phase 2 calibration at the 30 Hz frame rate: schedule oracle and the event reflex with a
fixed reaction latency (no model calls).

    python -m ringsim.p2.calibrate --out outputs/ringsim2/calibration.json
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

LATENCIES = (0.0, 1 / 30, 2 / 30, 0.1, 4 / 30, 5 / 30, 0.2, 0.3, 0.5, 1.0)


def _run(args):
    kind, lat, seed = args
    os.environ.setdefault("MUJOCO_GL", "egl")
    from ..baselines import Reactive, ScheduleOracle
    from ..game import Game
    from .world import FRAME, PHYS
    g = Game(render=False, tick=FRAME, phys_dt=PHYS)
    g.reset(seed)
    ctl = ScheduleOracle(g) if kind == "oracle" else Reactive(g, latency=lat)
    while not g.done:
        g.step(ctl.act(g), observe=False)
    r = g.result()
    return kind, lat, r["caught"], r["drops"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--start", type=int, default=50000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    seeds = range(a.start, a.start + a.episodes)
    jobs = [("oracle", 0.0, s) for s in seeds] + [("event", L, s) for L in LATENCIES for s in seeds]
    agg = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for kind, lat, c, d in ex.map(_run, jobs, chunksize=4):
            key = "oracle" if kind == "oracle" else f"event@{lat:.4f}"
            v = agg.setdefault(key, {"caught": 0, "drops": 0, "latency_s": lat})
            v["caught"] += c
            v["drops"] += d
    for v in agg.values():
        v["catch_rate"] = round(v["caught"] / v["drops"], 4)
    out = {"fps": 30, "seeds": f"{a.start}-{a.start + a.episodes - 1}", "controllers": agg}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    for k, v in agg.items():
        print(k, v["caught"], v["drops"], v["catch_rate"])


if __name__ == "__main__":
    main()
