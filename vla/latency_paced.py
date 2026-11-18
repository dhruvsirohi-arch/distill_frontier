"""Does real-time pacing change inference latency? (adds "paced" to latency.json)

The main benchmark runs inferences back to back. A deployed loop waits for the next 40 ms tick,
and how it waits matters: sleeping lets the CPU core idle, and the next inference starts cold.

  .venv/bin/python -m vla.latency_paced --ckpt outputs/vla/runs/m2_early/best.pt
"""
from __future__ import annotations

import argparse
import json
import time

from viz.viz import setup_gl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/vla/runs/m2_early/best.pt")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--out", default="outputs/vla/latency.json")
    args = ap.parse_args()
    setup_gl("record")
    import numpy as np
    import torch

    from env.common import REPO, write_json
    from vla.common import load_vla_config
    from vla.env import TickEnv
    from vla.latency import pct
    from vla.policy import load_policy

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = load_vla_config()
    cfg["vla"] = v = ck["vla_cfg"]
    env = TickEnv(cfg, render=True)
    dt = 1.0 / v["control_hz"]
    seeds = [v["seeds"]["eval_start"] + 30000 + i for i in range(args.episodes + 1)]

    def run(mode):
        pol = load_policy(args.ckpt)
        ok = 0
        for k, seed in enumerate(seeds):
            obs = env.reset(seed)
            pol.reset(seed)
            if k == 1:
                pol.latencies_ms.clear()
                ok = 0
            done = False
            while not done:
                t0 = time.perf_counter()
                obs, done, info = env.step(pol.act(obs))
                if mode == "sleep":
                    time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
                elif mode == "busy_wait":
                    while time.perf_counter() - t0 < dt:
                        pass
                elif mode == "sleep_then_spin_5ms":
                    time.sleep(max(0.0, dt - 0.005 - (time.perf_counter() - t0)))
                    while time.perf_counter() - t0 < dt:
                        pass
            ok += bool(info["success"])
        return {"n": args.episodes, "success_rate": ok / args.episodes, "inference_ms": pct(np.array(pol.latencies_ms))}

    out = {}
    for mode in ("unpaced", "sleep", "sleep_then_spin_5ms", "busy_wait"):
        out[mode] = run(mode)
        print(mode, out[mode]["success_rate"], {k: round(out[mode]["inference_ms"][k], 2) for k in ("p50", "p90", "p99", "max")}, flush=True)
    path = REPO / args.out
    res = json.loads(path.read_text())
    res["paced"] = out
    write_json(path, res)


if __name__ == "__main__":
    main()
