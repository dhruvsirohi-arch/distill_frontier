"""Closed-loop latency and success for precision / torch.compile variants (adds to latency.json).

  .venv/bin/python -m vla.latency_variants --ckpt outputs/vla/runs/m2_early/best.pt
"""
from __future__ import annotations

import argparse
import json
import time

from viz.viz import setup_gl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/vla/runs/m2_early/best.pt")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--out", default="outputs/vla/latency.json")
    args = ap.parse_args()
    setup_gl("record")
    import numpy as np
    import torch

    from env.common import REPO, write_json
    from vla.common import load_vla_config
    from vla.env import TickEnv
    from vla.latency import pct
    from vla.policy import load_policy, rollout

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = load_vla_config()
    cfg["vla"] = v = ck["vla_cfg"]
    env = TickEnv(cfg, render=True)
    seeds = list(range(v["seeds"]["eval_start"] + 30000, v["seeds"]["eval_start"] + 30000 + args.episodes))
    out = {}
    for name, amp, comp, steps in [("eager_bf16", True, False, 4), ("eager_fp32", False, False, 4),
                                   ("compiled_bf16", True, True, 4), ("compiled_fp32", False, True, 4),
                                   ("compiled_fp32_f2", False, True, 2), ("compiled_fp32_f1", False, True, 1)]:
        p = load_policy(args.ckpt, amp=amp, flow_steps=steps)
        t0 = time.perf_counter()
        if comp:
            m = p.model
            m.vision, m.encoder, m.decoder = torch.compile(m.vision), torch.compile(m.encoder), torch.compile(m.decoder)
        for s in seeds[:2]:
            rollout(env, p, s)
        warm_s = time.perf_counter() - t0
        p.latencies_ms.clear()
        ok = sum(bool(rollout(env, p, s)["success"]) for s in seeds)
        out[name] = {"n": args.episodes, "success_rate": ok / args.episodes, "flow_steps": steps, "warmup_s": warm_s,
                     "inference_ms": pct(np.array(p.latencies_ms))}
        print(name, out[name]["success_rate"], {k: round(out[name]["inference_ms"][k], 2) for k in ("p50", "p99", "max")}, flush=True)
    path = REPO / args.out
    res = json.loads(path.read_text())
    res["closed_loop_variants"] = out
    write_json(path, res)


if __name__ == "__main__":
    main()
