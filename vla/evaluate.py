"""Benchmark a trained VLA in closed loop on held-out seeds.

Reports success vs the scripted expert (by failure type and pad speed), inference latency, and a
language test: the same scene with a different instruction must lead to a different outcome.

  .venv/bin/python -m vla.evaluate --ckpt outputs/vla/runs/m1/best.pt --n 100 --video 10
"""
from __future__ import annotations

import argparse
import time
from collections import Counter

from viz.viz import setup_gl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--video", type=int, default=10)
    ap.add_argument("--exec-horizon", type=int, default=None)
    ap.add_argument("--flow-steps", type=int, default=None)
    ap.add_argument("--tag", default="eval")
    ap.add_argument("--no-expert", action="store_true")
    args = ap.parse_args()
    setup_gl("record")

    import imageio.v2 as iio2
    import numpy as np
    import torch

    from env.common import REPO, write_json
    from vla.env import TEMPLATES, TickEnv
    from vla.expert import TickExpert
    from vla.policy import load_policy, rollout
    from vla.run_expert import run_episode

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    from vla.common import load_vla_config

    cfg = load_vla_config()
    cfg["vla"] = ck["vla_cfg"]
    v = cfg["vla"]
    pol = load_policy(args.ckpt, exec_horizon=args.exec_horizon, flow_steps=args.flow_steps)
    env = TickEnv(cfg, render=True)
    out = REPO / "outputs" / "vla"
    vid_dir = out / "videos" / args.tag
    vid_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(v["seeds"]["eval_start"], v["seeds"]["eval_start"] + args.n))

    # ---- warm up (first CUDA calls are slow and would pollute latency numbers)
    rollout(env, pol, seeds[0])
    pol.latencies_ms.clear()

    # ---- main closed-loop evaluation
    eps, fails, t0 = [], Counter(), time.time()
    for k, seed in enumerate(seeds):
        frames = [] if k < args.video else None
        info = rollout(env, pol, seed, frames=frames)
        t = env.task
        eps.append({"seed": seed, "success": bool(info["success"]), "failure_type": info["failure_type"],
                    "ticks": env.stats.ticks, "pad_speed": t["pads"][t["pad"]]["speed"],
                    "instruction": t["instruction"], "robot_env_ticks": env.stats.robot_env_ticks})
        if not info["success"]:
            fails[info["failure_type"]] += 1
        if frames is not None:
            tag = "ok" if info["success"] else info["failure_type"]
            iio2.mimwrite(vid_dir / f"seed{seed}_{tag}.mp4", frames, fps=v["control_hz"], codec="libx264",
                          macro_block_size=1, quality=6)
    wall = time.time() - t0
    lat = np.array(pol.latencies_ms)
    ok = np.array([e["success"] for e in eps])
    speeds = np.array([e["pad_speed"] for e in eps])
    lo, hi = v["pads"][0]["speed"]
    edges = np.linspace(lo, hi, 4)
    by_speed = {}
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (speeds >= a) & (speeds <= b)
        by_speed[f"{a * 100:.1f}-{b * 100:.1f} cm/s"] = {"n": int(sel.sum()), "success": float(ok[sel].mean()) if sel.any() else None}
    tick_ms = 1000.0 / v["control_hz"]
    res = {
        "checkpoint": str(args.ckpt), "train_step": ck["step"], "n": args.n,
        "success_rate": float(ok.mean()), "failures": dict(fails), "by_pad_speed": by_speed,
        "mean_ticks_success": float(np.mean([e["ticks"] for e in eps if e["success"]])) if ok.any() else None,
        "latency_ms": {"p50": float(np.percentile(lat, 50)), "p90": float(np.percentile(lat, 90)),
                       "p99": float(np.percentile(lat, 99)), "max": float(lat.max()), "n_inferences": int(len(lat))},
        "control": {"tick_ms": tick_ms, "exec_horizon": pol.exec_horizon, "flow_steps": pol.flow_steps,
                    "chunk": pol.model.chunk, "fits_in_tick": bool(np.percentile(lat, 99) < tick_ms),
                    "max_policy_hz_if_replanning_every_tick": float(1000.0 / np.percentile(lat, 99))},
        "params_M": sum(p.numel() for p in pol.model.parameters()) / 1e6,
        "wall_time_s": round(wall, 1),
    }

    # ---- expert on the same seeds
    if not args.no_expert:
        xenv = TickEnv(cfg, render=False)
        xp = TickExpert(cfg, xenv)
        xok, xticks = 0, []
        for seed in seeds:
            info, _ = run_episode(xenv, xp, seed)
            xok += info["success"]
            xticks.append(xenv.stats.ticks)
        res["expert"] = {"success_rate": xok / len(seeds), "mean_ticks": float(np.mean(xticks))}

    # ---- language test: same scene, the other bowl / pad named
    nb, npad = len(v["bowls"]), len(v["pads"])
    if nb * npad > 1:
        lang = Counter()
        n_lang = min(args.n, 50)
        for seed in seeds[:n_lang]:
            base = env.sample_task(seed)
            for bi in range(nb):
                for pi in range(npad):
                    if (bi, pi) == (base["bowl"], base["pad"]):
                        continue
                    task = dict(base, bowl=bi, pad=pi,
                                instruction=TEMPLATES[seed % len(TEMPLATES)].format(bowl=env.bowl_names[bi], pad=env.pad_names[pi]))
                    info = rollout(env, pol, seed, task=task)
                    lang["n"] += 1
                    lang["success"] += int(info["success"])
                    if not info["success"]:
                        lang[info["failure_type"]] += 1
        res["language_counterfactual"] = {**dict(lang), "success_rate": lang["success"] / max(1, lang["n"])}

    # ---- unseen phrasing
    para = ["could you put the {bowl} bowl onto the {pad} pad", "{bowl} bowl to the {pad} pad please",
            "grab the {bowl} bowl and drop it on the {pad} pad"]
    pok = 0
    n_para = min(args.n, 30)
    for k, seed in enumerate(seeds[:n_para]):
        base = env.sample_task(seed)
        task = dict(base, instruction=para[k % len(para)].format(bowl=env.bowl_names[base["bowl"]], pad=env.pad_names[base["pad"]]))
        pok += rollout(env, pol, seed, task=task)["success"]
    res["unseen_phrasing"] = {"n": n_para, "success_rate": pok / n_para}

    write_json(out / f"results_{args.tag}.json", {"summary": res, "episodes": eps})
    import json

    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
