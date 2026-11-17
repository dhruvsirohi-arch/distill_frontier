"""Run the tick-level expert and report success. `--video N` saves MP4s of the first N episodes.

  .venv/bin/python -m vla.run_expert --n 200
"""
from __future__ import annotations

import argparse
import time
from collections import Counter

import numpy as np

from viz.viz import setup_gl


def run_episode(env, pol, seed, frames=None, noise=None):
    obs = env.reset(seed)
    pol.reset(seed)
    done, info = False, {}
    phases = []
    while not done:
        a = pol.act()
        phases.append(pol.phase)
        if noise is not None:
            a = noise(a)
        obs, done, info = env.step(a)
        if frames is not None:
            im = env.render(["front", "scene", "wrist"], frames["renderer"])
            frames["list"].append(np.concatenate(list(im.values()), axis=1))
    return info, phases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--video", type=int, default=0)
    ap.add_argument("--config", default=None, help="override yaml, e.g. configs/vla_m2.yaml")
    ap.add_argument("--tag", default="expert")
    args = ap.parse_args()
    setup_gl("record")
    import mujoco

    from env.common import REPO, write_json
    from vla.common import load_vla_config
    from vla.env import TickEnv
    from vla.expert import TickExpert

    cfg = load_vla_config(args.config)
    start, n = cfg["vla"]["seeds"]["expert_check"]
    start, n = (args.start if args.start is not None else start), (args.n or n)
    env = TickEnv(cfg, render=False)
    pol = TickExpert(cfg, env)
    rend = mujoco.Renderer(env.m, 240, 320) if args.video else None
    out = REPO / "outputs" / "vla"
    fails, ticks, t0, ok = Counter(), [], time.time(), 0
    for k, seed in enumerate(range(start, start + n)):
        frames = {"renderer": rend, "list": []} if k < args.video else None
        info, phases = run_episode(env, pol, seed, frames)
        ok += info["success"]
        ticks.append(env.stats.ticks)
        if not info["success"]:
            fails[info["failure_type"]] += 1
            print(f"seed {seed}: FAIL {info['failure_type']} last_phase={phases[-1]} {env.stats.detail}", flush=True)
        if frames is not None:
            import imageio.v2 as iio2
            (out / "videos" / args.tag).mkdir(parents=True, exist_ok=True)
            iio2.mimwrite(out / "videos" / args.tag / f"seed{seed:06d}.mp4", frames["list"],
                          fps=cfg["vla"]["control_hz"], codec="libx264", macro_block_size=1, quality=6)
    res = {"n": n, "start": start, "successes": ok, "success_rate": ok / n, "failures": dict(fails),
           "mean_ticks": float(np.mean(ticks)), "wall_time_s": round(time.time() - t0, 1)}
    print(res)
    write_json(out / f"{args.tag}_check.json", res)


if __name__ == "__main__":
    main()
