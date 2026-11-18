"""Prompt the VLA and watch it act.

  # record a video of one instruction on a random scene
  .venv/bin/python -m vla.play --instruction "put the green bowl on the yellow pad" --seed 7

  # live MuJoCo window; type a new instruction in the terminal at any time (needs a desktop session)
  .venv/bin/python -m vla.play --live

The scene (bowl positions, pad motion) comes from --seed. The instruction is free text; if it names
a bowl colour and a pad colour the episode is scored against it, otherwise it just runs.
"""
from __future__ import annotations

import argparse
import os
import queue
import threading
import time

DEFAULT_CKPT = "outputs/vla/runs/m2_early/best.pt"


def parse_target(text: str, bowls, pads):
    """(bowl index, pad index) named in the text, or None. Looks for '<colour> bowl' / '<colour> pad'."""
    words = text.lower().replace(",", " ").replace(".", " ").split()

    def find(names, noun):
        for a, b in zip(words, words[1:]):
            if b.startswith(noun) and a in names:
                return names.index(a)
        return None

    bi, pi = find(bowls, "bowl"), find(pads, "pad")
    return None if bi is None or pi is None else (bi, pi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--instruction", default=None, help="free text; default: the instruction sampled for the seed")
    ap.add_argument("--seed", type=int, default=0, help="scene index (offset into held-out seeds)")
    ap.add_argument("--live", action="store_true", help="open a MuJoCo window and accept typed instructions")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--out", default="outputs/vla/videos/play")
    args = ap.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")        # policy cameras render offscreen on the GPU

    import imageio.v2 as iio2
    import mujoco
    import numpy as np
    import torch

    from env.common import REPO
    from vla.common import load_vla_config
    from vla.env import TickEnv, TickStats
    from vla.policy import load_policy

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = load_vla_config()
    cfg["vla"] = v = ck["vla_cfg"]
    pol = load_policy(args.ckpt)
    env = TickEnv(cfg, render=True)
    bowls, pads = env.bowl_names, env.pad_names
    print(f"bowls: {bowls}   pads: {pads}", flush=True)

    typed: queue.Queue[str] = queue.Queue()
    viewer = None
    if args.live:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(env.m, env.d)
        viewer.cam.lookat[:] = [0.14, 0.0, 0.05]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 0.75, 180, -35

        def reader():
            while True:
                try:
                    line = input().strip()
                except EOFError:
                    return
                if line:
                    typed.put(line)

        threading.Thread(target=reader, daemon=True).start()
        print("type an instruction and press enter at any time; 'n' = next scene, 'q' = quit", flush=True)

    def set_instruction(text):
        env.instruction = text
        tgt = parse_target(text, bowls, pads)
        if tgt:
            env.target_bowl, env.target_pad = tgt
        env.tick, env._hold, env.stats = 0, 0, TickStats(seed=env.seed)      # fresh time budget and score
        print(f"  instruction: {text!r}" + ("" if tgt else "   (no '<colour> bowl' + '<colour> pad' found: not scored)"), flush=True)
        return tgt is not None

    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    ep, quit_ = 0, False
    while not quit_ and (args.live or ep < args.episodes):
        seed = v["seeds"]["eval_start"] + 20000 + args.seed + ep
        obs = env.reset(seed)
        pol.reset(seed)
        print(f"scene {args.seed + ep}", flush=True)
        scored = set_instruction(args.instruction or env.instruction)
        frames, done, info, nxt = [], False, {}, False
        while not (nxt or quit_):
            t0 = time.perf_counter()
            while not typed.empty():
                line = typed.get()
                if line == "q":
                    quit_ = True
                elif line == "n":
                    nxt = True
                else:
                    scored = set_instruction(line)
                    done = False
            if viewer is not None and not viewer.is_running():
                quit_ = True
            if quit_ or nxt:
                break
            obs["instruction"] = env.instruction
            obs, fin, info = env.step(pol.act(obs))
            if fin and not done:
                done = True
                if env.stats.success and scored:
                    print(f"  success after {env.tick} ticks ({env.tick * env.dt:.1f} s)", flush=True)
                elif scored:
                    print(f"  failed: {info['failure_type']}", flush=True)
                else:
                    print("  finished (not scored)", flush=True)
                if not args.live:
                    break
                print("  type another instruction, 'n' for the next scene, or 'q'", flush=True)
            if viewer is not None:
                viewer.sync()
                time.sleep(max(0.0, env.dt - (time.perf_counter() - t0)))
            else:
                frames.append(np.concatenate([obs["images"][c] for c in pol.cams], axis=1))
        if frames:
            path = out_dir / f"scene{args.seed + ep}.mp4"
            iio2.mimwrite(path, frames, fps=v["control_hz"], codec="libx264", macro_block_size=1, quality=7)
            print(f"  video: {path}", flush=True)
        ep += 1
    if viewer is not None:
        viewer.close()
    lat = np.array(pol.latencies_ms[3:] or [0.0])
    print(f"inference p50 {np.percentile(lat, 50):.1f} ms over {len(lat)} calls")


if __name__ == "__main__":
    main()
