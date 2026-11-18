"""Mid-episode instruction switch: the hook a slower reasoning layer would use.

The episode starts with one instruction. A few ticks after the policy has lifted the bowl, the
instruction changes to name the other pad, and the episode is scored against the new instruction.
The policy never saw an instruction change during training.

  .venv/bin/python -m vla.switch_demo --ckpt outputs/vla/runs/m2/best.pt --n 30 --video 6
"""
from __future__ import annotations

import argparse
from collections import Counter

from viz.viz import setup_gl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--video", type=int, default=6)
    ap.add_argument("--delay", type=int, default=10, help="ticks between first grasp and the switch")
    ap.add_argument("--tag", default="switch")
    args = ap.parse_args()
    setup_gl("record")

    import imageio.v2 as iio2
    import torch

    from env.common import REPO, write_json
    from vla.common import load_vla_config
    from vla.env import TEMPLATES, TickEnv
    from vla.policy import load_policy, rollout

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = load_vla_config()
    cfg["vla"] = v = ck["vla_cfg"]
    assert len(v["pads"]) > 1, "needs a scene with two pads"
    pol = load_policy(args.ckpt)
    env = TickEnv(cfg, render=True)
    vid_dir = REPO / "outputs" / "vla" / "videos" / args.tag
    vid_dir.mkdir(parents=True, exist_ok=True)

    eps, fails = [], Counter()
    for k in range(args.n):
        seed = v["seeds"]["eval_start"] + 5000 + k
        st = {"grasp_tick": None, "switch_tick": None}

        def instruction_fn(tick, env):
            if st["grasp_tick"] is None and env.grasped(env.target_bowl):
                st["grasp_tick"] = tick
            if st["switch_tick"] is None and st["grasp_tick"] is not None and tick >= st["grasp_tick"] + args.delay:
                st["switch_tick"] = tick
                env.target_pad = 1 - env.target_pad            # score against the new instruction
                env.instruction = TEMPLATES[seed % len(TEMPLATES)].format(
                    bowl=env.bowl_names[env.target_bowl], pad=env.pad_names[env.target_pad])
            return env.instruction

        frames = [] if k < args.video else None
        info = rollout(env, pol, seed, frames=frames, instruction_fn=instruction_fn)
        ok = bool(info["success"]) and st["switch_tick"] is not None
        eps.append({"seed": seed, "success": ok, "failure_type": info["failure_type"], "ticks": env.stats.ticks,
                    "switch_tick": st["switch_tick"], "final_instruction": env.instruction})
        if not ok:
            fails[info["failure_type"] or "no_switch"] += 1
        if frames is not None:
            iio2.mimwrite(vid_dir / f"seed{seed}_{'ok' if ok else info['failure_type']}.mp4", frames,
                          fps=v["control_hz"], codec="libx264", macro_block_size=1, quality=6)
    res = {"checkpoint": args.ckpt, "n": args.n, "delay_ticks": args.delay,
           "success_rate": sum(e["success"] for e in eps) / args.n, "failures": dict(fails)}
    write_json(REPO / "outputs" / "vla" / f"results_{args.tag}.json", {"summary": res, "episodes": eps})
    print(res)


if __name__ == "__main__":
    main()
