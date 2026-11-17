"""Generate expert demonstrations for the VLA: one HDF5 per successful episode.

  images/<cam>   T JPEG-encoded frames (observation before the action at that tick)
  proprio        T x 6 joint positions          action   T x 6 clean expert joint targets
  state/*        privileged signals for analysis (tcp, bowl, pad, phase)
  attrs          seed, instruction, task (json), perturbed

A fraction of episodes is collected with perturbed execution (vla/noise.py) so the data contains
recoveries. Failed episodes are not written; they are counted in manifest.json.

  .venv/bin/python -m vla.generate --n 1000
"""
from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from viz.viz import setup_gl

_W = {}


def _init_worker(cfg):
    from vla.env import TickEnv
    from vla.expert import TickExpert

    env = TickEnv(cfg, render=True)
    _W.update(cfg=cfg, env=env, pol=TickExpert(cfg, env))


def _jpeg(im, quality):
    from PIL import Image

    b = io.BytesIO()
    Image.fromarray(im).save(b, format="JPEG", quality=quality)
    return np.frombuffer(b.getvalue(), dtype=np.uint8)


def _run_one(job):
    import h5py

    from vla.expert import PHASES
    from vla.noise import BurstNoise

    seed, out_dir, perturb_frac, quality = job
    env, pol = _W["env"], _W["pol"]
    perturbed = bool(np.random.default_rng([seed, 17]).random() < perturb_frac)
    noise = BurstNoise(seed) if perturbed else None
    obs = env.reset(seed)
    pol.reset(seed)
    imgs = {c: [] for c in env.cams}
    rec = {k: [] for k in ("proprio", "action", "tcp", "bowl", "pad", "phase")}
    done = False
    while not done:
        a = pol.act()
        for c in env.cams:
            imgs[c].append(_jpeg(obs["images"][c], quality))
        rec["proprio"].append(obs["proprio"])
        rec["action"].append(a.astype(np.float32))
        rec["tcp"].append(env.tcp_pos())
        rec["bowl"].append(env.bowl_pos(env.target_bowl))
        rec["pad"].append(env.pad_pos(env.target_pad))
        rec["phase"].append(PHASES.index(pol.phase))
        obs, done, info = env.step(noise(a) if noise else a)
    meta = {"seed": seed, "success": bool(info["success"]), "failure_type": info["failure_type"],
            "ticks": env.stats.ticks, "perturbed": perturbed, "instruction": env.instruction,
            "bowl": env.bowl_names[env.target_bowl], "pad": env.pad_names[env.target_pad], "file": None}
    if info["success"]:
        path = Path(out_dir) / f"ep{seed:07d}.h5"
        with h5py.File(path, "w") as f:
            vl = h5py.vlen_dtype(np.uint8)
            for c in env.cams:
                ds = f.create_dataset(f"images/{c}", (len(imgs[c]),), dtype=vl)
                for i, b in enumerate(imgs[c]):
                    ds[i] = b
            f.create_dataset("proprio", data=np.array(rec["proprio"], np.float32))
            f.create_dataset("action", data=np.array(rec["action"], np.float32))
            for k in ("tcp", "bowl", "pad"):
                f.create_dataset(f"state/{k}", data=np.array(rec[k], np.float32))
            f.create_dataset("state/phase", data=np.array(rec["phase"], np.int8))
            f.attrs["seed"], f.attrs["instruction"], f.attrs["perturbed"] = seed, env.instruction, perturbed
            f.attrs["task"] = json.dumps(env.task)
        meta["file"] = path.name
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--perturb-frac", type=float, default=0.6)
    ap.add_argument("--quality", type=int, default=92)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    setup_gl("record")

    from env.common import REPO, write_json
    from vla.common import load_vla_config

    cfg = load_vla_config(args.config)
    start = cfg["vla"]["seeds"][f"{args.split}_start"]
    out = Path(args.out) if args.out else REPO / cfg["vla"]["paths"]["data"] / args.split
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(s, str(out), args.perturb_frac, args.quality) for s in range(start, start + args.n)]
    t0, metas = time.time(), []
    with mp.get_context("spawn").Pool(args.workers, initializer=_init_worker, initargs=(cfg,)) as pool:
        for k, meta in enumerate(pool.imap_unordered(_run_one, jobs, chunksize=2)):
            metas.append(meta)
            if (k + 1) % 100 == 0:
                ok = sum(m["success"] for m in metas)
                print(f"{k + 1}/{len(jobs)} episodes, {ok} ok, {time.time() - t0:.0f}s", flush=True)
    metas.sort(key=lambda m: m["seed"])
    ok = [m for m in metas if m["success"]]
    fails = {}
    for m in metas:
        if not m["success"]:
            fails[m["failure_type"]] = fails.get(m["failure_type"], 0) + 1
    summary = {"split": args.split, "episodes": len(metas), "kept": len(ok),
               "kept_perturbed": sum(m["perturbed"] for m in ok), "ticks": sum(m["ticks"] for m in ok),
               "failures": fails, "wall_time_s": round(time.time() - t0, 1), "perturb_frac": args.perturb_frac}
    write_json(out / "manifest.json", {"summary": summary, "episodes": metas})
    print(summary)


if __name__ == "__main__":
    main()
