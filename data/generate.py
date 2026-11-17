"""Generate expert demonstrations in parallel and save ALL episodes (successes + failures) as raw JSONL.

One JSONL row per decision step, carrying: episode_id, seed, step, observation, history, action,
result, final success/failure type, variation parameters and episode metrics.

  .venv/bin/python -m data.generate                    # ~2000 episodes on training seeds
  .venv/bin/python -m data.generate --seeds 20001 20007 --out outputs/data/raw_topup_episodes.jsonl
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from env.common import REPO, get_logger, load_config, set_seeds, write_json

_W = {}


def _init_worker(cfg, record_every, stage):
    from env.so101_env import BowlEnv
    from expert.policy import ExpertPolicy
    from viz.viz import RecordViz

    viz = RecordViz(cfg, stage) if record_every > 0 else None
    _W.update(env=BowlEnv(cfg, viz=viz), pol=ExpertPolicy(cfg), viz=viz, record_every=record_every)


def _run_one(job):
    from env.rollout import run_episode

    idx, seed, prefix = job
    viz = _W["viz"]
    if viz is not None:
        viz.enabled = idx % _W["record_every"] == 0
    ep = run_episode(_W["env"], _W["pol"], seed, episode_id=f"{prefix}{seed:06d}")
    ep["video"] = viz.last_path if (viz is not None and viz.enabled) else None
    return ep


def episode_rows(ep: dict):
    for st in ep["steps"]:
        yield {
            "episode_id": ep["episode_id"], "seed": ep["seed"], "step": st["step"],
            "observation": st["observation"], "history": st["history"], "action": st["action"],
            "result": st["result"], "final_success": ep["success"], "failure_type": ep["failure_type"],
            "n_steps": ep["n_steps"], "variation": ep["variation"], "spawn": ep["spawn"],
            "metrics": ep["metrics"], "success_detail": ep["success_detail"], "video": ep.get("video"),
        }


def generate(cfg, seeds, out_path, workers, record_every, stage, prefix="ep", log=None):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    jobs = [(i, int(s), prefix) for i, s in enumerate(seeds)]
    t0 = time.time()
    n_ok = 0
    eps_meta = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init_worker, initargs=(cfg, record_every, stage)) as pool, \
            open(out_path, "w") as f:
        for k, ep in enumerate(pool.imap(_run_one, jobs, chunksize=4)):
            for row in episode_rows(ep):
                f.write(json.dumps(row) + "\n")
            n_ok += ep["success"]
            eps_meta.append((ep["success"], ep["failure_type"]))
            if log and (k + 1) % 200 == 0:
                log.info(f"{k + 1}/{len(jobs)} episodes, success so far {n_ok / (k + 1):.1%}, "
                         f"{time.time() - t0:.0f}s")
    dt = time.time() - t0
    fails = {}
    for ok, ft in eps_meta:
        if not ok:
            fails[ft] = fails.get(ft, 0) + 1
    return {"episodes": len(jobs), "successes": n_ok, "success_rate": n_ok / max(1, len(jobs)),
            "failures": fails, "wall_time_s": round(dt, 1), "workers": workers, "out": str(out_path)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--n", type=int, default=None, help="number of training episodes")
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="explicit seed list (top-up)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--record-every", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    log = get_logger("generate", cfg)
    set_seeds(cfg["seeds"]["global"])
    sc = cfg["seeds"]
    if args.seeds:
        seeds = args.seeds
    else:
        lo, hi = sc["train_range"]
        n = args.n or sc["num_train_episodes"]
        seeds = sorted(np.random.default_rng(sc["global"]).choice(np.arange(lo, hi), n, replace=False).tolist())
    out = args.out or str(REPO / cfg["paths"]["raw_jsonl"])
    workers = args.workers or cfg["datagen"]["workers"]
    rec = cfg["datagen"]["record_every"] if args.record_every is None else args.record_every
    # workers render MP4s when rec > 0: pick the GL backend (EGL on headless Linux) before they spawn,
    # so they inherit MUJOCO_GL; otherwise mujoco defaults to GLFW and fails without a display
    from viz.viz import setup_gl

    setup_gl("record" if rec > 0 else "none")
    log.info(f"generating {len(seeds)} episodes with {workers} workers, recording every {rec}th -> {out}")
    summary = generate(cfg, seeds, out, workers, rec, stage="datagen", log=log)
    write_json(Path(out).with_suffix(".summary.json"), summary)
    log.info("datagen summary: " + json.dumps(summary))


if __name__ == "__main__":
    main()
