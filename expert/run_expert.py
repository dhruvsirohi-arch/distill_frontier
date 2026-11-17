"""Run the scripted expert on N seeds (default: configs seeds.expert_check) and report success rate.

  .venv/bin/mjpython -m expert.run_expert --viz live          # watch it (macOS needs mjpython)
  .venv/bin/python -m expert.run_expert --viz record --n 5    # record MP4s
"""
from __future__ import annotations

import argparse
import collections
import json
import time

from viz.viz import add_viz_args, make_viz, resolve_mode
from env.common import REPO, get_logger, load_config, set_seeds, write_json


def main():
    ap = add_viz_args(argparse.ArgumentParser())
    ap.add_argument("--config", default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--min-success", type=float, default=0.9)
    args = ap.parse_args()
    cfg = load_config(args.config)
    mode = resolve_mode(args, cfg)
    from env.rollout import run_episode
    from env.so101_env import BowlEnv
    from expert.policy import ExpertPolicy

    log = get_logger("expert_check", cfg)
    set_seeds(cfg["seeds"]["global"])
    start, n = cfg["seeds"]["expert_check"]
    start = args.start if args.start is not None else start
    n = args.n if args.n is not None else n
    viz = make_viz(mode, cfg, stage="expert_check")
    env = BowlEnv(cfg, viz=viz)
    pol = ExpertPolicy(cfg)
    t0 = time.time()
    eps = []
    for s in range(start, start + n):
        ep = run_episode(env, pol, s, episode_id=f"expert_seed{s}")
        eps.append(ep)
        log.info(f"seed {s}: {'OK ' if ep['success'] else 'FAIL'} {ep['failure_type']:<14} steps={ep['n_steps']}")
    if viz:
        viz.close()
    k = sum(e["success"] for e in eps)
    rate = k / max(1, n)
    summary = {"n": n, "start": start, "successes": k, "success_rate": rate,
               "failures": dict(collections.Counter(e["failure_type"] for e in eps if not e["success"])),
               "steps_hist": dict(collections.Counter(e["n_steps"] for e in eps)),
               "wall_time_s": round(time.time() - t0, 1)}
    write_json(REPO / "outputs" / "expert_check.json", summary)
    log.info("expert check: " + json.dumps(summary))
    if rate < args.min_success:
        log.error(f"expert success {rate:.0%} < {args.min_success:.0%}: fix env/IK/policy before continuing")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
