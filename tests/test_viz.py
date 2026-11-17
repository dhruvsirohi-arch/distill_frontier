"""Rendering must never change simulation results."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from env.common import load_config
from env.rollout import run_episode
from env.so101_env import BowlEnv
from expert.policy import ExpertPolicy
from viz.viz import RecordViz


def _run(cfg, viz, seed):
    env = BowlEnv(cfg, viz=viz)
    ep = run_episode(env, ExpertPolicy(cfg), seed, episode_id=f"det_{seed}")
    return ep, env.d.qpos.copy(), env.d.time


def test_viz_does_not_change_outcome(tmp_path):
    cfg = load_config()
    for seed in (7, 11):
        ep0, q0, t0 = _run(cfg, None, seed)
        viz = RecordViz(cfg, "test_determinism", lowres=True, out_root=str(tmp_path))
        ep1, q1, t1 = _run(cfg, viz, seed)
        assert Path(viz.last_path).exists()
        assert ep0["steps"] == ep1["steps"]
        assert ep0["success"] == ep1["success"]
        assert np.array_equal(q0, q1) and t0 == t1
