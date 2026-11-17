import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from env.common import load_config
from env.rollout import run_episode
from env.so101_env import BowlEnv
from expert.policy import ExpertPolicy


@pytest.fixture(scope="module")
def env():
    return BowlEnv(load_config())


def test_ik_accuracy(env):
    for target in [[0.15, -0.05, 0.03], [0.22, 0.0, 0.10], [0.10, 0.15, 0.08], [0.20, -0.12, 0.05]]:
        q, ok, pe, re = env.ik.solve(target)
        assert ok and pe < 0.002 and re < 0.05
        p, R = env.ik.fk(q)
        assert np.linalg.norm(p - target) < 0.002
        assert R[2, 2] > 0.99  # fingers point straight down


def test_unreachable_rejected(env):
    env.reset(0)
    for tgt in [{"x": 0.6, "y": 0.0, "z": 0.05}, {"x": 0.15, "y": 0.0, "z": 0.4}, {"x": 0.04, "y": 0.0, "z": 0.05}]:
        obs, done, info = env.step({"action": "move_to", **tgt})
        assert info["result"] == "rejected_unreachable"
        assert obs["last_result"] == "rejected_unreachable"


def test_reset_deterministic(env):
    a = env.reset(123)
    q1 = env.d.qpos.copy()
    b = env.reset(123)
    assert a == b and np.array_equal(q1, env.d.qpos)
    c = env.reset(124)
    assert c["bowl"] != a["bowl"]


def test_tracking(env):
    env.reset(0)
    env.step({"action": "open_gripper"})
    obs, _, info = env.step({"action": "move_to", "x": 0.16, "y": -0.02, "z": 0.10})
    assert info["result"] == "success"
    assert abs(obs["ee"]["x"] - 0.16) < 0.003 and abs(obs["ee"]["z"] - 0.10) < 0.003


def test_success_check_positive_and_negative(env):
    pol = ExpertPolicy(env.cfg)
    ep = run_episode(env, pol, 3)
    assert ep["success"] and ep["success_detail"]["in_region"] and ep["success_detail"]["upright"]
    # declaring done immediately must fail (bowl still on the floor)
    env.reset(3)
    _, done, info = env.step({"action": "done"})
    assert done and not info["success"] and info["failure_type"] == "failed_grasp"


def test_step_cap(env):
    env.reset(5)
    done = False
    n = 0
    while not done:
        _, done, info = env.step({"action": "open_gripper"})
        n += 1
    assert n == env.cfg["env"]["max_decision_steps"] and info["failure_type"] in ("step_limit", "failed_grasp")
