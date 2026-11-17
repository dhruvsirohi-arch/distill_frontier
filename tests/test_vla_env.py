import numpy as np
import pytest

from vla.common import load_vla_config
from vla.env import TickEnv
from vla.expert import TickExpert
from vla.run_expert import run_episode


@pytest.fixture(scope="module")
def cfg():
    return load_vla_config()


@pytest.fixture(scope="module")
def env(cfg):
    return TickEnv(cfg, render=False)


def hold(env):
    return np.concatenate([env.home_q, [env.v["gripper"]["open"]]])


def test_pad_ref_is_bounded_triangle_wave(env):
    env.reset(3)
    A, sp = env.v["pads"][0]["travel"], env.task["pads"][0]["speed"]
    ts = np.arange(0, 30, 0.002)
    s = np.array([env.pad_ref(0, t) for t in ts])
    assert abs(s[0] - env.task["pads"][0]["s0"]) < 1e-9
    assert s.min() >= -A - 1e-9 and s.max() <= A + 1e-9
    assert np.allclose(np.abs(np.diff(s)), sp * 0.002, atol=1e-6) or np.max(np.abs(np.diff(s))) <= sp * 0.002 + 1e-9
    assert s.max() > A - 1e-3 and s.min() < -A + 1e-3   # it does reverse at both ends


def test_pad_follows_reference_and_carries_bowl(env):
    env.reset(5)
    a = env.bowl_qadr[0]
    env.d.qpos[a:a + 3] = [*env.pad_pos(0), env.pad_top[0] + 0.001]
    start = env.pad_pos(0).copy()
    for _ in range(250):    # 10 s, includes reversals
        env.step(hold(env))
    rel, resting = env.on_pad(0, 0)
    assert resting and np.linalg.norm(rel) < 0.005          # bowl rode along, did not slide off
    assert abs(env.d.qpos[env.pad_qadr[0]] - env.pad_ref(0, env.d.time)) < 0.002
    assert np.linalg.norm(env.pad_pos(0) - start) > 1e-3 or env.task["pads"][0]["speed"] < 0.011


def test_reset_deterministic(env):
    env.reset(11)
    for _ in range(20):
        env.step(hold(env))
    q1 = env.d.qpos.copy()
    env.reset(11)
    for _ in range(20):
        env.step(hold(env))
    assert np.array_equal(q1, env.d.qpos)
    assert env.sample_task(11) == env.sample_task(11)
    assert env.sample_task(11) != env.sample_task(12)


def test_doing_nothing_fails(env):
    env.reset(1)
    done = False
    while not done:
        _, done, info = env.step(hold(env))
    assert not info["success"] and info["failure_type"] == "never_grasped"


def test_bowl_on_pad_but_still_held_is_not_success(env):
    env.reset(2)
    a = env.bowl_qadr[0]
    env.d.qpos[a:a + 3] = [*env.pad_pos(0), env.pad_top[0] + 0.001]
    for _ in range(10):
        env.step(hold(env))
    ok, detail = env.placed()
    assert ok                                   # resting, released, centred
    c = env.contacts()
    c["robot_bowl"][0] = True                   # same state but with the arm touching the bowl
    assert not env.placed(c)[0]


def test_expert_succeeds(cfg, env):
    pol = TickExpert(cfg, env)
    for seed in (0, 1, 2, 3):
        info, _ = run_episode(env, pol, seed)
        assert info["success"], (seed, info, env.stats.detail)


def test_observation_images(cfg):
    env = TickEnv(cfg, render=True)
    obs = env.reset(0)
    n = cfg["vla"]["image_size"]
    assert set(obs["images"]) == set(cfg["vla"]["cameras"])
    assert all(im.shape == (n, n, 3) and im.dtype == np.uint8 for im in obs["images"].values())
    assert obs["proprio"].shape == (6,) and "bowl" in obs["instruction"]
