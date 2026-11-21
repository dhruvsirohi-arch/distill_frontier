"""Reaction Ring simulator: rules, judge, determinism and the sensor boundary."""
import os

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")

from ringsim.baselines import Idle, Parked, ScheduleOracle  # noqa: E402
from ringsim.game import COUNTDOWN, INTERVAL, Game, Schedule  # noqa: E402


@pytest.fixture(scope="module")
def game():
    return Game(render=False)


def play(game, seed, policy_cls, **kw):
    game.reset(seed)
    pol = policy_cls(game)
    for k, v in kw.items():
        setattr(pol, k, v)
    while not game.done:
        game.step(pol.act(game), observe=False)
    return game.result()


def test_schedule_follows_manual_difficulty_one():
    s = Schedule.sample(7, 8)
    assert sorted(s.order) == list(range(8))
    gaps = np.diff([COUNTDOWN] + s.release_t)
    assert (gaps >= INTERVAL[0] - 0.021).all() and (gaps <= INTERVAL[1] + 0.021).all()
    assert Schedule.sample(7, 8) == s and Schedule.sample(8, 8) != s


def test_magnets_hold_until_release(game):
    game.reset(3)
    z0 = game.d.xpos[game.stick_body].copy()[:, 2]
    while game.t < game.schedule.release_t[0] - 0.05:
        game.step(None, observe=False)
    assert np.allclose(game.d.xpos[game.stick_body][:, 2], z0, atol=2e-3)
    while game.t < game.schedule.release_t[0] + 0.2:
        game.step(None, observe=False)
    first = game.schedule.order[0]
    assert game.d.xpos[game.stick_body[first]][2] < z0[first] - 0.1


def test_idle_robot_catches_nothing_and_every_stick_lands(game):
    res = play(game, 11, Idle)
    assert res["caught"] == 0
    assert [e["outcome"] for e in res["events"]] == ["floor"] * 8


def test_schedule_oracle_solves_the_task(game):
    caught = sum(play(game, s, ScheduleOracle)["caught"] for s in (0, 1, 4))
    assert caught >= 22


def test_parked_jaws_score_only_with_open_jaws(game):
    res = play(game, 21, Parked)
    assert res["caught"] == 2                           # two fixed guesses, timed close
    trapped = play(game, 21, Parked, trap=True)
    assert trapped["caught"] == 0
    assert sum(e["outcome"] == "trapped" for e in trapped["events"]) == 2


def test_replay_is_deterministic(game):
    a = play(game, 5, ScheduleOracle)
    b = play(game, 5, ScheduleOracle)
    assert a == b


def test_sensor_observation_has_no_object_state():
    g = Game(render=True, img_size=(64, 48))
    obs = g.reset(2)
    assert set(obs) == {"t", "rgb", "depth", "joints", "odometry", "pinch", "grip"}
    assert obs["rgb"].shape == (48, 128, 3) and obs["depth"].shape == (48, 128)
    text = repr({k: v for k, v in obs.items() if k not in ("rgb", "depth")})
    assert "stick" not in text and "schedule" not in text
