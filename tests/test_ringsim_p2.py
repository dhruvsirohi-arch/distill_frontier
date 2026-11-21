import json
import os

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")

from ringsim.p2 import budget
from ringsim.p2.agent import SCHEMA, build_inputs, instructions
from ringsim.p2.sensors import workspace_cloud
from ringsim.p2.world import FRAME, World


@pytest.fixture(scope="module")
def world():
    return World(50000)


def test_frame_clock_and_releases_on_grid(world):
    g = world.game
    assert g.substeps == 16 and abs(g.tick_dt - FRAME) < 1e-12
    for t in g.schedule.release_t:
        assert abs(t * 30 - round(t * 30)) < 0.01      # stored to 0.1 ms; fires at the frame


def test_lidar_sees_all_hanging_sticks(world):
    pts = workspace_cloud(world.snapshot()["scan"]) / 100.0
    slots = world.game.slots
    for x, y, _ in slots:
        near = np.hypot(pts[:, 0] - x, pts[:, 1] - y) < 0.04
        assert near.sum() >= 5, (x, y)
    # nothing on the robot itself (self-filter): no points near the torso axis
    assert (np.hypot(pts[:, 0], pts[:, 1]) > 0.3).all()


def test_scan_published_at_10hz():
    w = World(50001)
    stamps = []
    for _ in range(30):
        w.step()
        stamps.append(w.scan["stamp"])
    new = sorted(set(stamps))[1:]
    assert np.allclose(np.diff(new), 0.1, atol=1e-6)


def test_proprio_messages(world):
    s = world.snapshot()
    js = s["joint_states"]
    assert len(js["position"]) == len(world.proprio.names) == 19
    assert abs(np.linalg.norm(s["imu"]["orientation_wxyz"]) - 1) < 1e-3
    assert 9.5 < s["imu"]["linear_acceleration"][2] < 10.1        # gravity, at rest


def test_inputs_and_schema(world):
    inputs, text = build_inputs(world.snapshot(), "memo")
    assert [p["type"] for p in inputs] == ["text", "image", "image", "image", "image"]
    head = json.loads(text.split("\n")[0][len("Observation "):])
    assert head["your_notes"] == "memo" and "stick" not in json.dumps(head)
    assert SCHEMA["required"][0] == "reasoning"
    instr = instructions("standard", world.proprio.names)
    assert "every frame" in instr and "release schedule" not in instr


def test_budget_guard(tmp_path, monkeypatch):
    f = tmp_path / "lim.json"
    monkeypatch.setenv("RINGSIM_LIMITS_FILE", str(f))
    f.write_text(json.dumps({"primary": {"usedPercent": 10}, "secondary": {"usedPercent": 50}}))
    assert budget.wait_for_room() == 0
    f.write_text(json.dumps({"primary": {"usedPercent": 10}, "secondary": {"usedPercent": 95}}))
    with pytest.raises(budget.BudgetExhausted):
        budget.wait_for_room()
