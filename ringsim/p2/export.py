"""Export per-frame trajectories of Phase 2 episodes by deterministic replay of the recorded
commands (outcomes are checked against the original run).

    python -m ringsim.p2.export outputs/ringsim2/astra_realtime.jsonl --out outputs/ringsim2/traj
Writes one <mode>_<seed>.npz per episode with, at every 30 Hz frame:
  t                    game clock (s)
  qpos, qvel           full MuJoCo state (robot joints, base, sticks)
  joint_names          the 19 joints in joint_pos/joint_vel/joint_effort order
  joint_pos/vel/effort sensor_msgs/JointState as published
  pinch_left/right     gripper pinch points in base_link (m)
  grip_left/right      jaw opening (1 open, 0 closed)
  cmd_left/right       commanded pinch target [x, y, z, grip] in force at that frame
  imu_quat, imu_gyro, imu_accel
  stick_pos            (frames, n_sticks, 3) stick centres in the world frame
  stick_state          hanging / falling / resolved / cleared per stick
plus the release schedule, the per-drop outcomes, and the frames at which Astra was called and
its commands took effect.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def export(rec, out_dir):
    os.environ.setdefault("MUJOCO_GL", "egl")
    from .world import World
    w = World(rec["seed"], render=False)
    g = w.game
    actions = {int(k): a for k, a in rec.get("applied", [])}
    cols = {k: [] for k in ("t", "qpos", "qvel", "joint_pos", "joint_vel", "joint_effort",
                            "pinch_left", "pinch_right", "grip_left", "grip_right", "cmd_left",
                            "cmd_right", "imu_quat", "imu_gyro", "imu_accel", "stick_pos",
                            "stick_state")}

    def grab():
        m = w.msgs
        c = g.ctrl.cmd
        cols["t"].append(g.t)
        cols["qpos"].append(g.d.qpos.copy())
        cols["qvel"].append(g.d.qvel.copy())
        cols["joint_pos"].append(m["joint_states"]["position"])
        cols["joint_vel"].append(m["joint_states"]["velocity"])
        cols["joint_effort"].append(m["joint_states"]["effort"])
        for s in ("left", "right"):
            cols[f"pinch_{s}"].append(m["tf"]["pinch"][s])
            cols[f"grip_{s}"].append(m["tf"]["grip"][s])
            cols[f"cmd_{s}"].append(list(c[s]["pos"]) + [c[s]["grip"]])
        cols["imu_quat"].append(m["imu"]["orientation_wxyz"])
        cols["imu_gyro"].append(m["imu"]["angular_velocity"])
        cols["imu_accel"].append(m["imu"]["linear_acceleration"])
        cols["stick_pos"].append(g.d.xpos[g.stick_body].copy())
        cols["stick_state"].append(list(g.state))

    grab()
    while not w.done:
        w.step(actions.get(g.tick))
        grab()
    res = g.result()
    same = [e["outcome"] for e in res["events"]] == [e["outcome"] for e in rec["events"]]
    calls = rec.get("decisions", [])
    arrays = {k: np.asarray(v) for k, v in cols.items()}
    arrays["stick_state"] = np.asarray(cols["stick_state"], dtype="U9")
    path = Path(out_dir) / f"{rec['mode']}_{rec['seed']}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, **arrays, joint_names=np.asarray(w.proprio.names),
        release_slot=np.asarray(g.schedule.order), release_t=np.asarray(g.schedule.release_t),
        slots=np.asarray(g.slots),
        outcomes=np.asarray([e["outcome"] for e in res["events"]]),
        call_frame=np.asarray([d.get("frame", -1) for d in calls]),
        call_t_obs=np.asarray([d.get("t_obs", np.nan) for d in calls], float),
        call_t_applied=np.asarray([d.get("t_applied", d.get("t_obs") if rec["mode"] == "standard"
                                         and d.get("action") else np.nan) or np.nan
                                   for d in calls], float),
        meta=json.dumps({"mode": rec["mode"], "seed": rec["seed"], "fps": 30,
                         "caught": res["caught"], "reproduced": same}))
    return path, same


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(a.jsonl)]
    recs = [r for r in recs if "fatal" not in r and (a.seed is None or r["seed"] == a.seed)]
    ok = 0
    for r in recs:
        path, same = export(r, a.out)
        ok += same
        print(path, "reproduced" if same else "MISMATCH")
    print(f"{ok}/{len(recs)} exported and reproduced")


if __name__ == "__main__":
    main()
