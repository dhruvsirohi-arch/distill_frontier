"""Low-level robot controller: the skill layer the exam agents command.

Agents send Cartesian goals; this module turns them into actuator setpoints every control tick,
like the motion controller on a real robot. It uses only proprioception (joint encoders and wheel
odometry), never object state.

Action (all fields optional; a missing field keeps its previous command):
    base:       [v, w]   forward speed (m/s, |v|<=0.6) and turn rate (rad/s, |w|<=2.0)
    torso_yaw:  rad      waist rotation, |yaw|<=1.5
    left/right: {"pos": [x, y, z], "grip": g}
                pos   pinch point (between the fingertips) in the robot base frame, metres:
                      x forward, y left, z above the floor. The gripper is held level with the
                      jaws closing horizontally, approaching radially from the torso axis.
                grip  jaw opening, 1 = fully open (80 mm), 0 = closed.
"""
from __future__ import annotations

import copy

import mujoco
import numpy as np

from .scene import ARM_JOINTS, SIDES

V_MAX, W_MAX, TORSO_MAX = 0.6, 2.0, 1.5
# joint trajectory speed limits (rad/s): shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw
JOINT_SPEED = np.array([12.0, 12.0, 12.0, 12.0, 16.0, 16.0, 16.0])
READY = {
    "base": [0.0, 0.0], "torso_yaw": 0.0,
    "left": {"pos": [0.38, 0.20, 1.30], "grip": 1.0},
    "right": {"pos": [0.38, -0.20, 1.30], "grip": 1.0},
}


def _yaw_quat(yaw):
    return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])


def merge_action(prev: dict, new: dict | None) -> dict:
    """Fill missing fields of `new` from `prev` and clamp everything to the robot's limits."""
    out = copy.deepcopy(prev)
    if not new:
        return out
    if "base" in new and new["base"] is not None:
        v, w = (float(x) for x in new["base"])
        out["base"] = [float(np.clip(v, -V_MAX, V_MAX)), float(np.clip(w, -W_MAX, W_MAX))]
    if new.get("torso_yaw") is not None:
        out["torso_yaw"] = float(np.clip(float(new["torso_yaw"]), -TORSO_MAX, TORSO_MAX))
    for side in SIDES:
        arm = new.get(side)
        if not arm:
            continue
        if arm.get("pos") is not None:
            p = np.asarray(arm["pos"], float).reshape(3)
            if np.isfinite(p).all():
                out[side]["pos"] = np.clip(p, [-0.3, -0.9, 0.5], [0.9, 0.9, 1.9]).tolist()
        if arm.get("grip") is not None:
            out[side]["grip"] = float(np.clip(float(arm["grip"]), 0.0, 1.0))
    return out


class Controller:
    def __init__(self, model: mujoco.MjModel):
        self.m = model
        self.scratch = mujoco.MjData(model)
        jid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        aid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        self.base_q = [model.jnt_qposadr[jid(n)] for n in ("base_x", "base_y", "base_yaw")]
        self.base_v = [model.jnt_dofadr[jid(n)] for n in ("base_x", "base_y", "base_yaw")]
        self.base_act = [aid(n) for n in ("base_x", "base_y", "base_yaw")]
        self.torso_q = model.jnt_qposadr[jid("torso_Joint")]
        self.torso_act = aid("torso_Joint")
        self.arm = {}
        for side in SIDES:
            names = [f"{side}_{a}_joint" for a in ARM_JOINTS]
            ids = [jid(n) for n in names]
            self.arm[side] = dict(
                qadr=np.array([model.jnt_qposadr[i] for i in ids]),
                dadr=np.array([model.jnt_dofadr[i] for i in ids]),
                lo=model.jnt_range[ids, 0] + 0.02, hi=model.jnt_range[ids, 1] - 0.02,
                act=np.array([aid(n) for n in names]),
                fingers=[aid(f"{side}_dex1_finger_joint_{k}") for k in (1, 2)],
                site=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_pinch"),
                hand=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_dex1_base_link"),
            )
        self.cmd = copy.deepcopy(READY)
        self.q_goal = {}
        self.seeds = {}

    # ---- proprioception --------------------------------------------------------------------------
    def base_pose(self, d):
        x, y, yaw = (d.qpos[i] for i in self.base_q)
        return np.array([x, y]), float(yaw)

    def to_world(self, d, p_base):
        xy, yaw = self.base_pose(d)
        c, s = np.cos(yaw), np.sin(yaw)
        p = np.asarray(p_base, float)
        return np.array([xy[0] + c * p[0] - s * p[1], xy[1] + s * p[0] + c * p[1], p[2]])

    def to_base(self, d, p_world):
        xy, yaw = self.base_pose(d)
        c, s = np.cos(yaw), np.sin(yaw)
        dx, dy = p_world[0] - xy[0], p_world[1] - xy[1]
        return np.array([c * dx + s * dy, -s * dx + c * dy, p_world[2]])

    # ---- inverse kinematics ----------------------------------------------------------------------
    def solve(self, d, side, target_world, torso_yaw, iters=25):
        """IK with restarts: warm start from the last goal; if that does not converge, retry from
        the nearest precomputed seeds and keep the best."""
        start = self.q_goal.get(side, d.qpos[self.arm[side]["qadr"]]).copy()
        q, err = self._dls(d, side, target_world, torso_yaw, start, iters)
        if err > 0.004 and self.seeds.get(side) is not None:
            local = self.to_base(d, target_world)
            pts, qs = self.seeds[side]
            # seeds are expressed for torso yaw 0; rotate the query into the torso frame
            c, s_ = np.cos(-torso_yaw), np.sin(-torso_yaw)
            lq = np.array([c * local[0] - s_ * local[1], s_ * local[0] + c * local[1], local[2]])
            for k in np.argsort(np.linalg.norm(pts - lq, axis=1))[:4]:
                q2, e2 = self._dls(d, side, target_world, torso_yaw, qs[k].copy(), 60)
                if e2 < err - 1e-4:
                    q, err = q2, e2
                if err < 0.002:
                    break
        self.q_goal[side] = q
        return q, err

    def build_seeds(self, d):
        """Grid of converged arm postures over the catching workspace (torso yaw 0)."""
        self.seeds = {}
        for side in SIDES:
            sgn = 1 if side == "left" else -1
            pts, qs = [], []
            q_prev = d.qpos[self.arm[side]["qadr"]].copy()
            for z in (0.85, 1.0, 1.15, 1.3, 1.45):
                for r in (0.3, 0.42, 0.52, 0.62):
                    for ang in np.radians(np.arange(-40, 111, 15)):
                        p = np.array([r * np.cos(ang), sgn * r * np.sin(ang), z])
                        best = None
                        for q0 in ([q_prev] + [qs[k] for k in np.argsort(
                                np.linalg.norm(np.array(pts) - p, axis=1))[:2]] if pts else [q_prev]):
                            q, e = self._dls(d, side, self.to_world(d, p), 0.0, q0.copy(), 80)
                            if best is None or e < best[1]:
                                best = (q, e)
                        if best[1] < 0.003:
                            pts.append(p); qs.append(best[0]); q_prev = best[0]
            self.seeds[side] = (np.array(pts), np.array(qs))

    def _dls(self, d, side, target_world, torso_yaw, q, iters):
        """Damped least squares on the 7 arm joints: pinch point to the target, gripper z axis up,
        approach direction radial from the torso (weak preference)."""
        m, sd, a = self.m, self.scratch, self.arm[side]
        sd.qpos[:] = d.qpos
        sd.qpos[self.torso_q] = torso_yaw
        torso_xy = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")][:2]
        rad = np.asarray(target_world[:2]) - torso_xy
        yaw = np.arctan2(rad[1], rad[0])
        R_des = np.zeros(9)
        mujoco.mju_quat2Mat(R_des, _yaw_quat(yaw))
        R_des = R_des.reshape(3, 3)
        jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
        w_pos, w_tilt, w_yaw, lam = 1.0, 0.35, 0.08, 0.02
        for _ in range(iters):
            sd.qpos[a["qadr"]] = q
            mujoco.mj_kinematics(m, sd)
            mujoco.mj_comPos(m, sd)
            e_p = target_world - sd.site_xpos[a["site"]]
            R = sd.xmat[a["hand"]].reshape(3, 3)
            # rotation error (world frame) from current to desired
            Re = R_des @ R.T
            qe = np.zeros(4); mujoco.mju_mat2Quat(qe, Re.ravel())
            e_r = np.zeros(3); mujoco.mju_quat2Vel(e_r, qe, 1.0)
            mujoco.mj_jacSite(m, sd, jp, None, a["site"])
            mujoco.mj_jacBody(m, sd, None, jr, a["hand"])
            Jp, Jr = jp[:, a["dadr"]], jr[:, a["dadr"]]
            W = np.array([w_tilt, w_tilt, w_yaw])
            J = np.vstack([w_pos * Jp, W[:, None] * Jr])
            e = np.concatenate([w_pos * e_p, W * e_r])
            dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), e)
            q = np.clip(q + dq, a["lo"], a["hi"])
            if np.linalg.norm(e_p) < 1e-3 and np.linalg.norm(e_r[:2]) < 0.02:
                break
        sd.qpos[a["qadr"]] = q
        mujoco.mj_kinematics(m, sd)
        err = np.linalg.norm(target_world - sd.site_xpos[a["site"]])
        tilt = np.arccos(np.clip(sd.xmat[a["hand"]][8], -1, 1))
        return q, float(err + 0.05 * max(tilt - 0.1, 0))

    # ---- per control tick --------------------------------------------------------------------------
    def set_command(self, d, action):
        self.cmd = merge_action(self.cmd, action)
        c = self.cmd
        d.ctrl[self.torso_act] = c["torso_yaw"]
        self.ik_error = {}
        for side in SIDES:
            a = self.arm[side]
            target = self.to_world(d, c[side]["pos"])
            q, err = self.solve(d, side, target, c["torso_yaw"])
            self.q_target[side] = q
            self.ik_error[side] = err
            open_q = 0.041 * c[side]["grip"]
            for f in a["fingers"]:
                d.ctrl[f] = open_q

    def drive(self, d):
        """Called every physics step: rate-limited joint setpoints toward the IK goal (the joint
        trajectory generator) and differential-drive kinematics (no sideways slip)."""
        dt = self.m.opt.timestep
        for side in SIDES:
            a = self.arm[side]
            sp = self.q_setpoint[side]
            step = np.clip(self.q_target[side] - sp, -JOINT_SPEED * dt, JOINT_SPEED * dt)
            self.q_setpoint[side] = sp + step
            d.ctrl[a["act"]] = self.q_setpoint[side]
        v, w = self.cmd["base"]
        yaw = d.qpos[self.base_q[2]]
        d.ctrl[self.base_act[0]] = v * np.cos(yaw)
        d.ctrl[self.base_act[1]] = v * np.sin(yaw)
        d.ctrl[self.base_act[2]] = w

    def reset(self, d):
        self.cmd = copy.deepcopy(READY)
        self.q_goal = {}
        if not self.seeds:
            mujoco.mj_forward(self.m, d)
            self.build_seeds(d)
            self.q_goal = {}
        # pre-solve the ready pose and place the arms there
        for _ in range(3):
            for side in SIDES:
                q, _ = self.solve(d, side, self.to_world(d, READY[side]["pos"]), 0.0, iters=60)
                d.qpos[self.arm[side]["qadr"]] = q
            mujoco.mj_forward(self.m, d)
        self.q_setpoint = {s: d.qpos[self.arm[s]["qadr"]].copy() for s in SIDES}
        self.q_target = {s: self.q_setpoint[s].copy() for s in SIDES}
        self.set_command(d, None)
        for side in SIDES:
            for k, f in enumerate(self.arm[side]["fingers"]):
                jnt = self.m.actuator_trnid[f, 0]
                d.qpos[self.m.jnt_qposadr[jnt]] = 0.041
        mujoco.mj_forward(self.m, d)
