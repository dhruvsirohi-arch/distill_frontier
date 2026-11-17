"""Damped-least-squares IK for a top-down SO-101 TCP pose.

Target orientation: fingers pointing straight down, jaw closing direction aligned with the radial
direction from the pan axis to the target. With the SO-101's planar arm this is exactly solvable
with 5 DOF. IK runs on a private MjData, so it never touches the simulation state.
"""
from __future__ import annotations

import mujoco
import numpy as np

from env.scene import PAN_AXIS_XY

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def topdown_rotation(target_xy) -> np.ndarray:
    u = np.asarray(target_xy, float) - PAN_AXIS_XY
    n = np.linalg.norm(u)
    u = u / n if n > 1e-6 else np.array([1.0, 0.0])
    x = np.array([u[0], u[1], 0.0])
    z = np.array([0.0, 0.0, 1.0])
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


class TopDownIK:
    def __init__(self, model: mujoco.MjModel, pos_tol=0.002, rot_tol=0.05, max_iters=150, damping=0.01):
        self.m = model
        self.d = mujoco.MjData(model)
        self.site = model.site("tcp").id
        self.qadr = np.array([model.jnt_qposadr[model.joint(j).id] for j in ARM_JOINTS])
        self.dofadr = np.array([model.jnt_dofadr[model.joint(j).id] for j in ARM_JOINTS])
        self.lo = np.array([model.jnt_range[model.joint(j).id][0] for j in ARM_JOINTS])
        self.hi = np.array([model.jnt_range[model.joint(j).id][1] for j in ARM_JOINTS])
        self.pos_tol, self.rot_tol, self.max_iters, self.damping = pos_tol, rot_tol, max_iters, damping
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))
        # canonical elbow-up, top-down seed configurations
        self.seeds = [np.array([0.0, -0.3, 0.6, 1.3, 0.0]), np.array([0.0, 0.2, 0.2, 1.2, 0.0]),
                      np.array([0.0, -0.8, 1.2, 1.1, 0.0])]

    def fk(self, q):
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[self.site].copy(), self.d.site_xmat[self.site].reshape(3, 3).copy()

    def _solve_from(self, target, Rdes, q0):
        q = np.clip(q0.copy(), self.lo, self.hi)
        lam2 = self.damping ** 2
        for _ in range(self.max_iters):
            p, R = self.fk(q)
            ep = target - p
            er = 0.5 * sum(np.cross(R[:, i], Rdes[:, i]) for i in range(3))
            if np.linalg.norm(ep) < self.pos_tol * 0.25 and np.linalg.norm(er) < self.rot_tol * 0.25:
                break
            mujoco.mj_comPos(self.m, self.d)
            mujoco.mj_jacSite(self.m, self.d, self.jacp, self.jacr, self.site)
            J = np.vstack([self.jacp[:, self.dofadr], 0.3 * self.jacr[:, self.dofadr]])
            e = np.concatenate([ep, 0.3 * er])
            dq = J.T @ np.linalg.solve(J @ J.T + lam2 * np.eye(6), e)
            n = np.linalg.norm(dq)
            if n > 0.3:
                dq *= 0.3 / n
            q = np.clip(q + dq, self.lo, self.hi)
        p, R = self.fk(q)
        perr = np.linalg.norm(target - p)
        rerr = np.linalg.norm(0.5 * sum(np.cross(R[:, i], Rdes[:, i]) for i in range(3)))
        return q, perr, rerr

    def solve(self, target, q_init=None):
        """Return (q[5], ok, pos_err, rot_err). Tries the warm start first, then canonical seeds."""
        target = np.asarray(target, float)
        Rdes = topdown_rotation(target[:2])
        u = target[:2] - PAN_AXIS_XY
        pan_guess = float(np.arctan2(-u[1], u[0]))  # sign verified by tests/test_env.py
        cands = []
        inits = ([np.asarray(q_init, float)] if q_init is not None else [])
        for s in self.seeds:
            s = s.copy()
            s[0] = pan_guess
            inits.append(s)
        best = None
        for q0 in inits:
            q, pe, re = self._solve_from(target, Rdes, q0)
            ok = pe < self.pos_tol and re < self.rot_tol
            cand = (not ok, pe + 0.1 * re, q, pe, re)
            cands.append(cand)
            if ok and q_init is not None and q0 is inits[0]:
                best = cand
                break
        if best is None:
            best = min(cands, key=lambda c: (c[0], c[1]))
        _, _, q, pe, re = best
        return q, bool(pe < self.pos_tol and re < self.rot_tol), float(pe), float(re)
