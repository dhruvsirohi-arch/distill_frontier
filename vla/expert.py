"""Tick-level scripted expert for the moving-pad task.

Uses privileged simulator state. Every tick it picks a Cartesian goal for the TCP from the current
phase, steps a speed-limited distance from the *measured* TCP toward it, and solves top-down IK
for the joint targets. Because the command is a function of the measured state, it self-corrects
after perturbations (which is what makes noise-injected data collection useful).
"""
from __future__ import annotations

import numpy as np

from env.scene import PAN_AXIS_XY

PHASES = ("approach", "descend", "close", "lift", "transit", "lower", "release", "retreat", "rest")


def _radial(xy):
    u = np.asarray(xy, float) - PAN_AXIS_XY
    n = np.linalg.norm(u)
    u = u / n if n > 1e-6 else np.array([1.0, 0.0])
    return u, np.array([-u[1], u[0]])


class TickExpert:
    name = "expert"

    def __init__(self, cfg: dict, env):
        self.cfg, self.x, self.env = cfg, cfg["vla"]["expert"], env
        b = cfg["vla"]["bowl"]
        self.R_mid, self.H = b["radius"] - b["wall"] / 2, b["height"]
        self.g_open, self.g_closed = cfg["vla"]["gripper"]["open"], cfg["vla"]["gripper"]["closed"]
        rc = cfg["env"]["reach"]
        self.tol = cfg["vla"]["success"]["center_tol"]
        self.r_lim, self.z_lim = (rc["r"][0] + 0.005, rc["r"][1] - 0.008), rc["z"]

    def reset(self, seed: int):
        rng = np.random.default_rng([int(seed), 7])
        x = self.x
        u = lambda r: float(rng.uniform(*r))
        self.p = {
            "pregrasp_height": u(x["pregrasp_height"]), "grasp_depth": u(x["grasp_depth"]),
            "lift_height": u(x["lift_height"]), "place_drop": u(x["place_drop"]),
            "place_dx": float(rng.uniform(-1, 1) * x["place_offset"]),
            "place_dy": float(rng.uniform(-1, 1) * x["place_offset"]),
            "retreat_height": u(x["retreat_height"]), "max_speed": u(x["max_speed"]),
        }
        self.phase, self.phase_ticks = "approach", 0
        self.grip = self.g_open
        self.q_last = None
        return self.p

    def _set(self, phase):
        self.phase, self.phase_ticks = phase, 0

    # ------------------------------------------------------------------ geometry
    def _clamp(self, p):
        """Keep a TCP goal inside the top-down reachable annulus."""
        p = np.array(p, float)
        rel = p[:2] - PAN_AXIS_XY
        r = np.linalg.norm(rel)
        rc = np.clip(r, *self.r_lim)
        if r > 1e-6:
            p[:2] = PAN_AXIS_XY + rel * rc / r
        p[2] = np.clip(p[2], *self.z_lim)
        return p

    def _place_tcp(self, s, lead):
        """TCP xy that puts the held bowl's centre on the (lead-compensated) placement point."""
        bi, pi = s["target_bowl"], s["target_pad"]
        ee, bowl = s["tcp"], s["bowls"][bi]
        u, t = _radial(ee[:2])
        off = bowl[:2] - ee[:2]
        a_r, a_t = float(off @ u), float(off @ t)
        c = s["pads"][pi] + np.array([self.p["place_dx"], self.p["place_dy"]]) + s["pad_vel"][pi] * lead
        q = c - off
        for _ in range(6):      # the offset rotates with the pan angle
            uu, tt = _radial(q)
            q = c - a_r * uu - a_t * tt
        return q

    # ------------------------------------------------------------------ decision
    def act(self) -> np.ndarray:
        env, p = self.env, self.p
        s = env.state()
        bi, pi = s["target_bowl"], s["target_pad"]
        tcp, bowl = s["tcp"], s["bowls"][bi]
        held = s["grasped"][bi]
        grip_q = s["q"][5]
        lead = self.x["lead_time"]
        self.phase_ticks += 1

        # lost the bowl while carrying it -> start over
        if self.phase in ("lift", "transit", "lower") and not held and self.phase_ticks > 3:
            self._set("approach")

        # grasp the rim point nearest the arm; lead the bowl if it is riding on a pad
        bxy = bowl[:2] + env.bowl_vel(bi)[:2] * lead
        u, _ = _radial(bxy)
        g = np.array([*(bxy - self.R_mid * u), bowl[2] + self.H - p["grasp_depth"]])
        pre = g + np.array([0, 0, p["pregrasp_height"]])
        speed = p["max_speed"]

        if self.phase == "approach":
            self.grip = self.g_open
            goal = pre
            if np.linalg.norm(tcp - pre) < 0.008 and grip_q > 0.6 * self.g_open:
                self._set("descend")
        if self.phase == "descend":
            goal, speed = g, 0.6 * p["max_speed"]
            if np.linalg.norm(tcp[:2] - g[:2]) > 0.012:      # bowl got pushed: re-align from above
                self._set("approach"); goal = pre
            elif np.linalg.norm(tcp - g) < 0.004:
                self._set("close")
        if self.phase == "close":
            self.grip = self.g_closed
            goal = tcp if held else g
            if held and self.phase_ticks >= 8:
                self._hold_xy = tcp[:2].copy()
                self._set("lift")
            elif self.phase_ticks > 25:
                self._set("approach"); goal = pre
        if self.phase == "lift":
            goal = np.array([*self._hold_xy, p["lift_height"]])
            if tcp[2] > p["lift_height"] - 0.015:
                self._set("transit")
        if self.phase in ("transit", "lower", "release"):
            q = self._place_tcp(s, lead)
            hang = tcp[2] - bowl[2]
            z_low = env.pad_top[pi] + hang + p["place_drop"]
            if self.phase == "transit":
                goal = np.array([*q, p["lift_height"]])
                if np.linalg.norm(tcp[:2] - q) < 0.010:
                    self._set("lower")
            if self.phase == "lower":
                goal, speed = np.array([*q, z_low]), 0.6 * p["max_speed"]
                if tcp[2] - z_low < 0.004 and np.linalg.norm(tcp[:2] - q) < 0.008:
                    self._z_rel = tcp[2]
                    self._set("release")
            if self.phase == "release":
                self.grip = self.g_open
                # keep riding with the pad at the release height while the jaws open
                goal = np.array([*(tcp[:2] + s["pad_vel"][pi] * (env.dt + lead)), self._z_rel])
                if grip_q > 0.7 * self.g_open and self.phase_ticks >= 6:
                    self._set("retreat")
        if self.phase == "retreat":
            goal = np.array([*(tcp[:2] + s["pad_vel"][pi] * (env.dt + lead)), p["retreat_height"]])
            if tcp[2] > p["retreat_height"] - 0.015:
                self._set("rest")
        if self.phase == "rest":
            goal, speed = np.array(self.x["rest_tcp"]), 0.6 * p["max_speed"]
            rel, resting = env.on_pad(bi, pi)
            if self.phase_ticks > 10 and not (resting and np.all(np.abs(rel) < 0.8 * self.tol)):
                self._set("approach"); goal = pre      # missed or knocked off: pick it up again

        goal = self._clamp(goal)
        d = goal - tcp
        n = np.linalg.norm(d)
        step = speed * env.dt * 1.6          # servo lags the command, so lead it a little
        cmd = goal if n <= step else tcp + d * step / n
        q, ok, *_ = env.ik.solve(self._clamp(cmd), s["q"][:5])
        if not ok and self.q_last is not None:
            q = self.q_last
        self.q_last = q
        return np.concatenate([q, [self.grip]])
