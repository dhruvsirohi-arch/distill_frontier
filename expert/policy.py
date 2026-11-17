"""Scripted pick-and-place expert.

Decides using ONLY the observation JSON (plus the last-2-actions history that Qwen also sees) and
a set of per-episode variation parameters sampled from the episode seed. It has no access to sim
internals. The decision rule is memoryless w.r.t. hidden state, so phases are inferable from what
Qwen sees, and the expert recovers from failed grasps the same way Qwen could.
"""
from __future__ import annotations

import math

import numpy as np

PAN_AXIS_XY = np.array([0.0388353, 0.0])  # public robot geometry (base frame), not sim state


def _angdiff(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def _radial(xy):
    u = np.asarray(xy, float) - PAN_AXIS_XY
    n = np.linalg.norm(u)
    u = u / n if n > 1e-6 else np.array([1.0, 0.0])
    return u, np.array([-u[1], u[0]])


class ExpertPolicy:
    name = "expert"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.x = cfg["expert"]
        b = cfg["env"]["bowl"]
        self.R_mid = b["radius"] - b["wall"] / 2
        self.H = b["height"]
        self.r_max = cfg["env"]["reach"]["r"][1] - 0.012
        self.params = None
        self.seed = 0

    # ------------------------------------------------------------------ per-episode variation
    def reset(self, seed: int, obs: dict):
        rng = np.random.default_rng([int(seed), 7])
        x, tg = self.x, obs["target"]
        u = lambda r: float(rng.uniform(*r))
        mx = tg["sx"] / 2 - x["place_margin"]
        my = tg["sy"] / 2 - x["place_margin"]
        self.params = {
            "pregrasp_height": u(x["pregrasp_height"]),
            "approach_tangent_offset": u(x["approach_tangent_offset"]),
            "grasp_depth": u(x["grasp_depth"]),
            "grasp_offset_deg": u(x["grasp_offset_deg"]),
            "lift_height": u(x["lift_height"]),
            "n_waypoints": int(rng.random() < x["waypoint_prob"]),
            "waypoint_jitter": [float(v) for v in rng.uniform(-1, 1, 3) * np.array(x["waypoint_jitter"])],
            "place_dx": float(rng.uniform(-mx, mx)),
            "place_dy": float(rng.uniform(-my, my)),
            "place_drop": u(x["place_drop"]),
            "retreat_height": u(x["retreat_height"]),
            "open_first": bool(rng.random() < x["open_first_prob"]),
        }
        self.seed = int(seed)
        return self.params

    # ------------------------------------------------------------------ geometry helpers
    def grasp_plan(self, obs):
        """Grasp point on the rim, consistent with bowl yaw (avoid the lug), and the pregrasp."""
        p = self.params
        b = np.array([obs["bowl"]["x"], obs["bowl"]["y"]])
        u, _ = _radial(b)
        lug = math.radians(obs["bowl"]["yaw"])
        clear = math.radians(self.x["lug_clearance_deg"])
        near = math.atan2(-u[1], -u[0])
        far = math.atan2(u[1], u[0])
        delta = math.radians(p["grasp_offset_deg"])
        side = "near"
        phi = near + delta
        if abs(_angdiff(lug, phi)) < clear:
            phi = near - delta  # flip the offset away from the lug
        if abs(_angdiff(lug, phi)) < clear:
            far_r = np.linalg.norm(b + self.R_mid * np.array([math.cos(far), math.sin(far)]) - PAN_AXIS_XY)
            if far_r <= self.r_max:
                side, phi = "far", far + delta
                if abs(_angdiff(lug, phi)) < clear:
                    phi = far - delta
            else:  # rotate just far enough from the lug on the near side
                s = 1.0 if _angdiff(near, lug) >= 0 else -1.0
                phi = lug + s * clear
        g_xy = b + self.R_mid * np.array([math.cos(phi), math.sin(phi)])
        g = np.array([g_xy[0], g_xy[1], obs["bowl"]["z"] + self.H - p["grasp_depth"]])
        _, t = _radial(g_xy)
        pre = g + np.array([*(t * p["approach_tangent_offset"]), p["pregrasp_height"]])
        return g, pre, side

    def place_plan(self, obs):
        """TCP targets that put the bowl center at the chosen placement point."""
        p, tg = self.params, obs["target"]
        ee = np.array([obs["ee"]["x"], obs["ee"]["y"], obs["ee"]["z"]])
        bowl = np.array([obs["bowl"]["x"], obs["bowl"]["y"], obs["bowl"]["z"]])
        u, t = _radial(ee[:2])
        off = bowl[:2] - ee[:2]
        a_r, a_t = float(off @ u), float(off @ t)
        c = np.array([tg["x"] + p["place_dx"], tg["y"] + p["place_dy"]])

        def tcp_for(c):
            q = c - off
            for _ in range(6):
                uu, tt = _radial(q)
                q = c - a_r * uu - a_t * tt
            return q

        q = tcp_for(c)
        # keep the TCP inside reach by pulling the placement toward the region center if needed
        for _ in range(10):
            if np.linalg.norm(q - PAN_AXIS_XY) <= self.r_max:
                break
            c = c + 0.25 * (np.array([tg["x"], tg["y"]]) - c)
            q = tcp_for(c)
        hang = ee[2] - bowl[2]
        z_place = tg["z"] + hang + p["place_drop"]
        low = np.array([q[0], q[1], z_place])
        high = np.array([q[0], q[1], p["lift_height"]])
        # intermediate waypoint: swing back from the placement toward the spawn side (polar around
        # the pan axis so it always stays inside the reachable annulus)
        jit = np.array(p["waypoint_jitter"])
        rel = q - PAN_AXIS_XY
        th = math.atan2(rel[1], rel[0]) - math.radians(35.0 + 10.0 * jit[0] / self.x["waypoint_jitter"][0])
        r = float(np.clip(np.linalg.norm(rel) + jit[1], 0.12, 0.24))
        wp = np.array([PAN_AXIS_XY[0] + r * math.cos(th), PAN_AXIS_XY[1] + r * math.sin(th),
                       p["lift_height"] + 0.01 + abs(jit[2])])
        return low, high, wp

    # ------------------------------------------------------------------ decision
    def _move(self, target, step, scale=1.0):
        rng = np.random.default_rng([self.seed, 11, int(step)])
        n = np.clip(rng.normal(0, self.x["noise_std"] * scale, 3), -self.x["noise_clip"] * scale,
                    self.x["noise_clip"] * scale)
        v = np.asarray(target, float) + n
        return {"action": "move_to", "x": round(float(v[0]), 3), "y": round(float(v[1]), 3),
                "z": round(float(v[2]), 3)}

    def act(self, obs: dict, history: list[dict]) -> dict:
        tol = self.x["near_tol"]
        step = obs["step"]
        ee = np.array([obs["ee"]["x"], obs["ee"]["y"], obs["ee"]["z"]])
        tg = obs["target"]
        last = history[-1]["action"] if history else None
        prev = history[-2]["action"] if len(history) > 1 else None
        closed = obs["gripper"] == "closed"
        on_table = abs(obs["bowl"]["z"] - tg["z"]) < 0.008

        if obs["grasped"]:
            low, high, wp = self.place_plan(obs)
            if np.linalg.norm(ee[:2] - low[:2]) < tol and abs(ee[2] - low[2]) < 0.01:
                return {"action": "open_gripper"}
            if np.linalg.norm(ee[:2] - low[:2]) < tol:
                return self._move(low, step, 0.5)
            if ee[2] < self.params["lift_height"] - 0.015:
                return self._move([ee[0], ee[1], self.params["lift_height"]], step)
            last_wp = (last == "move_to" and np.linalg.norm(
                np.array([history[-1]["x"], history[-1]["y"], history[-1]["z"]]) - wp) < 0.01)
            if self.params["n_waypoints"] and not last_wp and np.linalg.norm(ee - wp) > tol:
                return self._move(wp, step, 2.0)
            return self._move(high, step)

        if on_table:  # released (or dropped) on the table: open, retreat, finish
            if closed:
                return {"action": "open_gripper"}
            if ee[2] < self.params["retreat_height"] - 0.01:
                return self._move([ee[0], ee[1], self.params["retreat_height"]], step)
            return {"action": "done"}

        g, pre, _ = self.grasp_plan(obs)
        if closed:
            if np.linalg.norm(ee - g) < tol:  # failed grasp: let go and re-approach
                return {"action": "open_gripper"}
            if self.params["open_first"] or np.linalg.norm(ee - pre) < tol:
                return {"action": "open_gripper"}
            return self._move(pre, step)
        if last == "open_gripper" and prev == "close_gripper":
            return self._move(pre, step)
        if np.linalg.norm(ee - g) < 0.008:
            return {"action": "close_gripper"}
        if np.linalg.norm(ee - pre) < tol or (np.linalg.norm(ee[:2] - g[:2]) < 0.02 and ee[2] < pre[2]):
            return self._move(g, step)
        return self._move(pre, step)
