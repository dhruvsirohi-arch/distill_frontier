"""Tick-level SO-101 environment for the VLA: joint-position actions at control rate, camera
observations, bowls to be placed on pads that slide back and forth on rails.

  action       6 floats: absolute joint position targets (5 arm joints + gripper), radians
  observation  {"images": {cam: uint8 HxWx3}, "proprio": 6 joint positions, "instruction": str}

Privileged simulator state (`state()`) is for the scripted expert and for scoring only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from env.ik import ARM_JOINTS, TopDownIK
from vla.scene import build_model

TEMPLATES = [
    "put the {bowl} bowl on the {pad} pad",
    "place the {bowl} bowl on the {pad} pad",
    "pick up the {bowl} bowl and put it on the {pad} pad",
    "move the {bowl} bowl onto the {pad} pad",
    "set the {bowl} bowl down on the {pad} pad",
    "the {bowl} bowl goes on the {pad} pad",
]


@dataclass
class TickStats:
    seed: int = 0
    ticks: int = 0
    success: bool = False
    failure_type: str = ""
    ever_grasped: bool = False
    grasped_wrong_bowl: bool = False
    robot_env_ticks: int = 0
    detail: dict = field(default_factory=dict)


class TickEnv:
    def __init__(self, cfg: dict, render: bool = True):
        self.cfg, self.v = cfg, cfg["vla"]
        v = self.v
        self.m = m = build_model(cfg)
        self.d = mujoco.MjData(m)
        self.n_sub = int(round(1.0 / (v["control_hz"] * m.opt.timestep)))
        self.dt = self.n_sub * m.opt.timestep
        icfg = cfg["env"]["ik"]
        self.ik = TopDownIK(m, icfg["pos_tol"], icfg["rot_tol"], icfg["max_iters"], icfg["damping"])
        joints = ARM_JOINTS + ["gripper"]
        self.q_adr = np.array([m.jnt_qposadr[m.joint(j).id] for j in joints])
        self.act_ids = np.array([m.actuator(j).id for j in joints])
        self.act_lo = m.actuator_ctrlrange[self.act_ids, 0].copy()
        self.act_hi = m.actuator_ctrlrange[self.act_ids, 1].copy()
        self.tcp = m.site("tcp").id
        robot_root = m.body("base").id
        self.robot_body = np.array([m.body_rootid[b] == robot_root and b != 0 for b in range(m.nbody)])
        self.fixed_jaw_body = m.body("gripper").id
        self.moving_jaw_body = m.body("moving_jaw_so101_v1").id

        self.bowl_names = [b["name"] for b in v["bowls"]]
        self.pad_names = [p["name"] for p in v["pads"]]
        self.bowl_body = [m.body(f"bowl_{n}").id for n in self.bowl_names]
        self.bowl_qadr = [m.jnt_qposadr[m.joint(f"bowl_{n}_free").id] for n in self.bowl_names]
        self.bowl_dadr = [m.jnt_dofadr[m.joint(f"bowl_{n}_free").id] for n in self.bowl_names]
        self.pad_body = [m.body(f"pad_{n}").id for n in self.pad_names]
        self.pad_qadr = [m.jnt_qposadr[m.joint(f"pad_{n}_slide").id] for n in self.pad_names]
        self.pad_dadr = [m.jnt_dofadr[m.joint(f"pad_{n}_slide").id] for n in self.pad_names]
        self.pad_act = [m.actuator(f"pad_{n}_drive").id for n in self.pad_names]
        self.pad_axis = [np.array(p["axis"], float) / np.linalg.norm(p["axis"]) for p in v["pads"]]
        self.pad_top = [p["height"] for p in v["pads"]]
        self.env_geoms = {m.geom("floor").id} | {m.geom(f"pad_{n}_geom").id for n in self.pad_names}
        self._bowl_of_body = {b: i for i, b in enumerate(self.bowl_body)}

        self.home_q, ok, *_ = self.ik.solve(v["expert"]["rest_tcp"])
        assert ok, "home pose unreachable"
        self.cams = list(v["cameras"])
        self.renderer = mujoco.Renderer(m, v["image_size"], v["image_size"]) if render else None
        # nominal values for visual randomization
        self._cam_pos0 = m.cam_pos.copy()
        self._light_pos0 = m.light_pos.copy()
        self._light_diffuse0 = m.light_diffuse.copy()
        self._mat_rgba0 = m.mat_rgba.copy()
        self._world_cams = [m.camera(c).id for c in self.cams if m.cam_bodyid[m.camera(c).id] == 0]

    # ------------------------------------------------------------------ episode setup
    def sample_task(self, seed: int) -> dict:
        """Everything that varies per episode, as a pure function of the seed."""
        v = self.v
        rng = np.random.default_rng([int(seed), 101])
        # bowls: rejection-sample so they do not overlap
        R = v["bowl"]["radius"]
        for _ in range(200):
            pos = [(rng.uniform(*b["spawn"]["x"]), rng.uniform(*b["spawn"]["y"])) for b in v["bowls"]]
            if all(math.dist(pos[i], pos[j]) > 2 * R + 0.012
                   for i in range(len(pos)) for j in range(i)):
                break
        pads = []
        for p in v["pads"]:
            pads.append({"speed": float(rng.uniform(*p["speed"])),
                         "s0": float(rng.uniform(-p["travel"], p["travel"])),
                         "dir": int(rng.choice([-1, 1]))})
        bi, pi = int(rng.integers(len(v["bowls"]))), int(rng.integers(len(v["pads"])))
        text = TEMPLATES[int(rng.integers(len(TEMPLATES)))].format(bowl=self.bowl_names[bi], pad=self.pad_names[pi])
        rz = v["randomize"]
        return {"bowl_xy": [list(map(float, p)) for p in pos],
                "bowl_yaw": [float(rng.uniform(-math.pi, math.pi)) for _ in pos],
                "pads": pads, "bowl": bi, "pad": pi, "instruction": text,
                "light": float(rng.uniform(*rz["light"])),
                "light_dpos": (rng.uniform(-1, 1, 3) * rz["light_pos_jitter"]).tolist(),
                "cam_dpos": (rng.uniform(-1, 1, (len(self._world_cams), 3)) * rz["cam_pos_jitter"]).tolist(),
                "floor_tint": (1 + rng.uniform(-1, 1, 3) * rz["floor_tint"]).tolist()}

    def reset(self, seed: int, task: dict | None = None, randomize_visuals: bool = True):
        m, d, v = self.m, self.d, self.v
        self.seed = int(seed)
        self.task = t = task or self.sample_task(seed)
        self.instruction = t["instruction"]
        self.target_bowl, self.target_pad = t["bowl"], t["pad"]
        # visuals
        m.cam_pos[:] = self._cam_pos0
        m.light_pos[:] = self._light_pos0
        m.light_diffuse[:] = self._light_diffuse0
        m.mat_rgba[:] = self._mat_rgba0
        if randomize_visuals:
            for k, c in enumerate(self._world_cams):
                m.cam_pos[c] = self._cam_pos0[c] + np.array(t["cam_dpos"][k])
            m.light_pos[0] = self._light_pos0[0] + np.array(t["light_dpos"])
            m.light_diffuse[0] = t["light"]
            m.mat_rgba[m.material("grid").id, :3] = np.clip(np.array(t["floor_tint"]), 0, 1.3)

        mujoco.mj_resetData(m, d)
        g = v["gripper"]
        d.qpos[self.q_adr[:5]] = self.home_q
        d.qpos[self.q_adr[5]] = g["open"]
        self.ctrl = np.concatenate([self.home_q, [g["open"]]])
        d.ctrl[self.act_ids] = self.ctrl
        for i, (xy, yaw) in enumerate(zip(t["bowl_xy"], t["bowl_yaw"])):
            a = self.bowl_qadr[i]
            d.qpos[a:a + 7] = [xy[0], xy[1], 0.0005, math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        for i, p in enumerate(t["pads"]):
            d.qpos[self.pad_qadr[i]] = p["s0"]
            d.qvel[self.pad_dadr[i]] = p["dir"] * p["speed"]
            d.ctrl[self.pad_act[i]] = p["s0"]
        self._steps = 0
        self.tick = 0
        self._hold = 0
        self.stats = TickStats(seed=self.seed)
        mujoco.mj_forward(m, d)
        return self.observation()

    # ------------------------------------------------------------------ pad motion
    def pad_ref(self, i: int, t: float) -> float:
        """Triangle wave: constant speed along the rail, reversing at +-travel."""
        A = self.v["pads"][i]["travel"]
        p = self.task["pads"][i]
        u = (p["s0"] + A + p["dir"] * p["speed"] * t) % (4 * A) if p["dir"] > 0 else \
            (A - p["s0"] + p["speed"] * t) % (4 * A)
        tri = u if u <= 2 * A else 4 * A - u          # 0..2A
        s = tri - A
        return s if p["dir"] > 0 else -s

    # ------------------------------------------------------------------ stepping
    def step(self, action):
        """Apply joint targets for one control tick. Returns (obs, done, info)."""
        m, d = self.m, self.d
        a = np.clip(np.asarray(action, float), self.act_lo, self.act_hi)
        prev = self.ctrl
        for k in range(self.n_sub):
            f = (k + 1) / self.n_sub
            d.ctrl[self.act_ids] = prev + f * (a - prev)   # ramp, so 25 Hz targets do not jerk the servos
            t = (self._steps + 1) * m.opt.timestep
            for i, act in enumerate(self.pad_act):
                d.ctrl[act] = self.pad_ref(i, t)
            mujoco.mj_step(m, d)
            self._steps += 1
        self.ctrl = a
        self.tick += 1

        c = self.contacts()
        self.stats.robot_env_ticks += int(c["robot_env"])
        for i in range(len(self.bowl_body)):
            if c["fixed_jaw"][i] and c["moving_jaw"][i]:
                if i == self.target_bowl:
                    self.stats.ever_grasped = True
                else:
                    self.stats.grasped_wrong_bowl = True
        placed, detail = self.placed(c)
        self._hold = self._hold + 1 if placed else 0
        done = False
        if self._hold >= int(round(self.v["success"]["hold_time"] / self.dt)):
            done, self.stats.success = True, True
        elif self.tick >= self.v["max_ticks"]:
            done = True
            self.stats.failure_type = self._failure_type(detail)
        if done:
            self.stats.ticks, self.stats.detail = self.tick, detail
        info = {"success": self.stats.success, "failure_type": self.stats.failure_type} if done else {}
        return self.observation(), done, info

    # ------------------------------------------------------------------ state queries
    def qpos(self):
        return self.d.qpos[self.q_adr].copy()

    def tcp_pos(self):
        return self.d.site_xpos[self.tcp].copy()

    def bowl_pos(self, i):
        a = self.bowl_qadr[i]
        return self.d.qpos[a:a + 3].copy()

    def bowl_vel(self, i):
        a = self.bowl_dadr[i]
        return self.d.qvel[a:a + 3].copy()

    def bowl_up_cos(self, i):
        return float(self.d.xmat[self.bowl_body[i]].reshape(3, 3)[2, 2])

    def pad_pos(self, i):
        return self.d.xpos[self.pad_body[i]][:2].copy()

    def pad_vel(self, i):
        return self.pad_axis[i] * self.d.qvel[self.pad_dadr[i]]

    def contacts(self) -> dict:
        m, d = self.m, self.d
        n = len(self.bowl_body)
        out = {"robot_env": False, "robot_bowl": [False] * n, "fixed_jaw": [False] * n, "moving_jaw": [False] * n}
        for k in range(d.ncon):
            g1, g2 = d.contact.geom[k]
            b1, b2 = m.geom_bodyid[g1], m.geom_bodyid[g2]
            r1, r2 = self.robot_body[b1], self.robot_body[b2]
            if r1 == r2:
                continue
            g_other, b_rob = (g2, b1) if r1 else (g1, b2)
            b_other = m.geom_bodyid[g_other]
            if g_other in self.env_geoms:
                out["robot_env"] = True
            elif b_other in self._bowl_of_body:
                i = self._bowl_of_body[b_other]
                out["robot_bowl"][i] = True
                if b_rob == self.fixed_jaw_body:
                    out["fixed_jaw"][i] = True
                elif b_rob == self.moving_jaw_body:
                    out["moving_jaw"][i] = True
        return out

    def grasped(self, i, c=None) -> bool:
        c = c or self.contacts()
        return bool(c["fixed_jaw"][i] and c["moving_jaw"][i])

    def on_pad(self, bi, pi):
        """(dx, dy) of bowl centre from pad centre, and whether the bowl rests at pad-top height."""
        p = self.bowl_pos(bi)
        rel = p[:2] - self.pad_pos(pi)
        return rel, bool(abs(p[2] - self.pad_top[pi]) < 0.006)

    def placed(self, c=None):
        c = c or self.contacts()
        sc = self.v["success"]
        bi, pi = self.target_bowl, self.target_pad
        rel, resting = self.on_pad(bi, pi)
        vrel = self.bowl_vel(bi) - np.array([*self.pad_vel(pi), 0.0])
        g = self.v["gripper"]
        detail = {
            "offset": [float(rel[0]), float(rel[1])],
            "centered": bool(np.all(np.abs(rel) < sc["center_tol"])),
            "on_pad": resting,
            "upright": bool(self.bowl_up_cos(bi) > math.cos(math.radians(sc["max_tilt_deg"]))),
            "gripper_open": bool(self.d.qpos[self.q_adr[5]] > 0.5 * (g["open"] + g["closed"])),
            "arm_clear": not c["robot_bowl"][bi],
            "moving_with_pad": bool(np.linalg.norm(vrel) < sc["max_rel_speed"]),
        }
        return all(v for k, v in detail.items() if k != "offset"), detail

    def _failure_type(self, detail):
        bi = self.target_bowl
        if not self.stats.ever_grasped:
            return "wrong_bowl" if self.stats.grasped_wrong_bowl else "never_grasped"
        if detail["on_pad"] and detail["centered"]:
            return "not_released" if not (detail["gripper_open"] and detail["arm_clear"]) else "unstable"
        for pi in range(len(self.pad_body)):
            rel, resting = self.on_pad(bi, pi)
            half = np.array(self.v["pads"][pi]["size"]) / 2
            if pi != self.target_pad and resting and np.all(np.abs(rel) < half):
                return "wrong_pad"
        if self.grasped(bi):
            return "still_holding"
        return "missed_pad"

    def state(self) -> dict:
        """Privileged state for the expert / logging."""
        c = self.contacts()
        return {
            "t": self.d.time, "tick": self.tick, "tcp": self.tcp_pos(), "q": self.qpos(),
            "bowls": [self.bowl_pos(i) for i in range(len(self.bowl_body))],
            "pads": [self.pad_pos(i) for i in range(len(self.pad_body))],
            "pad_vel": [self.pad_vel(i) for i in range(len(self.pad_body))],
            "grasped": [self.grasped(i, c) for i in range(len(self.bowl_body))],
            "robot_bowl": c["robot_bowl"], "target_bowl": self.target_bowl, "target_pad": self.target_pad,
        }

    # ------------------------------------------------------------------ observation
    def render(self, cams=None, renderer=None) -> dict:
        r = renderer or self.renderer
        out = {}
        for cam in cams or self.cams:
            r.update_scene(self.d, camera=cam)
            out[cam] = r.render().copy()
        return out

    def observation(self) -> dict:
        obs = {"proprio": self.qpos().astype(np.float32), "instruction": self.instruction}
        if self.renderer is not None:
            obs["images"] = self.render()
        return obs
