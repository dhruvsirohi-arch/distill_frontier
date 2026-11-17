"""SO-101 bowl pick-and-place environment with a high-level JSON action space.

Actions (dicts, one per decision step):
  {"action": "move_to", "x": .., "y": .., "z": ..}   straight-line Cartesian TCP move, top-down
  {"action": "open_gripper"} / {"action": "close_gripper"} / {"action": "done"}

The scripted controller (IK + interpolation) executes each primitive in MuJoCo. Rendering hooks
(`viz`) only *read* MjModel/MjData, so visualization never changes simulation results.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from env.ik import ARM_JOINTS, TopDownIK
from env.scene import PAN_AXIS_XY, build_model

RESULTS = ("success", "failed", "collision", "rejected_unreachable", "invalid")


def r3(v):
    return round(float(v), 3)


def bowl_yaw_deg(quat) -> float:
    w, x, y, z = quat
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def spawn_for_seed(seed: int, cfg: dict):
    """Bowl spawn (x, y, yaw_deg) for a seed. Cheap: lets curation pick seeds for thin bins."""
    rng = np.random.default_rng(int(seed))
    sp = cfg["env"]["spawn"]
    bx, by = rng.uniform(*sp["x"]), rng.uniform(*sp["y"])
    return float(bx), float(by), float(rng.uniform(*sp["yaw_deg"]))


@dataclass
class EpisodeStats:
    seed: int = 0
    path_length: float = 0.0
    n_collisions: int = 0
    n_rejected: int = 0
    n_failed_grasps: int = 0
    n_invalid: int = 0
    ever_grasped: bool = False
    dropped: bool = False
    done_called: bool = False
    success: bool = False
    failure_type: str = ""
    success_detail: dict = field(default_factory=dict)


class BowlEnv:
    def __init__(self, cfg: dict, viz=None):
        self.cfg = cfg
        self.ecfg = cfg["env"]
        self.m = build_model(cfg)
        self.d = mujoco.MjData(self.m)
        self.viz = viz
        icfg = self.ecfg["ik"]
        self.ik = TopDownIK(self.m, icfg["pos_tol"], icfg["rot_tol"], icfg["max_iters"], icfg["damping"])
        m = self.m
        self.arm_qadr = self.ik.qadr
        self.arm_act = np.array([m.actuator(j).id for j in ARM_JOINTS])
        self.grip_act = m.actuator("gripper").id
        self.grip_qadr = m.jnt_qposadr[m.joint("gripper").id]
        self.bowl_body = m.body("bowl").id
        self.bowl_qadr = m.jnt_qposadr[m.joint("bowl_free").id]
        self.bowl_dadr = m.jnt_dofadr[m.joint("bowl_free").id]
        self.tcp = m.site("tcp").id
        robot_root = m.body("base").id
        self.robot_body = np.array([m.body_rootid[b] == robot_root and b != 0 for b in range(m.nbody)])
        self.fixed_jaw_body = m.body("gripper").id
        self.moving_jaw_body = m.body("moving_jaw_so101_v1").id
        self.env_geoms = {m.geom("floor").id, m.geom("table").id}
        self.table_top = self.ecfg["table"]["height"]
        self.n_sub_per_frame = None
        self.home_q, ok, *_ = self.ik.solve([0.16, -0.06, 0.15])
        assert ok, "home pose unreachable"
        self.rng = np.random.default_rng(0)

    # ------------------------------------------------------------------ state queries
    def tcp_pos(self):
        return self.d.site_xpos[self.tcp].copy()

    def bowl_pose(self):
        q = self.d.qpos[self.bowl_qadr:self.bowl_qadr + 7]
        return q[:3].copy(), q[3:7].copy()

    def bowl_up_cos(self):
        return float(self.d.xmat[self.bowl_body].reshape(3, 3)[2, 2])

    def _contacts(self):
        """Return flags: robot-env, robot-bowl, fixed-jaw-bowl, moving-jaw-bowl."""
        m, d = self.m, self.d
        robot_env = robot_bowl = fj = mj = False
        for i in range(d.ncon):
            g1, g2 = d.contact.geom[i]
            b1, b2 = m.geom_bodyid[g1], m.geom_bodyid[g2]
            r1, r2 = self.robot_body[b1], self.robot_body[b2]
            if r1 == r2:
                continue
            g_other, b_rob = (g2, b1) if r1 else (g1, b2)
            b_other = m.geom_bodyid[g_other]
            if g_other in self.env_geoms:
                robot_env = True
            elif b_other == self.bowl_body:
                robot_bowl = True
                if b_rob == self.fixed_jaw_body:
                    fj = True
                elif b_rob == self.moving_jaw_body:
                    mj = True
        return robot_env, robot_bowl, fj, mj

    def is_grasped(self):
        _, _, fj, mj = self._contacts()
        return bool(fj and mj and self.gripper_closed)

    # ------------------------------------------------------------------ simulation core
    def _step_sim(self, n, monitor=None):
        """Advance n physics steps. `monitor(step_idx)` may return True to abort early."""
        fps = self.cfg["viz"]["fps"]
        every = max(1, int(round(1.0 / (fps * self.m.opt.timestep))))
        for k in range(n):
            mujoco.mj_step(self.m, self.d)
            self._t_steps += 1
            if self.viz is not None and self._t_steps % every == 0:
                self.viz.on_frame(self.m, self.d, self._viz_info())
            if monitor is not None and monitor(k):
                return k + 1
        return n

    def _viz_info(self):
        return {"episode_id": self.episode_id, "seed": self.seed, "step": self.step_idx,
                "action": self._cur_action, "policy": self.policy_name}

    def _secs(self, t):
        return max(1, int(round(t / self.m.opt.timestep)))

    # ------------------------------------------------------------------ episode API
    def reset(self, seed: int, episode_id: str = "", policy_name: str = ""):
        self.seed, self.episode_id, self.policy_name = int(seed), episode_id or f"seed{seed}", policy_name
        bx, by, yaw_deg = spawn_for_seed(seed, self.cfg)
        yaw = math.radians(yaw_deg)
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[self.arm_qadr] = self.home_q
        self.d.qpos[self.grip_qadr] = self.ecfg["gripper"]["closed"]
        self.d.qpos[self.bowl_qadr:self.bowl_qadr + 3] = [bx, by, 0.0005]
        self.d.qpos[self.bowl_qadr + 3:self.bowl_qadr + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        self.d.ctrl[self.arm_act] = self.home_q
        self.d.ctrl[self.grip_act] = self.ecfg["gripper"]["closed"]
        self.gripper_closed = True
        self.step_idx = 0
        self.last_result = "none"
        self._cur_action = None
        self._t_steps = 0
        self.stats = EpisodeStats(seed=int(seed))
        self.spawn = {"x": float(bx), "y": float(by), "yaw_deg": float(math.degrees(yaw))}
        mujoco.mj_forward(self.m, self.d)
        if self.viz is not None:
            self.viz.on_episode_start(self.m, self.d, self._viz_info())
        self._step_sim(self._secs(0.3))
        self._was_grasped = False
        return self.observation()

    def observation(self) -> dict:
        p, q = self.bowl_pose()
        e = self.tcp_pos()
        tg = self.ecfg["target"]
        return {
            "step": self.step_idx,
            "bowl": {"x": r3(p[0]), "y": r3(p[1]), "z": r3(p[2]), "yaw": int(round(bowl_yaw_deg(q)))},
            "target": {"x": r3(tg["center"][0]), "y": r3(tg["center"][1]), "z": r3(self.table_top),
                       "sx": r3(tg["size"][0]), "sy": r3(tg["size"][1])},
            "ee": {"x": r3(e[0]), "y": r3(e[1]), "z": r3(e[2])},
            "gripper": "closed" if self.gripper_closed else "open",
            "grasped": self.is_grasped(),
            "last_result": self.last_result,
        }

    def step(self, action: dict | None):
        """Execute one decision step. `action=None` means invalid output (counts as a failed step)."""
        self._cur_action = action
        if self.viz is not None:
            self.viz.on_action(action, self._viz_info())
        done = False
        if action is None:
            result = "invalid"
            self.stats.n_invalid += 1
        else:
            kind = action["action"]
            if kind == "move_to":
                result = self._move_to(np.array([action["x"], action["y"], action["z"]], float))
            elif kind == "open_gripper":
                result = self._gripper(False)
            elif kind == "close_gripper":
                result = self._gripper(True)
            elif kind == "done":
                result = "success"
                done = True
                self.stats.done_called = True
            else:
                result = "invalid"
        # drop detection: bowl was held, gripper still closed, no longer held
        grasped = self.is_grasped()
        if self._was_grasped and self.gripper_closed and not grasped:
            self.stats.dropped = True
        self._was_grasped = grasped
        self.stats.ever_grasped |= grasped
        self.step_idx += 1
        self.last_result = result
        if done:
            self._finish()
        elif self.step_idx >= self.ecfg["max_decision_steps"]:
            done = True
            self._finish(step_limit=True)
        info = {"result": result}
        if done:
            info["success"] = self.stats.success
            info["failure_type"] = self.stats.failure_type
            if self.viz is not None:
                self.viz.on_episode_end(self.m, self.d, self._viz_info(), self.stats)
        return self.observation(), done, info

    # ------------------------------------------------------------------ primitives
    def reachable(self, target):
        rc = self.ecfg["reach"]
        r = float(np.linalg.norm(target[:2] - PAN_AXIS_XY))
        if not (rc["z"][0] <= target[2] <= rc["z"][1] and rc["r"][0] <= r <= rc["r"][1]):
            return False, None
        q, ok, *_ = self.ik.solve(target, self.d.qpos[self.arm_qadr])
        return ok, q

    def _move_to(self, target):
        if not np.all(np.isfinite(target)):
            self.stats.n_rejected += 1
            return "rejected_unreachable"
        ok, q_goal = self.reachable(target)
        if not ok:
            self.stats.n_rejected += 1
            return "rejected_unreachable"
        mc = self.ecfg["motion"]
        start = self.tcp_pos()
        dist = float(np.linalg.norm(target - start))
        n_wp = max(1, int(math.ceil(dist / mc["waypoint_spacing"])))
        qs = [self.d.qpos[self.arm_qadr].copy()]
        for i in range(1, n_wp + 1):
            p = start + (target - start) * i / n_wp
            q, ok, *_ = self.ik.solve(p, qs[-1])
            if not ok:  # straight line leaves the top-down workspace
                self.stats.n_rejected += 1
                return "rejected_unreachable"
            qs.append(q)
        qs = np.array(qs)
        T = max(mc["min_move_time"], dist / mc["cart_speed"])
        n = self._secs(T)
        hit = {"flag": False}

        def monitor(k):
            s = (k + 1) / n
            s = 10 * s**3 - 15 * s**4 + 6 * s**5  # min-jerk time scaling
            f = s * n_wp
            i = min(int(f), n_wp - 1)
            a = f - i
            self.d.ctrl[self.arm_act] = (1 - a) * qs[i] + a * qs[i + 1]
            if self._collision_now():
                hit["flag"] = True
                return True
            return False

        prev = self.tcp_pos()
        self._move_bowl_start = self.bowl_pose()[0]
        self._step_sim(n, monitor)
        if not hit["flag"]:
            def settle_monitor(k):
                if self._collision_now():
                    hit["flag"] = True
                    return True
                return False

            self._step_sim(self._secs(mc["settle_time"]), settle_monitor)
        if hit["flag"]:  # freeze where we are
            self.d.ctrl[self.arm_act] = self.d.qpos[self.arm_qadr]
            self._step_sim(self._secs(0.1))
        self.stats.path_length += float(np.linalg.norm(self.tcp_pos() - prev))
        if hit["flag"]:
            self.stats.n_collisions += 1
            return "collision"
        return "success"

    def _collision_now(self):
        robot_env, robot_bowl, _, _ = self._contacts()
        if robot_env:
            return True
        # with an open gripper, contact that disturbs the bowl (pushes or tips it) = hitting the rim
        if robot_bowl and not self.gripper_closed:
            p, _ = self.bowl_pose()
            moved = np.linalg.norm(p - self._move_bowl_start) > 0.004
            tipped = self.bowl_up_cos() < math.cos(math.radians(5))
            return bool(moved or tipped)
        return False

    def _gripper(self, close: bool):
        g = self.ecfg["gripper"]
        self.gripper_closed = close
        self.d.ctrl[self.grip_act] = g["closed"] if close else g["open"]
        self._step_sim(self._secs(self.ecfg["motion"]["gripper_time"]))
        if close:
            if self.is_grasped():
                return "success"
            self.stats.n_failed_grasps += 1
            return "failed"
        return "success"

    # ------------------------------------------------------------------ success check
    def placement_margin(self, p=None):
        """Signed distance from bowl center to the nearest target-region edge (positive = inside)."""
        if p is None:
            p, _ = self.bowl_pose()
        tg = self.ecfg["target"]
        dx = tg["size"][0] / 2 - abs(p[0] - tg["center"][0])
        dy = tg["size"][1] / 2 - abs(p[1] - tg["center"][1])
        return float(min(dx, dy))

    def _finish(self, step_limit=False):
        sc = self.ecfg["success"]
        n = self._secs(sc["settle_time"])
        max_speed = 0.0
        touched = False

        def mon(k):
            nonlocal max_speed, touched
            v = self.d.qvel[self.bowl_dadr:self.bowl_dadr + 3]
            max_speed = max(max_speed, float(np.linalg.norm(v)))
            touched |= self._contacts()[1]
            return False

        self._step_sim(n, mon)
        p, _ = self.bowl_pose()
        detail = {
            "margin": self.placement_margin(p),
            "on_table": bool(abs(p[2] - self.table_top) < 0.006),
            "upright": bool(self.bowl_up_cos() > math.cos(math.radians(sc["max_tilt_deg"]))),
            "gripper_open": not self.gripper_closed,
            "arm_clear": not touched,
            "stable": max_speed < sc["max_speed"],
        }
        detail["in_region"] = detail["margin"] > 0
        ok = all(v for k, v in detail.items() if k != "margin") and not step_limit
        self.stats.success = bool(ok)
        self.stats.success_detail = detail
        if ok:
            self.stats.failure_type = ""
        elif self.stats.dropped:
            self.stats.failure_type = "dropped_bowl"
        elif not self.stats.ever_grasped:
            self.stats.failure_type = "failed_grasp"
        elif step_limit:
            self.stats.failure_type = "step_limit"
        else:
            self.stats.failure_type = "missed_target"
