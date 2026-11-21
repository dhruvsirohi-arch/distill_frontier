"""Scripted reference controllers. They establish that the task is solvable and calibrate it.

  ScheduleOracle   privileged: knows the release schedule; hands wait under the next sticks.
  Reactive         learns which stick dropped only `latency` seconds after release, either from
                   the simulator (detector="event") or from its own head camera ("vision").
  Parked           both hands wait permanently under two fixed sticks (pure guessing).
  Idle             never moves.
All return an action dict (or None) once per control tick through `act(game_or_obs)`.
"""
from __future__ import annotations

import numpy as np

from .game import TICK
from .robot import READY

G = 9.81
WAIT_Z = 1.40          # pinch height under the hanging sticks (stick bottoms at 1.55 m)
LOW_Z = 1.05           # lowest level grasp the arms reach at the bar radius


def side_for(slot_xy):
    return "left" if slot_xy[1] >= 0 else "right"


def home(side):
    return {"pos": list(READY[side]["pos"][:2]) + [WAIT_Z], "grip": 1.0}


class ScheduleOracle:
    """Privileged plan: before each release, park the free hand under that stick, close shortly
    after release. Uses the schedule only, so it can be submitted as an open-loop trajectory."""
    lead = 0.30
    close_after = 0.16
    hold = 0.26                       # the catch registers about 0.38 s after release

    def __init__(self, game):
        self.slots = game.slots
        self.plan = []                         # (t_start, t_close, t_free, side, slot)
        free = {"left": -1.0, "right": -1.0}
        for slot, tr in zip(game.schedule.order, game.schedule.release_t):
            y = self.slots[slot][1]
            able = [s for s in ("left", "right") if (y > -0.2 if s == "left" else y < 0.2)]
            pref = side_for(self.slots[slot])
            side = min(able, key=lambda s: (max(free[s] - (tr - self.lead), 0), s != pref))
            self.plan.append((tr - self.lead, tr + self.close_after,
                              tr + self.close_after + self.hold, side, slot))
            free[side] = tr + self.close_after + self.hold

    def act(self, game):
        t = game.t
        cmd = {s: home(s) for s in ("left", "right")}
        cmd["base"], cmd["torso_yaw"] = [0.0, 0.0], 0.0
        for side in ("left", "right"):
            mine = [p for p in self.plan if p[3] == side and p[0] <= t < p[2]]
            if mine:
                t0, tc, tf, _, slot = mine[0]
                x, y, _ = self.slots[slot]
                cmd[side] = {"pos": [x, y, WAIT_Z], "grip": 0.0 if t >= tc else 1.0}
            else:
                upcoming = [p for p in self.plan if p[3] == side and p[0] > t]
                if upcoming:                  # drift toward the next assignment early
                    x, y, _ = self.slots[upcoming[0][4]]
                    cmd[side] = {"pos": [x, y, WAIT_Z], "grip": 1.0}
        return cmd


class Reactive:
    """Hands wait centrally; after it knows a stick fell (release + latency), the nearest free
    hand intercepts the stick lower down its path and closes as it passes."""

    def __init__(self, game, latency=0.0, detector="event", stick_length=0.40):
        self.latency = latency
        self.detector = detector
        self.slots = game.slots
        self.bottom0 = game.machine.magnet_z - stick_length
        self.top0 = game.machine.magnet_z
        self.known = {}                       # slot -> release time it believes
        self.job = {"left": None, "right": None}
        self.vision = VisionDetector() if detector == "vision" else None
        self.game = game

    def _detect(self, game, obs):
        t = game.t
        if self.detector == "event":
            for dr in game.drops:
                if dr.release_t + self.latency <= t + 1e-9 and dr.slot not in self.known:
                    self.known[dr.slot] = dr.release_t
        else:
            # Vision: detections are delayed by `latency` (a FIFO of camera frames).
            for slot in self.vision.update(t, obs, self.slots, game):
                self.known.setdefault(slot, None)
                self.vision_pending.append((t + self.latency, slot, t))

    vision_pending: list

    def act(self, game, obs=None):
        t = game.t
        if self.detector == "vision":
            if not hasattr(self, "_vp"):
                self._vp = True
                self.vision_pending = []
            self._detect(game, obs)
            ready = [(slot, t_seen) for (ta, slot, t_seen) in self.vision_pending if ta <= t + 1e-9]
            self.vision_pending = [p for p in self.vision_pending if p[0] > t + 1e-9]
            new = []
            for slot, t_seen in ready:
                # the detector needs about 1 cm of fall (two frames) to see motion
                new.append((slot, t_seen - 2 * game.tick_dt))
        else:
            before = set(self.known)
            self._detect(game, obs)
            new = [(s, self.known[s]) for s in self.known if s not in before]
        for slot, t_rel in new:
            pref = side_for(self.slots[slot])
            other = "right" if pref == "left" else "left"
            reach_other = abs(self.slots[slot][1]) < 0.2
            side = other if (self.job[pref] is not None and self.job[other] is None
                             and reach_other) else pref
            self.job[side] = dict(slot=slot, t_rel=t_rel, t0=t)
        cmd = {"base": [0.0, 0.0], "torso_yaw": 0.0}
        for side in ("left", "right"):
            job = self.job[side]
            if job is None:
                cmd[side] = home(side)
                continue
            x, y, _ = self.slots[job["slot"]]
            if "z" not in job:
                # intercept low enough that the hand (about 0.18 s to arrive) gets there before the
                # stick's bottom; close once ~12 cm of stick has passed between the jaws
                t_arrive = (t - job["t_rel"]) + 0.18
                z = self.bottom0 - 0.5 * G * t_arrive ** 2 - 0.03
                job["z"] = float(np.clip(z, LOW_Z, WAIT_Z))
                t_in = np.sqrt(2 * (self.bottom0 - job["z"] + 0.12) / G)
                job["t_close"] = job["t_rel"] + max(t_in, t_arrive)
            grip = 0.0 if t >= job["t_close"] else 1.0
            cmd[side] = {"pos": [float(x), float(y), job["z"]], "grip": grip}
            if t - job["t_rel"] > 0.9:
                self.job[side] = None
        return cmd


class VisionDetector:
    """Camera-only drop detector. In the first head image it finds each hanging stick (pink
    connected component) and the pixels just below its magnet; a stick has dropped once most of
    those pixels stop being pink. Slots are matched to components by projecting the machine's
    published geometry through the robot's own calibrated head cameras."""

    WIN = 3

    def __init__(self):
        self.ref = None

    @staticmethod
    def pink(rgb):
        r, g, b = (rgb[..., k].astype(int) for k in range(3))
        return (r > 100) & (r - g > 60) & (b > 30) & (r - b > 25)

    def update(self, t, obs, slots, game):
        from scipy import ndimage
        mask = self.pink(obs["rgb"])
        if self.ref is None:
            self.ref = True
            self.anchor = {}        # slot -> (top probe, its pink count, bottom probe)
            lab, n = ndimage.label(mask)
            proj = self._project(game, slots)
            comp_of = {}
            for slot, pts in proj.items():
                hits = [lab[int(v), int(u)] for u, v in pts if lab[int(v), int(u)]]
                if hits:
                    comp_of[slot] = max(set(hits), key=hits.count)
            # leftover components (e.g. partly hidden behind the bar) go to the nearest slot
            for k in set(range(1, n + 1)) - set(comp_of.values()):
                rows, cols = np.nonzero(lab == k)
                if len(rows) < 40:
                    continue
                c = np.array([cols.mean(), rows.mean()])
                free = [s for s in proj if s not in comp_of]
                if free:
                    best = min(free, key=lambda s: np.min(np.linalg.norm(np.array(proj[s]) - c, axis=1)))
                    if np.min(np.linalg.norm(np.array(proj[best]) - c, axis=1)) < 60:
                        comp_of[best] = k
            for slot, k in comp_of.items():
                rows, cols = np.nonzero(lab == k)
                top, bot = rows.min(), rows.max()
                tc = int(round(cols[rows <= top + 3].mean()))
                bc = int(round(cols[rows >= bot - 3].mean()))
                tp = (int(top) + 2, tc)
                bp = (int(bot) + 4, bc)
                self.anchor[slot] = (tp, self._count(mask, *tp), bp)
            self.fired = set()
            return []
        out = []
        for slot, (tp, base, bp) in self.anchor.items():
            if slot in self.fired:
                continue
            if self._count(mask, *tp) < 0.4 * base or self._count(mask, *bp, rows=(0, 5)) >= 10:
                self.fired.add(slot)
                out.append(slot)
        return out

    def _count(self, mask, r, c, rows=(-2, 3), half_w=8):
        """Pink pixels in a window that is wide horizontally: torso and base sway of about a
        degree shifts the image a few pixels sideways when the arms swing."""
        return int(mask[max(r + rows[0], 0):r + rows[1] + 1,
                        max(c - half_w, 0):c + half_w + 1].sum())

    def _project(self, game, slots):
        import mujoco
        m, d = game.m, game.d
        W, H = game.img_w, game.img_h
        out = {}
        for panel, cam in enumerate(("head_left", "head_right")):
            cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam)
            pos, R = d.cam_xpos[cid], d.cam_xmat[cid].reshape(3, 3)
            f = 0.5 * H / np.tan(np.radians(m.cam_fovy[cid]) / 2)
            for s, p in enumerate(slots):
                if s in out:
                    continue
                pts = []
                for z in np.linspace(p[2] - 0.02, p[2] - 0.38, 12):
                    q = R.T @ (np.array([p[0], p[1], z]) - pos)
                    if q[2] >= -0.05:
                        continue
                    u, v = W / 2 + f * q[0] / -q[2], H / 2 - f * q[1] / -q[2]
                    if 0 <= u < W and 0 <= v < H:
                        pts.append((u + panel * W, v))
                if len(pts) >= 4:
                    out[s] = pts
        return out


class Parked:
    """Both hands wait under two fixed sticks (slots 2 and 5) for the whole game: pure guessing.
    Default: jaws open, closed with perfect timing when that stick drops (an upper bound on
    guessing). trap=True: jaws held at 35% (narrower than the stick) with no timing at all."""
    trap = False

    def __init__(self, game, slots=(2, 5)):
        self.slots = game.slots
        self.pick = slots

    def act(self, game, obs=None):
        cmd = {"base": [0.0, 0.0], "torso_yaw": 0.0}
        for side, slot in zip(("left", "right"), self.pick):
            x, y, _ = self.slots[slot]
            if self.trap:
                grip = 0.35
            else:
                dr = game.drop_of_slot[slot]
                grip = 0.0 if game.t >= dr.release_t + ScheduleOracle.close_after else 1.0
            cmd[side] = {"pos": [x, y, WAIT_Z], "grip": grip}
        return cmd


class Idle:
    def __init__(self, game):
        pass

    def act(self, game, obs=None):
        return None
