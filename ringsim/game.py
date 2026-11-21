"""Reaction Ring game: sticks drop one at a time in a random order; catch each before it lands.

Rules (as in the arcade machine, difficulty 1, hints off):
  * A 3 s countdown, then each stick is released once, in a seeded random order, 0.5-1.5 s apart.
  * A stick counts as caught when one gripper holds it (both jaw pads in contact) for 0.2 s
    without it having touched the floor. A stick already in a gripper's jaws (pad contact) when
    its magnet releases is a foul and does not score.
  * The stick must fall into open jaws: if its bottom end enters a gripper whose jaws are already
    narrower than 36 mm (the stick is 32 mm thick), a resulting hold is "trapped" and does not
    score. Without this rule a hand parked under a stick with half-closed jaws catches it with no
    reaction at all (the stick wedges between the fingertips).
  * A stick that touches the floor, or is not caught within 1.5 s, is a miss.
  * Caught and missed sticks are cleared from the play area (an attendant collects them).

The judge, the schedule and the stick poses are simulator-side only. Sensor-mode agents receive
head RGB-D images and proprioception, nothing else.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import mujoco
import numpy as np

from .robot import Controller
from .scene import SIDES, Machine, compile_model

PHYS_DT = 0.002
TICK = 0.04                       # control period, 25 Hz
SUBSTEPS = int(round(TICK / PHYS_DT))
COUNTDOWN = 3.0
INTERVAL = (0.5, 1.5)             # manual difficulty 1
HOLD_TIME = 0.20
OPEN_GAP = 0.036                  # jaws must be at least this open when the stick enters them
STICK_HALF = 0.20
MISS_TIMEOUT = 1.5
CLEAR_DELAY = 0.4
IMG_W, IMG_H = 320, 240           # per head camera


@dataclass
class Drop:
    slot: int
    release_t: float
    outcome: str = "pending"      # pending | caught | floor | timeout | foul | trapped
    outcome_t: float | None = None
    hand: str | None = None
    first_pad_contact_t: float | None = None
    entry_gap_mm: dict = field(default_factory=dict)   # side -> jaw gap when the stick entered


@dataclass
class Schedule:
    order: list[int]
    release_t: list[float]

    @classmethod
    def sample(cls, seed: int, n: int):
        rng = np.random.default_rng(seed)
        order = rng.permutation(n).tolist()
        gaps = rng.uniform(*INTERVAL, size=n)
        t = COUNTDOWN + np.cumsum(gaps)
        # releases land on the control grid so every mode sees them at the same tick boundary
        return cls(order=order, release_t=[round(round(x / TICK) * TICK, 4) for x in t])


class Game:
    def __init__(self, machine: Machine = Machine(), render: bool = True,
                 img_size=(IMG_W, IMG_H)):
        self.machine = machine
        self.spec, self.m = compile_model(machine)
        self.d = mujoco.MjData(self.m)
        self.ctrl = Controller(self.m)
        self.n = machine.n_sticks
        self.render_enabled = render
        self.renderer = None
        self.img_w, self.img_h = img_size
        name = lambda t, n: mujoco.mj_name2id(self.m, t, n)
        G, B, J = mujoco.mjtObj.mjOBJ_GEOM, mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_JOINT
        self.stick_geom = np.array([name(G, f"stick{i}") for i in range(self.n)])
        self.stick_body = np.array([name(B, f"stick{i}") for i in range(self.n)])
        self.stick_qadr = np.array([self.m.jnt_qposadr[name(J, f"stick{i}")] for i in range(self.n)])
        self.stick_dadr = np.array([self.m.jnt_dofadr[name(J, f"stick{i}")] for i in range(self.n)])
        self.geom_to_stick = {g: i for i, g in enumerate(self.stick_geom)}
        self.floor = name(G, "floor")
        self.pad = {}
        for side in SIDES:
            for k in (1, 2):
                self.pad[name(G, f"{side}_pad{k}")] = (side, k)
        self.slots = machine.slots()
        self.finger_q = {side: [self.m.jnt_qposadr[self.m.actuator_trnid[f, 0]]
                                for f in self.ctrl.arm[side]["fingers"]] for side in SIDES}
        self.pinch_site = {side: self.ctrl.arm[side]["site"] for side in SIDES}

    # ---- episode control --------------------------------------------------------------------------
    def reset(self, seed: int):
        self.seed = seed
        self.schedule = Schedule.sample(seed, self.n)
        mujoco.mj_resetData(self.m, self.d)
        self.d.eq_active[:] = 1
        mujoco.mj_forward(self.m, self.d)
        self.ctrl.reset(self.d)
        self.tick = 0
        self.drops = [Drop(slot=s, release_t=t)
                      for s, t in zip(self.schedule.order, self.schedule.release_t)]
        self.state = ["hanging"] * self.n          # hanging | falling | resolved | cleared
        self.drop_of_slot = {dr.slot: dr for dr in self.drops}
        self.hold_since = [None] * self.n
        self.hold_side = [None] * self.n
        self.floor_touched = [False] * self.n
        self.resolved_at = [None] * self.n
        self.actions = []                            # (tick, action) applied, for replay
        self.end_t = self.schedule.release_t[-1] + MISS_TIMEOUT + 0.2
        return self.observe() if self.render_enabled else None

    @property
    def t(self):
        return self.tick * TICK

    @property
    def done(self):
        return self.t >= self.end_t - 1e-9 or all(s in ("resolved", "cleared") for s in self.state)

    def step(self, action: dict | None = None, ticks: int = 1, observe: bool = True):
        """Apply `action` (merged into the held command) and advance `ticks` control periods."""
        if action:
            self.actions.append((self.tick, action))
        for k in range(ticks):
            if self.done:
                break
            self.ctrl.set_command(self.d, action if k == 0 else None)
            for _ in range(SUBSTEPS):
                self._release_due()
                self.ctrl.drive(self.d)
                mujoco.mj_step(self.m, self.d)
                self._judge()
            self.tick += 1
            self._clear_resolved()
        obs = self.observe() if (observe and self.render_enabled) else None
        return obs, self.done

    # ---- referee ----------------------------------------------------------------------------------
    def _release_due(self):
        t = self.d.time
        for dr in self.drops:
            i = dr.slot
            if self.state[i] == "hanging" and t >= dr.release_t - 1e-9:
                self.state[i] = "falling"
                self.d.eq_active[i] = 0
                if self._pad_contacts(i):
                    dr.outcome, dr.outcome_t = "foul", t
                    self.state[i] = "resolved"
                    self.resolved_at[i] = t

    def _pad_contacts(self, i):
        sides = {}
        g = self.stick_geom[i]
        for c in self.d.contact[:self.d.ncon]:
            other = c.geom2 if c.geom1 == g else c.geom1 if c.geom2 == g else None
            if other is not None and other in self.pad:
                side, k = self.pad[other]
                sides.setdefault(side, set()).add(k)
        return sides

    def _judge(self):
        t = self.d.time
        con = self.d.contact[:self.d.ncon]
        touching = {}
        for c in con:
            for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if a in self.geom_to_stick:
                    i = self.geom_to_stick[a]
                    if b == self.floor:
                        self.floor_touched[i] = True
                    elif b in self.pad:
                        side, k = self.pad[b]
                        touching.setdefault(i, {}).setdefault(side, set()).add(k)
        for dr in self.drops:
            i = dr.slot
            if self.state[i] != "falling":
                continue
            self._entry_check(dr)
            if i in touching and dr.first_pad_contact_t is None:
                dr.first_pad_contact_t = round(t, 4)
            if self.floor_touched[i]:
                dr.outcome, dr.outcome_t = "floor", round(t, 4)
            elif t - dr.release_t > MISS_TIMEOUT:
                dr.outcome, dr.outcome_t = "timeout", round(t, 4)
            else:
                held = [s for s, ks in touching.get(i, {}).items() if ks == {1, 2}]
                if held:
                    if self.hold_since[i] is None or self.hold_side[i] != held[0]:
                        self.hold_since[i], self.hold_side[i] = t, held[0]
                    elif t - self.hold_since[i] >= HOLD_TIME - 1e-9:
                        gap = dr.entry_gap_mm.get(held[0])
                        ok = gap is None or gap >= OPEN_GAP * 1000
                        dr.outcome = "caught" if ok else "trapped"
                        dr.outcome_t, dr.hand = round(t, 4), held[0]
                else:
                    self.hold_since[i] = None
                    self.hold_side[i] = None
            if dr.outcome != "pending":
                self.state[i] = "resolved"
                self.resolved_at[i] = t

    def _entry_check(self, dr):
        """Record each gripper's jaw gap at the moment the stick's bottom end drops past the top
        of its pads while the stick is between the fingers."""
        i = dr.slot
        b = self.stick_body[i]
        axis = self.d.xmat[b].reshape(3, 3)[:, 2]
        bottom = self.d.xpos[b] - STICK_HALF * axis * np.sign(axis[2] or 1)
        for side in SIDES:
            if side in dr.entry_gap_mm:
                continue
            p = self.d.site_xpos[self.pinch_site[side]]
            if np.hypot(*(bottom[:2] - p[:2])) < 0.045 and p[2] - 0.015 < bottom[2] < p[2] + 0.015:
                gap = sum(self.d.qpos[q] for q in self.finger_q[side]) - 0.002
                dr.entry_gap_mm[side] = round(1000 * gap, 1)

    def _clear_resolved(self):
        for i in range(self.n):
            if self.state[i] == "resolved":
                dr = self.drop_of_slot[i]
                delay = 0.0 if dr.outcome == "caught" else CLEAR_DELAY
                if self.d.time - self.resolved_at[i] >= delay - 1e-9:
                    q = self.stick_qadr[i]
                    # lay it on the floor in the attendant's bin behind the robot
                    self.d.qpos[q:q + 3] = [-2.0 - 0.05 * i, 1.5, 0.05]
                    self.d.qpos[q + 3:q + 7] = [0.7071, 0.7071, 0, 0]
                    self.d.qvel[self.stick_dadr[i]:self.stick_dadr[i] + 6] = 0
                    self.state[i] = "cleared"
        mujoco.mj_forward(self.m, self.d)

    # ---- sensors ----------------------------------------------------------------------------------
    def _render(self, cam, depth=False):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.m, self.img_h, self.img_w)
        r = self.renderer
        if depth:
            r.enable_depth_rendering()
        r.update_scene(self.d, cam)
        out = r.render().copy()
        if depth:
            r.disable_depth_rendering()
        return out

    def observe(self):
        rgb = np.concatenate([self._render("head_left"), self._render("head_right")], axis=1)
        depth = np.concatenate([self._render("head_left", True),
                                self._render("head_right", True)], axis=1).astype(np.float32)
        return dict(t=round(self.t, 3), rgb=rgb, depth=depth, **self.proprio())

    def proprio(self):
        m, d, c = self.m, self.d, self.ctrl
        joints = {}
        for side in SIDES:
            a = c.arm[side]
            joints[side + "_arm"] = np.round(d.qpos[a["qadr"]], 4).tolist()
        joints["torso_yaw"] = round(float(d.qpos[c.torso_q]), 4)
        xy, yaw = c.base_pose(d)
        pinch, grip = {}, {}
        for side in SIDES:
            pinch[side] = np.round(c.to_base(d, d.site_xpos[c.arm[side]["site"]]), 4).tolist()
            fq = [d.qpos[m.jnt_qposadr[m.actuator_trnid[f, 0]]] for f in c.arm[side]["fingers"]]
            grip[side] = round(float(np.mean(fq) / 0.041), 3)
        v = d.qvel[c.base_v]
        return dict(joints=joints,
                    odometry=dict(x=round(float(xy[0]), 4), y=round(float(xy[1]), 4),
                                  yaw=round(yaw, 4),
                                  v=round(float(np.hypot(v[0], v[1])), 3), w=round(float(v[2]), 3)),
                    pinch=pinch, grip=grip)

    # ---- privileged access (privileged mode and baselines only) -------------------------------------
    def privileged_state(self):
        sticks = []
        for i in range(self.n):
            sticks.append(dict(slot=i, state=self.state[i],
                               pos=np.round(self.d.xpos[self.stick_body[i]], 4).tolist(),
                               vel_z=round(float(self.d.qvel[self.stick_dadr[i] + 2]), 3)))
        return dict(t=round(self.t, 3), sticks=sticks,
                    schedule=[dict(slot=s, release_t=t) for s, t in
                              zip(self.schedule.order, self.schedule.release_t)],
                    slots=np.round(self.slots, 4).tolist(), robot=self.proprio())

    def result(self):
        caught = sum(dr.outcome == "caught" for dr in self.drops)
        return dict(seed=self.seed, caught=caught, drops=len(self.drops),
                    events=[asdict(dr) for dr in self.drops])
