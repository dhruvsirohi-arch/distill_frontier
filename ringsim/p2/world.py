"""Phase 2 world: the Reaction Ring game stepped one 30 fps camera frame at a time, with a sensor
bus that holds the latest message on every topic plus a short camera history.

  physics      480 Hz (16 steps per frame)
  control      30 Hz: a command takes effect at the next frame boundary
  camera       30 Hz head RGB-D (both panels)
  lidar        10 Hz: each scan is swept over three frames and published, stamped, on the third
  proprio      30 Hz joint states, IMU, odometry, end-effector TF
"""
from __future__ import annotations

from collections import deque

import numpy as np

from ..game import Game
from .sensors import Mid360, Proprio

FPS = 30
FRAME = 1.0 / FPS
PHYS = 1.0 / 480
LIDAR_EVERY = 3
HISTORY = 12                      # frames kept for clips (0.4 s)


class World:
    def __init__(self, seed: int, render: bool = True):
        self.game = Game(render=render, tick=FRAME, phys_dt=PHYS)
        self.game.reset(seed)
        self.seed = seed
        self.lidar = Mid360(self.game, seed)
        self.proprio = Proprio(self.game, seed)
        self.frames: deque = deque(maxlen=HISTORY)
        self.frame_no = 0
        self.scan = None
        self.render = render
        self._capture()

    @property
    def t(self):
        return self.game.t

    @property
    def done(self):
        return self.game.done

    def _capture(self):
        g = self.game
        if self.render:
            o = g.observe()
            self.frames.append(dict(frame=self.frame_no, t=round(g.t, 4), rgb=o["rgb"],
                                    depth=o["depth"]))
        if self.scan is None:
            self.scan = self.lidar.scan()              # first frame: a complete scan
        else:
            done = self.lidar.sweep(self.frame_no % LIDAR_EVERY, LIDAR_EVERY)
            if done is not None:
                self.scan = done
        p = self.proprio
        self.msgs = dict(joint_states=p.joint_state(), imu=p.imu(), odom=p.odom(), tf=p.tf())

    def step(self, action=None):
        """Apply `action` at this frame boundary and advance one frame; sensors publish."""
        self.game.step(action, ticks=1, observe=False)
        self.frame_no += 1
        self._capture()

    def snapshot(self, clip=(3, 6)):
        """Everything the agent may see now. Arrays are never mutated after publication, so the
        snapshot can be handed to another thread."""
        cur = self.frames[-1]
        hist = list(self.frames)
        past = [hist[max(len(hist) - 1 - k, 0)] for k in clip]
        return dict(frame=self.frame_no, t=round(self.t, 4), rgb=cur["rgb"], depth=cur["depth"],
                    clip=[(f["t"], f["rgb"]) for f in past], scan=self.scan,
                    held=_copy_cmd(self.game.ctrl.cmd), **self.msgs)


def _copy_cmd(c):
    return {"base": list(c["base"]), "torso_yaw": float(c["torso_yaw"]),
            "left": {"pos": [float(v) for v in c["left"]["pos"]], "grip": float(c["left"]["grip"])},
            "right": {"pos": [float(v) for v in c["right"]["pos"]],
                      "grip": float(c["right"]["grip"])}}
