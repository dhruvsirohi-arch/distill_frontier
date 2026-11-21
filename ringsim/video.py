"""Render an episode to MP4 from spectator cameras (plus the robot's head view)."""
from __future__ import annotations

import imageio.v2 as imageio
import mujoco
import numpy as np


class Recorder:
    def __init__(self, game, path, cams=("overview", "front"), size=(640, 480), head=True,
                 fps=25):
        self.g, self.cams, self.head = game, cams, head
        self.r = mujoco.Renderer(game.m, size[1], size[0])
        self.hr = mujoco.Renderer(game.m, size[1] // 2, size[0] // 2) if head else None
        self.w = imageio.get_writer(path, fps=fps, codec="libx264", quality=7,
                                    macro_block_size=8)

    def frame(self):
        tiles = []
        for c in self.cams:
            self.r.update_scene(self.g.d, c)
            tiles.append(self.r.render())
        top = np.concatenate(tiles, axis=1)
        if self.head:
            hs = []
            for c in ("head_left", "head_right"):
                self.hr.update_scene(self.g.d, c)
                hs.append(self.hr.render())
            h = np.concatenate(hs, axis=1)
            pad = np.zeros((h.shape[0], top.shape[1] - h.shape[1], 3), np.uint8) + 20
            top = np.concatenate([top, np.concatenate([h, pad], axis=1)], axis=0)
        self._overlay(top)
        self.w.append_data(top)

    def _overlay(self, img):
        # score bar: one cell per drop, green caught / red missed / grey pending
        g = self.g
        for k, dr in enumerate(g.drops):
            col = {"caught": (40, 200, 70), "pending": (90, 90, 90)}.get(dr.outcome, (220, 50, 50))
            if dr.outcome == "pending" and g.state[dr.slot] == "falling":
                col = (240, 200, 40)
            img[6:20, 8 + k * 18:22 + k * 18] = col
        t = int(min(g.t / max(g.end_t, 1e-6), 1) * (img.shape[1] - 16))
        img[24:27, 8:8 + t] = (200, 200, 200)

    def close(self):
        self.w.close()
