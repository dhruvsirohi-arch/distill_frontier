"""Action perturbation for data collection (DART-style): the perturbed action is executed, the
expert's clean action is what gets recorded, so the dataset contains recoveries from off-nominal
states. Light white noise every tick plus occasional short pushes on the arm joints."""
from __future__ import annotations

import numpy as np


class BurstNoise:
    def __init__(self, seed: int, white_std=0.004, burst_prob=0.02, burst_len=(3, 8), burst_mag=0.08):
        self.rng = np.random.default_rng([int(seed), 13])
        self.white_std, self.burst_prob, self.burst_len, self.burst_mag = white_std, burst_prob, burst_len, burst_mag
        self.left, self.offset = 0, np.zeros(6)

    def __call__(self, a: np.ndarray) -> np.ndarray:
        if self.left == 0 and self.rng.random() < self.burst_prob:
            self.left = int(self.rng.integers(self.burst_len[0], self.burst_len[1] + 1))
            self.offset = np.zeros(6)
            self.offset[:5] = self.rng.uniform(-1, 1, 5) * self.burst_mag
        n = np.zeros(6)
        n[:5] = self.rng.normal(0, self.white_std, 5)
        if self.left > 0:
            self.left -= 1
            n += self.offset
        return a + n
