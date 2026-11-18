"""Dataset over the per-episode HDF5 files written by vla/generate.py."""
from __future__ import annotations

import io
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image


# grounding targets: xy of the named bowl and the named pad, roughly unit scale
AUX_CENTER = np.array([0.18, 0.0, 0.18, 0.0], np.float32)
AUX_SCALE = 0.1


def decode(buf) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(bytes(buf))).convert("RGB"))


def list_episodes(root) -> list[Path]:
    root = Path(root)
    man = json.loads((root / "manifest.json").read_text())
    return [root / e["file"] for e in man["episodes"] if e["file"]]


def action_targets(action: np.ndarray, proprio: np.ndarray, t: int, chunk: int) -> np.ndarray:
    """Chunk of actions starting at tick t, padded by repeating the last one. Arm joints are
    expressed relative to the current joint positions, the gripper target stays absolute."""
    idx = np.minimum(np.arange(t, t + chunk), len(action) - 1)
    a = action[idx].copy()
    a[:, :5] -= proprio[t, :5]
    return a


def compute_stats(files, chunk: int) -> dict:
    acts, pros = [], []
    for f in files:
        with h5py.File(f, "r") as h:
            a, p = h["action"][:], h["proprio"][:]
        pros.append(p)
        acts.append(np.stack([action_targets(a, p, t, chunk) for t in range(0, len(a), 2)]))
    acts, pros = np.concatenate(acts).reshape(-1, a.shape[1]), np.concatenate(pros)
    floor = 0.02        # rad; joints that barely move (wrist roll) must not have their noise blown up
    return {"action_mean": acts.mean(0).tolist(), "action_std": np.maximum(acts.std(0), floor).tolist(),
            "proprio_mean": pros.mean(0).tolist(), "proprio_std": np.maximum(pros.std(0), floor).tolist()}


class EpisodeDataset(torch.utils.data.Dataset):
    """One sample per tick: current + earlier frame for each camera, proprio, action chunk."""

    def __init__(self, files, cams, chunk, history_offset, stats, text_ids: dict):
        self.files, self.cams, self.chunk, self.hist = [str(f) for f in files], cams, chunk, history_offset
        self.stats = {k: np.asarray(v, np.float32) for k, v in stats.items()}
        self.index, self.text_id, self.arr = [], [], []
        for i, f in enumerate(self.files):
            with h5py.File(f, "r") as h:        # low-dimensional signals are small: keep them in RAM
                aux = np.concatenate([h["state/bowl"][:, :2], h["state/pad"][:, :2]], axis=1)
                self.arr.append((h["action"][:], h["proprio"][:], ((aux - AUX_CENTER) / AUX_SCALE).astype(np.float32)))
                self.text_id.append(text_ids[h.attrs["instruction"]])
            self.index.extend((i, t) for t in range(len(self.arr[-1][0])))
        self._h5 = {}                           # per-worker open file handles (opened lazily)

    def __len__(self):
        return len(self.index)

    def _file(self, i):
        if i not in self._h5:
            if len(self._h5) >= 256:
                self._h5.pop(next(iter(self._h5))).close()
            self._h5[i] = h5py.File(self.files[i], "r")
        return self._h5[i]

    def __getitem__(self, k):
        i, t = self.index[k]
        h, s = self._file(i), self.stats
        action, proprio, aux = self.arr[i]
        ep = {"action": action, "proprio": proprio}
        tp = max(t - self.hist, 0)
        views = [decode(h[f"images/{c}"][t]) for c in self.cams] + [decode(h[f"images/{c}"][tp]) for c in self.cams]
        images = torch.from_numpy(np.stack(views)).permute(0, 3, 1, 2)           # uint8 [V, 3, H, W]
        a = action_targets(ep["action"], ep["proprio"], t, self.chunk)
        return {"images": images,
                "proprio": torch.from_numpy((ep["proprio"][t] - s["proprio_mean"]) / s["proprio_std"]),
                "action": torch.from_numpy((a - s["action_mean"]) / s["action_std"]),
                "aux": torch.from_numpy(aux[t]), "text_id": self.text_id[i]}
