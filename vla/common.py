"""Config for the VLA stages: base config.yaml plus configs/vla.yaml under cfg["vla"]."""
from __future__ import annotations

import yaml

from env.common import REPO, load_config


def load_vla_config(path: str | None = None) -> dict:
    cfg = load_config()
    with open(path or REPO / "configs" / "vla.yaml") as f:
        cfg["vla"] = yaml.safe_load(f)
    return cfg
