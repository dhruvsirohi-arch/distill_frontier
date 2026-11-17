"""Config for the VLA stages: base config.yaml plus configs/vla.yaml under cfg["vla"].
An optional override file (e.g. configs/vla_m2.yaml) is deep-merged on top; lists are replaced."""
from __future__ import annotations

import yaml

from env.common import REPO, load_config


def _merge(a: dict, b: dict) -> dict:
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _merge(a[k], v)
        else:
            a[k] = v
    return a


def load_vla_config(override: str | None = None) -> dict:
    cfg = load_config()
    with open(REPO / "configs" / "vla.yaml") as f:
        cfg["vla"] = yaml.safe_load(f)
    if override:
        with open(override) as f:
            _merge(cfg["vla"], yaml.safe_load(f))
    return cfg
