"""Config loading, seeding and logging shared by every stage."""
from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]


def load_config(path: str | None = None) -> dict:
    path = Path(path) if path else REPO / "configs" / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def get_logger(stage: str, cfg: dict) -> logging.Logger:
    log_dir = REPO / cfg["paths"]["logs"]
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(stage)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(name)s] %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(log_dir / f"{stage}_{time.strftime('%Y%m%d_%H%M%S')}.log")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
