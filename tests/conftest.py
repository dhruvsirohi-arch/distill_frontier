"""Runs before any test module imports mujoco: pick the GL backend (EGL on headless Linux)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from viz.viz import setup_gl  # noqa: E402

setup_gl("record")
