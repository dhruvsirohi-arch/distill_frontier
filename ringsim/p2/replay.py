"""Replay a Phase 2 episode from its applied commands, check the outcome reproduces, and
optionally render (a) a spectator video and (b) the exact images Astra was shown at each call.

    python -m ringsim.p2.replay outputs/ringsim2/astra_standard.jsonl --seed 50000 \
        --video out.mp4 --frames out_frames/
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def replay(rec, video=None, frames_dir=None):
    os.environ.setdefault("MUJOCO_GL", "egl")
    import numpy as np
    from PIL import Image

    from .sensors import lidar_views
    from .world import FPS, World
    w = World(rec["seed"], render=True)
    actions = {int(k): a for k, a in rec.get("applied", [])}
    want = {d["frame"] for d in rec.get("decisions", []) if "frame" in d}
    recorder = None
    if video:
        from ..video import Recorder
        recorder = Recorder(w.game, video, fps=FPS)
    if frames_dir:
        Path(frames_dir).mkdir(parents=True, exist_ok=True)

    def dump():
        if frames_dir and w.frame_no in want:
            s = w.snapshot()
            rgb = s["rgb"]
            lid = lidar_views(s["scan"])
            lid = np.asarray(Image.fromarray(lid).resize((rgb.shape[0] * lid.shape[1] // lid.shape[0],
                                                          rgb.shape[0])))
            Image.fromarray(np.concatenate([rgb, lid], 1)).save(
                Path(frames_dir) / f"{w.frame_no:04d}.jpg", quality=80)
    dump()
    while not w.done:
        w.step(actions.get(w.game.tick))
        if recorder:
            recorder.frame()
        dump()
    if recorder:
        recorder.close()
    return w.game.result()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--video")
    ap.add_argument("--frames")
    args = ap.parse_args()
    recs = [json.loads(l) for l in open(args.jsonl)]
    recs = [r for r in recs if "fatal" not in r and (args.seed is None or r["seed"] == args.seed)]
    bad = 0
    for r in recs:
        one = args.seed is not None
        res = replay(r, args.video if one else None, args.frames if one else None)
        same = [e["outcome"] for e in res["events"]] == [e["outcome"] for e in r["events"]]
        bad += not same
        print(r["seed"], r["caught"], "->", res["caught"], "reproduced" if same else "MISMATCH")
    print(f"{len(recs) - bad}/{len(recs)} episodes reproduced")


if __name__ == "__main__":
    main()
