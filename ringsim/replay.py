"""Replay a recorded exam episode from its applied commands in a fresh simulator, check that the
outcome reproduces, and optionally render a video.

    python -m ringsim.replay outputs/ringsim/astra_realtime.jsonl --seed 50000 --video out.mp4
"""
from __future__ import annotations

import argparse
import json
import os


def replay(rec, video=None):
    os.environ.setdefault("MUJOCO_GL", "egl")
    from .game import Game
    game = Game(render=False)
    game.reset(rec["seed"])
    actions = {int(k): a for k, a in rec.get("applied", [])}
    recorder = None
    if video:
        from .video import Recorder
        recorder = Recorder(game, video)
    while not game.done:
        game.step(actions.get(game.tick), observe=False)
        if recorder:
            recorder.frame()
    if recorder:
        recorder.close()
    return game.result()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--video")
    args = ap.parse_args()
    recs = [json.loads(l) for l in open(args.jsonl)]
    recs = [r for r in recs if "fatal" not in r and (args.seed is None or r["seed"] == args.seed)]
    bad = 0
    for r in recs:
        res = replay(r, args.video if args.seed is not None else None)
        same = [e["outcome"] for e in res["events"]] == [e["outcome"] for e in r["events"]]
        bad += not same
        print(r["seed"], r["caught"], "->", res["caught"], "reproduced" if same else "MISMATCH")
    print(f"{len(recs) - bad}/{len(recs)} episodes reproduced")


if __name__ == "__main__":
    main()
