"""How fast can GPT-6.2-Astra answer one control decision? Profiles decision latency for input
payloads from the full sensor turn down to text only, with fresh vs reused threads.

    python -m ringsim.latency_profile --n 20 --out outputs/ringsim/latency_profile.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")


def main():
    from PIL import Image

    from .astra import Astra, depth_png, png_part
    from .exam import SENSOR_SCHEMA, _sensor_turn_inputs, sensor_instructions
    from .game import Game

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    g = Game(render=True)
    obs = g.reset(50000)
    small = np.asarray(Image.fromarray(obs["rgb"]).resize((320, 120)))
    tiny_schema = {"type": "object", "additionalProperties": False, "required": ["slot"],
                   "properties": {"slot": {"type": "integer"}}}
    full_instr = sensor_instructions("realtime")
    configs = {
        "full sensor turn (3 images, fresh thread)":
            (full_instr, lambda: _sensor_turn_inputs(obs, None, g.ctrl.cmd, ""), SENSOR_SCHEMA, False),
        "1 RGB image 640x240 (fresh thread)":
            (full_instr, lambda: [{"type": "text", "text": f"t={obs['t']}"}, png_part(obs["rgb"])],
             SENSOR_SCHEMA, False),
        "1 RGB image 320x120 (fresh thread)":
            (full_instr, lambda: [{"type": "text", "text": f"t={obs['t']}"}, png_part(small)],
             SENSOR_SCHEMA, False),
        "1 RGB image 320x120 (reused thread)":
            (full_instr, lambda: [{"type": "text", "text": f"t={obs['t']}"}, png_part(small)],
             SENSOR_SCHEMA, True),
        "text only, one-integer answer (fresh thread)":
            ("Answer with JSON.", lambda: [{"type": "text", "text": "Which slot 0-7? Say 3."}],
             tiny_schema, False),
    }
    out = {}
    with Astra(effort="low") as astra:
        for name, (instr, make, schema, reuse) in configs.items():
            lat, toks, th = [], [], None
            for _ in range(args.n):
                t0 = time.monotonic()
                if th is None or not reuse:
                    th = astra.new_thread(instr)
                _, info = astra.ask(th, make(), schema)
                lat.append(time.monotonic() - t0)
                toks.append((info["usage"] or {}).get("inputTokens"))
            out[name] = {"n": len(lat), "median_s": round(float(np.median(lat)), 2),
                         "p10_s": round(float(np.percentile(lat, 10)), 2),
                         "p90_s": round(float(np.percentile(lat, 90)), 2),
                         "min_s": round(float(np.min(lat)), 2),
                         "input_tokens_median": float(np.median([t for t in toks if t])),
                         "samples_s": [round(x, 2) for x in lat]}
            print(name, {k: v for k, v in out[name].items() if k != "samples_s"}, flush=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
