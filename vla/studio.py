"""Live stress-test studio: drive the VLA from a browser while changing how the pads move.

  .venv/bin/python -m vla.studio                       # then open http://127.0.0.1:8770
  .venv/bin/python -m vla.studio --host 0.0.0.0        # reachable from another machine

The page shows a live view of the simulation and the two images the model sees, and lets you set
the instruction, pad speed and pad travel while the policy runs.

How it behaves:
  Run           starts an episode with your instruction (and resumes if paused). If the scene on
                screen has already been used, a new scene is loaded first.
  New scene     loads a new layout. It only starts running if Auto-run is on.
  Auto-run      when on, every finished episode is followed by a new scene and the same instruction.
  Time limit    an episode that has not succeeded by then is recorded as failed; "none" disables it.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DEFAULT_CKPT = "outputs/vla/runs/m2_early/best.pt"
PAGE = Path(__file__).with_name("studio.html")


class Stream:
    """Latest JPEG frame, shared with any number of MJPEG clients."""

    def __init__(self):
        self.cond, self.jpeg = threading.Condition(), None

    def push(self, rgb, quality=82):
        from PIL import Image

        b = io.BytesIO()
        Image.fromarray(rgb).save(b, format="JPEG", quality=quality)
        with self.cond:
            self.jpeg = b.getvalue()
            self.cond.notify_all()

    def serve(self, handler):
        handler.send_response(200)
        handler.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        handler.send_header("Cache-Control", "no-store")
        handler.end_headers()
        try:
            while True:
                with self.cond:
                    self.cond.wait(timeout=1.0)
                    buf = self.jpeg
                if buf is not None:
                    handler.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                        + str(len(buf)).encode() + b"\r\n\r\n" + buf + b"\r\n")
        except (BrokenPipeError, ConnectionResetError, OSError):
            return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sleep", action="store_true", help="pace with sleep instead of busy-waiting (slower inference)")
    args = ap.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")

    import mujoco
    import numpy as np
    import torch

    from vla.common import load_vla_config
    from vla.env import TEMPLATES, TickEnv, TickStats
    from vla.play import parse_target
    from vla.policy import load_policy

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = load_vla_config()
    cfg["vla"] = v = ck["vla_cfg"]
    train_speed = [float(x) for x in v["pads"][0]["speed"]]
    train_travel = float(v["pads"][0]["travel"])

    class StudioEnv(TickEnv):
        """TickEnv whose pad speed and travel can change while it runs."""

        def __init__(self, cfg):
            super().__init__(cfg, render=True)
            n = len(self.pad_names)
            self.speed, self.travel = float(np.mean(train_speed)), train_travel
            self._s, self._dir = [0.0] * n, [1] * n
            self._rail = [self.m.geom(f"pad_{p}_rail").id for p in self.pad_names]
            self._jnt = [self.m.joint(f"pad_{p}_slide").id for p in self.pad_names]
            self._half = [max(p["size"]) / 2 for p in self.v["pads"]]
            self.set_travel(self.travel)

        def set_travel(self, travel):
            self.travel = float(travel)
            for i in range(len(self.pad_names)):
                self.m.jnt_range[self._jnt[i]] = [-self.travel - 0.01, self.travel + 0.01]
                self.m.geom_size[self._rail[i], 0] = self.travel + self._half[i]

        def studio_task(self, seed):
            t = self.sample_task(seed)
            for p in t["pads"]:
                p["s0"] = p["s0"] / train_travel * self.travel
                p["speed"] = self.speed
            return t

        def reset(self, seed, task=None, randomize_visuals=True):
            task = task or self.studio_task(seed)
            self._s = [p["s0"] for p in task["pads"]]
            self._dir = [p["dir"] for p in task["pads"]]
            return super().reset(seed, task, randomize_visuals)

        def pad_ref(self, i, t):          # called once per physics step per pad
            dt, A, sp = self.m.opt.timestep, self.travel, self.speed
            s = self._s[i] + self._dir[i] * sp * dt
            if s > A:
                self._dir[i], s = -1, max(A, s - 2 * sp * dt)
            elif s < -A:
                self._dir[i], s = 1, min(-A, s + 2 * sp * dt)
            self._s[i] = s
            return s

    env = StudioEnv(cfg)
    pol = load_policy(args.ckpt)
    bowls, pads = env.bowl_names, env.pad_names

    # big viewer camera (separate from the 224 px model cameras)
    VW, VH = 960, 600
    env.m.vis.global_.offwidth, env.m.vis.global_.offheight = max(VW, env.m.vis.global_.offwidth), max(VH, env.m.vis.global_.offheight)
    view = mujoco.Renderer(env.m, VH, VW)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    CAMS = {"overview": (155.0, -34.0, 0.95), "top": (180.0, -89.0, 1.0), "side": (90.0, -20.0, 0.95), "front": (180.0, -24.0, 0.95)}
    cam.lookat[:] = [0.12, 0.0, 0.03]
    cam.azimuth, cam.elevation, cam.distance = CAMS["overview"]

    main_stream, input_stream = Stream(), Stream()
    cmds: queue.Queue = queue.Queue()
    lock = threading.Lock()
    snapshot = {}

    # ------------------------------------------------------------------ state
    # phase: "ready"   scene loaded, no episode yet (arm holds, pads move)
    #        "running" an episode is in progress
    #        "done"    the episode ended; result says how
    # `spec` is the user's instruction and stays until they change it. "random" fields are drawn
    # again for every episode. Pause only freezes time; commands still apply.
    S = {"phase": "ready", "result": None, "failure": "", "auto": False, "paused": False, "scored": True,
         "episode": 0, "history": [], "hz": 0.0, "seed": args.seed, "dirty": False, "limit_s": 20.0,
         "spec": {"bowl": "random", "pad": "random", "text": ""}, "final_t": 0.0, "t_done": 0.0,
         "max_speed": 0.0, "max_travel": 0.0, "switched": False}

    def resolve(spec):
        """Instruction text for one episode from the sticky spec."""
        if spec.get("text"):
            return spec["text"]
        b = spec["bowl"] if spec["bowl"] in bowls else bowls[int(np.random.randint(len(bowls)))]
        p = spec["pad"] if spec["pad"] in pads else pads[int(np.random.randint(len(pads)))]
        return TEMPLATES[0].format(bowl=b, pad=p)

    def clean_spec(c):
        sp = c.get("spec") or {}
        return {"bowl": sp.get("bowl") if sp.get("bowl") in bowls else "random",
                "pad": sp.get("pad") if sp.get("pad") in pads else "random",
                "text": str(sp.get("text") or "").strip()[:160]}

    def set_goal(text):
        env.instruction = text
        tgt = parse_target(text, bowls, pads)
        if tgt:
            env.target_bowl, env.target_pad = tgt
        S["scored"] = tgt is not None
        env._hold = 0
        pol.set_instruction(text)
        pol.queue = []                              # replan on the next tick

    def log(result, failure=""):
        S["history"].append({"episode": S["episode"], "instruction": env.instruction, "speed": round(S["max_speed"] * 100, 2),
                             "travel": round(S["max_travel"] * 100, 2), "result": result, "failure": failure,
                             "seconds": round(env.tick * env.dt, 1), "switched": S["switched"]})
        S["history"] = S["history"][-400:]

    def end_episode(result, failure=""):
        S.update(phase="done", result=result, failure=failure, t_done=time.time(), final_t=env.tick * env.dt)
        log(result, failure)

    def load_scene(seed=None):
        """Put a fresh layout on screen. Does not start an episode."""
        nonlocal obs
        if S["phase"] == "running":
            end_episode("stopped")
        S["seed"] = S["seed"] + 1 if seed is None else seed
        obs = env.reset(v["seeds"]["eval_start"] + 40000 + S["seed"])
        pol.reset(S["seed"])
        env.instruction = ""
        S.update(phase="ready", result=None, failure="", dirty=False, final_t=0.0)

    def start_episode(same_scene=False):
        """Run the user's instruction: in the scene on screen if it is untouched, otherwise in a new one."""
        if S["phase"] == "running":
            end_episode("stopped")
        if S["dirty"]:
            load_scene(seed=S["seed"] if same_scene else None)
        env.v["max_ticks"] = int(round(S["limit_s"] * v["control_hz"])) if S["limit_s"] else 10 ** 9
        env.tick, env._hold, env.stats = 0, 0, TickStats(seed=env.seed)
        pol.reset(S["seed"])
        set_goal(resolve(S["spec"]))
        S["episode"] += 1
        S.update(phase="running", result=None, failure="", dirty=True, switched=False,
                 max_speed=env.speed, max_travel=env.travel)

    def handle(c):
        k = c.get("type")
        if k == "run":                              # the Run button: always starts an episode, and resumes
            S["spec"] = clean_spec(c)
            S["paused"] = False
            start_episode()
        elif k == "switch":                         # change the goal of the episode that is running
            if S["phase"] == "running":
                S["spec"] = clean_spec(c)
                set_goal(resolve(S["spec"]))
                S["switched"] = True
        elif k == "retry":
            S["paused"] = False
            start_episode(same_scene=True)
        elif k == "new_scene":
            load_scene()
            if S["auto"]:
                start_episode()
        elif k == "auto":
            S["auto"] = bool(c["value"])
        elif k == "pause":
            S["paused"] = bool(c["value"])
        elif k == "limit":
            S["limit_s"] = float(np.clip(float(c["value"]), 0.0, 600.0))
            env.v["max_ticks"] = int(round(S["limit_s"] * v["control_hz"])) if S["limit_s"] else 10 ** 9
        elif k == "speed":
            env.speed = float(np.clip(float(c["value"]), 0.0, 0.30))
            S["max_speed"] = max(S["max_speed"], env.speed)
        elif k == "travel":
            env.set_travel(float(np.clip(float(c["value"]), 0.01, 0.10)))
            S["max_travel"] = max(S["max_travel"], env.travel)
        elif k == "clear":
            S["history"] = []
        elif k == "camera":
            if c.get("preset") in CAMS:
                cam.azimuth, cam.elevation, cam.distance = CAMS[c["preset"]]
            cam.azimuth = float(cam.azimuth + float(c.get("daz", 0.0)))
            cam.elevation = float(np.clip(cam.elevation + float(c.get("del", 0.0)), -89.0, -5.0))
            cam.distance = float(np.clip(cam.distance * float(c.get("zoom", 1.0)), 0.35, 2.0))

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body: bytes, ctype: str):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/view.mjpg":
                main_stream.serve(self)
            elif path == "/inputs.mjpg":
                input_stream.serve(self)
            elif path in ("/view.jpg", "/inputs.jpg"):
                self._send((main_stream if path == "/view.jpg" else input_stream).jpeg or b"", "image/jpeg")
            elif path == "/state":
                with lock:
                    body = json.dumps(snapshot).encode()
                self._send(body, "application/json")
            elif path in ("/", "/index.html"):
                self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
            else:
                self.send_error(404)

        def do_POST(self):
            try:
                n = int(self.headers.get("Content-Length", 0))
                cmds.put(json.loads(self.rfile.read(min(n, 4096)) or b"{}"))
                self._send(b"{}", "application/json")
            except (ValueError, json.JSONDecodeError):
                self.send_error(400)

    server = ThreadingHTTPServer((args.host, args.port), H)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"VLA studio: http://{'127.0.0.1' if args.host in ('0.0.0.0', '') else args.host}:{args.port}/   (Ctrl-C to stop)", flush=True)

    obs = None
    load_scene(seed=args.seed)
    obs["instruction"] = "warm up"
    pol.act(obs)                                    # first CUDA call is slow; do it before anyone watches
    pol.reset(args.seed)
    pol.latencies_ms.clear()
    dts = []
    try:
        while True:
            t0 = time.perf_counter()
            while not cmds.empty():
                try:
                    handle(cmds.get())
                except (KeyError, ValueError, TypeError) as e:
                    print("bad command:", e, flush=True)
            if not S["paused"]:
                if S["phase"] == "running":
                    obs["instruction"] = env.instruction
                    obs, done, info = env.step(pol.act(obs))
                    if done:
                        ok = bool(info.get("success"))
                        if not S["scored"]:
                            end_episode("unscored")
                        else:
                            end_episode("success" if ok else "failed", "" if ok else info.get("failure_type", ""))
                else:                               # ready / done: hold the arm, keep the pads moving
                    obs, _, _ = env.step(env.ctrl)
                    if S["auto"] and (S["phase"] == "ready" or time.time() - S["t_done"] > 1.2):
                        start_episode()
            view.update_scene(env.d, camera=cam)
            main_stream.push(view.render())
            input_stream.push(np.concatenate([obs["images"][c] for c in pol.cams], axis=1), quality=88)

            lat = np.array(pol.latencies_ms[-200:] or [0.0])
            bi, pi = env.target_bowl, env.target_pad
            _, resting = env.on_pad(bi, pi)
            running = S["phase"] == "running"
            snap = {"phase": S["phase"], "result": S["result"], "failure": S["failure"], "auto": S["auto"],
                    "paused": S["paused"], "scored": S["scored"], "episode": S["episode"], "scene": S["seed"],
                    "spec": S["spec"], "instruction": env.instruction, "bowls": bowls, "pads": pads,
                    "target_bowl": bowls[bi], "target_pad": pads[pi],
                    "t": round(env.tick * env.dt if running else S["final_t"], 2), "limit_s": S["limit_s"],
                    "hold": round(env._hold * env.dt, 2) if running else 0.0, "hold_needed": v["success"]["hold_time"],
                    "grasped": bool(env.grasped(bi)) if S["phase"] != "ready" else False,
                    "on_pad": bool(resting) if S["phase"] != "ready" else False,
                    "speed": env.speed, "travel": env.travel, "train_speed": train_speed, "train_travel": train_travel,
                    "latency_p50": float(np.percentile(lat, 50)), "latency_max": float(lat.max()), "loop_hz": S["hz"],
                    "control_hz": v["control_hz"], "exec_horizon": pol.exec_horizon, "history": S["history"]}
            with lock:
                snapshot.clear()
                snapshot.update(snap)
            # Busy-wait to the next tick. Sleeping lets the core idle and the next inference then
            # takes ~23 ms instead of ~10 ms (see vla/latency_paced.py); --sleep trades that for an idle core.
            if args.sleep:
                time.sleep(max(0.0, env.dt - (time.perf_counter() - t0)))
            else:
                while time.perf_counter() - t0 < env.dt:
                    pass
            dts.append(time.perf_counter() - t0)
            if len(dts) >= 25:
                S["hz"] = round(len(dts) / sum(dts), 1)
                dts.clear()
    except KeyboardInterrupt:
        print("\nstopped")
        server.shutdown()


if __name__ == "__main__":
    main()
