"""Visualization for every stage that runs MuJoCo.

Modes (``--viz`` flag / ``viz.mode`` in configs):
  live    on-screen passive viewer (macOS: run under ``mjpython``), real-time with speed multiplier,
          overlay text, camera switching (keys 1=front, 2=top, 0=free camera).
  stream  offscreen render -> MJPEG over HTTP (http://127.0.0.1:<port>/) with an action/event log.
  record  offscreen render (front | top side by side + overlay) -> MP4 under outputs/videos/<stage>/.
  none    no rendering.

All modes only *read* MjModel/MjData, so the same seed gives the same outcome with viz on or off
(tests/test_viz.py checks this).
"""
from __future__ import annotations

import io
import json
import os
import platform
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np


def setup_gl(mode: str) -> str:
    """Pick the GL backend BEFORE mujoco is imported. Returns the effective viz mode."""
    if mode == "live" and not display_available():
        print("[viz] no display detected -> falling back from 'live' to 'stream'", flush=True)
        mode = "stream"
    # Offscreen rendering on Linux always goes through EGL, even when a desktop session exists:
    # via X/GLFW every process queues on the X server, which serializes parallel workers.
    if platform.system() == "Linux" and mode in ("stream", "record"):
        os.environ.setdefault("MUJOCO_GL", "egl")
    return mode


def display_available() -> bool:
    if os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY"):
        return bool(os.environ.get("DISPLAY"))  # X forwarding only
    if platform.system() == "Darwin":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def overlay_lines(info: dict, n_success: int, n_done: int) -> list[str]:
    a = info.get("action")
    return [
        f"policy: {info.get('policy', '')}   episode: {info.get('episode_id', '')}   seed: {info.get('seed', '')}",
        f"step: {info.get('step', 0)}   success so far: {n_success}/{n_done}",
        f"action: {json.dumps(a) if a is not None else ('<invalid>' if 'action' in info and info.get('step', 0) else '-')}",
    ]


class BaseViz:
    def __init__(self, cfg: dict):
        self.cfg = cfg["viz"]
        self.n_success = 0
        self.n_done = 0
        self.last_info: dict = {}
        self._wall0 = None
        self._sim0 = 0.0

    # hooks called by BowlEnv
    def on_episode_start(self, m, d, info):
        self.last_info = dict(info)
        self._wall0, self._sim0 = time.time(), d.time

    def on_action(self, action, info):
        self.last_info = dict(info)

    def on_frame(self, m, d, info):
        self.last_info = dict(info)

    def on_episode_end(self, m, d, info, stats):
        self.n_done += 1
        self.n_success += int(stats.success)

    def close(self):
        pass

    def _pace(self, d):
        speed = float(self.cfg.get("speed", 1.0))
        if speed <= 0 or self._wall0 is None:
            return
        target = self._wall0 + (d.time - self._sim0) / speed
        dt = target - time.time()
        if dt > 0:
            time.sleep(dt)


class _Renderer:
    """Offscreen renderer producing a (front | top) frame with text overlay."""

    def __init__(self, m, width, height, cameras):
        import mujoco
        from PIL import ImageFont

        self.r = mujoco.Renderer(m, height=height, width=width)
        self.cameras = cameras
        self.font = ImageFont.load_default(size=max(10, height // 26))

    def frame(self, d, lines, banner=None):
        from PIL import Image, ImageDraw

        imgs = []
        for cam in self.cameras:
            self.r.update_scene(d, camera=cam)
            imgs.append(self.r.render())
        img = Image.fromarray(np.concatenate(imgs, axis=1))
        dr = ImageDraw.Draw(img)
        y = 4
        for ln in lines:
            dr.text((6, y), ln, fill=(255, 255, 255), font=self.font, stroke_width=2, stroke_fill=(0, 0, 0))
            y += self.font.size + 3
        if banner:
            text, color = banner
            dr.text((6, img.height - self.font.size - 8), text, fill=color, font=self.font,
                    stroke_width=2, stroke_fill=(0, 0, 0))
        return img

    def close(self):
        self.r.close()


class RecordViz(BaseViz):
    def __init__(self, cfg, stage: str, lowres: bool = False, out_root="outputs/videos"):
        super().__init__(cfg)
        w, h = (self.cfg["lowres"] if lowres else (self.cfg["width"], self.cfg["height"]))
        self.size = (int(w), int(h))
        self.dir = Path(out_root) / stage
        self.dir.mkdir(parents=True, exist_ok=True)
        self.rend = None
        self.frames = []
        self.enabled = True  # can be toggled per episode (record every Nth)
        self.last_path = None

    def on_episode_start(self, m, d, info):
        super().on_episode_start(m, d, info)
        self.frames = []
        if self.enabled and self.rend is None:
            self.rend = _Renderer(m, *self.size, self.cfg["cameras"])
        if self.enabled:
            self.frames.append(np.asarray(self.rend.frame(d, overlay_lines(info, self.n_success, self.n_done))))

    def on_frame(self, m, d, info):
        super().on_frame(m, d, info)
        if self.enabled:
            self.frames.append(np.asarray(self.rend.frame(d, overlay_lines(info, self.n_success, self.n_done))))

    def on_episode_end(self, m, d, info, stats):
        super().on_episode_end(m, d, info, stats)
        if not self.enabled:
            return
        import imageio.v2 as iio2

        ok = stats.success
        banner = ("SUCCESS" if ok else f"FAIL: {stats.failure_type}", (80, 255, 80) if ok else (255, 80, 80))
        last = np.asarray(self.rend.frame(d, overlay_lines(info, self.n_success, self.n_done), banner))
        self.frames.extend([last] * int(self.cfg["fps"]))  # hold result for 1 s
        path = self.dir / f"{info['episode_id']}.mp4"
        iio2.mimwrite(path, self.frames, fps=self.cfg["fps"], codec="libx264", macro_block_size=1,
                      quality=6)
        self.last_path = str(path)
        self.frames = []

    def close(self):
        if self.rend is not None:
            self.rend.close()


class StreamViz(BaseViz):
    PAGE = """<!doctype html><html><head><meta charset=utf-8><title>SO-101 stream</title>
<style>body{font-family:system-ui;background:#111;color:#eee;margin:16px}img{max-width:100%;border:1px solid #444}
#log{font:12px monospace;white-space:pre-wrap;height:220px;overflow:auto;background:#000;padding:8px}</style></head>
<body><h3>SO-101 bowl pick-and-place — live stream</h3><img src="/stream"><h4>events</h4><div id=log></div>
<script>let n=0;setInterval(async()=>{const r=await fetch('/events?since='+n);const j=await r.json();
for(const e of j.events){document.getElementById('log').textContent+=e+'\\n';}n=j.next;
const l=document.getElementById('log');l.scrollTop=l.scrollHeight;},500);</script></body></html>"""

    def __init__(self, cfg, port=None):
        super().__init__(cfg)
        self.port = int(port or self.cfg["stream_port"])
        self.rend = None
        self.jpeg = None
        self.cond = threading.Condition()
        self.events: list[str] = []
        viz = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path.startswith("/stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    try:
                        while True:
                            with viz.cond:
                                viz.cond.wait(timeout=1.0)
                                buf = viz.jpeg
                            if buf is None:
                                continue
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf + b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        return
                elif self.path.startswith("/events"):
                    since = int(self.path.split("since=")[-1]) if "since=" in self.path else 0
                    body = json.dumps({"events": viz.events[since:], "next": len(viz.events)}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    body = viz.PAGE.encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", self.port), H)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        print(f"[viz] MJPEG stream at http://127.0.0.1:{self.port}/", flush=True)

    def _push(self, m, d, info, banner=None):
        if self.rend is None:
            self.rend = _Renderer(m, self.cfg["width"], self.cfg["height"], self.cfg["cameras"])
        img = self.rend.frame(d, overlay_lines(info, self.n_success, self.n_done), banner)
        b = io.BytesIO()
        img.save(b, format="JPEG", quality=80)
        with self.cond:
            self.jpeg = b.getvalue()
            self.cond.notify_all()

    def on_episode_start(self, m, d, info):
        super().on_episode_start(m, d, info)
        self.events.append(f"[{info['policy']}] episode {info['episode_id']} (seed {info['seed']}) start")
        self._push(m, d, info)

    def on_action(self, action, info):
        super().on_action(action, info)
        self.events.append(f"  step {info['step']}: {json.dumps(action) if action is not None else '<invalid>'}")

    def on_frame(self, m, d, info):
        super().on_frame(m, d, info)
        self._push(m, d, info)
        self._pace(d)

    def on_episode_end(self, m, d, info, stats):
        super().on_episode_end(m, d, info, stats)
        msg = "SUCCESS" if stats.success else f"FAIL: {stats.failure_type}"
        self.events.append(f"  -> {msg}   (running {self.n_success}/{self.n_done})")
        self._push(m, d, info, (msg, (80, 255, 80) if stats.success else (255, 80, 80)))
        time.sleep(0.5 / max(float(self.cfg.get("speed", 1.0)), 1e-3))

    def close(self):
        self.server.shutdown()
        if self.rend is not None:
            self.rend.close()


class LiveViz(BaseViz):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.handle = None
        self._cam_req = None

    def _key(self, keycode):
        if keycode in (ord("0"), ord("1"), ord("2")):
            self._cam_req = keycode - ord("0")

    def _ensure(self, m, d):
        import mujoco
        import mujoco.viewer

        if self.handle is None:
            self.handle = mujoco.viewer.launch_passive(m, d, key_callback=self._key)
            self._set_cam(1)
            print("[viz] live viewer: keys 1=front camera, 2=top camera, 0=free camera", flush=True)
        if self._cam_req is not None:
            self._set_cam(self._cam_req)
            self._cam_req = None
        return mujoco

    def _set_cam(self, idx):
        import mujoco

        with self.handle.lock():
            if idx == 0:
                self.handle.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            else:
                self.handle.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                self.handle.cam.fixedcamid = self.handle.m.camera(self.cfg["cameras"][idx - 1]).id

    def _sync(self, m, d, info, extra=None):
        mujoco = self._ensure(m, d)
        if not self.handle.is_running():
            return
        lines = overlay_lines(info, self.n_success, self.n_done)
        if extra:
            lines.append(extra)
        self.handle.set_texts((mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                               "\n".join(ln.split(":")[0] for ln in lines),
                               "\n".join(":".join(ln.split(":")[1:]).strip() for ln in lines)))
        self.handle.sync()

    def on_episode_start(self, m, d, info):
        super().on_episode_start(m, d, info)
        self._sync(m, d, info)

    def on_frame(self, m, d, info):
        super().on_frame(m, d, info)
        self._sync(m, d, info)
        self._pace(d)

    def on_episode_end(self, m, d, info, stats):
        super().on_episode_end(m, d, info, stats)
        self._sync(m, d, info, "result: " + ("SUCCESS" if stats.success else f"FAIL {stats.failure_type}"))
        time.sleep(0.8 / max(float(self.cfg.get("speed", 1.0)), 1e-3))

    def close(self):
        if self.handle is not None:
            self.handle.close()


def make_viz(mode: str, cfg: dict, stage: str = "misc", lowres: bool = False):
    """Factory. Call viz.setup_gl(mode) before importing mujoco in entry points."""
    mode = (mode or "none").lower()
    if mode == "none":
        return None
    if mode == "live":
        if platform.system() == "Darwin":
            import mujoco.viewer as mv

            if mv._MJPYTHON is None:
                print("[viz] 'live' on macOS needs mjpython (e.g. .venv/bin/mjpython -m ...). "
                      "Falling back to 'stream'.", flush=True)
                return StreamViz(cfg)
        return LiveViz(cfg)
    if mode == "stream":
        return StreamViz(cfg)
    if mode == "record":
        return RecordViz(cfg, stage, lowres=lowres)
    raise ValueError(f"unknown viz mode {mode}")


def add_viz_args(ap):
    ap.add_argument("--viz", default=None, choices=["none", "live", "stream", "record"],
                    help="visualization mode (default: configs viz.mode)")
    ap.add_argument("--speed", type=float, default=None, help="real-time speed multiplier (live/stream)")
    return ap


def resolve_mode(args, cfg) -> str:
    mode = args.viz or cfg["viz"]["mode"]
    if args.speed is not None:
        cfg["viz"]["speed"] = args.speed
    return setup_gl(mode)
