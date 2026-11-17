"""Render / transcode the footage for the showcase site (docs/).

  .venv/bin/python -m showcase.render            # everything
  .venv/bin/python -m showcase.render --only demos hero

* demos       clean cinematic renders of scripted-expert training episodes (curated train seeds)
* hero        the final Qwen policy on held-out eval seeds, concatenated
* progress    the real training-checkpoint recordings (step 0 -> 1498), front camera cropped
* evalwall    all 100 held-out Qwen eval recordings, re-encoded for the web + thumbnails

Rendering only *reads* MjData. The "studio look" edits (dark floor, fog, hidden spawn marker) touch only
visual model fields (textures, material/geom colours, vis.*), never anything physics reads, so outcomes are
identical to the training/eval runs (same seeds, same policies).
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from viz.viz import setup_gl

setup_gl("record")

import imageio.v2 as iio  # noqa: E402
import imageio_ffmpeg  # noqa: E402
import mujoco  # noqa: E402
from PIL import Image  # noqa: E402

from env.common import REPO, get_logger, load_config  # noqa: E402
from env.rollout import run_episode  # noqa: E402
from env.so101_env import BowlEnv  # noqa: E402
from viz.viz import BaseViz  # noqa: E402

OUT = REPO / "docs" / "assets"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
BG = np.array([13, 14, 17], np.uint8)


def stylize(m):
    """Visual-only restyle: dark studio floor with a fine grid, flat dark sky, fog, hidden spawn marker."""
    for name, fn in (("grid", _floor_tex), ("sky", _sky_tex)):
        t = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TEXTURE, name)
        if t < 0:
            continue
        w, h, c = m.tex_width[t], m.tex_height[t], m.tex_nchannel[t]
        a = m.tex_adr[t]
        m.tex_data[a:a + w * h * c] = fn(w, h, c).reshape(-1)
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "grid")
    m.mat_reflectance[g] = 0.06
    m.mat_texrepeat[g] = [24, 24]
    for geom, rgba in (("spawn_marker", [0, 0, 0, 0]), ("target_region", [0.25, 0.95, 0.55, 0.45]),
                       ("pedestal", [0.30, 0.32, 0.37, 1])):
        i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, geom)
        if i >= 0:
            m.geom_rgba[i] = rgba
    m.vis.quality.shadowsize = 8192
    m.vis.quality.offsamples = 8
    m.vis.map.fogstart, m.vis.map.fogend = 1.05, 2.1
    m.vis.rgba.fog[:] = [*(BG / 255.0), 1]
    m.vis.headlight.ambient[:] = [0.42, 0.42, 0.45]
    m.vis.headlight.diffuse[:] = [0.62, 0.62, 0.62]


def _floor_tex(w, h, c):
    img = np.empty((h, w, c), np.uint8)
    img[:] = [22, 24, 29][:c]
    step = max(4, w // 8)
    for k in range(0, w, step):
        img[:, k:k + 2] = [38, 42, 50][:c]
        img[k:k + 2, :] = [38, 42, 50][:c]
    return img


def _sky_tex(w, h, c):
    img = np.empty((h, w, c), np.uint8)
    img[:] = BG[:c]
    return img


class CinemaRec(BaseViz):
    """Free-camera recorder with a slow orbit. No overlay text (captions live in the page)."""

    def __init__(self, cfg, size=(1280, 720), az=(232.0, 248.0), el=-20.0, dist=0.74,
                 lookat=(0.10, 0.04, 0.14), stream_to: Path | None = None, crf=27):
        super().__init__(cfg)
        self.stream_to, self.crf, self.writer, self.last = stream_to, crf, None, None
        self.size, self.az, self.el, self.dist, self.lookat = size, az, el, dist, lookat
        self.r = None
        self.frames: list[np.ndarray] = []
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.t = 0
        self.outcome = None

    def _shot(self, d):
        u = min(1.0, self.t / (25 * 9.0))  # orbit spans ~9 s
        self.cam.azimuth = self.az[0] + (self.az[1] - self.az[0]) * (0.5 - 0.5 * np.cos(np.pi * u))
        self.cam.elevation, self.cam.distance = self.el, self.dist
        self.cam.lookat[:] = self.lookat
        self.r.update_scene(d, self.cam)
        self.r.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1
        img = self.r.render().copy()
        if self.writer is not None:
            self.writer.append_data(img)
            self.last = img
        else:
            self.frames.append(img)
        self.t += 1

    def on_episode_start(self, m, d, info):
        super().on_episode_start(m, d, info)
        if self.r is None:
            stylize(m)
            self.r = mujoco.Renderer(m, height=self.size[1], width=self.size[0])
        self.frames, self.t = [], 0
        if self.stream_to is not None:
            self.stream_to.parent.mkdir(parents=True, exist_ok=True)
            self.writer = iio.get_writer(self.stream_to, fps=self.cfg["fps"], codec="libx264", macro_block_size=1,
                                         quality=None, ffmpeg_params=["-crf", str(self.crf), "-preset", "slow",
                                                                      "-movflags", "+faststart"])
        self._shot(d)

    def on_frame(self, m, d, info):
        super().on_frame(m, d, info)
        self._shot(d)

    def on_episode_end(self, m, d, info, stats):
        super().on_episode_end(m, d, info, stats)
        self.outcome = stats.success
        for _ in range(int(self.cfg["fps"] * 0.8)):
            self._shot(d)
        if self.writer is not None:
            self.writer.close()
            self.writer = None
            Image.fromarray(self.last).save(self.stream_to.with_suffix(".jpg"), quality=80)

    def close(self):
        if self.r is not None:
            self.r.close()


def write_mp4(frames, path: Path, fps=25, crf=23):
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.mimwrite(path, frames, fps=fps, codec="libx264", macro_block_size=1, quality=None,
                 ffmpeg_params=["-crf", str(crf), "-preset", "slow", "-movflags", "+faststart"])
    Image.fromarray(frames[len(frames) // 3]).save(path.with_suffix(".jpg"), quality=82)


def transcode(src: Path, dst: Path, crop: str | None = None, crf=26, scale: str | None = None):
    dst.parent.mkdir(parents=True, exist_ok=True)
    vf = ",".join(x for x in [f"crop={crop}" if crop else None, f"scale={scale}:flags=lanczos" if scale else None] if x)
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-i", str(src)] + (["-vf", vf] if vf else []) + [
        "-c:v", "libx264", "-crf", str(crf), "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-an", str(dst)]
    subprocess.run(cmd, check=True)


def last_frame(src: Path, crop_box=None) -> Image.Image:
    fr = iio.mimread(src, memtest=False)
    img = Image.fromarray(fr[-1])
    return img.crop(crop_box) if crop_box else img


def render_demos(cfg, log, n=6):
    from expert.policy import ExpertPolicy

    eps = json.load(open(REPO / "outputs/data/curated/episodes.json"))
    train = [e for e in eps if e["split"] == "train"]
    # spread over the workspace: pick distinct xy bins with varied yaw, waypoint and grasp side
    picked, seen = [], set()
    for e in sorted(train, key=lambda e: -e["score"]):
        key = e["bin"].split("_")[0]
        if key in seen:
            continue
        want_far = len(picked) in (1, 4)
        if (e["grasp_side"] == "far") != want_far:
            continue
        if len(picked) in (2, 5) and e["variation"].get("n_waypoints", 0) == 0:
            continue
        seen.add(key)
        picked.append(e)
        if len(picked) == n:
            break
    pol = ExpertPolicy(cfg)
    meta = []
    for i, e in enumerate(picked):
        rec = CinemaRec(cfg, size=(960, 540), az=(222 + 5 * i, 238 + 5 * i))
        env = BowlEnv(cfg, viz=rec)
        ep = run_episode(env, pol, e["seed"], episode_id=e["episode_id"])
        write_mp4(rec.frames, OUT / "video" / f"demo_{i}.mp4", crf=25)
        rec.close()
        meta.append({"file": f"assets/video/demo_{i}.mp4", "seed": e["seed"], "success": ep["success"],
                     "steps": ep["n_steps"], "yaw": round(e["spawn"]["yaw_deg"]), "grasp_side": e["grasp_side"],
                     "waypoints": e["variation"].get("n_waypoints", 0)})
        log.info(f"demo {i}: seed {e['seed']} success={ep['success']} steps={ep['n_steps']}")
    return meta


def render_hero(cfg, log, seeds=(100000, 100003, 100012, 100021)):
    from eval.qwen_policy import QwenPolicy

    pol = QwenPolicy.load(str(REPO / cfg["paths"]["merged_dir"]), None, cfg["eval"]["max_new_tokens"], name="qwen")
    frames, meta = [], []
    for i, s in enumerate(seeds):
        rec = CinemaRec(cfg, size=(1280, 720), az=(226 + 8 * i, 246 + 8 * i))
        env = BowlEnv(cfg, viz=rec)
        ep = run_episode(env, pol, s, episode_id=f"hero_{s}", history_len=cfg["train"]["history_len"])
        log.info(f"hero seed {s}: success={ep['success']} steps={ep['n_steps']} invalid={ep['metrics']['n_invalid']}")
        meta.append({"seed": s, "success": ep["success"], "steps": ep["n_steps"]})
        if ep["success"]:
            frames.extend(rec.frames)
        rec.close()
    write_mp4(frames, OUT / "video" / "hero.mp4", crf=24)
    return meta


def render_progress(cfg, log, seed=100500):
    s = json.load(open(REPO / "outputs/train_summary.json"))
    meta = []
    for c in s["ckpt"]:
        step = c["step"]
        src = REPO / f"outputs/videos/train_ckpt_{step}/ckpt{step}_seed{seed}.mp4"
        dst = OUT / "video" / f"ckpt_{step}.mp4"
        transcode(src, dst, crop="480:360:0:0", crf=24, scale="640:480")
        last_frame(dst).save(dst.with_suffix(".jpg"), quality=82)
        idx = cfg["seeds"]["ckpt_eval_seeds"].index(seed)
        meta.append({"step": step, "file": f"assets/video/ckpt_{step}.mp4", "success_of_5": c["success"],
                     "this_seed": c["failures"][idx] == "" if idx < len(c["failures"]) else None})
        log.info(f"progress ckpt {step}: {c['success']}/5")
    # the final checkpoint's one failure (seed 100504): honest failure-mode clip
    src = REPO / "outputs/videos/train_ckpt_1498/ckpt1498_seed100504.mp4"
    if src.exists():
        dst = OUT / "video" / "fail_1498.mp4"
        transcode(src, dst, crop="480:360:0:0", crf=24, scale="640:480")
        last_frame(dst).save(dst.with_suffix(".jpg"), quality=82)
    return meta


def render_evalwall(cfg, log, base=False):
    """Replay the scored Qwen eval with identical batching (lockstep chunks of 25, same seeds and episode ids)
    and re-render it in the studio look. Outcomes are checked against outputs/results.json."""
    from env.rollout import run_episodes_lockstep
    from eval.qwen_policy import QwenPolicy

    key, pre = ("qwen_base", "b") if base else ("qwen", "q")
    r = json.load(open(REPO / ("outputs/results_base.json" if base else "outputs/results.json")))
    ref = {e["seed"]: e for e in r[key]["episodes"]}
    seeds = sorted(ref)
    model = cfg["train"]["base_model"] if base else str(REPO / cfg["paths"]["merged_dir"])
    pol = QwenPolicy.load(model, None, cfg["eval"]["max_new_tokens"], name=key)
    meta, mismatch = [], []
    for b0 in range(0, len(seeds), 25):
        chunk = seeds[b0:b0 + 25]
        recs = [CinemaRec(cfg, size=(480, 270), az=(228, 244), stream_to=OUT / "eval" / f"{pre}{s}.mp4", crf=28)
                for s in chunk]
        envs = [BowlEnv(cfg, viz=rc) for rc in recs]
        res = run_episodes_lockstep(envs, pol, chunk, [f"{key}_{s}" for s in chunk], cfg["train"]["history_len"])
        for ep, rc in zip(res, recs):
            rc.close()
            e = ref[ep["seed"]]
            fail = "" if ep["success"] else ("invalid_json" if ep["metrics"]["n_invalid"] else ep["failure_type"])
            if ep["success"] != e["success"] or fail != e["failure"]:
                mismatch.append((ep["seed"], e["success"], ep["success"]))
            meta.append({"seed": e["seed"], "success": e["success"], "failure": e["failure"], "steps": e["n_steps"],
                         "x": round(e["spawn"]["x"], 4), "y": round(e["spawn"]["y"], 4),
                         "yaw": round(e["spawn"]["yaw_deg"]), "replay_matches": ep["success"] == e["success"]})
        log.info(f"eval wall: {b0 + len(chunk)}/100 rendered, mismatches so far {mismatch}")
    log.info(f"eval wall done: {len(meta)} clips, {len(mismatch)} outcome mismatches vs results.json {mismatch}")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=["demos", "hero", "progress", "evalwall"])
    args = ap.parse_args()
    cfg = load_config(None)
    cfg["viz"]["fps"] = 25
    log = get_logger("showcase", cfg)
    mpath = REPO / "docs" / "assets" / "media.json"
    media = json.loads(mpath.read_text()) if mpath.exists() else {}
    for k, fn in (("demos", render_demos), ("hero", render_hero), ("progress", render_progress),
                  ("evalwall", render_evalwall), ("basewall", lambda c, lg: render_evalwall(c, lg, base=True))):
        if k in args.only:
            media[k] = fn(cfg, log)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(media, indent=1))
    log.info(f"media -> {mpath}")


if __name__ == "__main__":
    main()
