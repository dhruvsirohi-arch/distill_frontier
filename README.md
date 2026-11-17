# distill_frontier

Distill the GPTs and Claudes for robotics.

## SO-101 bowl pick-and-place: scripted expert → curated SFT data → Qwen LoRA policy

End-to-end pipeline in MuJoCo with the official **MuJoCo Menagerie `robotstudio_so101`** model (unmodified,
loaded via `MjSpec`; we only *add* world geometry, cameras and a TCP site):

1. `env/` simulation + high-level JSON primitives (IK + Cartesian interpolation executor)
2. `viz/` one visualization module (live / stream / record) used by every stage
3. `expert/` scripted expert that sees only the observation JSON Qwen sees
4. `data/generate.py` parallel demo generation → raw JSONL (all episodes, never modified)
5. `data/curate.py` quality filters, score, stratified diverse selection, top-up, splits, report
6. `train/sft.py` LoRA SFT of Qwen3-1.7B (thinking off) under a hard wall-clock budget
7. `eval/evaluate.py` 100 held-out seeds, Qwen vs expert, `results.json` + gallery

Everything is configured in **`configs/config.yaml`** (seeds, ranges, hyperparameters, model names, viz).
No paid services; secrets (optional `HF_TOKEN`) are read from the environment only.

## Setup (tested: macOS 26, Apple M5 24 GB, Python 3.12, mujoco 4.0.1, torch 2.15 MPS, transformers 5.23, peft 0.21)

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install mujoco torch transformers peft trl accelerate datasets pyyaml matplotlib scipy imageio imageio-ffmpeg pytest
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git assets/menagerie
git -C assets/menagerie sparse-checkout set robotstudio_so101
```

### Linux + NVIDIA (tested: Ubuntu 24.04, RTX 4090 24 GB, i9-14900K, Python 3.12.3, torch 2.15 cu132)

Versions are pinned in `pyproject.toml` / `uv.lock`:

```bash
uv sync --frozen          # creates .venv with the exact locked versions
# then clone assets/menagerie as above
```

* Headless rendering uses **EGL** (needs `libEGL_nvidia`, shipped with the driver). Every stage that renders
  selects it automatically before importing mujoco; `MUJOCO_GL` set in the environment wins.
* Measured here: data generation 2000 eps ≈ 1.5 min (9 workers), SFT ≈ 9 min end to end, eval (100 Qwen + 100 expert eps, lowres videos) ≈ 13 min (0.11 s/step at
  bs 4, 3296 steps = full 2 epochs, ~13 GB VRAM; well inside the 50-min budget), first run also downloads
  Qwen3-1.7B (~3.4 GB).
* If `nvidia-smi` says *Driver/library version mismatch* after a driver update, reboot; CUDA in torch kept
  working in that state, but don't rely on it.

## Commands (each stage is standalone and rerunnable; `./run_all.sh` runs them all)

| stage | command |
|---|---|
| tests (IK, determinism, success check, viz-invariance) | `.venv/bin/python -m pytest -q tests` |
| watch the expert live (macOS needs `mjpython`) | `.venv/bin/mjpython -m expert.run_expert --viz live --n 5 --speed 1` |
| expert check (50 seeds, must be ≥ 90 %) | `.venv/bin/python -m expert.run_expert` |
| data generation (2000 eps, 9 workers, MP4 every 50th) | `.venv/bin/python -m data.generate` |
| curation + report | `.venv/bin/python -m data.curate` |
| SFT (50-min budget, checkpoint videos) | `.venv/bin/python -m train.sft` (`--viz stream` to watch checkpoints) |
| evaluation + gallery | `.venv/bin/python -m eval.evaluate` |

### Visualization (`--viz` flag or `viz.mode` in config)
* `live` – passive MuJoCo viewer synced to real time × `--speed`; overlay shows episode id, seed, step,
  current action JSON, policy, running success count. Keys **1** front camera, **2** top camera, **0** free.
  On macOS it must run under `.venv/bin/mjpython`; under plain `python` (or with no display / over SSH) it
  falls back to `stream` and says so.
* `stream` – offscreen render streamed as MJPEG at `http://127.0.0.1:8765/` (URL printed) with an action /
  success-event log. (Rerun is planned as a later replacement.)
* `record` – MP4s (front | top side by side + overlay) under `outputs/videos/<stage>/`.
* Linux headless: `MUJOCO_GL=egl` is set automatically for offscreen modes; macOS uses CGL.
* Rendering only reads `MjData`; `tests/test_viz.py` checks the same seed gives bit-identical results with
  recording on and off.

## Design notes / decisions

* **Arm on a 12 cm pedestal.** The SO-101 wrist-flex limit (±95°) only allows a vertical (top-down) gripper
  when the TCP is < ~9 cm above the arm base, so with the base on the floor it could not lift the bowl over a
  table. Mounting the base on a pedestal (like clamping to a bench) gives top-down reach r ∈ [0.09, 0.27] m for
  z ∈ [0.02, 0.17] m. The robot MJCF itself is untouched.
* **Action space.** `move_to` = straight-line Cartesian TCP move with the fingers pointing down and the jaw
  aligned radially from the pan axis (exactly solvable with 5 DOF); damped-least-squares IK on a private
  `MjData`; min-jerk timing. Targets outside the reach annulus / IK failures are **rejected without motion**
  and reported as `last_result: "rejected_unreachable"`. A move is aborted as `collision` if the arm touches
  the floor/table, or the open gripper pushes/tips the bowl. Episodes are capped at 15 decisions.
* **Success** = bowl center inside the target region, resting on the table, upright (< 15° tilt), gripper
  open, arm not touching it, bowl speed < 2 cm/s during a 0.5 s settle.
* **Bowl yaw is meaningful**: a small red lug on the rim marks yaw and physically blocks grasps there; the
  expert picks a grasp point consistent with yaw (avoids the lug, switches to the far rim when needed).
* **Expert** decides from the observation JSON + last 2 actions + per-episode variation parameters sampled
  from the seed (approach side/offset, pre-grasp height, grasp depth & rim offset, lift height, 0/1
  waypoints, placement point, release height, retreat height, open-first order, Gaussian target noise).
* **Model I/O in integer millimetres.** Qwen tokenizes digits individually, so the prompt layer
  (`train/prompt.py`, shared by training and eval) converts observation/action numbers to integer mm
  (lossless: expert targets are already rounded to 1 mm; round-trip verified). Env and data stay in meters.
* **Loss** on assistant tokens only, `lm_head` evaluated only at assistant positions (memory), per-example
  weights by action type from curation (`close_gripper`, `done` ×2, `open_gripper` ×1.5).
* **Time budget.** After 20 probe steps the trainer estimates total wall-clock including checkpoint
  episodes; if > 50 min it caps the optimizer steps and logs `BUDGET EXCEEDED ...` in `outputs/logs/`.

## Outputs

* `outputs/data/raw_episodes.jsonl` (+ `raw_topup_r*.jsonl` if thin bins were topped up) – raw, all episodes
* `outputs/data/curated/{train,val}.jsonl`, `episodes.json`; `outputs/curation_report.md` + `outputs/curation/*.png`
* `outputs/qwen_lora/adapter/`, `outputs/qwen_lora/merged/`, `outputs/train_summary.json`
* `outputs/videos/{expert_check,datagen,train_ckpt_<step>,eval/qwen,eval/expert}/`
* `outputs/results.json`, `outputs/eval_episodes.jsonl`, `outputs/eval_gallery.html`
* `outputs/logs/*.log` – every stage logs to file + stdout

## Results

See `RESULTS.md` (written after the full run).
