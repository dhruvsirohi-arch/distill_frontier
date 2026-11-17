#!/usr/bin/env bash
# End-to-end pipeline: env tests -> expert check -> data generation -> curation -> SFT -> eval.
# Usage: ./run_all.sh            (VIZ=live|stream|record|none for the expert preview, default none)
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
VIZ=${VIZ:-none}
mkdir -p outputs/logs

echo "== 0. tests (IK, reset determinism, success check, viz-does-not-change-results)"
$PY -m pytest -q tests

echo "== 1. offscreen render smoke test (1 s clip) + expert check on 50 seeds (must be >= 90%)"
$PY - <<'PY'
import sys; sys.path.insert(0, ".")
from viz.viz import setup_gl, RecordViz
setup_gl("record")
from env.common import load_config
from env.so101_env import BowlEnv, EpisodeStats
cfg = load_config(); v = RecordViz(cfg, "smoke"); env = BowlEnv(cfg, viz=v)
env.reset(0, episode_id="render_test_1s", policy_name="smoke"); env._step_sim(env._secs(1.0))
s = EpisodeStats(); s.success = True; v.on_episode_end(env.m, env.d, env._viz_info(), s)
print("1 s test clip:", v.last_path)
PY
if [ "$VIZ" = "live" ]; then .venv/bin/mjpython -m expert.run_expert --viz live --n 3; fi
$PY -m expert.run_expert --viz none

echo "== 2. data generation (~2000 expert episodes, parallel)"
$PY -m data.generate

echo "== 3. curation (+ targeted top-up for thin bins)"
$PY -m data.curate

echo "== 4. LoRA SFT (hard 50-min budget, checkpoint episodes recorded)"
$PY -m train.sft

echo "== 5. evaluation on held-out seeds 100000-100099 (Qwen + expert)"
$PY -m eval.evaluate
echo "done: outputs/results.json, outputs/eval_gallery.html, outputs/curation_report.md"
