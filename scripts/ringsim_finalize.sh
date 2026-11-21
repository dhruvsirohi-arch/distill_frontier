#!/usr/bin/env bash
# After the overnight driver: unloaded latency profile, rebuild report, commit + push the branch.
cd "$(dirname "$0")/.."
export MUJOCO_GL=egl PYTHONPATH=.
until grep -q "all done" outputs/ringsim/overnight.log 2>/dev/null; do sleep 60; done
.venv/bin/python -m ringsim.latency_profile --n 20 --out outputs/ringsim/latency_profile.json \
    > outputs/ringsim/latency_profile.log 2>&1
.venv/bin/python scripts/build_ringsim_report.py >> outputs/ringsim/overnight.log 2>&1
[ "$(git branch --show-current)" = "reaction-ring-sim" ] || exit 1
git add ringsim scripts docs/ringsim outputs/ringsim
git commit -q -m "Reaction Ring report: final results (standard, fast profile, slow-motion sweep, latency profile)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q origin reaction-ring-sim
echo "[$(date '+%F %T')] finalize: branch pushed (exit $?)" >> outputs/ringsim/overnight.log
