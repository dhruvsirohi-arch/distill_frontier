#!/usr/bin/env bash
# Phase 2 runs (budget-scaled), unattended and restart-safe: every runner resumes, pauses through
# the Codex 5-hour window, and stops at 92% of the weekly limit (never into paid credits).
set -u
cd "$(dirname "$0")/.."
export MUJOCO_GL=egl PYTHONPATH=. RINGSIM_LIMITS_FILE=outputs/ringsim2/rate_limits.json
PY=.venv/bin/python
O=outputs/ringsim2
mkdir -p $O
log() { echo "[$(date '+%F %T')] $*" >> $O/run.log; }
run() {  # mode episodes workers [extra args]
  local mode=$1 n=$2 w=$3; shift 3
  log "start $mode ($n episodes)"
  $PY -m ringsim.p2.exam --mode $mode --episodes $n --workers $w "$@" \
      --out $O/astra_$mode.jsonl >> $O/astra_$mode.log 2>&1
  log "end $mode: $(wc -l < $O/astra_$mode.jsonl 2>/dev/null) records"
}
run standard 1 1
run realtime 100 2
run pipelined 3 1
run privileged 20 2
log "all done"
