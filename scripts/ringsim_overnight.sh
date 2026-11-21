#!/usr/bin/env bash
# Reaction Ring latency experiments, run unattended (restart-safe: every runner resumes).
# Waits for the Standard run so latency measurements are not distorted by concurrent load.
set -u
cd "$(dirname "$0")/.."
export MUJOCO_GL=egl PYTHONPATH=.
PY=.venv/bin/python
O=outputs/ringsim
log() { echo "[$(date '+%F %T')] $*" >> $O/overnight.log; }

log "waiting for the Standard run"
while pgrep -f "[r]ingsim.exam --mode standard" >/dev/null; do sleep 60; done
# finish any Standard seeds that were interrupted
$PY -m ringsim.exam --mode standard --episodes 100 --workers 8 --out $O/astra_standard.jsonl >> $O/astra_standard.log 2>&1
log "standard complete"

[ -s $O/latency_profile.json ] || $PY -m ringsim.latency_profile --n 20 --out $O/latency_profile.json >> $O/latency_profile.log 2>&1
log "latency profile complete"

$PY -m ringsim.exam --mode realtime --profile fast --episodes 100 --workers 4 \
    --out $O/astra_realtime_fast.jsonl >> $O/astra_realtime_fast.log 2>&1
log "realtime fast complete"

for k in 2 5 10 20 40 80; do
  n=30; [ $k -ge 40 ] && n=20
  $PY -m ringsim.exam --mode realtime --slowdown $k --episodes $n --workers 8 \
      --out $O/astra_realtime_slow${k}x.jsonl >> $O/astra_realtime_slow${k}x.log 2>&1
  log "slow-motion ${k}x complete"
done

$PY -m ringsim.summary --dir $O > $O/summary.txt 2>&1
$PY scripts/build_ringsim_report.py >> $O/overnight.log 2>&1
log "all done"
