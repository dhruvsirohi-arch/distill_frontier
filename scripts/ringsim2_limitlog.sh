#!/usr/bin/env bash
# Append the newest Codex usage reading every minute (for cost accounting in the report).
cd "$(dirname "$0")/.."
F=outputs/ringsim2/rate_limits.json; O=outputs/ringsim2/usage_log.csv
[ -s $O ] || echo "time,primary_pct,weekly_pct,reading_age_s" > $O
while pgrep -f "[r]ingsim2_run.sh" >/dev/null; do
  python3 -c "
import json,time
l=json.load(open('$F'));print(f\"{time.strftime('%F %T')},{l['primary']['usedPercent']},{l['secondary']['usedPercent']},{round(time.time()-l.get('at',0))}\")" >> $O 2>/dev/null
  sleep 60
done
