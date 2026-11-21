#!/usr/bin/env bash
# After the overnight driver finishes: commit the rebuilt report + results to the feature branch
# and push it (never main).
cd "$(dirname "$0")/.."
until grep -q "all done" outputs/ringsim/overnight.log 2>/dev/null; do sleep 120; done
[ "$(git branch --show-current)" = "reaction-ring-sim" ] || exit 1
git add docs/ringsim outputs/ringsim
git commit -q -m "Reaction Ring report: final results (standard, fast profile, slow-motion sweep, latency profile)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q origin reaction-ring-sim
echo "[$(date '+%F %T')] branch pushed: $?" >> outputs/ringsim/overnight.log
