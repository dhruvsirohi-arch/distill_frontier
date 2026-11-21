"""Keep runs inside the Codex plan's limits and never into paid credits.

Every Astra client writes its newest rate-limit reading to $RINGSIM_LIMITS_FILE. Drivers call
`wait_for_room()` before starting an episode (and Standard mode before each call, since its
simulation is paused anyway): it sleeps through the 5-hour window when that is nearly used and
raises BudgetExhausted once the weekly limit reaches the stop level.
"""
from __future__ import annotations

import json
import os
import time

WINDOW_PAUSE = 93.0
WEEKLY_STOP = 92.0


class BudgetExhausted(RuntimeError):
    pass


def read():
    path = os.environ.get("RINGSIM_LIMITS_FILE")
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def wait_for_room(log=print, window_pause=WINDOW_PAUSE, weekly_stop=WEEKLY_STOP):
    """Returns seconds waited."""
    waited = 0.0
    while True:
        lim = read()
        if not lim:
            return waited
        weekly = (lim.get("secondary") or {}).get("usedPercent", 0)
        if weekly >= weekly_stop or lim.get("rateLimitReachedType"):
            raise BudgetExhausted(f"weekly limit at {weekly}% (stop at {weekly_stop}%)")
        prim = lim.get("primary") or {}
        if prim.get("usedPercent", 0) < window_pause:
            return waited
        reset = prim.get("resetsAt", time.time() + 600)
        if reset <= time.time():
            # the reading is stale (window already reset); wait for a fresh one from any call
            if time.time() - lim.get("at", 0) > 120:
                return waited
        pause = max(60.0, reset - time.time() + 90)
        log(f"5-hour window at {prim.get('usedPercent')}%: pausing {pause / 60:.0f} min")
        time.sleep(pause)
        waited += pause
        # after the reset the file is stale until the next call: trust the reset time
        lim["primary"]["usedPercent"] = 0
        path = os.environ.get("RINGSIM_LIMITS_FILE")
        with open(path, "w") as f:
            json.dump(lim, f)
