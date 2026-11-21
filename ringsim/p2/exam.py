"""Phase 2 Reaction Ring exam: GPT-6.2-Astra with eyes (30 fps camera, 10 Hz lidar, 30 Hz
proprioception) and a full record of its thinking.

  standard    simulation paused while Astra thinks; it is called on every frame
  realtime    simulation runs at wall-clock speed; one call in flight; each call gets the newest
              sensors; the next call starts when the previous reply lands
  pipelined   as realtime, but a new call starts every --period seconds (up to --pool in flight);
              a reply is applied unless one based on newer sensors was already applied
  privileged  full state, simulate/submit plans (as Phase 1, at the 30 Hz control rate)

Per episode: a compact record in --out (JSONL) and a full trace of every call in
<out dir>/traces/<run>/<seed>.jsonl.gz: the exact text sent, the reasoning summaries the server
streamed, reasoning-token counts, the raw reply, the parsed reasoning/action/notes, and timings.

    python -m ringsim.p2.exam --mode standard --episodes 100 --workers 8 \
        --out outputs/ringsim2/astra_standard.jsonl
"""
from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import os
import queue
import threading
import time
import traceback
from pathlib import Path

import numpy as np

from . import budget

SEED0 = 50000
MAX_CALLS = 900                   # standard: more than the frames in any game
GRACE_S = 90.0                    # realtime: let calls in flight at the end finish (for the trace)


def _clean(a):
    from ..robot import READY, merge_action
    return merge_action(READY, a)


def _merge(prev, new):
    from ..robot import merge_action
    return merge_action(prev, new)


class Trace:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows = []
        self.lock = threading.Lock()

    def add(self, row):
        with self.lock:
            self.rows.append(row)

    def close(self):
        with gzip.open(self.path, "wt") as f:
            for r in sorted(self.rows, key=lambda r: r.get("call", 0)):
                f.write(json.dumps(r, default=float) + "\n")


def _summary_text(info):
    return [s for it in info.get("reasoning", []) for s in it.get("summary", [])]


def _call(astra, instr, snap, notes, rec, trace):
    """One stateless decision. Fills `rec` (compact) and writes the full trace row."""
    from ..astra import AstraError
    from .agent import SCHEMA, build_inputs
    t_build = time.monotonic()
    row = {"call": rec["call"], "frame": snap["frame"], "t_obs": snap["t"]}
    try:
        inputs, text = build_inputs(snap, notes)
        row["input_text"] = text
        rec["build_s"] = round(time.monotonic() - t_build, 3)
        th = astra.new_thread(instr)
        out, info = astra.ask(th, inputs, SCHEMA)
        action = _clean(out["action"])
        new_notes = str(out.get("notes", ""))[:400]
        reasoning = out.get("reasoning", {})
        rec.update(latency_s=info["latency_s"], first_token_s=info["first_token_s"],
                   usage=info["usage"], reasoning=reasoning, notes=new_notes, action=action,
                   n_summary=len(_summary_text(info)))
        row.update(latency_s=info["latency_s"], first_token_s=info["first_token_s"],
                   usage=info["usage"], reasoning_items=info["reasoning"],
                   reply_raw=info["raw"], reasoning=reasoning, action=action, notes=new_notes)
        return action, new_notes
    except (AstraError, KeyError, TypeError, ValueError) as e:
        rec["error"] = str(e)[:300]
        row["error"] = str(e)[:1000]
        return None, None
    finally:
        trace.add(row)


# ------------------------------------------------------------------------------------------------
def run_standard(seed, effort, trace):
    from ..astra import Astra
    from .agent import instructions
    from .world import World
    w = World(seed)
    instr = instructions("standard", w.proprio.names)
    decisions, notes = [], ""
    with Astra(effort=effort, summary="detailed") as astra:
        while not w.done and len(decisions) < MAX_CALLS:
            snap = w.snapshot()
            rec = {"call": len(decisions), "frame": snap["frame"], "t_obs": snap["t"]}
            for attempt in range(4):
                # the simulation is paused, so waiting out a usage window changes nothing
                waited = budget.wait_for_room()
                if waited:
                    rec["paused_for_limit_s"] = round(rec.get("paused_for_limit_s", 0) + waited)
                action, new_notes = _call(astra, instr, snap, notes, rec, trace)
                err = rec.get("error", "").lower()
                if action is None and ("limit" in err or "rate" in err) and attempt < 3:
                    rec.pop("error")
                    rec["retries"] = rec.get("retries", 0) + 1
                    time.sleep(120)
                    continue
                break
            if new_notes is not None:
                notes = new_notes
            rec["action"] = action
            decisions.append(rec)
            w.step(action)
        limits = astra.rate_limits
    return w, decisions, {"rate_limits": limits}


def run_realtime(seed, effort, trace, pipelined=False, period=0.25, pool=40):
    from ..astra import Astra
    from .agent import instructions
    from .world import FRAME, World
    w = World(seed)
    instr = instructions("pipelined" if pipelined else "realtime", w.proprio.names)
    n_proc = pool if pipelined else 1
    procs = [None] * n_proc

    def start(k):
        procs[k] = Astra(effort=effort, summary="detailed")
        procs[k].new_thread(instr)                     # warm the connection before the clock
    starters = [threading.Thread(target=start, args=(k,)) for k in range(n_proc)]
    for s in starters:
        s.start()
    for s in starters:
        s.join()
    free: queue.Queue = queue.Queue()
    for p in procs:
        free.put(p)
    replies: queue.Queue = queue.Queue()
    decisions, workers = [], []
    state = {"notes": "", "notes_t": -1.0, "applied_t": -1.0}

    def ask(astra, snap, notes, rec):
        try:
            action, new_notes = _call(astra, instr, snap, notes, rec, trace)
            rec["wall_reply"] = time.monotonic()
            replies.put((rec, action, new_notes))
        finally:
            free.put(astra)

    stats = {"overruns": 0, "pool_exhausted": 0}
    t0 = time.monotonic()
    t_next, last_dispatch = t0, -1e9
    in_flight = 0
    try:
        while not w.done:
            arrived = []
            while True:
                try:
                    arrived.append(replies.get_nowait())
                except queue.Empty:
                    break
            in_flight -= len(arrived)
            action, best = None, None
            for rec, act, new_notes in arrived:
                if new_notes is not None and rec["t_obs"] > state["notes_t"]:
                    state["notes"], state["notes_t"] = new_notes, rec["t_obs"]
                if act is not None and rec["t_obs"] > state["applied_t"] and \
                        (best is None or rec["t_obs"] > best["t_obs"]):
                    best, action = rec, act
            for rec, act, _ in arrived:
                if act is not None and rec is not best:
                    rec["superseded"] = True          # a reply on newer sensors already won
            if best is not None:
                state["applied_t"] = best["t_obs"]
                best.update(t_applied=round(w.t, 4), applied=True,
                            wall_applied=round(time.monotonic() - t0, 3))
            now = time.monotonic()
            want = (in_flight == 0) if not pipelined else (now - last_dispatch >= period - 1e-6)
            if want:
                try:
                    astra = free.get_nowait()
                except queue.Empty:
                    astra = None
                    stats["pool_exhausted"] += 1
                if astra is not None:
                    snap = w.snapshot()
                    rec = {"call": len(decisions), "frame": snap["frame"], "t_obs": snap["t"],
                           "wall_obs": round(now - t0, 3)}
                    decisions.append(rec)
                    in_flight += 1
                    last_dispatch = now
                    th = threading.Thread(target=ask, args=(astra, snap, state["notes"], rec),
                                          daemon=True)
                    th.start()
                    workers.append(th)
            w.step(action)
            t_next += FRAME
            slack = t_next - time.monotonic()
            if slack > 0:
                time.sleep(slack)
            else:
                stats["overruns"] += 1
                stats["max_lag_ms"] = max(stats.get("max_lag_ms", 0), round(-slack * 1000, 1))
        t_end = time.monotonic()
        for th in workers:                   # finish thoughts already in progress (trace only)
            th.join(timeout=max(0.0, GRACE_S - (time.monotonic() - t_end)))
        for rec in decisions:
            if "t_applied" not in rec:
                rec["applied"] = False
                if rec.get("wall_reply") is not None and rec["wall_reply"] > t_end:
                    rec["arrived_after_end"] = True
                elif rec.get("wall_reply") is None:
                    rec["unfinished"] = True
            if rec.get("wall_reply") is not None:
                rec["wall_reply"] = round(rec["wall_reply"] - t0, 3)
            rec["action"] = rec.get("action")
        limits = next((p.rate_limits for p in procs if p and p.rate_limits), None)
    finally:
        for p in procs:
            if p:
                p.close()
    stats["rate_limits"] = limits
    return w, decisions, stats


# ------------------------------------------------------------------------------------------------
def play_plan(seed, plan):
    from ..game import Game
    from .world import FRAME, PHYS
    game = Game(render=False, tick=FRAME, phys_dt=PHYS)
    game.reset(seed)
    keys = sorted(((float(k["t"]), k["action"]) for k in plan), key=lambda x: x[0])
    k, trace = 0, []
    while not game.done:
        action = None
        while k < len(keys) and keys[k][0] <= game.t + 1e-9:
            action = _clean(keys[k][1])
            k += 1
        game.step(action, observe=False)
        p = game.proprio()["pinch"]
        trace.append((round(game.t, 2), {s: np.round(p[s], 3).tolist() for s in p}))
    return game, trace


def _sim_report(seed, plan):
    game, trace = play_plan(seed, plan)
    tr = dict(trace)
    rows = []
    for e in game.result()["events"]:
        diag = {}
        for dt in (0.0, 0.2, 0.3):
            t = round(round((e["release_t"] + dt) * 30) / 30, 2)
            if t in tr:
                diag[f"pinch_at_release+{dt}"] = tr[t]
        rows.append({**e, "stick_xy": np.round(game.slots[e["slot"]][:2], 3).tolist(), **diag})
    return {"caught": game.result()["caught"], "drops": rows}


def run_privileged(seed, effort, trace, turns_allowed=8):
    from ..astra import Astra, AstraError
    from ..game import Game
    from .agent import PRIV_SCHEMA2, priv_instructions
    from .world import FRAME, PHYS
    game = Game(render=False, tick=FRAME, phys_dt=PHYS)
    game.reset(seed)
    turns, plan, submitted = [], [], False
    with Astra(effort=effort, summary="detailed") as astra:
        th = astra.new_thread(priv_instructions())
        msg = {"turn": 1, "turns_allowed": turns_allowed, "state_at_start": game.privileged_state(),
               "note": "Reply with op 'simulate' to test a plan or 'submit' to finish."}
        for k in range(turns_allowed):
            if k == turns_allowed - 1:
                msg["note"] = "Final turn: your reply is the submission whatever op you choose."
            rec = {"turn": k + 1}
            row = {"call": k, "input_text": json.dumps(msg, default=float)}
            try:
                out, info = astra.ask(th, [{"type": "text", "text": row["input_text"]}],
                                      PRIV_SCHEMA2)
                rec.update(latency_s=info["latency_s"], usage=info["usage"], op=out["op"],
                           keyframes=len(out["plan"]), reasoning=out.get("reasoning", "")[:2000],
                           notes=out.get("notes", "")[:400])
                row.update(latency_s=info["latency_s"], usage=info["usage"],
                           reasoning_items=info["reasoning"], reply_raw=info["raw"])
                plan = out["plan"]
            except (AstraError, KeyError, TypeError) as e:
                rec["error"] = row["error"] = str(e)[:300]
                trace.add(row)
                turns.append(rec)
                msg = {"turn": k + 2, "error": "your last reply failed: " + str(e)[:200]}
                continue
            trace.add(row)
            turns.append(rec)
            if out["op"] == "submit" or k == turns_allowed - 1:
                submitted = True
                break
            report = _sim_report(seed, plan)
            rec["sim_caught"] = report["caught"]
            msg = {"turn": k + 2, "turns_left": turns_allowed - k - 1, "simulation": report}
        limits = astra.rate_limits
    final, _ = play_plan(seed, plan)
    return final, turns, plan, submitted, limits


# ------------------------------------------------------------------------------------------------
def episode(mode, seed, effort, out_dir, period=0.25, pool=40):
    os.environ.setdefault("MUJOCO_GL", "egl")
    t0 = time.time()
    run = mode if mode != "realtime" else "realtime"
    trace = Trace(Path(out_dir) / "traces" / run / f"{seed}.jsonl.gz")
    rec = {"phase": 2, "mode": mode, "seed": seed, "model": "gpt-6.2-astra", "effort": effort,
           "fps": 30, "lidar_hz": 10}
    try:
        if mode == "privileged":
            game, turns, plan, submitted, limits = run_privileged(seed, effort, trace)
            rec.update(turns=turns, plan=plan, submitted=submitted, rate_limits=limits)
        else:
            if mode == "standard":
                w, decisions, stats = run_standard(seed, effort, trace)
            else:
                w, decisions, stats = run_realtime(seed, effort, trace, mode == "pipelined",
                                                   period, pool)
                rec.update(period=period if mode == "pipelined" else None,
                           pool=pool if mode == "pipelined" else 1)
            game = w.game
            rec["decisions"] = decisions
            rec.update(stats)
        rec.update(game.result())
        rec["applied"] = [[int(k), a] for k, a in game.actions]
    except Exception:
        rec["fatal"] = traceback.format_exc()[-2000:]
    finally:
        trace.close()
    rec["wall_s"] = round(time.time() - t0, 1)
    return rec


def _worker(args):
    return episode(*args)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("privileged", "standard", "realtime", "pipelined"),
                    required=True)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--start", type=int, default=SEED0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--period", type=float, default=0.25)
    ap.add_argument("--pool", type=int, default=40)
    ap.add_argument("--stop-at", type=float, default=budget.WEEKLY_STOP,
                    help="stop launching episodes once the weekly limit is this %% used")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            if "fatal" not in r:
                done.add(r["seed"])
    todo = [s for s in range(args.start, args.start + args.episodes) if s not in done]
    print(f"{args.mode}: {len(done)} done, {len(todo)} to run", flush=True)
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
    ctx = mp.get_context("spawn")
    jobs = [(args.mode, s, args.effort, str(out.parent), args.period, args.pool) for s in todo]
    running = set()
    stop = False
    with ProcessPoolExecutor(args.workers, mp_context=ctx, max_tasks_per_child=1) as ex, \
            out.open("a") as f:
        while (jobs and not stop) or running:
            while jobs and not stop and len(running) < args.workers:
                try:
                    budget.wait_for_room(weekly_stop=args.stop_at,
                                         log=lambda m: print(m, flush=True))
                except budget.BudgetExhausted as e:
                    print(f"stopping: {e}", flush=True)
                    stop = True
                    break
                running.add(ex.submit(episode, *jobs.pop(0)))
            if not running:
                break
            finished, running = wait(running, return_when=FIRST_COMPLETED)
            for fut in finished:
                rec = fut.result()
                f.write(json.dumps(rec, default=float) + "\n")
                f.flush()
                used = ((rec.get("rate_limits") or {}).get("secondary") or {}).get("usedPercent")
                status = rec["fatal"].strip().splitlines()[-1] if "fatal" in rec else \
                    f"{rec['caught']}/{rec['drops']}"
                print(f"seed {rec['seed']}: {status}  ({rec['wall_s']} s, weekly {used}%)",
                      flush=True)

if __name__ == "__main__":
    main()
