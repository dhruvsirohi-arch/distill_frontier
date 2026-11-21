"""Reaction Ring exam: Privileged, Standard and Realtime protocols for GPT-6.2-Astra.

  privileged  The agent runs alongside the simulator: it reads the full state (stick poses, the
              release schedule, the judging rule), may simulate candidate plans from a reset as
              often as its turn budget allows, then submits one plan. The submission is replayed
              in a fresh simulator and that replay is graded.
  standard    The simulator is a separate process. The agent sees only robot sensors (two head
              RGB panels, depth, proprioception) and never object poses or the success rule.
              The simulation pauses while the agent thinks; each reply says how long to run before
              the next observation. No resets. The live episode is graded.
  realtime    As standard, but the simulation runs on the wall clock (25 Hz) while the agent
              thinks. The previous command is held until a new one arrives.

Usage:
  python -m ringsim.exam --mode realtime --episodes 100 --workers 4 --out outputs/ringsim/astra_realtime.jsonl
"""
from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import queue
import threading
import time
import traceback
from pathlib import Path

import numpy as np

SEED0 = 50000
MAX_STANDARD_DECISIONS = 600
PRIV_TURNS = 8

# ------------------------------------------------------------------------------------------------
# prompts
# ------------------------------------------------------------------------------------------------
ROBOT = """\
ROBOT: Unitree G1-D: a wheeled base, a humanoid torso on a lift, two 7-joint arms with parallel
grippers (jaws open up to 80 mm). You command it through its motion controller:
  base       [v, w]: forward speed in m/s (|v| <= 0.6) and turn rate in rad/s (|w| <= 2.0)
  torso_yaw  waist rotation in rad (|yaw| <= 1.5); the head cameras turn with the torso
  left/right {"pos": [x, y, z], "grip": g}
             pos: where the gripper's pinch point (between the fingertips) should go, in metres
             in the robot base frame: x forward, y to the robot's left, z up from the floor.
             The controller keeps the gripper level, jaws closing horizontally, pointing outward
             from the torso. Reachable at heights 1.0-1.45 m and 0.25-0.62 m from the torso
             axis; the left arm covers the left side and the front, the right arm the right side
             and the front. Hands move about 0.3 m in 0.2 s.
             grip: 1 = jaws fully open, 0 = closed. Closing fully takes about 0.1 s.
A command stays in force until you send another. Start pose: left pos [0.38, 0.2, 1.3], right
pos [0.38, -0.2, 1.3], both grips 1, base still, torso 0."""

GAME = """\
GAME: Reaction Ring. You stand at the centre of an arcade machine whose curved overhead bar holds
8 foam sticks (40 cm long, 3 cm thick, pink) on electromagnets, on a half circle of about 0.5 m
radius around you, from your left side through the front to your right side. The bar is about
1.95 m above the floor, so the hanging sticks span roughly 1.55-1.95 m. After a 3 s countdown
(game clock t = 0 to 3 s) the magnets release the sticks one at a time in a random order, with
0.5-1.5 s between releases. A released stick free-falls and reaches the floor about 0.55 s
later. Catch as many as you can by closing a gripper around a falling stick. A caught stick is
taken out of your hand automatically. Do not touch hanging sticks before they drop, and catch
with open jaws: a stick that lands on jaws already narrower than the stick does not count. There
is no hint about which stick drops next."""

SENSORS = """\
SENSORS each turn: the game clock t; image 1: head RGB, two panels side by side (left half looks
forward-left, right half forward-right; each panel 320x240, about 106 degrees wide); image 2:
head depth for the same panels, grayscale from black (0 m) to white (2.5 m or more); image 3: the
RGB from your previous turn (for spotting motion); joint readings, the measured pinch points and
jaw openings, wheel odometry, the command currently in force, and the notes you left yourself."""

STANDARD_RULES = """\
TIMING (Standard): the simulation is paused while you think. Your reply's advance_ms (40-2000,
in steps of 40 ms) says how long the simulation runs with your new command before you get the
next observation. Short steps let you react; long steps save turns."""

REALTIME_RULES = """\
TIMING (Realtime): the simulation keeps running while you think. Your command is applied the
moment your reply arrives, and you then receive a fresh observation. advance_ms is ignored.
Thinking time is game time: the clock in your next observation shows how much passed."""

PRIV = """\
MODE: Privileged. You run alongside the simulator and can read its full state, including the
exact release schedule, every stick's pose, and the judging rule. You may simulate candidate
plans from the episode's start as often as your turn budget allows; each simulation reports
what happened. Then submit one plan. The submitted plan is replayed in a fresh simulator and
that replay is scored.
JUDGE: a stick is caught when both jaw pads of one gripper touch it continuously for 0.2 s
before it touches the floor and within 1.5 s of release. A stick with a pad touching it at the
moment its magnet releases is a foul (no score). If the stick's bottom end enters a gripper
whose jaw gap is under 36 mm (the stick is 32 mm thick), the hold is "trapped" (no score). The game ends 1.7 s after the last release.
PLAN: a list of keyframes {"t": seconds on the game clock, "action": {...}} in time order; each
action takes effect at the first 40 ms control tick at or after t and holds until the next
keyframe. Every action must contain base, torso_yaw, left and right (send the full command).
Physics facts: control tick 40 ms; a stick hanging from magnet height 1.95 m has its bottom at
1.55 m; after release its bottom is at 1.55 - 4.905 t^2 m. Gripper jaws close at about
0.4 m/s per finger (80 mm to 0 in about 0.1 s)."""

ARM_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["pos", "grip"],
              "properties": {"pos": {"type": "array", "items": {"type": "number"},
                                     "minItems": 3, "maxItems": 3},
                             "grip": {"type": "number"}}}
ACTION_SCHEMA = {"type": "object", "additionalProperties": False,
                 "required": ["base", "torso_yaw", "left", "right"],
                 "properties": {"base": {"type": "array", "items": {"type": "number"},
                                         "minItems": 2, "maxItems": 2},
                                "torso_yaw": {"type": "number"},
                                "left": ARM_SCHEMA, "right": ARM_SCHEMA}}
SENSOR_SCHEMA = {"type": "object", "additionalProperties": False,
                 "required": ["action", "advance_ms", "notes"],
                 "properties": {"action": ACTION_SCHEMA,
                                "advance_ms": {"type": "integer"},
                                "notes": {"type": "string"}}}
PRIV_SCHEMA = {"type": "object", "additionalProperties": False,
               "required": ["op", "plan", "notes"],
               "properties": {"op": {"type": "string", "enum": ["simulate", "submit"]},
                              "plan": {"type": "array", "items": {
                                  "type": "object", "additionalProperties": False,
                                  "required": ["t", "action"],
                                  "properties": {"t": {"type": "number"},
                                                 "action": ACTION_SCHEMA}}},
                              "notes": {"type": "string"}}}


def sensor_instructions(mode, profile="full"):
    rules = STANDARD_RULES if mode == "standard" else REALTIME_RULES
    return "\n\n".join([
        "You control a real-time robot in a physics simulation. Reply with JSON only: your next "
        "command (action), advance_ms, and short notes (<= 400 characters) that will be shown "
        "back to you next turn (each turn you see only the current sensors, the previous image "
        "and these notes).", ROBOT, GAME, FAST_SENSORS if profile == "fast" else SENSORS, rules])


def priv_instructions():
    return "\n\n".join([
        "You control a robot in a physics simulation. Reply with JSON only.", ROBOT, GAME, PRIV])


# ------------------------------------------------------------------------------------------------
# sensor-mode episodes
# ------------------------------------------------------------------------------------------------
def _sensor_turn_inputs(obs, prev_rgb, held, notes):
    from .astra import depth_png, png_part
    text = {"t": obs["t"], "pinch": obs["pinch"], "grip": obs["grip"],
            "odometry": obs["odometry"], "joints": obs["joints"],
            "command_in_force": held, "your_notes": notes}
    parts = [{"type": "text", "text": "Observation " + json.dumps(text)},
             png_part(obs["rgb"]), png_part(depth_png(obs["depth"]))]
    parts.append(png_part(prev_rgb if prev_rgb is not None else obs["rgb"]))
    return parts


def _fast_turn_inputs(obs, prev_rgb, held, notes):
    """Lowest-latency payload: one downscaled RGB image and compact text."""
    from PIL import Image

    from .astra import png_part
    small = np.asarray(Image.fromarray(obs["rgb"]).resize((320, 120)))
    text = {"t": obs["t"], "pinch": obs["pinch"], "grip": obs["grip"], "your_notes": notes}
    return [{"type": "text", "text": "Observation " + json.dumps(text)}, png_part(small)]


FAST_SENSORS = """SENSORS each turn (fast profile): the game clock t, one head RGB image (two panels side by side,
left half looks forward-left, right half forward-right, downscaled to 320x120), the measured
pinch points and jaw openings, and the notes you left yourself."""


def _clean_action(a):
    from .robot import merge_action, READY
    return merge_action(READY, a)        # clamp to limits, fill anything malformed


def run_standard(seed, effort, log):
    from .astra import Astra, AstraError
    from .game import Game, TICK
    game = Game(render=True)
    obs = game.reset(seed)
    decisions, notes, prev_rgb = [], "", None
    held = copy.deepcopy(game.ctrl.cmd)
    with Astra(effort=effort) as astra:
        instr = sensor_instructions("standard")
        while not game.done and len(decisions) < MAX_STANDARD_DECISIONS:
            rec = {"t_obs": obs["t"]}
            try:
                th = astra.new_thread(instr)
                out, info = astra.ask(th, _sensor_turn_inputs(obs, prev_rgb, held, notes),
                                      SENSOR_SCHEMA)
                action = _clean_action(out["action"])
                ticks = int(np.clip(round(out["advance_ms"] / 1000 / TICK), 1, 50))
                notes = str(out.get("notes", ""))[:400]
                rec.update(latency_s=info["latency_s"], usage=info["usage"], action=action,
                           advance_ticks=ticks, notes=notes)
            except (AstraError, KeyError, TypeError, ValueError) as e:
                action, ticks = None, 1
                rec.update(error=str(e)[:300])
            decisions.append(rec)
            prev_rgb = obs["rgb"]
            obs, _ = game.step(action, ticks=ticks)
            held = copy.deepcopy(game.ctrl.cmd)
    return game, decisions


def run_realtime(seed, effort, log, slowdown=1.0, profile="full"):
    """Simulator ticks on a wall-clock grid of 40 ms x slowdown; one decision outstanding at a
    time. slowdown > 1 runs the world in slow motion (a time-dilation probe of reaction speed).
    profile="fast": one small image, compact text, one reused thread (prompt caching)."""
    from .astra import Astra, AstraError
    from .game import Game, TICK
    game = Game(render=True)
    obs = game.reset(seed)
    replies: queue.Queue = queue.Queue()
    decisions = []
    state = {"busy": False, "notes": "", "prev_rgb": None}
    astra = Astra(effort=effort)
    instr = sensor_instructions("realtime", profile)
    make_inputs = _fast_turn_inputs if profile == "fast" else _sensor_turn_inputs
    shared = {}

    def ask(obs, held, rec):
        try:
            th = shared["thread"] if profile == "fast" else astra.new_thread(instr)
            out, info = astra.ask(th, make_inputs(obs, state["prev_rgb"], held,
                                                  state["notes"]), SENSOR_SCHEMA)
            rec.update(latency_s=info["latency_s"], usage=info["usage"])
            replies.put((rec, _clean_action(out["action"]), str(out.get("notes", ""))[:400]))
            state["prev_rgb"] = obs["rgb"]
        except Exception as e:
            if game.done:                       # still thinking when the game ended
                rec.update(unfinished_at_end=True,
                           elapsed_s=round(time.monotonic() - rec["wall_obs"], 3))
            else:
                rec.update(error=str(e)[:300],
                           latency_s=round(time.monotonic() - rec["wall_obs"], 3))
            replies.put((rec, None, None))

    try:
        shared["thread"] = astra.new_thread(instr)   # warm the connection before the clock starts
        overruns = 0
        t0 = time.monotonic()
        t_next = t0
        while not game.done:
            action = None
            while True:
                try:
                    rec, act, new_notes = replies.get_nowait()
                except queue.Empty:
                    break
                state["busy"] = False
                if act is not None:
                    action = act
                    state["notes"] = new_notes
                    rec.update(action=act, t_applied=round(game.t, 3),
                               wall_applied=round(time.monotonic() - t0, 3))
            if not state["busy"]:
                # a fresh frame, captured at the moment the request is sent
                obs = game.observe()
                held = game.ctrl.cmd if action is None else _merge(game.ctrl.cmd, action)
                rec = {"t_obs": obs["t"], "wall_obs": time.monotonic()}
                decisions.append(rec)
                state["busy"] = True
                threading.Thread(target=ask, args=(obs, copy.deepcopy(held), rec),
                                 daemon=True).start()
            game.step(action, ticks=1, observe=False)
            t_next += TICK * slowdown
            slack = t_next - time.monotonic()
            if slack > 0:
                time.sleep(slack)
            else:
                overruns += 1
    finally:
        astra.close()
    for rec in decisions:
        rec["wall_obs"] = round(rec["wall_obs"] - t0, 3)
    return game, decisions, overruns


def _merge(prev, new):
    from .robot import merge_action
    return merge_action(prev, new)


# ------------------------------------------------------------------------------------------------
# privileged
# ------------------------------------------------------------------------------------------------
def play_plan(seed, plan, render=False):
    """Run a keyframe plan from the start of the seeded episode in a fresh simulator."""
    from .game import Game, TICK
    game = Game(render=render)
    game.reset(seed)
    keys = sorted(((float(k["t"]), k["action"]) for k in plan), key=lambda x: x[0])
    k = 0
    trace = []
    while not game.done:
        action = None
        while k < len(keys) and keys[k][0] <= game.t + 1e-9:
            action = _clean_action(keys[k][1])
            k += 1
        game.step(action, observe=False)
        trace.append((round(game.t, 2), game.proprio()["pinch"]))
    return game, trace


def _sim_report(seed, plan):
    game, trace = play_plan(seed, plan)
    rows = []
    tr = dict(trace)
    for e in game.result()["events"]:
        slot = e["slot"]
        diag = {}
        for dt in (0.0, 0.2, 0.3):
            t = round(round((e["release_t"] + dt) / 0.04) * 0.04, 2)
            if t in tr:
                diag[f"pinch_at_release+{dt}"] = tr[t]
        rows.append({**e, "stick_xy": np.round(game.slots[slot][:2], 3).tolist(), **diag})
    return {"caught": game.result()["caught"], "drops": rows}


def run_privileged(seed, effort, log):
    from .astra import Astra, AstraError
    from .game import Game
    game = Game(render=False)
    game.reset(seed)
    state = game.privileged_state()
    turns, plan, submitted = [], [], False
    with Astra(effort=effort) as astra:
        th = astra.new_thread(priv_instructions())
        msg = {"turn": 1, "turns_allowed": PRIV_TURNS, "state_at_start": state,
               "note": "Reply with op 'simulate' to test a plan or 'submit' to finish."}
        for k in range(PRIV_TURNS):
            last = k == PRIV_TURNS - 1
            if last:
                msg["note"] = "Final turn: your reply is the submission whatever op you choose."
            rec = {"turn": k + 1}
            try:
                out, info = astra.ask(th, [{"type": "text", "text": json.dumps(msg)}], PRIV_SCHEMA)
                rec.update(latency_s=info["latency_s"], usage=info["usage"], op=out["op"],
                           keyframes=len(out["plan"]), notes=out.get("notes", "")[:400])
                plan = out["plan"]
            except (AstraError, KeyError, TypeError) as e:
                rec.update(error=str(e)[:300])
                turns.append(rec)
                msg = {"turn": k + 2, "error": "your last reply failed: " + str(e)[:200]}
                continue
            turns.append(rec)
            if out["op"] == "submit" or last:
                submitted = True
                break
            report = _sim_report(seed, plan)
            rec["sim_caught"] = report["caught"]
            msg = {"turn": k + 2, "turns_left": PRIV_TURNS - k - 1, "simulation": report}
    final, _ = play_plan(seed, plan)          # the graded replay, fresh simulator
    return final, turns, plan, submitted


# ------------------------------------------------------------------------------------------------
# runner
# ------------------------------------------------------------------------------------------------
def episode(mode, seed, effort, slowdown=1.0, profile="full"):
    os.environ.setdefault("MUJOCO_GL", "egl")
    t0 = time.time()
    rec = {"mode": mode, "seed": seed, "model": "gpt-6.2-astra", "effort": effort,
           "slowdown": slowdown, "profile": profile}
    try:
        if mode == "standard":
            game, decisions = run_standard(seed, effort, None)
            rec["decisions"] = decisions
        elif mode == "realtime":
            game, decisions, overruns = run_realtime(seed, effort, None, slowdown, profile)
            rec["decisions"], rec["tick_overruns"] = decisions, overruns
        else:
            game, turns, plan, submitted = run_privileged(seed, effort, None)
            rec.update(turns=turns, plan=plan, submitted=submitted)
        rec.update(game.result())
        rec["applied"] = [[int(k), a] for k, a in game.actions]   # for replay / video
    except Exception as e:
        rec["fatal"] = traceback.format_exc()[-1500:]
    rec["wall_s"] = round(time.time() - t0, 1)
    return rec


def _worker(args):
    return episode(*args)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("privileged", "standard", "realtime"), required=True)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--start", type=int, default=SEED0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--slowdown", type=float, default=1.0,
                    help="realtime only: wall seconds per simulated second")
    ap.add_argument("--profile", choices=("full", "fast"), default="full",
                    help="realtime only: sensor payload")
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
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, maxtasksperchild=1) as pool, out.open("a") as f:
        for rec in pool.imap_unordered(_worker, [(args.mode, s, args.effort, args.slowdown, args.profile) for s in todo]):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            status = rec.get("fatal", "").splitlines()[-1] if "fatal" in rec else \
                f"{rec['caught']}/{rec['drops']}"
            print(f"seed {rec['seed']}: {status}  ({rec['wall_s']} s)", flush=True)


if __name__ == "__main__":
    main()
