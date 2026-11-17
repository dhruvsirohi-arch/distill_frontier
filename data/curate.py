"""Curate raw expert episodes into a high-quality, diverse SFT dataset.

Reads raw JSONL (never modifies it), writes curated train/val JSONL + episodes.json, and
outputs/curation_report.md with plots under outputs/curation/. Rerunnable; deterministic.

Pipeline:
  1. episode-level quality filters (drop whole episode on any failure, reasons recorded)
  2. single quality score (weights in configs/curation.score_weights)
  3. stratified selection: bins = bowl start xy cell (grid_xy) x yaw bin; up to K per bin, spread
     across placement cells and variation params; near-duplicate removal within a bin
  4. thin bins (< min_per_bin) -> targeted top-up expert episodes for that bin, then re-curate
  5. episode-level train/val split stratified across bins; step-level phase/action stats and
     loss weights (upweight close_gripper/done via weights, never by duplication)

  .venv/bin/python -m data.curate
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

from env.common import REPO, get_logger, load_config, write_json

PAN_AXIS_XY = np.array([0.0388353, 0.0])
PHASES = ["approach", "pre-grasp", "grasp", "lift", "transport", "place", "release", "retreat", "done"]


# ---------------------------------------------------------------------------------------- loading
def load_episodes(paths):
    eps = collections.OrderedDict()
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                e = eps.setdefault(r["episode_id"], {k: r[k] for k in (
                    "episode_id", "seed", "final_success", "failure_type", "n_steps", "variation",
                    "spawn", "metrics", "success_detail", "video")} | {"steps": [], "source": p.name})
                e["steps"].append({k: r[k] for k in ("step", "observation", "history", "action", "result")})
    for e in eps.values():
        e["steps"].sort(key=lambda s: s["step"])
    return list(eps.values())


# ---------------------------------------------------------------------------------------- binning
def xy_cell(x, y, cfg):
    sp, (gx, gy) = cfg["env"]["spawn"], cfg["curation"]["grid_xy"]
    ix = int(np.clip((x - sp["x"][0]) / (sp["x"][1] - sp["x"][0]) * gx, 0, gx - 1))
    iy = int(np.clip((y - sp["y"][0]) / (sp["y"][1] - sp["y"][0]) * gy, 0, gy - 1))
    return ix, iy


def yaw_bin(yaw_deg, cfg):
    nb = cfg["curation"]["yaw_bins"]
    return int(np.clip((yaw_deg + 180.0) / 360.0 * nb, 0, nb - 1))


def place_cell(x, y, cfg):
    tg, (px, py) = cfg["env"]["target"], cfg["curation"]["place_grid"]
    ix = int(np.clip((x - (tg["center"][0] - tg["size"][0] / 2)) / tg["size"][0] * px, 0, px - 1))
    iy = int(np.clip((y - (tg["center"][1] - tg["size"][1] / 2)) / tg["size"][1] * py, 0, py - 1))
    return ix, iy


# ---------------------------------------------------------------------------------------- step labels
def phase_of(step, cfg):
    a, o = step["action"], step["observation"]
    if a is None:
        return "invalid"
    k = a["action"]
    if k == "done":
        return "done"
    if k == "close_gripper":
        return "grasp"
    on_table = abs(o["bowl"]["z"] - o["target"]["z"]) < 0.008
    if k == "open_gripper":
        return "release" if (o["grasped"] or on_table) else "approach"
    # move_to
    tgt = np.array([a["x"], a["y"], a["z"]])
    ee = np.array([o["ee"]["x"], o["ee"]["y"], o["ee"]["z"]])
    if o["grasped"]:
        if np.linalg.norm(tgt[:2] - ee[:2]) < 0.01 and tgt[2] > ee[2] + 0.01:
            return "lift"
        if np.linalg.norm(tgt[:2] - ee[:2]) < 0.015 and tgt[2] < ee[2] - 0.01:
            return "place"
        return "transport"
    if on_table:
        return "retreat"
    rim_top = o["bowl"]["z"] + cfg["env"]["bowl"]["height"]
    return "pre-grasp" if tgt[2] < rim_top + 0.005 else "approach"


def grasp_side(e):
    for s in e["steps"]:
        if s["action"] and s["action"]["action"] == "close_gripper" and s["result"] == "success":
            o = s["observation"]
            r_ee = np.linalg.norm(np.array([o["ee"]["x"], o["ee"]["y"]]) - PAN_AXIS_XY)
            r_b = np.linalg.norm(np.array([o["bowl"]["x"], o["bowl"]["y"]]) - PAN_AXIS_XY)
            return "near" if r_ee < r_b else "far"
    return "none"


def move_targets(e):
    return [(s["action"]["x"], s["action"]["y"], s["action"]["z"]) for s in e["steps"]
            if s["action"] and s["action"]["action"] == "move_to"]


def redundant(e, cfg):
    """Repeated move_to to (nearly) the same point, or gripper open/close toggling."""
    d = cfg["curation"]["redundant_move_dist"]
    prev = None
    for s in e["steps"]:
        a = s["action"]
        if a is None:
            return True
        if prev is not None:
            if a["action"] == "move_to" and prev["action"] == "move_to":
                if math.dist((a["x"], a["y"], a["z"]), (prev["x"], prev["y"], prev["z"])) < d:
                    return True
            if {a["action"], prev["action"]} == {"open_gripper", "close_gripper"}:
                return True
            if a["action"] == prev["action"] and a["action"] != "move_to":
                return True
        prev = a
    return False


def collided_with_rim_or_table(e):
    return any(s["result"] == "collision" for s in e["steps"]) or e["metrics"]["n_collisions"] > 0


# ---------------------------------------------------------------------------------------- core
def annotate(eps, cfg):
    for e in eps:
        sp = e["spawn"]
        e["cell"] = xy_cell(sp["x"], sp["y"], cfg)
        e["yaw_bin"] = yaw_bin(sp["yaw_deg"], cfg)
        e["bin"] = f"x{e['cell'][0]}y{e['cell'][1]}_yaw{e['yaw_bin']}"
        fb = e["metrics"]["final_bowl"]
        e["place_cell"] = place_cell(fb[0], fb[1], cfg)
        e["grasp_side"] = grasp_side(e)
        v = e["variation"]
        e["var_key"] = (e["place_cell"], v["n_waypoints"], int(v["approach_tangent_offset"] >= 0), e["grasp_side"])


def quality_filter(eps, cfg):
    cc = cfg["curation"]
    succ_steps = collections.defaultdict(list)
    for e in eps:
        if e["final_success"]:
            succ_steps[e["cell"]].append(e["n_steps"])
    p90 = {c: float(np.percentile(v, cc["step_pct"])) for c, v in succ_steps.items()}
    passed, reasons = [], collections.Counter()
    for e in eps:
        r = []
        if not e["final_success"]:
            r.append(f"not_success:{e['failure_type']}")
        if e["metrics"]["n_rejected"] > 0:
            r.append("rejected_action")
        if collided_with_rim_or_table(e):
            r.append("collision")
        if e["metrics"]["n_failed_grasps"] > 0:
            r.append("failed_grasp_attempt")
        if e["n_steps"] > p90.get(e["cell"], 99):
            r.append("too_many_steps")
        if redundant(e, cfg):
            r.append("redundant_action")
        m = e["metrics"]["place_margin"]
        if not (isinstance(m, float) and m >= cc["min_place_margin"]):
            r.append("low_place_margin")
        e["drop_reasons"] = r
        if r:
            reasons[r[0].split(":")[0] if not r[0].startswith("not_success") else r[0]] += 1
        else:
            passed.append(e)
    return passed, reasons, p90


def score(passed, cfg):
    w = cfg["curation"]["score_weights"]
    if not passed:
        return
    steps = np.array([e["n_steps"] for e in passed], float)
    path = np.array([e["metrics"]["path_length"] for e in passed], float)
    marg = np.array([e["metrics"]["place_margin"] for e in passed], float)
    pre = np.array([e["variation"]["pregrasp_height"] for e in passed], float)

    def norm(x):
        return (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else np.zeros_like(x)

    s_steps = 1 - norm(steps)
    s_path = 1 - norm(path)
    s_marg = norm(marg)
    s_pre = 1 - norm(np.abs(pre - np.median(pre)))
    total = (w["steps"] * s_steps + w["path_length"] * s_path + w["place_margin"] * s_marg
             + w["pregrasp_consistency"] * s_pre) / sum(w.values())
    for e, s in zip(passed, total):
        e["score"] = float(s)


def dup_distance(a, b):
    sa, sb = a["spawn"], b["spawn"]
    dyaw = abs((sa["yaw_deg"] - sb["yaw_deg"] + 180) % 360 - 180)
    d_start = math.hypot(sa["x"] - sb["x"], sa["y"] - sb["y"]) + 0.0005 * dyaw  # 1 deg ~ 0.5 mm
    ta = [s["action"]["action"] for s in a["steps"]]
    tb = [s["action"]["action"] for s in b["steps"]]
    if ta != tb:
        return float("inf")
    ma, mb = np.array(move_targets(a)), np.array(move_targets(b))
    d_act = float(np.mean(np.linalg.norm(ma - mb, axis=1))) if len(ma) else 0.0
    return max(d_start, d_act)


def select(passed, cfg):
    """Per bin: up to K, greedy round-robin over variation groups (best score first), dedup."""
    cc = cfg["curation"]
    by_bin = collections.defaultdict(list)
    for e in passed:
        by_bin[e["bin"]].append(e)
    selected, n_dups = [], 0
    for b in sorted(by_bin):
        cands = sorted(by_bin[b], key=lambda e: -e["score"])
        picked, counts = [], collections.Counter()
        field_counts = [collections.Counter() for _ in range(4)]
        while cands and len(picked) < cc["max_per_bin"]:
            # prefer candidates whose variation values are least represented so far, then score
            def novelty(e):
                return sum(field_counts[i][v] for i, v in enumerate(e["var_key"]))

            best = min(cands, key=lambda e: (novelty(e), -e["score"]))
            cands.remove(best)
            if any(dup_distance(best, p) < cc["dup_dist"] for p in picked):
                n_dups += 1
                best["drop_reasons"] = ["near_duplicate"]
                continue
            picked.append(best)
            for i, v in enumerate(best["var_key"]):
                field_counts[i][v] += 1
        selected.extend(picked)
    return selected, by_bin, n_dups


def all_bins(cfg):
    gx, gy = cfg["curation"]["grid_xy"]
    return [f"x{i}y{j}_yaw{k}" for i in range(gx) for j in range(gy) for k in range(cfg["curation"]["yaw_bins"])]


def topup_seeds(thin_bins, cfg, used: set, start: int, per_bin: int):
    from env.so101_env import spawn_for_seed

    need = {b: per_bin for b in thin_bins}
    seeds = []
    s = start
    while any(v > 0 for v in need.values()) and s < cfg["seeds"]["eval_start"]:
        if s not in used:
            x, y, yaw = spawn_for_seed(s, cfg)
            ix, iy = xy_cell(x, y, cfg)
            b = f"x{ix}y{iy}_yaw{yaw_bin(yaw, cfg)}"
            if need.get(b, 0) > 0:
                seeds.append(s)
                need[b] -= 1
        s += 1
    return seeds, s


def split(selected, cfg):
    frac = cfg["curation"]["val_fraction"]
    rng = np.random.default_rng(cfg["seeds"]["global"])
    by_bin = collections.defaultdict(list)
    for e in selected:
        by_bin[e["bin"]].append(e)
    cum, cum_val = 0, 0
    for b in sorted(by_bin):
        es = by_bin[b]
        order = rng.permutation(len(es))
        cum += len(es)
        n_val = int(math.floor(cum * frac + 0.5)) - cum_val
        for r, i in enumerate(order):
            es[i]["split"] = "val" if r < n_val else "train"
        cum_val += max(0, min(n_val, len(es)))


# ---------------------------------------------------------------------------------------- report
SER_RAW, SER_CUR = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax, title):
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def make_plots(raw, passed, selected, cfg, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
    files = {}
    sp, tg = cfg["env"]["spawn"], cfg["env"]["target"]
    gx, gy = cfg["curation"]["grid_xy"]

    def heat(ax, xs, ys, ext, bins, title):
        h, _, _ = np.histogram2d(xs, ys, bins=bins, range=[ext[:2], ext[2:]])
        im = ax.imshow(h.T, origin="lower", extent=ext, cmap="Blues", aspect="auto", vmin=0)
        for i in range(h.shape[0]):
            for j in range(h.shape[1]):
                cx = ext[0] + (i + 0.5) * (ext[1] - ext[0]) / h.shape[0]
                cy = ext[2] + (j + 0.5) * (ext[3] - ext[2]) / h.shape[1]
                ax.text(cx, cy, int(h[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if h[i, j] > 0.6 * h.max() else INK)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        plt.colorbar(im, ax=ax, fraction=0.046)

    ext_s = [sp["x"][0], sp["x"][1], sp["y"][0], sp["y"][1]]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    heat(axs[0], [e["spawn"]["x"] for e in raw], [e["spawn"]["y"] for e in raw], ext_s, [gx, gy], "Bowl start, raw (episodes per cell)")
    heat(axs[1], [e["spawn"]["x"] for e in selected], [e["spawn"]["y"] for e in selected], ext_s, [gx, gy], "Bowl start, curated")
    fig.tight_layout()
    files["start_heatmap"] = out_dir / "start_heatmap.png"
    fig.savefig(files["start_heatmap"], dpi=110)
    plt.close(fig)

    ext_p = [tg["center"][0] - tg["size"][0] / 2, tg["center"][0] + tg["size"][0] / 2,
             tg["center"][1] - tg["size"][1] / 2, tg["center"][1] + tg["size"][1] / 2]
    raw_ok = [e for e in raw if e["final_success"]]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    heat(axs[0], [e["metrics"]["final_bowl"][0] for e in raw_ok], [e["metrics"]["final_bowl"][1] for e in raw_ok], ext_p, [5, 4], "Placement, raw successes")
    heat(axs[1], [e["metrics"]["final_bowl"][0] for e in selected], [e["metrics"]["final_bowl"][1] for e in selected], ext_p, [5, 4], "Placement, curated")
    fig.tight_layout()
    files["place_heatmap"] = out_dir / "place_heatmap.png"
    fig.savefig(files["place_heatmap"], dpi=110)
    plt.close(fig)

    # action types + phases (curated steps)
    acts = collections.Counter(s["action"]["action"] for e in selected for s in e["steps"])
    phs = collections.Counter(phase_of(s, cfg) for e in selected for s in e["steps"])
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    k = ["move_to", "open_gripper", "close_gripper", "done"]
    axs[0].bar(k, [acts[x] for x in k], color=SER_CUR, width=0.6)
    _style(axs[0], "Action types (curated steps)")
    pk = [p for p in PHASES if phs[p]]
    axs[1].bar(pk, [phs[p] for p in pk], color=SER_CUR, width=0.6)
    axs[1].tick_params(axis="x", rotation=30)
    _style(axs[1], "Phases (curated steps)")
    fig.tight_layout()
    files["actions"] = out_dir / "actions_phases.png"
    fig.savefig(files["actions"], dpi=110)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    bins = np.arange(5, 17) - 0.5
    axs[0].hist([e["n_steps"] for e in raw], bins=bins, color=SER_RAW, alpha=0.55, label="raw")
    axs[0].hist([e["n_steps"] for e in selected], bins=bins, color=SER_CUR, alpha=0.85, label="curated")
    axs[0].legend(frameon=False)
    _style(axs[0], "Steps per episode")
    axs[1].hist([e["score"] for e in passed], bins=30, color=SER_RAW, alpha=0.55, label="passed quality")
    axs[1].hist([e["score"] for e in selected], bins=30, color=SER_CUR, alpha=0.85, label="selected")
    axs[1].legend(frameon=False)
    _style(axs[1], "Quality score")
    fig.tight_layout()
    files["steps_score"] = out_dir / "steps_score.png"
    fig.savefig(files["steps_score"], dpi=110)
    plt.close(fig)

    keys = [("pregrasp_height", "pre-grasp height (m)"), ("approach_tangent_offset", "approach offset (m)"),
            ("grasp_offset_deg", "grasp offset (deg)"), ("lift_height", "lift height (m)"),
            ("place_dx", "placement dx (m)"), ("place_dy", "placement dy (m)")]
    fig, axs = plt.subplots(2, 4, figsize=(13, 5.5))
    for ax, (key, lab) in zip(axs.flat, keys):
        ax.hist([e["variation"][key] for e in raw], bins=20, color=SER_RAW, alpha=0.55, density=True, label="raw")
        ax.hist([e["variation"][key] for e in selected], bins=20, color=SER_CUR, alpha=0.85, density=True, label="curated")
        _style(ax, lab)
    ax = axs.flat[6]
    cats = ["0 wp", "1 wp", "near rim", "far rim"]
    rawc = [sum(e["variation"]["n_waypoints"] == 0 for e in raw), sum(e["variation"]["n_waypoints"] == 1 for e in raw),
            sum(e["grasp_side"] == "near" for e in raw), sum(e["grasp_side"] == "far" for e in raw)]
    curc = [sum(e["variation"]["n_waypoints"] == 0 for e in selected), sum(e["variation"]["n_waypoints"] == 1 for e in selected),
            sum(e["grasp_side"] == "near" for e in selected), sum(e["grasp_side"] == "far" for e in selected)]
    xx = np.arange(len(cats))
    ax.bar(xx - 0.18, np.array(rawc) / len(raw), 0.34, color=SER_RAW, label="raw")
    ax.bar(xx + 0.18, np.array(curc) / max(1, len(selected)), 0.34, color=SER_CUR, label="curated")
    ax.set_xticks(xx, cats)
    _style(ax, "fraction of episodes")
    ax.legend(frameon=False, fontsize=8)
    axs.flat[7].axis("off")
    axs.flat[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    files["variation"] = out_dir / "variation.png"
    fig.savefig(files["variation"], dpi=110)
    plt.close(fig)
    return files


def write_report(path, ctx, files):
    rel = lambda p: Path(p).relative_to(path.parent).as_posix()
    L = ["# Curation report", "",
         f"Raw sources: {', '.join(ctx['sources'])}. Raw data is never modified; this report is regenerated by "
         "`python -m data.curate`.", "",
         "## Counts", "",
         "| stage | episodes |", "|---|---|",
         f"| raw (incl. top-up) | {ctx['n_raw']} |",
         f"| passed quality filters | {ctx['n_passed']} |",
         f"| near-duplicates removed | {ctx['n_dups']} |",
         f"| selected (stratified) | {ctx['n_selected']} |",
         f"| → train / val episodes | {ctx['n_train']} / {ctx['n_val']} |",
         f"| → train / val observation→action pairs | {ctx['pairs_train']} / {ctx['pairs_val']} |", "",
         "### Drop reasons (first failing filter per episode)", "", "| reason | episodes |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in sorted(ctx["reasons"].items(), key=lambda kv: -kv[1])]
    L += ["", "Filters: final success; no rejected/unreachable actions; no collisions (robot–table/floor, "
          "or open gripper disturbing the bowl); no failed grasp attempts; steps ≤ p90 of successes in the "
          f"same start cell; no redundant actions (move_to within {ctx['cfg']['redundant_move_dist']*1000:.0f} mm of the "
          "previous target, gripper toggling, repeated gripper command); placement margin ≥ "
          f"{ctx['cfg']['min_place_margin']*1000:.0f} mm inside the target region.", "",
          "## Quality score", "",
          "`score = (w_steps·(1−norm(steps)) + w_path·(1−norm(ee_path)) + w_margin·norm(place_margin) + "
          "w_pre·(1−norm(|pregrasp_h − median|))) / Σw`, min-max normalized over passing episodes. "
          f"Weights: `{json.dumps(ctx['cfg']['score_weights'])}`.", "",
          "## Diversity / stratification", "",
          f"Bins = bowl start cell ({ctx['cfg']['grid_xy'][0]}×{ctx['cfg']['grid_xy'][1]} over the spawn region) × "
          f"{ctx['cfg']['yaw_bins']} yaw bins = {ctx['n_bins']} bins; up to K={ctx['cfg']['max_per_bin']} per bin. "
          "Within a bin, picks are greedy: least-represented variation values first (placement cell "
          f"{ctx['cfg']['place_grid'][0]}×{ctx['cfg']['place_grid'][1]}, waypoint count, approach side, grasp rim side), "
          f"then score. Near-duplicates (start state + action trajectory within {ctx['cfg']['dup_dist']*1000:.0f} mm) are dropped.", "",
          f"Top-up rounds run: {ctx['topup_rounds']} (episodes added: {ctx['n_topup']}).", "",
          f"Thin bins (< {ctx['cfg']['min_per_bin']} selected) after top-up: "
          f"{', '.join(ctx['thin_bins']) if ctx['thin_bins'] else 'none'}.", "",
          f"Selected per bin: min {ctx['per_bin_min']}, median {ctx['per_bin_med']}, max {ctx['per_bin_max']}.", "",
          f"![start]({rel(files['start_heatmap'])})", "", f"![place]({rel(files['place_heatmap'])})", "",
          "## Step-level balance", "",
          "| action type | steps | loss weight |", "|---|---|---|"]
    L += [f"| {k} | {v} | {ctx['cfg']['loss_weights'].get(k, 1.0)} |" for k, v in ctx["acts"].items()]
    L += ["", "| phase | steps |", "|---|---|"] + [f"| {k} | {v} |" for k, v in ctx["phases"].items()]
    L += ["", "`close_gripper` and `done` occur once per episode; they are upweighted through per-example "
          "loss weights in training, not duplicated.", "",
          f"![actions]({rel(files['actions'])})", "", f"![steps]({rel(files['steps_score'])})", "",
          "## Variation parameters", "", f"![variation]({rel(files['variation'])})", ""]
    path.write_text("\n".join(L))


# ---------------------------------------------------------------------------------------- main
def curate_once(cfg, sources):
    raw = load_episodes(sources)
    annotate(raw, cfg)
    passed, reasons, p90 = quality_filter(raw, cfg)
    score(passed, cfg)
    selected, by_bin, n_dups = select(passed, cfg)
    return raw, passed, reasons, selected, by_bin, n_dups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-topup", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    cc = cfg["curation"]
    log = get_logger("curate", cfg)
    raw_path = REPO / cfg["paths"]["raw_jsonl"]
    topup_dir = raw_path.parent
    sources = [raw_path] + sorted(topup_dir.glob("raw_topup_r*.jsonl"))
    rounds_done = len(sources) - 1
    n_topup = 0
    while True:
        raw, passed, reasons, selected, by_bin, n_dups = curate_once(cfg, sources)
        counts = collections.Counter(e["bin"] for e in selected)
        thin = [b for b in all_bins(cfg) if counts[b] < cc["min_per_bin"]]
        log.info(f"raw={len(raw)} passed={len(passed)} selected={len(selected)} thin_bins={len(thin)}")
        if not thin or args.no_topup or rounds_done >= cc["max_topup_rounds"]:
            break
        used = {e["seed"] for e in raw}
        start = cfg["seeds"]["topup_start"] + 10000 * rounds_done
        seeds, _ = topup_seeds(thin, cfg, used, start, cc["topup_per_thin_bin"])
        rounds_done += 1
        out = topup_dir / f"raw_topup_r{rounds_done}.jsonl"
        log.info(f"top-up round {rounds_done}: {len(thin)} thin bins {thin} -> {len(seeds)} targeted seeds -> {out.name}")
        subprocess.run([sys.executable, "-m", "data.generate", "--seeds", *map(str, seeds), "--out", str(out),
                        "--record-every", "0"] + (["--config", args.config] if args.config else []),
                       check=True, cwd=REPO)
        sources.append(out)
    n_topup = sum(1 for e in raw if e["source"].startswith("raw_topup"))

    if len(selected) < cc["min_episodes"]:
        log.error(f"only {len(selected)} curated episodes (< {cc['min_episodes']}). Stopping: dataset too thin.")
        raise SystemExit(2)

    split(selected, cfg)
    out_dir = REPO / cfg["paths"]["curated_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    lw = cc["loss_weights"]
    pairs = collections.Counter()
    with open(out_dir / "train.jsonl", "w") as ft, open(out_dir / "val.jsonl", "w") as fv:
        for e in selected:
            f = ft if e["split"] == "train" else fv
            for s in e["steps"]:
                a = s["action"]
                f.write(json.dumps({
                    "episode_id": e["episode_id"], "seed": e["seed"], "split": e["split"], "bin": e["bin"],
                    "step": s["step"], "observation": s["observation"], "history": s["history"], "action": a,
                    "action_type": a["action"], "phase": phase_of(s, cfg), "loss_weight": lw.get(a["action"], 1.0),
                }) + "\n")
                pairs[e["split"]] += 1
    write_json(out_dir / "episodes.json", [{k: e[k] for k in ("episode_id", "seed", "bin", "split", "score",
                                            "n_steps", "grasp_side", "spawn", "variation")} for e in selected])

    counts = collections.Counter(e["bin"] for e in selected)
    per_bin = [counts[b] for b in all_bins(cfg)]
    files = make_plots(raw, passed, selected, cfg, REPO / "outputs" / "curation")
    ctx = {
        "sources": [Path(s).name for s in sources], "n_raw": len(raw), "n_passed": len(passed), "n_dups": n_dups,
        "n_selected": len(selected), "n_train": sum(e["split"] == "train" for e in selected),
        "n_val": sum(e["split"] == "val" for e in selected), "pairs_train": pairs["train"], "pairs_val": pairs["val"],
        "reasons": dict(reasons), "cfg": cc, "n_bins": len(all_bins(cfg)), "topup_rounds": rounds_done,
        "n_topup": n_topup, "thin_bins": [b for b in all_bins(cfg) if counts[b] < cc["min_per_bin"]],
        "per_bin_min": min(per_bin), "per_bin_med": int(np.median(per_bin)), "per_bin_max": max(per_bin),
        "acts": dict(collections.Counter(s["action"]["action"] for e in selected for s in e["steps"]).most_common()),
        "phases": {p: c for p in PHASES if (c := sum(phase_of(s, cfg) == p for e in selected for s in e["steps"]))},
    }
    write_report(REPO / "outputs" / "curation_report.md", ctx, files)
    write_json(REPO / "outputs" / "curation_summary.json", {k: v for k, v in ctx.items() if k != "cfg"})
    log.info("curation summary: " + json.dumps({k: v for k, v in ctx.items() if k not in ("cfg",)}))


if __name__ == "__main__":
    main()
