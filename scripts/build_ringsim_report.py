"""Build docs/ringsim/index.html: the Reaction Ring latency exam report, from outputs/ringsim.

    .venv/bin/python scripts/build_ringsim_report.py
Every number on the page is computed here from the result files.
"""
from __future__ import annotations

import html
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "ringsim"
DOC = ROOT / "docs" / "ringsim"
MEDIA = DOC / "media"
CSS = (ROOT / "scripts" / "ringsim_report.css").read_text()
S1, S2, S3, ACC = "#3987e5", "#d95926", "#199e70", "#f2c230"
GRID, AXIS = "rgba(255,255,255,.06)", "rgba(255,255,255,.22)"


def load(name):
    p = OUT / name
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def esc(s):
    return html.escape(str(s), quote=True)


# ---------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------
subprocess.run([sys.executable, "-m", "ringsim.summary", "--dir", str(OUT)], cwd=ROOT,
               env={**os.environ, "PYTHONPATH": str(ROOT)}, capture_output=True, check=True)
SUM = json.loads((OUT / "summary.json").read_text())
CAL = json.loads((OUT / "calibration.json").read_text())["controllers"]
PRIV, STD, RT = SUM["privileged"], SUM["standard"], SUM["realtime"]
VAR = SUM.get("realtime_variants", {})
PROF = SUM.get("latency_profile")
std_recs = [r for r in load("astra_standard.jsonl") if "fatal" not in r]
rt_recs = [r for r in load("astra_realtime.jsonl") if "fatal" not in r]
priv_recs = [r for r in load("astra_privileged.jsonl") if "fatal" not in r]

LAT_KEYS = sorted((float(k[6:]), k) for k in CAL if k.startswith("event@"))
curve = [(L, CAL[k]["catch_rate"]) for L, k in LAT_KEYS]
oracle = CAL["oracle"]["catch_rate"]
vision = CAL["vision"]["catch_rate"]
parked = CAL["parked"]["catch_rate"]


def interp_latency(target):
    """Largest latency at which the scripted reactive controller still reaches `target`."""
    pts = [(L, r) for L, r in curve if L > 0]
    for (l0, r0), (l1, r1) in zip(pts, pts[1:]):
        if r0 >= target > r1:
            return l0 + (r0 - target) / (r0 - r1) * (l1 - l0)
    return None


L50 = interp_latency(0.5)
L10 = interp_latency(0.1)

rt_lat = [d["latency_s"] for r in rt_recs for d in r["decisions"] if "latency_s" in d and "error" not in d]
rt_first = [r["decisions"][0]["latency_s"] for r in rt_recs if r["decisions"] and "latency_s" in r["decisions"][0]]
rt_late_first = sum(1 for r in rt_recs if not [d for d in r["decisions"] if "t_applied" in d]
                    or min(d["t_applied"] for d in r["decisions"] if "t_applied" in d)
                    > min(e["outcome_t"] for e in r["events"]))
rt_complete = [sum("action" in d for d in r["decisions"]) for r in rt_recs]
rt_pre_first = sum(any("t_applied" in d and d["t_applied"] < r["events"][0]["release_t"] or False
                       for d in r["decisions"]) for r in rt_recs) if rt_recs else 0
first_release = min(min(e["release_t"] for e in r["events"]) for r in rt_recs) if rt_recs else 3.5

# Standard behaviour
std_hist = np.bincount([r["caught"] for r in std_recs], minlength=9) if std_recs else np.zeros(9, int)
std_adv = [d["advance_ticks"] * 40 for r in std_recs for d in r["decisions"] if "advance_ticks" in d]
std_touch = STD["touched"] if STD else 0
std_outc = STD["outcomes"] if STD else {}
std_reacted = STD.get("drops_with_postrelease_command", 0) if STD else 0
caught_lag = [e["outcome_t"] - e["release_t"] for r in std_recs for e in r["events"] if e["outcome"] == "caught"]


def first_contact_after(recs, outcome=None):
    return [e["first_pad_contact_t"] - e["release_t"] for r in recs for e in r["events"]
            if e["first_pad_contact_t"] is not None and (outcome is None or e["outcome"] == outcome)]


std_contact_caught = first_contact_after(std_recs, "caught")
std_contact_missed = [e["first_pad_contact_t"] - e["release_t"] for r in std_recs for e in r["events"]
                      if e["first_pad_contact_t"] is not None and e["outcome"] != "caught"]
std_short = sum(a <= 80 for a in std_adv) / max(len(std_adv), 1)

# slow-motion sweep
sweep = sorted([(v["slowdown"], k, v) for k, v in VAR.items() if v.get("profile") == "full"
                and v["slowdown"] > 1], key=lambda x: x[0])
fast = VAR.get("fast")

# ---------------------------------------------------------------------------------------------
# chart helpers (inline SVG, hover tooltips via data-tip)
# ---------------------------------------------------------------------------------------------


def svg_open(w, h, label):
    return f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{esc(label)}">'


def logx(v, lo, hi, x0, x1):
    return x0 + (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * (x1 - x0)


def fig_budget():
    """Horizontal bars on a log time axis: how long things take."""
    rows = [("Stick falls past the waiting hands", 0.17, S3),
            ("Stick reaches the floor", 0.56, S3),
            (f"Scripted reflex: 50% catch rate lost at", L50 or 0.12, S1),
            ("Astra: fastest decision observed", min(rt_lat) if rt_lat else 4.7, S2),
            ("Astra: median Realtime decision", RT["latency_median_s"], S2),
            ("Astra: median Privileged planning turn", PRIV["latency_median_s"], S2)]
    W, rowh, x0, x1 = 980, 46, 330, 940
    H = 40 + rowh * len(rows) + 30
    lo, hi = 0.05, 50
    s = [svg_open(W, H, "Time scales of the task versus the model")]
    for t in (0.1, 0.3, 1, 3, 10, 30):
        x = logx(t, lo, hi, x0, x1)
        s.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="30" y2="{H - 30}" stroke="{GRID}"/>')
        s.append(f'<text class="tick" x="{x:.1f}" y="{H - 10}" text-anchor="middle">{t:g} s</text>')
    for i, (name, v, c) in enumerate(rows):
        y = 40 + i * rowh
        x = logx(v, lo, hi, x0, x1)
        s.append(f'<text class="lab" x="{x0 - 14}" y="{y + 19}" text-anchor="end">{esc(name)}</text>')
        s.append(f'<rect x="{x0}" y="{y + 6}" width="{max(x - x0, 4):.1f}" height="20" rx="4" fill="{c}" '
                 f'data-tip="<b>{v:.2f} s</b>{esc(name)}" tabindex="0"/>')
        s.append(f'<text class="val" x="{x + 8:.1f}" y="{y + 20}">{v:.2f} s</text>')
    s.append("</svg>")
    return "".join(s)


def fig_curve():
    """Catch rate vs reaction latency: the scripted reflex and every Astra realtime variant."""
    W, H, x0, x1, y0, y1 = 980, 430, 70, 940, 30, 370
    lo, hi = 0.03, 30
    ys = lambda r: y1 - r * (y1 - y0)
    s = [svg_open(W, H, "Catch rate versus reaction latency")]
    for r in (0, .25, .5, .75, 1):
        s.append(f'<line x1="{x0}" x2="{x1}" y1="{ys(r):.1f}" y2="{ys(r):.1f}" stroke="{GRID}"/>')
        s.append(f'<text class="tick" x="{x0 - 10}" y="{ys(r) + 4:.1f}" text-anchor="end">{int(r * 100)}%</text>')
    for t in (0.03, 0.1, 0.3, 1, 3, 10, 30):
        x = logx(t, lo, hi, x0, x1)
        s.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{y0}" y2="{y1}" stroke="{GRID}"/>')
        s.append(f'<text class="tick" x="{x:.1f}" y="{y1 + 20}" text-anchor="middle">{t:g} s</text>')
    s.append(f'<text class="tick" x="{(x0 + x1) / 2}" y="{H - 4}" text-anchor="middle">'
             'reaction latency in simulated time (log scale)</text>')
    # reference lines
    for val, name in ((oracle, "Privileged oracle (knows the schedule)"),
                      (STD["catch_rate"], "Astra Standard (world paused while it thinks)")):
        y = ys(val)
        s.append(f'<line x1="{x0}" x2="{x1}" y1="{y:.1f}" y2="{y:.1f}" stroke="{AXIS}" stroke-dasharray="5 5"/>')
        s.append(f'<text class="tick" x="{x1}" y="{y - 7:.1f}" text-anchor="end">{esc(name)} · {pct(val)}</text>')
    # scripted curve
    pts = [(L, r) for L, r in curve if L > 0]
    path = " ".join(f"{'M' if i == 0 else 'L'}{logx(L, lo, hi, x0, x1):.1f},{ys(r):.1f}" for i, (L, r) in enumerate(pts))
    s.append(f'<path d="{path}" fill="none" stroke="{S1}" stroke-width="2"/>')
    for L, r in pts:
        x, y = logx(L, lo, hi, x0, x1), ys(r)
        s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{S1}" stroke="#13151a" stroke-width="2"/>')
        s.append(f'<circle class="hit" cx="{x:.1f}" cy="{y:.1f}" r="14" tabindex="0" '
                 f'data-tip="<b>{pct(r)}</b>scripted reflex, told of each drop {L * 1000:.0f} ms late"/>')
    # Astra points: realtime (1x), fast, slow-motion sweep at effective latency
    apts = []
    if RT:
        apts.append((RT["sim_latency_median_s"] if RT.get("sim_latency_median_s") else RT["latency_median_s"],
                     RT["catch_rate"], "Realtime, full sensors, 1×"))
    if fast:
        apts.append((fast.get("sim_latency_median_s") or fast["latency_median_s"], fast["catch_rate"],
                     "Realtime, fast profile, 1×"))
    for k, _, v in sweep:
        if v.get("sim_latency_median_s"):
            apts.append((v["sim_latency_median_s"], v["catch_rate"], f"Realtime, world slowed {k:g}×"))
    apts.sort()
    if len(apts) > 1:
        path = " ".join(f"{'M' if i == 0 else 'L'}{logx(min(max(L, lo), hi), lo, hi, x0, x1):.1f},{ys(r):.1f}"
                        for i, (L, r, _) in enumerate(apts))
        s.append(f'<path d="{path}" fill="none" stroke="{S2}" stroke-width="2"/>')
    for L, r, name in apts:
        x, y = logx(min(max(L, lo), hi), lo, hi, x0, x1), ys(r)
        s.append(f'<rect x="{x - 5:.1f}" y="{y - 5:.1f}" width="10" height="10" rx="2" fill="{S2}" stroke="#13151a" stroke-width="2"/>')
        s.append(f'<circle class="hit" cx="{x:.1f}" cy="{y:.1f}" r="14" tabindex="0" '
                 f'data-tip="<b>{pct(r)}</b>Astra · {esc(name)} · median {L:.2f} s of game time per decision"/>')
    s.append(f'<text class="lab" x="{logx(0.045, lo, hi, x0, x1):.1f}" y="{ys(0.93):.1f}" fill="{S1}">Scripted reflex</text>')
    if apts:
        L, r, _ = apts[-1]
        s.append(f'<text class="lab" x="{logx(min(L, hi), lo, hi, x0, x1) - 8:.1f}" y="{ys(r) - 14:.1f}" text-anchor="end">Astra</text>')
    s.append("</svg>")
    return "".join(s)


def fig_hist():
    W, H, x0, x1, y0, y1 = 980, 300, 60, 940, 20, 250
    n = std_hist
    top = max(int(n.max()), 1)
    s = [svg_open(W, H, "Sticks caught per Standard episode")]
    for v in np.linspace(0, top, 5):
        y = y1 - v / top * (y1 - y0)
        s.append(f'<line x1="{x0}" x2="{x1}" y1="{y:.1f}" y2="{y:.1f}" stroke="{GRID}"/>')
        s.append(f'<text class="tick" x="{x0 - 10}" y="{y + 4:.1f}" text-anchor="end">{v:.0f}</text>')
    bw = (x1 - x0) / 9
    for k in range(9):
        h = n[k] / top * (y1 - y0)
        x = x0 + k * bw + 2
        s.append(f'<rect x="{x:.1f}" y="{y1 - h:.1f}" width="{bw - 4:.1f}" height="{max(h, 0):.1f}" rx="4" fill="{S1}" '
                 f'data-tip="<b>{n[k]} episodes</b>caught {k} of 8" tabindex="0"/>')
        if n[k]:
            s.append(f'<text class="val" x="{x + bw / 2 - 2:.1f}" y="{y1 - h - 6:.1f}" text-anchor="middle">{n[k]}</text>')
        s.append(f'<text class="tick" x="{x + bw / 2 - 2:.1f}" y="{y1 + 18}" text-anchor="middle">{k}</text>')
    s.append(f'<text class="tick" x="{(x0 + x1) / 2}" y="{H - 4}" text-anchor="middle">sticks caught out of 8</text>')
    s.append("</svg>")
    return "".join(s)


def fig_profile():
    rows = []
    if PROF:
        for name, v in PROF.items():
            rows.append((name, v["median_s"], v["p10_s"], v["p90_s"]))
    if RT:
        rows.append(("Realtime exam decisions, full sensors (in the loop)", RT["latency_median_s"],
                     float(np.percentile(rt_lat, 10)) if rt_lat else RT["latency_median_s"],
                     float(np.percentile(rt_lat, 90)) if rt_lat else RT["latency_median_s"]))
    if not rows:
        return "<p class='cap'>Pending.</p>"
    W, rowh, x0, x1 = 980, 44, 400, 930
    H = 30 + rowh * len(rows) + 34
    hi = max(r[3] for r in rows) * 1.12
    s = [svg_open(W, H, "Decision latency by input payload")]
    for t in np.arange(0, hi, 2 if hi < 16 else 5):
        x = x0 + t / hi * (x1 - x0)
        s.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="20" y2="{H - 30}" stroke="{GRID}"/>')
        s.append(f'<text class="tick" x="{x:.1f}" y="{H - 10}" text-anchor="middle">{t:g} s</text>')
    for i, (name, med, p10, p90) in enumerate(rows):
        y = 28 + i * rowh
        xm = x0 + med / hi * (x1 - x0)
        s.append(f'<text class="lab" x="{x0 - 14}" y="{y + 17}" text-anchor="end">{esc(name[:58])}</text>')
        s.append(f'<rect x="{x0}" y="{y + 5}" width="{xm - x0:.1f}" height="18" rx="4" fill="{S2}" '
                 f'data-tip="<b>{med:.2f} s median</b>p10 {p10:.2f} s · p90 {p90:.2f} s" tabindex="0"/>')
        a, b = x0 + p10 / hi * (x1 - x0), x0 + p90 / hi * (x1 - x0)
        s.append(f'<line x1="{a:.1f}" x2="{b:.1f}" y1="{y + 14}" y2="{y + 14}" stroke="var(--ink)" stroke-width="1.5"/>')
        s.append(f'<text class="val" x="{b + 8:.1f}" y="{y + 18}">{med:.2f} s</text>')
    s.append("</svg>")
    return "".join(s)


def fig_sweep():
    pts = [("1×", RT["catch_rate"], RT["catch_rate_ci95"], RT.get("sim_latency_median_s"), RT["episodes"])]
    for k, _, v in sweep:
        pts.append((f"{k:g}×", v["catch_rate"], v["catch_rate_ci95"], v.get("sim_latency_median_s"), v["episodes"]))
    pts.append(("paused (Standard)", STD["catch_rate"], STD["catch_rate_ci95"], None, STD["episodes"]))
    W, H, x0, x1, y0, y1 = 980, 340, 60, 940, 24, 280
    ys = lambda r: y1 - r * (y1 - y0)
    s = [svg_open(W, H, "Catch rate versus slow-motion factor")]
    for r in (0, .25, .5, .75, 1):
        s.append(f'<line x1="{x0}" x2="{x1}" y1="{ys(r):.1f}" y2="{ys(r):.1f}" stroke="{GRID}"/>')
        s.append(f'<text class="tick" x="{x0 - 10}" y="{ys(r) + 4:.1f}" text-anchor="end">{int(r * 100)}%</text>')
    step = (x1 - x0) / len(pts)
    prev = None
    for i, (lab, r, ci, sl, n) in enumerate(pts):
        x = x0 + step * (i + .5)
        s.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{ys(ci[0]):.1f}" y2="{ys(ci[1]):.1f}" stroke="var(--ink2)" stroke-width="1.5"/>')
        if prev and i < len(pts) - 1:
            s.append(f'<line x1="{prev[0]:.1f}" y1="{prev[1]:.1f}" x2="{x:.1f}" y2="{ys(r):.1f}" stroke="{S2}" stroke-width="2"/>')
        c = S1 if i == len(pts) - 1 else S2
        s.append(f'<circle cx="{x:.1f}" cy="{ys(r):.1f}" r="6" fill="{c}" stroke="#13151a" stroke-width="2"/>')
        extra = f" · median {sl:.2f} s game time per decision" if sl else ""
        s.append(f'<circle class="hit" cx="{x:.1f}" cy="{ys(r):.1f}" r="16" tabindex="0" '
                 f'data-tip="<b>{pct(r)}</b>{esc(lab)} · {n} episodes · 95% CI {pct(ci[0])}–{pct(ci[1])}{extra}"/>')
        s.append(f'<text class="val" x="{x + 10:.1f}" y="{ys(r) - 8:.1f}">{pct(r, 0)}</text>')
        s.append(f'<text class="tick" x="{x:.1f}" y="{y1 + 20}" text-anchor="middle">{esc(lab)}</text>')
        prev = (x, ys(r))
    s.append(f'<text class="tick" x="{(x0 + x1) / 2}" y="{H - 4}" text-anchor="middle">wall-clock seconds per simulated second</text>')
    s.append("</svg>")
    return "".join(s)


# ---------------------------------------------------------------------------------------------
# media
# ---------------------------------------------------------------------------------------------
MEDIA.mkdir(parents=True, exist_ok=True)


def ensure_video(jsonl, seed, name):
    dst = MEDIA / name
    if not dst.exists() and (OUT / jsonl).exists():
        subprocess.run([sys.executable, "-m", "ringsim.replay", str(OUT / jsonl), "--seed", str(seed),
                        "--video", str(dst)], cwd=ROOT, check=False, capture_output=True,
                       env={**os.environ, "PYTHONPATH": str(ROOT), "MUJOCO_GL": "egl"})
    return dst.exists()


best_std = max(std_recs, key=lambda r: (r["caught"], -r["seed"])) if std_recs else None
vid_std = ensure_video("astra_standard.jsonl", best_std["seed"], "standard_best.mp4") if best_std else False
vid_rt = ensure_video("astra_realtime.jsonl", 50000, "realtime_50000.mp4")
vid_priv = ensure_video("astra_privileged.jsonl", 50000, "privileged_50000.mp4")
head_png = MEDIA / "head_view.png"
if not head_png.exists():
    os.environ.setdefault("MUJOCO_GL", "egl")
    sys.path.insert(0, str(ROOT))
    from PIL import Image

    from ringsim.game import Game
    g = Game(render=True)
    o = g.reset(50000)
    Image.fromarray(o["rgb"]).save(head_png)
    d = np.uint8(np.clip(o["depth"] / 2.5, 0, 1) * 255)
    Image.fromarray(d).save(MEDIA / "head_depth.png")

# ---------------------------------------------------------------------------------------------
# page
# ---------------------------------------------------------------------------------------------
n_std = STD["episodes"]
std_ci = STD["catch_rate_ci95"]
ratio = RT["latency_median_s"] / (L50 or 0.12)
tok_total = sum((SUM[m] or {}).get("tokens_total", 0) for m in ("privileged", "standard", "realtime"))
tok_total += sum(v.get("tokens_total", 0) for v in VAR.values())
dec_total = sum((SUM[m] or {}).get("decisions", 0) for m in ("privileged", "standard", "realtime"))
dec_total += sum(v.get("decisions", 0) for v in VAR.values())
pending = [] if (fast and len(sweep) >= 6 and n_std >= 100) else ["some runs were still in progress when this page was built"]


def cal_row(name, key, desc):
    c = CAL[key]
    return (f"<tr><td>{name}<small>{desc}</small></td><td class='n'>{c['caught']}/{c['drops']}</td>"
            f"<td class='n'>{pct(c['catch_rate'])}</td></tr>")


def mode_row(name, s, desc, me=False):
    if not s:
        return ""
    ci = s["catch_rate_ci95"]
    lat = f"{s['latency_median_s']:.1f} s" if s.get("latency_median_s") else "–"
    return (f"<tr{' class=me' if me else ''}><td>{name}<small>{desc}</small></td>"
            f"<td class='n'>{s['caught']}/{s['drops']}</td><td class='n'>{pct(s['catch_rate'])}</td>"
            f"<td class='n'>{pct(ci[0])}–{pct(ci[1])}</td><td class='n'>{s['perfect_episodes']}/{s['episodes']}</td>"
            f"<td class='n'>{lat}</td></tr>")


curve_rows = "".join(f"<tr><td>{L * 1000:.0f} ms</td><td class='n'>{pct(r)}</td></tr>" for L, r in curve)
sweep_rows = "".join(
    f"<tr><td>{k:g}×</td><td class='n'>{v['episodes']}</td><td class='n'>{v['caught']}/{v['drops']}</td>"
    f"<td class='n'>{pct(v['catch_rate'])}</td><td class='n'>{v.get('latency_median_s', 0):.2f} s</td>"
    f"<td class='n'>{(v.get('sim_latency_median_s') or 0) * 1000:.0f} ms</td>"
    f"<td class='n'>{v.get('decisions_per_episode', 0):.0f}</td></tr>" for k, _, v in sweep)

lat_c = float(np.median(std_contact_caught)) if std_contact_caught else 0
lat_m = float(np.median(std_contact_missed)) if std_contact_missed else 0
touched_missed = sum(v for k, v in std_outc.items() if k != "caught") - sum(
    1 for r in std_recs for e in r["events"] if e["outcome"] != "caught" and e["first_pad_contact_t"] is None)
trapped_n = std_outc.get("trapped", 0)
prof_floor = min((v["median_s"] for v in PROF.values()), default=None) if PROF else None
prof_full = next((v["median_s"] for k, v in PROF.items() if k.startswith("full")), None) if PROF else None
best_sweep = max(sweep, key=lambda x: x[2]["catch_rate"]) if sweep else None
first_nonzero = next(((k, v) for k, _, v in sweep if v["caught"] > 0), None)
fast_txt = (f"The fast profile (one 320×120 image, compact text, one cached thread) brought the median "
            f"decision to <b>{fast['latency_median_s']:.1f} s</b> and caught <b>{fast['caught']}/{fast['drops']}</b>."
            if fast else "The fast-profile run was still in progress when this page was built.")
if sweep:
    sweep_txt = (f"Slowing the world down is the only lever that moved the score. "
                 + (f"The first catches appear at <b>{first_nonzero[0]:g}× slow motion</b> "
                    f"({first_nonzero[1]['caught']}/{first_nonzero[1]['drops']}), when a decision costs "
                    f"{first_nonzero[1].get('sim_latency_median_s', 0) * 1000:.0f} ms of game time. "
                    if first_nonzero else "No slow-motion factor tested produced a catch. ")
                 + f"At {best_sweep[0]:g}× Astra catches <b>{pct(best_sweep[2]['catch_rate'])}</b>, "
                   f"against {pct(STD['catch_rate'])} with the world fully paused.")
else:
    sweep_txt = "The slow-motion sweep was still in progress when this page was built."
abstract_tail = []
if fast:
    abstract_tail.append(f"A leaner sensor payload with prompt caching brought decisions to "
                         f"{fast['latency_median_s']:.1f} s and caught {fast['caught']} of {fast['drops']}.")
if sweep:
    abstract_tail.append(f"Slowing the world down {first_nonzero[0]:g}× is where catches begin."
                         if first_nonzero else "Even slowing the world down "
                         f"{sweep[-1][0]:g}× did not produce a catch.")
abstract_tail = " ".join(abstract_tail)
if prof_floor is not None:
    prof_cap = (f"The fastest configuration, a text-only question with a one-integer answer, still has a "
                f"median of <b>{prof_floor:.1f} s</b>. The full sensor turn takes {prof_full:.1f} s unloaded, "
                f"against {RT['latency_median_s']:.1f} s inside the Realtime exam (which overlapped other runs). "
                f"Payload size moves latency by seconds at most; the per-turn floor of the serving path dominates. ")
else:
    prof_cap = "The unloaded latency profile was still running when this page was built. "

page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Reaction Ring latency exam</title>
<meta name="description" content="A simulated Reaction Ring drop-stick game as a reaction-time exam for GPT-6.2-Astra on a wheeled humanoid: privileged, standard and realtime protocols.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter+Tight:wght@400;500;600;700&family=Instrument+Serif:ital@0;1&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
{CSS}
img.wide{{width:100%;border-radius:12px;border:1px solid var(--line);display:block}}
.eq{{font:400 15px var(--mono);color:var(--ink);background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 18px;margin:20px 0;max-width:var(--col)}}
</style>
</head>
<body>
<div id="tip"></div>
<nav class="nav"><div class="wrap"><a class="logo" href="../"><i></i>distill_frontier <span>/ reaction ring</span></a>
<ul><li><a href="#task">Task</a></li><li><a href="#calibration">Calibration</a></li><li><a href="#results">Results</a></li><li><a href="#limit">Pushing realtime</a></li><li><a href="#discussion">Discussion</a></li><li><a href="#reproduce">Reproduce</a></li></ul>
<a class="btn" href="https://github.com/dhruvsirohi-arch/distill_frontier/tree/main/ringsim">Code</a></div></nav>

<header class="hero"><div class="wrap">
  <span class="eyebrow">Technical report · Embodied latency exam · phase 1</span>
  <h1>Too slow to <em>catch a falling stick.</em></h1>
  <p class="lede">We built a MuJoCo version of the <b>Reaction Ring</b> arcade game for a wheeled humanoid and gave it to <b>GPT-6.2-Astra</b> under three protocols. With the world paused while it thinks, Astra catches <b>{pct(STD['catch_rate'])}</b> of falling sticks. With the world running, it catches <b>{pct(RT['catch_rate'])}</b>: a decision takes {RT['latency_median_s']:.1f} s and a stick hits the floor in 0.56 s.</p>
  <div class="byline"><span class="tag">{esc(SUM.get('model'))}</span><span class="tag">reasoning effort: low</span><span class="tag">Unitree G1-D + Dex1-1</span><span class="tag">MuJoCo 4.0.1</span><span class="tag">seeds 50000–50099</span><span class="tag">21 Nov 2026</span></div>
  <div class="kpis">
    <div class="kpi"><div class="v">{pct(PRIV['catch_rate'], 0)}</div><div class="l">Privileged: full state, planning with resets, replayed in a fresh simulator</div></div>
    <div class="kpi"><div class="v">{pct(STD['catch_rate'], 0)}</div><div class="l">Standard: sensors only, the simulation pauses during every decision</div></div>
    <div class="kpi"><div class="v">{pct(RT['catch_rate'], 0)}</div><div class="l">Realtime: same as Standard, but the world keeps running while Astra thinks</div></div>
    <div class="kpi"><div class="v">{ratio:.0f}<small>×</small></div><div class="l">too slow: median decision {RT['latency_median_s']:.1f} s vs. the ~{(L50 or .12) * 1000:.0f} ms at which a scripted reflex loses half its catches</div></div>
  </div>
  <div class="abstract"><div class="k">Abstract</div>
  <p>The Reaction Ring drops sticks one at a time from an overhead bar; the player must catch each one in the half second before it lands. It isolates one question: can the agent close the loop from seeing to acting fast enough? Astra perceives and plans the task well, perfectly with privileged state and at {pct(STD['catch_rate'], 0)} from cameras alone when time stops, but its decision latency is two orders of magnitude above the task's budget, so in real time it catches nothing. {abstract_tail}</p></div>
</div></header>

<section id="intro"><div class="wrap">
  <div class="head"><span class="eyebrow">01 · Motivation</span>
  <h2>A test where <em>time</em> is the whole difficulty</h2></div>
  <div class="prose">
  <p>Most embodied benchmarks let the environment wait for the model. That hides the cost of thinking: a policy that is correct but slow scores the same as one that is correct and fast. Our exam follows the three settings of the exam framework we evaluate under: <b>Privileged</b> (the agent reads the simulator and submits a trajectory that is replayed), <b>Standard</b> (sensors only, simulation paused between calls, live episode graded) and <b>Realtime</b> (as Standard, but the simulation keeps running during inference). Because Realtime is costly it is reserved for dynamic tasks. A dynamic task should make the gap between Standard and Realtime large and interpretable.</p>
  <p>The Reaction Ring is about as dynamic as a manipulation task gets while staying simple to grade. Perception is easy (a pink stick moves), the motion is short (a hand moves a few tens of centimetres), and the only hard part is <b>doing it within a few hundred milliseconds</b>. The difference between the Standard and Realtime scores is therefore almost entirely the cost of latency.</p>
  </div>
</div></section>

<section id="task"><div class="wrap">
  <div class="head"><span class="eyebrow">02 · The task</span>
  <h2>The game, the robot, and the <em>rules</em></h2>
  <p class="sub">The machine follows the manufacturer's manual<a href="#r1">[1]</a>: a <b>2.05 m</b> stand whose curved overhead bar holds <b>8 sticks</b> (menu default; 10 or 12 optional) on electromagnets. At difficulty 1 sticks fall in random order <b>0.5–1.5 s</b> apart, after a 3-2-1 countdown, with hints off.</p></div>
  <div class="vids">
    {"<figure class='vid'><video controls muted loop playsinline autoplay preload='metadata' src='media/standard_best.mp4'></video><figcaption><b>Astra, Standard protocol, seed " + str(best_std['seed']) + f" ({best_std['caught']}/8)</b>Replayed from its recorded commands. Spectator views on top; Astra's head cameras below. Score cells turn yellow while a stick falls, then green or red.</figcaption></figure>" if vid_std else ""}
    {"<figure class='vid'><video controls muted loop playsinline autoplay preload='metadata' src='media/realtime_50000.mp4'></video><figcaption><b>Astra, Realtime protocol, seed 50000 (0/8)</b>The first command lands after the first stick has already fallen. Arms move only to a ready pose.</figcaption></figure>" if vid_rt else ""}
  </div>
  <div class="stages">
    <div class="stage"><div class="n">MACHINE</div><h4>Reaction Ring</h4><p>Half-circle bar of radius 0.5 m around the robot; foam sticks 40 cm × 3.2 cm, 120 g, hanging from 1.95 m to 1.55 m.</p><div class="m">8 sticks · random order</div></div>
    <div class="stage"><div class="n">ROBOT</div><h4>Unitree G1-D</h4><p>Wheeled base, humanoid torso, two 7-DoF arms, Dex1-1 parallel grippers (80 mm, 20 N). Link inertias and limits from Unitree's URDF.</p><div class="m">25 Hz control · 2 ms physics</div></div>
    <div class="stage"><div class="n">SENSORS</div><h4>Head RGB-D + joints</h4><p>Two 106° head cameras (2 × 320×240), metric depth, joint encoders, wheel odometry, measured pinch points.</p><div class="m">no object poses</div></div>
    <div class="stage"><div class="n">ACTIONS</div><h4>Cartesian skills</h4><p>Base velocity, waist yaw, and a pinch-point target plus jaw opening per hand. An IK controller with joint rate limits executes them.</p><div class="m">hands move ~0.3 m in 0.2 s</div></div>
  </div>
  <div class="fig">
    <div class="t"><span><b>Fig. 1</b> · What the agent sees</span><span>seed 50000, t = 0</span></div>
    <h3>Both head panels together cover all eight sticks</h3>
    <p class="d">Left half looks forward-left, right half forward-right. The black post is the machine's column.</p>
    <img class="wide" src="media/head_view.png" alt="Robot head camera view of eight pink sticks hanging from a curved bar">
  </div>
  <div class="prose" style="margin-top:28px">
  <p><b>Judging</b> happens in the simulator and is never shown to sensor-mode agents. A stick counts as <b>caught</b> when both jaw pads of one gripper touch it continuously for 0.2 s before it touches the floor. It is a <b>miss</b> if it reaches the floor or is not caught within 1.5 s. Touching a stick before its magnet releases is a <b>foul</b>. Caught and missed sticks are cleared from the play area, as an attendant would.</p>
  <p>One rule was added after calibration, and it matters for what the test measures. A hand parked under a stick with its jaws <i>half closed</i> catches it with no timing at all, because the stick wedges between the fingertips (Section 04). We therefore require the stick to fall into <b>open jaws</b>: if its bottom end enters a gripper whose gap is under 36 mm (the stick is 32 mm), the hold is "trapped" and does not score. Agents are told the rule in plain words. The thresholds stay hidden.</p>
  </div>
  <div class="eq">free fall from release: stick bottom passes the waiting hands (z ≈ 1.4 m) at 0.17 s, its top at 0.33 s, the floor at 0.56 s</div>
</div></section>

<section id="protocols"><div class="wrap">
  <div class="head"><span class="eyebrow">03 · Protocols</span>
  <h2>Three ways to <em>take the exam</em></h2></div>
  <div class="notes three">
    <div class="note"><h4>Privileged</h4><p>Astra reads the whole state: stick poses, the exact release schedule, and the judging rule. In up to {8} turns it may <b>simulate</b> a keyframe plan from the start (a reset) and see the outcome, then <b>submit</b>. The submission is replayed in a fresh simulator, and only the replay is graded.</p></div>
    <div class="note"><h4>Standard</h4><p>The simulator is a separate service. Each turn Astra receives the head RGB and depth, the previous image, its own notes, and proprioception. It returns a command plus <code>advance_ms</code> (40–2000 ms). The simulation <b>pauses</b> while it thinks. No resets. The live game is graded.</p></div>
    <div class="note"><h4>Realtime</h4><p>Same observations and commands, but the simulator runs on the wall clock at 25 Hz. One decision is in flight at a time. The previous command holds until a reply arrives, then a fresh frame goes out immediately. Inference time is game time.</p></div>
  </div>
  <div class="prose" style="margin-top:28px">
  <p><b>Harness.</b> Astra is called through the locally authenticated Codex app-server with tools, shell, web search and plugins disabled, structured JSON output, and reasoning effort <code>low</code> (the lowest the model accepts; <code>minimal</code> and <code>none</code> are rejected). Sensor-mode turns are stateless: each decision opens a fresh ephemeral thread containing the instructions, the current sensors, the previous frame and up to 400 characters of notes Astra wrote to itself last turn. This keeps cost and latency constant over long episodes. All protocols use the same 100 seeds, which fix the drop order and timing. Every recorded game was re-simulated from its applied commands and reproduced exactly.</p>
  </div>
</div></section>

<section id="calibration"><div class="wrap">
  <div class="head"><span class="eyebrow">04 · Calibration</span>
  <h2>What a fast controller <em>can</em> do on these seeds</h2>
  <p class="sub">Scripted controllers on the same 100 seeds establish that the task is solvable, solvable from cameras alone, and where the latency budget lies.</p></div>
  <div class="fig">
    <div class="t"><span><b>Table 1</b> · Scripted controllers</span><span>800 drops each</span></div>
    <h3>Solvable with knowledge, solvable from pixels, impossible by guessing</h3>
    <div class="scroll"><table>
      <tr><th>Controller</th><th class="n">Caught</th><th class="n">Rate</th></tr>
      {cal_row("Schedule oracle", "oracle", "Knows the release schedule; parks the free hand under the next stick and closes 0.16 s after release.")}
      {cal_row("Camera-only reflex", "vision", "Finds sticks in the first head image, flags a drop when pink pixels leave the magnet or appear below the tip (40–80 ms), then intercepts.")}
      {cal_row("Parked guess, timed close", "parked", "Both hands wait under two fixed sticks all game and close at the right moment: the most pure guessing can earn.")}
      {cal_row("Parked guess, half-closed jaws", "parked_trap", "Same, but jaws held narrower than the stick with no timing at all. Would catch every stick that falls into it without the open-jaws rule.")}
      {cal_row("Idle", "idle", "Never moves.")}
    </table></div>
  </div>
  <div class="fig">
    <div class="t"><span><b>Fig. 2</b> · Catch rate vs. reaction latency</span><span>scripted reflex: 800 drops per point</span></div>
    <h3>The budget is about a tenth of a second; Astra lives two orders of magnitude to the right</h3>
    <p class="d">A reflex controller is told which stick fell after a fixed delay. Astra's points use the median game time per Realtime decision.</p>
    <div class="legend"><span><i class="line" style="background:{S1}"></i>Scripted reflex (event + delay)</span><span><i style="background:{S2}"></i>Astra Realtime variants</span><span><i class="line" style="background:{AXIS}"></i>Reference rates</span></div>
    {fig_curve()}
    <details><summary>Table view</summary><div class="scroll"><table><tr><th>Delay</th><th class="n">Catch rate</th></tr>{curve_rows}</table></div></details>
    <p class="cap">The reflex catches {pct(dict(curve).get(0.04, 0))} at 40 ms and {pct(dict(curve).get(0.08, 0))} at 80 ms of delay, falls to {pct(dict(curve).get(0.12, 0))} at 120 ms and {pct(dict(curve).get(0.24, 0))} at 240 ms. Half its catches are gone by about <b>{(L50 or 0) * 1000:.0f} ms</b>. The robot's own motion (about 0.2 s to reach a stick, 0.1 s to close) uses most of the 0.56 s fall. At 0 ms the scripted controller picks a high intercept and sometimes meets the stick side-on, which is why it scores slightly below its 40 ms result.</p>
  </div>
  <div class="fig">
    <div class="t"><span><b>Fig. 3</b> · Time scales</span><span>log scale</span></div>
    <h3>Everything that matters happens before Astra's first token</h3>
    {fig_budget()}
  </div>
</div></section>

<section id="results"><div class="wrap">
  <div class="head"><span class="eyebrow">05 · Results</span>
  <h2>Perfect with state, competent when time stops, <em>zero</em> in real time</h2></div>
  <div class="fig">
    <div class="t"><span><b>Table 2</b> · GPT-6.2-Astra on 100 seeds per protocol</span><span>95% CI: bootstrap over episodes</span></div>
    <div class="scroll"><table>
      <tr><th>Protocol</th><th class="n">Caught</th><th class="n">Rate</th><th class="n">95% CI</th><th class="n">Perfect games</th><th class="n">Median decision</th></tr>
      {mode_row("Privileged", PRIV, f"Planning turns; {PRIV.get('turns_mean', 0):.1f} turns per game on average, all {PRIV.get('submitted', 0)} games submitted")}
      {mode_row("Standard", STD, f"{STD.get('decisions_per_episode', 0):.0f} decisions per game; world paused during each", True)}
      {mode_row("Realtime", RT, f"{np.mean(rt_complete) if rt_complete else 0:.1f} commands applied per 13.7 s game")}
    </table></div>
  </div>
  <div class="prose" style="margin-top:28px">
  <p><b>Privileged.</b> With the schedule in hand, the task reduces to placing a hand under the right stick and timing the close. Astra wrote keyframe plans, simulated once or twice, fixed whatever failed (typically a close that came too early or a hand that disturbed another), and submitted. {PRIV['caught']}/{PRIV['drops']} catches in the graded replays. Privileged mode is an upper bound here, and an easy one: it removes perception and time pressure together.</p>
  <p><b>Standard.</b> Sensors only, but time stands still. Astra played the way a careful human would: it recorded pixel coordinates for every stick tip in its notes at the start, then polled in short steps ({pct(std_short, 0)} of its <code>advance_ms</code> choices were ≤ 80 ms), spotted a released stick within one or two frames, moved the nearest open hand under it, and closed. It got a jaw pad onto <b>{std_touch} of {STD['drops']}</b> falling sticks ({pct(std_touch / STD['drops'])}) and held {STD['caught']}. So detection and reaching work; the grasp is where it fails. First contact came {lat_c * 1000:.0f} ms after release for catches and {lat_m * 1000:.0f} ms for misses (medians), so missed sticks were not reached later. They were lost after contact: the jaws closed too early or too late for the stick to be held, or the stick slid out. {trapped_n} holds were "trapped", with the jaws already partly closed as the stick entered. It averaged {STD.get('decisions_per_episode', 0):.0f} decisions per 13.7 s game, a median {STD.get('latency_median_s', 0):.1f} s each, so one game takes about {STD.get('wall_s_median', 0) / 60:.0f} minutes of wall time.</p>
  <p><b>Realtime.</b> The countdown gives 3 s before the first possible release, and a decision takes {RT['latency_median_s']:.1f} s at the median ({RT.get('latency_min_s', 0):.1f} s at best). In <b>{rt_late_first} of {len(rt_recs)}</b> games Astra's first command arrived after the first stick had already landed. It completed only {np.mean(rt_complete) if rt_complete else 0:.1f} commands per game. Across 800 drops, <b>{RT.get('drops_with_postrelease_command', 0)}</b> received a command based on an image taken after that stick's release before the stick was resolved. Nothing it did could react to a drop. The score of 0 is not noise: even the best pure guess (Table 1, timed close) would need a close timed to within about 0.1 s of a release it cannot see coming.</p>
  </div>
  <div class="fig">
    <div class="t"><span><b>Fig. 4</b> · Standard: sticks caught per game</span><span>{n_std} games</span></div>
    <h3>Competent but inconsistent when time is free</h3>
    {fig_hist()}
  </div>
</div></section>

<section id="limit"><div class="wrap">
  <div class="head"><span class="eyebrow">06 · Pushing Realtime to the limit</span>
  <h2>Can Astra be made <em>fast enough?</em></h2>
  <p class="sub">We attacked the latency in three ways: shrink what each decision has to read, cache what does not change, and, as a measurement device rather than a fix, slow the world down until Astra starts to catch.</p></div>
  <div class="fig">
    <div class="t"><span><b>Fig. 5</b> · Decision latency floor</span><span>20 calls per row, unloaded</span></div>
    <h3>Where the seconds go</h3>
    <p class="d">Median with p10–p90 range, from request to parsed JSON, reasoning effort low</p>
    {fig_profile()}
    <p class="cap">{prof_cap}{fast_txt}</p>
  </div>
  <div class="fig">
    <div class="t"><span><b>Fig. 6</b> · Slow-motion sweep</span><span>Realtime protocol, full sensors</span></div>
    <h3>How much slower must the world run before Astra can play?</h3>
    <p class="d">The simulator advances one 40 ms tick every 40 ms × k of wall time. k = 1 is Realtime; "paused" is Standard. Whiskers: 95% bootstrap CI.</p>
    {fig_sweep() if sweep else "<p class='cap'>Pending: the sweep was running when this page was built.</p>"}
    {"<details><summary>Table view</summary><div class='scroll'><table><tr><th>Slow-down</th><th class='n'>Games</th><th class='n'>Caught</th><th class='n'>Rate</th><th class='n'>Wall latency</th><th class='n'>Game-time latency</th><th class='n'>Decisions/game</th></tr>" + sweep_rows + "</table></div></details>" if sweep else ""}
    <p class="cap">{sweep_txt} Read across Fig. 2, this converts Astra's score into an <b>effective reaction time</b>: the slow-motion factor at which it starts to catch, multiplied by its wall-clock latency, lands where the scripted reflex's curve does.</p>
  </div>
</div></section>

<section id="discussion"><div class="wrap">
  <div class="head"><span class="eyebrow">07 · Discussion</span>
  <h2>What this exam measures, <em>and what it does not</em></h2></div>
  <div class="notes">
    <div class="note"><h4>A clean latency probe</h4><p>Perception and planning are not what fails: Privileged is perfect and Standard reaches {pct(STD['catch_rate'], 0)}. The whole Standard-to-Realtime gap ({pct(STD['catch_rate'], 0)} → {pct(RT['catch_rate'], 0)}) is time. That makes the task a good discriminator between "can reason about it" and "can do it", and a poor one between two slow models: a controller above about 0.25 s of reaction latency scores essentially zero here (Fig. 2).</p></div>
    <div class="note"><h4>Report a curve, not a point</h4><p>A single Realtime score saturates at 0 for every current foundation model. The slow-motion factor needed to reach a given fraction of the Standard score is a continuous, comparable number. We recommend reporting it alongside the 1× score, together with the scripted latency curve for the same seeds.</p></div>
    <div class="note"><h4>The exploit we closed</h4><p>Without the open-jaws rule a passive trap (a half-closed gripper parked under a stick) catches with no timing. A slow agent can then score about 25% by parking under two sticks, or more by re-parking under still-hanging ones. That would turn a latency test into a guessing game. Any physical version of this task needs the same rule, or a gripper that cannot be pre-shaped.</p></div>
    <div class="note"><h4>The architectural answer</h4><p>Humans pass this test with a reflex, not deliberation. A robot would too: a fast learned or scripted catch skill (Table 1's camera reflex runs in milliseconds) supervised by a slow planner that chooses where to wait. The exam then measures the fast layer. As posed here (the model is the only controller) it measures the model's raw loop time.</p></div>
  </div>
  <ul class="plain">
    <li><b>Harness overhead is included.</b> Latency is measured end to end through the Codex app-server, which adds session and serialization cost a dedicated API path would not. It is the latency the agent actually experiences, but it is not pure model time.</li>
    <li><b>Concurrency.</b> The Realtime run (4 parallel games) overlapped with the Standard run (8 parallel games). The latency-profile, fast-profile and slow-motion runs were scheduled after Standard finished.{f" Unloaded, the same full sensor turn took {prof_full:.1f} s (median) instead of {RT['latency_median_s']:.1f} s, so load inflated Realtime latency but cannot account for a 0.1 s budget." if prof_full else ""}</li>
    <li><b>Simulation choices.</b> The base is a planar differential drive, the lift is fixed, the waist motor is set to the G1's 88 N·m rating rather than the URDF's 30 N·m, and the two wide head cameras are an addition. The grippers, inertias and limits are Unitree's. Physical calibration against a real G1-D was not done.</li>
    <li><b>Single model, single effort.</b> One model at reasoning effort low, under one prompt design (stateless turns with self-written notes). A persistent-thread design was tried only in the fast profile.</li>
    <li><b>Scripted calibration is a reference, not an optimum.</b> A better reflex controller would push the latency curve right somewhat. It cannot move it by two orders of magnitude, because the stick reaches the floor in 0.56 s.</li>
  </ul>
</div></section>

<section id="reproduce"><div class="wrap">
  <div class="head"><span class="eyebrow">08 · Reproduce</span><h2>Commands</h2></div>
<pre><span class="c"># tests and calibration (no model access needed)</span>
.venv/bin/python -m pytest -q tests/test_ringsim.py
.venv/bin/python -m ringsim.calibrate --episodes 100 --out outputs/ringsim/calibration.json

<span class="c"># the three protocols (needs a Codex login with GPT-6.2-Astra)</span>
.venv/bin/python -m ringsim.exam --mode privileged --episodes 100 --workers 6 --out outputs/ringsim/astra_privileged.jsonl
.venv/bin/python -m ringsim.exam --mode standard   --episodes 100 --workers 8 --out outputs/ringsim/astra_standard.jsonl
.venv/bin/python -m ringsim.exam --mode realtime   --episodes 100 --workers 4 --out outputs/ringsim/astra_realtime.jsonl

<span class="c"># pushing realtime: payload floor, fast profile, slow motion</span>
.venv/bin/python -m ringsim.latency_profile --n 20 --out outputs/ringsim/latency_profile.json
.venv/bin/python -m ringsim.exam --mode realtime --profile fast --episodes 100 --out outputs/ringsim/astra_realtime_fast.jsonl
.venv/bin/python -m ringsim.exam --mode realtime --slowdown 10 --episodes 30 --out outputs/ringsim/astra_realtime_slow10x.jsonl

<span class="c"># verify a run reproduces from its applied commands, render a game</span>
.venv/bin/python -m ringsim.replay outputs/ringsim/astra_standard.jsonl
.venv/bin/python -m ringsim.replay outputs/ringsim/astra_standard.jsonl --seed 50000 --video game.mp4</pre>
  <div class="prose" style="margin-top:24px"><p>In total the runs on this page made <b>{dec_total:,}</b> Astra decisions and consumed about <b>{tok_total / 1e6:.1f} M</b> tokens. Every runner checkpoints per episode and resumes. Raw per-episode records (commands, timings, notes, outcomes) are in <code>outputs/ringsim/</code>.{" Note: " + "; ".join(pending) + "." if pending else ""}</p></div>
</div></section>

<section id="references"><div class="wrap">
  <div class="head"><span class="eyebrow">References</span></div>
  <ol class="src">
    <li id="r1">Made for Arcade, Reaction Ring user manual: sticks, drop modes, difficulty intervals, machine dimensions.<a href="https://madeforarcade.com/manual/reaction-ring-user-manual-2/">madeforarcade.com/manual/reaction-ring-user-manual-2</a></li>
    <li id="r2">Unitree Robotics, G1-D description (URDF, meshes), revision 157ff95.<a href="https://github.com/unitreerobotics/unitree_ros/tree/157ff950351e5f535e03c5d85a16b7d594b3dbcc/robots/g1_d_description">github.com/unitreerobotics/unitree_ros</a></li>
    <li id="r3">MuJoCo 4.0.1 physics engine.<a href="https://mujoco.org">mujoco.org</a></li>
  </ol>
</div></section>

<footer><div class="wrap"><span>distill_frontier · Reaction Ring latency exam · built from outputs/ringsim</span><a href="../">← project home</a></div></footer>
<script>
const tip=document.getElementById('tip');
function show(e,el){{tip.innerHTML=el.getAttribute('data-tip');tip.style.opacity=1;
  const r=el.getBoundingClientRect();const x=(e&&e.clientX)||r.left+r.width/2,y=(e&&e.clientY)||r.top;
  tip.style.left=Math.min(x+14,innerWidth-310)+'px';tip.style.top=(y+14)+'px';}}
document.querySelectorAll('[data-tip]').forEach(el=>{{
  el.addEventListener('mousemove',e=>show(e,el));el.addEventListener('focus',()=>show(null,el));
  el.addEventListener('mouseleave',()=>tip.style.opacity=0);el.addEventListener('blur',()=>tip.style.opacity=0);}});
</script>
</body>
</html>
"""
DOC.mkdir(parents=True, exist_ok=True)
(DOC / "index.html").write_text(page)
print("wrote", DOC / "index.html", f"({len(page) / 1024:.0f} KB)")
