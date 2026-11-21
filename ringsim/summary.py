"""Summarise exam runs and the calibration into one JSON + a markdown table.

    python -m ringsim.summary --dir outputs/ringsim
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def bootstrap(caught, drops, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    c, d = np.array(caught), np.array(drops)
    idx = rng.integers(0, len(c), size=(n, len(c)))
    rates = c[idx].sum(1) / d[idx].sum(1)
    return [round(float(np.percentile(rates, 2.5)), 4), round(float(np.percentile(rates, 97.5)), 4)]


def reacted(rec):
    """Drops for which a command based on an image taken after release took effect before the
    drop was resolved (the only way a latency test can be passed by reacting)."""
    n = 0
    for e in rec["events"]:
        for dcs in rec.get("decisions", []):
            applied = dcs.get("t_applied", dcs["t_obs"] if rec["mode"] == "standard" else None)
            if "action" in dcs and applied is not None and \
                    dcs["t_obs"] > e["release_t"] and applied < (e["outcome_t"] or 1e9):
                n += 1
                break
    return n


def summarise(recs):
    ok = [r for r in recs if "fatal" not in r]
    if not ok:
        return None
    caught = [r["caught"] for r in ok]
    drops = [r["drops"] for r in ok]
    outcomes = {}
    for r in ok:
        for e in r["events"]:
            outcomes[e["outcome"]] = outcomes.get(e["outcome"], 0) + 1
    s = {"episodes": len(ok), "failed_episodes": len(recs) - len(ok),
         "caught": int(sum(caught)), "drops": int(sum(drops)),
         "catch_rate": round(sum(caught) / sum(drops), 4),
         "catch_rate_ci95": bootstrap(caught, drops),
         "perfect_episodes": sum(c == d for c, d in zip(caught, drops)),
         "outcomes": outcomes, "touched": sum(e["first_pad_contact_t"] is not None
                                              for r in ok for e in r["events"])}
    lat = [d["latency_s"] for r in ok for d in r.get("decisions", r.get("turns", []))
           if "latency_s" in d and "error" not in d]
    if lat:
        s.update(decisions=len(lat), decisions_per_episode=round(len(lat) / len(ok), 1),
                 latency_median_s=round(float(np.median(lat)), 2),
                 latency_p95_s=round(float(np.percentile(lat, 95)), 2),
                 latency_min_s=round(float(np.min(lat)), 2))
    sim_lat = [d["t_applied"] - d["t_obs"] for r in ok for d in r.get("decisions", [])
               if "t_applied" in d]
    if sim_lat:
        s["sim_latency_median_s"] = round(float(np.median(sim_lat)), 3)
        s["sim_latency_min_s"] = round(float(np.min(sim_lat)), 3)
    errs = sum("error" in d for r in ok for d in r.get("decisions", r.get("turns", [])))
    s["decision_errors"] = errs
    if ok[0]["mode"] in ("standard", "realtime"):
        s["drops_with_postrelease_command"] = sum(reacted(r) for r in ok)
    if ok[0]["mode"] == "privileged":
        s["submitted"] = sum(r.get("submitted", False) for r in ok)
        s["turns_mean"] = round(float(np.mean([len(r["turns"]) for r in ok])), 2)
    tok = [d["usage"]["totalTokens"] for r in ok for d in r.get("decisions", r.get("turns", []))
           if d.get("usage")]
    if tok:
        s["tokens_total"] = int(sum(tok))
    s["wall_s_median"] = round(float(np.median([r["wall_s"] for r in ok])), 1)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/ringsim")
    args = ap.parse_args()
    d = Path(args.dir)
    out = {"model": "gpt-6.2-astra", "seeds": "50000-50099"}
    for mode in ("privileged", "standard", "realtime"):
        out[mode] = summarise(load(d / f"astra_{mode}.jsonl"))
    # realtime variants: astra_realtime_<tag>.jsonl (fast profile, slow-motion sweeps)
    out["realtime_variants"] = {}
    for f in sorted(d.glob("astra_realtime_*.jsonl")):
        recs = load(f)
        s = summarise(recs)
        if s:
            s["slowdown"] = recs[0].get("slowdown", 1.0)
            s["profile"] = recs[0].get("profile", "full")
            out["realtime_variants"][f.stem.replace("astra_realtime_", "")] = s
    lp = d / "latency_profile.json"
    if lp.exists():
        out["latency_profile"] = {k: {kk: vv for kk, vv in v.items() if kk != "samples_s"}
                                  for k, v in json.loads(lp.read_text()).items() if v.get("n")}
    cal = d / "calibration.json"
    if cal.exists():
        out["calibration"] = {k: {kk: v[kk] for kk in ("caught", "drops", "catch_rate")}
                              for k, v in json.loads(cal.read_text())["controllers"].items()}
    (d / "summary.json").write_text(json.dumps(out, indent=1))
    print("| Mode | Caught | Rate (95% CI) | Perfect games | Median decision |")
    print("|---|---:|---|---:|---:|")
    rows = [(m, out[m]) for m in ("privileged", "standard", "realtime")]
    rows += [("realtime_" + k, v) for k, v in out["realtime_variants"].items()]
    for mode, s in rows:
        if s:
            ci = s["catch_rate_ci95"]
            print(f"| {mode} | {s['caught']}/{s['drops']} | {s['catch_rate']:.1%} "
                  f"({ci[0]:.1%}-{ci[1]:.1%}) | {s['perfect_episodes']}/{s['episodes']} | "
                  f"{s.get('latency_median_s', '-')} s |")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
