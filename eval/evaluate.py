"""Evaluate Qwen (and the scripted expert as the reference ceiling) on 100 held-out seeds.

Seeds 100000-100099 (never used in training/curation/checkpoints). Greedy decoding, strict JSON
schema validation (invalid output = failed step). Records every episode at low resolution by
default (configs eval.record: none | lowres | highres) and builds outputs/eval_gallery.html.

  .venv/bin/python -m eval.evaluate
  .venv/bin/python -m eval.evaluate --policies expert --record none
"""
from __future__ import annotations

import argparse
import collections
import html
import json
import time
from pathlib import Path

from viz.viz import setup_gl
from env.common import REPO, get_logger, load_config, set_seeds, write_json

FAILURES = ["failed_grasp", "dropped_bowl", "missed_target", "invalid_json", "step_limit"]


def classify(ep):
    if ep["success"]:
        return ""
    if ep["metrics"]["n_invalid"] > 0:
        return "invalid_json"
    return ep["failure_type"] or "other"


def make_recorder(cfg, record, policy, i):
    from viz.viz import RecordViz

    if record == "none" or (record == "highres" and i >= cfg["eval"]["highres_subset"]):
        return None
    return RecordViz(cfg, f"eval/{policy}", lowres=(record == "lowres"), out_root=str(REPO / "outputs" / "videos"))


def run_expert(cfg, seeds, record, log):
    from env.rollout import run_episode
    from env.so101_env import BowlEnv
    from expert.policy import ExpertPolicy

    pol = ExpertPolicy(cfg)
    eps = []
    for i, s in enumerate(seeds):
        viz = make_recorder(cfg, record, "expert", i)
        env = BowlEnv(cfg, viz=viz)
        ep = run_episode(env, pol, s, episode_id=f"expert_{s}")
        ep["video"] = viz.last_path if viz else None
        if viz:
            viz.close()
        eps.append(ep)
    log.info(f"expert: {sum(e['success'] for e in eps)}/{len(eps)}")
    return eps


def run_qwen(cfg, seeds, record, model_path, adapter, batch, log):
    from env.rollout import run_episodes_lockstep
    from env.so101_env import BowlEnv
    from eval.qwen_policy import QwenPolicy

    pol = QwenPolicy.load(model_path, adapter, cfg["eval"]["max_new_tokens"], name="qwen")
    eps = []
    for b0 in range(0, len(seeds), batch):
        chunk = seeds[b0:b0 + batch]
        vizs = [make_recorder(cfg, record, "qwen", b0 + j) for j in range(len(chunk))]
        envs = [BowlEnv(cfg, viz=v) for v in vizs]
        t0 = time.time()
        res = run_episodes_lockstep(envs, pol, chunk, [f"qwen_{s}" for s in chunk], cfg["train"]["history_len"])
        for ep, v in zip(res, vizs):
            ep["video"] = v.last_path if v else None
            if v:
                v.close()
        eps.extend(res)
        log.info(f"qwen: episodes {b0 + len(chunk)}/{len(seeds)}, success so far "
                 f"{sum(e['success'] for e in eps)}/{len(eps)} ({time.time() - t0:.0f}s for this batch)")
    return eps


def summarize(eps, cfg):
    from data.curate import xy_cell

    n = len(eps)
    succ = [e for e in eps if e["success"]]
    fails = collections.Counter(classify(e) for e in eps if not e["success"])
    per_bin = collections.defaultdict(lambda: [0, 0])
    for e in eps:
        c = xy_cell(e["spawn"]["x"], e["spawn"]["y"], cfg)
        per_bin[f"x{c[0]}y{c[1]}"][0] += int(e["success"])
        per_bin[f"x{c[0]}y{c[1]}"][1] += 1
    return {
        "success": len(succ), "n": n, "success_rate": len(succ) / max(1, n),
        "mean_steps_to_success": round(sum(e["n_steps"] for e in succ) / len(succ), 2) if succ else None,
        "failure_breakdown": {k: fails.get(k, 0) for k in FAILURES} | {k: v for k, v in fails.items() if k not in FAILURES},
        "invalid_steps": sum(e["metrics"]["n_invalid"] for e in eps),
        "per_bin": {k: {"success": v[0], "n": v[1]} for k, v in sorted(per_bin.items())},
        "episodes": [{"seed": e["seed"], "success": e["success"], "failure": classify(e), "n_steps": e["n_steps"],
                      "spawn": e["spawn"], "video": e.get("video")} for e in eps],
    }


def print_table(results, cfg):
    pols = list(results)
    w = 16
    print("\n" + "=" * (24 + w * len(pols)))
    print(f"{'held-out seeds':<24}" + "".join(f"{p:>{w}}" for p in pols))
    print("-" * (24 + w * len(pols)))
    print(f"{'success / 100':<24}" + "".join(f"{results[p]['success']:>{w}}" for p in pols))
    print(f"{'mean steps to success':<24}" + "".join(f"{str(results[p]['mean_steps_to_success']):>{w}}" for p in pols))
    for f in FAILURES:
        print(f"{'  fail: ' + f:<24}" + "".join(f"{results[p]['failure_breakdown'].get(f, 0):>{w}}" for p in pols))
    print("-" * (24 + w * len(pols)))
    gx, gy = cfg["curation"]["grid_xy"]
    for p in pols:
        print(f"\nsuccess per workspace cell ({p}); rows = y (high->low), cols = x (near->far)")
        for j in reversed(range(gy)):
            row = []
            for i in range(gx):
                b = results[p]["per_bin"].get(f"x{i}y{j}")
                row.append(f"{b['success']}/{b['n']}" if b else "  -")
            print("   " + "  ".join(f"{c:>5}" for c in row))
    print("=" * (24 + w * len(pols)) + "\n")


def gallery(results, path):
    cards = []
    for pol, r in results.items():
        for e in r["episodes"]:
            if not e["video"]:
                continue
            rel = Path(e["video"]).resolve().relative_to(path.parent.resolve()).as_posix()
            ok = e["success"]
            label = "success" if ok else e["failure"]
            cards.append(
                f'<figure class="card" data-policy="{pol}" data-outcome="{"success" if ok else "failure"}" '
                f'data-failure="{html.escape(label)}"><video src="{rel}" muted loop playsinline preload="metadata"></video>'
                f'<figcaption><span class="pill {"ok" if ok else "bad"}">{html.escape(label)}</span>'
                f'<span class="meta">{pol} · seed {e["seed"]} · {e["n_steps"]} steps</span></figcaption></figure>')
    summary = " · ".join(f"<b>{p}</b> {r['success']}/{r['n']}" for p, r in results.items())
    pols = "".join(f'<option value="{p}">{p}</option>' for p in results)
    fl = sorted({c.split('data-failure="')[1].split('"')[0] for c in cards})
    fopts = "".join(f'<option value="{f}">{f}</option>' for f in fl)
    page = f"""<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Eval Gallery</title><style>
:root{{--bg:#fcfcfb;--card:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--ok:#0b7a3b;--okbg:#e3f4ea;--bad:#b42318;--badbg:#fdecea}}
@media (prefers-color-scheme:dark){{:root{{--bg:#1a1a19;--card:#232322;--ink:#f0efec;--ink2:#c3c2b7;--line:#3a3a38;--ok:#7fd6a0;--okbg:#17321f;--bad:#f2a49b;--badbg:#3b1d1a}}}}
body{{margin:0;padding:16px;background:var(--bg);color:var(--ink);font:14px system-ui,sans-serif}}
h1{{font-size:20px;margin:0 0 4px}} .sum{{color:var(--ink2);margin-bottom:12px}}
.bar{{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px}} select{{padding:6px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--ink)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}}
.card{{margin:0;background:var(--card);border:1px solid var(--line);border-radius:8px;overflow:hidden}}
video{{width:100%;display:block;background:#000}} figcaption{{padding:6px 8px;display:flex;gap:8px;align-items:center;flex-wrap:wrap}}
.pill{{font-size:12px;padding:2px 8px;border-radius:99px;font-weight:600}} .ok{{color:var(--ok);background:var(--okbg)}} .bad{{color:var(--bad);background:var(--badbg)}}
.meta{{color:var(--ink2);font-size:12px}} .hidden{{display:none}}</style></head><body>
<h1>Held-out evaluation (seeds 100000–100099)</h1><div class=sum>{summary} · hover a clip to play</div>
<div class=bar><select id=pol><option value="">all policies</option>{pols}</select>
<select id=out><option value="">all outcomes</option><option value="success">success</option><option value="failure">failure</option></select>
<select id=fail><option value="">all labels</option>{fopts}</select><span id=count class=meta></span></div>
<div class=grid id=grid>{''.join(cards)}</div>
<script>
const f=()=>{{const p=pol.value,o=out.value,l=fail.value;let n=0;
document.querySelectorAll('.card').forEach(c=>{{const s=(!p||c.dataset.policy===p)&&(!o||c.dataset.outcome===o)&&(!l||c.dataset.failure===l);
c.classList.toggle('hidden',!s);n+=s}});count.textContent=n+' clips';}};
[pol,out,fail].forEach(e=>e.onchange=f);f();
document.querySelectorAll('video').forEach(v=>{{v.parentElement.onmouseenter=()=>v.play();v.parentElement.onmouseleave=()=>v.pause();}});
</script></body></html>"""
    path.write_text(page)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--policies", nargs="+", default=["qwen", "expert"])
    ap.add_argument("--record", default=None, choices=["none", "lowres", "highres"])
    ap.add_argument("--model", default=None, help="merged model dir (default: paths.merged_dir)")
    ap.add_argument("--adapter", default=None, help="optional: base model + this LoRA adapter instead of merged")
    ap.add_argument("--batch", type=int, default=25, help="episodes run in lockstep per batched generate")
    args = ap.parse_args()
    cfg = load_config(args.config)
    record = args.record or cfg["eval"]["record"]
    setup_gl("record" if record != "none" else "none")
    log = get_logger("eval", cfg)
    set_seeds(cfg["seeds"]["global"])
    s0, n = cfg["seeds"]["eval_start"], cfg["seeds"]["eval_count"]
    seeds = list(range(s0, s0 + n))
    assert n == 100 and min(seeds) >= 100000
    results, raw = {}, {}
    t0 = time.time()
    for pol in args.policies:
        if pol == "expert":
            eps = run_expert(cfg, seeds, record, log)
        else:
            model = args.model or str(REPO / cfg["paths"]["merged_dir"])
            if args.adapter:
                model = cfg["train"]["base_model"]
            eps = run_qwen(cfg, seeds, record, model, args.adapter, args.batch, log)
        results[pol] = summarize(eps, cfg)
        raw[pol] = eps
    out = REPO / "outputs"
    write_json(out / "results.json", {"seeds": [seeds[0], seeds[-1]], "wall_time_s": round(time.time() - t0, 1),
                                      **results})
    with open(out / "eval_episodes.jsonl", "w") as f:
        for pol, eps in raw.items():
            for e in eps:
                f.write(json.dumps(e) + "\n")
    gallery(results, out / "eval_gallery.html")
    print_table(results, cfg)
    log.info(f"results -> {out / 'results.json'}, gallery -> {out / 'eval_gallery.html'}")


if __name__ == "__main__":
    main()
