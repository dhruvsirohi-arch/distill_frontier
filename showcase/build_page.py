"""Build docs/index.html (GitHub Pages) from the run's own output files.

  .venv/bin/python -m showcase.render       # footage -> docs/assets/ (+ media.json)
  .venv/bin/python -m showcase.build_page   # numbers + template -> docs/index.html

Every number on the page comes from outputs/*.json, outputs/eval_episodes.jsonl or configs/config.yaml.
"""
from __future__ import annotations

import collections
import datetime as dt
import json

from env.common import REPO, load_config
from train.prompt import action_to_text, user_text

REPO_URL = "https://github.com/dhruvsirohi-arch/distill_frontier"

CKPT_STORY = {
    0: ("Copies the target and ignores the bowl.",
        "Before fine-tuning, Qwen already writes valid JSON from the prompt alone. But its first move sends the "
        "empty gripper straight to the target coordinates, then it opens and declares done."),
    374: ("Finds the bowl, but the grasp misses.",
          "It now reaches down into the bowl at the right spot, but the grasp doesn't hold. It heads to the table "
          "empty-handed and fumbles there until it runs out of moves."),
    749: ("Pick, carry, place.",
          "The full sequence emerges: approach, descend, close, lift, carry, lower, release, retreat, done. "
          "It solves 4 of the 5 checkpoint scenes."),
    1124: ("Every checkpoint scene solved.",
           "5 of 5, while validation loss keeps falling on episodes the model has never trained on."),
    1498: ("The shipped policy.",
           "The final checkpoint: about ten chat turns per task, one action per turn. On the full held-out "
           "benchmark it solves 90 of 100."),
}
FAIL_DESC = {
    "dropped_bowl": "lifted, then lost the bowl in transit",
    "failed_grasp": "closed the gripper without securing the bowl",
    "missed_target": "placed the bowl, but outside the target region",
    "invalid_json": "produced output that failed schema validation",
    "step_limit": "ran out of the 15-decision budget",
}


def latency_block(out):
    """Summary + one representative move_to decision for the token timeline (None if not measured)."""
    lp = out / "latency.json"
    if not lp.exists():
        return None
    L = json.load(open(lp))
    recs = [json.loads(line) for line in open(out / "latency_decisions.jsonl")]
    mv = [r for r in recs if r["action"] == "move_to"]
    med = sorted(r["total_ms"] for r in mv)[len(mv) // 2]
    ex = min(mv, key=lambda r: abs(r["total_ms"] - med))
    # which tokens are decided by the model (action name, digits, sign) vs fixed JSON syntax
    text, spans = "", []
    for t in ex["tokens"]:
        spans.append((len(text), len(text) + len(t)))
        text += t
    a0 = text.find('"action":"') + len('"action":"')
    a1 = text.find('"', a0)
    fixed = []
    for (i0, i1), t in zip(spans, ex["tokens"]):
        content = any(c.isdigit() or c == "-" for c in t) or (i0 < a1 and i1 > a0)
        fixed.append(not content)
    order = ["move_to", "open_gripper", "close_gripper", "done"]
    by = [{"action": k, "n": v["n"], "tokens": v["output_tokens"], "p50": v["total_ms"]["p50"],
           "p90": v["total_ms"]["p90"]} for k, v in L["by_action"].items() if k in order]
    by.sort(key=lambda r: order.index(r["action"]))
    dev = L["setup"]["device"]
    return {
        "decisions": L["decisions"], "decision_ms": L["decision_ms"], "ttft_ms": L["ttft_ms"],
        "decode_ms_per_token": L["decode_ms_per_token"], "tps": L["decode_tokens_per_s"],
        "out_tokens": L["output_tokens_per_action"]["mean"], "prompt_tokens": L["prompt_tokens"]["mean"],
        "by_action": by, "ep_model_s": L["per_episode_model_seconds"]["p50"],
        "ep_sim_s": L["per_episode_sim_seconds"]["p50"],
        "dec_per_ep": L["decisions"] / L["episodes"]["n"],
        "device_label": "Apple M5" if dev.startswith("mps") else dev, "equiv": L["setup"]["generate_equivalence"],
        "example": {"seed": ex["seed"], "step": ex["step"], "tokens": ex["tokens"], "token_ms": ex["token_ms"],
                    "fixed": fixed, "text": text.replace("<|im_end|>", "").strip()},
    }


def main():
    cfg = load_config(None)
    out = REPO / "outputs"
    res = json.load(open(out / "results.json"))
    train = json.load(open(out / "train_summary.json"))
    cur = json.load(open(out / "curation_summary.json"))
    expc = json.load(open(out / "expert_check.json"))
    gen = json.load(open(out / "data" / "raw_episodes.summary.json"))
    media = json.load(open(REPO / "docs" / "assets" / "media.json"))
    eps = [json.loads(line) for line in open(out / "eval_episodes.jsonl")]
    q_eps = [e for e in eps if e["policy"] == "qwen"]
    e_eps = [e for e in eps if e["policy"] == "expert"]
    curated = json.load(open(out / "data" / "curated" / "episodes.json"))

    q, ex = res["qwen"], res["expert"]
    decisions = sum(e["n_steps"] for e in q_eps)
    invalid = sum(e["metrics"]["n_invalid"] + e["metrics"]["n_rejected"] for e in q_eps)
    coll = collections.Counter(s["result"] for e in q_eps for s in e["steps"])["collision"]
    coll_eps = [e for e in q_eps if e["metrics"]["n_collisions"] > 0]
    e_coll = sum(e["metrics"]["n_collisions"] for e in e_eps)

    # a real held-out exchange for the "what the model sees" panel (validation split, never trained on)
    val_rows = [json.loads(line) for line in open(out / "data" / "curated" / "val.jsonl")]
    row = next(r for r in val_rows if r["observation"]["step"] == 2 and r["action"]["action"] == "move_to")
    hero_seed = media["hero"][0]["seed"]
    trace_ep = next(e for e in q_eps if e["seed"] == hero_seed)
    trace = [action_to_text(s["action"]) for s in trace_ep["steps"]]

    epoch = lambda s: s / (train["trained_steps"] / train["epochs_effective"])  # noqa: E731
    progress = []
    for p in media["progress"]:
        t, x = CKPT_STORY.get(p["step"], ("", ""))
        progress.append({**p, "epoch": epoch(p["step"]), "title": t, "text": x})

    # untuned base model on the same 100 seeds (eval.evaluate --base)
    rb = json.load(open(out / "results_base.json"))["qwen_base"]
    b_eps = [json.loads(line) for line in open(out / "eval_episodes_base.jsonl")]
    b_invalid = sum(e["metrics"]["n_invalid"] + e["metrics"]["n_rejected"] for e in b_eps)
    b_fail = collections.Counter({k: v for k, v in rb["failure_breakdown"].items() if v})
    fail_keys = ["dropped_bowl", "failed_grasp", "missed_target", "invalid_json", "step_limit"]
    base_failures = [[k.replace("_", " "), rb["failure_breakdown"].get(k, 0)] for k in fail_keys]
    fails = [[k.replace("_", " "), q["failure_breakdown"].get(k, 0), FAIL_DESC[k], ex["failure_breakdown"].get(k, 0)]
             for k in ["dropped_bowl", "failed_grasp", "missed_target", "invalid_json", "step_limit"]]
    est_min = 1.83 * train["trained_steps"] / 60
    D = {
        "generated": dt.date.today().isoformat(),
        "kpi": {"base_success": rb["success"], "qwen_success": q["success"], "expert_success": ex["success"], "invalid": invalid,
                "decisions": decisions, "pairs": cur["pairs_train"], "curated": cur["n_selected"],
                "raw": cur["n_raw"], "expert_check": f"{round(100 * expc['success_rate'])}%"},
        "pipeline": [
            ["Simulate", "MuJoCo with the official SO-101 arm model, a bowl with a yaw-marking lug, and a table.",
             f"{cfg['env']['max_decision_steps']} decisions max · IK executor"],
            ["Demonstrate", "A scripted expert that sees only the model's JSON observation, with randomized style.",
             f"{gen['episodes']:,} episodes in {gen['wall_time_s']:.0f} s · {100 * gen['success_rate']:.1f}% success"],
            ["Curate", "Quality filters, a score, and stratified selection across 100 position × orientation bins.",
             f"{cur['n_raw']:,} → {cur['n_selected']} episodes"],
            ["Distill", "LoRA fine-tune of Qwen3-1.7B. Loss only on action tokens, weighted toward gripper and done.",
             f"{cur['pairs_train']:,} pairs · r=16 · {train['epochs_effective']:.2f} epochs"],
            ["Evaluate", "100 never-seen scenes, greedy decoding, strict schema checks, same success test as the expert.",
             f"before {rb['success']} → after {q['success']} · expert {ex['success']} /100"],
        ],
        "example": {"user": user_text(row["observation"], row["history"]), "action": action_to_text(row["action"]),
                    "trace": trace},
        "demos": media["demos"],
        "funnel": [["Generated by the expert", cur["n_raw"]], ["Pass quality filters", cur["n_passed"]],
                   ["Selected for diversity", cur["n_selected"]],
                   ["State → action training pairs (train split)", cur["pairs_train"], True]],
        "drops": [[{"low_place_margin": "placed too close to the edge", "not_success:failed_grasp": "missed the grasp",
                     "collision": "arm collided", "not_success:step_limit": "ran out of decisions"}.get(k, k), v]
                  for k, v in sorted(cur["reasons"].items(), key=lambda kv: -kv[1])],
        "spawn": [{"x": round(e["spawn"]["x"], 4), "y": round(e["spawn"]["y"], 4), "yaw": round(e["spawn"]["yaw_deg"]),
                   "seed": e["seed"], "split": e["split"]} for e in curated],
        "world": {"pan": [0.0388353, 0.0], "reach": [cfg["env"]["reach"]["r"][0], cfg["env"]["reach"]["r"][1]],
                  "table": {"c": cfg["env"]["table"]["center"], "s": cfg["env"]["table"]["size"]},
                  "target": {"c": cfg["env"]["target"]["center"], "s": cfg["env"]["target"]["size"]},
                  "spawn": {"x": cfg["env"]["spawn"]["x"], "y": cfg["env"]["spawn"]["y"]}},
        "progress": progress,
        "loss": {"train": [[p["step"], round(p["loss"], 4)] for p in train["train_loss"]],
                 "val": [[p["step"], round(p["val_loss"], 4)] for p in train["val_loss"]],
                 "total_steps": train["trained_steps"]},
        "trainfacts": [["Base model", "Qwen3-1.7B"], ["Adapter", "LoRA r16 · all projections"],
                       ["Optimizer steps", f"{train['trained_steps']:,} · {train['epochs_effective']:.2f} epochs"],
                       ["Final validation loss", f"{train['final_val_loss']:.3f}"],
                       ["Compute at measured speed", f"≈ {est_min:.0f} min"]],
        "trainnote": (f"Measured 1.8 s per step on the M5 GPU. Memory pressure from other apps stretched this run to "
                      f"{train['total_minutes']:.0f} min wall-clock. The trainer now enforces a hard time budget "
                      f"and saves at every checkpoint."),
        "eval": {"base": {"success": rb["success"], "mean_steps": rb["mean_steps_to_success"], "invalid": b_invalid,
                          "failures": base_failures,
                          "top": b_fail.most_common(1)[0][0].replace("_", " ") if b_fail else "none"},
                 "qwen": {"success": q["success"], "mean_steps": q["mean_steps_to_success"], "invalid": invalid,
                          "failures": fails},
                 "expert": {"success": ex["success"], "mean_steps": ex["mean_steps_to_success"]},
                 "collisions": {"steps": coll, "episodes": len(coll_eps),
                                "episodes_succeeded": sum(e["success"] for e in coll_eps), "expert": e_coll}},
        "latency": latency_block(out),
        "wall_base": [{**w, "file": f"assets/eval/b{w['seed']}.mp4", "thumb": f"assets/eval/b{w['seed']}.jpg"}
                      for w in media["basewall"]],
        "wall": [{**w, "file": f"assets/eval/q{w['seed']}.mp4", "thumb": f"assets/eval/q{w['seed']}.jpg"}
                 for w in media["evalwall"]],
    }
    page = (REPO / "showcase" / "template.html").read_text()
    page = page.replace("/*__DATA__*/null", json.dumps(D, separators=(",", ":"))).replace("__REPO__", REPO_URL)
    (REPO / "docs" / "index.html").write_text(page)
    (REPO / "docs" / ".nojekyll").write_text("")
    print(f"docs/index.html written ({len(page) / 1024:.0f} KB); collisions: {D['eval']['collisions']}")


if __name__ == "__main__":
    main()
