"""Per-decision and per-token latency of the Qwen policy, measured the way a single robot would run it.

Batch size 1 (one robot), greedy decoding, thinking disabled: every output token is part of the action
JSON (+ the end-of-turn token). The decoding loop is written out token by token with a device sync after
each token, so we get:
  * prefill / time-to-first-token (TTFT): the prompt (~250 tokens) is processed and token 1 is chosen
  * per-token decode latency for every following token
  * total latency per decision (= time from observation to complete, parseable action)
The manual loop is checked against model.generate (identical tokens) on the first decisions.

  .venv/bin/python -m eval.latency                 # 100 held-out eval seeds
  .venv/bin/python -m eval.latency --n 10          # quick run

Writes outputs/latency.json (summary) and outputs/latency_decisions.jsonl (every decision, every token).
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time

import numpy as np
import torch

from env.common import REPO, get_logger, load_config, set_seeds, write_json
from eval.qwen_policy import QwenPolicy
from train.prompt import parse_action, prompt_text


def _sync(dev):
    if dev.type == "mps":
        torch.mps.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()


class TimedQwenPolicy(QwenPolicy):
    """QwenPolicy.act with an explicit greedy loop and per-token timing (batch size 1)."""

    def __init__(self, *a, check_first=40, **k):
        super().__init__(*a, **k)
        self.records: list[dict] = []
        self.check_first = check_first
        self.checked = self.matched = 0
        self.ctx: dict = {}

    @torch.no_grad()
    def decode(self, prompt: str):
        m, dev = self.model, self.model.device
        ids = self.tok(prompt, return_tensors="pt").input_ids.to(dev)
        stop = {self.im_end, self.tok.eos_token_id}
        toks, times = [], []
        _sync(dev)
        t0 = time.perf_counter()
        out = m(input_ids=ids, use_cache=True)
        nxt = int(out.logits[0, -1].argmax())  # .item() forces the device sync
        t = time.perf_counter()
        toks.append(nxt)
        times.append(t - t0)
        past = out.past_key_values
        while nxt not in stop and len(toks) < self.max_new_tokens:
            t1 = time.perf_counter()
            out = m(input_ids=torch.tensor([[nxt]], device=dev), past_key_values=past, use_cache=True)
            nxt = int(out.logits[0, -1].argmax())
            times.append(time.perf_counter() - t1)
            toks.append(nxt)
            past = out.past_key_values
        return ids.shape[1], toks, times, time.perf_counter() - t0

    @torch.no_grad()
    def act(self, obs, history):
        prompt = prompt_text(self.tok, obs, history)
        n_prompt, toks, times, total = self.decode(prompt)
        text = self.tok.decode(toks, skip_special_tokens=True)
        action = parse_action(text)
        if self.checked < self.check_first:  # the timing loop must produce exactly what generate() produces
            ref = super().act(obs, history)[1]
            self.checked += 1
            self.matched += int(ref == text.strip())
        self.records.append({
            **self.ctx, "step": obs["step"], "action": action["action"] if action else "invalid",
            "prompt_tokens": n_prompt, "output_tokens": len(toks),
            "tokens": [self.tok.decode([t]) for t in toks],
            "ttft_ms": round(1000 * times[0], 2), "token_ms": [round(1000 * x, 2) for x in times],
            "total_ms": round(1000 * total, 2)})
        return action, text.strip()


def pct(xs, q):
    return round(float(np.percentile(xs, q)), 1)


def stats(xs):
    return {"mean": round(statistics.fmean(xs), 1), "p50": pct(xs, 50), "p90": pct(xs, 90), "p99": pct(xs, 99),
            "min": round(min(xs), 1), "max": round(max(xs), 1), "n": len(xs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--model", default=None, help="merged model dir (default: paths.merged_dir)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    log = get_logger("latency", cfg)
    set_seeds(cfg["seeds"]["global"])
    from env.rollout import run_episode
    from env.so101_env import BowlEnv

    model_path = args.model or str(REPO / cfg["paths"]["merged_dir"])
    base = QwenPolicy.load(model_path, None, cfg["eval"]["max_new_tokens"], name="qwen")
    pol = TimedQwenPolicy(base.model, base.tok, cfg["eval"]["max_new_tokens"], name="qwen")
    dev = pol.model.device
    log.info(f"device {dev}, dtype {pol.model.dtype}, batch size 1, greedy, thinking off")

    # warm-up (kernel compilation / allocator) on a training seed; not recorded
    env = BowlEnv(cfg)
    run_episode(env, pol, 1, episode_id="warmup", history_len=cfg["train"]["history_len"])
    pol.records.clear()
    pol.checked = pol.matched = 0

    seeds = list(range(cfg["seeds"]["eval_start"], cfg["seeds"]["eval_start"] + args.n))
    eps = []
    t_all = time.time()
    for s in seeds:
        pol.ctx = {"seed": s}
        ep = run_episode(env, pol, s, episode_id=f"lat_{s}", history_len=cfg["train"]["history_len"])
        eps.append({"seed": s, "success": ep["success"], "n_steps": ep["n_steps"],
                    "sim_seconds": round(env.d.time, 2)})
        if len(eps) % 10 == 0:
            recs = pol.records
            log.info(f"{len(eps)}/{len(seeds)} episodes, {len(recs)} decisions, "
                     f"p50 {pct([r['total_ms'] for r in recs], 50)} ms/decision")

    R = pol.records
    total = [r["total_ms"] for r in R]
    ttft = [r["ttft_ms"] for r in R]
    decode = [x for r in R for x in r["token_ms"][1:]]
    by_type = {}
    for a in sorted({r["action"] for r in R}):
        rr = [r for r in R if r["action"] == a]
        by_type[a] = {"n": len(rr), "output_tokens": round(statistics.fmean(r["output_tokens"] for r in rr), 2),
                      "total_ms": stats([r["total_ms"] for r in rr])}
    max_len = max(r["output_tokens"] for r in R)
    by_pos = []
    for i in range(max_len):
        xs = [r["token_ms"][i] for r in R if len(r["token_ms"]) > i]
        by_pos.append({"index": i + 1, "n": len(xs), "mean_ms": round(statistics.fmean(xs), 2),
                       "p90_ms": pct(xs, 90)})
    per_ep = {}
    for r in R:
        per_ep[r["seed"]] = per_ep.get(r["seed"], 0.0) + r["total_ms"]
    ep_model_s = [v / 1000 for v in per_ep.values()]
    summary = {
        "setup": {"model": cfg["train"]["base_model"] + " + LoRA (merged)", "device": str(dev),
                  "dtype": str(pol.model.dtype).replace("torch.", ""), "batch_size": 1, "decoding": "greedy",
                  "thinking": "off (no reasoning tokens; every output token is part of the action)",
                  "machine": f"{platform.machine()} {platform.system()} {platform.mac_ver()[0]}",
                  "torch": torch.__version__, "seeds": [seeds[0], seeds[-1]],
                  "generate_equivalence": f"{pol.matched}/{pol.checked} decisions identical to model.generate"},
        "decisions": len(R),
        "prompt_tokens": stats([r["prompt_tokens"] for r in R]),
        "output_tokens_per_action": stats([r["output_tokens"] for r in R]),
        "decision_ms": stats(total),
        "ttft_ms": stats(ttft),
        "decode_ms_per_token": stats(decode),
        "decode_tokens_per_s": round(1000 / statistics.fmean(decode), 1),
        "by_action": by_type,
        "by_token_position": by_pos,
        "per_episode_model_seconds": stats(ep_model_s),
        "per_episode_sim_seconds": stats([e["sim_seconds"] for e in eps]),
        "episodes": {"n": len(eps), "success": sum(e["success"] for e in eps)},
        "wall_time_s": round(time.time() - t_all, 1),
    }
    write_json(REPO / "outputs" / "latency.json", summary)
    with open(REPO / "outputs" / "latency_decisions.jsonl", "w") as f:
        for r in R:
            f.write(json.dumps(r) + "\n")
    d, t, k = summary["decision_ms"], summary["ttft_ms"], summary["decode_ms_per_token"]
    print("\n" + "=" * 64)
    print(f"Qwen policy latency, batch 1, {dev}, {len(R)} decisions over {len(eps)} held-out episodes")
    print("-" * 64)
    print(f"{'':28}{'mean':>8}{'p50':>8}{'p90':>8}{'p99':>8}")
    for name, s in (("per decision (ms)", d), ("time to first token (ms)", t), ("per decode token (ms)", k)):
        print(f"{name:28}{s['mean']:>8}{s['p50']:>8}{s['p90']:>8}{s['p99']:>8}")
    print(f"output tokens per action: mean {summary['output_tokens_per_action']['mean']}, "
          f"prompt tokens: mean {summary['prompt_tokens']['mean']}")
    for a, v in by_type.items():
        print(f"  {a:14} n={v['n']:4}  tokens {v['output_tokens']:5}  p50 {v['total_ms']['p50']} ms")
    print(f"model time per episode: p50 {summary['per_episode_model_seconds']['p50']} s "
          f"(sim time p50 {summary['per_episode_sim_seconds']['p50']} s)")
    print(f"{summary['setup']['generate_equivalence']}; success at batch 1: {summary['episodes']['success']}/{len(eps)}")
    print("=" * 64)


if __name__ == "__main__":
    main()
