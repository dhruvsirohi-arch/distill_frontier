"""LoRA SFT of Qwen on curated expert data, with a hard wall-clock budget and in-sim checkpoints.

- chat format from train/prompt.py (system + user[observation, last 2 actions] -> assistant[action JSON])
- loss on assistant tokens only, per-example loss weights from curation (close_gripper/done upweighted)
- memory: lm_head is applied only at assistant positions; gradient checkpointing on
- budget: after `probe_steps`, estimate total runtime incl. checkpoint episodes; if it exceeds
  `budget_minutes`, cap the number of optimizer steps (i.e. reduce epochs/data) and log it loudly
- checkpoints at `ckpt_fractions`: run 5 fixed held-out episodes with the current adapter, record
  them to outputs/videos/train_ckpt_<step>/ (or stream), log success + val loss

  .venv/bin/python -m train.sft                 # full run
  .venv/bin/python -m train.sft --benchmark 10  # time 10 steps only
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from viz.viz import add_viz_args, resolve_mode, setup_gl  # noqa: E402  (sets GL backend before mujoco import)
from env.common import REPO, get_logger, load_config, set_seeds, write_json  # noqa: E402


def load_rows(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def build_examples(rows, tok, max_len):
    from train.prompt import completion_text, prompt_text

    ex, n_trunc = [], 0
    for r in rows:
        p = tok(prompt_text(tok, r["observation"], r["history"]), add_special_tokens=False)["input_ids"]
        c = tok(completion_text(tok, r["action"]), add_special_tokens=False)["input_ids"]
        ids = p + c
        if len(ids) > max_len:  # keep the completion, trim the prompt head (system prompt) - logged
            n_trunc += 1
            ids = ids[-max_len:]
            p = ids[: len(ids) - len(c)]
        ex.append({"ids": ids, "n_prompt": len(p), "w": float(r["loss_weight"]), "type": r["action_type"]})
    return ex, n_trunc


def collate(batch, pad_id, device):
    import torch

    L = max(len(b["ids"]) for b in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    pos_b, pos_t, tgt, owner = [], [], [], []
    for i, b in enumerate(batch):
        n = len(b["ids"])
        ids[i, :n] = torch.tensor(b["ids"])
        att[i, :n] = 1
        for t in range(b["n_prompt"], n):  # predict token t from position t-1
            pos_b.append(i)
            pos_t.append(t - 1)
            tgt.append(b["ids"][t])
            owner.append(i)
    w = torch.tensor([b["w"] for b in batch], dtype=torch.float32)
    return {k: v.to(device) for k, v in {
        "ids": ids, "att": att, "pos_b": torch.tensor(pos_b), "pos_t": torch.tensor(pos_t),
        "tgt": torch.tensor(tgt), "owner": torch.tensor(owner), "w": w}.items()}


def weighted_loss(model, batch, use_weights=True):
    """Mean CE over each example's assistant tokens, weighted per example, averaged over the batch."""
    import torch
    import torch.nn.functional as F

    core = model.get_base_model()  # Qwen*ForCausalLM with LoRA layers injected
    h = core.model(input_ids=batch["ids"], attention_mask=batch["att"]).last_hidden_state
    logits = core.lm_head(h[batch["pos_b"], batch["pos_t"]]).float()
    tok_loss = F.cross_entropy(logits, batch["tgt"], reduction="none")
    B = batch["w"].shape[0]
    per_ex = torch.zeros(B, device=tok_loss.device).index_add_(0, batch["owner"], tok_loss)
    cnt = torch.zeros(B, device=tok_loss.device).index_add_(0, batch["owner"], torch.ones_like(tok_loss))
    per_ex = per_ex / cnt.clamp(min=1)
    w = batch["w"] if use_weights else torch.ones_like(batch["w"])
    return (per_ex * w).sum() / B, per_ex.mean()


def evaluate_val(model, examples, bs, pad_id, device, limit=None):
    import torch

    model.eval()
    ex = examples[:limit] if limit else examples
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(ex), bs):
            b = collate(ex[i:i + bs], pad_id, device)
            _, unweighted = weighted_loss(model, b, use_weights=False)
            tot += float(unweighted) * len(ex[i:i + bs])
            n += len(ex[i:i + bs])
    model.train()
    return tot / max(1, n)


def ckpt_episodes(cfg, model, tok, step, mode, log):
    """Run the fixed held-out checkpoint episodes with the current adapter; record/stream them."""
    from env.rollout import run_episodes_lockstep
    from env.so101_env import BowlEnv
    from eval.qwen_policy import QwenPolicy
    from viz.viz import RecordViz, make_viz

    t0 = time.time()
    seeds = cfg["seeds"]["ckpt_eval_seeds"][: cfg["train"]["ckpt_episodes"]]
    stage = f"train_ckpt_{step}"
    envs = []
    for s in seeds:
        if mode == "stream":
            v = None  # one shared stream below would interleave frames; record + stream first env
        else:
            v = RecordViz(cfg, stage, out_root=str(REPO / "outputs" / "videos")) if mode != "none" else None
        envs.append(BowlEnv(cfg, viz=v))
    if mode == "stream":
        envs[0].viz = make_viz("stream", cfg)
        for e in envs[1:]:
            e.viz = RecordViz(cfg, stage, out_root=str(REPO / "outputs" / "videos"))
    model.eval()
    pol = QwenPolicy(model, tok, cfg["eval"]["max_new_tokens"], name=f"qwen@{step}")
    eps = run_episodes_lockstep(envs, pol, seeds, [f"ckpt{step}_seed{s}" for s in seeds],
                                cfg["train"]["history_len"])
    tok.padding_side = "right"
    model.train()
    for e in envs:
        if e.viz is not None:
            e.viz.close()
    k = sum(e["success"] for e in eps)
    res = {"step": step, "success": k, "n": len(eps), "failures": [e["failure_type"] for e in eps],
           "invalid_steps": sum(e["metrics"]["n_invalid"] for e in eps), "seconds": round(time.time() - t0, 1)}
    log.info(f"[ckpt {step}] held-out success {k}/{len(eps)} failures={res['failures']} "
             f"invalid_steps={res['invalid_steps']} ({res['seconds']}s) videos -> outputs/videos/{stage}/")
    return res


def main():
    ap = add_viz_args(argparse.ArgumentParser())
    ap.add_argument("--config", default=None)
    ap.add_argument("--benchmark", type=int, default=0, help="only time this many steps")
    args = ap.parse_args()
    cfg = load_config(args.config)
    mode = resolve_mode(args, cfg)
    if mode == "none":
        mode = "record"  # checkpoints are always recorded (that is the point of them)
    if mode == "live":
        mode = "stream"  # the passive viewer would block training on macOS; use stream instead
    setup_gl(mode)  # mode may have changed above; mujoco is not imported yet (EGL on headless Linux)
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tc = cfg["train"]
    log = get_logger("sft", cfg)
    set_seeds(tc["seed"])
    T0 = time.time()
    budget_s = tc["budget_minutes"] * 60
    from eval.qwen_policy import pick_device

    device = pick_device()
    base = tc["base_model"]
    try:
        tok = AutoTokenizer.from_pretrained(base)
    except Exception as e:  # noqa: BLE001
        log.warning(f"{base} unavailable ({e}); falling back to {tc['fallback_model']}")
        base = tc["fallback_model"]
        tok = AutoTokenizer.from_pretrained(base)
    tok.padding_side = "right"
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16).to(device)
    if tc.get("grad_checkpointing", False):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=tc["lora_r"], lora_alpha=tc["lora_alpha"],
                                             lora_dropout=tc["lora_dropout"], target_modules=tc["target_modules"],
                                             task_type="CAUSAL_LM"))
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"model {base} on {device}, bf16, LoRA r={tc['lora_r']} trainable params {n_tr/1e6:.1f}M")

    cdir = REPO / cfg["paths"]["curated_dir"]
    train_ex, tr_trunc = build_examples(load_rows(cdir / "train.jsonl"), tok, tc["max_seq_len"])
    val_ex, va_trunc = build_examples(load_rows(cdir / "val.jsonl"), tok, tc["max_seq_len"])
    lens = [len(e["ids"]) for e in train_ex]
    log.info(f"train examples {len(train_ex)} (truncated {tr_trunc}), val {len(val_ex)} (truncated {va_trunc}); "
             f"tokens/example mean {sum(lens)/len(lens):.0f} max {max(lens)}")

    bs = tc["batch_size"]
    steps_per_epoch = math.ceil(len(train_ex) / bs)
    planned = steps_per_epoch * tc["epochs"]
    total = {"steps": planned}
    rng = random.Random(tc["seed"])
    order = []
    for _ in range(tc["epochs"]):
        idx = list(range(len(train_ex)))
        rng.shuffle(idx)
        order += [idx[i:i + bs] for i in range(0, len(idx), bs)]

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=tc["lr"], weight_decay=0.0)
    warm = max(1, int(tc["warmup_ratio"] * planned))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, 1 - s / max(1, total["steps"])))

    history = {"train_loss": [], "val_loss": [], "ckpt": [], "budget_events": []}
    sync = (torch.mps.synchronize if device == "mps" else torch.cuda.synchronize if device == "cuda" else (lambda: None))

    # baseline checkpoint (step 0) also measures checkpoint cost for the budget estimate
    ckpt_cost = 0.0
    if not args.benchmark:
        r = ckpt_episodes(cfg, model, tok, 0, mode, log)
        history["ckpt"].append(r)
        ckpt_cost = r["seconds"]
    ckpt_steps = sorted({max(1, int(round(f * planned))) for f in tc["ckpt_fractions"]})
    model.train()
    t_train = time.time()
    t_probe = None
    step = 0
    run_loss = 0.0
    stop = {"flag": False, "why": ""}
    import signal
    def _graceful(signum, frame):  # Ctrl-C / SIGTERM: finish the current step, then save adapter + merged
        stop["flag"], stop["why"] = True, f"signal {signum}"
    signal.signal(signal.SIGINT, _graceful)
    signal.signal(signal.SIGTERM, _graceful)
    while step < total["steps"] and not stop["flag"]:
        batch = collate([train_ex[i] for i in order[step]], pad_id, device)
        loss, raw = weighted_loss(model, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        run_loss += float(raw.detach())
        if step % 10 == 0:
            history["train_loss"].append({"step": step, "loss": run_loss / 10})
            log.info(f"step {step}/{total['steps']} loss {run_loss/10:.4f} lr {sched.get_last_lr()[0]:.2e} "
                     f"elapsed {(time.time()-T0)/60:.1f} min")
            run_loss = 0.0
        if step == 2:
            sync()
            t_probe = time.time()  # exclude warm-up steps from the rate estimate
        if step == tc["probe_steps"] + 2 or (args.benchmark and step == args.benchmark):
            sync()
            sec_per_step = (time.time() - t_probe) / (step - 2)
            n_ck = sum(1 for s in ckpt_steps if s > step)
            overhead = 180.0  # val loss passes + adapter save + merge/save, measured ~2-3 min
            est = (time.time() - T0) + (total["steps"] - step) * sec_per_step + n_ck * (ckpt_cost or 60) + overhead
            msg = (f"BUDGET: {sec_per_step:.2f}s/step, {total['steps']} steps planned -> estimated total "
                   f"{est/60:.1f} min (budget {tc['budget_minutes']} min, ckpt cost {ckpt_cost:.0f}s x {n_ck})")
            log.info(msg)
            history["budget_events"].append(msg)
            if args.benchmark:
                write_json(REPO / "outputs" / "logs" / "sft_benchmark.json",
                           {"sec_per_step": sec_per_step, "planned_steps": planned, "estimated_min": est / 60,
                            "batch_size": bs, "device": device})
                return
            if est > budget_s:
                spare = budget_s - (time.time() - T0) - n_ck * (ckpt_cost or 60) - overhead
                new_total = max(step + 10, int(spare / sec_per_step) + step)
                msg = (f"BUDGET EXCEEDED: reducing training from {total['steps']} to {new_total} steps "
                       f"({new_total/steps_per_epoch:.2f} epochs instead of {tc['epochs']}) to stay under "
                       f"{tc['budget_minutes']} min")
                log.warning(msg)
                history["budget_events"].append(msg)
                total["steps"] = new_total
                ckpt_steps = sorted({max(step + 1, int(round(f * new_total))) for f in tc["ckpt_fractions"]})
        # hard wall-clock guard (the step-20 estimate can be wrong if the machine starts swapping)
        if not args.benchmark and step % 10 == 0 and (time.time() - T0) + 180.0 > budget_s:
            stop["flag"], stop["why"] = True, "wall-clock budget"
        if stop["flag"]:
            msg = (f"STOPPED EARLY at step {step}/{total['steps']} ({step/steps_per_epoch:.2f} epochs): "
                   f"{stop['why']}, elapsed {(time.time()-T0)/60:.1f} min")
            log.warning(msg)
            history["budget_events"].append(msg)
            total["steps"] = step
            break
        if step in ckpt_steps:
            vl = evaluate_val(model, val_ex, 16, pad_id, device, limit=256 if step != total["steps"] else None)
            history["val_loss"].append({"step": step, "val_loss": vl})
            log.info(f"[ckpt {step}] val loss {vl:.4f}")
            history["ckpt"].append(ckpt_episodes(cfg, model, tok, step, mode, log))
            model.save_pretrained(REPO / cfg["paths"]["adapter_dir"])  # survive a crash/kill
            model.train()
    train_wall = time.time() - t_train

    if total["steps"] not in ckpt_steps:
        vl = evaluate_val(model, val_ex, 16, pad_id, device)
        history["val_loss"].append({"step": total["steps"], "val_loss": vl})
    adir, mdir = REPO / cfg["paths"]["adapter_dir"], REPO / cfg["paths"]["merged_dir"]
    model.save_pretrained(adir)
    tok.save_pretrained(adir)
    merged = model.merge_and_unload()
    merged.config.use_cache = True
    merged.save_pretrained(mdir)
    tok.save_pretrained(mdir)
    total_wall = time.time() - T0
    summary = {"base_model": base, "device": device, "train_examples": len(train_ex), "val_examples": len(val_ex),
               "planned_steps": planned, "trained_steps": total["steps"],
               "epochs_effective": total["steps"] / steps_per_epoch, "batch_size": bs,
               "train_loop_minutes": round(train_wall / 60, 2), "total_minutes": round(total_wall / 60, 2),
               "final_val_loss": history["val_loss"][-1]["val_loss"] if history["val_loss"] else None,
               "adapter_dir": str(adir), "merged_dir": str(mdir), **history}
    write_json(REPO / "outputs" / "train_summary.json", summary)
    log.info(f"done: total wall-clock {total_wall/60:.1f} min (train loop {train_wall/60:.1f} min), "
             f"final val loss {summary['final_val_loss']:.4f}; adapter -> {adir}, merged -> {mdir}")


if __name__ == "__main__":
    main()
