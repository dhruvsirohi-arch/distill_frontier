"""Train the VLA by behaviour cloning (flow matching on action chunks).

  .venv/bin/python -m vla.train --run m1                    # full run
  .venv/bin/python -m vla.train --run overfit --overfit 10 --steps 1500   # sanity check
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

from viz.viz import setup_gl


def augment(images, shift, brightness):
    """images: uint8 [B, V, 3, H, W] on GPU -> float [0, 1] with random shift and brightness."""
    import torch
    import torch.nn.functional as F

    b, v, c, h, w = images.shape
    x = images.flatten(0, 1).float() / 255.0
    if shift > 0:
        x = F.pad(x, (shift,) * 4, mode="replicate")
        ox = torch.randint(0, 2 * shift + 1, (b * v,), device=x.device)
        oy = torch.randint(0, 2 * shift + 1, (b * v,), device=x.device)
        iy = (oy[:, None] + torch.arange(h, device=x.device))[:, None, :, None]     # [N, 1, H, 1]
        ix = (ox[:, None] + torch.arange(w, device=x.device))[:, None, None, :]     # [N, 1, 1, W]
        n = torch.arange(b * v, device=x.device)[:, None, None, None]
        ch = torch.arange(c, device=x.device)[None, :, None, None]
        x = x[n, ch, iy, ix]
    if brightness > 0:
        g = 1 + (torch.rand(b * v, 1, 1, 1, device=x.device) * 2 - 1) * brightness
        x = (x * g).clamp(0, 1)
    return x.unflatten(0, (b, v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="m1")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--overfit", type=int, default=0, help="train on N val episodes and roll out on their seeds")
    ap.add_argument("--episodes", type=int, default=0, help="use only the first N training episodes")
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--eval-episodes", type=int, default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-history", action="store_true", help="ablation: previous frame = current frame")
    args = ap.parse_args()
    setup_gl("record")

    import h5py
    import numpy as np
    import torch
    from torch.utils.tensorboard import SummaryWriter

    from env.common import REPO, set_seeds, write_json
    from vla.common import load_vla_config
    from vla.data import EpisodeDataset, compute_stats, list_episodes
    from vla.env import TEMPLATES, TickEnv
    from vla.model import VLA, TextEncoder
    from vla.policy import VLAPolicy, rollout, save_checkpoint

    cfg = load_vla_config(args.config)
    v, mc, tc = cfg["vla"], cfg["vla"]["model"], cfg["vla"]["train"]
    if args.no_history:
        mc["history_offset"] = 0
    steps = args.steps or tc["steps"]
    eval_every = args.eval_every or tc["eval_every"]
    n_eval = args.eval_episodes or tc["eval_episodes"]
    set_seeds(tc["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = "cuda"
    data_root = REPO / v["paths"]["data"]
    run_dir = REPO / v["paths"]["runs"] / args.run
    run_dir.mkdir(parents=True, exist_ok=True)

    val_files = list_episodes(data_root / "val")
    if args.overfit:
        train_files = val_files[: args.overfit]
        val_files = train_files
        eval_seeds = [int(h5py.File(f, "r").attrs["seed"]) for f in train_files][:n_eval]
    else:
        train_files = list_episodes(data_root / "train")
        if args.episodes:
            train_files = train_files[: args.episodes]
        eval_seeds = list(range(v["seeds"]["val_start"] + 5000, v["seeds"]["val_start"] + 5000 + n_eval))
    stats = compute_stats(train_files, mc["chunk"])

    # every instruction the task can produce, encoded once by the frozen text tower
    texts = sorted({t.format(bowl=b["name"], pad=p["name"]) for t in TEMPLATES for b in v["bowls"] for p in v["pads"]})
    text_enc = TextEncoder(mc["text"]).to(dev)
    th, tm = text_enc(texts)
    text_ids = {t: i for i, t in enumerate(texts)}
    text_cache = {t: (th[i].cpu(), tm[i].cpu()) for t, i in text_ids.items()}

    mk = lambda files: EpisodeDataset(files, v["cameras"], mc["chunk"], mc["history_offset"], stats, text_ids)
    train_ds, val_ds = mk(train_files), mk(val_files)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=tc["batch_size"], shuffle=True, drop_last=True,
                                         num_workers=tc["workers"], persistent_workers=True, pin_memory=True,
                                         prefetch_factor=4)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=tc["batch_size"], shuffle=True,
                                             num_workers=4, persistent_workers=True)
    print(f"train: {len(train_files)} episodes / {len(train_ds)} ticks; val: {len(val_files)} / {len(val_ds)}; "
          f"{steps} steps", flush=True)

    model = VLA(mc, n_cams=len(v["cameras"])).to(dev)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    vis_params = [p for p in model.vision.parameters() if p.requires_grad]
    vis_ids = {id(p) for p in vis_params}
    new_params = [p for p in model.parameters() if p.requires_grad and id(p) not in vis_ids]
    opt = torch.optim.AdamW([{"params": new_params, "lr": tc["lr"]}, {"params": vis_params, "lr": tc["lr_vision"]}],
                            weight_decay=tc["weight_decay"], betas=(0.9, 0.95))
    base_lrs = [g["lr"] for g in opt.param_groups]

    def lr_scale(s):
        if s < tc["warmup_steps"]:
            return (s + 1) / tc["warmup_steps"]
        return 0.5 * (1 + math.cos(math.pi * (s - tc["warmup_steps"]) / max(1, steps - tc["warmup_steps"]))) * 0.98 + 0.02

    def batch_loss(m, b, train):
        img = b["images"].to(dev, non_blocking=True)
        img = augment(img, tc["aug_shift"], tc["aug_brightness"]) if train else img.float() / 255.0
        tid = b["text_id"].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return m.loss(img, th[tid], tm[tid], b["proprio"].to(dev), b["action"].to(dev))

    env = TickEnv(cfg, render=True)
    tb = SummaryWriter(run_dir / "tb")
    history = {"train_loss": [], "val": []}

    def evaluate(step):
        ema.eval()
        with torch.no_grad():
            vl = float(np.mean([batch_loss(ema, b, False).item() for _, b in zip(range(20), val_loader)]))
        pol = VLAPolicy(ema, stats, v["cameras"], None, text_cache, device=dev)
        fails, ok, ticks = {}, 0, []
        for seed in eval_seeds:
            info = rollout(env, pol, seed)
            ok += info["success"]
            ticks.append(env.stats.ticks)
            if not info["success"]:
                fails[info["failure_type"]] = fails.get(info["failure_type"], 0) + 1
        lat = np.array(pol.latencies_ms[5:])
        rec = {"step": step, "val_loss": vl, "success": ok / len(eval_seeds), "n": len(eval_seeds), "failures": fails,
               "mean_ticks": float(np.mean(ticks)), "latency_ms_p50": float(np.percentile(lat, 50))}
        history["val"].append(rec)
        tb.add_scalar("val/loss", vl, step)
        tb.add_scalar("val/success", rec["success"], step)
        print(f"[eval] {json.dumps(rec)}", flush=True)
        save_checkpoint(run_dir / "last.pt", ema, stats, cfg, text_cache, step, rec)
        best = max(r["success"] for r in history["val"])
        if rec["success"] >= best:
            save_checkpoint(run_dir / "best.pt", ema, stats, cfg, text_cache, step, rec)
        write_json(run_dir / "history.json", history)

    step, t0, run_loss = 0, time.time(), 0.0
    model.train()
    while step < steps:
        for b in loader:
            for g, lr in zip(opt.param_groups, base_lrs):
                g["lr"] = lr * lr_scale(step)
            loss = batch_loss(model, b, True)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            opt.step()
            step += 1
            d = min(tc["ema"], (1 + step) / (10 + step))
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    if pm.requires_grad:
                        pe.lerp_(pm, 1 - d)
            run_loss += loss.item()
            if step % 50 == 0:
                history["train_loss"].append({"step": step, "loss": run_loss / 50})
                tb.add_scalar("train/loss", run_loss / 50, step)
                if step % 250 == 0:
                    print(f"step {step}/{steps} loss {run_loss / 50:.4f} lr {opt.param_groups[0]['lr']:.2e} "
                          f"{(time.time() - t0) / step:.3f} s/step", flush=True)
                run_loss = 0.0
            if step % eval_every == 0 or step == steps:
                evaluate(step)
                model.train()
            if step >= steps:
                break
    write_json(run_dir / "history.json", {**history, "minutes": (time.time() - t0) / 60,
                                          "params_M": sum(p.numel() for p in model.parameters()) / 1e6})
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
