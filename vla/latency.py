"""Characterize action-generation latency of a trained VLA.

  .venv/bin/python -m vla.latency --ckpt outputs/vla/runs/m2_early/best.pt

Writes outputs/vla/latency.json with: cold start, per-component breakdown, precision / flow-step /
torch.compile sweeps, CPU-only timing, instruction-change cost, simulator cost, and closed-loop
latency distributions plus success for different flow steps and replan intervals.
All GPU timings bracket the work with torch.cuda.synchronize().
"""
from __future__ import annotations

import argparse
import time

from viz.viz import setup_gl


def pct(x):
    import numpy as np

    x = np.asarray(x, float)
    return {"n": int(len(x)), "mean": float(x.mean()), "min": float(x.min()), "p50": float(np.percentile(x, 50)),
            "p90": float(np.percentile(x, 90)), "p99": float(np.percentile(x, 99)),
            "p999": float(np.percentile(x, 99.9)), "max": float(x.max()), "std": float(x.std())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/vla/runs/m2_early/best.pt")
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--sweep-episodes", type=int, default=50)
    ap.add_argument("--out", default="outputs/vla/latency.json")
    ap.add_argument("--skip-compile", action="store_true")
    args = ap.parse_args()
    setup_gl("record")

    import mujoco
    import numpy as np
    import torch

    from env.common import REPO, write_json
    from vla.common import load_vla_config
    from vla.env import TickEnv
    from vla.policy import load_policy, rollout

    def sync():
        torch.cuda.synchronize()

    def timeit(fn, reps, warm=30, cuda=True):
        for _ in range(warm):
            fn()
        out = []
        for _ in range(reps):
            if cuda:
                sync()
            t0 = time.perf_counter()
            fn()
            if cuda:
                sync()
            out.append((time.perf_counter() - t0) * 1000)
        return out

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = load_vla_config()
    cfg["vla"] = v = ck["vla_cfg"]
    env = TickEnv(cfg, render=True)
    seeds = list(range(v["seeds"]["eval_start"] + 30000, v["seeds"]["eval_start"] + 30000 + args.episodes))
    res = {"checkpoint": args.ckpt, "hardware": {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                                                  "mujoco": mujoco.__version__},
           "control": {"hz": v["control_hz"], "tick_ms": 1000 / v["control_hz"], "chunk": v["model"]["chunk"],
                       "exec_horizon": v["model"]["exec_horizon"], "flow_steps": v["model"]["flow_steps"],
                       "history_offset": v["model"]["history_offset"], "image_size": v["image_size"],
                       "cameras": v["cameras"]}}

    # ---------------------------------------------------------------- cold start
    t0 = time.perf_counter()
    pol = load_policy(args.ckpt)
    sync()
    load_s = time.perf_counter() - t0
    obs = env.reset(seeds[0])
    t0 = time.perf_counter()
    pol.act(obs)
    sync()
    first_ms = (time.perf_counter() - t0) * 1000
    res["cold_start"] = {"load_s": load_s, "first_inference_ms": first_ms}
    model = pol.model
    res["params_M"] = {"total": sum(p.numel() for p in model.parameters()) / 1e6,
                       "vision": sum(p.numel() for p in model.vision.parameters()) / 1e6,
                       "encoder": sum(p.numel() for p in model.encoder.parameters()) / 1e6,
                       "decoder": sum(p.numel() for p in model.decoder.parameters()) / 1e6,
                       "text_encoder_offline": sum(p.numel() for p in pol.text_encoder.parameters()) / 1e6}
    print("cold start", res["cold_start"], flush=True)

    # ---------------------------------------------------------------- real observations to time on
    obs_list = []
    obs = env.reset(seeds[0])
    pol.reset(0)
    done = False
    while not done:
        obs_list.append(obs)
        obs, done, _ = env.step(pol.act(obs))
    obs_list = obs_list[:: max(1, len(obs_list) // 32)]
    pol.set_instruction(obs_list[0]["instruction"])
    k = [0]

    def next_obs():
        k[0] = (k[0] + 1) % len(obs_list)
        return obs_list[k[0]]

    # ---------------------------------------------------------------- component breakdown (default settings)
    def stage_times(amp, steps, reps):
        dt = torch.bfloat16
        st = {n: [] for n in ("upload", "vision", "fusion", "flow", "post", "total")}
        mark = [0.0]

        def lap(name):
            sync()
            now = time.perf_counter()
            st[name].append((now - mark[0]) * 1000)
            mark[0] = now

        for i in range(reps + 30):
            o = next_obs()
            sync()
            start = mark[0] = time.perf_counter()
            with torch.no_grad(), torch.autocast("cuda", dtype=dt, enabled=amp):
                x = pol._to_tensor(o["images"])
                q = torch.from_numpy(o["proprio"]).cuda().float()
                pro = ((q - pol.stats["proprio_mean"]) / pol.stats["proprio_std"])[None]
                lap("upload")
                tok = model.encode_images(x)
                lap("vision")
                vis = torch.cat([tok, tok], dim=0)[None]
                memory, pad = model.fuse(vis, pol.text, pol.text_mask, pro)
                lap("fusion")
                a = torch.zeros(1, model.chunk, model.action_dim, device="cuda", dtype=memory.dtype)
                for s in range(steps):
                    t = torch.full((1,), s / steps, device="cuda")
                    a = a + model.velocity(a, t, memory, pad) / steps
                lap("flow")
                a = a.float()[0] * pol.stats["action_std"] + pol.stats["action_mean"]
                a[:, :5] += q[:5]
                a = a.cpu().numpy()
                lap("post")
            st["total"].append((time.perf_counter() - start) * 1000)
        return {n: pct(x[30:]) for n, x in st.items()}

    res["breakdown"] = stage_times(True, v["model"]["flow_steps"], args.reps)
    print("breakdown p50", {n: round(s["p50"], 2) for n, s in res["breakdown"].items()}, flush=True)

    # ---------------------------------------------------------------- precision x flow steps
    sweep = {}
    for amp in (True, False):
        for steps in (1, 2, 4, 8):
            s = stage_times(amp, steps, args.reps // 2)
            sweep[f"{'bf16' if amp else 'fp32'}_f{steps}"] = {"total": s["total"], "flow": s["flow"], "vision": s["vision"]}
    res["precision_flow_sweep"] = sweep
    print("sweep p50", {n: round(s["total"]["p50"], 2) for n, s in sweep.items()}, flush=True)

    # ---------------------------------------------------------------- cost of encoding an uncached earlier frame
    x4 = torch.cat([pol._to_tensor(obs_list[0]["images"]), pol._to_tensor(obs_list[1]["images"])])
    x2 = x4[:2]

    def enc(x):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model.encode_images(x)

    res["vision_batch"] = {"2_images": pct(timeit(lambda: enc(x2), 300)), "4_images": pct(timeit(lambda: enc(x4), 300))}

    # ---------------------------------------------------------------- instruction change
    new_texts = [f"please place the {b} bowl on top of the {p} pad now #{i}" for i in range(40)
                 for b in ("red", "green") for p in ("blue", "yellow")]
    it = iter(new_texts)

    def new_instruction():
        pol.set_instruction(next(it))

    res["instruction_change"] = {"uncached_text_encode_ms": pct(timeit(new_instruction, 100, warm=10)),
                                 "note": "cached instructions cost a dictionary lookup; the next action chunk "
                                         "then costs one normal inference"}
    pol.set_instruction(obs_list[0]["instruction"])

    # ---------------------------------------------------------------- memory
    torch.cuda.reset_peak_memory_stats()
    for _ in range(20):
        pol.queue = []
        pol.act(next_obs())
    res["gpu_memory_MB"] = {"allocated": torch.cuda.memory_allocated() / 2**20,
                            "peak_allocated": torch.cuda.max_memory_allocated() / 2**20}

    # ---------------------------------------------------------------- simulator cost per control tick
    env.reset(seeds[1])
    a0 = env.qpos().copy()
    phys, rend = [], []
    for _ in range(300):
        t0 = time.perf_counter()
        for _ in range(env.n_sub):
            mujoco.mj_step(env.m, env.d)
        t1 = time.perf_counter()
        env.render()
        rend.append((time.perf_counter() - t1) * 1000)
        phys.append((t1 - t0) * 1000)
    res["simulator_per_tick_ms"] = {"physics": pct(phys), "render_two_cameras": pct(rend)}
    del a0

    # ---------------------------------------------------------------- closed loop
    def closed_loop(n, **kw):
        p = load_policy(args.ckpt, **kw)
        rollout(env, p, seeds[0])
        p.latencies_ms.clear()
        ok, fails, act_ms, ticks = 0, {}, [], 0
        for seed in seeds[:n]:
            obs = env.reset(seed)
            p.reset(seed)
            done = False
            while not done:
                t0 = time.perf_counter()
                a = p.act(obs)
                act_ms.append((time.perf_counter() - t0) * 1000)
                obs, done, info = env.step(a)
            ticks += env.stats.ticks
            ok += info["success"]
            if not info["success"]:
                fails[info["failure_type"]] = fails.get(info["failure_type"], 0) + 1
        lat = np.array(p.latencies_ms)
        act_ms = np.array(act_ms)
        return {"n": n, "success_rate": ok / n, "failures": fails, "exec_horizon": p.exec_horizon,
                "flow_steps": p.flow_steps, "inference_ms": pct(lat), "per_tick_ms": pct(act_ms),
                "mean_compute_per_tick_ms": float(act_ms.sum() / ticks),
                "duty_cycle": float(act_ms.sum() / (ticks * 1000 / v["control_hz"])),
                "raw": lat}

    base = closed_loop(args.episodes)
    raw = base.pop("raw")
    res["closed_loop_default"] = base
    res["closed_loop_default"]["inference_ms_samples"] = [round(float(x), 3) for x in raw]
    print("closed loop default", {k2: base[k2] for k2 in ("success_rate", "mean_compute_per_tick_ms")},
          {k2: round(base["inference_ms"][k2], 2) for k2 in ("p50", "p99", "max")}, flush=True)
    sw = {}
    for name, kw in [("f1", {"flow_steps": 1}), ("f2", {"flow_steps": 2}), ("f8", {"flow_steps": 8}),
                     ("h1", {"exec_horizon": 1}), ("h2", {"exec_horizon": 2}), ("h8", {"exec_horizon": 8}),
                     ("h16", {"exec_horizon": 16})]:
        r = closed_loop(args.sweep_episodes, **kw)
        r.pop("raw")
        sw[name] = r
        print(name, r["success_rate"], round(r["inference_ms"]["p50"], 2), round(r["mean_compute_per_tick_ms"], 2), flush=True)
    res["closed_loop_sweep"] = sw

    # ---------------------------------------------------------------- CPU only
    cpu = load_policy(args.ckpt, device="cpu", amp=False)
    cpu.set_instruction(obs_list[0]["instruction"])

    def cpu_step():
        cpu.queue = []
        cpu.act(next_obs())

    res["cpu_only"] = {"threads": torch.get_num_threads(), "inference_ms": pct(timeit(cpu_step, 60, warm=5, cuda=False))}
    print("cpu p50", round(res["cpu_only"]["inference_ms"]["p50"], 1), flush=True)

    # ---------------------------------------------------------------- torch.compile (optional, last: it mutates the model)
    if not args.skip_compile:
        try:
            t0 = time.perf_counter()
            model.vision = torch.compile(model.vision)
            model.encoder = torch.compile(model.encoder)
            model.decoder = torch.compile(model.decoder)

            def step():
                pol.queue = []
                pol.tokens.clear()
                pol.frames.clear()
                pol.act(next_obs())

            for _ in range(40):
                step()
            sync()
            compile_s = time.perf_counter() - t0
            res["torch_compile"] = {"compile_s": compile_s, "inference_ms": pct(timeit(step, args.reps // 2))}
            print("compiled p50", round(res["torch_compile"]["inference_ms"]["p50"], 2), flush=True)
        except Exception as e:  # noqa: BLE001
            res["torch_compile"] = {"error": repr(e)[:400]}
            print("compile failed", repr(e)[:200], flush=True)

    write_json(REPO / args.out, res)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
