"""Run-time wrapper: observation in, joint targets out, at control rate.

Keeps the image history and the current action chunk, replans every `exec_horizon` ticks, caches
image tokens so the "previous frame" is never encoded twice, and records inference latency.
The instruction can be changed at any tick (`set_instruction`), which forces an immediate replan:
this is the interface a slower reasoning layer would drive.
"""
from __future__ import annotations

import time
from collections import deque

import numpy as np
import torch

from vla.model import VLA, TextEncoder


class VLAPolicy:
    name = "vla"

    def __init__(self, model: VLA, stats: dict, cams, text_encoder: TextEncoder | None = None,
                 text_cache: dict | None = None, device="cuda", exec_horizon=None, flow_steps=None,
                 amp=True, deterministic=True):
        self.model, self.cams, self.device = model.to(device).eval(), list(cams), device
        m = model.mcfg
        self.hist = m["history_offset"]
        self.exec_horizon = exec_horizon or m["exec_horizon"]
        self.flow_steps = flow_steps or m["flow_steps"]
        self.stats = {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in stats.items()}
        self.text_encoder, self.text_cache = text_encoder, dict(text_cache or {})
        self.amp, self.deterministic = amp, deterministic
        self.latencies_ms: list[float] = []
        self.reset()

    def reset(self, seed: int = 0):
        self.tokens = deque(maxlen=self.hist + 1)      # per-tick image tokens (None until computed)
        self.frames = deque(maxlen=self.hist + 1)      # per-tick raw frames
        self.queue: list[np.ndarray] = []
        self.gen = torch.Generator(device=self.device).manual_seed(int(seed))
        self.instruction = None

    def set_instruction(self, text: str):
        if text == self.instruction:
            return
        if text not in self.text_cache:
            self.text_encoder.to(self.device)
            h, mask = self.text_encoder([text])
            self.text_cache[text] = (h[0].cpu(), mask[0].cpu())
        h, mask = self.text_cache[text]
        self.text, self.text_mask = h[None].to(self.device), mask[None].to(self.device)
        self.instruction = text
        self.queue = []                                # replan now

    def _to_tensor(self, frames: dict) -> torch.Tensor:
        x = np.stack([frames[c] for c in self.cams])
        return torch.from_numpy(x).to(self.device).permute(0, 3, 1, 2).float() / 255.0

    @torch.no_grad()
    def act(self, obs: dict) -> np.ndarray:
        self.set_instruction(obs["instruction"])
        self.frames.append(obs["images"])
        self.tokens.append(None)
        if not self.queue:
            if self.device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp and self.device == "cuda"):
                if self.tokens[-1] is None:
                    self.tokens[-1] = self.model.encode_images(self._to_tensor(self.frames[-1]))
                if self.tokens[0] is None:             # earlier frame was not an inference tick
                    self.tokens[0] = self.model.encode_images(self._to_tensor(self.frames[0]))
                vis = torch.cat([self.tokens[-1], self.tokens[0]], dim=0)[None]
                q = torch.from_numpy(obs["proprio"]).to(self.device).float()
                pro = ((q - self.stats["proprio_mean"]) / self.stats["proprio_std"])[None]
                noise = None
                if self.deterministic:
                    noise = torch.zeros(1, self.model.chunk, self.model.action_dim, device=self.device)
                a = self.model.sample(vis, self.text, self.text_mask, pro, self.flow_steps, noise)[0]
            a = a * self.stats["action_std"] + self.stats["action_mean"]
            a[:, :5] += q[:5]                          # arm targets are relative to current joints
            a = a.cpu().numpy()
            if self.device == "cuda":
                torch.cuda.synchronize()
            self.latencies_ms.append((time.perf_counter() - t0) * 1000)
            self.queue = list(a[: self.exec_horizon])
        return self.queue.pop(0)


def save_checkpoint(path, model: VLA, stats, cfg, text_cache, step, extra=None):
    torch.save({"model": {k: v for k, v in model.state_dict().items()}, "stats": stats, "vla_cfg": cfg["vla"],
                "text_cache": text_cache, "step": step, "extra": extra or {}}, path)


def load_policy(path, device="cuda", **kw) -> VLAPolicy:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    v = ck["vla_cfg"]
    model = VLA(v["model"], n_cams=len(v["cameras"]))
    model.load_state_dict(ck["model"])
    return VLAPolicy(model, ck["stats"], v["cameras"], TextEncoder(v["model"]["text"]), ck["text_cache"],
                     device=device, **kw)


def rollout(env, policy: VLAPolicy, seed: int, task=None, frames=None, instruction_fn=None):
    """One closed-loop episode. `instruction_fn(tick, env) -> str | None` can override the instruction."""
    obs = env.reset(seed, task)
    policy.reset(seed)
    done, info = False, {}
    while not done:
        if instruction_fn is not None:
            text = instruction_fn(env.tick, env)
            if text:
                obs["instruction"] = text
        a = policy.act(obs)
        obs, done, info = env.step(a)
        if frames is not None:
            frames.append(np.concatenate([obs["images"][c] for c in policy.cams], axis=1))
    return info
