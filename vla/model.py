"""A small vision-language-action model.

  images (current + one earlier frame per camera) -> DINOv2-S patch tokens, pooled to 64 per image
  instruction -> frozen CLIP text tokens (computed once per instruction, never at control rate)
  proprio -> one token
  fusion: transformer encoder over all tokens
  action head: flow matching. A transformer decoder turns a noisy 16-tick action chunk into a
  velocity toward the clean chunk, cross-attending to the fused tokens; a few Euler steps at
  inference. The encoder runs once per inference, only the light decoder runs per flow step.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def sinusoid(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(1000.0) * torch.arange(half, device=t.device) / half)
    a = t[:, None].float() * freqs[None] * 100.0
    return torch.cat([a.sin(), a.cos()], dim=-1)


class TextEncoder(nn.Module):
    """Frozen CLIP text tower. Used offline / once per instruction."""

    def __init__(self, name: str):
        super().__init__()
        from transformers import CLIPTextModel, CLIPTokenizer

        self.tok = CLIPTokenizer.from_pretrained(name)
        self.model = CLIPTextModel.from_pretrained(name).eval().requires_grad_(False)
        self.dim = self.model.config.hidden_size

    @torch.no_grad()
    def forward(self, texts: list[str], max_len: int = 24):
        b = self.tok(texts, padding="max_length", max_length=max_len, truncation=True, return_tensors="pt")
        b = {k: v.to(self.model.device) for k, v in b.items()}
        h = self.model(**b).last_hidden_state
        return h.float(), b["attention_mask"].bool()


class VLA(nn.Module):
    def __init__(self, mcfg: dict, n_cams: int = 2, action_dim: int = 6, proprio_dim: int = 6, text_dim: int = 512):
        super().__init__()
        from transformers import AutoModel

        d = mcfg["d_model"]
        self.mcfg, self.d, self.chunk, self.action_dim = mcfg, d, mcfg["chunk"], action_dim
        self.n_views = 2 * n_cams                       # current + previous frame for each camera
        self.vision = AutoModel.from_pretrained(mcfg["vision"])
        vd = self.vision.config.hidden_size
        self.vision.requires_grad_(False)
        k = mcfg["vision_trainable_blocks"]
        if k > 0:
            for blk in self.vision.encoder.layer[-k:]:
                blk.requires_grad_(True)
            self.vision.layernorm.requires_grad_(True)
        self.pool = mcfg["pool"]
        self.register_buffer("img_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("img_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

        self.vis_proj = nn.Linear(vd, d)
        self.view_emb = nn.Parameter(torch.zeros(self.n_views, d))
        self.patch_pos = nn.Parameter(torch.zeros(64 * 4, d))     # sized lazily to the pooled grid
        self.text_proj = nn.Linear(text_dim, d)
        self.text_pos = nn.Parameter(torch.zeros(32, d))
        self.proprio_proj = nn.Sequential(nn.Linear(proprio_dim, d), nn.GELU(), nn.Linear(d, d))
        enc = nn.TransformerEncoderLayer(d, mcfg["n_heads"], 4 * d, dropout=0.0, activation="gelu",
                                         batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, mcfg["enc_layers"], enable_nested_tensor=False)
        self.enc_norm = nn.LayerNorm(d)

        self.act_in = nn.Linear(action_dim, d)
        self.act_pos = nn.Parameter(torch.zeros(self.chunk, d))
        self.time_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        dec = nn.TransformerDecoderLayer(d, mcfg["n_heads"], 4 * d, dropout=0.0, activation="gelu",
                                         batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec, mcfg["dec_layers"])
        self.act_out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, action_dim))
        for p in (self.view_emb, self.patch_pos, self.text_pos, self.act_pos):
            nn.init.normal_(p, std=0.02)
        nn.init.zeros_(self.act_out[1].weight)
        nn.init.zeros_(self.act_out[1].bias)

    # ------------------------------------------------------------------ encoders
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """images: [N, 3, H, W] float in [0, 1] -> [N, T, d] pooled, projected patch tokens."""
        x = (images - self.img_mean) / self.img_std
        h = self.vision(pixel_values=x).last_hidden_state[:, 1:]          # drop CLS
        n, t, c = h.shape
        g = int(math.isqrt(t))
        h = h.transpose(1, 2).reshape(n, c, g, g)
        h = F.avg_pool2d(h, self.pool).flatten(2).transpose(1, 2)        # [N, (g/pool)^2, c]
        return self.vis_proj(h) + self.patch_pos[: h.shape[1]]

    def fuse(self, vis: torch.Tensor, text: torch.Tensor, text_mask: torch.Tensor, proprio: torch.Tensor):
        """vis: [B, n_views, T, d] image tokens; text: [B, L, text_dim]; proprio: [B, P] (normalized)."""
        b = vis.shape[0]
        vis = (vis + self.view_emb[None, :, None]).flatten(1, 2)
        txt = self.text_proj(text) + self.text_pos[: text.shape[1]]
        pro = self.proprio_proj(proprio)[:, None]
        x = torch.cat([vis, txt, pro], dim=1)
        pad = torch.cat([torch.zeros(b, vis.shape[1], dtype=torch.bool, device=x.device), ~text_mask,
                         torch.zeros(b, 1, dtype=torch.bool, device=x.device)], dim=1)
        return self.enc_norm(self.encoder(x, src_key_padding_mask=pad)), pad

    def velocity(self, noisy: torch.Tensor, t: torch.Tensor, memory, pad) -> torch.Tensor:
        x = self.act_in(noisy) + self.act_pos + self.time_mlp(sinusoid(t, self.d))[:, None]
        return self.act_out(self.decoder(x, memory, memory_key_padding_mask=pad))

    # ------------------------------------------------------------------ training / inference
    def loss(self, images, text, text_mask, proprio, actions, weight=None):
        """images: [B, n_views, 3, H, W]; actions: [B, chunk, A] (normalized)."""
        b = images.shape[0]
        vis = self.encode_images(images.flatten(0, 1)).unflatten(0, (b, self.n_views))
        memory, pad = self.fuse(vis, text, text_mask, proprio)
        noise = torch.randn_like(actions)
        t = torch.sigmoid(torch.randn(b, device=actions.device))          # logit-normal over (0, 1)
        xt = (1 - t)[:, None, None] * noise + t[:, None, None] * actions
        v = self.velocity(xt, t, memory, pad)
        err = F.mse_loss(v.float(), actions - noise, reduction="none")
        if weight is not None:
            err = err * weight
        return err.mean()

    @torch.no_grad()
    def sample(self, vis, text, text_mask, proprio, steps: int | None = None, noise=None):
        """vis: [B, n_views, T, d] precomputed image tokens -> [B, chunk, A] normalized actions."""
        steps = steps or self.mcfg["flow_steps"]
        memory, pad = self.fuse(vis, text, text_mask, proprio)
        b = vis.shape[0]
        x = noise if noise is not None else torch.randn(b, self.chunk, self.action_dim, device=vis.device)
        x = x.to(memory.dtype)
        for i in range(steps):
            t = torch.full((b,), i / steps, device=vis.device)
            x = x + self.velocity(x, t, memory, pad) / steps
        return x.float()
