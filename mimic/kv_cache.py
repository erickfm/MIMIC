"""Streaming inference with a KV cache for sliding-window relpos models (cfg.attn_window > 0).

A model trained with `--attn-window W` on long chunks gives every frame the same computation
whether it is run over a long sequence or one frame at a time: each layer's query sees only
that layer's last W keys, and relpos has no absolute position. So keeping each layer's last W
keys/values and running one new frame per step reproduces the full forward exactly, at ~1/W
of the per-frame cost of recomputing a W-frame window.

    cache = RelPosKVCache(model, batch_size=B)
    for frame in frames:              # frame: dict of (B, 1, ...) tensors, the encoder's inputs
        preds = cache.step(frame)     # same dict as model(frames)[:, -1:]
    cache.reset([3, 7])               # rows 3 and 7 start new episodes (empty history)

Each row keeps its own clock, so rows can start episodes at different times. An empty cache is
exactly a training chunk that starts at the first frame of a game (no history, no mock frames).

Only the mimic relpos transformer exists, so every window model is supported.
"""
from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn.functional as Fn

from .model import CausalSelfAttentionRelPos, FramePredictor


class RelPosKVCache:
    def __init__(self, model: FramePredictor, batch_size: int, dtype: torch.dtype = None):
        model = getattr(model, "_orig_mod", model)
        w = getattr(model.cfg, "attn_window", 0)
        assert w > 0, "KV cache needs a sliding-window model (cfg.attn_window > 0)"
        self.model, self.w, self.B = model, w, batch_size
        attn0 = model.blocks[0].self_attn
        assert isinstance(attn0, CausalSelfAttentionRelPos)
        p = next(model.parameters())
        H, hs = attn0.n_head, attn0.hs
        n = len(model.blocks)
        self.K = torch.zeros(n, batch_size, H, w, hs, device=p.device, dtype=dtype or p.dtype)
        self.V = torch.zeros_like(self.K)
        # Ring buffer: frame t lives in slot t % w. Everything that changes per step is a device
        # tensor, so a step can be captured once as a CUDA graph and replayed.
        self.t = torch.zeros(batch_size, dtype=torch.long, device=p.device)   # frames seen, per row
        self.slots = torch.arange(w, device=p.device)
        self.rows = torch.arange(batch_size, device=p.device)

    def reset(self, rows=None):
        """Empty the cache for `rows` (all rows when None): their next step is an episode start."""
        if rows is None:
            self.K.zero_(); self.V.zero_(); self.t.zero_()
            return
        r = torch.as_tensor(rows, device=self.t.device, dtype=torch.long)
        self.K[:, r] = 0; self.V[:, r] = 0; self.t[r] = 0

    @torch.no_grad()
    def step(self, frame: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        m, w = self.model, self.w
        x = m.encoder(frame)                                   # (B, 1, D)
        B, _, D = x.shape
        slot = self.t % w                                       # (B,) this frame's slot per row
        dist = (slot[:, None] - self.slots[None]) % w           # (B, w) each slot's distance back
        empty = (self.slots[None] > self.t[:, None])[:, None, None]   # (B,1,1,w) not yet written
        for li, blk in enumerate(m.blocks):
            a = blk.self_attn
            h = blk.ln_1(x)
            q, k, v = a.c_attn(h).split(D, dim=2)
            q = q.view(B, 1, a.n_head, a.hs).transpose(1, 2)    # (B, H, 1, hs)
            kv_dt = self.K.dtype
            self.K[li][self.rows, :, slot] = k.view(B, a.n_head, a.hs).to(kv_dt)
            self.V[li][self.rows, :, slot] = v.view(B, a.n_head, a.hs).to(kv_dt)
            # the training path scores distance d with Er[block_size-1-d]
            Er = a.Er[a.block_size - 1 - dist].to(q.dtype)      # (B, w, hs)
            att = (q @ self.K[li].transpose(-2, -1)
                   + torch.einsum("bhqd,bwd->bhqw", q, Er)) * (1.0 / math.sqrt(a.hs))
            att = att.masked_fill(empty, float("-inf"))
            y = Fn.softmax(att.float(), dim=-1).to(q.dtype) @ self.V[li]
            y = y.transpose(1, 2).reshape(B, 1, D)
            x = x + a.c_proj(y)
            if blk._use_swiglu:
                x = x + blk.mlp(blk.ln_2(x))
            else:
                x = x + blk.mlp.c_proj(Fn.gelu(blk.mlp.c_fc(blk.ln_2(x))))
        self.t += 1
        return m.heads(m.final_norm(x))
