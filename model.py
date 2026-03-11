#!/usr/bin/env python3
# model.py ─ FRAME next-frame predictor
# ------------------------------------
# 1. Hyper-parameter bundle
# 2. Causal self-attention & TransformerBlock
# 3. PredictionHeads
# 4. FramePredictor (Encoder ➜ Transformer ➜ Heads)
# ------------------------------------

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as Fn

# Source-of-truth enum maps
from cat_maps import (
    STAGE_MAP,
    CHARACTER_MAP,
    ACTION_MAP,
    PROJECTILE_TYPE_MAP,
)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Hyper-parameter bundle
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    # model size
    d_model: int         = 1024
    nhead: int           = 8
    num_layers: int      = 4
    dim_feedforward: int = 4096      # 4x expansion (was 2048)
    dropout: float       = 0.0

    # sequence length
    max_seq_len: int     = 60

    # fixed categorical vocab sizes
    num_stages: int       = len(STAGE_MAP)
    num_ports: int        = 4
    num_characters: int   = len(CHARACTER_MAP)
    num_actions: int      = len(ACTION_MAP)
    num_costumes: int     = 6
    num_c_dirs: int       = 5
    num_proj_types: int   = len(PROJECTILE_TYPE_MAP)
    num_proj_subtypes: int= 40

# ─────────────────────────────────────────────────────────────────────────────
# 2. Attention + Transformer block
# ─────────────────────────────────────────────────────────────────────────────
class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with causal masking via Flash/SDPA."""
    def __init__(self, d_model: int, nhead: int, dropout: float, max_seq_len: int):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        out = Fn.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block with causal self-attention."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.self_attn = CausalSelfAttention(cfg.d_model, cfg.nhead, cfg.dropout, cfg.max_seq_len)
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.drop1 = nn.Dropout(cfg.dropout)

        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.dim_feedforward),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.dim_feedforward, cfg.d_model),
        )
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.drop2 = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop1(self.self_attn(self.norm1(x)))
        x = x + self.drop2(self.ff(self.norm2(x)))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 3. Output heads
# ─────────────────────────────────────────────────────────────────────────────
class PredictionHeads(nn.Module):
    """
    Outputs per next-frame target (broadcasts over T for autoregressive training):
      - main_xy          raw regression        (MSE, clamp to [0,1] at inference)
      - L_val, R_val     raw regression        (MSE, clamp to [0,1] at inference)
      - c_dir_logits     5-way logits          (Focal CE)
      - btn_logits       12-way logits         (BCE)
    """
    def __init__(self, d_model: int, hidden: int = 256, btn_threshold: float = 0.5):
        super().__init__()
        self.btn_threshold = btn_threshold

        def build(out_dim: int):
            return nn.Sequential(
                nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, out_dim),
            )

        self.main_head  = build(2)
        self.L_head     = build(1)
        self.R_head     = build(1)
        self.cdir_head  = build(5)
        self.btn_head   = build(12)

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        return dict(
            main_xy=self.main_head(h),
            L_val=self.L_head(h),
            R_val=self.R_head(h),
            c_dir_logits=self.cdir_head(h),
            btn_logits=self.btn_head(h),
        )

    def threshold_buttons(self, btn_logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(btn_logits) >= self.btn_threshold


# ─────────────────────────────────────────────────────────────────────────────
# 4. Full model: Encoder → Pos-emb → N×Transformer → Heads
# ─────────────────────────────────────────────────────────────────────────────
class FramePredictor(nn.Module):
    """
    1. Encodes structured frames via FrameEncoder → (B, T, d_model)
    2. Adds learned positional embeddings
    3. Runs a stack of causal Transformer blocks
    4. Feeds all positions into PredictionHeads (autoregressive training)
    """
    def __init__(self, cfg: ModelConfig, encoder: Optional[nn.Module] = None):
        super().__init__()
        from frame_encoder import FrameEncoder

        self.cfg = cfg
        self.encoder = encoder or FrameEncoder(
            num_stages=cfg.num_stages,
            num_ports=cfg.num_ports,
            num_characters=cfg.num_characters,
            num_actions=cfg.num_actions,
            num_costumes=cfg.num_costumes,
            num_proj_types=cfg.num_proj_types,
            num_proj_subtypes=cfg.num_proj_subtypes,
            d_model=cfg.d_model,
            num_c_dirs=cfg.num_c_dirs,
        )

        self.pos_emb = nn.Parameter(torch.randn(1, cfg.max_seq_len, cfg.d_model) * 0.02)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.heads = PredictionHeads(cfg.d_model)

        self.apply(self._init_weights)
        residual_std = 0.02 / math.sqrt(2 * cfg.num_layers)
        for blk in self.blocks:
            nn.init.normal_(blk.self_attn.out_proj.weight, std=residual_std)
            nn.init.normal_(blk.ff[-1].weight, std=residual_std)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, frames: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = self.encoder(frames)                         # (B, T, d_model)
        T = x.size(1)
        x = x + self.pos_emb[:, :T]
        for blk in self.blocks:
            x = blk(x)
        x = self.final_norm(x)                           # (B, T, d_model)
        return self.heads(x)                             # all T positions
