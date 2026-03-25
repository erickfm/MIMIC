#!/usr/bin/env python3
# model.py ─ MIMIC next-frame predictor
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
from .cat_maps import (
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
    # temporal backbone
    d_model: int         = 1024
    nhead: int           = 8
    num_layers: int      = 4
    dim_feedforward: int = 4096
    dropout: float       = 0.1

    # frame encoder
    encoder_type: str    = "hybrid16"
    d_intra: int         = 256
    encoder_nlayers: int = 2
    k_query: int         = 1
    scaled_emb: bool     = False

    # sequence length
    max_seq_len: int     = 60

    # positional encoding
    pos_enc: str         = "rope"

    # attention variant
    attn_variant: str    = "standard"
    n_kv_heads: int      = 0

    # loss / output configuration
    stick_loss: str      = "clusters"
    btn_loss: str        = "focal"
    no_opp_inputs: bool  = True
    no_self_inputs: bool = True

    # prediction heads
    head_hidden: int            = 256
    n_stick_clusters: int       = 63
    n_shoulder_bins: int        = 4
    autoregressive_heads: bool  = True

    # fixed categorical vocab sizes
    num_stages: int       = len(STAGE_MAP)
    num_ports: int        = 4
    num_characters: int   = len(CHARACTER_MAP)
    num_actions: int      = len(ACTION_MAP)
    num_costumes: int     = 6
    num_c_dirs: int       = 5
    num_proj_types: int   = len(PROJECTILE_TYPE_MAP)
    num_proj_subtypes: int= 40


MODEL_PRESETS = {
    "tiny":         dict(d_model=256,  nhead=4,  num_layers=4, dim_feedforward=1024),
    "small":        dict(d_model=512,  nhead=8,  num_layers=4, dim_feedforward=2048),
    "medium":       dict(d_model=768,  nhead=8,  num_layers=4, dim_feedforward=3072),
    "base":         dict(d_model=1024, nhead=8,  num_layers=4, dim_feedforward=4096),
    "shallow":      dict(d_model=1024, nhead=8,  num_layers=2, dim_feedforward=4096),
    "deep":         dict(d_model=512,  nhead=8,  num_layers=8, dim_feedforward=2048),
    "wide-shallow": dict(d_model=1536, nhead=12, num_layers=2, dim_feedforward=6144),
    "xlarge":       dict(d_model=1024, nhead=8,  num_layers=8, dim_feedforward=4096),
    "xxlarge":      dict(d_model=1536, nhead=12, num_layers=8, dim_feedforward=6144),
    "huge":         dict(d_model=2048, nhead=16, num_layers=12, dim_feedforward=8192,
                         d_intra=512, head_hidden=512),
    "giant":        dict(d_model=3072, nhead=24, num_layers=12, dim_feedforward=12288,
                         d_intra=512, head_hidden=768),
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. Positional encoding helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sinusoidal_embeddings(max_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(max_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)


def _apply_rope(q: torch.Tensor, k: torch.Tensor, head_dim: int) -> tuple:
    """Apply rotary position embeddings to q, k of shape (B, nhead, T, head_dim)."""
    T = q.size(2)
    device = q.device
    half = head_dim // 2
    theta = 1.0 / (10000.0 ** (torch.arange(0, half, device=device).float() / half))
    pos = torch.arange(T, device=device).float()
    freqs = torch.outer(pos, theta)
    cos_f = freqs.cos().unsqueeze(0).unsqueeze(0)
    sin_f = freqs.sin().unsqueeze(0).unsqueeze(0)

    def rotate(x):
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * cos_f - x2 * sin_f, x2 * cos_f + x1 * sin_f], dim=-1)

    return rotate(q), rotate(k)


def _alibi_slopes(nhead: int) -> torch.Tensor:
    closest_pow2 = 2 ** math.floor(math.log2(nhead))
    base = 2.0 ** (-(2.0 ** -(math.log2(closest_pow2) - 3)))
    slopes = torch.pow(base, torch.arange(1, closest_pow2 + 1).float())
    if closest_pow2 < nhead:
        extra_base = 2.0 ** (-(2.0 ** -(math.log2(2 * closest_pow2) - 3)))
        extra = torch.pow(extra_base, torch.arange(1, 2 * (nhead - closest_pow2) + 1, 2).float())
        slopes = torch.cat([slopes, extra])
    return slopes


def _sliding_window_mask(T: int, window: int, device: torch.device) -> torch.Tensor:
    """Causal mask with sliding window: attend to max `window` past positions."""
    row = torch.arange(T, device=device).unsqueeze(1)
    col = torch.arange(T, device=device).unsqueeze(0)
    mask = (col <= row) & (row - col < window)
    return mask.float().log()  # 0 for valid, -inf for masked


# ─────────────────────────────────────────────────────────────────────────────
# 3. Attention + Transformer block
# ─────────────────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with support for RoPE, ALiBi, GQA, sliding window."""

    def __init__(self, d_model: int, nhead: int, dropout: float, max_seq_len: int,
                 pos_enc: str = "learned", attn_variant: str = "standard",
                 n_kv_heads: int = 0, window_size: int = 30):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.d_model = d_model
        self.dropout = dropout
        self.pos_enc = pos_enc
        self.attn_variant = attn_variant
        self.window_size = window_size

        self.n_kv_heads = n_kv_heads if (n_kv_heads and n_kv_heads < nhead) else nhead
        assert nhead % self.n_kv_heads == 0, f"nhead ({nhead}) must be divisible by n_kv_heads ({self.n_kv_heads})"
        self.kv_group_size = nhead // self.n_kv_heads

        q_dim = nhead * self.head_dim
        kv_dim = self.n_kv_heads * self.head_dim
        self.q_proj = nn.Linear(d_model, q_dim)
        self.k_proj = nn.Linear(d_model, kv_dim)
        self.v_proj = nn.Linear(d_model, kv_dim)
        self.out_proj = nn.Linear(d_model, d_model)

        if pos_enc == "alibi":
            slopes = _alibi_slopes(nhead)
            self.register_buffer("alibi_slopes", slopes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.kv_group_size > 1:
            k = k.repeat_interleave(self.kv_group_size, dim=1)
            v = v.repeat_interleave(self.kv_group_size, dim=1)

        if self.pos_enc == "rope":
            q, k = _apply_rope(q, k, self.head_dim)

        attn_mask = None
        use_causal = True

        if self.pos_enc == "alibi":
            pos = torch.arange(T, device=x.device)
            dist = pos.unsqueeze(0) - pos.unsqueeze(1)  # (T, T)
            causal = (dist <= 0).float().log()
            bias = -self.alibi_slopes.view(-1, 1, 1) * dist.abs().unsqueeze(0).float()
            attn_mask = (bias + causal).unsqueeze(0)  # (1, nhead, T, T)
            use_causal = False

        if self.attn_variant == "sliding":
            sw_mask = _sliding_window_mask(T, self.window_size, x.device)
            if attn_mask is not None:
                attn_mask = attn_mask + sw_mask.unsqueeze(0).unsqueeze(0)
            else:
                attn_mask = sw_mask.unsqueeze(0).unsqueeze(0)
            use_causal = False

        out = Fn.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=use_causal,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block with configurable attention."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.self_attn = CausalSelfAttention(
            cfg.d_model, cfg.nhead, cfg.dropout, cfg.max_seq_len,
            pos_enc=cfg.pos_enc, attn_variant=cfg.attn_variant,
            n_kv_heads=cfg.n_kv_heads,
        )
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
    """Autoregressive prediction heads with discrete cluster outputs.

    Sticks use n_stick_clusters logits for main, n_shoulder_bins for L/R.

    Heads are chained autoregressively:
      L(h) -> R(h,L) -> cdir(h,L,R) -> main(h,L,R,cdir)
      -> btn_A(h,...) -> btn_B(h,...,A) -> btn_X(h,...,A,B) -> ...
    with .detach() on conditioning to prevent gradient cascade.

    Button order: A, B, X, Y, Z, L, R, START, D_UP, D_DOWN, D_LEFT, D_RIGHT
    """

    # Button prediction order: action buttons first, modifiers last
    # Drop START(7) and D-pad(8-11) — ~0% press rate, pure noise
    BTN_ORDER = [0, 1, 2, 3, 4, 5, 6]  # A, B, X, Y, Z, L, R
    N_BUTTONS = 12  # total button dim in targets (unchanged for compatibility)
    N_ACTIVE_BUTTONS = 7  # buttons we actually predict

    def __init__(self, d_model: int, hidden: int = 256, btn_threshold: float = 0.2,
                 stick_loss: str = "clusters",
                 n_stick_clusters: int = 63, n_shoulder_bins: int = 4,
                 autoregressive: bool = True):
        super().__init__()
        self.btn_threshold = btn_threshold
        self.stick_loss = stick_loss
        self.n_stick_clusters = n_stick_clusters
        self.n_shoulder_bins = n_shoulder_bins

        main_out = n_stick_clusters
        lr_out = n_shoulder_bins

        def _head(in_dim: int, out_dim: int):
            return nn.Sequential(
                nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim),
            )

        ctx_after_L    = lr_out
        ctx_after_R    = ctx_after_L + lr_out
        ctx_after_cdir = ctx_after_R + 5
        ctx_after_main = ctx_after_cdir + main_out

        self.L_head    = _head(d_model,                    lr_out)
        self.R_head    = _head(d_model + ctx_after_L,      lr_out)
        self.cdir_head = _head(d_model + ctx_after_R,      5)
        self.main_head = _head(d_model + ctx_after_cdir,   main_out)

        # Autoregressive button heads: each predicts 1 logit, conditioned on
        # the backbone context + all previously predicted buttons
        # Only predict active buttons (A,B,X,Y,Z,L,R); skip START/D-pad
        btn_base_dim = d_model + ctx_after_main
        self.btn_heads = nn.ModuleList([
            _head(btn_base_dim + i, 1)  # +i for the i previous button predictions
            for i in range(self.N_ACTIVE_BUTTONS)
        ])

    def forward(self, h: torch.Tensor, btn_targets: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass. During training, btn_targets (B,T,12) provides ground
        truth for teacher forcing the autoregressive button chain. During
        inference, btn_targets=None and each button uses its own prediction."""
        L = self.L_head(h)
        ctx = L.detach()
        R = self.R_head(torch.cat([h, ctx], dim=-1))
        ctx = torch.cat([ctx, R.detach()], dim=-1)
        cdir = self.cdir_head(torch.cat([h, ctx], dim=-1))
        ctx = torch.cat([ctx, cdir.detach()], dim=-1)
        main = self.main_head(torch.cat([h, ctx], dim=-1))
        ctx = torch.cat([ctx, main.detach()], dim=-1)

        # Autoregressive button prediction (active buttons only: A,B,X,Y,Z,L,R)
        btn_base = torch.cat([h, ctx], dim=-1)  # (B, T, btn_base_dim)
        btn_logits_list = []
        btn_ctx_parts = []

        for i, idx in enumerate(self.BTN_ORDER):
            # Input: base context + all previous button values
            if btn_ctx_parts:
                btn_input = torch.cat([btn_base] + btn_ctx_parts, dim=-1)
            else:
                btn_input = btn_base

            logit = self.btn_heads[i](btn_input)  # (B, T, 1)
            btn_logits_list.append(logit)

            # For conditioning the next button: use ground truth during training,
            # own prediction during inference
            if btn_targets is not None:
                btn_val = btn_targets[..., idx:idx+1].float().detach()
            else:
                btn_val = (torch.sigmoid(logit) > self.btn_threshold).float().detach()
            btn_ctx_parts.append(btn_val)

        # Stack into (B, T, 12) — active buttons get predicted logits,
        # dropped buttons (START, D-pad) stay at 0 (= sigmoid 0.5, but never
        # above threshold so they're never pressed)
        btn_logits = torch.full((*h.shape[:-1], self.N_BUTTONS), -10.0,
                                device=h.device, dtype=h.dtype)  # large negative = sigmoid ~0
        for i, idx in enumerate(self.BTN_ORDER):
            btn_logits[..., idx] = btn_logits_list[i].squeeze(-1)

        return dict(
            main_xy=main,
            L_val=L,
            R_val=R,
            c_dir_logits=cdir,
            btn_logits=btn_logits,
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
        from .frame_encoder import build_encoder

        self.cfg = cfg
        self.encoder = encoder or build_encoder(
            encoder_type=cfg.encoder_type,
            d_model=cfg.d_model,
            d_intra=cfg.d_intra,
            dropout=cfg.dropout,
            nlayers=cfg.encoder_nlayers,
            k_query=cfg.k_query,
            scaled_emb=cfg.scaled_emb,
            num_stages=cfg.num_stages,
            num_ports=cfg.num_ports,
            num_characters=cfg.num_characters,
            num_actions=cfg.num_actions,
            num_costumes=cfg.num_costumes,
            num_proj_types=cfg.num_proj_types,
            num_proj_subtypes=cfg.num_proj_subtypes,
            num_c_dirs=cfg.num_c_dirs,
            no_opp_inputs=cfg.no_opp_inputs,
            no_self_inputs=cfg.no_self_inputs,
        )

        if cfg.pos_enc == "learned":
            self.pos_emb = nn.Parameter(torch.randn(1, cfg.max_seq_len, cfg.d_model) * 0.02)
        elif cfg.pos_enc == "sinusoidal":
            self.register_buffer("pos_emb", _sinusoidal_embeddings(cfg.max_seq_len, cfg.d_model))
        else:
            self.pos_emb = None

        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.heads = PredictionHeads(
            cfg.d_model,
            hidden=cfg.head_hidden,
            stick_loss=cfg.stick_loss,
            n_stick_clusters=cfg.n_stick_clusters,
            n_shoulder_bins=cfg.n_shoulder_bins,
            autoregressive=cfg.autoregressive_heads,
        )

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

    def forward(self, frames: Dict[str, torch.Tensor],
                btn_targets: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        x = self.encoder(frames)                         # (B, T, d_model)
        if self.pos_emb is not None:
            T = x.size(1)
            x = x + self.pos_emb[:, :T]
        for blk in self.blocks:
            x = blk(x)
        x = self.final_norm(x)                           # (B, T, d_model)
        return self.heads(x, btn_targets=btn_targets)    # all T positions
