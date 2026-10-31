#!/usr/bin/env python3
# model.py ─ MIMIC next-frame predictor
# ------------------------------------
# 1. Hyper-parameter bundle
# 2. Shaw relative-position attention & the mimic transformer block
# 3. Prediction heads (MimicPredictionHeads)
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
    encoder_type: str    = "mimic_flat"

    # sequence length
    max_seq_len: int     = 60

    # positional encoding: Shaw relative-position attention is the only one implemented
    pos_enc: str         = "relpos"

    # modern components
    use_rmsnorm: bool    = False
    use_swiglu: bool     = False

    # Per-input-column sigmoid gate on the encoder's final projection.
    # When True and trained with an L1 penalty on the gate, the sigmoid
    # values act as learned feature importance (0 = pruned, 1 = kept).
    use_input_gate: bool = False

    # inputs
    no_opp_inputs: bool  = True
    no_self_inputs: bool = True

    # prediction heads (the mimic heads: single-label buttons, combined 3-class shoulder, LN)
    n_stick_clusters: int           = 63
    mimic_mode: bool                = True   # the only mode; False only in pre-mimic checkpoints
    mimic_minimal_features: bool    = False  # legacy 9-col gamestate; removed 2026-10-22 (load refuses)
    mimic_controller_encoding: bool = False  # one-hot controller feedback (stick clusters + button combos)
    # Conditional heads (--cond-heads): each mimic head is conditioned on the one-hot CLASS of the
    # earlier heads (the true class in training, the sampled class at inference) instead of their
    # detached logits, so the four heads model the joint distribution. Same parameter shapes.
    cond_heads: bool                = False
    # Training-time kernel for the Shaw relpos attention (--relpos-impl): "fp32" (the explicit
    # FP32 math), "sdpa" (Shaw term as an additive mask into fused SDPA) or "flex"
    # (FlexAttention; no attention dropout, needs
    # torch.compile). Speed only: eval and inference always run the fp32 path.
    relpos_impl: str                = "fp32"
    # Sliding-window relpos attention (--attn-window): >0 limits every layer to the last N frames,
    # so training can use long chunks and inference can stream with a KV cache (mimic/kv_cache.py).
    attn_window: int                = 0
    # LLM-architecture variants (2026-10-25 proxy sweep; all off = the original block). Set from
    # train.py --arch key=value. Full-causal models only.
    qk_norm: bool                   = False   # RMSNorm on q and k per head (OLMo 2 / Gemma 3)
    attn_gate: bool                 = False   # sigmoid gate on the attention output (Qwen gated attention)
    mlp_act: str                    = "gelu"  # "relu2": ReLU squared (modded-nanogpt)
    zero_init_proj: bool            = False   # zero-init the attention / MLP output projections
    logit_softcap: float            = 0.0     # >0: head logits -> c * tanh(logits / c) (Gemma 2)
    value_residual: bool            = False   # mix every layer's values with layer 0's (ResFormer)
    unet_skips: bool                = False   # first-half block outputs added to the mirrored second half
    canon: bool                     = False   # causal depthwise conv (k=4) before attention and MLP
    n_controller_combos: int        = 5       # number of button combo classes (from controller_combos.json)

    # Legacy hal_* fields — kept for checkpoint backwards compatibility.
    # __post_init__ migrates them onto the mimic_* fields above. Always
    # read the mimic_* fields in new code.
    hal_mode: bool                  = False
    hal_minimal_features: bool      = False
    hal_controller_encoding: bool   = False

    # fixed categorical vocab sizes
    num_stages: int       = len(STAGE_MAP)
    num_ports: int        = 4
    num_characters: int   = len(CHARACTER_MAP)
    num_actions: int      = len(ACTION_MAP)
    num_costumes: int     = 6
    num_c_dirs: int       = 5
    num_proj_types: int   = len(PROJECTILE_TYPE_MAP)
    num_proj_subtypes: int= 40

    def __post_init__(self):
        # Migrate legacy hal_* fields onto mimic_* fields.
        # If the caller set hal_mode=True but didn't touch mimic_mode,
        # honor hal_mode. Forward-going code should only read mimic_*.
        if self.hal_mode and not self.mimic_mode:
            self.mimic_mode = True
        if self.hal_minimal_features and not self.mimic_minimal_features:
            self.mimic_minimal_features = True
        if self.hal_controller_encoding and not self.mimic_controller_encoding:
            self.mimic_controller_encoding = True
        # Keep legacy fields in sync so old code paths still read them
        # correctly without a sweep.
        self.hal_mode = self.mimic_mode
        self.hal_minimal_features = self.mimic_minimal_features
        self.hal_controller_encoding = self.mimic_controller_encoding


MODEL_PRESETS = {
    # M: the production model (~20M params; bootstrapped from HAL's GPTv5Controller, then diverged).
    "M": dict(d_model=512, nhead=8, num_layers=6, dim_feedforward=2048,
              dropout=0.2, max_seq_len=180, pos_enc="relpos",
              num_stages=6, num_characters=27, num_actions=396, num_c_dirs=9),
    # S: half-width M (~5.4M params; same head_dim 64, 4x FFN, depth 6), the fast research proxy.
    "S": dict(d_model=256, nhead=4, num_layers=6, dim_feedforward=1024,
              dropout=0.2, max_seq_len=180, pos_enc="relpos",
              num_stages=6, num_characters=27, num_actions=396, num_c_dirs=9),
    # L: 1.5x-width M (~44M params; same head_dim 64, 4x FFN, depth 6). Added 2026-10-29 (bc-L-v1).
    "L": dict(d_model=768, nhead=12, num_layers=6, dim_feedforward=3072,
              dropout=0.2, max_seq_len=180, pos_enc="relpos",
              num_stages=6, num_characters=27, num_actions=396, num_c_dirs=9),
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. Shaw relative-position attention (skew trick, Shaw et al. 2018)
# ─────────────────────────────────────────────────────────────────────────────

def _skew(QEr: torch.Tensor) -> torch.Tensor:
    """Efficient relative-position skew trick (Music Transformer / Shaw 2018)."""
    padded = Fn.pad(QEr, (1, 0))
    B, nh, nr, nc = padded.shape
    reshaped = padded.reshape(B, nh, nc, nr)
    return reshaped[:, :, 1:, :]


class CausalSelfAttentionRelPos(nn.Module):
    """HAL's relative-position causal self-attention.

    Uses combined QKV projection, learnable relative position table Er,
    and the skew trick for efficient relative-position bias computation.
    """

    def __init__(self, n_embd: int = 512, n_head: int = 8,
                 block_size: int = 1024, dropout: float = 0.2, impl: str = "fp32",
                 window: int = 0, qk_norm: bool = False, gate: bool = False,
                 value_residual: bool = False):
        super().__init__()
        assert impl in ("fp32", "sdpa", "flex"), impl
        assert 0 <= window <= block_size, f"attn_window {window} exceeds the relpos table ({block_size})"
        self.impl = impl
        self.window = window          # >0: each query sees only the last `window` keys (incl. itself)
        self._flex_masks = {}
        self.n_head = n_head
        self.hs = n_embd // n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.Er = nn.Parameter(torch.randn(block_size, self.hs))
        self.q_norm = RMSNorm(self.hs) if qk_norm else None
        self.k_norm = RMSNorm(self.hs) if qk_norm else None
        self.c_gate = nn.Linear(n_embd, n_embd) if gate else None
        self.v_mix = nn.Parameter(torch.tensor([0.5, 0.5])) if value_residual else None
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.register_buffer("bias",
            torch.tril(torch.ones(block_size, block_size)).view(
                1, 1, block_size, block_size))
        if window > 0:   # separate, non-persistent: `bias` stays the plain causal mask in checkpoints
            band = self.bias - torch.tril(torch.ones(block_size, block_size), diagonal=-window)
            self.register_buffer("band", band, persistent=False)
        self.block_size = block_size

    def _mask(self, L: int) -> torch.Tensor:
        if self.window > 0 and L > self.block_size:
            ar = torch.arange(L, device=self.bias.device)
            dd = ar[:, None] - ar[None, :]
            return ((dd >= 0) & (dd < self.window)).float()[None, None]
        return (self.band if self.window > 0 else self.bias)[:, :, :L, :L]

    def forward(self, x: torch.Tensor, v0: Optional[torch.Tensor] = None):
        """x: (B, L, D). With value_residual, v0 is layer 0's values (None in layer 0) and the call
        returns (out, values) so the caller can pass layer 0's on; otherwise it returns out."""
        B, L, D = x.size()
        q, k, v = self.c_attn(x).split(D, dim=2)
        k = k.view(B, L, self.n_head, self.hs).transpose(1, 2)
        q = q.view(B, L, self.n_head, self.hs).transpose(1, 2)
        v = v.view(B, L, self.n_head, self.hs).transpose(1, 2)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        if self.v_mix is not None and v0 is not None:
            v = self.v_mix[0] * v + self.v_mix[1] * v0
        v_out = v
        if self.training and self.impl != "fp32" and (
                self.impl != "flex" or torch.compiler.is_compiling()):
            y = self._fast_attention(q, k, v, L)
        elif self.window > 0 and L >= 2 * self.window:   # banded: O(L * W), not O(L^2)
            y = self._banded(q, k, v, L)
        else:
            y = self._dense(q, k, v, L)
        y = y.transpose(1, 2).contiguous().view(B, L, D)
        if self.c_gate is not None:
            y = y * torch.sigmoid(self.c_gate(x))
        out = self.resid_dropout(self.c_proj(y))
        return (out, v_out) if self.v_mix is not None else out

    def _dense(self, q, k, v, L):
        B = q.size(0)

        # Attention math in FP32 to avoid BF16 overflow on Q@K^T + Srel
        with torch.autocast("cuda", enabled=False):
            q, k = q.float(), k.float()
            if self.window > 0:   # score by distance (< window), so L may exceed block_size
                w = self.window
                QEr = q @ self.Er.flip(0)[:w].float().transpose(0, 1)          # (B, H, L, w)
                ar = torch.arange(L, device=q.device)
                d = (ar[:, None] - ar[None, :]).clamp(0, w - 1)                 # masked entries clamp
                Srel = QEr.gather(-1, d.expand(B, self.n_head, L, L))
            else:
                start = self.block_size - L
                Er_t = self.Er[start:, :].float().transpose(0, 1)
                QEr = q @ Er_t
                Srel = _skew(QEr)

            QK_t = q @ k.transpose(-2, -1)
            scale = 1.0 / math.sqrt(k.size(-1))
            att = (QK_t + Srel) * scale
            att = att.masked_fill(self._mask(L) == 0, float("-inf"))
            att = Fn.softmax(att, dim=-1)

        att = self.attn_dropout(att)
        return att @ v

    def _banded(self, q, k, v, L):
        """The fp32 window path above, computed per block of W queries against the W - 1 keys
        before the block plus the block itself: the same scores, mask and softmax, with memory
        O(L * W) instead of O(L^2), so whole games (~6,000+ frames) fit."""
        w = self.window
        Bq = w
        nb = -(-L // Bq)
        P = nb * Bq - L                                   # tail padding (queries discarded)
        with torch.autocast("cuda", enabled=False):
            q, k = q.float(), k.float()
            B, H, _, hs = q.shape
            QEr = q @ self.Er.flip(0)[:w].float().transpose(0, 1)                   # (B, H, L, w)
            qb = Fn.pad(q, (0, 0, 0, P)).view(B, H, nb, Bq, hs)
            QErb = Fn.pad(QEr, (0, 0, 0, P)).view(B, H, nb, Bq, w)
            kp = Fn.pad(k, (0, 0, w - 1, P))                                        # front: masked
            vp = Fn.pad(v, (0, 0, w - 1, P))
            S = Bq + w - 1                                                          # keys per block
            kb = kp.unfold(2, S, Bq).transpose(-1, -2)                              # (B, H, nb, S, hs)
            vb = vp.unfold(2, S, Bq).transpose(-1, -2)
            r = torch.arange(Bq, device=q.device)[:, None]
            c = torch.arange(S, device=q.device)[None, :]
            d = r - c + (w - 1)                                                     # i - j
            blk = torch.arange(nb, device=q.device)[:, None, None]
            j = blk * Bq - (w - 1) + c                                              # absolute key
            ok = (d >= 0) & (d < w) & (j >= 0)                                      # (nb, Bq, S)
            Srel = QErb.gather(-1, d.clamp(0, w - 1).expand(B, H, nb, Bq, S))
            att = (qb @ kb.transpose(-1, -2) + Srel) * (1.0 / math.sqrt(hs))
            att = att.masked_fill(~ok, float("-inf"))
            att = Fn.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ vb.to(att.dtype)                                                  # (B, H, nb, Bq, hs)
        return y.reshape(B, H, nb * Bq, hs)[:, :, :L]

    def _srel(self, q, L):
        """Shaw relative term Srel[i, j] = q_i . Er[block_size-1-(i-j)] for the dense paths. With a
        window, indexed by distance (< window), so L may exceed block_size; else the skew trick."""
        if self.window > 0:
            w = self.window
            QEr = q @ self.Er.flip(0)[:w].transpose(0, 1).to(q.dtype)          # (B, H, L, w)
            ar = torch.arange(L, device=q.device)
            d = (ar[:, None] - ar[None, :]).clamp(0, w - 1)                     # masked entries clamp
            return QEr.gather(-1, d.expand(q.size(0), q.size(1), L, L))
        return _skew(q @ self.Er[self.block_size - L:, :].transpose(0, 1).to(q.dtype))

    def _fast_attention(self, q, k, v, L):
        """Training-only kernels for the same Q@K^T + Srel attention (see relpos_impl)."""
        scale = 1.0 / math.sqrt(k.size(-1))
        if self.impl == "sdpa":
            bias = (self._srel(q, L) * scale).masked_fill(self._mask(L) == 0, float("-inf"))
            return Fn.scaled_dot_product_attention(q, k, v, attn_mask=bias,
                                                   dropout_p=self.attn_dropout.p)
        # flex: column r of QEr is relative offset r - (L-1); pad so masked j > i stays in range
        from torch.nn.attention.flex_attention import flex_attention, create_block_mask
        w = self.window
        if w > 0:   # only distances 0..w-1 exist: Er[block_size-1-d] scores distance d
            QEr = (q @ self.Er.flip(0)[:w].transpose(0, 1).to(q.dtype)) * scale   # (B,H,L,w)
            def score_mod(score, b, h, i, j):
                return score + QEr[b, h, i, torch.clamp(i - j, 0, w - 1)]
            mask_mod = lambda b, h, i, j: (i >= j) & (i - j < w)
        else:
            QEr = Fn.pad((q @ self.Er[self.block_size - L:, :].transpose(0, 1).to(q.dtype)) * scale,
                         (0, L - 1))
            def score_mod(score, b, h, i, j):
                return score + QEr[b, h, i, j - i + L - 1]
            mask_mod = lambda b, h, i, j: i >= j
        key = (L, q.device)
        if key not in self._flex_masks:
            self._flex_masks[key] = create_block_mask(mask_mod, None, None, L, L, device=q.device)
        return flex_attention(q, k, v, score_mod=score_mod, block_mask=self._flex_masks[key],
                              scale=scale)


class MimicTransformerBlock(nn.Module):
    """HAL's pre-norm Transformer block with relative-position attention.
    Supports modern components (RMSNorm, SwiGLU) via config flags."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        use_rmsnorm = getattr(cfg, 'use_rmsnorm', False)
        use_swiglu = getattr(cfg, 'use_swiglu', False)
        Norm = RMSNorm if use_rmsnorm else nn.LayerNorm
        self.ln_1 = Norm(cfg.d_model)
        self.self_attn = CausalSelfAttentionRelPos(
            n_embd=cfg.d_model, n_head=cfg.nhead,
            block_size=1024, dropout=cfg.dropout,
            impl=getattr(cfg, "relpos_impl", "fp32"),
            window=getattr(cfg, "attn_window", 0),
            qk_norm=getattr(cfg, "qk_norm", False), gate=getattr(cfg, "attn_gate", False),
            value_residual=getattr(cfg, "value_residual", False))
        self.ln_2 = Norm(cfg.d_model)
        self._relu2 = getattr(cfg, "mlp_act", "gelu") == "relu2"
        canon = getattr(cfg, "canon", False)
        self.canon_a = CausalConv(cfg.d_model) if canon else None
        self.canon_c = CausalConv(cfg.d_model) if canon else None
        if use_swiglu:
            self.mlp = SwiGLU(cfg.d_model, cfg.dim_feedforward, cfg.dropout)
        else:
            self.mlp = nn.ModuleDict(dict(
                c_fc=nn.Linear(cfg.d_model, cfg.dim_feedforward),
                c_proj=nn.Linear(cfg.dim_feedforward, cfg.d_model),
            ))
        self._use_swiglu = use_swiglu
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, v0: Optional[torch.Tensor] = None):
        """Returns x, or (x, values) for value_residual models (see CausalSelfAttentionRelPos)."""
        h = self.ln_1(x)
        if self.canon_a is not None:
            h = self.canon_a(h)
        a = self.self_attn(h, v0)
        v = None
        if isinstance(a, tuple):
            a, v = a
        x = x + a
        h = self.ln_2(x)
        if self.canon_c is not None:
            h = self.canon_c(h)
        if self._use_swiglu:
            x = x + self.mlp(h)
        else:
            h = self.mlp.c_fc(h)
            h = Fn.relu(h).square() if self._relu2 else Fn.gelu(h)
            h = self.dropout(self.mlp.c_proj(h))
            x = x + h
        return x if v is None else (x, v)


# ─────────────────────────────────────────────────────────────────────────────
# 2b. Optional block components: RMSNorm, SwiGLU
# ─────────────────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich 2019).
    Faster than LayerNorm: no mean subtraction, one learnable param (gamma)."""
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class SwiGLU(nn.Module):
    """SwiGLU feedforward: SiLU(xW_gate) * (xW_up) -> W_down.
    Uses 2/3 * dim_feedforward for gate/up to match param count of standard FFN."""
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float = 0.0):
        super().__init__()
        # Scale hidden dim to 2/3 so total params ≈ standard FFN (2 * d * ff)
        hidden = int(2 * dim_feedforward / 3)
        self.w_gate = nn.Linear(d_model, hidden, bias=False)
        self.w_up = nn.Linear(d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_down(Fn.silu(self.w_gate(x)) * self.w_up(x)))


class CausalConv(nn.Module):
    """Canon layer (Physics of Language Models 4.1): x + depthwise causal conv over the last k
    positions. Zero-init, so it starts as the identity."""
    def __init__(self, d: int, k: int = 4):
        super().__init__()
        self.k = k
        self.conv = nn.Conv1d(d, d, k, groups=d, bias=False)
        nn.init.zeros_(self.conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # (B, T, d)
        h = Fn.pad(x.transpose(1, 2), (self.k - 1, 0))
        return x + self.conv(h).transpose(1, 2)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Output heads
# ─────────────────────────────────────────────────────────────────────────────
def _cond_sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """One class per position from softmax(logits / T) via Gumbel-max (argmax when T <= 0 or the
    logits are not finite): the cond heads' inference sampler, CUDA-graph capturable."""
    lg = torch.nan_to_num(logits.float(), nan=-1e9, posinf=1e9, neginf=-1e9)
    if temperature <= 0:
        return lg.argmax(dim=-1)
    u = torch.rand_like(lg).clamp_(1e-20, 1.0)
    return (lg / temperature - torch.log(-torch.log(u))).argmax(dim=-1)


class SoftCap(nn.Module):
    """c * tanh(x / c): bounds logits to (-c, c) (Gemma 2 / modded-nanogpt)."""
    def __init__(self, c: float):
        super().__init__()
        self.c = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c * torch.tanh(x / self.c)


class MimicPredictionHeads(nn.Module):
    """HAL-style prediction heads: single-label buttons, combined shoulder, LayerNorm.

    Autoregressive chain:
      shoulder(h) -> cdir(h,S) -> main(h,S,C) -> buttons(h,S,C,M)

    Buttons: 5-class softmax (A, B, jump=X|Y, Z, NO_BUTTON)
    Shoulder: combined max(L,R) -> 3-class (0.0, 0.4, 1.0)
    Heads use LayerNorm and hidden=in_dim//2 (matching HAL).
    """

    # Single-label button classes: A=0, B=1, jump=2, Z=3, NO_BUTTON=4
    N_BTN_CLASSES = 5

    def __init__(self, d_model: int, n_stick_clusters: int = 63,
                 n_shoulder_bins: int = 3, n_cdir: int = 5,
                 n_btn_classes: int = 5, cond: bool = False, softcap: float = 0.0):
        super().__init__()
        self.cond = cond
        self.softcap = softcap
        # Inference sampling temperature for cond heads (they sample inside forward).
        self.sample_temperature = 1.0
        self.n_stick_clusters = n_stick_clusters
        self.n_shoulder_bins = n_shoulder_bins
        self.n_btn_classes = n_btn_classes

        def _head(in_dim: int, out_dim: int):
            h = in_dim // 2
            layers = [nn.LayerNorm(in_dim), nn.Linear(in_dim, h), nn.GELU(), nn.Linear(h, out_dim)]
            if softcap > 0:
                layers.append(SoftCap(softcap))
            return nn.Sequential(*layers)

        shldr_out = n_shoulder_bins
        cdir_out = n_cdir
        main_out = n_stick_clusters
        btn_out = n_btn_classes

        self.shoulder_head = _head(d_model, shldr_out)
        self.cdir_head     = _head(d_model + shldr_out, cdir_out)
        self.main_head     = _head(d_model + shldr_out + cdir_out, main_out)
        self.btn_head      = _head(d_model + shldr_out + cdir_out + main_out, btn_out)

    def forward(self, h: torch.Tensor, btn_targets: Optional[torch.Tensor] = None,
                cond_classes: Optional[tuple] = None) -> Dict[str, torch.Tensor]:
        if self.cond:
            return self._forward_cond(h, cond_classes)
        shoulder = self.shoulder_head(h)
        ctx = shoulder.detach()
        cdir = self.cdir_head(torch.cat([h, ctx], dim=-1))
        ctx = torch.cat([ctx, cdir.detach()], dim=-1)
        main = self.main_head(torch.cat([h, ctx], dim=-1))
        ctx = torch.cat([ctx, main.detach()], dim=-1)
        btn = self.btn_head(torch.cat([h, ctx], dim=-1))

        return dict(
            main_xy=main,
            shoulder_val=shoulder,  # combined shoulder (not separate L/R)
            c_dir_logits=cdir,
            btn_logits=btn,         # (B, T, 5) single-label logits
        )

    def _forward_cond(self, h, cond_classes):
        """Conditional chain shoulder -> c -> main -> buttons. cond_classes = (shoulder, c, main)
        class tensors (B, T) in training / val (teacher forcing: the loss is the joint NLL). None
        at inference: each head is sampled (Gumbel-max at self.sample_temperature; argmax when
        <= 0) and its one-hot fed to the next; the samples are returned as "cond_sample"
        (B, T, 4) in (main, c, shoulder, button) order."""
        oh = lambda idx, n: Fn.one_hot(idx.long(), n).to(h.dtype)
        n_s, n_c, n_m = (self.shoulder_head[3].out_features, self.cdir_head[3].out_features,
                         self.main_head[3].out_features)
        sampling = cond_classes is None
        shoulder = self.shoulder_head(h)
        s_idx = _cond_sample(shoulder, self.sample_temperature) if sampling else cond_classes[0]
        ctx = oh(s_idx, n_s)
        cdir = self.cdir_head(torch.cat([h, ctx], dim=-1))
        c_idx = _cond_sample(cdir, self.sample_temperature) if sampling else cond_classes[1]
        ctx = torch.cat([ctx, oh(c_idx, n_c)], dim=-1)
        main = self.main_head(torch.cat([h, ctx], dim=-1))
        m_idx = _cond_sample(main, self.sample_temperature) if sampling else cond_classes[2]
        ctx = torch.cat([ctx, oh(m_idx, n_m)], dim=-1)
        btn = self.btn_head(torch.cat([h, ctx], dim=-1))
        out = dict(main_xy=main, shoulder_val=shoulder, c_dir_logits=cdir, btn_logits=btn)
        if sampling:
            b_idx = _cond_sample(btn, self.sample_temperature)
            out["cond_sample"] = torch.stack([m_idx, c_idx, s_idx, b_idx], dim=-1)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 4. Full model: Encoder → N×Transformer → Heads
# ─────────────────────────────────────────────────────────────────────────────
class FramePredictor(nn.Module):
    """
    1. Encodes structured frames via MimicFlatEncoder → (B, T, d_model)
    2. Runs a stack of causal relpos Transformer blocks
    3. Feeds all positions into the mimic heads (autoregressive training)
    """
    def __init__(self, cfg: ModelConfig, encoder: Optional[nn.Module] = None):
        super().__init__()
        from .frame_encoder import build_encoder

        self.cfg = cfg
        self.encoder = encoder or build_encoder(
            encoder_type=cfg.encoder_type,
            d_model=cfg.d_model,
            dropout=cfg.dropout,
            num_stages=cfg.num_stages,
            num_characters=cfg.num_characters,
            num_actions=cfg.num_actions,
            no_self_inputs=cfg.no_self_inputs,
            mimic_controller_encoding=cfg.mimic_controller_encoding,
            n_controller_combos=cfg.n_controller_combos,
            use_input_gate=cfg.use_input_gate,
        )

        if cfg.pos_enc != "relpos":
            raise ValueError(f"only Shaw relpos attention is implemented, not pos_enc={cfg.pos_enc!r} "
                             f"(the RoPE/xPos/ALiBi/learned variants were removed 2026-10-22; git history)")
        self.blocks = nn.ModuleList([MimicTransformerBlock(cfg) for _ in range(cfg.num_layers)])
        FinalNorm = RMSNorm if getattr(cfg, 'use_rmsnorm', False) else nn.LayerNorm
        self.final_norm = FinalNorm(cfg.d_model)
        if cfg.mimic_minimal_features:
            raise ValueError("minimal-features (9-col gamestate) models are no longer supported "
                             "(removed 2026-10-22; git history)")
        if not cfg.mimic_mode:
            raise ValueError("only the mimic heads are implemented (the multi-label PredictionHeads "
                             "were removed 2026-10-22; git history)")
        self.heads = MimicPredictionHeads(
            cfg.d_model,
            n_stick_clusters=cfg.n_stick_clusters,
            n_shoulder_bins=3,  # 3-class combined shoulder (LOW/MID/HIGH)
            n_cdir=cfg.num_c_dirs,
            n_btn_classes=cfg.n_controller_combos if cfg.mimic_controller_encoding else 5,
            cond=getattr(cfg, "cond_heads", False),
            softcap=getattr(cfg, "logit_softcap", 0.0),
        )
        variants = [k for k in ("qk_norm", "attn_gate", "zero_init_proj", "value_residual",
                                "unet_skips", "canon") if getattr(cfg, k, False)]
        if getattr(cfg, "mlp_act", "gelu") != "gelu":
            variants.append("mlp_act")
        if getattr(cfg, "logit_softcap", 0.0):
            variants.append("logit_softcap")
        self.tokens_per_frame = getattr(self.encoder, "tokens_per_frame", 1)
        if self.tokens_per_frame > 1:
            variants.append(cfg.encoder_type)
        if variants and getattr(cfg, "attn_window", 0):
            raise ValueError(f"{variants} are implemented for full-causal models only (the streaming "
                             "KV cache and the stream RL update re-implement the plain block)")
        # fast_policy.forward_last / kv_cache re-implement the plain block; anything else uses forward
        self.plain_blocks = set(variants) <= {"zero_init_proj", "attn_gate"}   # forward_last has these
        self.unet_w = (nn.Parameter(torch.ones(cfg.num_layers // 2))
                       if getattr(cfg, "unet_skips", False) else None)

        self.apply(self._init_weights)
        residual_std = 0.02 / math.sqrt(2 * max(cfg.num_layers, 1))
        std = 0.0 if getattr(cfg, "zero_init_proj", False) else residual_std
        for blk in self.blocks:
            nn.init.normal_(blk.self_attn.c_proj.weight, std=std)
            if isinstance(blk.mlp, SwiGLU):
                nn.init.normal_(blk.mlp.w_down.weight, std=std)
            else:
                nn.init.normal_(blk.mlp.c_proj.weight, std=std)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def features(self, frames: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encoder + backbone + final norm: the (B, T, d_model) input to the heads."""
        x = self.encoder(frames)                         # (B, T, d_model)
        n, v0, skips = len(self.blocks), None, []
        for i, blk in enumerate(self.blocks):
            if self.unet_w is not None and i >= n - len(self.unet_w):   # mirrored first-half output
                x = x + self.unet_w[n - 1 - i] * skips[n - 1 - i]
            out = blk(x, v0)
            if isinstance(out, tuple):
                x, v = out
                v0 = v if v0 is None else v0
            else:
                x = out
            if self.unet_w is not None and i < len(self.unet_w):
                skips.append(x)
        x = self.final_norm(x)
        if self.tokens_per_frame > 1:   # the heads read each frame's last token (self)
            x = x[:, self.tokens_per_frame - 1::self.tokens_per_frame]
        return x

    def forward(self, frames: Dict[str, torch.Tensor],
                btn_targets: Optional[torch.Tensor] = None,
                cond_classes: Optional[tuple] = None) -> Dict[str, torch.Tensor]:
        x = self.features(frames)                        # (B, T, d_model)
        if cond_classes is not None:
            return self.heads(x, btn_targets=btn_targets, cond_classes=cond_classes)
        return self.heads(x, btn_targets=btn_targets)    # all T positions
