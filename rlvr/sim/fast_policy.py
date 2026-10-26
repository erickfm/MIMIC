"""Inference-only fast path for the `mimic` (Shaw relpos) model: predictions for the newest frame.

Inference uses only the last position's heads, but FramePredictor.forward runs the final block
and the four autoregressive heads on all T positions. forward_last runs the first n-1 blocks on
the whole window as before (the last block's keys need them) and the final block's attention
query, MLP, final norm and heads on the last position only. Same math as the full forward's last
row; for the last query the relative-position skew is the identity (rel index = key index), so
Srel[-1] = (q_last @ Er^T) without the skew.
"""
import math

import torch
import torch.nn.functional as Fn


def _last_query_attention(attn, x):
    """CausalSelfAttentionRelPos.forward for the last query position only."""
    B, L, D = x.size()
    q, k, v = attn.c_attn(x).split(D, dim=2)
    q = q[:, -1:].view(B, 1, attn.n_head, attn.hs).transpose(1, 2)
    k = k.view(B, L, attn.n_head, attn.hs).transpose(1, 2)
    v = v.view(B, L, attn.n_head, attn.hs).transpose(1, 2)
    with torch.autocast("cuda", enabled=False):   # same FP32 attention math as training
        q, k = q.float(), k.float()
        start = attn.block_size - L
        Er_t = attn.Er[start:, :].float().transpose(0, 1)
        srel = q @ Er_t                               # (B, H, 1, L): the last row of skew(Q Er^T)
        att = (q @ k.transpose(-2, -1) + srel) * (1.0 / math.sqrt(k.size(-1)))
        att = Fn.softmax(att, dim=-1)                 # the last query sees every key: no mask
    y = att.to(v.dtype) @ v if att.dtype != v.dtype else att @ v
    y = y.transpose(1, 2).contiguous().view(B, 1, D)
    if attn.c_gate is not None:   # --arch attn_gate=1
        y = y * torch.sigmoid(attn.c_gate(x[:, -1:]))
    return attn.c_proj(y)


def forward_last(model, frames):
    """model(frames) restricted to the last position: the same dict of head outputs, T = 1."""
    if getattr(model.cfg, "attn_window", 0) > 0:
        raise NotImplementedError("forward_last supports full-causal models only; stream "
                                  "sliding-window models with mimic/kv_cache.RelPosKVCache")
    if not getattr(model, "plain_blocks", True):   # architecture variants: the plain forward
        return {k: v[:, -1:] for k, v in model(frames).items()}
    x = model.encoder(frames)
    for blk in model.blocks[:-1]:
        x = blk(x)
    blk = model.blocks[-1]
    h = x[:, -1:] + _last_query_attention(blk.self_attn, blk.ln_1(x))
    if blk._use_swiglu:
        h = h + blk.mlp(blk.ln_2(h))
    else:
        h = h + blk.mlp.c_proj(Fn.gelu(blk.mlp.c_fc(blk.ln_2(h))))
    return model.heads(model.final_norm(h))
