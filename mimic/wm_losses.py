"""World-model per-column losses.

Targets come from the same shard fields as inputs, shifted by +1 frame.
All numeric targets are in the shard's normalized space; we apply MSE
directly on them. Flags are already {0, 1} in the shard; BCE-with-logits
on the raw head output.

Returns a dict with per-column losses and a `total` field (weighted sum).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import torch
import torch.nn.functional as F


@dataclass
class WMLossWeights:
    action_self: float = 1.0
    action_opp: float = 1.0
    numeric_self: float = 1.0
    numeric_opp: float = 1.0
    flags_self: float = 0.5
    flags_opp: float = 0.5
    # Action-elapsed scalar regression (frames-since-action-change at t+1).
    # Softer signal than raw action CE — has non-trivial gradient on the
    # many intra-animation frames where the action didn't change. Scaled
    # by 1/action_elapsed_scale before loss to match Huber's range.
    action_elapsed_self: float = 0.25
    action_elapsed_opp: float = 0.25
    action_elapsed_scale: float = 30.0
    # Discretized-counter CE weights (only used when the model has
    # discretize_counters=True and the loss receives the matching int
    # targets). Each applies to both self and opp symmetrically.
    percent_w: float = 0.5
    stock_w: float = 0.5
    jumps_w: float = 0.3
    hitlag_w: float = 0.3
    hitstun_w: float = 0.3
    # "huber" (default) or "mse" for numeric cols. Huber is much better
    # behaved on the velocity/hitlag/hitstun cols, which can have huge
    # z-scored tails on legacy shards where those stats are near-zero.
    numeric_loss: str = "huber"
    huber_delta: float = 1.0


def compute_wm_loss(
    preds: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    weights: WMLossWeights = None,
) -> Dict[str, torch.Tensor]:
    """Per-column losses. All inputs are (B, T, *) shape.

    preds: output of WorldModelHeads.forward(h). Keys:
      self_action_logits, opp_action_logits,
      self_numeric_pred,  opp_numeric_pred,
      self_flags_logits,  opp_flags_logits

    targets: shifted-by-1 state dict. Keys:
      self_action, opp_action (int64),
      self_numeric, opp_numeric (float, already normalized),
      self_flags, opp_flags (float, 0/1).
    """
    w = weights or WMLossWeights()
    out: Dict[str, torch.Tensor] = {}

    # Cross-entropy on action. Flatten (B, T, C) → (B*T, C) and (B, T) → (B*T).
    for side in ("self", "opp"):
        logits = preds[f"{side}_action_logits"]
        tgt = targets[f"{side}_action"].long()
        B, T, C = logits.shape
        out[f"action_{side}"] = F.cross_entropy(
            logits.reshape(B * T, C), tgt.reshape(B * T)
        )

    # Numeric regression. Default Huber (SmoothL1) — MSE blows up on
    # the poorly-normalized velocity/hitlag/hitstun cols. When
    # discretize_counters is active (detected by the presence of percent
    # CE logits in preds AND the pred head's shrunken output width),
    # the 13-col target gets sliced down to the 8 continuous cols only.
    discretize_on = "self_percent_logits" in preds
    for side in ("self", "opp"):
        pred = preds[f"{side}_numeric_pred"]
        tgt = targets[f"{side}_numeric"].float()
        if discretize_on and tgt.shape[-1] != pred.shape[-1]:
            # 13-col shard target → select the 8 continuous columns:
            # pos_x(0), pos_y(1), 5 speeds(5..9), shield(12).
            CONT_COLS = [0, 1, 5, 6, 7, 8, 9, 12]
            tgt = tgt[..., CONT_COLS]
        if w.numeric_loss == "mse":
            out[f"numeric_{side}"] = F.mse_loss(pred, tgt)
        else:
            out[f"numeric_{side}"] = F.smooth_l1_loss(
                pred, tgt, beta=w.huber_delta
            )

    # BCE-with-logits on flags.
    for side in ("self", "opp"):
        logits = preds[f"{side}_flags_logits"]
        tgt = targets[f"{side}_flags"].float()
        out[f"flags_{side}"] = F.binary_cross_entropy_with_logits(logits, tgt)

    # Action-elapsed regression (optional — only when the head is present
    # AND the shard provides the target). Target is scaled by
    # 1/action_elapsed_scale so the loss magnitude sits alongside the
    # Huber numeric loss (~1 nominal).
    has_elapsed = (
        "self_action_elapsed_pred" in preds
        and "self_action_elapsed" in targets
    )
    if has_elapsed:
        for side in ("self", "opp"):
            pred = preds[f"{side}_action_elapsed_pred"]
            tgt = targets[f"{side}_action_elapsed"].float() / w.action_elapsed_scale
            out[f"action_elapsed_{side}"] = F.smooth_l1_loss(
                pred, tgt, beta=w.huber_delta
            )

    out["total"] = (
        w.action_self * out["action_self"]
        + w.action_opp * out["action_opp"]
        + w.numeric_self * out["numeric_self"]
        + w.numeric_opp * out["numeric_opp"]
        + w.flags_self * out["flags_self"]
        + w.flags_opp * out["flags_opp"]
    )
    if has_elapsed:
        out["total"] = (
            out["total"]
            + w.action_elapsed_self * out["action_elapsed_self"]
            + w.action_elapsed_opp * out["action_elapsed_opp"]
        )

    # Discretized counter CE heads (percent, stock, jumps, hitlag, hitstun,
    # elapsed-when-discretized). Only compute when both the head and the
    # int target tensors are present. Weights are per-side symmetric.
    if discretize_on:
        COUNTER_WEIGHTS = {
            "percent": w.percent_w,
            "stock": w.stock_w,
            "jumps": w.jumps_w,
            "hitlag": w.hitlag_w,
            "hitstun": w.hitstun_w,
            "elapsed": w.action_elapsed_self,  # reuse elapsed weight
        }
        for side in ("self", "opp"):
            for name, wt in COUNTER_WEIGHTS.items():
                logits = preds[f"{side}_{name}_logits"]
                tgt_key = f"{side}_{name}_int"
                if tgt_key not in targets:
                    continue
                tgt = targets[tgt_key].long()
                B, T, C = logits.shape
                ce = F.cross_entropy(
                    logits.reshape(B * T, C), tgt.reshape(B * T)
                )
                out[f"{name}_{side}"] = ce
                out["total"] = out["total"] + wt * ce

    return out


@torch.no_grad()
def compute_wm_metrics(
    preds: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """Lightweight per-head metrics, complement to compute_wm_loss.

    - action top-1 accuracy (self, opp)
    - numeric MSE (self, opp) — duplicates the loss but without reduction-mix
    - flag accuracy (self, opp) — threshold at 0
    - action-transition top-1 accuracy (only on frames where action changes).
    """
    metrics: Dict[str, float] = {}

    for side in ("self", "opp"):
        logits = preds[f"{side}_action_logits"]
        tgt = targets[f"{side}_action"].long()
        pred_idx = logits.argmax(dim=-1)
        metrics[f"action_{side}_acc"] = (pred_idx == tgt).float().mean().item()

        # Action-transition accuracy: compare against *previous* action to
        # mask "stayed the same" frames. First frame has no previous — drop.
        #   change_mask[t] = tgt[t] != tgt[t-1]   (for t >= 1)
        if tgt.size(1) >= 2:
            change = tgt[:, 1:] != tgt[:, :-1]
            if change.any():
                correct = (pred_idx[:, 1:] == tgt[:, 1:]) & change
                metrics[f"action_{side}_change_acc"] = (
                    correct.float().sum() / change.float().sum()
                ).item()
            else:
                metrics[f"action_{side}_change_acc"] = float("nan")

        pred_num = preds[f"{side}_numeric_pred"]
        tgt_num = targets[f"{side}_numeric"].float()
        if tgt_num.shape[-1] != pred_num.shape[-1]:
            # discretize_counters active — target is still full 13 cols, pred
            # has 8 continuous cols only. Slice target to match.
            CONT_COLS = [0, 1, 5, 6, 7, 8, 9, 12]
            tgt_num = tgt_num[..., CONT_COLS]
        metrics[f"numeric_{side}_mse"] = F.mse_loss(pred_num, tgt_num).item()

        flag_logits = preds[f"{side}_flags_logits"]
        tgt_flg = targets[f"{side}_flags"].float()
        flag_pred = (flag_logits > 0).float()
        metrics[f"flags_{side}_acc"] = (flag_pred == tgt_flg).float().mean().item()

        # Action-elapsed MAE in raw frames (pred is in scaled space, so
        # multiply by the default scale to report a human-readable number).
        elapsed_key = f"{side}_action_elapsed_pred"
        if elapsed_key in preds and f"{side}_action_elapsed" in targets:
            pred = preds[elapsed_key] * 30.0
            tgt = targets[f"{side}_action_elapsed"].float()
            metrics[f"action_elapsed_{side}_mae"] = (
                (pred - tgt).abs().mean().item()
            )

        # Discretized counter accuracies (when heads are present)
        for name in ("percent", "stock", "jumps", "hitlag", "hitstun", "elapsed"):
            lk = f"{side}_{name}_logits"
            tk = f"{side}_{name}_int"
            if lk in preds and tk in targets:
                pred_idx = preds[lk].argmax(dim=-1)
                tgt = targets[tk].long()
                metrics[f"{name}_{side}_acc"] = (pred_idx == tgt).float().mean().item()

    return metrics
