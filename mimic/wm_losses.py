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

    # MSE on numeric columns (mean-reduced over all dims).
    for side in ("self", "opp"):
        pred = preds[f"{side}_numeric_pred"]
        tgt = targets[f"{side}_numeric"].float()
        out[f"numeric_{side}"] = F.mse_loss(pred, tgt)

    # BCE-with-logits on flags.
    for side in ("self", "opp"):
        logits = preds[f"{side}_flags_logits"]
        tgt = targets[f"{side}_flags"].float()
        out[f"flags_{side}"] = F.binary_cross_entropy_with_logits(logits, tgt)

    out["total"] = (
        w.action_self * out["action_self"]
        + w.action_opp * out["action_opp"]
        + w.numeric_self * out["numeric_self"]
        + w.numeric_opp * out["numeric_opp"]
        + w.flags_self * out["flags_self"]
        + w.flags_opp * out["flags_opp"]
    )
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
        metrics[f"numeric_{side}_mse"] = F.mse_loss(pred_num, tgt_num).item()

        flag_logits = preds[f"{side}_flags_logits"]
        tgt_flg = targets[f"{side}_flags"].float()
        flag_pred = (flag_logits > 0).float()
        metrics[f"flags_{side}_acc"] = (flag_pred == tgt_flg).float().mean().item()

    return metrics
