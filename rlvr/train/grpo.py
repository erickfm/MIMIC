"""GRPO loss for single-step RLVR (DeepSeek / Open-R1 style).

Given a batch of N rollouts per prompt, N * B total rollouts:
  1. Group-normalize rewards within each prompt's group -> advantages.
  2. PPO-clipped surrogate objective.
  3. Schulman low-bias KL estimator against a frozen reference model.
  4. loss = -mean(surrogate) + beta * mean(kl).

All tensors are one-dim over the (B*N,) axis. No time dimension.

Reference for the canonical GRPO formulation: DeepSeek R1 + TRL
GRPOTrainer. The math here is the same; we just strip the token loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class GRPOConfig:
    clip_eps: float = 0.2
    kl_beta: float = 0.01
    advantage_eps: float = 1e-8    # added to group std to avoid /0
    # If True, shift/scale by group mean+std. If False, only mean-center
    # (Dr. GRPO variant). Default: both, matching reference GRPO.
    normalize_by_std: bool = True


def group_normalize(rewards: torch.Tensor, group_size: int, eps: float = 1e-8,
                    normalize_by_std: bool = True) -> torch.Tensor:
    """Reshape (B*N,) -> (B, N), z-score within group, flatten back."""
    assert rewards.dim() == 1, f"rewards must be 1-D, got {rewards.shape}"
    total = rewards.shape[0]
    assert total % group_size == 0, (
        f"total rewards ({total}) not divisible by group_size ({group_size})"
    )
    B = total // group_size
    r = rewards.view(B, group_size)
    mean = r.mean(dim=1, keepdim=True)
    centered = r - mean
    if normalize_by_std:
        std = r.std(dim=1, keepdim=True, unbiased=False)
        adv = centered / (std + eps)
    else:
        adv = centered
    return adv.view(total)


def grpo_loss(
    logprobs_theta: torch.Tensor,     # (B*N,) — grad flows
    logprobs_old: torch.Tensor,       # (B*N,) — detached snapshot at rollout
    logprobs_ref: torch.Tensor,       # (B*N,) — detached, from frozen ref
    rewards: torch.Tensor,            # (B*N,) — from verifier in [0, 1]
    group_size: int,
    cfg: GRPOConfig = GRPOConfig(),
) -> Dict[str, torch.Tensor]:
    """Compute the GRPO loss + diagnostics.

    Returns dict with: loss (scalar), pg_loss, kl, advantage_mean,
    advantage_std, ratio_mean, clip_frac, reward_mean.
    """
    assert logprobs_theta.shape == logprobs_old.shape == logprobs_ref.shape == rewards.shape

    advantages = group_normalize(
        rewards, group_size, cfg.advantage_eps, cfg.normalize_by_std
    ).detach()

    # Importance ratio pi_theta / pi_old
    log_ratio = logprobs_theta - logprobs_old
    ratio = torch.exp(log_ratio)

    # Clipped PPO surrogate (maximize -> negated for loss)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * advantages
    # GRPO surrogate is the min (since advantage can be +/- and we clip
    # both sides).
    pg_obj = torch.minimum(unclipped, clipped)
    pg_loss = -pg_obj.mean()

    # Schulman low-bias KL estimator:
    #   KL(theta || ref) ~= exp(lp_ref - lp_theta) - (lp_ref - lp_theta) - 1
    # Detaches the ref side (already a no-grad snapshot); grad flows
    # into theta's log-probs.
    lr_diff = logprobs_ref - logprobs_theta
    kl = (lr_diff.exp() - lr_diff - 1.0).mean()

    loss = pg_loss + cfg.kl_beta * kl

    clip_frac = ((ratio < 1.0 - cfg.clip_eps) |
                 (ratio > 1.0 + cfg.clip_eps)).float().mean()

    return {
        "loss": loss,
        "pg_loss": pg_loss.detach(),
        "kl": kl.detach(),
        "advantage_mean": advantages.mean().detach(),
        "advantage_std": advantages.std(unbiased=False).detach(),
        "ratio_mean": ratio.mean().detach(),
        "clip_frac": clip_frac.detach(),
        "reward_mean": rewards.mean().detach(),
    }
