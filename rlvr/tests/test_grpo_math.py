"""GRPO math unit tests.

Gate: synthetic inputs with known gold answers. We verify:
  1. group_normalize produces unit-variance, zero-mean per group.
  2. Ratio-1 (theta == old) path: unclipped surrogate = advantage; loss
     = -mean(advantage) + beta * KL(theta, ref).
  3. Clipped-ratio path: out-of-band ratios are clipped symmetrically.
  4. KL estimator = 0 when theta == ref, positive otherwise.
  5. reward_mean / advantage_mean are diagnostic-correct.
"""
from __future__ import annotations

import pytest
import torch

from rlvr.train.grpo import GRPOConfig, grpo_loss, group_normalize


def test_group_normalize_two_groups_hand_computed():
    # Two groups of 4 rollouts. Group 0: [1, 1, 1, 1] (zero advantage).
    # Group 1: [0, 0, 1, 1] (mean=0.5, std=0.5, advantages [-1, -1, 1, 1]).
    rewards = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0])
    adv = group_normalize(rewards, group_size=4)
    assert adv[:4].abs().max().item() < 1e-6   # group 0 all zero
    assert torch.allclose(adv[4:], torch.tensor([-1.0, -1.0, 1.0, 1.0]), atol=1e-6)


def test_group_normalize_no_std_variant():
    rewards = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    adv = group_normalize(rewards, group_size=4, normalize_by_std=False)
    # Group 0 mean=0.5 -> advantages [-0.5, -0.5, 0.5, 0.5]
    # Group 1 mean=0.25 -> [-0.25, -0.25, -0.25, 0.75]
    expected = torch.tensor([-0.5, -0.5, 0.5, 0.5, -0.25, -0.25, -0.25, 0.75])
    assert torch.allclose(adv, expected, atol=1e-6)


def test_loss_when_ratio_is_one_and_ref_equals_theta():
    """theta == old -> ratio = 1 everywhere, unclipped surrogate is just
    the advantage. theta == ref -> KL = 0. So loss = -mean(advantage)."""
    # 2 groups of 4 rollouts.
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    logprobs_theta = torch.zeros(8)   # all 0, so exp(0) = 1
    logprobs_old = torch.zeros(8)     # same -> ratio = 1
    logprobs_ref = torch.zeros(8)     # same -> KL = 0

    out = grpo_loss(
        logprobs_theta, logprobs_old, logprobs_ref, rewards,
        group_size=4,
    )
    # advantages are mean-0-std-1 within each group of [1,0,1,0] and
    # [1,1,0,0]: in each group, std = 0.5 (unbiased=False), values
    # (r-0.5)/0.5 = -1 or 1, mean 0. So mean(advantage) = 0 -> pg_loss
    # = 0 -> total loss = 0 (KL also 0).
    assert abs(out["loss"].item()) < 1e-6, f"loss={out['loss'].item()}"
    assert abs(out["kl"].item()) < 1e-6
    assert abs(out["pg_loss"].item()) < 1e-6
    assert abs(out["reward_mean"].item() - 0.5) < 1e-6


def test_kl_positive_when_theta_differs_from_ref():
    logprobs_theta = torch.zeros(4)          # e.g. prob ~1 for each
    logprobs_old = torch.zeros(4)
    # Ref has lower log-probs -> theta has diverged upward
    logprobs_ref = torch.tensor([-0.5, -0.5, -0.5, -0.5])
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])

    out = grpo_loss(
        logprobs_theta, logprobs_old, logprobs_ref, rewards,
        group_size=4,
    )
    # KL(theta || ref) = mean(exp(ref - theta) - (ref - theta) - 1)
    # = mean(exp(-0.5) - (-0.5) - 1) = exp(-0.5) + 0.5 - 1 ≈ 0.1065
    assert abs(out["kl"].item() - 0.106531) < 1e-4


def test_clip_fraction_counts_out_of_band_ratios():
    # theta_logprobs - old_logprobs = 0.5 -> ratio = exp(0.5) ≈ 1.649
    # With clip_eps=0.2, everything outside [0.8, 1.2] is clipped.
    # 1.649 > 1.2 -> all rollouts clipped.
    logprobs_theta = torch.tensor([0.5, 0.5, 0.5, 0.5])
    logprobs_old = torch.zeros(4)
    logprobs_ref = torch.zeros(4)
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])

    out = grpo_loss(
        logprobs_theta, logprobs_old, logprobs_ref, rewards,
        group_size=4, cfg=GRPOConfig(clip_eps=0.2),
    )
    assert out["clip_frac"].item() == 1.0
    # Ratio mean should equal exp(0.5)
    assert abs(out["ratio_mean"].item() - 2.718281828 ** 0.5) < 1e-4


def test_gradient_flows_through_theta():
    """loss.backward() must produce gradients on logprobs_theta."""
    logprobs_theta = torch.zeros(4, requires_grad=True)
    logprobs_old = torch.zeros(4)
    logprobs_ref = torch.zeros(4)
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])

    out = grpo_loss(
        logprobs_theta, logprobs_old, logprobs_ref, rewards, group_size=4,
    )
    out["loss"].backward()
    assert logprobs_theta.grad is not None
    assert torch.isfinite(logprobs_theta.grad).all()


def test_group_size_mismatch_raises():
    rewards = torch.zeros(7)
    lp = torch.zeros(7)
    with pytest.raises(AssertionError):
        grpo_loss(lp, lp, lp, rewards, group_size=4)
