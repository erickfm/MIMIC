"""Multi-step PPO update for online RL.

Given a batch of Episodes (each with stored T-frame context snapshots
per frame + old logprobs + ref logprobs), this:

  1. Computes discounted MC returns per episode.
  2. Group-normalizes across all frames in the batch -> advantages.
  3. For each frame, re-forwards the policy on the stored context +
     computes current logprob of the sampled action under the 4 heads.
  4. PPO-clipped surrogate + Schulman KL estimator to the frozen ref.

Memory: episode frames are stored CPU-side. For the update we build
chunks of M frames, move to GPU, forward, discard — controls peak
memory.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn.functional as F

from rlvr.online.trajectory import Episode, group_normalize_across_episodes


@dataclass
class OnlinePPOConfig:
    clip_eps: float = 0.2
    kl_beta: float = 0.01
    gamma: float = 1.0
    advantage_eps: float = 1e-8
    minibatch_frames: int = 64        # chunk size for GPU forward
    normalize_by_std: bool = True


def _last_logits(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 3:
        return t[:, -1, :]
    return t


def _logprob_at_indices(logits_dict, sampled_indices):
    """Sum log-prob across the 4 heads at the sampled indices.
    logits_dict values: (B, T, V) or (B, V).
    sampled_indices: (B, 4) long tensor."""
    def _lp(head, idx):
        lg = _last_logits(head).float()
        lg = torch.nan_to_num(lg, nan=-1e9, posinf=1e9, neginf=-1e9)
        logp = F.log_softmax(lg, dim=-1)
        return logp.gather(-1, idx.unsqueeze(-1)).squeeze(-1)

    m_idx = sampled_indices[:, 0]
    s_idx = sampled_indices[:, 1]
    c_idx = sampled_indices[:, 2]
    b_idx = sampled_indices[:, 3]
    return (
        _lp(logits_dict["main_xy"], m_idx)
        + _lp(logits_dict["shoulder_val"], s_idx)
        + _lp(logits_dict["c_dir_logits"], c_idx)
        + _lp(logits_dict["btn_logits"], b_idx)
    )


def _compute_advantages(episodes: List[Episode], gamma: float, normalize_by_std: bool,
                        eps: float) -> torch.Tensor:
    """Flatten per-frame returns across all episodes + z-score."""
    returns_per_ep = [ep.returns(gamma=gamma) for ep in episodes]
    if not returns_per_ep:
        return torch.empty(0)
    flat = torch.cat(returns_per_ep, dim=0).float()
    mean = flat.mean()
    if normalize_by_std:
        std = flat.std(unbiased=False)
        return (flat - mean) / (std + eps)
    return flat - mean


def ppo_update(
    model,
    episodes: List[Episode],
    optimizer,
    cfg: OnlinePPOConfig = OnlinePPOConfig(),
    device: str = "cuda",
) -> Dict[str, float]:
    """Run one PPO epoch over the collected episodes.

    Returns a metrics dict averaged over the batch (mean reward, KL,
    clip frac, loss, etc.).
    """
    if not episodes:
        return {"n_frames": 0, "n_episodes": 0}

    # --- flatten frames across episodes and stack tensors ---
    all_frames = []
    ep_boundaries = []
    for ep in episodes:
        ep_boundaries.append(len(all_frames))
        all_frames.extend(ep.frames)
    N = len(all_frames)
    if N == 0:
        return {"n_frames": 0, "n_episodes": len(episodes)}

    advantages = _compute_advantages(
        episodes, cfg.gamma, cfg.normalize_by_std, cfg.advantage_eps
    ).to(device).detach()

    sampled_indices = torch.stack(
        [fr.sampled_indices for fr in all_frames], dim=0
    ).to(device)                                                 # (N, 4)
    logprobs_old = torch.stack(
        [fr.logprob_old for fr in all_frames], dim=0
    ).to(device).float().detach()                                # (N,)
    logprobs_ref = torch.stack(
        [fr.logprob_ref for fr in all_frames], dim=0
    ).to(device).float().detach()                                # (N,)
    rewards_end = torch.tensor(
        [fr.reward + (ep.terminal_reward if i == len(ep.frames) - 1 else 0.0)
         for ep in episodes
         for i, fr in enumerate(ep.frames)],
        dtype=torch.float32,
    ).to(device)

    # --- re-forward in minibatches with gradient ---
    all_logprobs_theta = []
    model.train()
    optimizer.zero_grad()
    minibatch = cfg.minibatch_frames
    total_loss = 0.0
    total_kl = 0.0
    total_clip = 0.0
    total_n = 0

    for start in range(0, N, minibatch):
        end = min(start + minibatch, N)
        chunk = all_frames[start:end]
        # Build (B, T, ...) batch from chunk.obs dicts.
        batch = {}
        for k in chunk[0].obs:
            tensors = [fr.obs[k].unsqueeze(0) for fr in chunk]   # each (1, T, ...)
            batch[k] = torch.cat(tensors, dim=0).to(device)

        logits = model(batch)
        lp_theta = _logprob_at_indices(logits, sampled_indices[start:end])
        all_logprobs_theta.append(lp_theta.detach())

        # Per-chunk PPO objective (bptt cheap since per-chunk).
        adv_chunk = advantages[start:end]
        lp_old_chunk = logprobs_old[start:end]
        lp_ref_chunk = logprobs_ref[start:end]

        log_ratio = lp_theta - lp_old_chunk
        ratio = torch.exp(log_ratio)
        unclipped = ratio * adv_chunk
        clipped = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_chunk
        pg_obj = torch.minimum(unclipped, clipped)
        pg_loss = -pg_obj.mean()

        # Schulman low-bias KL
        lr_diff = lp_ref_chunk - lp_theta
        kl = (lr_diff.exp() - lr_diff - 1.0).mean()

        loss = pg_loss + cfg.kl_beta * kl
        (loss * (end - start) / N).backward()   # accumulate, scaled by chunk frac

        total_loss += float(loss.item()) * (end - start)
        total_kl += float(kl.item()) * (end - start)
        total_clip += float(
            ((ratio < 1 - cfg.clip_eps) | (ratio > 1 + cfg.clip_eps)).float().mean().item()
        ) * (end - start)
        total_n += (end - start)

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return {
        "loss": total_loss / max(total_n, 1),
        "kl": total_kl / max(total_n, 1),
        "clip_frac": total_clip / max(total_n, 1),
        "reward_mean": float(rewards_end.mean().item()),
        "advantage_mean": float(advantages.mean().item()),
        "advantage_std": float(advantages.std(unbiased=False).item()),
        "grad_norm": float(grad_norm),
        "n_frames": total_n,
        "n_episodes": len(episodes),
    }
