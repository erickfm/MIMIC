"""Per-frame trajectory buffer + GAE advantages for online RL.

An `Episode` is a contiguous window of frames inside a skill scenario.
Each frame stores: observation tensor dict (for re-forward on update),
sampled head indices, logprob under the policy at sampling time (old),
logprob under the frozen reference, and frame-local reward (usually 0
except at terminal frames).

GAE is computed per-episode with no bootstrap beyond the episode end —
we use pure Monte-Carlo returns since episodes are short (10-60 frames
for L-cancel / shield), so high-variance isn't a real concern.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


@dataclass
class FrameRecord:
    obs: Dict[str, torch.Tensor]       # frame dict suitable for build_context_batch
    sampled_indices: torch.Tensor      # shape (4,) int64: main, shldr, cdir, btn
    logprob_old: torch.Tensor          # scalar tensor, sum over 4 heads
    logprob_ref: torch.Tensor          # scalar tensor, sum over 4 heads
    reward: float = 0.0
    # carried for diagnostics / credit assignment debug
    game_frame_id: int = 0


@dataclass
class Episode:
    task_id: str
    frames: List[FrameRecord] = field(default_factory=list)
    terminal_reward: float = 0.0       # dense reward at final frame (if any)
    start_game_frame: int = 0
    end_game_frame: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.frames)

    def returns(self, gamma: float = 1.0) -> torch.Tensor:
        """Discounted MC returns, backward-scanned. Terminal reward lives
        on the final frame; per-frame rewards on others (usually 0).
        With gamma=1 (default for short episodes) this is just a cumulative
        sum from the end."""
        n = len(self.frames)
        out = torch.zeros(n)
        r_next = 0.0
        # Terminal reward is the final frame's reward + terminal_reward.
        for t in range(n - 1, -1, -1):
            reward_t = self.frames[t].reward
            if t == n - 1:
                reward_t = reward_t + self.terminal_reward
            r_next = reward_t + gamma * r_next
            out[t] = r_next
        return out


def group_normalize_across_episodes(
    returns_per_episode: List[torch.Tensor],
    eps: float = 1e-8,
) -> List[torch.Tensor]:
    """Pool all frame returns across episodes, z-score, split back.
    This gives us GRPO-style advantages at the batch level — no critic
    needed, and the group is 'all frames in this update batch.'"""
    if not returns_per_episode:
        return []
    flat = torch.cat(returns_per_episode, dim=0).float()
    mean = flat.mean()
    std = flat.std(unbiased=False)
    normalized_flat = (flat - mean) / (std + eps)
    out = []
    i = 0
    for r in returns_per_episode:
        n = r.shape[0]
        out.append(normalized_flat[i:i + n])
        i += n
    return out
