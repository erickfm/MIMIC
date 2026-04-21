"""Online-task protocol: episode start, episode end, terminal reward.

An OnlineTask is a streaming state machine. The actor pushes libmelee
GameStates one at a time; the task watches for scenario starts and ends,
and emits a reward at scenario completion.

Unlike the offline Task which has `tag_frames(replay)`, online tasks
don't know the future — they must decide episode boundaries from the
state-so-far. That's the whole point of online: we grade actual policy
behavior, not pre-tagged opportunities.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class EpisodeOutcome:
    """What a task returns when an episode ends."""
    terminal_reward: float             # added to the last frame's reward
    per_frame_reward: Optional[List[float]] = None  # dense rewards (same len as episode); default 0
    metadata: Dict[str, Any] = None


class OnlineTask(Protocol):
    id: str
    description: str
    def should_start(self, state_history) -> bool:
        """Given the streaming state history (latest frame last), should
        an episode begin at the current frame?"""
        ...

    def should_end(self, state_history, episode_start_idx: int) -> bool:
        """Should the episode ending at the current frame close now?"""
        ...

    def compute_outcome(self, state_history, episode_start_idx: int) -> EpisodeOutcome:
        """Score the completed episode. Called once after should_end
        returns True. Tasks that require a post-match .slp parse (e.g.
        the L-cancel ground-truth label) should stash the pending frame
        in metadata and implement `enrich_with_replay`.

        Returning `terminal_reward=float('nan')` signals 'reward deferred
        to enrich_with_replay'; the actor holds these episodes until the
        match's .slp is written.
        """
        ...

    def enrich_with_replay(self, episodes, slp_path, self_port: int) -> list:
        """Optional hook: read the .slp at `slp_path` post-match and
        finalize rewards for any episodes the task marked as pending.
        Default: identity (no enrichment needed).
        Returns the (possibly filtered) list of episodes."""
        return episodes
