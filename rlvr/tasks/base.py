"""Task and Verifier Protocols.

A Task is a skill the model can attempt, identifiable from game state.
Each task module exports one Task and one Verifier; both share a
`task_id` string. The tagger calls `Task.tag_frames(replay)` once per
replay at corpus-scan time to find prompt-relevant frames. The training
loop calls `Verifier(prompt, sampled_ctrl)` per-rollout to grade a
policy-sampled action.

Verifier is pure (no side effects, no randomness, no external state) —
same input always produces the same reward.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Protocol

from rlvr.state.gamestate import ControllerInput


@dataclass(frozen=True, slots=True)
class FrameRow:
    """One row emitted by Task.tag_frames — one prompt worth of data.

    The tagger serializes these to events.parquet. The sampler reads
    rows back and joins against a replay id to reconstruct prompt
    context.
    """
    replay_id: str
    player_port: int                  # which port the task is about (1-indexed)
    frame_idx: int                    # Slippi frame_id, matches peppi frame_ids
    task_id: str
    character: int                    # libmelee Character enum value (self)
    stage: int                        # libmelee Stage enum value
    task_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Prompt:
    """One prompt consumed by the rollout harness.

    `state_context` is the replay's own state trajectory for the T
    frames ending at `frame_idx`. The model will consume this as input
    and emit one controller action.
    """
    task_id: str
    replay_id: str
    player_port: int
    frame_idx: int
    state_context: tuple             # tuple of GameState, length up to T
    task_metadata: Dict[str, Any]


class Task(Protocol):
    id: str
    description: str
    def tag_frames(self, replay) -> Iterator[FrameRow]:
        """Walk a parsed peppi Replay and yield one FrameRow per prompt-
        relevant frame for this task.

        This is the ONLY place in the pipeline that looks ahead or
        across frames; it runs once at tagger time. Free to use any
        peppi column including ground-truth labels like post.l_cancel.
        """
        ...


class Verifier(Protocol):
    task_id: str
    def __call__(self, prompt: Prompt, sampled_ctrl: ControllerInput) -> float:
        """Grade a policy-sampled controller action for this prompt.
        Returns a reward in [0, 1]."""
        ...
