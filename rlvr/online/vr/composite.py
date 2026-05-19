"""VRModule interface + CompositeVRTask.

A VR (verifiable reward) is a streaming reward component — a `VRModule`.
Each match the actor runs one `CompositeVRTask` — which implements the
existing `OnlineTask` protocol (`rlvr/online/episode.py`) — and the
composite holds a weighted list of VRModules. Every in-game frame it sums
their per-frame rewards into one reward stream over the whole-match
episode.

Architecture (see the plan file / docs/vr-proposals/): episode = one whole
match. The actor opens the episode on the first in-game frame, calls
`observe()` every frame, and scores+closes it at the match-end menu
transition. `should_end` never fires mid-match.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

from rlvr.online.episode import EpisodeOutcome


class VRModule:
    """Base class for a verifiable-reward component.

    Lifecycle, driven by `CompositeVRTask`:
      - `reset()`    — once, at match start.
      - `observe()`  — every in-game frame; returns this VR's *unweighted*
                       per-frame reward (0 on most frames; situational VRs
                       run their streaming state machine internally and
                       emit on the resolving frame).
      - `finalize()` — once, at match end; default 0.0.
      - `metadata()` — once, at match end; per-match diagnostics.
    """

    id: str = "vr"

    def reset(self) -> None:
        """Clear per-match state. Override if the VR holds state."""

    def observe(self, state_history) -> float:
        """Per-frame unweighted reward contribution. Must be overridden."""
        raise NotImplementedError

    def finalize(self, state_history) -> float:
        """End-of-match terminal reward contribution. Default 0."""
        return 0.0

    def metadata(self) -> Dict[str, Any]:
        """Per-match diagnostics. Default empty."""
        return {}


class CompositeVRTask:
    """Runs a weighted set of `VRModule`s as one whole-match episode.

    Implements the `OnlineTask` protocol:
      - `should_start`    — True on the first in-game frame; resets modules.
      - `observe`         — sum `wᵢ · moduleᵢ.observe(...)` per frame.
      - `should_end`      — always False; the actor scores+closes the
                            episode at the match-end menu transition.
      - `compute_outcome` — the accumulated per-frame vector + summed
                            `finalize` terminal.
    """

    def __init__(self, modules: Sequence[VRModule],
                 weights: Sequence[float], self_port: int = 1):
        if len(modules) != len(weights):
            raise ValueError(
                f"modules ({len(modules)}) and weights ({len(weights)}) "
                f"length mismatch")
        self.modules: List[VRModule] = list(modules)
        self.weights: List[float] = [float(w) for w in weights]
        self.self_port = self_port
        self.id = "+".join(m.id for m in self.modules) or "composite_vr"
        self.description = "Composite VR: " + ", ".join(
            f"{m.id}*{w:g}" for m, w in zip(self.modules, self.weights))
        self._reward_vec: List[float] = []

    # -- OnlineTask protocol -------------------------------------------------
    def should_start(self, state_history) -> bool:
        """Open the whole-match episode on the first in-game frame."""
        self._reward_vec = []
        for m in self.modules:
            m.reset()
        return True

    def observe(self, state_history) -> float:
        """Per-frame: accumulate the weighted sum over every module."""
        total = 0.0
        for m, w in zip(self.modules, self.weights):
            total += w * float(m.observe(state_history))
        self._reward_vec.append(total)
        return total

    def should_end(self, state_history, episode_start_idx: int) -> bool:
        """Never close mid-match — the actor scores+closes the whole-match
        episode at the match-end menu transition."""
        return False

    def compute_outcome(self, state_history, episode_start_idx: int) -> EpisodeOutcome:
        """Return the accumulated per-frame reward vector plus the summed
        terminal (`finalize`) contributions."""
        terminal = 0.0
        meta: Dict[str, Any] = {}
        for m, w in zip(self.modules, self.weights):
            terminal += w * float(m.finalize(state_history))
            md = m.metadata()
            if md:
                meta[m.id] = md
        per_frame = list(self._reward_vec)
        meta["n_frames"] = len(per_frame)
        meta["reward_sum"] = float(sum(per_frame) + terminal)
        outcome = EpisodeOutcome(
            terminal_reward=float(terminal),
            per_frame_reward=per_frame,
            metadata=meta,
        )
        self._reward_vec = []
        return outcome

    def enrich_with_replay(self, episodes, slp_path, self_port: int) -> list:
        """No post-match .slp enrichment — every VR scores from live
        state. Identity."""
        return episodes
