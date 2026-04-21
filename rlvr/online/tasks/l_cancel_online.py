"""Online L-cancel task (post-match replay enrichment).

libmelee's PlayerState doesn't expose the game's `post.l_cancel` column;
it's only in the raw .slp event stream (peppi exposes it). So the online
L-cancel task works in two passes:

  1. During the match, identify aerial-attack -> landing episodes and
     mark them pending (terminal_reward = NaN).
  2. After the match, the actor reads the just-written .slp with peppi,
     looks up `post.l_cancel` at each episode's landing frame, and sets
     the terminal reward (1 = success, 0 = fail, episode discarded if
     the landing wasn't L-cancel-eligible at all).

This relies on Dolphin's `save_replays=True` writing a .slp per match,
which the inference stack already uses.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import List, Optional

import numpy as np

from rlvr.online.episode import EpisodeOutcome, OnlineTask
from rlvr.online.trajectory import Episode
from rlvr.state.peppi_adapter import Replay


log = logging.getLogger("rlvr.online.l_cancel")


# libmelee Action enum values
AERIAL_STATES = {65, 66, 67, 68, 69}
LANDING_AIR_STATES = {70, 71, 72, 73, 74}
GENERIC_LANDING_STATES = {42, 43}

TASK_ID = "l_cancel_online"


class LCancelOnlineTask:
    id = TASK_ID
    description = (
        "Online L-cancel: episodes = bot aerial-attack -> landing. "
        "Reward deferred to post-match .slp parse for peppi ground-truth."
    )

    def __init__(self, self_port: int = 1):
        self.self_port = self_port

    def _self_ps(self, state_history, offset: int = -1):
        gs = state_history[offset]
        for p in gs.players:
            if p.port == self.self_port:
                return p
        return None

    def should_start(self, state_history) -> bool:
        if len(state_history) < 2:
            return False
        prev = self._self_ps(state_history, -2)
        curr = self._self_ps(state_history, -1)
        if prev is None or curr is None:
            return False
        was_aerial = prev.action in AERIAL_STATES
        is_aerial = curr.action in AERIAL_STATES
        return (is_aerial and not was_aerial
                and not curr.on_ground
                and curr.hitstun_frames_left == 0)

    def should_end(self, state_history, episode_start_idx: int) -> bool:
        curr = self._self_ps(state_history, -1)
        if curr is None:
            return True
        if curr.action in LANDING_AIR_STATES:
            return True
        if curr.action in GENERIC_LANDING_STATES:
            return True
        if curr.action not in AERIAL_STATES:
            return True
        return False

    def compute_outcome(self, state_history, episode_start_idx: int) -> EpisodeOutcome:
        curr = self._self_ps(state_history, -1)
        if curr is None:
            return EpisodeOutcome(terminal_reward=0.0, metadata={"result": "aborted"})
        landing_action = int(curr.action)
        landing_frame_id = int(state_history[-1].frame_idx)
        if curr.action in LANDING_AIR_STATES:
            # Ground truth unknown until post-match enrichment.
            return EpisodeOutcome(
                terminal_reward=float("nan"),
                metadata={
                    "pending": True,
                    "landing_frame_id": landing_frame_id,
                    "landing_state": landing_action,
                    "result": "pending_lcancel_check",
                },
            )
        # Not a LANDING_AIR_* — either a non-L-cancel landing or an
        # aerial interruption. Skip training signal on these.
        return EpisodeOutcome(
            terminal_reward=0.0,
            metadata={
                "pending": False,
                "result": "ineligible",
                "landing_state": landing_action,
            },
        )

    def enrich_with_replay(
        self, episodes: List[Episode], slp_path: Path, self_port: int,
    ) -> List[Episode]:
        """Read the .slp with peppi and assign terminal_reward for any
        pending episodes. Discards episodes where post.l_cancel == 0 at
        the landing frame (the landing wasn't a real L-cancel case)."""
        pending = [
            i for i, ep in enumerate(episodes)
            if ep.metadata.get("pending") and math.isnan(ep.terminal_reward)
        ]
        if not pending:
            return episodes
        try:
            replay = Replay(Path(slp_path))
        except Exception as e:
            log.warning("couldn't parse %s for enrichment: %s", slp_path, e)
            # Drop pending episodes we can't score.
            return [ep for i, ep in enumerate(episodes) if i not in pending]

        # Locate the self-player column by port.
        target_pi = None
        for pi, port in enumerate(replay.player_ports):
            if port == self_port:
                target_pi = pi
                break
        if target_pi is None:
            log.warning("port %d not found in %s; dropping %d pending episodes",
                        self_port, slp_path, len(pending))
            return [ep for i, ep in enumerate(episodes) if i not in pending]

        lc_col = replay.l_cancel_per_player(target_pi)  # uint8 array
        frame_ids = replay.frame_ids                     # sorted (dedup'd)

        out = []
        n_success = n_fail = n_ineligible = 0
        for i, ep in enumerate(episodes):
            if i not in pending:
                out.append(ep)
                continue
            lf_id = ep.metadata["landing_frame_id"]
            # Map frame_id -> dedup index
            idx = int(np.searchsorted(frame_ids, lf_id))
            if idx >= len(frame_ids) or int(frame_ids[idx]) != lf_id:
                log.warning("frame_id %d not in .slp (len %d); skipping",
                            lf_id, len(frame_ids))
                n_ineligible += 1
                continue
            lc = int(lc_col[idx])
            if lc == 1:
                ep.terminal_reward = 1.0
                ep.metadata["result"] = "l_cancel_success"
                ep.metadata["pending"] = False
                ep.metadata["lc_code"] = 1
                out.append(ep)
                n_success += 1
            elif lc == 2:
                ep.terminal_reward = 0.0
                ep.metadata["result"] = "l_cancel_missed"
                ep.metadata["pending"] = False
                ep.metadata["lc_code"] = 2
                out.append(ep)
                n_fail += 1
            else:
                n_ineligible += 1

        log.info("enrich: %d success, %d fail, %d ineligible (dropped)",
                 n_success, n_fail, n_ineligible)
        return out
