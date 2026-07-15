"""Online L-cancel task (post-match replay enrichment).

The reward is REALIZED AVOIDABLE LAG, not the engine's `post.l_cancel`
flag. The flag miscounts: it marks ledge-slides and hit-interrupted
landings as "misses" even though they cost zero frames (see
tools/lcancel_analysis.py and docs/research-notes-2026-06-29.md). The
correct target is

    avoidable_lag = max(0, realized_landing_lag - cancelled_min[move])

where realized_landing_lag is the number of frames actually spent in
the LANDING_AIR_* state and cancelled_min is the per-move L-cancelled
lag floor. reward = 1.0 iff avoidable_lag == 0.

Live libmelee state can't measure this (the landing run's length isn't
known until it ends, and rollbacks can rewrite it), so the task works
in two passes:

  1. During the match, identify aerial-attack -> landing episodes and
     mark them pending (terminal_reward = NaN).
  2. After the match, the actor reads the just-written .slp with peppi,
     bounds the landing-state run at each episode's landing frame, and
     scores avoidable lag. Landings the opponent interrupted (exit into
     damage/dead states) carry no input-timing signal and are dropped.

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

# Per-move L-cancelled landing-lag floor (frames), keyed by
# LANDING_AIR state: 70=NAIR 71=FAIR 72=BAIR 73=UAIR 74=DAIR.
# Measured as the median realized lag of clean (grounded-exit,
# flag=1) landings over the master-Fox corpus
# (tools/lcancel_analysis.py, docs/research-notes-2026-07-05.md).
# FOX-SPECIFIC — other characters have different landing lag.
CANCELLED_MIN = {70: 7, 71: 11, 72: 10, 73: 9, 74: 9}

# Landing-run exit categories (mirrors tools/lcancel_analysis.py).
# Exits into damage/dead mean the opponent cut the landing short —
# realized lag is opponent-determined there, not an input-timing
# signal, so those episodes are dropped.
DEAD_STATES = set(range(0, 11))
DAMAGE_STATES = set(range(75, 92))

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
        # Not a LANDING_AIR_* — either a non-L-cancel landing (e.g. an
        # AUTOCANCEL into GENERIC_LANDING, which is optimal play) or an
        # aerial interruption (hit/grabbed — opponent-caused). Neither
        # carries L-cancel timing signal. terminal_reward must be NaN,
        # NOT 0.0: with group-normalized advantages a 0.0 reward
        # PUNISHES these episodes as hard as a missed L-cancel, i.e.
        # trains against autocancels. NaN + pending=False marks the
        # episode as unscoreable so the actor discards it.
        return EpisodeOutcome(
            terminal_reward=float("nan"),
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
        pending episodes by realized avoidable lag (see module
        docstring). Drops episodes whose landing can't be scored:
        frame not in the replay, live/replay state mismatch (rollback),
        replay truncated mid-landing, or opponent-interrupted exit."""
        pending = [
            i for i, ep in enumerate(episodes)
            if ep.metadata.get("pending") and math.isnan(ep.terminal_reward)
        ]
        if not pending:
            return episodes
        # libmelee closes the .slp on match-end transition, but the
        # OS may still be flushing dirty pages when we try to parse.
        # peppi surfaces this as "I/O error: failed to fill whole buffer".
        # Retry a few times with backoff before giving up.
        import time as _t
        replay = None
        last_err: Optional[Exception] = None
        for attempt in range(6):  # ~ 0.05 + 0.1 + 0.2 + 0.4 + 0.8 + 1.6 = 3.15s max
            try:
                replay = Replay(Path(slp_path))
                break
            except Exception as e:
                last_err = e
                _t.sleep(0.05 * (2 ** attempt))
        if replay is None:
            log.warning("couldn't parse %s for enrichment after 6 retries: %s",
                        slp_path, last_err)
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

        st = np.asarray(replay._post[target_pi]["state"]).astype(int)
        lc_col = replay.l_cancel_per_player(target_pi)  # uint8 array
        frame_ids = replay.frame_ids                     # sorted (dedup'd)
        n = len(st)

        out = []
        n_success = n_fail = n_dropped = 0
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
                n_dropped += 1
                continue
            landing_state = int(st[idx])
            if landing_state not in CANCELLED_MIN:
                # Live obs said LANDING_AIR but the replay disagrees
                # (rollback rewrote the frame). No ground truth — drop.
                n_dropped += 1
                continue
            # Bound the landing-state run [t, j).
            t = idx
            while t > 0 and st[t - 1] == landing_state:
                t -= 1
            j = idx
            while j < n and st[j] == landing_state:
                j += 1
            if j >= n:
                # Replay ended mid-landing: lag truncated, can't score.
                n_dropped += 1
                continue
            exit_state = int(st[j])
            if exit_state in DEAD_STATES or exit_state in DAMAGE_STATES:
                # Opponent interrupted the landing — no timing signal.
                n_dropped += 1
                continue
            lag = j - t
            avoidable = max(0, lag - CANCELLED_MIN[landing_state])
            success = avoidable == 0
            ep.terminal_reward = 1.0 if success else 0.0
            ep.metadata["result"] = ("l_cancel_success" if success
                                     else "l_cancel_missed")
            ep.metadata["pending"] = False
            ep.metadata["realized_lag"] = lag
            ep.metadata["avoidable_lag"] = avoidable
            ep.metadata["exit_state"] = exit_state
            # Engine flag kept for diagnostics only — NOT the reward.
            ep.metadata["lc_code"] = int(lc_col[idx])
            out.append(ep)
            if success:
                n_success += 1
            else:
                n_fail += 1

        log.info("enrich: %d success, %d miss, %d unscorable (dropped)",
                 n_success, n_fail, n_dropped)
        return out
