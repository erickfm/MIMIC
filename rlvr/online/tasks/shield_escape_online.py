"""Online escape-pressured-shield task.

Episode start: bot is in SHIELD-family, shield_strength dropped by
>= DAMAGE_DELTA_THRESHOLD in the last 1-3 frames (absorbed a hit),
AND current shield strength < SMALL_SHIELD_THRESHOLD (small shield).
Episode end: bot exits SHIELD-family (roll, spotdodge, release, hit-out,
jump, grab, OoS attack, or shield break).

Terminal reward:
  +1.0   clean escape (ROLL / SPOTDODGE / SHIELD_RELEASE / jump / grab /
         OoS smash)
   0.0   took a hit out of shield (damage family) — neutral, didn't
         escape cleanly but didn't break either
  -1.0   shield broke (SHIELD_BREAK_*)

All reward is in the task's own observable state — no post-match parse.
"""
from __future__ import annotations

from rlvr.online.episode import EpisodeOutcome, OnlineTask


TASK_ID = "shield_escape_online"

SHIELD_FAMILY = {178, 179, 180, 181, 182}
SHIELD_ACTIONABLE = {178, 179, 182}   # START, SHIELD, REFLECT
SHIELD_STUN = 181
ESCAPE_STATES = {188, 189, 196, 197, 198, 233, 234, 235}   # rolls + spotdodge family
SHIELD_BREAK_STATES = {205, 206, 207, 208, 209, 210, 211}
DAMAGE_STATES = set(range(75, 91))

SMALL_SHIELD_THRESHOLD = 30.0
DAMAGE_DELTA_THRESHOLD = 8.0
DAMAGE_DELTA_LOOKBACK = 3


class ShieldEscapeOnlineTask:
    id = TASK_ID
    description = (
        f"Online shield-escape: episode starts when bot absorbs a hit "
        f"of >= {DAMAGE_DELTA_THRESHOLD} strength leaving shield "
        f"< {SMALL_SHIELD_THRESHOLD}. Ends on any shield-exit. "
        f"Reward +1 clean escape, 0 hit-out, -1 shield break."
    )

    def __init__(self, self_port: int = 1):
        self.self_port = self_port
        # Track shield-strength history per-step so should_start can see
        # the last-N shield values without looking through full state.
        # The history_len is bounded by the actor's state_history
        # deque; we just read the last few GameStates ourselves.

    def _self_ps(self, state_history, offset: int = -1):
        gs = state_history[offset]
        for p in gs.players:
            if p.port == self.self_port:
                return p
        return None

    def should_start(self, state_history) -> bool:
        if len(state_history) < DAMAGE_DELTA_LOOKBACK + 1:
            return False
        curr = self._self_ps(state_history, -1)
        if curr is None:
            return False
        if curr.action not in SHIELD_ACTIONABLE:
            return False
        if curr.shield_strength >= SMALL_SHIELD_THRESHOLD:
            return False
        # Check for a big shield-damage drop in the recent past.
        max_delta = 0.0
        for k in range(1, DAMAGE_DELTA_LOOKBACK + 1):
            prev = self._self_ps(state_history, -1 - k)
            if prev is None:
                break
            delta = float(prev.shield_strength) - float(curr.shield_strength)
            if delta > max_delta:
                max_delta = delta
        return max_delta >= DAMAGE_DELTA_THRESHOLD

    def should_end(self, state_history, episode_start_idx: int) -> bool:
        curr = self._self_ps(state_history, -1)
        if curr is None:
            return True
        if curr.action in SHIELD_FAMILY:
            # Keep running while in shield family (stun counted as still
            # inside the episode).
            return False
        return True

    def compute_outcome(self, state_history, episode_start_idx: int) -> EpisodeOutcome:
        curr = self._self_ps(state_history, -1)
        if curr is None:
            return EpisodeOutcome(terminal_reward=0.0,
                                  metadata={"result": "aborted"})
        s = int(curr.action)
        if s in SHIELD_BREAK_STATES:
            return EpisodeOutcome(terminal_reward=-1.0,
                                  metadata={"result": "shield_break"})
        if s in ESCAPE_STATES:
            return EpisodeOutcome(terminal_reward=1.0,
                                  metadata={"result": "escape",
                                            "escape_state": s})
        if s in DAMAGE_STATES:
            return EpisodeOutcome(terminal_reward=0.0,
                                  metadata={"result": "hit_out",
                                            "final_state": s})
        # Other exits (shield release, jump-out, grab, etc.) count as
        # escape — any intentional transition out of shield.
        return EpisodeOutcome(terminal_reward=1.0,
                              metadata={"result": "exit",
                                        "final_state": s})
