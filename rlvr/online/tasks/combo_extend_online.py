"""Online combo-extension task.

The first verifiable reward operationalized from the V(s) discovery
work (see `docs/research-notes-2026-05-13.md`). Discovery surfaced
`combo_on_opp_damage` as one of the strongest cross-matchup signals —
high-V states within macro-matched bins are systematically the ones
where the bot is dealing damage during an opp punish sequence.

Episode start: opp transitions FROM not-in-punish-state INTO
in-punish-state. Punish state = opp.action ∈ DAMAGE range (75-91) OR
opp.hitstun_frames_left > 0 OR opp.action ∈ GRABBED capture range
(223-232). This is the same composite slippistats uses for combo
detection (see `value/derived_features.py`).

Episode end: opp has been continuously OUT of punish state for K=20
consecutive frames. The window tolerates juggle gaps (transient 1-19
frame non-hitstun moments mid-combo) but treats a sustained recovery
as the combo's end.

Terminal reward: cumulative damage dealt to opp during the episode,
clipped to [0, 80] and rescaled to [0, 1]. We use `opp.percent_end -
opp.percent_start` directly. If opp loses a stock mid-combo their
percent resets to 0, which would mis-score a kill confirm as low-
reward — the stock-loss handler treats that as a successful KO and
emits a +1.0 terminal reward. Sub-threshold combos (< 5%
total damage and no kill) emit 0 to avoid rewarding tap-hits that
were never going to extend.

No `enrich_with_replay` needed — all signal is in libmelee
`PlayerState` fields we already have.
"""
from __future__ import annotations

from rlvr.online.episode import EpisodeOutcome, OnlineTask


TASK_ID = "combo_extend_online"

# Action-state ranges (libmelee Action enum integers). These mirror
# slippistats's `is_damaged` / `is_grabbed` ranges (see
# slippistats/enums/state.py:191-194 and value/derived_features.py).
# DAMAGE_HIGH_1 (75) through DAMAGE_FLY_ROLL (91) covers all generic
# damage-take states.
DAMAGE_START = 75
DAMAGE_END = 91
# CAPTURE_PULLED_HIGH (223) through CAPTURE_FOOT (232) covers being
# held by an opp's grab.
CAPTURE_START = 223
CAPTURE_END = 232

# How many consecutive frames opp must be out of punish state before
# we close the combo. Tolerates juggle gaps at K-1 frames. 20 frames =
# ~1/3 second.
COMBO_END_GAP = 20

# Don't reward tap-hits that didn't develop. Below this raw-percent
# delta the combo gets 0 reward (unless a stock was taken).
MIN_DAMAGE_REWARD = 5.0

# Bound the reward so a 70% combo doesn't dwarf a 30% combo by 2x
# in the gradient signal — diminishing returns past the cap.
MAX_DAMAGE_REWARD = 80.0

# Reward for a stock taken mid-combo (kill confirm).
STOCK_TAKEN_REWARD = 1.0


class ComboExtendOnlineTask:
    id = TASK_ID
    description = (
        "Online combo-extend: episode starts when opp enters punish "
        f"state (action {DAMAGE_START}-{DAMAGE_END} OR grabbed "
        f"{CAPTURE_START}-{CAPTURE_END} OR hitstun > 0). Ends when "
        f"opp out of punish state for {COMBO_END_GAP} consecutive "
        f"frames OR opp dies. Reward = clip(opp damage dealt during "
        f"episode, 0, {MAX_DAMAGE_REWARD}) / {MAX_DAMAGE_REWARD}, or "
        f"+{STOCK_TAKEN_REWARD} on stock taken. Sub-threshold "
        f"(< {MIN_DAMAGE_REWARD}%) gets 0."
    )

    def __init__(self, self_port: int = 1):
        self.self_port = self_port
        # Per-episode state set in should_start, used by should_end +
        # compute_outcome. The `episode_start_idx` the actor passes
        # becomes unreliable once it slides off the deque (deque
        # maxlen=256 < max episode 600), so we snapshot the open-frame
        # opp state here instead of looking it up via the index.
        self._episode_start_opp_state = None  # (start_percent, start_stock) or None
        self._frames_since_start = 0  # incremented per should_end call

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------
    def _opp_ps(self, state_history, offset: int = -1):
        """Return opp PlayerState at history[offset], or None."""
        gs = state_history[offset]
        for p in gs.players:
            if p.port != self.self_port:
                return p
        return None

    def _self_ps(self, state_history, offset: int = -1):
        """Return self PlayerState at history[offset], or None."""
        gs = state_history[offset]
        for p in gs.players:
            if p.port == self.self_port:
                return p
        return None

    @staticmethod
    def _in_punish_state(ps) -> bool:
        """slippistats-style 'is being punished' composite."""
        if ps is None:
            return False
        a = int(ps.action)
        if DAMAGE_START <= a <= DAMAGE_END:
            return True
        if CAPTURE_START <= a <= CAPTURE_END:
            return True
        if float(ps.hitstun_frames_left) > 0.0:
            return True
        return False

    # ------------------------------------------------------------------
    # OnlineTask protocol
    # ------------------------------------------------------------------
    def should_start(self, state_history) -> bool:
        """Open an episode on the ascending edge of opp entering
        punish state. Requires at least 2 frames of history to detect
        the transition. Snapshots opp start-state to instance vars."""
        if len(state_history) < 2:
            return False
        curr = self._opp_ps(state_history, -1)
        prev = self._opp_ps(state_history, -2)
        if curr is None or prev is None:
            return False
        # Ascending edge: prev not in punish, curr in punish.
        firing = self._in_punish_state(curr) and not self._in_punish_state(prev)
        if firing:
            self._episode_start_opp_state = (float(curr.percent), int(curr.stock))
            self._frames_since_start = 0
        return firing

    def should_end(self, state_history, episode_start_idx: int) -> bool:
        """Close an episode when opp has been out of punish state for
        COMBO_END_GAP consecutive frames, OR opp lost a stock during
        the episode.

        Uses an instance frame counter (incremented here) rather than
        episode_start_idx so we're robust to the deque sliding off the
        original open frame for long episodes.
        """
        self._frames_since_start += 1

        curr = self._opp_ps(state_history, -1)
        if curr is None:
            return True

        # Stock-loss terminates immediately (kill confirm).
        if self._episode_start_opp_state is not None:
            _, start_stock = self._episode_start_opp_state
            if int(curr.stock) < start_stock:
                return True

        # Need COMBO_END_GAP frames elapsed in the episode before we'd
        # consider closing on the K-frame gap criterion.
        if self._frames_since_start < COMBO_END_GAP:
            return False

        # Look at the last COMBO_END_GAP frames. If opp was in punish
        # state on any of them, the combo is still alive.
        history_to_check = min(COMBO_END_GAP, len(state_history))
        for k in range(history_to_check):
            offset = -1 - k
            ps = self._opp_ps(state_history, offset)
            if self._in_punish_state(ps):
                return False
        return True

    def compute_outcome(self, state_history, episode_start_idx: int) -> EpisodeOutcome:
        """Score the combo: damage dealt to opp during episode, with
        a special-case reward if opp lost a stock."""
        end_opp = self._opp_ps(state_history, -1)
        start_state = self._lookup_start_state(state_history, episode_start_idx)
        outcome = self._score(start_state, end_opp)
        # Reset per-episode state so a stale snapshot doesn't leak
        # into the next episode (e.g., after abortive close).
        self._episode_start_opp_state = None
        self._frames_since_start = 0
        return outcome

    def _score(self, start_state, end_opp) -> EpisodeOutcome:
        if start_state is None or end_opp is None:
            return EpisodeOutcome(terminal_reward=0.0,
                                  metadata={"result": "aborted"})
        start_percent, start_stock = start_state

        # Stock taken mid-combo = kill confirm. Top reward.
        if int(end_opp.stock) < start_stock:
            return EpisodeOutcome(
                terminal_reward=STOCK_TAKEN_REWARD,
                metadata={"result": "stock_taken",
                          "start_percent": start_percent,
                          "stocks_taken": start_stock - int(end_opp.stock)},
            )

        damage = max(0.0, float(end_opp.percent) - start_percent)

        if damage < MIN_DAMAGE_REWARD:
            return EpisodeOutcome(
                terminal_reward=0.0,
                metadata={"result": "sub_threshold",
                          "damage": damage,
                          "start_percent": start_percent,
                          "end_percent": float(end_opp.percent)},
            )

        clipped = min(damage, MAX_DAMAGE_REWARD)
        return EpisodeOutcome(
            terminal_reward=clipped / MAX_DAMAGE_REWARD,
            metadata={"result": "combo",
                      "damage": damage,
                      "start_percent": start_percent,
                      "end_percent": float(end_opp.percent)},
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _lookup_start_state(self, state_history, episode_start_idx: int):
        """Return (start_opp_percent, start_opp_stock) at episode open.

        Two sources, in priority order:
        1. Instance-state snapshot from `should_start` — authoritative
           when the actor drives the full lifecycle.
        2. `state_history[episode_start_idx]` fallback — used by tests
           that directly call compute_outcome without driving
           should_start (the deque-indexing assumption holds for short
           synthetic histories).
        """
        if self._episode_start_opp_state is not None:
            return self._episode_start_opp_state
        n_history = len(state_history)
        if 0 <= episode_start_idx < n_history:
            gs = state_history[episode_start_idx]
            for p in gs.players:
                if p.port != self.self_port:
                    return (float(p.percent), int(p.stock))
        return None
