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

from typing import Optional

from rlvr.online.episode import EpisodeOutcome, OnlineTask


TASK_ID = "combo_extend_online"

# Action-state ranges (libmelee Action enum integers). These mirror
# slippistats/stats/common.py is_damaged / is_grabbed / is_dying /
# is_downed / is_teching / is_dodging — see also value/derived_features.py
# for the V(s) discovery side.
# DAMAGE_HIGH_1 (75) through DAMAGE_FLY_ROLL (91): generic hit reactions.
DAMAGE_START = 75
DAMAGE_END = 91
# CAPTURE_PULLED_HIGH (223) through CAPTURE_FOOT (232): in opp's grab.
CAPTURE_START = 223
CAPTURE_END = 232
# DYING_START (0) through DYING_END (10): blast-zone death animations.
# CRITICAL — keeps the combo open through the kill blow so the stock
# decrement registers before the K-gap closes the episode.
DYING_START = 0
DYING_END = 10
# DOWN_BOUND_*  (183) through DOWN_SPOT_D (198): knocked-on-ground
# (missed tech or hit-into-floor). Common in tech-chase combos.
DOWN_START = 183
DOWN_END = 198
# TECH_START (199) through TECH_END (204): tech-in-place / -roll /
# -jump. Bot is still pressuring the opp here.
TECH_START = 199
TECH_END = 204
# DODGE_START (233) through DODGE_END (236): spot dodge / roll /
# airdodge. Opp's escape attempt; counts as "still being pressured"
# because the bot's punish opportunity hasn't ended yet.
DODGE_START = 233
DODGE_END = 236
# THROWN_FORWARD (239) through THROWN_DOWN_2 (243): the throw
# animations opp is forced through after a grab. CAPTURE covers the
# "being held" portion (223-232) but not the throw itself, so without
# this range a grab → throw → followup combo gets segmented during
# the ~10-30 throw frames. libmelee's hitstun_frames_left may also
# fire during throws but isn't reliable across Slippi/Ishiiruka
# versions; the explicit range is safer.
THROWN_START = 239
THROWN_END = 243
# FALL_SPECIAL_START (35) through FALL_SPECIAL_END (37): post-up-B
# helpless-fall. Opp can't act → punish opportunity.
FALL_SPECIAL_START = 35
FALL_SPECIAL_END = 37
# GUARD_BREAK_START (205) through GUARD_BREAK_END (211): opp's
# shield broke → stunned. Definitively a punish state.
GUARD_BREAK_START = 205
GUARD_BREAK_END = 211
# GUARD_START (178) through GUARD_END (182): opp is shielding.
# slippistats keeps the combo alive through mid-combo shield
# pressure. Counts as "still being pressured" — no false-positive
# damage since shield blocks.
GUARD_START = 178
GUARD_END = 182
# LEDGE_ACTION_START (252) through LEDGE_ACTION_END (263): opp is
# on the ledge or doing a ledge option. slippistats counts this as
# combo continuation (you're still edge-guarding).
LEDGE_ACTION_START = 252
LEDGE_ACTION_END = 263
# Two command-grab ranges (266-304 and 327-338): character-specific
# command grabs (Bowser side-B, Yoshi egg-lay, DK cargo, Kirby inhale,
# etc.). Fox doesn't have one, but the opp could be any character.
COMMAND_GRAB_RANGE1_START = 266
COMMAND_GRAB_RANGE1_END = 304
COMMAND_GRAB_RANGE2_START = 327
COMMAND_GRAB_RANGE2_END = 338

# How many consecutive frames opp must be out of punish state before
# we close the combo. Mirrors slippistats COMBO_LENIENCY = 45 frames
# (~3/4 sec). Previously 20, which was too aggressive — split
# tech-chases, re-grab attempts, and DI-wait punishes into multiple
# short sub-threshold episodes.
COMBO_END_GAP = 45

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
        self._episode_start_self_stock: Optional[int] = None
        self._frames_since_start = 0  # incremented per should_end call
        # Track the highest opp percent reached during the open episode.
        # Necessary because opp's percent resets to 0 on respawn — if the
        # combo killed them, end-of-episode opp.percent is 0 and a naive
        # end-minus-start damage calc would report 0 damage for a
        # successful kill. Peak captures the damage that was actually
        # dealt before the death blast.
        self._max_opp_percent = 0.0
        # Count distinct *moves* within the episode (slippistats-style).
        # A "move" is a group of hits that share the same bot action
        # state — Fox's dair drill is ONE move with 6-8 hits; a
        # jab→grab→throw is 3 separate moves. We require n_moves >= 2
        # for a "combo" reward — single-move multi-hit punishes (like a
        # solo dair drill) DON'T count as extension.
        #
        # Tracking follows slippistats's combo_computer.py:181-221:
        #   - On each frame, track whether bot's action state has
        #     "changed since the last hit". If yes OR if action_frame
        #     reset (state_age went down — same action restarted, e.g.
        #     jab1 → jab1 again), clear last_hit_animation.
        #   - When opp percent increases (a hit landed):
        #     * if last_hit_animation is None → new move starting,
        #       n_moves += 1
        #     * last_hit_animation := bot's action at this frame
        self._n_moves: int = 0
        self._last_hit_animation: Optional[int] = None
        self._last_action_frame: float = 0.0
        self._last_opp_percent_for_move_count: float = 0.0

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
        """Faithful port of slippistats's combo-continuation set
        (combo_computer.py:260-277). Opp is "still being punished /
        not yet escaped" when any of:
          - damaged (75-91)               — the canonical hit reaction
          - grabbed / captured (223-232)  — in our grab
          - command-grabbed (266-304, 327-338) — character-specific
            grabs (Bowser side-B, Kirby inhale, etc.)
          - thrown (239-243)              — being thrown out of grab
          - hitstun bitflag (live-stream signal)
          - in hitlag (the freeze frames between hit-connect and the
            hit's stun starting)
          - dying (0-10)                  — death animation
          - downed (183-198)              — knocked on ground
          - teching (199-204)             — tech in place / roll / jump
          - dodging (233-236)             — spot dodge / roll / airdodge
          - shielding (178-182)           — mid-combo shield pressure
          - shield-broken (205-211)       — stunned from shield break
          - special-fall (35-37)          — post-up-B helpless
          - ledge action (252-263)        — at the ledge, edgeguard
          - off-stage flag                — opp is off the stage
        Skipped slippistats checks (intentional):
          - is_maybe_juggled (needs stage geometry, position-based)
          - is_upb_lag (needs prev_state diff; rare effect)
        """
        if ps is None:
            return False
        a = int(ps.action)
        if DAMAGE_START <= a <= DAMAGE_END: return True
        if CAPTURE_START <= a <= CAPTURE_END: return True
        if COMMAND_GRAB_RANGE1_START <= a <= COMMAND_GRAB_RANGE1_END: return True
        if COMMAND_GRAB_RANGE2_START <= a <= COMMAND_GRAB_RANGE2_END: return True
        if THROWN_START <= a <= THROWN_END: return True
        if DYING_START <= a <= DYING_END: return True
        if DOWN_START <= a <= DOWN_END: return True
        if TECH_START <= a <= TECH_END: return True
        if DODGE_START <= a <= DODGE_END: return True
        if GUARD_START <= a <= GUARD_END: return True
        if GUARD_BREAK_START <= a <= GUARD_BREAK_END: return True
        if FALL_SPECIAL_START <= a <= FALL_SPECIAL_END: return True
        if LEDGE_ACTION_START <= a <= LEDGE_ACTION_END: return True
        if float(ps.hitstun_frames_left) > 0.0: return True
        if float(getattr(ps, "hitlag_left", 0.0)) > 0.0: return True
        if bool(getattr(ps, "off_stage", False)): return True
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
            # Damage-or-grab safeguard: a legitimate combo trigger
            # must either (a) deal damage this frame, or (b) put opp
            # into a grab/capture state. Otherwise the transition was
            # something like shield-stun-edge-case / hitstun-bit-glitch
            # / library noise — opp didn't actually get hit. Without
            # this filter the bot could be rewarded for spurious
            # state-machine triggers that didn't correspond to real
            # hits.
            pct_increased = float(curr.percent) > float(prev.percent)
            a = int(curr.action)
            in_any_grab = (
                CAPTURE_START <= a <= CAPTURE_END
                or COMMAND_GRAB_RANGE1_START <= a <= COMMAND_GRAB_RANGE1_END
                or COMMAND_GRAB_RANGE2_START <= a <= COMMAND_GRAB_RANGE2_END
            )
            if not (pct_increased or in_any_grab):
                firing = False
        if firing:
            # IMPORTANT: capture *prev*'s percent (before the punish-
            # triggering hit applied), not curr's. The ascending-edge
            # frame is the frame the engine wrote both DAMAGE-action
            # and the post-hit percent — so curr.percent already
            # includes the initiating hit's damage. Snapshotting curr
            # would credit only follow-up hits and read a clean single-
            # hit punish as 0% damage. With prev.percent the initial
            # hit's damage shows up in the total too.
            self._episode_start_opp_state = (float(prev.percent), int(prev.stock))
            # Snapshot bot stock at episode open. Slippistats terminates
            # the combo if the *player* (us) dies mid-window:
            # combo_computer.py:295 `... or player_did_lose_stock`.
            self_prev = self._self_ps(state_history, -2)
            self._episode_start_self_stock = (int(self_prev.stock)
                                               if self_prev is not None
                                               else None)
            self._frames_since_start = 0
            self._max_opp_percent = float(curr.percent)
            # Move counting: the triggering hit counts as move #1 ONLY
            # if opp actually took damage (real hit). Pure grabs start
            # at 0 moves; the throw/follow-up will register as move #1.
            self._last_opp_percent_for_move_count = float(curr.percent)
            if float(curr.percent) > float(prev.percent):
                self._n_moves = 1
                # Snapshot bot's action at the moment the hit landed.
                # We look at SELF, since slippistats keys move identity
                # on the bot's animation state.
                self_curr = self._self_ps(state_history, -1)
                if self_curr is not None:
                    self._last_hit_animation = int(self_curr.action)
                    self._last_action_frame = float(self_curr.action_frame)
                else:
                    self._last_hit_animation = None
                    self._last_action_frame = 0.0
            else:
                self._n_moves = 0
                self._last_hit_animation = None
                self._last_action_frame = 0.0
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

        # Track peak opp percent for kill-damage capture. After
        # respawn opp.percent resets to 0, so we can't read end-of-
        # episode percent for damage; we have to remember the max
        # achieved during the combo. This is the slippistats approach
        # (`combo.current_percent = opponent_frame.post.percent` per
        # frame, only when `did_lose_stock` is False).
        start_stock = (self._episode_start_opp_state[1]
                       if self._episode_start_opp_state else 4)
        if int(curr.stock) >= start_stock:
            self._max_opp_percent = max(self._max_opp_percent,
                                        float(curr.percent))
            # Move counting — slippistats-style. Track bot's animation
            # state and decide whether a new percent-increase counts as
            # a new move (different animation OR animation reset) or
            # the same move (multi-hit drill, e.g. dair).
            self_curr = self._self_ps(state_history, -1)
            if self_curr is not None:
                self_action = int(self_curr.action)
                self_action_frame = float(self_curr.action_frame)
                # Detect "animation changed" or "animation reset":
                animation_advanced = (
                    self._last_hit_animation is not None and (
                        self_action != self._last_hit_animation
                        or self_action_frame < self._last_action_frame
                    )
                )
                if animation_advanced:
                    # Bot moved on to a different move — next damage
                    # tick will register as a new move.
                    self._last_hit_animation = None
                # Damage delta = a hit landed this frame.
                damage_taken_this_frame = (
                    float(curr.percent) - self._last_opp_percent_for_move_count
                )
                if damage_taken_this_frame > 0.0:
                    if self._last_hit_animation is None:
                        # New move detected.
                        self._n_moves += 1
                    self._last_hit_animation = self_action
                    self._last_action_frame = self_action_frame
                    self._last_opp_percent_for_move_count = float(curr.percent)

        # Stock-loss terminates immediately (kill confirm).
        if self._episode_start_opp_state is not None:
            _, start_stock = self._episode_start_opp_state
            if int(curr.stock) < start_stock:
                return True

        # Player-death termination (slippistats player_did_lose_stock).
        # If the bot dies mid-combo (opp counter-KO'd us), the combo
        # is over — we're no longer the one applying pressure.
        if self._episode_start_self_stock is not None:
            self_curr = self._self_ps(state_history, -1)
            if (self_curr is not None
                    and int(self_curr.stock) < self._episode_start_self_stock):
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
        """Score the combo: damage dealt to opp during episode. Uses
        peak opp percent (not end-of-episode) so kills are scored from
        the damage delivered before respawn-reset. Stock takes are
        reported in metadata for analytics but do NOT receive a flat
        bonus reward — that contaminates the combo-extension signal
        (e.g. opp self-destructs late in an unrelated combo would get
        +1.0 free)."""
        end_opp = self._opp_ps(state_history, -1)
        start_state = self._lookup_start_state(state_history, episode_start_idx)
        # Fall back to end_opp.percent when peak wasn't tracked (e.g.
        # tests that bypass should_start/should_end). In real combat
        # _max_opp_percent is always populated.
        effective_max = self._max_opp_percent
        if end_opp is not None:
            effective_max = max(effective_max, float(end_opp.percent))
        # Fall back move-count: if state machine wasn't driven (test
        # path), infer at least 1 move if end_pct > start_pct.
        effective_moves = self._n_moves
        if effective_moves == 0 and start_state is not None and end_opp is not None:
            if float(end_opp.percent) > start_state[0]:
                effective_moves = 1
        outcome = self._score(start_state, end_opp,
                              effective_max, effective_moves)
        # Reset per-episode state so a stale snapshot doesn't leak
        # into the next episode (e.g., after abortive close).
        self._episode_start_opp_state = None
        self._episode_start_self_stock = None
        self._frames_since_start = 0
        self._max_opp_percent = 0.0
        self._n_moves = 0
        self._last_hit_animation = None
        self._last_action_frame = 0.0
        self._last_opp_percent_for_move_count = 0.0
        return outcome

    def _score(self, start_state, end_opp,
               max_opp_percent: float, n_moves: int) -> EpisodeOutcome:
        if start_state is None or end_opp is None:
            return EpisodeOutcome(terminal_reward=0.0,
                                  metadata={"result": "aborted"})
        start_percent, start_stock = start_state

        # Use peak percent — captures kill damage even after respawn.
        damage = max(0.0, float(max_opp_percent) - start_percent)
        stock_was_taken = int(end_opp.stock) < start_stock

        # n_moves < 2 → single move, not a combo. The task is combo
        # extension; a single move (even a 30% upsmash, or a 6-hit
        # dair drill the bot just held the button for) doesn't extend
        # anything. Classify + emit 0 reward.
        if n_moves < 2:
            return EpisodeOutcome(
                terminal_reward=0.0,
                metadata={"result": "single_hit",
                          "damage": damage,
                          "n_moves": n_moves,
                          "start_percent": start_percent,
                          "end_percent": float(end_opp.percent)},
            )

        if damage < MIN_DAMAGE_REWARD:
            return EpisodeOutcome(
                terminal_reward=0.0,
                metadata={"result": "sub_threshold",
                          "damage": damage,
                          "n_moves": n_moves,
                          "start_percent": start_percent,
                          "end_percent": float(end_opp.percent)},
            )

        clipped = min(damage, MAX_DAMAGE_REWARD)
        # Stock takes still tracked in metadata for analytics, but are
        # NOT rewarded with a flat +1.0 bonus. The combo's *damage*
        # determines the reward: a real combo kill at 0% reaches a
        # peak around 120-130% naturally (clipped to 80) → reward ≈ 1.0
        # anyway. Pure SDs (opp dies without us hitting them) get 0.
        return EpisodeOutcome(
            terminal_reward=clipped / MAX_DAMAGE_REWARD,
            metadata={"result": "combo_kill" if stock_was_taken else "combo",
                      "damage": damage,
                      "n_moves": n_moves,
                      "start_percent": start_percent,
                      "max_percent": float(max_opp_percent),
                      "stock_was_taken": stock_was_taken},
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
