"""Streaming slippistats — a faithful streaming port of slippistats' batch
stat logic for live `libmelee`-style player state.

slippistats (`~/.local/lib/python3.12/site-packages/slippistats/`) computes
its stats in batch over a parsed `.slp`. The online RL actor needs the same
predicates and state machines *streaming*, frame-by-frame, on live state.
This module is that port. It is the shared dependency of every VR module
(`rlvr/online/vr/`) and supersedes the ad-hoc port that lived inside the
retired `combo_extend_online.py`.

Porting discipline (see CLAUDE.md "Porting slippistats logic"): the
predicates and trackers below were ported branch-by-branch from
`slippistats/stats/common.py`, `combo_computer.py`, and `tech_compute`,
adapted to the fields a live player-state object exposes (action enum,
percent, stock, hitstun_frames_left, hitlag_left, off_stage, on_ground,
action_frame) rather than slippistats' parsed-frame bitflags.

Player/GameState objects here follow the `rlvr` shape: `gs.players` is an
iterable of player objects each with `.port`, `.action`, `.percent`,
`.stock`, `.hitstun_frames_left`, `.action_frame`, `.position`, etc.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# --------------------------------------------------------------------------
# Action-state ranges (libmelee Action enum integers).
# Ported from slippistats/enums/state.py:ActionRange, cross-checked against
# the 6-rounds-debugged ranges in the retired combo_extend_online.py.
# --------------------------------------------------------------------------
# DYING_* (0-10): blast-zone death animations. Kept in the punish set so a
# kill blow stays "in punish" long enough for the stock decrement to land.
DYING_START, DYING_END = 0, 10
# FALL_SPECIAL_* (35-37): post-up-B helpless fall.
FALL_SPECIAL_START, FALL_SPECIAL_END = 35, 37
# DAMAGE_HIGH_1 (75) .. DAMAGE_FLY_ROLL (91): generic hit reactions.
DAMAGE_START, DAMAGE_END = 75, 91
# GUARD_ON (178) .. GUARD_REFLECT (182): shielding.
GUARD_START, GUARD_END = 178, 182
# DOWN_BOUND_* (183) .. DOWN_SPOT_D (198): knocked-on-ground (missed tech).
DOWN_START, DOWN_END = 183, 198
# TECH_* (199-204): tech in place / roll / jump.
TECH_START, TECH_END = 199, 204
# GUARD_BREAK_* (205-211): shield broke -> stunned.
GUARD_BREAK_START, GUARD_BREAK_END = 205, 211
# CAPTURE_PULLED_HIGH (223) .. CAPTURE_FOOT (232): in opponent's grab.
CAPTURE_START, CAPTURE_END = 223, 232
# DODGE_* (233-236): spot dodge / roll / airdodge.
DODGE_START, DODGE_END = 233, 236
# THROWN_FORWARD (239) .. THROWN_DOWN_2 (243): the throw animations.
THROWN_START, THROWN_END = 239, 243
# LEDGE_ACTION_* (252-263): hanging from / acting off the ledge.
LEDGE_ACTION_START, LEDGE_ACTION_END = 252, 263
# Two character-specific command-grab ranges.
COMMAND_GRAB_RANGE1_START, COMMAND_GRAB_RANGE1_END = 266, 304
COMMAND_GRAB_RANGE2_START, COMMAND_GRAB_RANGE2_END = 327, 338

# Mirrors slippistats COMBO_LENIENCY — frames out of punish before a combo
# is considered ended.
COMBO_LENIENCY = 45


# --------------------------------------------------------------------------
# State accessors
# --------------------------------------------------------------------------
def _act(ps) -> int:
    """Action-state as an int (`ps.action` may be an enum or an int)."""
    a = ps.action
    return int(getattr(a, "value", a))


def get_player(gs, port: int):
    """Return the player object on `port`, or None."""
    for p in gs.players:
        if int(p.port) == int(port):
            return p
    return None


def get_opponent(gs, self_port: int):
    """Return the first player whose port != self_port, or None."""
    for p in gs.players:
        if int(p.port) != int(self_port):
            return p
    return None


# --------------------------------------------------------------------------
# Stateless predicates  (ports of slippistats/stats/common.py)
# --------------------------------------------------------------------------
def is_damaged(ps) -> bool:
    """In a generic hit-reaction state (slippistats `is_damaged`, DAMAGE
    range). The DOWN_DAMAGE jab-reset states slippistats also folds in are
    covered by the DOWN range inside `in_punish_state`."""
    if ps is None:
        return False
    return DAMAGE_START <= _act(ps) <= DAMAGE_END


def is_teching(ps) -> bool:
    """Tech / knockdown situation — slippistats `is_teching`: TECH range
    (199-204) plus the DOWN range (183-198, missed-tech/downed). Wall and
    ceiling techs (FLY_REFLECT_*) are intentionally omitted — rare, and not
    relevant to floor-tech punishes."""
    if ps is None:
        return False
    a = _act(ps)
    return (TECH_START <= a <= TECH_END) or (DOWN_START <= a <= DOWN_END)


def is_downed(ps) -> bool:
    """Knocked on the ground / missed tech (slippistats `is_downed`)."""
    if ps is None:
        return False
    return DOWN_START <= _act(ps) <= DOWN_END


def is_dying(ps) -> bool:
    """In a blast-zone death animation (slippistats `is_dying`)."""
    if ps is None:
        return False
    return DYING_START <= _act(ps) <= DYING_END


def is_special_fall(ps) -> bool:
    """Post-up-B helpless fall (slippistats `is_special_fall`)."""
    if ps is None:
        return False
    return FALL_SPECIAL_START <= _act(ps) <= FALL_SPECIAL_END


def on_ledge(ps) -> bool:
    """Hanging from / acting off the ledge (slippistats `is_ledge_action`)."""
    if ps is None:
        return False
    return LEDGE_ACTION_START <= _act(ps) <= LEDGE_ACTION_END


def in_hitstun(ps) -> bool:
    """In hitstun. slippistats reads a bitflag; live state exposes
    `hitstun_frames_left` directly."""
    if ps is None:
        return False
    return float(getattr(ps, "hitstun_frames_left", 0.0) or 0.0) > 0.0


def in_hitlag(ps) -> bool:
    """In hitlag (the freeze frames between connect and stun)."""
    if ps is None:
        return False
    return float(getattr(ps, "hitlag_left", 0.0) or 0.0) > 0.0


def is_offstage(ps) -> bool:
    """Off the stage. Uses libmelee's `off_stage` flag rather than
    slippistats' position-vs-stage-bounds computation."""
    if ps is None:
        return False
    return bool(getattr(ps, "off_stage", False))


def in_any_grab(ps) -> bool:
    """In a regular grab/capture or a character-specific command grab."""
    if ps is None:
        return False
    a = _act(ps)
    return (
        (CAPTURE_START <= a <= CAPTURE_END)
        or (COMMAND_GRAB_RANGE1_START <= a <= COMMAND_GRAB_RANGE1_END)
        or (COMMAND_GRAB_RANGE2_START <= a <= COMMAND_GRAB_RANGE2_END)
    )


def in_punish_state(ps) -> bool:
    """Faithful streaming port of slippistats' combo-continuation set
    (`combo_computer.py:260-277`). The player is "still being punished /
    not yet escaped" when any of: damaged (75-91), grabbed/captured
    (223-232), command-grabbed, thrown (239-243), dying (0-10), downed
    (183-198), teching (199-204), dodging (233-236), shielding (178-182),
    shield-broken (205-211), special-fall (35-37), ledge action (252-263),
    in hitstun, in hitlag, or off-stage.

    Intentionally skipped (vs slippistats): `is_maybe_juggled` (needs stage
    geometry) and `is_upb_lag` (needs a prev-state diff; rare effect)."""
    if ps is None:
        return False
    a = _act(ps)
    if DAMAGE_START <= a <= DAMAGE_END:
        return True
    if CAPTURE_START <= a <= CAPTURE_END:
        return True
    if COMMAND_GRAB_RANGE1_START <= a <= COMMAND_GRAB_RANGE1_END:
        return True
    if COMMAND_GRAB_RANGE2_START <= a <= COMMAND_GRAB_RANGE2_END:
        return True
    if THROWN_START <= a <= THROWN_END:
        return True
    if DYING_START <= a <= DYING_END:
        return True
    if DOWN_START <= a <= DOWN_END:
        return True
    if TECH_START <= a <= TECH_END:
        return True
    if DODGE_START <= a <= DODGE_END:
        return True
    if GUARD_START <= a <= GUARD_END:
        return True
    if GUARD_BREAK_START <= a <= GUARD_BREAK_END:
        return True
    if FALL_SPECIAL_START <= a <= FALL_SPECIAL_END:
        return True
    if LEDGE_ACTION_START <= a <= LEDGE_ACTION_END:
        return True
    if in_hitstun(ps):
        return True
    if in_hitlag(ps):
        return True
    if is_offstage(ps):
        return True
    return False


def recently_in_hitstun_or_damage(state_history, port: int, window: int) -> bool:
    """SD-gate heuristic, shared by `stock-delta` and `low-percent-kill`:
    was the player on `port` in hitstun / a DAMAGE state / hitlag at any
    point in the last `window` frames of `state_history`? A real kill is
    preceded by the killing hit (hitstun + damage); a clean self-destruct
    is not."""
    n = len(state_history)
    for k in range(1, min(window, n) + 1):
        ps = get_player(state_history[-k], port)
        if ps is None:
            continue
        if in_hitstun(ps) or in_hitlag(ps) or is_damaged(ps):
            return True
    return False


# --------------------------------------------------------------------------
# Streaming trackers
# --------------------------------------------------------------------------
class OppHitRecencyTracker:
    """Streaming SD-gate: did the opponent take a hit shortly before
    dying? Distinguishes a kill the bot earned from an opponent
    self-destruct (the exogenous noise the design wants filtered out).

    Why stateful, not a backward scan from the death frame: the `stock`
    count decrements a long, *variable* time after the killing hit. On a
    top/star KO the character sits in a DEAD action state (range 0-10)
    for 1.5-3 s before `stock` ticks down — a fixed look-back window from
    the decrement frame lands entirely inside those DEAD frames and never
    reaches the hit. This tracker carries a decay counter that DEAD
    frames pass through *untouched* ('transparent'), so the
    hit -> fly -> dead -> (long wait) -> decrement sequence still gates
    True. The counter only decays on frames where the opponent is alive
    and not in a hit reaction, so a genuine self-destruct (no hit in the
    `hit_memory` alive-frames before death) still gates False."""

    def __init__(self, hit_memory: int = 90):
        self.hit_memory = int(hit_memory)
        self.reset()

    def reset(self) -> None:
        self._decay = 0

    def update(self, opp_ps) -> None:
        """Advance one frame with the opponent's PlayerState (or None)."""
        if opp_ps is None:
            return
        if in_hitstun(opp_ps) or in_hitlag(opp_ps) or is_damaged(opp_ps):
            self._decay = self.hit_memory
        elif is_dying(opp_ps):
            pass                       # DEAD frames are transparent
        elif self._decay > 0:
            self._decay -= 1

    @property
    def recently_hit(self) -> bool:
        return self._decay > 0


class MoveCounter:
    """Streaming port of slippistats' combo move-counter
    (`combo_computer.py:175-221`). A "move" is a group of hits sharing one
    bot action state — a multi-hit drill is one move. Distinct moves are
    counted by watching whether the bot's action state has changed (or
    restarted: `action_frame` went backward) since the last connecting hit."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.n_moves = 0
        self._last_hit_animation: Optional[int] = None
        self._last_action_frame = 0.0
        self._last_opp_percent = 0.0

    def seed(self, self_action: int, self_action_frame: float,
             opp_percent: float, opp_prev_percent: float) -> None:
        """Called on the combo-opening frame. The triggering hit counts as
        move #1 iff the opponent actually took damage this frame."""
        self._last_opp_percent = opp_percent
        if opp_percent > opp_prev_percent:
            self.n_moves = 1
            self._last_hit_animation = self_action
            self._last_action_frame = self_action_frame
        else:
            self.n_moves = 0
            self._last_hit_animation = None
            self._last_action_frame = 0.0

    def update(self, self_action: int, self_action_frame: float,
               opp_percent: float) -> None:
        """Per-frame update while a combo is open."""
        animation_advanced = (
            self._last_hit_animation is not None
            and (self_action != self._last_hit_animation
                 or self_action_frame < self._last_action_frame)
        )
        if animation_advanced:
            self._last_hit_animation = None
        damage = opp_percent - self._last_opp_percent
        if damage > 0.0:
            if self._last_hit_animation is None:
                self.n_moves += 1
            self._last_hit_animation = self_action
            self._last_action_frame = self_action_frame
            self._last_opp_percent = opp_percent


@dataclass
class ComboResult:
    """Emitted by `ComboTracker.update` on the frame a combo closes."""
    n_moves: int
    damage: float          # peak opp percent reached − percent at combo start
    did_kill: bool         # the combo ended because the opponent lost a stock


class ComboTracker:
    """Streaming port of `combo_computer.py`'s punish-sequence detection.
    A combo opens on the ascending edge of the opponent entering punish
    state (gated by a damage-or-grab safeguard), stays open while the
    opponent keeps re-entering punish state within `COMBO_LENIENCY` frames,
    and closes on opponent stock loss, bot stock loss, or the leniency gap.
    Peak opponent percent is tracked so a kill's damage is captured before
    the respawn percent-reset."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._active = False
        self._start_opp_percent = 0.0
        self._start_opp_stock: Optional[int] = None
        self._start_self_stock: Optional[int] = None
        self._peak_opp_percent = 0.0
        self._reset_counter = 0
        self._moves = MoveCounter()
        self._prev_opp_percent: Optional[float] = None

    @property
    def active(self) -> bool:
        return self._active

    def update(self, self_ps, opp_ps) -> Optional[ComboResult]:
        """Feed one frame. Returns a `ComboResult` on the frame a combo
        closes, else None."""
        result: Optional[ComboResult] = None
        if self_ps is None or opp_ps is None:
            return None
        opp_pct = float(opp_ps.percent)
        opp_stk = int(opp_ps.stock)
        self_stk = int(self_ps.stock)
        self_act = _act(self_ps)
        self_af = float(getattr(self_ps, "action_frame", 0.0) or 0.0)

        if not self._active:
            if in_punish_state(opp_ps) and self._prev_opp_percent is not None:
                prev_pct = self._prev_opp_percent
                # Damage-or-grab safeguard: a real combo trigger either
                # dealt damage this frame or put the opp in a grab —
                # otherwise the punish-state edge was library noise.
                if opp_pct > prev_pct or in_any_grab(opp_ps):
                    self._active = True
                    self._start_opp_percent = prev_pct
                    self._start_opp_stock = opp_stk
                    self._start_self_stock = self_stk
                    self._peak_opp_percent = opp_pct
                    self._reset_counter = 0
                    self._moves.reset()
                    self._moves.seed(self_act, self_af, opp_pct, prev_pct)
        else:
            # Track peak percent + move count only while the opp hasn't
            # respawned (percent resets to 0 on death).
            if opp_stk >= (self._start_opp_stock or 0):
                self._peak_opp_percent = max(self._peak_opp_percent, opp_pct)
                self._moves.update(self_act, self_af, opp_pct)
            # Leniency counter.
            if in_punish_state(opp_ps):
                self._reset_counter = 0
            else:
                self._reset_counter += 1
            # Termination.
            terminate = False
            did_kill = False
            if self._start_opp_stock is not None and opp_stk < self._start_opp_stock:
                terminate, did_kill = True, True
            elif (self._start_self_stock is not None
                    and self_stk < self._start_self_stock):
                terminate = True
            elif self._reset_counter > COMBO_LENIENCY:
                terminate = True
            if terminate:
                result = ComboResult(
                    n_moves=self._moves.n_moves,
                    damage=max(0.0, self._peak_opp_percent - self._start_opp_percent),
                    did_kill=did_kill,
                )
                self._active = False

        self._prev_opp_percent = opp_pct
        return result


@dataclass
class TechResult:
    """Emitted by `TechTracker.update` on the frame a tech situation ends."""
    was_punished: bool     # the player exited the tech situation into a hit


class TechTracker:
    """Streaming port of slippistats `tech_compute`. A tech situation is
    open while `is_teching` (TECH ∪ DOWN). slippistats' `was_punished`
    rule: the situation was punished iff the player *exited* it directly
    into a damaged state."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._active = False
        self._prev_teching = False

    def update(self, self_ps) -> Optional[TechResult]:
        """Feed one frame. Returns a `TechResult` when a tech situation
        closes, else None."""
        result: Optional[TechResult] = None
        if self_ps is None:
            return None
        curr_teching = is_teching(self_ps)
        if not curr_teching:
            if self._prev_teching and self._active:
                # Faithful to tech_compute: punished iff the exit state
                # is itself a damaged state.
                result = TechResult(was_punished=is_damaged(self_ps))
                self._active = False
        else:
            if not self._prev_teching:
                self._active = True
        self._prev_teching = curr_teching
        return result


@dataclass
class RecoveryResult:
    """Emitted by `RecoveryTracker.update` when a recovery situation ends."""
    succeeded: bool        # made it back (landed / grabbed ledge) vs died


class RecoveryTracker:
    """Streaming implementation of slippistats' commented-out
    `recovery_compute` sketch. A recovery situation opens when the player
    is knocked off-stage — `off_stage AND in_hitstun` (the sketch's start
    condition). It closes on success (back on stage / grabbed ledge) or
    failure (lost a stock while still recovering). A *voluntary* trip
    off-stage is not in hitstun, so it opens nothing."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._active = False
        self._prev_knocked_off = False
        self._start_stock: Optional[int] = None

    def update(self, self_ps) -> Optional[RecoveryResult]:
        """Feed one frame. Returns a `RecoveryResult` when a recovery
        situation closes, else None."""
        result: Optional[RecoveryResult] = None
        if self_ps is None:
            return None
        offstage = is_offstage(self_ps)
        knocked_off = offstage and in_hitstun(self_ps)
        stock = int(self_ps.stock)

        if not self._active:
            # Open only on the ascending edge of being *knocked* off.
            if knocked_off and not self._prev_knocked_off:
                self._active = True
                self._start_stock = stock
        else:
            # Failure: bot crossed the blast zone (entered DYING action
            # range 0-10) while still recovering. Check this BEFORE the
            # success branch because under the infinite_time gecko the
            # game doesn't decrement stock (stock<start_stock never
            # fires), and the bot respawns on the platform with
            # on_ground=True, which would falsely match the success
            # branch. The dying-action rising edge correctly captures
            # off-stage deaths in either gecko mode.
            if is_dying(self_ps):
                result = RecoveryResult(succeeded=False)
                self._active = False
            elif self._start_stock is not None and stock < self._start_stock:
                result = RecoveryResult(succeeded=False)
                self._active = False
            elif not offstage and (
                    bool(getattr(self_ps, "on_ground", False)) or on_ledge(self_ps)):
                result = RecoveryResult(succeeded=True)
                self._active = False

        self._prev_knocked_off = knocked_off
        return result
