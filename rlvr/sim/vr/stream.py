"""Streaming slippistats for the sim's libmelee GameStates.

slippistats (~/.local/lib/python3.12/site-packages/slippistats/) computes its stats in batch over a
parsed .slp; the VRs need the same predicates and state machines frame by frame. This is a
branch-by-branch port of `stats/common.py`, `stats/combo_computer.py:combo_compute` and
`stats/stats_computer.py:tech_compute` (plus the commented-out `recovery_compute` sketch), checked
constant by constant against `enums/state.py` (ActionRange / ActionState — slippistats' ids are
Melee's own; libmelee's names differ for a few, e.g. it calls 35 DEAD_FALL where Melee/slippistats
have FallSpecial).

Ported from the Dolphin-era rlvr/online/slippi_stream.py (git 2a7b786^) with these fixes:
  - hitstun / hitlag come from the Slippi state bit flags (flags.py), as in slippistats, not from
    libmelee's `hitstun_frames_left` (the post-frame misc float, nonzero in non-hitstun states).
  - `is_damaged` includes the jab-reset states DOWN_DAMAGE_U/D (185, 193), as in slippistats.
  - `is_teching` includes the wall/ceiling bounces FLY_REFLECT_WALL/CEIL (247, 248).
  - off-stage is slippistats' position test (y < -5 or x outside per-stage bounds), not libmelee's
    `off_stage` flag; the combo continuation set also has slippistats' `is_maybe_juggled`,
    `is_upb_lag` and the wavedash exclusion on dodging, which the old port skipped.
  - the combo opens and counts moves exactly as combo_compute does (start set = damaged / grabbed
    / command-grabbed / hitstun; a move is counted on any percent change while the opponent is in
    that set, keyed on the player's *previous-frame* state), instead of the old ascending-edge +
    damage-or-grab safeguard.
  - Dolphin artifacts removed: infinite-time-gecko handling, persistence across matches.
One deliberate deviation, documented at ComboTracker: combo damage is peak opponent percent minus
start percent (slippistats' end_percent reads the post-death 0% on kill combos).

GameStates are libmelee's: `gs.players` is a dict {port: PlayerState}, `gs.stage` a melee.Stage.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# --- ActionRange / ActionState (slippistats enums/state.py) --------------------------------------
DAMAGE_START, DAMAGE_END = 75, 91
DOWN_DAMAGE_U, DOWN_DAMAGE_D = 185, 193
CAPTURE_START, CAPTURE_END = 223, 232
GUARD_START, GUARD_END = 178, 182
GUARD_BREAK_START, GUARD_BREAK_END = 205, 211
LEDGE_ACTION_START, LEDGE_ACTION_END = 252, 263
DOWN_START, DOWN_END = 183, 198
TECH_START, TECH_END = 199, 204
FLY_REFLECT_WALL, FLY_REFLECT_CEIL = 247, 248
DODGE_START, DODGE_END = 233, 236
DYING_START, DYING_END = 0, 10
FALL_SPECIAL_START, FALL_SPECIAL_END = 35, 37
THROWN_START, THROWN_END = 239, 243
COMMAND_GRAB_RANGE1_START, COMMAND_GRAB_RANGE1_END = 266, 304
COMMAND_GRAB_RANGE2_START, COMMAND_GRAB_RANGE2_END = 327, 338
BARREL_WAIT = 293
CONTROLLED_JUMP_START, CONTROLLED_JUMP_END = 24, 34
KNEE_BEND = 24
ESCAPE_AIR = 236
LAND_FALL_SPECIAL = 43

COMBO_LENIENCY = 45          # slippistats combo_computer.py:32

# is_offstage x bounds and is_maybe_juggled heights, keyed by libmelee Stage name (common.py)
OFFSTAGE_X = {"FOUNTAIN_OF_DREAMS": 64, "YOSHIS_STORY": 56, "DREAMLAND": 73,
              "POKEMON_STADIUM": 88, "BATTLEFIELD": 67, "FINAL_DESTINATION": 89}
JUGGLE_Y = {"FOUNTAIN_OF_DREAMS": 42, "YOSHIS_STORY": 42, "DREAMLAND": 51,
            "POKEMON_STADIUM": 42, "BATTLEFIELD": 54, "FINAL_DESTINATION": 35}


# --- accessors ------------------------------------------------------------------------------------
def act(ps) -> int:
    a = ps.action
    return int(getattr(a, "value", a))


def stage_name(gs) -> str:
    s = getattr(gs, "stage", None)
    return getattr(s, "name", str(s))


def get_player(gs, port: int):
    return gs.players.get(int(port)) if gs is not None else None


def get_opponent(gs, self_port: int):
    """The other player (lowest other port; 1v1)."""
    if gs is None:
        return None
    for p in sorted(gs.players):
        if int(p) != int(self_port):
            return gs.players[p]
    return None


def opponent_port(gs, self_port: int) -> Optional[int]:
    for p in sorted(gs.players):
        if int(p) != int(self_port):
            return int(p)
    return None


# --- stateless predicates (common.py) -------------------------------------------------------------
def is_damaged(a: int) -> bool:
    return DAMAGE_START <= a <= DAMAGE_END or a == DOWN_DAMAGE_U or a == DOWN_DAMAGE_D


def is_grabbed(a: int) -> bool:
    return CAPTURE_START <= a <= CAPTURE_END


def is_cmd_grabbed(a: int) -> bool:
    return ((COMMAND_GRAB_RANGE1_START <= a <= COMMAND_GRAB_RANGE1_END)
            or (COMMAND_GRAB_RANGE2_START <= a <= COMMAND_GRAB_RANGE2_END)) and a != BARREL_WAIT


def in_any_grab(a: int) -> bool:
    return is_grabbed(a) or is_cmd_grabbed(a)


def is_teching(a: int) -> bool:
    return (TECH_START <= a <= TECH_END or DOWN_START <= a <= DOWN_END
            or a == FLY_REFLECT_CEIL or a == FLY_REFLECT_WALL)


def is_dying(a: int) -> bool:
    return DYING_START <= a <= DYING_END


def is_downed(a: int) -> bool:
    return DOWN_START <= a <= DOWN_END


def is_shielding(a: int) -> bool:
    return GUARD_START <= a <= GUARD_END


def is_shield_broken(a: int) -> bool:
    return GUARD_BREAK_START <= a <= GUARD_BREAK_END


def is_dodging(a: int) -> bool:
    return DODGE_START <= a <= DODGE_END


def is_ledge_action(a: int) -> bool:
    return LEDGE_ACTION_START <= a <= LEDGE_ACTION_END


def is_special_fall(a: int) -> bool:
    return FALL_SPECIAL_START <= a <= FALL_SPECIAL_END


def is_thrown(a: int) -> bool:
    return THROWN_START <= a <= THROWN_END


def is_upb_lag(a: int, prev_a: int) -> bool:
    return (a == LAND_FALL_SPECIAL and prev_a != LAND_FALL_SPECIAL and prev_a != KNEE_BEND
            and prev_a != ESCAPE_AIR
            and (prev_a <= CONTROLLED_JUMP_START or prev_a >= CONTROLLED_JUMP_END))


def in_hitstun(ps) -> bool:
    """Slippi hitstun bit flag (flags.py); libmelee's misc-float fallback on flagless replays."""
    f = getattr(ps, "hitstun_flag", None)
    if f is not None:
        return bool(f)
    return float(getattr(ps, "hitstun_frames_left", 0) or 0) > 0


def in_hitlag(ps) -> bool:
    f = getattr(ps, "hitlag_flag", None)
    if f is not None:
        return bool(f)
    return float(getattr(ps, "hitlag_left", 0) or 0) > 0


def is_offstage(ps, stage: str) -> bool:
    """slippistats is_offstage: y < -5, or x outside the stage's on-stage bounds (an unlisted stage
    has bounds (0, 0), so anything off x = 0 counts — as in slippistats)."""
    if float(ps.position.y) < -5:
        return True
    b = OFFSTAGE_X.get(stage, 0)
    return float(ps.position.x) < -b or float(ps.position.x) > b


def is_maybe_juggled(ps, stage: str) -> bool:
    if bool(ps.on_ground):           # slippistats: not airborne -> False
        return False
    h = JUGGLE_Y.get(stage)
    if h is None:
        return False
    return float(ps.position.y) >= h


def is_wavedashing(state_history, port: int) -> bool:
    """slippistats is_wavedashing, verbatim: ESCAPE_AIR now and LAND_FALL_SPECIAL in any of the
    3 frames before."""
    ps = get_player(state_history[-1], port)
    if ps is None or act(ps) != ESCAPE_AIR:
        return False
    for k in range(2, 5):
        if len(state_history) >= k:
            q = get_player(state_history[-k], port)
            if q is not None and act(q) == LAND_FALL_SPECIAL:
                return True
    return False


def did_lose_stock(ps, prev_ps) -> bool:
    return prev_ps is not None and int(prev_ps.stock) - int(ps.stock) > 0


def continues_combo(state_history, port: int) -> bool:
    """The combo_compute continuation set (combo_computer.py:255-275) for the player on `port`."""
    gs = state_history[-1]
    ps = get_player(gs, port)
    if ps is None:
        return False
    prev = get_player(state_history[-2], port) if len(state_history) >= 2 else None
    a = act(ps)
    st = stage_name(gs)
    return (is_damaged(a) or is_grabbed(a) or is_cmd_grabbed(a) or in_hitlag(ps) or in_hitstun(ps)
            or is_shielding(a) or is_offstage(ps, st)
            or (is_dodging(a) and not is_wavedashing(state_history, port))
            or is_dying(a) or is_downed(a) or is_teching(a) or is_ledge_action(a)
            or is_shield_broken(a) or is_maybe_juggled(ps, st) or is_special_fall(a)
            or (prev is not None and is_upb_lag(a, act(prev))))


def in_punish_state(state_history, port: int) -> bool:
    """'Being punished / not yet escaped' for neutral-win-loss: the combo continuation set plus the
    thrown animations (239-243), which combo_compute covers through hitstun."""
    ps = get_player(state_history[-1], port)
    return ps is not None and (continues_combo(state_history, port) or is_thrown(act(ps)))


# --- streaming trackers ---------------------------------------------------------------------------
class OppHitRecencyTracker:
    """Did the player take a hit shortly before dying? Separates a kill from a self-destruct. A
    decay counter, refreshed on hitstun / hitlag / damaged frames; DEAD frames are transparent, so
    a hit -> fly -> dead -> (stock decrement ~1.5 s later) sequence still gates True."""

    def __init__(self, hit_memory: int = 90):
        self.hit_memory = int(hit_memory)
        self.reset()

    def reset(self) -> None:
        self._decay = 0

    def update(self, ps) -> None:
        if ps is None:
            return
        a = act(ps)
        if in_hitstun(ps) or in_hitlag(ps) or is_damaged(a):
            self._decay = self.hit_memory
        elif is_dying(a):
            pass
        elif self._decay > 0:
            self._decay -= 1

    @property
    def recently_hit(self) -> bool:
        return self._decay > 0


@dataclass
class ComboResult:
    n_moves: int
    damage: float           # peak opponent percent (before any stock loss) - start percent
    did_kill: bool
    start_percent: float
    end_percent: float      # slippistats' end_percent (opponent percent on the frame before close)


class ComboTracker:
    """Streaming combo_compute for the player on `port` against the other player.

    Opens when the opponent is damaged / grabbed / command-grabbed / in hitstun; a move is counted
    on any percent change while the opponent is in that set, if the player's action changed (or
    restarted: action_frame went down) since the last hit — the last hit's animation is the
    player's previous-frame state, as in slippistats. Closes when the opponent loses a stock
    (did_kill), the player loses a stock, or the opponent is out of the continuation set for more
    than COMBO_LENIENCY frames. update() returns a ComboResult on the closing frame.

    Deviation: damage is the peak opponent percent reached before a stock loss minus the start
    percent; slippistats' total_damage() (end_percent - start_percent) reads the respawn 0% on kill
    combos. ComboResult.end_percent carries slippistats' value for comparison."""

    def __init__(self, port: int):
        self.port = int(port)
        self.reset()

    def reset(self) -> None:
        self.combo = None
        self.last_hit_animation = None

    @property
    def active(self) -> bool:
        return self.combo is not None

    def update(self, state_history) -> Optional[ComboResult]:
        if len(state_history) < 2:
            return None
        gs, pgs = state_history[-1], state_history[-2]
        p, pp = get_player(gs, self.port), get_player(pgs, self.port)
        op = opponent_port(gs, self.port)
        if p is None or pp is None or op is None:
            return None
        o, po = get_player(gs, op), get_player(pgs, op)
        if o is None or po is None:
            return None
        oa = act(o)
        o_damaged, o_grabbed, o_cmd = is_damaged(oa), is_grabbed(oa), is_cmd_grabbed(oa)
        o_hitstun = in_hitstun(o)
        dmg = float(o.percent) - float(po.percent)

        if act(p) != self.last_hit_animation or float(p.action_frame) < float(pp.action_frame):
            self.last_hit_animation = None

        if o_damaged or o_grabbed or o_cmd or o_hitstun:
            if self.combo is None:
                self.combo = dict(start_percent=float(po.percent), current_percent=float(o.percent),
                                  peak_percent=float(o.percent), n_moves=0, reset_counter=0)
            if dmg:                                   # slippistats: `if opnt_damage_taken:`
                if self.last_hit_animation is None:
                    self.combo["n_moves"] += 1
                self.last_hit_animation = act(pp)

        if self.combo is None:
            return None
        c = self.combo
        opp_lost = did_lose_stock(o, po)
        if not opp_lost:
            c["current_percent"] = float(o.percent)
            c["peak_percent"] = max(c["peak_percent"], float(o.percent))
        self_lost = did_lose_stock(p, pp)
        if continues_combo(state_history, op):
            c["reset_counter"] = 0
        else:
            c["reset_counter"] += 1
        if opp_lost or c["reset_counter"] > COMBO_LENIENCY or self_lost:
            self.combo = None
            return ComboResult(n_moves=c["n_moves"],
                               damage=max(0.0, c["peak_percent"] - c["start_percent"]),
                               did_kill=opp_lost, start_percent=c["start_percent"],
                               end_percent=float(po.percent))
        return None

    def flush(self, state_history) -> Optional[ComboResult]:
        """Close a combo still open when the match ends. The game-ending kill never reaches the
        stock decrement (the match is over first), so without this the winning combo is never
        scored; slippistats still lists it (it appends a combo when it opens). did_kill = the
        opponent is in a DEAD state on the last frame."""
        if self.combo is None or not state_history:
            return None
        c, self.combo = self.combo, None
        gs = state_history[-1]
        op = opponent_port(gs, self.port)
        o = get_player(gs, op) if op is not None else None
        return ComboResult(n_moves=c["n_moves"], damage=max(0.0, c["peak_percent"] - c["start_percent"]),
                           did_kill=o is not None and is_dying(act(o)), start_percent=c["start_percent"],
                           end_percent=c["current_percent"])


@dataclass
class TechResult:
    was_punished: bool


class TechTracker:
    """Streaming tech_compute: a tech situation is open while is_teching; when it ends, it was
    punished iff the exit state is_damaged."""

    def __init__(self, port: int):
        self.port = int(port)
        self.reset()

    def reset(self) -> None:
        self._active = False
        self._prev_teching = None

    def update(self, state_history) -> Optional[TechResult]:
        ps = get_player(state_history[-1], self.port)
        if ps is None:
            return None
        a = act(ps)
        curr = is_teching(a)
        prev = self._prev_teching
        self._prev_teching = curr
        if prev is None:                         # tech_compute skips frame 0
            return None
        if not curr:
            if prev and self._active:
                self._active = False
                return TechResult(was_punished=is_damaged(a))
            return None
        if not prev:
            self._active = True
        return None


@dataclass
class RecoveryResult:
    succeeded: bool


class RecoveryTracker:
    """slippistats' commented-out recovery_compute sketch, completed: a recovery opens on the frame
    the player goes off-stage (position test) while in hitstun; it fails if the player dies
    (DEAD action range) and succeeds on a ledge action or on landing (not airborne). The sketch's
    Randall exclusion (Yoshi's cloud) needs last_ground_id, which libmelee doesn't expose: landing
    on Randall counts as landing."""

    def __init__(self, port: int):
        self.port = int(port)
        self.reset()

    def reset(self) -> None:
        self._active = False
        self._prev_offstage = None

    def update(self, state_history) -> Optional[RecoveryResult]:
        gs = state_history[-1]
        ps = get_player(gs, self.port)
        if ps is None:
            return None
        st = stage_name(gs)
        off = is_offstage(ps, st)
        just_off = off and self._prev_offstage is False
        self._prev_offstage = off
        if not self._active:
            if just_off and in_hitstun(ps):
                self._active = True
            return None
        a = act(ps)
        if is_dying(a):
            self._active = False
            return RecoveryResult(succeeded=False)
        if is_ledge_action(a) or bool(ps.on_ground):
            self._active = False
            return RecoveryResult(succeeded=True)
        return None
