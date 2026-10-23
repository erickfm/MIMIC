"""Situational VRs: each runs a streaming state machine and emits on the frame its situation
resolves. Ported from git history (2a7b786^ rlvr/online/vr/*, rlvr/online/tasks/
shield_escape_online.py; 3c67390 rlvr/online/tasks/combo_extend_online.py) and rlvr/sim/lcancel.py.
"""
from __future__ import annotations

from rlvr.sim.vr.base import VRModule
from rlvr.sim.vr.stream import (
    ComboTracker, RecoveryTracker, TechTracker, act, get_opponent, get_player, in_any_grab,
    in_hitstun, in_punish_state, is_damaged, opponent_port,
)


class ComboLengthVR(VRModule):
    """At combo close: (clip(n_moves, 2, cap) - 2) / (cap - 2) — 2 moves = 0, cap+ moves = 1 — if
    n_moves >= 2 and the combo dealt >= min_damage %, else 0."""
    id = "combo_length"

    def __init__(self, self_port: int, cap: int = 8, min_damage: float = 5.0):
        super().__init__(self_port)
        self.cap, self.min_damage = cap, min_damage
        self.reset()

    def reset(self) -> None:
        self._tracker = ComboTracker(self.self_port)
        self._combos = self._rewarded = self._moves = 0

    def observe(self, state_history) -> float:
        res = self._tracker.update(state_history)
        return self._score(res) if res is not None else 0.0

    def _score(self, res) -> float:
        self._combos += 1
        self._moves += res.n_moves
        if res.n_moves < 2 or res.damage < self.min_damage:
            return 0.0
        self._rewarded += 1
        return (min(max(res.n_moves, 2), self.cap) - 2) / (self.cap - 2)

    def finalize(self, state_history) -> float:
        """Score a combo still open at match end (e.g. the game-ending kill combo)."""
        res = self._tracker.flush(state_history)
        return self._score(res) if res is not None else 0.0

    def metadata(self) -> dict:
        return {"combos": self._combos, "rewarded": self._rewarded, "total_moves": self._moves}


class ComboExtendVR(VRModule):
    """The first V(s)-discovered VR (combo_on_opp_damage). At combo close: min(damage, 80) / 80 if
    the combo had >= 2 moves and >= 5 % damage, else 0 — a kill combo scores by its damage, no
    stock bonus. Combo detection is the shared slippistats-faithful ComboTracker (the retired task
    had its own near-identical machine with a 45-frame out-of-punish gap)."""
    id = "combo_extend"

    def __init__(self, self_port: int, min_damage: float = 5.0, max_damage: float = 80.0):
        super().__init__(self_port)
        self.min_damage, self.max_damage = min_damage, max_damage
        self.reset()

    def reset(self) -> None:
        self._tracker = ComboTracker(self.self_port)
        self._combos = self._rewarded = 0
        self._damage = 0.0

    def observe(self, state_history) -> float:
        res = self._tracker.update(state_history)
        return self._score(res) if res is not None else 0.0

    def finalize(self, state_history) -> float:
        """Score a combo still open at match end (e.g. the game-ending kill combo)."""
        res = self._tracker.flush(state_history)
        return self._score(res) if res is not None else 0.0

    def _score(self, res) -> float:
        self._combos += 1
        if res.n_moves < 2 or res.damage < self.min_damage:
            return 0.0
        self._rewarded += 1
        self._damage += res.damage
        return min(res.damage, self.max_damage) / self.max_damage

    def metadata(self) -> dict:
        return {"combos": self._combos, "rewarded": self._rewarded, "damage": round(self._damage, 1)}


_NEG_INF = -10 ** 9


class NeutralWinLossVR(VRModule):
    """+win when the opponent enters punish state (ascending edge) while the bot has been out of
    punish for > K frames, -loss for the mirror; an edge counts only with damage dealt that frame
    or a grab (the proposal's safeguard). If the opposite edge follows within W frames it's a trade
    and a correction cancels the first one (net 0, a few frames apart). Punish state is
    stream.in_punish_state (the combo continuation set + thrown)."""
    id = "neutral_win_loss"

    def __init__(self, self_port: int, win: float = 1.0, loss: float = 1.0,
                 clean_frames: int = 30, trade_window: int = 4):
        super().__init__(self_port)
        self.win, self.loss, self.K, self.W = win, loss, clean_frames, trade_window
        self.reset()

    def reset(self) -> None:
        self._t = 0
        self._self_prev = self._opp_prev = False
        self._self_last = self._opp_last = _NEG_INF
        self._last_win_t = self._last_loss_t = None
        self._wins = self._losses = self._trades = 0

    def observe(self, state_history) -> float:
        gs = state_history[-1]
        me, opp = get_player(gs, self.self_port), get_opponent(gs, self.self_port)
        op = opponent_port(gs, self.self_port)
        if me is None or opp is None:
            return 0.0
        t = self._t
        self._t += 1
        me_p = in_punish_state(state_history, self.self_port)
        opp_p = in_punish_state(state_history, op)
        opp_edge, me_edge = opp_p and not self._opp_prev, me_p and not self._self_prev
        prev = state_history[-2] if len(state_history) >= 2 else None
        r = 0.0
        if opp_edge:
            prev_opp = get_opponent(prev, self.self_port) if prev is not None else None
            dealt = prev_opp is not None and float(opp.percent) > float(prev_opp.percent)
            if (t - self._self_last) > self.K and (dealt or in_any_grab(act(opp))):
                if self._last_loss_t is not None and (t - self._last_loss_t) <= self.W:
                    r += self.loss
                    self._last_loss_t = None
                    self._losses -= 1
                    self._trades += 1
                else:
                    r += self.win
                    self._wins += 1
                    self._last_win_t = t
        if me_edge:
            prev_me = get_player(prev, self.self_port) if prev is not None else None
            took = prev_me is not None and float(me.percent) > float(prev_me.percent)
            if (t - self._opp_last) > self.K and (took or in_any_grab(act(me))):
                if self._last_win_t is not None and (t - self._last_win_t) <= self.W:
                    r -= self.win
                    self._last_win_t = None
                    self._wins -= 1
                    self._trades += 1
                else:
                    r -= self.loss
                    self._losses += 1
                    self._last_loss_t = t
        if me_p:
            self._self_last = t
        if opp_p:
            self._opp_last = t
        self._self_prev, self._opp_prev = me_p, opp_p
        return r

    def metadata(self) -> dict:
        return {"neutral_wins": self._wins, "neutral_losses": self._losses, "trades": self._trades}


class RecoveryVR(VRModule):
    """-penalty when a recovery (knocked off-stage in hitstun) ends in death; a recovery that
    reaches the ledge or the ground is 0. Penalty-only, so going off-stage can't be farmed."""
    id = "recovery"

    def __init__(self, self_port: int, penalty: float = 0.25):
        super().__init__(self_port)
        self.penalty = penalty
        self.reset()

    def reset(self) -> None:
        self._tracker = RecoveryTracker(self.self_port)
        self._recoveries = self._failed = 0

    def observe(self, state_history) -> float:
        res = self._tracker.update(state_history)
        if res is None:
            return 0.0
        self._recoveries += 1
        if not res.succeeded:
            self._failed += 1
            return -self.penalty
        return 0.0

    def metadata(self) -> dict:
        return {"recoveries": self._recoveries, "failed": self._failed}


class TechVR(VRModule):
    """-penalty when a tech situation (TECH / DOWN / wall-ceiling bounce) ends directly in a damaged
    state; a clean exit is 0 (missing the tech on purpose costs nothing if it isn't punished)."""
    id = "tech"

    def __init__(self, self_port: int, penalty: float = 0.15):
        super().__init__(self_port)
        self.penalty = penalty
        self.reset()

    def reset(self) -> None:
        self._tracker = TechTracker(self.self_port)
        self._situations = self._punished = 0

    def observe(self, state_history) -> float:
        res = self._tracker.update(state_history)
        if res is None:
            return 0.0
        self._situations += 1
        if res.was_punished:
            self._punished += 1
            return -self.penalty
        return 0.0

    def metadata(self) -> dict:
        return {"tech_situations": self._situations, "punished": self._punished}


# --- L-cancel (rlvr/sim/lcancel.py's episode and reward, emitted on the resolving frame) ---------
AERIAL_STATES = {65, 66, 67, 68, 69}
LANDING_AIR_STATES = {70, 71, 72, 73, 74}
GENERIC_LANDING_STATES = {42, 43}
CANCELLED_MIN = {70: 7, 71: 11, 72: 10, 73: 9, 74: 9}      # Fox-specific; see rlvr/sim/lcancel.py
DEAD_STATES = set(range(0, 11))
LCANCEL_DAMAGE_STATES = set(range(75, 92))


class LCancelVR(VRModule):
    """An aerial starts (entered from a non-aerial, airborne, not in hitstun) and ends; if it ends
    in its LANDING_AIR state, the landing run's realized lag minus the move's cancelled floor is the
    avoidable lag: 0 -> `success`, else `miss`. Landings cut short by a hit/death and other exits
    (autocancel, interrupted) score nothing. Defaults are the old task's 1 / 0: as a dense reward
    that pays for *more* landings, so for mixing prefer success=0, miss=-1 (penalty-only).
    The start condition's hitstun test is the Slippi hitstun flag (rlvr/sim/lcancel.py reads
    libmelee's hitstun_frames_left)."""
    id = "l_cancel"

    def __init__(self, self_port: int, success: float = 1.0, miss: float = 0.0):
        super().__init__(self_port)
        self.success, self.miss = success, miss
        self.reset()

    def reset(self) -> None:
        self._prev = None
        self._active = False
        self._pending = None            # (landing state, run length)
        self._succ = self._miss = self._dropped = 0

    def observe(self, state_history) -> float:
        ps = get_player(state_history[-1], self.self_port)
        if ps is None:
            return 0.0
        a = act(ps)
        r = 0.0
        if self._pending is not None:
            state, run = self._pending
            if a == state:
                self._pending = (state, run + 1)
            else:
                self._pending = None
                if a in DEAD_STATES or a in LCANCEL_DAMAGE_STATES:
                    self._dropped += 1
                elif max(0, run - CANCELLED_MIN[state]) == 0:
                    self._succ += 1
                    r = self.success
                else:
                    self._miss += 1
                    r = self.miss
        if self._active:
            if a not in AERIAL_STATES or a in LANDING_AIR_STATES or a in GENERIC_LANDING_STATES:
                self._active = False
                if a in LANDING_AIR_STATES:
                    self._pending = (a, 1)
                else:
                    self._dropped += 1
        elif (self._prev is not None and a in AERIAL_STATES and self._prev not in AERIAL_STATES
              and not bool(ps.on_ground) and not in_hitstun(ps)):
            self._active = True
        self._prev = a
        return r

    def metadata(self) -> dict:
        return {"success": self._succ, "miss": self._miss, "dropped": self._dropped}


# --- shield escape (rlvr/online/tasks/shield_escape_online.py) -----------------------------------
SHIELD_FAMILY = {178, 179, 180, 181, 182}
SHIELD_ACTIONABLE = {178, 179, 182}           # GUARD_ON, GUARD, GUARD_REFLECT
SHIELD_BREAK_STATES = set(range(205, 212))
SMALL_SHIELD = 30.0
SHIELD_HIT = 8.0
SHIELD_LOOKBACK = 3


class ShieldEscapeVR(VRModule):
    """Escape a pressured shield. A shield hit (shield strength dropped by >= 8 vs any of the last
    1-3 frames, while in the shield family) arms the situation; it opens at the first actionable
    shield frame after it (GUARD_ON / GUARD / GUARD_REFLECT — i.e. once shieldstun ends) if the
    shield is then < 30 of 60, and resolves when the bot leaves the shield family: shield break ->
    `broken` (-1), hit out (a damaged state) -> `hit` (0), any other exit (roll, spotdodge,
    release, jump, grab, OoS attack) -> `escape` (+1).

    This is the offline tagger's definition (git ece68f9^ rlvr/tasks/escape_pressured_shield.py).
    The Dolphin online task (2a7b786^ shield_escape_online.py) required the 1-3-frame drop to
    coincide with an actionable frame, but the hit puts the shield in shieldstun first, so it could
    essentially never open — why the May online run "never triggered". Also fixed: its damage set
    was range(75, 91) (missed DAMAGE_FLY_ROLL 91), and its escape set listed the missed-tech getup
    rolls 188/189/196-198 (unreachable from shield; harmless)."""
    id = "shield_escape"

    def __init__(self, self_port: int, escape: float = 1.0, hit: float = 0.0, broken: float = -1.0):
        super().__init__(self_port)
        self.escape, self.hit, self.broken = escape, hit, broken
        self.reset()

    def reset(self) -> None:
        self._armed = self._active = False
        self._counts = {"escape": 0, "hit": 0, "broken": 0}

    def observe(self, state_history) -> float:
        ps = get_player(state_history[-1], self.self_port)
        if ps is None:
            return 0.0
        a = act(ps)
        if self._active:
            if a in SHIELD_FAMILY:
                return 0.0
            self._active = False
            if a in SHIELD_BREAK_STATES:
                self._counts["broken"] += 1
                return self.broken
            if is_damaged(a):
                self._counts["hit"] += 1
                return self.hit
            self._counts["escape"] += 1
            return self.escape
        if a not in SHIELD_FAMILY:
            self._armed = False
            return 0.0
        drop = 0.0
        for k in range(2, SHIELD_LOOKBACK + 2):
            if len(state_history) >= k:
                q = get_player(state_history[-k], self.self_port)
                if q is not None:
                    drop = max(drop, float(q.shield_strength) - float(ps.shield_strength))
        if drop >= SHIELD_HIT:
            self._armed = True
        if self._armed and a in SHIELD_ACTIONABLE:
            self._armed = False
            if float(ps.shield_strength) < SMALL_SHIELD:
                self._active = True
        return 0.0

    def metadata(self) -> dict:
        return dict(self._counts)
