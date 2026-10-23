"""Outcome VRs: stock_delta, damage_delta, low_percent_kill (ported from git 2a7b786^
rlvr/online/vr/{stock_delta,damage_delta,low_percent_kill}.py; docs/vr-proposals/)."""
from __future__ import annotations

import json
import os

from rlvr.sim.vr.base import VRModule
from rlvr.sim.vr.stream import (
    ComboTracker, OppHitRecencyTracker, act, get_opponent, get_player, is_dying,
)

# Alive frames of "recently hit" memory for the opponent self-destruct gate (DEAD frames are
# transparent, so this bounds only the alive, un-hit gap between the last hit and the death).
OPP_HIT_MEMORY = 90


class KillGate:
    """Was an opponent death earned? Yes if the opponent was hit within OPP_HIT_MEMORY alive frames
    OR the bot's combo (slippistats' ComboTracker) is still open — slippistats' own kill
    attribution. The hit memory alone (the Dolphin-era gate) scored edgeguard / gimp kills as self-
    destructs: on 40 master-Fox replays, 57 of 252 deaths came 113-183 frames after the last hit
    (knocked off, never made it back) with the bot's combo still open; 11 had neither (the true
    self-destruct candidates). Call update() every frame before reading earned()."""

    def __init__(self, self_port: int, hit_memory: int = OPP_HIT_MEMORY):
        self.hit = OppHitRecencyTracker(hit_memory)
        self.combo = ComboTracker(self_port)
        self._combo_open = False

    def update(self, state_history, opp) -> None:
        self.hit.update(opp)
        self._combo_open = self.combo.active      # before this frame's update: open at the death
        self.combo.update(state_history)

    def earned(self) -> bool:
        return self.hit.recently_hit or self._combo_open

    def next_life(self) -> None:
        self.hit.reset()


class StockDeltaVR(VRModule):
    """+1 when the opponent dies and the kill was earned (KillGate: a recent hit or the bot's combo
    still open; an opponent self-destruct is filtered), -1 when the bot dies (never gated). A death is the rising edge of the DEAD action range (0-10), which
    happens on the blast-zone frame, ~1.5-3 s before `stock` decrements. finalize() reconciles a
    stock decrement no observe() saw (normally 0)."""
    id = "stock_delta"

    def __init__(self, self_port: int, hit_memory: int = OPP_HIT_MEMORY):
        super().__init__(self_port)
        self.hit_memory = hit_memory
        self.reset()

    def reset(self) -> None:
        self._prev_self_dying = self._prev_opp_dying = False
        self._prev_self_stock = self._prev_opp_stock = None
        self._kills = self._deaths = self._filtered_opp_sds = 0
        self._gate = KillGate(self.self_port, self.hit_memory)

    def observe(self, state_history) -> float:
        gs = state_history[-1]
        me, opp = get_player(gs, self.self_port), get_opponent(gs, self.self_port)
        if me is None or opp is None:
            return 0.0
        self._gate.update(state_history, opp)
        me_dying, opp_dying = is_dying(act(me)), is_dying(act(opp))
        r = 0.0
        if opp_dying and not self._prev_opp_dying:
            if self._gate.earned():
                r += 1.0
                self._kills += 1
            else:
                self._filtered_opp_sds += 1
            self._gate.next_life()
        if me_dying and not self._prev_self_dying:
            r -= 1.0
            self._deaths += 1
        self._prev_self_dying, self._prev_opp_dying = me_dying, opp_dying
        self._prev_self_stock, self._prev_opp_stock = int(me.stock), int(opp.stock)
        return r

    def finalize(self, state_history) -> float:
        if not state_history:
            return 0.0
        gs = state_history[-1]
        me, opp = get_player(gs, self.self_port), get_opponent(gs, self.self_port)
        r = 0.0
        if opp is not None and self._prev_opp_stock is not None and int(opp.stock) < self._prev_opp_stock:
            n = self._prev_opp_stock - int(opp.stock)
            if self._gate.earned():
                r += n
                self._kills += n
        if me is not None and self._prev_self_stock is not None and int(me.stock) < self._prev_self_stock:
            n = self._prev_self_stock - int(me.stock)
            r -= n
            self._deaths += n
        return float(r)

    def metadata(self) -> dict:
        return {"kills": self._kills, "deaths": self._deaths, "filtered_opp_sds": self._filtered_opp_sds}


class DamageDeltaVR(VRModule):
    """+lam_give * (opponent percent rise) - lam_take * (bot percent rise), per frame, positive
    deltas only (a drop is a respawn reset, not healing). 0.04 per % (the Dolphin-era tuned value):
    a stock's worth of damage (~120%) is ~4.8."""
    id = "damage_delta"

    def __init__(self, self_port: int, lam_give: float = 0.04, lam_take: float = 0.04):
        super().__init__(self_port)
        self.lam_give, self.lam_take = lam_give, lam_take
        self.reset()

    def reset(self) -> None:
        self._prev_self = self._prev_opp = None
        self._dealt = self._taken = 0.0

    def observe(self, state_history) -> float:
        gs = state_history[-1]
        me, opp = get_player(gs, self.self_port), get_opponent(gs, self.self_port)
        if me is None or opp is None:
            return 0.0
        mp, op = float(me.percent), float(opp.percent)
        r = 0.0
        if self._prev_opp is not None and op - self._prev_opp > 0:
            r += self.lam_give * (op - self._prev_opp)
            self._dealt += op - self._prev_opp
        if self._prev_self is not None and mp - self._prev_self > 0:
            r -= self.lam_take * (mp - self._prev_self)
            self._taken += mp - self._prev_self
        self._prev_self, self._prev_opp = mp, op
        return r

    def metadata(self) -> dict:
        return {"damage_dealt": round(self._dealt, 1), "damage_taken": round(self._taken, 1)}


# Per-character kill-percent buckets (p15 of percent-at-death, libmelee character ids). The
# defaults are the measured 9-character table; kill_percent_buckets.json (the Dolphin-era bucket
# scan, 15 characters) extends/overrides it.
_DEFAULT_BUCKETS = {1: 105.0, 2: 100.0, 3: 140.0, 7: 120.0, 9: 125.0, 13: 135.0, 15: 110.0,
                    18: 95.0, 22: 90.0}
_FALLBACK_BUCKET = 110.0
_BUCKETS_JSON = os.path.join(os.path.dirname(__file__), "kill_percent_buckets.json")


def _load_buckets() -> dict:
    b = dict(_DEFAULT_BUCKETS)
    if os.path.exists(_BUCKETS_JSON):
        with open(_BUCKETS_JSON) as f:
            b.update({int(k): float(v) for k, v in json.load(f).items()})
    return b


class LowPercentKillVR(VRModule):
    """+bonus when the opponent dies, the kill was earned (KillGate, as stock_delta), and the death
    percent is below its character's bucket. Death percent = the running peak of the opponent's
    percent over that life (percent is already 0 during the DEAD frames, so a look-back would miss
    it)."""
    id = "low_percent_kill"

    def __init__(self, self_port: int, bonus: float = 0.5, hit_memory: int = OPP_HIT_MEMORY):
        super().__init__(self_port)
        self.bonus, self.hit_memory = bonus, hit_memory
        self.buckets = _load_buckets()
        self.reset()

    def reset(self) -> None:
        self._prev_opp_stock = None
        self._prev_opp_dying = False
        self._peak = 0.0
        self._kills = self._low_kills = 0
        self._gate = KillGate(self.self_port, self.hit_memory)

    def _score(self, opp) -> float:
        self._kills += 1
        ch = getattr(opp.character, "value", opp.character)
        r = 0.0
        if self._gate.earned() and self._peak < self.buckets.get(int(ch), _FALLBACK_BUCKET):
            r = self.bonus
            self._low_kills += 1
        self._gate.next_life()
        self._peak = 0.0
        return r

    def observe(self, state_history) -> float:
        opp = get_opponent(state_history[-1], self.self_port)
        if opp is None:
            return 0.0
        self._gate.update(state_history, opp)
        self._peak = max(self._peak, float(opp.percent))
        dying = is_dying(act(opp))
        r = self._score(opp) if dying and not self._prev_opp_dying else 0.0
        self._prev_opp_dying = dying
        self._prev_opp_stock = int(opp.stock)
        return r

    def finalize(self, state_history) -> float:
        if not state_history:
            return 0.0
        opp = get_opponent(state_history[-1], self.self_port)
        if opp is not None and self._prev_opp_stock is not None and int(opp.stock) < self._prev_opp_stock:
            return self._score(opp)
        return 0.0

    def metadata(self) -> dict:
        return {"kills": self._kills, "low_percent_kills": self._low_kills}
