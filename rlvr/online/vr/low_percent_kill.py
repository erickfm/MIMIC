"""low-percent-kill VR — a bonus on top of stock-delta for early kills.

When the opponent loses a stock, if the death percent is below the
per-character bucket (p15 of percent-at-death) *and* the kill was
bot-attributed (heuristic SD-gate: opponent recently in hitstun / a
DAMAGE state), add `+bonus`. See docs/vr-proposals/low-percent-kill.md.

Buckets keyed by libmelee character enum int. `_DEFAULT_BUCKETS` is the
measured 9-character table; `kill_percent_buckets.json` (written by the
step-8a bucket scan), if present, overrides/extends it. Unmeasured
characters use `_FALLBACK_BUCKET`.
"""
from __future__ import annotations

import json
import os

from rlvr.online.slippi_stream import get_opponent, recently_in_hitstun_or_damage
from rlvr.online.vr.composite import VRModule

_DEFAULT_BUCKETS = {
    1: 105.0,    # Fox
    2: 100.0,    # Captain Falcon
    3: 140.0,    # Donkey Kong
    7: 120.0,    # Sheik
    9: 125.0,    # Peach
    13: 135.0,   # Yoshi
    15: 110.0,   # Jigglypuff
    18: 95.0,    # Marth
    22: 90.0,    # Falco
}
_FALLBACK_BUCKET = 110.0
_BUCKETS_JSON = os.path.join(os.path.dirname(__file__), "kill_percent_buckets.json")


def _load_buckets() -> dict:
    buckets = dict(_DEFAULT_BUCKETS)
    if os.path.exists(_BUCKETS_JSON):
        try:
            with open(_BUCKETS_JSON) as f:
                for k, v in json.load(f).items():
                    buckets[int(k)] = float(v)
        except Exception:
            pass
    return buckets


class LowPercentKillVR(VRModule):
    id = "low_percent_kill"

    def __init__(self, self_port: int = 1, bonus: float = 0.5, window: int = 12):
        self.self_port = self_port
        self.bonus = bonus
        self.window = window
        self.buckets = _load_buckets()
        self.reset()

    def reset(self) -> None:
        self._prev_opp_stock = None
        self._kills = 0
        self._low_kills = 0

    def observe(self, state_history) -> float:
        opp_ps = get_opponent(state_history[-1], self.self_port)
        if opp_ps is None:
            return 0.0
        opp_stock = int(opp_ps.stock)
        reward = 0.0
        if self._prev_opp_stock is not None and opp_stock < self._prev_opp_stock:
            self._kills += 1
            # Death percent: opp.percent resets to 0 on the death frame, so
            # take the peak over the window just before it.
            death_pct = 0.0
            for k in range(1, min(self.window, len(state_history)) + 1):
                p = get_opponent(state_history[-k], self.self_port)
                if p is not None:
                    death_pct = max(death_pct, float(p.percent))
            bucket = self.buckets.get(int(opp_ps.character), _FALLBACK_BUCKET)
            gated = recently_in_hitstun_or_damage(
                state_history, opp_ps.port, self.window)
            if gated and death_pct < bucket:
                reward += self.bonus
                self._low_kills += 1
        self._prev_opp_stock = opp_stock
        return reward

    def metadata(self) -> dict:
        return {"kills": self._kills, "low_percent_kills": self._low_kills}
