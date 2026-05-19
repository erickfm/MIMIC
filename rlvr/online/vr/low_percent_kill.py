"""low-percent-kill VR — a bonus on top of stock-delta for early kills.

When the opponent loses a stock, if the death percent is below the
per-character bucket (p15 of percent-at-death) *and* the kill was
bot-attributed (the opponent was recently hit), add `+bonus`. See
docs/vr-proposals/low-percent-kill.md.

Both the kill-attribution gate and the death-percent read are *streaming*,
not backward scans from the stock-decrement frame. The `stock` counter
ticks down a long, variable time after the killing hit — the KO'd
character sits in a DEAD action state (range 0-10) for 1.5-3 s first — so
any fixed look-back window from the decrement frame lands entirely inside
DEAD frames, missing both the hit (gate) and the pre-death percent (which
has already reset to 0). The gate uses `OppHitRecencyTracker`
(DEAD-transparent); the death percent is the running peak of opponent
percent over the life, read at the stock-loss frame. (This is the same
stock-decrement-lag bug class fixed in `stock_delta` — see
docs/research-notes-2026-05-18.md.)

Buckets keyed by libmelee character enum int. `_DEFAULT_BUCKETS` is the
measured 9-character table; `kill_percent_buckets.json` (written by the
step-8a bucket scan), if present, overrides/extends it. Unmeasured
characters use `_FALLBACK_BUCKET`.
"""
from __future__ import annotations

import json
import os

from rlvr.online.slippi_stream import OppHitRecencyTracker, get_opponent
from rlvr.online.vr.composite import VRModule

# Alive-frames of "recently hit" memory for the kill-attribution gate.
# DEAD frames are transparent (see OppHitRecencyTracker), so this only
# bounds the gap of *alive*, non-hit frames between the last hit and death.
OPP_HIT_MEMORY = 90

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

    def __init__(self, self_port: int = 1, bonus: float = 0.5,
                 hit_memory: int = OPP_HIT_MEMORY):
        self.self_port = self_port
        self.bonus = bonus
        self.hit_memory = hit_memory
        self.buckets = _load_buckets()
        self.reset()

    def reset(self) -> None:
        self._prev_opp_stock = None
        self._opp_peak_percent = 0.0
        self._kills = 0
        self._low_kills = 0
        self._opp_hit = OppHitRecencyTracker(self.hit_memory)

    def observe(self, state_history) -> float:
        opp_ps = get_opponent(state_history[-1], self.self_port)
        if opp_ps is None:
            return 0.0
        # Advance both streaming trackers every frame — DEAD frames
        # included. The SD-gate counter treats DEAD as transparent; the
        # peak is unaffected (percent is already 0 by the DEAD frames).
        self._opp_hit.update(opp_ps)
        self._opp_peak_percent = max(self._opp_peak_percent,
                                     float(opp_ps.percent))
        opp_stock = int(opp_ps.stock)
        reward = 0.0
        if self._prev_opp_stock is not None and opp_stock < self._prev_opp_stock:
            reward += self._score_kill(opp_ps)
        self._prev_opp_stock = opp_stock
        return reward

    def _score_kill(self, opp_ps) -> float:
        """Score one opponent stock loss and reset the per-life streaming
        state (hit-recency + percent peak) for the opponent's next life."""
        self._kills += 1
        death_pct = self._opp_peak_percent
        bucket = self.buckets.get(int(opp_ps.character), _FALLBACK_BUCKET)
        reward = 0.0
        if self._opp_hit.recently_hit and death_pct < bucket:
            reward = self.bonus
            self._low_kills += 1
        self._opp_hit.reset()
        self._opp_peak_percent = 0.0
        return reward

    def finalize(self, state_history) -> float:
        """Reconcile a final opponent death the per-frame loop did not
        observe (rare: the actor stops calling observe before the last
        stock-0 frame). Mirrors StockDeltaVR.finalize. Normally 0."""
        if not state_history:
            return 0.0
        opp_ps = get_opponent(state_history[-1], self.self_port)
        if (opp_ps is not None and self._prev_opp_stock is not None
                and int(opp_ps.stock) < self._prev_opp_stock):
            return self._score_kill(opp_ps)
        return 0.0

    def metadata(self) -> dict:
        return {"kills": self._kills, "low_percent_kills": self._low_kills}
