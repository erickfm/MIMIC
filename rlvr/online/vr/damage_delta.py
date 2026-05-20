"""damage-delta VR — the dense shaping companion to stock-delta.

`+lam_give * Δopp_percent − lam_take * Δself_percent` per frame, positive
deltas only (a negative delta is a respawn reset, not healing). See
docs/vr-proposals/damage-delta.md.

`lam_give` and `lam_take` are separate knobs — their ratio is the bot's
aggression dial. Defaults bumped from the original 0.003 to 0.04 after
the first 7-VR run showed damage_delta contributing only 1.3% of total
gradient signal (live HUD data) — too small to actually shape behavior.
At 0.04, a full stock of damage (~120%) sums to ~4.8 — meaningfully
bigger than stock_delta's ±1, encoding that "the journey of dealing
damage" is the dominant per-event signal among the dense shaping VRs.
This is the per-event-magnitude knob; relative importance among VRs
lives in DEFAULT_VR_WEIGHTS (rlvr/online/vr/__init__.py), where this VR
sits at weight 1.0 (no doubling-up).
"""
from __future__ import annotations

from rlvr.online.slippi_stream import get_opponent, get_player
from rlvr.online.vr.composite import VRModule


class DamageDeltaVR(VRModule):
    id = "damage_delta"

    def __init__(self, self_port: int = 1,
                 lam_give: float = 0.04, lam_take: float = 0.04):
        self.self_port = self_port
        self.lam_give = lam_give
        self.lam_take = lam_take
        self.reset()

    def reset(self) -> None:
        self._prev_self_pct = None
        self._prev_opp_pct = None
        self._dealt = 0.0
        self._taken = 0.0

    def observe(self, state_history) -> float:
        gs = state_history[-1]
        self_ps = get_player(gs, self.self_port)
        opp_ps = get_opponent(gs, self.self_port)
        if self_ps is None or opp_ps is None:
            return 0.0
        self_pct = float(self_ps.percent)
        opp_pct = float(opp_ps.percent)
        reward = 0.0

        if self._prev_opp_pct is not None:
            d = opp_pct - self._prev_opp_pct
            if d > 0.0:                       # positive only — ignore respawn reset
                reward += self.lam_give * d
                self._dealt += d
        if self._prev_self_pct is not None:
            d = self_pct - self._prev_self_pct
            if d > 0.0:
                reward -= self.lam_take * d
                self._taken += d

        self._prev_self_pct = self_pct
        self._prev_opp_pct = opp_pct
        return reward

    def metadata(self) -> dict:
        return {"damage_dealt": round(self._dealt, 1),
                "damage_taken": round(self._taken, 1)}
