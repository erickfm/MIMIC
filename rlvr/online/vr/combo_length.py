"""combo-length VR — reward longer combos by move count.

A combo (detected by `ComboTracker`) scores, at close,
`(clip(n_moves, 2, CAP) − 2) / (CAP − 2)` — 2 moves = 0, CAP+ = 1.0 —
gated on ≥ 5% total damage and `n_moves ≥ 2`. See
docs/vr-proposals/combo-length.md.
"""
from __future__ import annotations

from rlvr.online.slippi_stream import ComboTracker, get_opponent, get_player
from rlvr.online.vr.composite import VRModule


class ComboLengthVR(VRModule):
    id = "combo_length"

    def __init__(self, self_port: int = 1, cap: int = 8, min_damage: float = 5.0):
        self.self_port = self_port
        self.cap = cap
        self.min_damage = min_damage
        self.reset()

    def reset(self) -> None:
        self._tracker = ComboTracker()
        self._combos = 0
        self._rewarded = 0
        self._total_moves = 0

    def observe(self, state_history) -> float:
        gs = state_history[-1]
        self_ps = get_player(gs, self.self_port)
        opp_ps = get_opponent(gs, self.self_port)
        if self_ps is None or opp_ps is None:
            return 0.0
        result = self._tracker.update(self_ps, opp_ps)
        if result is None:
            return 0.0
        self._combos += 1
        self._total_moves += result.n_moves
        # Damage-floor gate + n_moves >= 2 — see the VR doc.
        if result.n_moves < 2 or result.damage < self.min_damage:
            return 0.0
        n = min(max(result.n_moves, 2), self.cap)
        self._rewarded += 1
        return float(n - 2) / float(self.cap - 2)

    def metadata(self) -> dict:
        return {"combos": self._combos, "rewarded": self._rewarded,
                "total_moves": self._total_moves}
