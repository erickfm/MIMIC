"""recovery VR — a penalty for a failed recovery.

Penalty-only: a recovery situation opens when the bot is *knocked*
off-stage (`off_stage AND in_hitstun` — a voluntary trip opens nothing).
It closes on success (back on stage / grabbed ledge → 0) or failure (lost
the stock while still recovering → `−penalty`). See
docs/vr-proposals/recovery.md.
"""
from __future__ import annotations

from rlvr.online.slippi_stream import RecoveryTracker, get_player
from rlvr.online.vr.composite import VRModule


class RecoveryVR(VRModule):
    id = "recovery"

    def __init__(self, self_port: int = 1, penalty: float = 0.25):
        self.self_port = self_port
        self.penalty = penalty
        self.reset()

    def reset(self) -> None:
        self._tracker = RecoveryTracker()
        self._recoveries = 0
        self._failed = 0

    def observe(self, state_history) -> float:
        self_ps = get_player(state_history[-1], self.self_port)
        if self_ps is None:
            return 0.0
        result = self._tracker.update(self_ps)
        if result is None:
            return 0.0
        self._recoveries += 1
        if not result.succeeded:
            self._failed += 1
            return -self.penalty
        return 0.0

    def metadata(self) -> dict:
        return {"recoveries": self._recoveries, "failed": self._failed}
