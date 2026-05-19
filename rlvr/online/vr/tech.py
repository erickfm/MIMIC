"""tech VR — a penalty for being punished out of a tech situation.

Penalty-only: when the bot exits a tech situation directly into a hit
(`TechTracker` reports `was_punished`), emit `−penalty`; a clean tech is
0. Across re-freeze iterations of the baseline opponent this pressures
unpredictable teching. See docs/vr-proposals/tech.md.
"""
from __future__ import annotations

from rlvr.online.slippi_stream import TechTracker, get_player
from rlvr.online.vr.composite import VRModule


class TechVR(VRModule):
    id = "tech"

    def __init__(self, self_port: int = 1, penalty: float = 0.15):
        self.self_port = self_port
        self.penalty = penalty
        self.reset()

    def reset(self) -> None:
        self._tracker = TechTracker()
        self._tech_situations = 0
        self._punished = 0

    def observe(self, state_history) -> float:
        self_ps = get_player(state_history[-1], self.self_port)
        if self_ps is None:
            return 0.0
        result = self._tracker.update(self_ps)
        if result is None:
            return 0.0
        self._tech_situations += 1
        if result.was_punished:
            self._punished += 1
            return -self.penalty
        return 0.0

    def metadata(self) -> dict:
        return {"tech_situations": self._tech_situations,
                "punished": self._punished}
