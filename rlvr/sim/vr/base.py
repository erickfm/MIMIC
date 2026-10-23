"""VRModule: one verifiable-reward component.

Lifecycle (driven by the sim RL loop, one instance per rewarded player per env):
  reset()                    — at match start.
  observe(state_history)     — every in-game frame; returns this VR's *unweighted* reward for the
                               frame (0 on most frames; situational VRs run a streaming state
                               machine and emit on the frame the situation resolves).
  finalize(state_history)    — once at match end; default 0.
  metadata()                 — per-match diagnostics.

`state_history` is a deque of libmelee GameStates, newest last (maxlen rlvr.sim.vr.HISTORY).
Same semantics as the Dolphin-era rlvr/online/vr/composite.VRModule (git 2a7b786^).
"""
from typing import Any, Dict


class VRModule:
    id: str = "vr"

    def __init__(self, self_port: int):
        self.self_port = int(self_port)

    def reset(self) -> None:
        """Clear per-match state."""

    def observe(self, state_history) -> float:
        raise NotImplementedError

    def finalize(self, state_history) -> float:
        return 0.0

    def metadata(self) -> Dict[str, Any]:
        return {}
