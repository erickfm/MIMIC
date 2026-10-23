"""Verifiable rewards (VRs) for RL in the native sim.

Every VR the project has had code for, ported from git history to the sim's libmelee GameStates
(see stream.py for the slippistats port and what changed). One VRModule instance per rewarded
player per env; feed it a deque of GameStates (newest last, maxlen HISTORY) every in-game frame.

  modules = build("stock_delta:1,damage_delta:1,l_cancel:0.2", self_port=1)
  r = sum(w * m.observe(history) for m, w in modules)

HISTORY = 8 frames: the longest look-back any VR makes is 4 frames (shield_escape's 1-3-frame
shield-drop window plus the current frame; is_wavedashing's 3 frames before the current one);
everything else is carried in streaming state. 8 leaves margin.

Importing this package installs the Slippi state-flag hook (flags.py): hitstun / hitlag come from
the bit flags, as in slippistats. Import it in the worker before parsing any frame.
"""
from rlvr.sim.vr import flags as _flags
from rlvr.sim.vr.base import VRModule
from rlvr.sim.vr.outcome import DamageDeltaVR, LowPercentKillVR, StockDeltaVR
from rlvr.sim.vr.situational import (
    ComboExtendVR, ComboLengthVR, LCancelVR, NeutralWinLossVR, RecoveryVR, ShieldEscapeVR, TechVR,
)

_flags.install()

HISTORY = 8

VRS = {cls.id: cls for cls in (
    LCancelVR, ShieldEscapeVR, ComboExtendVR, StockDeltaVR, DamageDeltaVR, LowPercentKillVR,
    ComboLengthVR, NeutralWinLossVR, RecoveryVR, TechVR,
)}


def build(spec: str, self_port: int):
    """'id:weight,id:weight' (weight defaults to 1) -> [(VRModule, weight), ...]."""
    out = []
    for tok in (t.strip() for t in spec.split(",")):
        if not tok:
            continue
        vid, _, w = tok.partition(":")
        if vid not in VRS:
            raise ValueError(f"unknown VR {vid!r}; known: {sorted(VRS)}")
        out.append((VRS[vid](self_port), float(w) if w else 1.0))
    return out


__all__ = ["HISTORY", "VRS", "VRModule", "build"]
