"""Verifiable-reward (VR) modules for online RL.

See docs/vr-proposals/ for the per-VR designs and the plan file for the
whole-match `CompositeVRTask` architecture.

`VR_REGISTRY` maps a VR id (the `--vrs` CLI token) to its `VRModule`
class. `loop.py` builds a `CompositeVRTask` from it.
"""
from rlvr.online.vr.composite import CompositeVRTask, VRModule
from rlvr.online.vr.combo_length import ComboLengthVR
from rlvr.online.vr.damage_delta import DamageDeltaVR
from rlvr.online.vr.low_percent_kill import LowPercentKillVR
from rlvr.online.vr.neutral_win_loss import NeutralWinLossVR
from rlvr.online.vr.recovery import RecoveryVR
from rlvr.online.vr.stock_delta import StockDeltaVR
from rlvr.online.vr.tech import TechVR

VR_REGISTRY = {
    cls.id: cls for cls in (
        StockDeltaVR,
        DamageDeltaVR,
        NeutralWinLossVR,
        ComboLengthVR,
        LowPercentKillVR,
        TechVR,
        RecoveryVR,
    )
}

# Pre-seeded composite weights — the default when `--vr-weights` is not
# given. These are an *analytic* balance from the clash-review event-
# frequency estimates (docs/vr-proposals/, the plan's step 8b): the dense
# VRs are down-weighted so they do not bury `stock-delta` (~8 events/game
# at |r|=1). neutral-win fires ~30x/game and combos ~15x, hence 0.25 /
# 0.5; the sparse VRs sit at 1.0. A full empirical frequency measurement
# would refine these — treat them as a starting point, not a final tuning.
DEFAULT_VR_WEIGHTS = {
    "stock_delta": 1.0,
    "damage_delta": 1.0,
    "neutral_win_loss": 0.25,
    "combo_length": 0.5,
    "low_percent_kill": 1.0,
    "tech": 1.0,
    "recovery": 1.0,
}

__all__ = [
    "VRModule", "CompositeVRTask", "VR_REGISTRY", "DEFAULT_VR_WEIGHTS",
    "StockDeltaVR", "DamageDeltaVR", "NeutralWinLossVR", "ComboLengthVR",
    "LowPercentKillVR", "TechVR", "RecoveryVR",
]
