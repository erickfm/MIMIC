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
# given. Originally seeded from analytic event-frequency estimates (the
# plan's step 8b); refined after the first 7-VR run's live HUD data
# (see rlvr/eval/training_web/server.py per-VR Σ panel) showed:
#   - combo_length at weight 0.5 still ate 47% of total gradient signal
#     (events fired more often + at bigger magnitudes than estimated) →
#     down-weighted to 0.15 to bring it to ~15%.
#   - damage_delta contributed only 1.3% even with weight 1.0 — too
#     small to shape behavior. Fixed at the VR-internal level by
#     bumping lam_give / lam_take 0.003 → 0.015 (see damage_delta.py),
#     so weight 1.0 here now produces meaningful per-hit signal.
# neutral_win_loss stays down-weighted at 0.25 (still fires ~30x/match);
# the sparse VRs (stock, low_pct, tech, recovery) stay at 1.0.
# Tuned (after the first 7-VR run's live data) so that, given the
# empirically observed raw per-match magnitudes at weight=1.0
#   stock 0.86, damage 0.67 (post-lam-bump to 0.04), neutral 1.04,
#   combo 3.70, low_pct 0.16, tech 0.27, recovery 0.48,
# the *weighted* per-match magnitudes descend in this priority order:
#   stock_delta > damage_delta > neutral_win_loss > recovery
#     > combo_length > low_percent_kill > tech.
# Knob philosophy: each VR's *per-event magnitude* lives inside the VR
# module itself (e.g. damage_delta.lam_give/lam_take, low_percent_kill
# .bonus). The weight here is purely the "relative importance" dial —
# no doubling-up. combo_length's natural magnitude is ~4x the headline
# stock_delta, so it needs a heavy down-weight (0.08) to land at #5.
DEFAULT_VR_WEIGHTS = {
    "stock_delta": 1.0,
    "damage_delta": 1.0,
    "neutral_win_loss": 0.5,
    "recovery": 0.8,
    "combo_length": 0.08,
    "low_percent_kill": 1.3,
    "tech": 0.6,
}

__all__ = [
    "VRModule", "CompositeVRTask", "VR_REGISTRY", "DEFAULT_VR_WEIGHTS",
    "StockDeltaVR", "DamageDeltaVR", "NeutralWinLossVR", "ComboLengthVR",
    "LowPercentKillVR", "TechVR", "RecoveryVR",
]
