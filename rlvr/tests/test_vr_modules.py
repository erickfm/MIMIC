"""Tests for the 7 VR modules and CompositeVRTask (rlvr/online/vr/)."""
from __future__ import annotations

from rlvr.online.vr import (
    CompositeVRTask, ComboLengthVR, DamageDeltaVR, LowPercentKillVR,
    NeutralWinLossVR, RecoveryVR, StockDeltaVR, TechVR, VRModule, VR_REGISTRY,
)
from rlvr.tests._melee_fixtures import (
    DAMAGE, DYING, NEUTRAL, TECH, gs, history, ps,
)


def run_vr(vr, frames):
    """Feed GameStates one at a time; return the per-frame rewards."""
    h = history()
    vr.reset()
    out = []
    for fr in frames:
        h.append(fr)
        out.append(vr.observe(h))
    return out


def test_registry_has_seven():
    assert set(VR_REGISTRY) == {
        "stock_delta", "damage_delta", "neutral_win_loss", "combo_length",
        "low_percent_kill", "tech", "recovery",
    }


# --- stock_delta ---------------------------------------------------------
def test_stock_delta_kill_counts():
    frames = [
        gs(ps(1), ps(2, stock=4)),
        gs(ps(1), ps(2, stock=4, action=DAMAGE, hitstun=10.0)),
        gs(ps(1), ps(2, stock=3)),       # opp lost a stock (recently hit)
    ]
    r = run_vr(StockDeltaVR(self_port=1), frames)
    assert sum(r) == 1.0


def test_stock_delta_opp_sd_filtered():
    """An opponent stock loss with no recent hitstun/DAMAGE is a
    self-destruct — no +1."""
    frames = [gs(ps(1), ps(2, stock=4)) for _ in range(6)]
    frames.append(gs(ps(1), ps(2, stock=3)))   # clean SD
    r = run_vr(StockDeltaVR(self_port=1), frames)
    assert sum(r) == 0.0


def test_stock_delta_self_loss_always_penalized():
    frames = [
        gs(ps(1, stock=4), ps(2)),
        gs(ps(1, stock=3), ps(2)),       # bot SD — still −1, ungated
    ]
    r = run_vr(StockDeltaVR(self_port=1), frames)
    assert sum(r) == -1.0


def test_stock_delta_star_ko_dead_frames_transparent():
    """Top/star KO: the opponent is hit, then sits in a DEAD action state
    for many frames before `stock` finally decrements. The SD-gate must
    still credit the kill — DEAD frames are transparent, so the hit stays
    'remembered' across the long death sequence."""
    frames = [
        gs(ps(1), ps(2, stock=4)),
        gs(ps(1), ps(2, stock=4, action=DAMAGE, hitstun=12.0)),   # the hit
    ]
    # long DEAD stretch (star-KO delay) before the stock count ticks down
    frames += [gs(ps(1), ps(2, stock=4, action=DYING)) for _ in range(150)]
    frames.append(gs(ps(1), ps(2, stock=3, action=DYING)))
    r = run_vr(StockDeltaVR(self_port=1), frames)
    assert sum(r) == 1.0


def test_stock_delta_late_controlled_fall_filtered():
    """Opponent hit, then a long controlled fall (alive, not in a hit
    reaction) before dying — past the alive-frame hit-memory, so it gates
    out as a self-destruct."""
    frames = [
        gs(ps(1), ps(2, stock=4)),
        gs(ps(1), ps(2, stock=4, action=DAMAGE, hitstun=12.0)),
    ]
    frames += [gs(ps(1), ps(2, stock=4)) for _ in range(120)]   # alive, falling
    frames.append(gs(ps(1), ps(2, stock=3)))
    r = run_vr(StockDeltaVR(self_port=1), frames)
    assert sum(r) == 0.0


# --- damage_delta --------------------------------------------------------
def test_damage_delta_dealt_and_taken():
    vr = DamageDeltaVR(self_port=1, lam_give=0.01, lam_take=0.01)
    frames = [
        gs(ps(1, percent=0.0), ps(2, percent=0.0)),
        gs(ps(1, percent=0.0), ps(2, percent=20.0)),   # dealt 20
        gs(ps(1, percent=15.0), ps(2, percent=20.0)),  # took 15
    ]
    r = run_vr(vr, frames)
    assert abs(r[1] - 0.20) < 1e-6
    assert abs(r[2] - (-0.15)) < 1e-6


def test_damage_delta_respawn_reset_ignored():
    vr = DamageDeltaVR(self_port=1, lam_give=0.01, lam_take=0.01)
    frames = [
        gs(ps(1), ps(2, percent=100.0)),
        gs(ps(1), ps(2, percent=0.0)),    # respawn — negative delta, not healing
    ]
    r = run_vr(vr, frames)
    assert r[1] == 0.0


# --- neutral_win_loss ----------------------------------------------------
def test_neutral_win():
    frames = [gs(ps(1), ps(2)) for _ in range(35)]          # both clean
    frames.append(gs(ps(1), ps(2, action=DAMAGE, percent=10.0)))  # opp hit
    r = run_vr(NeutralWinLossVR(self_port=1), frames)
    assert sum(r) == 1.0


def test_neutral_loss():
    frames = [gs(ps(1), ps(2)) for _ in range(35)]
    frames.append(gs(ps(1, action=DAMAGE, percent=10.0), ps(2)))  # bot hit
    r = run_vr(NeutralWinLossVR(self_port=1), frames)
    assert sum(r) == -1.0


def test_neutral_trade_nets_zero():
    """Both players enter punish on the same frame — a trade nets 0."""
    vr = NeutralWinLossVR(self_port=1)
    frames = [gs(ps(1), ps(2)) for _ in range(35)]
    frames.append(gs(ps(1, action=DAMAGE, percent=10.0),
                      ps(2, action=DAMAGE, percent=10.0)))
    r = run_vr(vr, frames)
    assert sum(r) == 0.0
    assert vr.metadata()["trades"] == 1


# --- combo_length --------------------------------------------------------
def test_combo_length_three_move_combo():
    vr = ComboLengthVR(self_port=1, cap=8, min_damage=5.0)
    frames = [(ps(1), ps(2))]
    frames.append((ps(1, action=100), ps(2, action=DAMAGE, percent=12.0)))
    frames.append((ps(1, action=200), ps(2, action=DAMAGE, percent=24.0)))
    frames.append((ps(1, action=300), ps(2, action=DAMAGE, percent=36.0)))
    for _ in range(50):
        frames.append((ps(1), ps(2, action=NEUTRAL, percent=36.0)))
    r = run_vr(vr, [gs(s, o) for s, o in frames])
    # 3 moves -> (3-2)/(8-2) = 1/6
    assert abs(sum(r) - (1.0 / 6.0)) < 1e-6


def test_combo_length_single_move_zero():
    vr = ComboLengthVR(self_port=1)
    frames = [(ps(1), ps(2))]
    frames.append((ps(1, action=100), ps(2, action=DAMAGE, percent=12.0)))
    for _ in range(50):
        frames.append((ps(1), ps(2, action=NEUTRAL, percent=12.0)))
    r = run_vr(vr, [gs(s, o) for s, o in frames])
    assert sum(r) == 0.0     # 1 move -> not a combo


# --- low_percent_kill ----------------------------------------------------
def test_low_percent_kill_bonus():
    # Fox (char 1) bucket is 105; a kill at ~60% qualifies.
    frames = [gs(ps(1), ps(2, stock=4))]
    for _ in range(4):
        frames.append(gs(ps(1), ps(2, stock=4, action=DAMAGE,
                                    percent=60.0, hitstun=12.0)))
    frames.append(gs(ps(1), ps(2, stock=3, percent=0.0)))   # opp died
    r = run_vr(LowPercentKillVR(self_port=1, bonus=0.5), frames)
    assert sum(r) == 0.5


def test_low_percent_kill_high_percent_no_bonus():
    frames = [gs(ps(1), ps(2, stock=4))]
    for _ in range(4):
        frames.append(gs(ps(1), ps(2, stock=4, action=DAMAGE,
                                    percent=200.0, hitstun=12.0)))
    frames.append(gs(ps(1), ps(2, stock=3, percent=0.0)))
    r = run_vr(LowPercentKillVR(self_port=1), frames)
    assert sum(r) == 0.0


def test_low_percent_kill_sd_gated_out():
    """A low-percent opponent death with no recent hitstun (a SD) gets
    no bonus."""
    frames = [gs(ps(1), ps(2, stock=4, percent=30.0)) for _ in range(6)]
    frames.append(gs(ps(1), ps(2, stock=3, percent=0.0)))
    r = run_vr(LowPercentKillVR(self_port=1), frames)
    assert sum(r) == 0.0


# --- tech ----------------------------------------------------------------
def test_tech_punished_vs_clean():
    punished = run_vr(TechVR(self_port=1, penalty=0.15), [
        gs(ps(1, action=TECH), ps(2)),
        gs(ps(1, action=DAMAGE), ps(2)),     # exited tech into a hit
    ])
    assert sum(punished) == -0.15
    clean = run_vr(TechVR(self_port=1, penalty=0.15), [
        gs(ps(1, action=TECH), ps(2)),
        gs(ps(1, action=NEUTRAL), ps(2)),
    ])
    assert sum(clean) == 0.0


# --- recovery ------------------------------------------------------------
def test_recovery_failed_vs_success():
    failed = run_vr(RecoveryVR(self_port=1, penalty=0.25), [
        gs(ps(1, off_stage=True, hitstun=10.0, on_ground=False), ps(2)),
        gs(ps(1, off_stage=True, on_ground=False, stock=3), ps(2)),  # died
    ])
    assert sum(failed) == -0.25
    success = run_vr(RecoveryVR(self_port=1, penalty=0.25), [
        gs(ps(1, off_stage=True, hitstun=10.0, on_ground=False), ps(2)),
        gs(ps(1, off_stage=False, on_ground=True), ps(2)),           # made it back
    ])
    assert sum(success) == 0.0


# --- CompositeVRTask -----------------------------------------------------
class _ConstVR(VRModule):
    """A trivial VR that returns a fixed per-frame value."""
    def __init__(self, vid, value):
        self.id = vid
        self.value = value

    def observe(self, state_history):
        return self.value


def test_composite_weighted_sum():
    task = CompositeVRTask(
        [_ConstVR("a", 1.0), _ConstVR("b", 2.0)],
        weights=[0.5, 1.0], self_port=1,
    )
    h = history([gs(ps(1), ps(2))])
    assert task.should_start(h) is True
    for _ in range(3):
        task.observe(h)
    outcome = task.compute_outcome(h, 0)
    # per-frame reward = 0.5*1 + 1.0*2 = 2.5
    assert outcome.per_frame_reward == [2.5, 2.5, 2.5]
    assert len(outcome.per_frame_reward) == 3
    assert task.should_end(h, 0) is False


def test_composite_real_vrs_smoke():
    """A composite of the real objective VRs runs without error over a
    short synthetic match."""
    task = CompositeVRTask(
        [StockDeltaVR(self_port=1), DamageDeltaVR(self_port=1)],
        weights=[1.0, 1.0], self_port=1,
    )
    h = history()
    h.append(gs(ps(1), ps(2)))
    task.should_start(h)
    for pct in (0.0, 10.0, 25.0):
        h.append(gs(ps(1), ps(2, percent=pct, action=DAMAGE, hitstun=8.0)))
        task.observe(h)
    outcome = task.compute_outcome(h, 0)
    assert len(outcome.per_frame_reward) == 3
    assert outcome.metadata["n_frames"] == 3
