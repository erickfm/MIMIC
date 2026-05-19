"""Tests for rlvr/online/slippi_stream.py — the streaming-slippistats
predicates and trackers. Ports the state-machine coverage of the retired
test_combo_extend_online_fixtures.py onto ComboTracker / MoveCounter.
"""
from __future__ import annotations

from rlvr.online import slippi_stream as ss
from rlvr.tests._melee_fixtures import (
    DAMAGE, DOWN, DYING, GRABBED, NEUTRAL, TECH, gs, history, ps,
)


# --- predicates ----------------------------------------------------------
def test_in_punish_state():
    assert not ss.in_punish_state(ps(2, action=NEUTRAL))
    assert ss.in_punish_state(ps(2, action=DAMAGE))
    assert ss.in_punish_state(ps(2, action=GRABBED))
    assert ss.in_punish_state(ps(2, action=DOWN))
    assert ss.in_punish_state(ps(2, action=TECH))
    assert ss.in_punish_state(ps(2, action=DYING))
    # hitstun / off-stage make even a neutral action count
    assert ss.in_punish_state(ps(2, action=NEUTRAL, hitstun=5.0))
    assert ss.in_punish_state(ps(2, action=NEUTRAL, off_stage=True))


def test_is_damaged_teching_dying():
    assert ss.is_damaged(ps(2, action=DAMAGE))
    assert not ss.is_damaged(ps(2, action=NEUTRAL))
    assert ss.is_teching(ps(2, action=TECH))
    assert ss.is_teching(ps(2, action=DOWN))      # DOWN counts as teching
    assert not ss.is_teching(ps(2, action=NEUTRAL))
    assert ss.is_dying(ps(2, action=DYING))
    assert not ss.is_dying(ps(2, action=DAMAGE))


def test_offstage_hitstun():
    assert ss.is_offstage(ps(1, off_stage=True))
    assert not ss.is_offstage(ps(1, off_stage=False))
    assert ss.in_hitstun(ps(1, hitstun=3.0))
    assert not ss.in_hitstun(ps(1, hitstun=0.0))


def test_recently_in_hitstun_or_damage():
    # opp (port 2) was in hitstun 3 frames ago, clean since
    h = history([
        gs(ps(1), ps(2, hitstun=8.0)),
        gs(ps(1), ps(2, action=DAMAGE)),
        gs(ps(1), ps(2, action=NEUTRAL)),
    ])
    assert ss.recently_in_hitstun_or_damage(h, port=2, window=5)
    # window too short to reach the hit
    assert not ss.recently_in_hitstun_or_damage(h, port=2, window=1)
    # a clean history -> False
    clean = history([gs(ps(1), ps(2)) for _ in range(5)])
    assert not ss.recently_in_hitstun_or_damage(clean, port=2, window=5)


# --- MoveCounter ---------------------------------------------------------
def test_move_counter_distinct_moves():
    mc = ss.MoveCounter()
    mc.seed(self_action=100, self_action_frame=1.0,
            opp_percent=10.0, opp_prev_percent=0.0)
    assert mc.n_moves == 1
    # same move, no new damage
    mc.update(self_action=100, self_action_frame=2.0, opp_percent=10.0)
    assert mc.n_moves == 1
    # new action + damage -> move 2
    mc.update(self_action=200, self_action_frame=1.0, opp_percent=18.0)
    assert mc.n_moves == 2
    # another new action + damage -> move 3
    mc.update(self_action=100, self_action_frame=1.0, opp_percent=26.0)
    assert mc.n_moves == 3


def test_move_counter_multihit_drill_is_one_move():
    """A multi-hit drill (same action, percent rising every frame) is
    ONE move, not many."""
    mc = ss.MoveCounter()
    mc.seed(self_action=300, self_action_frame=1.0,
            opp_percent=5.0, opp_prev_percent=0.0)
    mc.update(self_action=300, self_action_frame=2.0, opp_percent=9.0)
    mc.update(self_action=300, self_action_frame=3.0, opp_percent=13.0)
    mc.update(self_action=300, self_action_frame=4.0, opp_percent=17.0)
    assert mc.n_moves == 1


# --- ComboTracker --------------------------------------------------------
def _feed(tracker, frames):
    """Feed (self_ps, opp_ps) pairs; return the list of non-None results."""
    out = []
    for self_ps, opp_ps in frames:
        r = tracker.update(self_ps, opp_ps)
        if r is not None:
            out.append(r)
    return out


def test_combo_tracker_basic_combo():
    ct = ss.ComboTracker()
    frames = [(ps(1), ps(2))]                              # frame 0: prime prev
    frames.append((ps(1, action=100), ps(2, action=DAMAGE, percent=10.0)))  # hit 1
    frames.append((ps(1, action=200), ps(2, action=DAMAGE, percent=20.0)))  # hit 2
    # opponent escapes — out of punish for > COMBO_LENIENCY frames
    for _ in range(ss.COMBO_LENIENCY + 2):
        frames.append((ps(1), ps(2, action=NEUTRAL, percent=20.0)))
    results = _feed(ct, frames)
    assert len(results) == 1
    r = results[0]
    assert r.n_moves == 2
    assert r.damage == 20.0
    assert not r.did_kill


def test_combo_tracker_kill_terminates():
    ct = ss.ComboTracker()
    frames = [
        (ps(1), ps(2)),
        (ps(1, action=100), ps(2, action=DAMAGE, percent=10.0)),
        (ps(1, action=200), ps(2, action=DAMAGE, percent=120.0)),
        (ps(1), ps(2, action=DYING, percent=0.0, stock=3)),  # opp lost a stock
    ]
    results = _feed(ct, frames)
    assert len(results) == 1
    assert results[0].did_kill


# --- TechTracker ---------------------------------------------------------
def test_tech_tracker_clean_vs_punished():
    # clean tech: enter TECH, exit to NEUTRAL
    tt = ss.TechTracker()
    assert tt.update(ps(1, action=TECH)) is None
    r = tt.update(ps(1, action=NEUTRAL))
    assert r is not None and not r.was_punished
    # punished tech: enter TECH, exit into a DAMAGE state
    tt2 = ss.TechTracker()
    assert tt2.update(ps(1, action=TECH)) is None
    r2 = tt2.update(ps(1, action=DAMAGE))
    assert r2 is not None and r2.was_punished


# --- RecoveryTracker -----------------------------------------------------
def test_recovery_tracker_success_and_failure():
    # success: knocked off-stage, then back on the ground
    rt = ss.RecoveryTracker()
    assert rt.update(ps(1, off_stage=True, hitstun=10.0, on_ground=False)) is None
    r = rt.update(ps(1, off_stage=False, on_ground=True))
    assert r is not None and r.succeeded
    # failure: knocked off, then lost the stock while still off-stage
    rt2 = ss.RecoveryTracker()
    rt2.update(ps(1, off_stage=True, hitstun=10.0, on_ground=False))
    r2 = rt2.update(ps(1, off_stage=True, on_ground=False, stock=3))
    assert r2 is not None and not r2.succeeded


def test_recovery_tracker_voluntary_offstage_does_not_open():
    """Going off-stage WITHOUT being in hitstun (a voluntary trip) opens
    no recovery situation — this is the anti-farming gate."""
    rt = ss.RecoveryTracker()
    rt.update(ps(1, off_stage=True, hitstun=0.0, on_ground=False))
    # landing back gives no result because nothing opened
    assert rt.update(ps(1, off_stage=False, on_ground=True)) is None
