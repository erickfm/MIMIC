"""Tests for the sim VR modules (rlvr/sim/vr/) on synthetic libmelee GameStates.

Includes the Dolphin-era test_vr_modules cases (git 2a7b786^). Three of those failed before the
2026-10-22 cleanup — test_stock_delta_kill_counts, test_stock_delta_self_loss_always_penalized,
test_low_percent_kill_bonus — because they dropped `stock` without the player ever entering a DEAD
action state. In Melee a stock is only lost after the DEAD animation (blast zone), and the modules
count deaths on that edge (it fires 1.5-3 s before the stock decrement), so the tests were wrong;
the ports below include the DEAD frames.
"""
from collections import deque

import melee
from melee.gamestate import GameState, PlayerState

import rlvr.sim.vr as V
from rlvr.sim.vr.flags import FlagPlayerState, _SLOTS
from rlvr.sim.vr.outcome import DamageDeltaVR, LowPercentKillVR, StockDeltaVR
from rlvr.sim.vr.situational import (
    ComboExtendVR, ComboLengthVR, LCancelVR, NeutralWinLossVR, RecoveryVR, ShieldEscapeVR, TechVR,
)
from rlvr.sim.vr.stream import ComboTracker, continues_combo, is_offstage

NEUTRAL, DAMAGE, GRABBED, TECH, DOWN, DYING = 14, 75, 223, 199, 183, 0
FALL, NAIR, NAIR_LANDING, SHIELD, ROLL_F, BREAK = 29, 65, 70, 179, 233, 205


def ps(*, action=NEUTRAL, percent=0.0, stock=4, hitstun=False, hitlag=False, action_frame=1,
       x=0.0, y=0.0, on_ground=True, shield=60.0, character=melee.Character.FOX):
    base = PlayerState()
    p = FlagPlayerState.__new__(FlagPlayerState)
    for s in _SLOTS:
        if hasattr(base, s):
            setattr(p, s, getattr(base, s))
    p.action = melee.Action(action)
    p.percent, p.stock, p.action_frame = percent, stock, action_frame
    p.hitstun_flag, p.hitlag_flag = hitstun, hitlag
    p.hitstun_frames_left = 10 if hitstun else 0
    p.position.x, p.position.y = x, y
    p.on_ground, p.shield_strength, p.character = on_ground, shield, character
    return p


def gs(me, opp, stage=melee.Stage.FINAL_DESTINATION):
    g = GameState()
    g.stage = stage
    g.players = {1: me, 2: opp}
    return g


def run(vr, frames):
    h, out = deque(maxlen=V.HISTORY), []
    vr.reset()
    for f in frames:
        h.append(f)
        out.append(vr.observe(h))
    return out


def test_registry_and_build():
    assert set(V.VRS) == {"l_cancel", "shield_escape", "combo_extend", "stock_delta", "damage_delta",
                          "low_percent_kill", "combo_length", "neutral_win_loss", "recovery", "tech"}
    mods = V.build("stock_delta:1,damage_delta:0.5,tech", self_port=2)
    assert [(m.id, w, m.self_port) for m, w in mods] == [
        ("stock_delta", 1.0, 2), ("damage_delta", 0.5, 2), ("tech", 1.0, 2)]


def test_flag_hook_on_real_replay():
    import glob
    path = sorted(glob.glob("data/raw_slp/fox_master_master/*.slp"))[0]
    c = melee.Console(is_dolphin=False, path=path, allow_old_version=True)
    c.connect()
    n = 0
    while n < 400:
        g = c.step()
        if g is None:
            break
        if g.menu_state in (melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH):
            for p in g.players.values():
                assert isinstance(p, FlagPlayerState) and p.hitstun_flag is not None
            n += 1
    assert n > 0


# --- stock_delta ---------------------------------------------------------------------------------
def test_stock_delta_kill_counts():
    frames = [gs(ps(), ps()),
              gs(ps(), ps(action=DAMAGE, hitstun=True)),
              gs(ps(), ps(action=DYING)),          # blast zone: the kill (recently hit)
              gs(ps(), ps(action=DYING, stock=3))]
    assert sum(run(StockDeltaVR(1), frames)) == 1.0


def test_stock_delta_opp_sd_filtered():
    frames = [gs(ps(), ps()) for _ in range(6)] + [gs(ps(), ps(action=DYING))]
    assert sum(run(StockDeltaVR(1), frames)) == 0.0


def test_stock_delta_self_loss_always_penalized():
    frames = [gs(ps(), ps()), gs(ps(action=DYING), ps()), gs(ps(action=DYING, stock=3), ps())]
    assert sum(run(StockDeltaVR(1), frames)) == -1.0


def test_stock_delta_star_ko_dead_frames_transparent():
    frames = [gs(ps(), ps()), gs(ps(), ps(action=DAMAGE, hitstun=True))]
    frames += [gs(ps(), ps(action=89)) for _ in range(150)]       # DAMAGE_FLY: tumbling, alive
    frames += [gs(ps(), ps(action=DYING)) for _ in range(150)] + [gs(ps(), ps(action=DYING, stock=3))]
    assert sum(run(StockDeltaVR(1), frames)) == 1.0


def test_stock_delta_late_controlled_fall_filtered():
    frames = [gs(ps(), ps()), gs(ps(), ps(action=DAMAGE, hitstun=True))]
    frames += [gs(ps(), ps(action=FALL, on_ground=False)) for _ in range(120)] + [gs(ps(), ps(action=DYING))]
    assert sum(run(StockDeltaVR(1), frames)) == 0.0


def test_stock_delta_finalize_reconciles_unseen_decrement():
    vr = StockDeltaVR(1)
    h = deque([gs(ps(), ps(stock=1)), gs(ps(), ps(stock=1, action=DAMAGE, hitstun=True))])
    vr.reset()
    for i in range(len(h)):
        vr.observe(deque(list(h)[:i + 1]))
    h.append(gs(ps(), ps(stock=0)))                 # never observed
    assert vr.finalize(h) == 1.0


# --- damage_delta --------------------------------------------------------------------------------
def test_damage_delta_dealt_and_taken():
    r = run(DamageDeltaVR(1, lam_give=0.01, lam_take=0.01),
            [gs(ps(), ps()), gs(ps(), ps(percent=20.0)), gs(ps(percent=15.0), ps(percent=20.0))])
    assert abs(r[1] - 0.20) < 1e-9 and abs(r[2] + 0.15) < 1e-9


def test_damage_delta_respawn_reset_ignored():
    r = run(DamageDeltaVR(1, lam_give=0.01, lam_take=0.01), [gs(ps(), ps(percent=100.0)), gs(ps(), ps())])
    assert r[1] == 0.0


# --- low_percent_kill ----------------------------------------------------------------------------
def _hit_at(pct, n=4):
    return [gs(ps(), ps(action=DAMAGE, percent=pct, hitstun=True)) for _ in range(n)]


def test_low_percent_kill_bonus():
    frames = [gs(ps(), ps())] + _hit_at(60.0) + [gs(ps(), ps(action=DYING)), gs(ps(), ps(action=DYING, stock=3))]
    assert sum(run(LowPercentKillVR(1, bonus=0.5), frames)) == 0.5


def test_low_percent_kill_high_percent_no_bonus():
    frames = [gs(ps(), ps())] + _hit_at(200.0) + [gs(ps(), ps(action=DYING))]
    assert sum(run(LowPercentKillVR(1), frames)) == 0.0


def test_low_percent_kill_sd_gated_out():
    frames = [gs(ps(), ps(percent=30.0)) for _ in range(6)] + [gs(ps(), ps(action=DYING))]
    assert sum(run(LowPercentKillVR(1), frames)) == 0.0


def test_low_percent_kill_star_ko_dead_frames_transparent():
    frames = [gs(ps(), ps())] + _hit_at(60.0)
    frames += [gs(ps(), ps(action=DYING)) for _ in range(150)] + [gs(ps(), ps(action=DYING, stock=3))]
    assert sum(run(LowPercentKillVR(1, bonus=0.5), frames)) == 0.5


def test_low_percent_kill_late_fall_gated_out():
    frames = [gs(ps(), ps())] + _hit_at(60.0)
    frames += [gs(ps(), ps(percent=60.0, action=FALL, on_ground=False)) for _ in range(120)]
    frames.append(gs(ps(), ps(action=DYING)))
    assert sum(run(LowPercentKillVR(1), frames)) == 0.0


# --- combos --------------------------------------------------------------------------------------
def _three_move_combo():
    f = [gs(ps(), ps())]
    f.append(gs(ps(action=44), ps(action=DAMAGE, percent=12.0, hitstun=True)))    # jab
    f.append(gs(ps(action=56), ps(action=DAMAGE, percent=24.0, hitstun=True)))    # different move
    f.append(gs(ps(action=60), ps(action=DAMAGE, percent=36.0, hitstun=True)))
    f += [gs(ps(), ps(percent=36.0)) for _ in range(50)]
    return f


def test_combo_length_three_move_combo():
    assert abs(sum(run(ComboLengthVR(1, cap=8, min_damage=5.0), _three_move_combo())) - 1 / 6) < 1e-9


def test_combo_length_single_move_zero():
    f = [gs(ps(), ps()), gs(ps(action=44), ps(action=DAMAGE, percent=12.0, hitstun=True))]
    f += [gs(ps(), ps(percent=12.0)) for _ in range(50)]
    assert sum(run(ComboLengthVR(1), f)) == 0.0


def test_combo_multihit_same_move_is_one_move():
    """A multi-hit move (same action, action_frame advancing) is one move, as in slippistats."""
    # the move starts a few frames before its first hit (a hit never lands on an action's first
    # frame; slippistats keys the move on the player's previous-frame state)
    f = [gs(ps(), ps()), gs(ps(action=69, action_frame=1), ps()), gs(ps(action=69, action_frame=2), ps())]
    for i, pct in enumerate((5.0, 10.0, 15.0, 20.0)):
        f.append(gs(ps(action=69, action_frame=3 + i), ps(action=DAMAGE, percent=pct, hitstun=True)))
    f += [gs(ps(), ps(percent=20.0)) for _ in range(50)]
    t, h, res = ComboTracker(1), deque(maxlen=V.HISTORY), []
    for g in f:
        h.append(g)
        r = t.update(h)
        if r:
            res.append(r)
    assert len(res) == 1 and res[0].n_moves == 1 and abs(res[0].damage - 20.0) < 1e-9


def test_combo_kill_damage_is_peak():
    f = [gs(ps(), ps())] + [gs(ps(action=44 + k), ps(action=DAMAGE, percent=30.0 * (k + 1), hitstun=True))
                            for k in range(3)]
    f += [gs(ps(), ps(action=DYING)) for _ in range(5)] + [gs(ps(), ps(action=DYING, stock=3))]
    t, h, res = ComboTracker(1), deque(maxlen=V.HISTORY), []
    for g in f:
        h.append(g)
        r = t.update(h)
        if r:
            res.append(r)
    assert res[0].did_kill and abs(res[0].damage - 90.0) < 1e-9


def test_combo_extend_scores_damage():
    assert abs(sum(run(ComboExtendVR(1), _three_move_combo())) - 36.0 / 80.0) < 1e-9


def test_offstage_and_continuation():
    st = "FINAL_DESTINATION"
    assert is_offstage(ps(x=100.0), st) and is_offstage(ps(y=-10.0), st) and not is_offstage(ps(x=50.0), st)
    h = deque([gs(ps(), ps(action=SHIELD))])
    assert continues_combo(h, 2)                     # mid-combo shield pressure keeps it alive


# --- neutral_win_loss ----------------------------------------------------------------------------
def test_neutral_win():
    f = [gs(ps(), ps()) for _ in range(35)] + [gs(ps(), ps(action=DAMAGE, percent=10.0, hitstun=True))]
    assert sum(run(NeutralWinLossVR(1), f)) == 1.0


def test_neutral_loss():
    f = [gs(ps(), ps()) for _ in range(35)] + [gs(ps(action=DAMAGE, percent=10.0, hitstun=True), ps())]
    assert sum(run(NeutralWinLossVR(1), f)) == -1.0


def test_neutral_trade_nets_zero():
    vr = NeutralWinLossVR(1)
    f = [gs(ps(), ps()) for _ in range(35)]
    f.append(gs(ps(action=DAMAGE, percent=10.0, hitstun=True), ps(action=DAMAGE, percent=10.0, hitstun=True)))
    assert sum(run(vr, f)) == 0.0 and vr.metadata()["trades"] == 1


def test_neutral_counter_hit_not_a_win():
    f = [gs(ps(), ps()) for _ in range(35)]
    f.append(gs(ps(action=DAMAGE, percent=10.0, hitstun=True), ps()))        # bot hit: a loss
    f += [gs(ps(), ps()) for _ in range(10)]                                   # bot recovers
    f.append(gs(ps(), ps(action=DAMAGE, percent=10.0, hitstun=True)))         # within K: counter-hit
    assert sum(run(NeutralWinLossVR(1), f)) == -1.0


# --- tech / recovery -----------------------------------------------------------------------------
def test_tech_punished_vs_clean():
    assert sum(run(TechVR(1, penalty=0.15), [gs(ps(), ps()), gs(ps(action=TECH), ps()),
                                             gs(ps(action=DAMAGE), ps())])) == -0.15
    assert sum(run(TechVR(1, penalty=0.15), [gs(ps(), ps()), gs(ps(action=TECH), ps()),
                                             gs(ps(), ps())])) == 0.0


def test_recovery_failed_vs_success():
    start = [gs(ps(x=50.0), ps()), gs(ps(x=100.0, on_ground=False, hitstun=True, action=89), ps())]
    failed = start + [gs(ps(x=120.0, on_ground=False, action=FALL), ps()), gs(ps(action=DYING), ps())]
    assert sum(run(RecoveryVR(1, penalty=0.25), failed)) == -0.25
    made_it = start + [gs(ps(x=95.0, on_ground=False, action=FALL), ps()), gs(ps(x=85.0, action=NEUTRAL), ps())]
    assert sum(run(RecoveryVR(1, penalty=0.25), made_it)) == 0.0


def test_recovery_voluntary_offstage_opens_nothing():
    f = [gs(ps(x=50.0), ps()), gs(ps(x=100.0, on_ground=False, action=FALL), ps()), gs(ps(action=DYING), ps())]
    vr = RecoveryVR(1)
    assert sum(run(vr, f)) == 0.0 and vr.metadata()["recoveries"] == 0


# --- l_cancel / shield_escape --------------------------------------------------------------------
def _aerial_landing(lag):
    f = [gs(ps(action=FALL, on_ground=False), ps())]
    f += [gs(ps(action=NAIR, on_ground=False), ps()) for _ in range(10)]
    f += [gs(ps(action=NAIR_LANDING), ps()) for _ in range(lag)] + [gs(ps(), ps())]
    return f


def test_l_cancel_success_and_miss():
    vr = LCancelVR(1)
    assert sum(run(vr, _aerial_landing(7))) == 1.0 and vr.metadata()["success"] == 1
    vr = LCancelVR(1, success=0.0, miss=-1.0)
    assert sum(run(vr, _aerial_landing(15))) == -1.0 and vr.metadata()["miss"] == 1


def test_l_cancel_landing_cut_short_dropped():
    f = _aerial_landing(3)[:-1] + [gs(ps(action=DAMAGE), ps())]
    vr = LCancelVR(1)
    assert sum(run(vr, f)) == 0.0 and vr.metadata()["dropped"] == 1


def _pressured_shield(exit_action):
    f = [gs(ps(action=SHIELD, shield=40.0), ps()) for _ in range(4)]
    f.append(gs(ps(action=181, shield=25.0), ps()))      # SHIELD_STUN: just absorbed 15
    f.append(gs(ps(action=SHIELD, shield=25.0), ps()))   # actionable, small, dropped >= 8
    f.append(gs(ps(action=exit_action, shield=25.0), ps()))
    return f


def test_shield_escape_outcomes():
    assert sum(run(ShieldEscapeVR(1), _pressured_shield(ROLL_F))) == 1.0
    assert sum(run(ShieldEscapeVR(1), _pressured_shield(BREAK))) == -1.0
    assert sum(run(ShieldEscapeVR(1), _pressured_shield(DAMAGE))) == 0.0
    vr = ShieldEscapeVR(1)
    run(vr, [gs(ps(action=SHIELD, shield=50.0), ps()) for _ in range(6)] + [gs(ps(action=ROLL_F), ps())])
    assert vr.metadata() == {"escape": 0, "hit": 0, "broken": 0}     # big shield: not pressured


def test_stock_delta_edgeguard_kill_is_earned():
    """Knocked off-stage, then a long fall/recovery with no further hit (past the 90-frame hit
    memory) before dying: the bot's combo is still open (off-stage continues it), so it's a kill,
    not a self-destruct. The Dolphin-era gate scored these as SDs (57 of 252 deaths on 40 replays)."""
    frames = [gs(ps(), ps(x=80.0)), gs(ps(), ps(action=DAMAGE, hitstun=True, x=95.0, on_ground=False))]
    frames += [gs(ps(), ps(action=FALL, x=110.0, y=-20.0, on_ground=False)) for _ in range(150)]
    frames.append(gs(ps(), ps(action=DYING, x=110.0, y=-110.0)))
    vr = StockDeltaVR(1)
    assert sum(run(vr, frames)) == 1.0 and vr.metadata()["filtered_opp_sds"] == 0
