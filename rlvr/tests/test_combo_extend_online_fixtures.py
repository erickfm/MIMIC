"""Unit tests for the combo-extend online task.

Two layers:
  1. State-machine fixtures — hand-built deques of synthetic GameStates
     drive should_start / should_end transitions. Validate the
     ascending-edge punish-entry detector, the juggle-gap tolerance,
     and the K-frame end condition.
  2. Outcome computation — synthetic histories with known percent
     deltas. Validate the damage-scaled reward formula, the
     sub-threshold zero-out, and the stock-taken kill-confirm reward.

Pattern mirrors test_l_cancel_fixtures.py — synthetic-only, no real
.slp dependency.
"""
from __future__ import annotations

from collections import deque
from dataclasses import replace

import pytest

from rlvr.state.gamestate import ControllerInput, GameState, PlayerState
from rlvr.online.tasks.combo_extend_online import (
    COMBO_END_GAP,
    ComboExtendOnlineTask,
    DAMAGE_END,
    DAMAGE_START,
    MAX_DAMAGE_REWARD,
    MIN_DAMAGE_REWARD,
    STOCK_TAKEN_REWARD,
    TASK_ID,
)


# --- Synthetic state factories ------------------------------------------

# Out-of-punish action: STANDING (libmelee Action.STANDING = 14).
NEUTRAL_ACTION = 14
# In-punish damage action: DAMAGE_HIGH_1 = 75 (= DAMAGE_START).
DAMAGE_ACTION = DAMAGE_START
# Action right at DAMAGE_END = DAMAGE_FLY_ROLL = 91.
DAMAGE_FLY_ROLL = DAMAGE_END
# A capture/grabbed action (CAPTURE_PULLED = 223 = CAPTURE_START).
GRABBED_ACTION = 223


def _ps(port: int, *, action: int = NEUTRAL_ACTION, percent: float = 0.0,
        stock: int = 4, hitstun: float = 0.0,
        action_frame: float = 1.0) -> PlayerState:
    """Make a minimal PlayerState with just the fields the task reads."""
    return PlayerState(
        character=1,  # Fox
        port=port,
        position_x=0.0, position_y=0.0,
        percent=percent,
        stock=stock,
        jumps_left=2,
        speed_air_x_self=0.0, speed_ground_x_self=0.0,
        speed_x_attack=0.0, speed_y_attack=0.0, speed_y_self=0.0,
        hitlag_left=0.0,
        hitstun_frames_left=hitstun,
        shield_strength=60.0,
        on_ground=True, off_stage=False, facing=True,
        invulnerable=False, moonwalkwarning=False,
        action=action,
        action_frame=action_frame,
        l_cancel=0,
        controller=ControllerInput.neutral(),
    )


def _gs(self_ps: PlayerState, opp_ps: PlayerState, frame: int = 0) -> GameState:
    """Bundle a (self, opp) into a GameState. self is port 1 (sorted first)."""
    return GameState(
        schema_version="v0.1",
        frame_idx=frame,
        stage=32,  # FINAL_DESTINATION
        players=(self_ps, opp_ps),
    )


def _history(frames):
    """Build a deque from a list of GameStates."""
    return deque(frames, maxlen=256)


def _self_neutral(percent: float = 0.0, stock: int = 4) -> PlayerState:
    return _ps(port=1, action=NEUTRAL_ACTION, percent=percent, stock=stock)


def _self_attack(action: int, percent: float = 0.0, stock: int = 4,
                  action_frame: float = 1.0) -> PlayerState:
    """Self in an attack animation. Tests use distinct `action` values
    to simulate the bot performing different moves between hits, which
    drives the move counter in the task."""
    return _ps(port=1, action=action, percent=percent, stock=stock,
               action_frame=action_frame)


def _opp_neutral(percent: float = 0.0, stock: int = 4) -> PlayerState:
    return _ps(port=2, action=NEUTRAL_ACTION, percent=percent, stock=stock)


def _opp_damaged(percent: float, stock: int = 4, hitstun: float = 10.0) -> PlayerState:
    return _ps(port=2, action=DAMAGE_ACTION, percent=percent, stock=stock,
               hitstun=hitstun)


def _opp_grabbed(percent: float, stock: int = 4) -> PlayerState:
    return _ps(port=2, action=GRABBED_ACTION, percent=percent, stock=stock)


# --- should_start tests --------------------------------------------------

def test_task_id_matches():
    """Sanity: the module-level constant matches the task instance id."""
    assert ComboExtendOnlineTask().id == TASK_ID


def test_should_start_fires_on_damage_entry():
    """Ascending edge: prev=neutral, curr=damaged → fire."""
    task = ComboExtendOnlineTask(self_port=1)
    history = _history([
        _gs(_self_neutral(), _opp_neutral(percent=20)),
        _gs(_self_neutral(), _opp_damaged(percent=25)),
    ])
    assert task.should_start(history) is True


def test_should_start_fires_on_grab_entry():
    """Grabbed counts as punish state — entering it should fire."""
    task = ComboExtendOnlineTask(self_port=1)
    history = _history([
        _gs(_self_neutral(), _opp_neutral(percent=30)),
        _gs(_self_neutral(), _opp_grabbed(percent=30)),
    ])
    assert task.should_start(history) is True


def test_should_start_fires_on_hitstun_with_damage():
    """Hitstun + percent increase = legitimate hit, fires."""
    task = ComboExtendOnlineTask(self_port=1)
    history = _history([
        _gs(_self_neutral(), _opp_neutral(percent=0, stock=4)),
        # action=NEUTRAL but hitstun > 0 AND percent went up = real hit.
        _gs(_self_neutral(),
            _ps(port=2, action=NEUTRAL_ACTION, percent=8, hitstun=5.0)),
    ])
    assert task.should_start(history) is True


def test_should_not_start_on_hitstun_without_damage():
    """Damage-or-grab safeguard: even if opp enters a punish state
    (e.g. hitstun bitflag set spuriously), if NO damage was dealt and
    opp wasn't grabbed, don't open an episode. This filters shield
    hits and any library-level hitstun-flag glitches."""
    task = ComboExtendOnlineTask(self_port=1)
    history = _history([
        _gs(_self_neutral(), _opp_neutral(percent=0, stock=4)),
        _gs(_self_neutral(),
            _ps(port=2, action=NEUTRAL_ACTION, percent=0, hitstun=5.0)),
    ])
    assert task.should_start(history) is False


def test_should_not_start_on_sustained_neutral():
    """Both frames out of punish — no fire."""
    task = ComboExtendOnlineTask(self_port=1)
    history = _history([
        _gs(_self_neutral(), _opp_neutral()),
        _gs(_self_neutral(), _opp_neutral()),
    ])
    assert task.should_start(history) is False


def test_should_not_start_on_sustained_punish():
    """Both frames already in punish state (mid-combo, not the edge)."""
    task = ComboExtendOnlineTask(self_port=1)
    history = _history([
        _gs(_self_neutral(), _opp_damaged(percent=20)),
        _gs(_self_neutral(), _opp_damaged(percent=25)),
    ])
    assert task.should_start(history) is False


def test_should_not_start_short_history():
    """Need at least 2 frames to detect ascending edge."""
    task = ComboExtendOnlineTask(self_port=1)
    history = _history([_gs(_self_neutral(), _opp_damaged(percent=20))])
    assert task.should_start(history) is False


# --- should_end tests ----------------------------------------------------

def test_should_not_end_during_active_combo():
    """Within COMBO_END_GAP frames of punish state → don't end."""
    task = ComboExtendOnlineTask(self_port=1)
    # Build: 5 frames of punish state. Episode started at frame 0.
    frames = [
        _gs(_self_neutral(), _opp_damaged(percent=20 + i))
        for i in range(5)
    ]
    history = _history(frames)
    assert task.should_end(history, episode_start_idx=0) is False


def test_should_end_after_K_frames_out_of_punish():
    """COMBO_END_GAP frames of sustained non-punish closes the episode.

    Tests must drive should_start first (matches the actor's lifecycle:
    open episode, then call should_end every frame).
    """
    task = ComboExtendOnlineTask(self_port=1)
    history = _history([
        _gs(_self_neutral(), _opp_neutral(percent=20)),
        _gs(_self_neutral(), _opp_damaged(percent=25)),
    ])
    assert task.should_start(history) is True

    # Now add COMBO_END_GAP+2 neutral frames, calling should_end each
    # frame. Final call (once frame counter reaches K) should fire.
    last_result = False
    for _ in range(COMBO_END_GAP + 2):
        history.append(_gs(_self_neutral(), _opp_neutral(percent=25)))
        last_result = task.should_end(history, episode_start_idx=1)
    assert last_result is True


def test_should_end_tolerates_juggle_gap():
    """A 5-frame neutral gap mid-combo should NOT close (juggle)."""
    task = ComboExtendOnlineTask(self_port=1)
    # Damage → 5 frames neutral → damage again. After this we're still
    # in active combo from the recent damage; should_end should not fire.
    frames = [_gs(_self_neutral(), _opp_damaged(percent=20))]
    for _ in range(5):
        frames.append(_gs(_self_neutral(), _opp_neutral(percent=22)))
    frames.append(_gs(_self_neutral(), _opp_damaged(percent=30)))
    history = _history(frames)
    assert task.should_end(history, episode_start_idx=0) is False


def test_should_end_on_stock_loss():
    """Stock decrement → end immediately (kill confirm)."""
    task = ComboExtendOnlineTask(self_port=1)
    # Drive should_start to snapshot opp's pre-death stock.
    history = _history([
        _gs(_self_neutral(), _opp_neutral(percent=70)),
        _gs(_self_neutral(), _opp_damaged(percent=80, stock=4)),
    ])
    assert task.should_start(history) is True

    # Stock decrement: should_end fires immediately regardless of
    # frame counter (kill-confirm path bypasses the K-frame gap).
    history.append(_gs(_self_neutral(), _opp_neutral(percent=0, stock=3)))
    assert task.should_end(history, episode_start_idx=1) is True


# --- compute_outcome tests -----------------------------------------------

def test_outcome_damage_combo():
    """Multi-MOVE combo dealing 30% damage → reward = 30/80 ≈ 0.375.
    Uses distinct self.action between hits so the move counter
    advances correctly (slippistats-style)."""
    task = ComboExtendOnlineTask(self_port=1)
    # Pre-hit neutral, then enter punish at 20%.
    history = _history([_gs(_self_neutral(), _opp_neutral(percent=20))])
    # Hit 1: bot uses action 100 (e.g., upsmash); opp goes 20 → 25.
    history.append(_gs(_self_attack(100), _opp_damaged(percent=25)))
    assert task.should_start(history) is True
    episode_start_idx = len(history) - 1
    # Hit 2: bot switches to action 101 (e.g., upair); opp at 35%.
    history.append(_gs(_self_attack(101), _opp_damaged(percent=35)))
    task.should_end(history, episode_start_idx)
    # Hit 3: action 102; opp at 50%.
    history.append(_gs(_self_attack(102), _opp_damaged(percent=50)))
    task.should_end(history, episode_start_idx)
    out = task.compute_outcome(history, episode_start_idx)
    assert out.terminal_reward == pytest.approx(30.0 / MAX_DAMAGE_REWARD)
    assert out.metadata["result"] == "combo"
    assert out.metadata["damage"] == pytest.approx(30.0)
    assert out.metadata["n_moves"] >= 2


def test_outcome_single_hit_zero():
    """A single-MOVE punish (30% upsmash, no followup) — not a combo
    by our definition. n_moves == 1 → 0 reward, classified single_hit.
    This is the load-bearing semantic: 'extension' requires ≥2 moves."""
    task = ComboExtendOnlineTask(self_port=1)
    history = _history([_gs(_self_neutral(), _opp_neutral(percent=20))])
    history.append(_gs(_self_attack(100), _opp_damaged(percent=50)))
    assert task.should_start(history) is True
    episode_start_idx = len(history) - 1
    # Opp recovers, no second hit (bot stays in same action).
    for _ in range(COMBO_END_GAP + 2):
        history.append(_gs(_self_attack(100), _opp_neutral(percent=50)))
        task.should_end(history, episode_start_idx)
    out = task.compute_outcome(history, episode_start_idx)
    assert out.terminal_reward == 0.0
    assert out.metadata["result"] == "single_hit"
    assert out.metadata["n_moves"] == 1


def test_outcome_multihit_same_move_is_single_hit():
    """Slippistats-style: a single multi-hit move (e.g., Fox dair
    drill) where the bot stays in the SAME action_state across many
    percent increases counts as ONE move → single_hit / 0 reward.
    This is the dair-spam reward-hack guard."""
    task = ComboExtendOnlineTask(self_port=1)
    DAIR_ACTION = 67  # any non-DAMAGE action, treated as 'one move'
    history = _history([_gs(_self_neutral(), _opp_neutral(percent=0))])
    # Drill: bot stays in same action while opp percent ticks up.
    history.append(_gs(_self_attack(DAIR_ACTION, action_frame=5),
                       _opp_damaged(percent=2)))
    assert task.should_start(history) is True
    episode_start_idx = len(history) - 1
    for i in range(2, 8):  # 6 more drill hits
        history.append(_gs(_self_attack(DAIR_ACTION, action_frame=5 + i),
                           _opp_damaged(percent=2 * i)))
        task.should_end(history, episode_start_idx)
    out = task.compute_outcome(history, episode_start_idx)
    # Despite 7 percent increases, the bot was in ONE action state →
    # one move → single_hit, 0 reward.
    assert out.terminal_reward == 0.0
    assert out.metadata["result"] == "single_hit"
    assert out.metadata["n_moves"] == 1


def test_outcome_sub_threshold_zero():
    """Multi-move but total damage < MIN_DAMAGE_REWARD → sub_threshold."""
    task = ComboExtendOnlineTask(self_port=1)
    history = _history([_gs(_self_neutral(), _opp_neutral(percent=20))])
    history.append(_gs(_self_attack(100), _opp_damaged(percent=21)))
    assert task.should_start(history) is True
    episode_start_idx = len(history) - 1
    history.append(_gs(_self_attack(101), _opp_damaged(percent=22)))
    task.should_end(history, episode_start_idx)
    history.append(_gs(_self_attack(102), _opp_damaged(percent=23)))
    task.should_end(history, episode_start_idx)
    out = task.compute_outcome(history, episode_start_idx)
    assert out.terminal_reward == 0.0
    assert out.metadata["result"] == "sub_threshold"


def test_outcome_combo_kill_uses_peak_percent():
    """A combo that ends in a stock loss is scored on peak damage
    delivered, not on a flat stock-taken bonus. The peak is tracked
    by should_end; we drive it here via the state machine."""
    task = ComboExtendOnlineTask(self_port=1)
    history = _history([_gs(_self_neutral(), _opp_neutral(percent=80, stock=4))])
    history.append(_gs(_self_attack(100), _opp_damaged(percent=90, stock=4)))
    assert task.should_start(history) is True
    episode_start_idx = len(history) - 1

    # Drive the combo up to 130% with distinct moves.
    for i, pct in enumerate((100, 110, 120, 130)):
        history.append(_gs(_self_attack(101 + i), _opp_damaged(percent=pct, stock=4)))
        assert task.should_end(history, episode_start_idx) is False

    # Opp dies, stock decrements → episode ends immediately.
    history.append(_gs(_self_attack(105), _opp_neutral(percent=0, stock=3)))
    assert task.should_end(history, episode_start_idx) is True

    out = task.compute_outcome(history, episode_start_idx)
    # Peak was 130, start was 80 → damage = 50 → reward = 50/80
    assert out.terminal_reward == pytest.approx(50.0 / MAX_DAMAGE_REWARD)
    assert out.metadata["result"] == "combo_kill"
    assert out.metadata["stock_was_taken"] is True


def test_outcome_pure_sd_no_episode_opens():
    """Opp self-destructs without us hitting them. With the
    damage-or-grab safeguard, no episode opens in the first place:
    opp entering DAMAGE state with zero percent change isn't a
    legitimate trigger."""
    task = ComboExtendOnlineTask(self_port=1)
    history = _history([_gs(_self_neutral(), _opp_neutral(percent=0, stock=4))])
    history.append(_gs(_self_neutral(), _opp_damaged(percent=0, stock=4)))
    # Crucially: should_start returns False because percent didn't
    # increase and opp wasn't grabbed.
    assert task.should_start(history) is False


def test_outcome_clipped_at_max():
    """120% damage from a multi-move combo clips to MAX/MAX = 1.0."""
    task = ComboExtendOnlineTask(self_port=1)
    history = _history([_gs(_self_neutral(), _opp_neutral(percent=20))])
    history.append(_gs(_self_attack(100), _opp_damaged(percent=70)))
    assert task.should_start(history) is True
    episode_start_idx = len(history) - 1
    history.append(_gs(_self_attack(101), _opp_damaged(percent=110)))
    task.should_end(history, episode_start_idx)
    history.append(_gs(_self_attack(102), _opp_damaged(percent=140)))
    task.should_end(history, episode_start_idx)
    out = task.compute_outcome(history, episode_start_idx)
    assert out.terminal_reward == pytest.approx(1.0)


def test_outcome_threshold_boundary():
    """Exactly MIN_DAMAGE_REWARD damage across 2+ moves rewards
    damage/MAX (boundary is inclusive)."""
    task = ComboExtendOnlineTask(self_port=1)
    history = _history([_gs(_self_neutral(), _opp_neutral(percent=20))])
    history.append(_gs(_self_attack(100), _opp_damaged(percent=22)))
    assert task.should_start(history) is True
    episode_start_idx = len(history) - 1
    history.append(_gs(_self_attack(101), _opp_damaged(percent=20 + MIN_DAMAGE_REWARD)))
    task.should_end(history, episode_start_idx)
    out = task.compute_outcome(history, episode_start_idx)
    assert out.terminal_reward == pytest.approx(MIN_DAMAGE_REWARD / MAX_DAMAGE_REWARD)


def test_outcome_aborted_if_start_out_of_history():
    """If episode_start_idx is out of bounds (deque ejected), abort safely."""
    task = ComboExtendOnlineTask(self_port=1)
    history = _history([_gs(_self_neutral(), _opp_neutral())])
    out = task.compute_outcome(history, episode_start_idx=999)
    assert out.terminal_reward == 0.0
    assert out.metadata["result"] == "aborted"


# --- Integration: full state-machine walk through a synthetic combo ----

def test_full_combo_lifecycle():
    """End-to-end: neutral → enter damage → sustained punish → recovery
    → end. Validates should_start, should_end timing, compute_outcome."""
    task = ComboExtendOnlineTask(self_port=1)
    history = _history([_gs(_self_neutral(), _opp_neutral(percent=0))])
    # Step 1: enter damage (should_start fires).
    history.append(_gs(_self_attack(100), _opp_damaged(percent=10)))
    assert task.should_start(history) is True
    episode_start_idx = len(history) - 1

    # Step 2-10: sustained punish, percent climbs, distinct moves.
    for i in range(2, 11):
        history.append(_gs(_self_attack(100 + i), _opp_damaged(percent=10 + i * 3)))
        assert task.should_end(history, episode_start_idx) is False

    # Step 11-onward: opp recovers (bot stays in last action; no new
    # damage so no new moves register).
    final_pct = 10 + 10 * 3
    ended = False
    for i in range(11, 11 + COMBO_END_GAP + 2):
        history.append(_gs(_self_attack(110), _opp_neutral(percent=final_pct)))
        if task.should_end(history, episode_start_idx):
            ended = True
            break
    assert ended is True

    # Outcome: start_percent is taken from the *prev* frame (pre-
    # punish) so the initial hit's damage counts. prev.percent = 0,
    # end.percent = 40. Damage = 40. Reward = 40/80 = 0.5.
    out = task.compute_outcome(history, episode_start_idx)
    assert out.terminal_reward == pytest.approx(40.0 / MAX_DAMAGE_REWARD)
    assert out.metadata["result"] == "combo"
