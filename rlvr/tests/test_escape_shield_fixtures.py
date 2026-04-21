"""Unit tests for escape_pressured_shield.

Verifier fixtures cover every valid escape route (release, grab, jump,
OoS attack, spotdodge, roll) and the boundary at TRIG_PRESSED_THRESHOLD
/ STICK_ESCAPE_THRESHOLD.

Tagger fixture uses a synthetic replay with one clear pressured-shield
event and checks window counts.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from rlvr.state.gamestate import ControllerInput
from rlvr.tasks.base import Prompt
from rlvr.tasks.escape_pressured_shield import (
    EscapePressuredShieldTask,
    EscapePressuredShieldVerifier,
    SHIELD, SHIELD_STUN,
    SMALL_SHIELD_THRESHOLD,
    TASK_ID,
    WINDOW_FRAMES,
)


def _dummy_prompt() -> Prompt:
    return Prompt(
        task_id=TASK_ID, replay_id="test", player_port=1,
        frame_idx=200, state_context=(),
        task_metadata={"shield_at_decision": 20.0, "damage_delta": 12.0},
    )


def _held_shield() -> ControllerInput:
    """Neutral baseline: trigger held, stick neutral, no buttons."""
    return replace(
        ControllerInput.neutral(),
        shoulder_l=1.0,
    )


# ---- verifier positive cases ----

@pytest.mark.parametrize("kwargs", [
    dict(shoulder_l=0.0),                              # release trigger
    dict(shoulder_l=1.0, z_button=True),               # grab OoS
    dict(shoulder_l=1.0, x_button=True),               # jump OoS (X)
    dict(shoulder_l=1.0, y_button=True),               # jump OoS (Y)
    dict(shoulder_l=1.0, a_button=True),               # OoS attack
    dict(shoulder_l=1.0, main_y=-0.6),                 # spotdodge (down)
    dict(shoulder_l=1.0, main_x=0.6),                  # roll forward
    dict(shoulder_l=1.0, main_x=-0.9),                 # roll backward
    dict(shoulder_r=0.5, z_button=True),               # grab via R-shield
    dict(l_button=True, z_button=True),                # digital L + Z
])
def test_verifier_positive(kwargs):
    v = EscapePressuredShieldVerifier()
    ctrl = replace(_held_shield(), **kwargs)
    # Overwrite shoulder if not specified in kwargs
    if "shoulder_l" not in kwargs and "shoulder_r" not in kwargs and "l_button" not in kwargs:
        ctrl = replace(ctrl, shoulder_l=1.0)
    assert v(_dummy_prompt(), ctrl) == 1.0


@pytest.mark.parametrize("kwargs,expected", [
    (dict(), 0.0),                             # pure held shield, no escape
    (dict(main_x=0.3), 0.0),                   # stick slightly sideways — below 0.5 threshold
    (dict(main_y=-0.45), 0.0),                 # stick slightly down — below threshold
    (dict(shoulder_l=0.35), 0.0),              # barely above trigger threshold, still holding shield
    (dict(shoulder_r=0.31), 0.0),              # R just above trigger threshold
])
def test_verifier_negative(kwargs, expected):
    v = EscapePressuredShieldVerifier()
    ctrl = replace(_held_shield(), **kwargs)
    assert v(_dummy_prompt(), ctrl) == expected


def test_verifier_threshold_boundary_release():
    # shoulder_l = 0.3 exactly: trigger is NOT held (strict >) -> escape (released).
    v = EscapePressuredShieldVerifier()
    ctrl = replace(ControllerInput.neutral(), shoulder_l=0.3)
    assert v(_dummy_prompt(), ctrl) == 1.0
    # 0.3001 is held
    ctrl = replace(ControllerInput.neutral(), shoulder_l=0.3001)
    assert v(_dummy_prompt(), ctrl) == 0.0


def test_verifier_threshold_boundary_stick():
    v = EscapePressuredShieldVerifier()
    base = _held_shield()
    # main_x = 0.5 exactly — not strictly greater than threshold, so NOT escape.
    assert v(_dummy_prompt(), replace(base, main_x=0.5)) == 0.0
    assert v(_dummy_prompt(), replace(base, main_x=-0.5)) == 0.0
    # Just above boundary is escape.
    assert v(_dummy_prompt(), replace(base, main_x=0.51)) == 1.0
    # main_y = -0.5 exactly — not escape (strict <).
    assert v(_dummy_prompt(), replace(base, main_y=-0.5)) == 0.0
    assert v(_dummy_prompt(), replace(base, main_y=-0.51)) == 1.0


# ---- tagger fixtures ----

class _FakeReplay:
    def __init__(self, state, shield, character=1, port=1, stage=25):
        self.path = type("P", (), {"stem": "synthetic"})()
        self.player_characters = [character]
        self.player_ports = [port]
        self.stage = stage
        self.num_frames = len(state)
        self.frame_ids = np.arange(self.num_frames, dtype=np.int32)
        self._post = [{
            "state": np.array(state, dtype=np.uint16),
            "shield": np.array(shield, dtype=np.float32),
        }]


def test_tagger_emits_on_pressured_event():
    """Build a replay with: 200 frames of normal play, then:
      - SHIELD for 20 frames (shield 60 -> 55)
      - hit absorbed (state SHIELD_STUN for 6 frames, shield drops 55 -> 20)
      - SHIELD (actionable) for 10 frames (shield 20)
    Expected: WINDOW_FRAMES prompts emitted at the first actionable
    frame after the stun."""
    N = 250
    state = [0] * 200
    state.extend([SHIELD] * 20)
    state.extend([SHIELD_STUN] * 6)
    state.extend([SHIELD] * 10)
    state.extend([0] * (N - len(state)))

    shield = [60.0] * 200
    # 60 -> 55 while in SHIELD
    shield.extend([60.0 - 0.28 * k for k in range(20)])
    # Big drop during SHIELD_STUN: 55 -> 20 over 6 frames
    start = shield[-1]
    for k in range(6):
        shield.append(start - (start - 20.0) * (k + 1) / 6)
    # Flat 20 during actionable
    shield.extend([20.0] * 10)
    shield.extend([20.0] * (N - len(shield)))

    replay = _FakeReplay(state, shield)
    task = EscapePressuredShieldTask()
    rows = list(task.tag_frames(replay))

    assert len(rows) == WINDOW_FRAMES, f"expected {WINDOW_FRAMES} rows, got {len(rows)}"
    # All rows should have shield_at_decision < SMALL_SHIELD_THRESHOLD
    for r in rows:
        assert r.task_metadata["shield_at_decision"] < SMALL_SHIELD_THRESHOLD
        assert r.task_metadata["damage_delta"] >= 8.0
    # Offsets 0..WINDOW_FRAMES-1 in order
    offsets = [r.task_metadata["decision_offset"] for r in rows]
    assert offsets == list(range(WINDOW_FRAMES))


def test_tagger_skips_when_shield_not_small():
    """Damage event but shield still > SMALL_SHIELD_THRESHOLD after -> no rows."""
    N = 250
    state = [0] * 200
    state.extend([SHIELD] * 10)
    state.extend([SHIELD_STUN] * 5)
    state.extend([SHIELD] * 10)
    state.extend([0] * (N - len(state)))

    shield = [60.0] * 200
    shield.extend([60.0] * 10)
    shield.extend([60.0 - 8.5 * (k + 1) / 5 for k in range(5)])  # drop ~8.5 → shield ~51.5
    shield.extend([51.5] * 10)
    shield.extend([51.5] * (N - len(shield)))

    replay = _FakeReplay(state, shield)
    task = EscapePressuredShieldTask()
    rows = list(task.tag_frames(replay))
    assert len(rows) == 0, "should not emit when shield_at_decision >= threshold"


def test_tagger_skips_small_damage():
    """Shield small, but the delta is below the damage threshold → no rows."""
    N = 250
    state = [0] * 200
    state.extend([SHIELD] * 30)
    state.extend([0] * (N - len(state)))
    shield = [60.0] * 200
    # Gradual decay, never drops by 8 in 3 frames
    shield.extend([60.0 - 0.28 * k for k in range(30)])
    shield.extend([shield[-1]] * (N - len(shield)))
    replay = _FakeReplay(state, shield)
    task = EscapePressuredShieldTask()
    assert len(list(task.tag_frames(replay))) == 0
