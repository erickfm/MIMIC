"""Unit tests for the L-cancel task.

Verifier tests: hand-crafted (prompt, ControllerInput) fixtures covering
all trigger modalities and the analog-threshold boundary.

Tagger tests:
  1. A real HF replay with known peppi l_cancel counts — the number of
     emitted rows must exactly equal (successes + failures) × 7,
     factoring in the short-history clamp at the start of the replay.
  2. Synthetic short-history: a single l_cancel=1 at frame index 3 (in
     rollback-deduped space) produces < 7 rows and is clamped properly.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from rlvr.state.gamestate import ControllerInput, GameState
from rlvr.tasks.base import Prompt
from rlvr.tasks.l_cancel import (
    LCancelTask,
    LCancelVerifier,
    TASK_ID,
    WINDOW_FRAMES,
)


# --- Verifier fixtures ---------------------------------------------------

def _dummy_prompt() -> Prompt:
    return Prompt(
        task_id=TASK_ID,
        replay_id="test",
        player_port=1,
        frame_idx=100,
        state_context=(),
        task_metadata={"offset_to_landing": -3, "replay_l_cancel_result": 1},
    )


@pytest.mark.parametrize("kwargs", [
    dict(l_button=True),
    dict(r_button=True),
    dict(z_button=True),
    dict(shoulder_l=0.5),
    dict(shoulder_r=1.0),
])
def test_verifier_positive(kwargs):
    v = LCancelVerifier()
    ctrl = replace(ControllerInput.neutral(), **kwargs)
    assert v(_dummy_prompt(), ctrl) == 1.0


@pytest.mark.parametrize("kwargs", [
    {},                            # all neutral
    dict(shoulder_l=0.25),          # below threshold
    dict(a_button=True),            # A doesn't count
])
def test_verifier_negative(kwargs):
    v = LCancelVerifier()
    ctrl = replace(ControllerInput.neutral(), **kwargs)
    assert v(_dummy_prompt(), ctrl) == 0.0


def test_verifier_boundary_exactly_threshold():
    # Strict > — exactly the threshold value is NOT considered pressed.
    v = LCancelVerifier()
    ctrl = replace(ControllerInput.neutral(), shoulder_l=0.3)
    assert v(_dummy_prompt(), ctrl) == 0.0
    ctrl = replace(ControllerInput.neutral(), shoulder_l=0.30001)
    assert v(_dummy_prompt(), ctrl) == 1.0


# --- Tagger fixture on a real replay ------------------------------------

_TEST_CACHE_SLP = (
    "/tmp/rlvr_test_cache/slp/"
    "master-master-01b5757ec64263d645617655.slp"
)


@pytest.fixture(scope="module")
def fixture_replay():
    from pathlib import Path
    from rlvr.state.peppi_adapter import Replay
    p = Path(_TEST_CACHE_SLP)
    if not p.exists():
        pytest.skip(f"fixture replay missing: {p} (run roundtrip test first to populate)")
    return Replay(p)


def test_tagger_row_count_matches_lcancel_ground_truth(fixture_replay):
    """For each player with an L-cancel label, the tagger must emit
    exactly WINDOW_FRAMES rows per landing — clamped for too-close-to-start."""
    task = LCancelTask()
    rows = list(task.tag_frames(fixture_replay))
    assert len(rows) > 0

    # Build ground truth from the columnar data.
    expected = 0
    for pi in range(len(fixture_replay.player_characters)):
        lc = fixture_replay.l_cancel_per_player(pi)
        landings = np.where(lc != 0)[0]
        for lf in landings:
            if lf < WINDOW_FRAMES:
                continue
            expected += WINDOW_FRAMES
    assert len(rows) == expected

    # Every row has result ∈ {1, 2} and offset ∈ [-7, -1].
    for r in rows:
        md = r.task_metadata
        assert md["replay_l_cancel_result"] in (1, 2)
        assert -WINDOW_FRAMES <= md["offset_to_landing"] <= -1


def test_tagger_metadata_fields_present(fixture_replay):
    task = LCancelTask()
    rows = list(task.tag_frames(fixture_replay))
    # Spot-check the first row.
    r0 = rows[0]
    for k in ("landing_frame_idx", "offset_to_landing",
              "replay_l_cancel_result", "aerial_action_state"):
        assert k in r0.task_metadata
    assert r0.task_id == TASK_ID
    assert r0.character in fixture_replay.player_characters
    assert r0.stage == fixture_replay.stage


def test_tagger_emits_both_success_and_failure_prompts(fixture_replay):
    """The test replay has both types; the tagger must surface both."""
    task = LCancelTask()
    rows = list(task.tag_frames(fixture_replay))
    results = {r.task_metadata["replay_l_cancel_result"] for r in rows}
    assert results == {1, 2}


# --- Synthetic short-history edge case -----------------------------------

class _FakeReplay:
    """Minimal peppi-Replay-shaped stub for testing tag_frames edge cases
    without real .slp data."""
    def __init__(self, lc_list, state_list, character=1, port=1, stage=25):
        self.path = type("P", (), {"stem": "synthetic"})()
        self.player_characters = [character]
        self.player_ports = [port]
        self.stage = stage
        self.num_frames = len(lc_list)
        self.frame_ids = np.arange(self.num_frames, dtype=np.int32)
        self._post = [{
            "state": np.array(state_list, dtype=np.uint16),
            "l_cancel": np.array(lc_list, dtype=np.uint8),
        }]

    def l_cancel_per_player(self, pi):
        return self._post[pi]["l_cancel"]


def test_tagger_clamps_short_history():
    # Landing at frame 3 — only 3 frames of history, not 7.
    lc = [0] * 20
    lc[3] = 1  # success at frame 3
    states = [14] * 20  # all STANDING — irrelevant here
    r = _FakeReplay(lc, states)
    task = LCancelTask()
    rows = list(task.tag_frames(r))
    # Frame 3 < WINDOW_FRAMES, so the tagger should emit ZERO rows
    # (the window would go negative, which we skip entirely).
    assert len(rows) == 0


def test_tagger_emits_full_window_when_history_sufficient():
    lc = [0] * 20
    lc[10] = 1  # success at frame 10 — plenty of history
    states = [66] * 20  # FAIR throughout
    r = _FakeReplay(lc, states)
    task = LCancelTask()
    rows = list(task.tag_frames(r))
    assert len(rows) == WINDOW_FRAMES
    # Offsets must be -7..-1 exactly, in order
    offsets = [row.task_metadata["offset_to_landing"] for row in rows]
    assert offsets == list(range(-WINDOW_FRAMES, 0))
    # Frame indices are 3..9 (landing at 10, window [3, 9])
    assert [r.frame_idx for r in rows] == list(range(3, 10))


def test_tagger_handles_adjacent_landings():
    # Two landings close together — windows are independent, can overlap.
    lc = [0] * 30
    lc[20] = 1
    lc[25] = 2
    states = [66] * 30
    r = _FakeReplay(lc, states)
    task = LCancelTask()
    rows = list(task.tag_frames(r))
    assert len(rows) == 2 * WINDOW_FRAMES
    results = {row.task_metadata["replay_l_cancel_result"] for row in rows}
    assert results == {1, 2}
