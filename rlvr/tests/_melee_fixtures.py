"""Synthetic GameState / PlayerState builders for the VR + slippi_stream
test suite. Salvaged from the retired test_combo_extend_online_fixtures.py.
"""
from __future__ import annotations

from collections import deque

from rlvr.state.gamestate import ControllerInput, GameState, PlayerState

# Action-state ints used across the tests.
NEUTRAL = 14      # Action.STANDING — out of punish, not teching
DAMAGE = 75       # DAMAGE_HIGH_1 (DAMAGE_START) — a hit reaction
GRABBED = 223     # CAPTURE_START — in a grab
TECH = 199        # TECH_START — tech in place
DOWN = 183        # DOWN_START — missed tech / downed
DYING = 0         # DYING_START — blast-zone death animation


def ps(port: int, *, action: int = NEUTRAL, percent: float = 0.0,
       stock: int = 4, hitstun: float = 0.0, hitlag: float = 0.0,
       action_frame: float = 1.0, character: int = 1,
       off_stage: bool = False, on_ground: bool = True) -> PlayerState:
    """A PlayerState with test-relevant fields set, the rest at defaults."""
    return PlayerState(
        character=character, port=port,
        position_x=0.0, position_y=0.0,
        percent=percent, stock=stock, jumps_left=2,
        speed_air_x_self=0.0, speed_ground_x_self=0.0, speed_x_attack=0.0,
        speed_y_attack=0.0, speed_y_self=0.0,
        hitlag_left=hitlag, hitstun_frames_left=hitstun,
        shield_strength=60.0,
        on_ground=on_ground, off_stage=off_stage, facing=True,
        invulnerable=False, moonwalkwarning=False,
        action=action, action_frame=action_frame, l_cancel=0,
        controller=ControllerInput.neutral(),
    )


def gs(self_ps: PlayerState, opp_ps: PlayerState, frame: int = 0) -> GameState:
    """Bundle (self on port 1, opp on port 2) into a GameState."""
    return GameState(schema_version="v0.1", frame_idx=frame, stage=32,
                     players=(self_ps, opp_ps))


def history(frames=()) -> deque:
    """A state-history deque (matches the actor's `_state_history`)."""
    return deque(frames, maxlen=256)
