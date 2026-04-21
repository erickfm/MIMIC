"""Canonical per-frame game state for RLVR.

The schema mirrors MIMIC's existing feature extraction: 13 numeric + 5 flag
columns per player (see /root/MIMIC/mimic/features.py:numeric_state and
/root/MIMIC/tools/inference_utils.py:_player_numeric_full). Fields are
named to match libmelee's `PlayerState` attributes so the MIMIC
build_frame helpers can consume these objects with light adaptation.

Parser-source-of-truth: peppi_py. See rlvr/state/peppi_adapter.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

SCHEMA_VERSION = "v0.1"


@dataclass(frozen=True, slots=True)
class ControllerInput:
    """One frame of controller input — what the player pressed.

    Axes and triggers are the physical analog values in the Slippi/peppi
    convention: joystick x/y in [-1, +1], triggers in [0, 1]. Buttons are
    booleans taken from peppi `pre.buttons_physical`.
    """
    main_x: float
    main_y: float
    c_x: float
    c_y: float
    shoulder_l: float
    shoulder_r: float
    a_button: bool
    b_button: bool
    x_button: bool
    y_button: bool
    z_button: bool
    l_button: bool
    r_button: bool
    start_button: bool
    d_up: bool
    d_down: bool
    d_left: bool
    d_right: bool

    @classmethod
    def neutral(cls) -> "ControllerInput":
        return cls(
            main_x=0.0, main_y=0.0, c_x=0.0, c_y=0.0,
            shoulder_l=0.0, shoulder_r=0.0,
            a_button=False, b_button=False, x_button=False, y_button=False,
            z_button=False, l_button=False, r_button=False, start_button=False,
            d_up=False, d_down=False, d_left=False, d_right=False,
        )


@dataclass(frozen=True, slots=True)
class PlayerState:
    """Per-player state at one frame. Mirrors MIMIC's 13-numeric + 5-flag
    shard schema plus a few fields verifiers/taggers need."""
    # Identity (libmelee enum integer values)
    character: int
    port: int

    # Numeric (13) — order matches mimic/features.py:numeric_state
    position_x: float
    position_y: float
    percent: float
    stock: int
    jumps_left: int
    speed_air_x_self: float
    speed_ground_x_self: float
    speed_x_attack: float
    speed_y_attack: float
    speed_y_self: float
    hitlag_left: float
    hitstun_frames_left: float
    shield_strength: float

    # Flags (5) — shard order
    on_ground: bool
    off_stage: bool
    facing: bool
    invulnerable: bool
    moonwalkwarning: bool

    # Action state (libmelee Action enum value)
    action: int

    # L-cancel label from the game engine.
    # 0 = not an L-cancel-eligible landing on this frame,
    # 1 = L-cancel succeeded, 2 = L-cancel failed.
    l_cancel: int

    # The controller this player PRESSED to produce this frame.
    controller: ControllerInput


@dataclass(frozen=True, slots=True)
class GameState:
    """One frame of a Melee game."""
    schema_version: str
    frame_idx: int          # Slippi frame ID (starts negative during countdown)
    stage: int              # libmelee Stage enum value
    players: Tuple[PlayerState, ...]  # sorted by port

    @property
    def num_players(self) -> int:
        return len(self.players)
