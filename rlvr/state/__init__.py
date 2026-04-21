"""Canonical game-state representation for RLVR.

Public API:
  GameState, PlayerState, ControllerInput — dataclasses.
  SCHEMA_VERSION — version string stamped into persisted artifacts.
  peppi_adapter.Replay, peppi_adapter.parse_replay — offline/tagger path.
  libmelee_adapter.parse_replay — compat / inference-time path.
"""
from rlvr.state.gamestate import (
    SCHEMA_VERSION,
    ControllerInput,
    GameState,
    PlayerState,
)

__all__ = [
    "SCHEMA_VERSION",
    "ControllerInput",
    "GameState",
    "PlayerState",
]
