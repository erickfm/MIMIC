"""libmelee -> GameState adapter.

Used for:
  (a) the roundtrip acceptance test that asserts peppi_adapter agrees
      with libmelee's own parse of the same .slp on the numeric fields
      the model consumes, and
  (b) forward compatibility: once this is wired into live inference, the
      same verifier code can score states coming out of live Dolphin.

Not used by the tagger or sampler hot path (peppi is much faster).
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from melee import Console, Menu

from rlvr.state.gamestate import (
    SCHEMA_VERSION,
    ControllerInput,
    GameState,
    PlayerState,
)


def _ps_from_libmelee(ps_lm, port: int) -> PlayerState:
    # libmelee's Controller state captured during replay parse mirrors
    # the pre-frame controller inputs — we decode it the same way the
    # peppi adapter does its ControllerInput, for an apples-to-apples
    # roundtrip comparison.
    cs = ps_lm.controller_state
    ctrl = ControllerInput(
        main_x=float(cs.main_stick[0]) * 2 - 1,  # libmelee uses [0,1]; scale to [-1,+1]
        main_y=float(cs.main_stick[1]) * 2 - 1,
        c_x=float(cs.c_stick[0]) * 2 - 1,
        c_y=float(cs.c_stick[1]) * 2 - 1,
        shoulder_l=float(cs.l_shoulder),
        shoulder_r=float(cs.r_shoulder),
        a_button=bool(cs.button.get("BUTTON_A", False)),
        b_button=bool(cs.button.get("BUTTON_B", False)),
        x_button=bool(cs.button.get("BUTTON_X", False)),
        y_button=bool(cs.button.get("BUTTON_Y", False)),
        z_button=bool(cs.button.get("BUTTON_Z", False)),
        l_button=bool(cs.button.get("BUTTON_L", False)),
        r_button=bool(cs.button.get("BUTTON_R", False)),
        start_button=bool(cs.button.get("BUTTON_START", False)),
        d_up=bool(cs.button.get("BUTTON_D_UP", False)),
        d_down=bool(cs.button.get("BUTTON_D_DOWN", False)),
        d_left=bool(cs.button.get("BUTTON_D_LEFT", False)),
        d_right=bool(cs.button.get("BUTTON_D_RIGHT", False)),
    )

    # libmelee exposes most fields directly on PlayerState.
    return PlayerState(
        character=int(ps_lm.character.value),
        port=port,
        position_x=float(ps_lm.position.x),
        position_y=float(ps_lm.position.y),
        percent=float(ps_lm.percent),
        stock=int(ps_lm.stock),
        jumps_left=int(ps_lm.jumps_left),
        speed_air_x_self=float(ps_lm.speed_air_x_self),
        speed_ground_x_self=float(ps_lm.speed_ground_x_self),
        speed_x_attack=float(ps_lm.speed_x_attack),
        speed_y_attack=float(ps_lm.speed_y_attack),
        speed_y_self=float(ps_lm.speed_y_self),
        hitlag_left=float(ps_lm.hitlag_left),
        hitstun_frames_left=float(ps_lm.hitstun_frames_left),
        shield_strength=float(ps_lm.shield_strength),
        on_ground=bool(ps_lm.on_ground),
        off_stage=bool(ps_lm.off_stage),
        facing=bool(ps_lm.facing),
        invulnerable=bool(ps_lm.invulnerable),
        moonwalkwarning=bool(ps_lm.moonwalkwarning),
        action=int(ps_lm.action.value),
        # libmelee does not expose post.l_cancel — the game-engine label
        # lives only in the raw .slp event stream. This adapter leaves
        # it at 0; peppi is the canonical source for l_cancel labels.
        l_cancel=0,
        controller=ctrl,
    )


def parse_replay(path: Path | str) -> List[GameState]:
    """Parse the entire replay via libmelee's Console.step() loop.

    Slower than peppi but emits bit-identical field semantics for the
    numerics MIMIC's encoder consumes — i.e. the exact values that
    tools/slp_to_shards.py writes into training shards.
    """
    console = Console(is_dolphin=False, path=str(path), allow_old_version=True)
    console.connect()

    states: List[GameState] = []
    while True:
        gs_lm = console.step()
        if gs_lm is None:
            break
        if gs_lm.menu_state != Menu.IN_GAME:
            continue
        players = sorted(gs_lm.players.items())
        player_objs = tuple(
            _ps_from_libmelee(ps_lm, port=int(port))
            for port, ps_lm in players
        )
        states.append(GameState(
            schema_version=SCHEMA_VERSION,
            frame_idx=int(gs_lm.frame),
            stage=int(gs_lm.stage.value),
            players=player_objs,
        ))
    return states
