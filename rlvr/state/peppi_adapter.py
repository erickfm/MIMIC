"""peppi_py -> GameState adapter.

peppi exposes replays as Apache Arrow columnar arrays. We keep the hot
path columnar for the tagger (see rlvr/tagger) and materialize GameState
dataclasses lazily only where needed (the sampler's prompt-context
window, which is T=180 frames).

Empirically verified facts about peppi's schema (see rlvr/tests):
  - peppi returns **external** Melee character IDs (CSS order) and
    **external** stage IDs, while libmelee's `Character`/`Stage` enums
    use a different internal mapping. We remap here so GameState
    exposes libmelee-compatible values — required because MIMIC was
    trained on libmelee-indexed embeddings. See _PEPPI_TO_LIBMELEE_CHAR
    / _STAGE below.
  - Action state IDs **do** match between peppi and libmelee
    (both libraries expose the raw SSBM Action State ID from the .slp
    event stream). No remap needed.
  - `post.l_cancel` is uint8: 0 = n/a, 1 = success, 2 = failure.
  - `pre.buttons_physical` is uint16 with GC-controller bit layout:
        bit 4 = Z,  bit 5 = R,  bit 6 = L,  bit 8 = A,
        bit 9 = B,  bit 10 = X, bit 11 = Y, bit 12 = Start.
        bits 0-3 = DPad L/R/D/U.
  - `pre.triggers_physical.l/.r` are raw analog [0, 1] values.
  - `pre.joystick.x/y` and `pre.cstick.x/y` are in [-1, +1].
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np
import peppi_py

from rlvr.state.gamestate import (
    SCHEMA_VERSION,
    ControllerInput,
    GameState,
    PlayerState,
)


# Peppi uses external (CSS-order) character IDs; libmelee's Character
# enum uses a different mapping. MIMIC was trained on libmelee-indexed
# embeddings, so we remap here. Observed values verified empirically on
# 50 master-master replays (see rlvr/tests); the rest are filled from
# the published external-character-ID table cross-referenced against
# melee.Character.
_PEPPI_TO_LIBMELEE_CHAR = {
    0: 2,    # CPTFALCON
    1: 3,    # DK
    2: 1,    # FOX
    3: 24,   # GAMEANDWATCH
    4: 4,    # KIRBY
    5: 5,    # BOWSER
    6: 6,    # LINK
    7: 17,   # LUIGI
    8: 0,    # MARIO
    9: 18,   # MARTH
    10: 16,  # MEWTWO
    11: 8,   # NESS
    12: 9,   # PEACH
    13: 12,  # PIKACHU
    14: 10,  # POPO (Ice Climbers — peppi exposes only the lead climber)
    15: 15,  # JIGGLYPUFF
    16: 13,  # SAMUS
    17: 14,  # YOSHI
    18: 19,  # ZELDA
    19: 7,   # SHEIK
    20: 22,  # FALCO
    21: 20,  # YLINK
    22: 21,  # DOC
    23: 26,  # ROY
    24: 23,  # PICHU
    25: 25,  # GANONDORF
}

# Peppi external stage IDs -> libmelee Stage enum values. Six tournament-
# legal stages plus NO_STAGE (0 maps to 0).
_PEPPI_TO_LIBMELEE_STAGE = {
    0: 0,     # NO_STAGE
    2: 8,     # FOUNTAIN_OF_DREAMS
    3: 18,    # POKEMON_STADIUM
    8: 6,     # YOSHIS_STORY
    28: 26,   # DREAMLAND
    31: 24,   # BATTLEFIELD
    32: 25,   # FINAL_DESTINATION
}

# GC-controller physical button bit indices, verified empirically on a
# master-master Fox replay (bit distribution matches expected Fox usage:
# R and Y dominant, Z+A+B supporting).
_BIT_DPAD_LEFT = 1 << 0
_BIT_DPAD_RIGHT = 1 << 1
_BIT_DPAD_DOWN = 1 << 2
_BIT_DPAD_UP = 1 << 3
_BIT_Z = 1 << 4
_BIT_R = 1 << 5
_BIT_L = 1 << 6
_BIT_A = 1 << 8
_BIT_B = 1 << 9
_BIT_X = 1 << 10
_BIT_Y = 1 << 11
_BIT_START = 1 << 12


class Replay:
    """Wraps a parsed peppi replay as columnar numpy arrays + metadata.

    This is the fast path for the tagger. Construct once per file, run
    whole-replay queries on .l_cancel_per_player / etc., and only
    materialize GameState objects at specific frames via .gamestate_at().
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._game = peppi_py.read_slippi(str(self.path))

        # Player-level metadata, keyed by port (0..3). Port values are
        # libmelee-compatible (1-indexed) and characters are remapped
        # from peppi external to libmelee enum values.
        self.player_ports: List[int] = []
        self.player_characters: List[int] = []
        for p in self._game.start.players:
            if p is None:
                continue
            # peppi Port enum is 0-indexed (P1=0, P2=1, ...); libmelee
            # uses 1-indexed port numbers.
            self.player_ports.append(int(p.port.value) + 1)
            self.player_characters.append(
                _PEPPI_TO_LIBMELEE_CHAR.get(p.character, p.character)
            )

        self.stage: int = _PEPPI_TO_LIBMELEE_STAGE.get(
            self._game.start.stage, self._game.start.stage
        )
        # Peppi's frame arrays include Slippi rollback duplicates — for
        # frames where the live netcode predicted one state and then
        # rolled back, both rows appear in the raw .slp event stream.
        # libmelee's replay parser keeps only the FIRST occurrence per
        # frame_id; we match that convention here so our adapter agrees
        # with MIMIC's training-time parser.
        raw_ids = self._game.frames.id.to_numpy(zero_copy_only=False)
        _, first_idx = np.unique(raw_ids, return_index=True)
        first_idx.sort()  # np.unique returns sorted-by-value; restore positional order
        self._raw_index: np.ndarray = first_idx
        self.frame_ids: np.ndarray = raw_ids[first_idx]
        self.num_frames: int = len(self.frame_ids)

        # Extract columnar arrays per player, applying the rollback
        # dedup up front so every stored array is already canonicalized.
        self._pre: List[dict] = []
        self._post: List[dict] = []
        for i, p in enumerate(self._game.start.players):
            if p is None:
                continue
            leader = self._game.frames.ports[i].leader
            self._pre.append(self._extract_pre_columns(leader.pre, first_idx))
            self._post.append(self._extract_post_columns(leader.post, first_idx))

    @staticmethod
    def _extract_pre_columns(pre, index: np.ndarray) -> dict:
        return {
            "buttons_physical": pre.buttons_physical.to_numpy(zero_copy_only=False)[index],
            "joystick_x": pre.joystick.x.to_numpy(zero_copy_only=False)[index],
            "joystick_y": pre.joystick.y.to_numpy(zero_copy_only=False)[index],
            "cstick_x": pre.cstick.x.to_numpy(zero_copy_only=False)[index],
            "cstick_y": pre.cstick.y.to_numpy(zero_copy_only=False)[index],
            "triggers_l": pre.triggers_physical.l.to_numpy(zero_copy_only=False)[index],
            "triggers_r": pre.triggers_physical.r.to_numpy(zero_copy_only=False)[index],
        }

    @staticmethod
    def _extract_post_columns(post, index: np.ndarray) -> dict:
        # state_flags is a 5-tuple of UInt8Arrays (one per byte of the
        # action state flag bitfield).
        sf = post.state_flags
        vel = post.velocities
        def arr(a):
            return a.to_numpy(zero_copy_only=False)[index]
        return {
            "state": arr(post.state),
            "position_x": arr(post.position.x),
            "position_y": arr(post.position.y),
            "percent": arr(post.percent),
            "stocks": arr(post.stocks),
            "jumps": arr(post.jumps),
            "airborne": arr(post.airborne),
            "direction": arr(post.direction),
            "hitlag": arr(post.hitlag),
            "misc_as": arr(post.misc_as),
            "shield": arr(post.shield),
            "l_cancel": arr(post.l_cancel),
            "state_flag_1": arr(sf[0]),
            "state_flag_2": arr(sf[1]),
            "state_flag_3": arr(sf[2]),
            "state_flag_4": arr(sf[3]),
            "state_flag_5": arr(sf[4]),
            "vel_self_x_air": arr(vel.self_x_air),
            "vel_self_x_ground": arr(vel.self_x_ground),
            "vel_self_y": arr(vel.self_y),
            "vel_knockback_x": arr(vel.knockback_x),
            "vel_knockback_y": arr(vel.knockback_y),
        }

    # -- tagger-friendly helpers (columnar) ---------------------------------

    def l_cancel_per_player(self, player_idx: int) -> np.ndarray:
        """Ground-truth L-cancel label per frame for one player.

        Values: 0 = not eligible, 1 = succeeded, 2 = failed.
        """
        return self._post[player_idx]["l_cancel"]

    def find_player_idx(self, character_id: int) -> Optional[int]:
        """Return the first positional player index whose character matches,
        or None. For dittos, iterate by calling this in a loop with
        distinct ports."""
        for i, c in enumerate(self.player_characters):
            if c == character_id:
                return i
        return None

    def all_player_indices(self, character_id: int) -> List[int]:
        """All positional player indices whose character matches."""
        return [i for i, c in enumerate(self.player_characters) if c == character_id]

    # -- sampler-friendly helpers (materialized) ---------------------------

    def gamestate_at(self, frame_idx: int) -> GameState:
        """Materialize the full GameState for one absolute frame position.

        `frame_idx` is the positional index into the peppi frame arrays
        (0..num_frames-1), not the Slippi frame ID.
        """
        players = []
        for pi in range(len(self._pre)):
            players.append(self._materialize_player(pi, frame_idx))
        return GameState(
            schema_version=SCHEMA_VERSION,
            frame_idx=int(self.frame_ids[frame_idx]),
            stage=self.stage,
            players=tuple(sorted(players, key=lambda p: p.port)),
        )

    def gamestate_range(self, start: int, stop: int) -> List[GameState]:
        """Materialize frames [start, stop) as a list of GameStates."""
        return [self.gamestate_at(i) for i in range(start, stop)]

    def _materialize_player(self, pi: int, fi: int) -> PlayerState:
        pre = self._pre[pi]
        post = self._post[pi]
        ctrl = _decode_controller(pre, fi)

        # 5 flags
        airborne = bool(post["airborne"][fi])
        on_ground = not airborne

        # off_stage = airborne AND outside stage_edge. For v0.1 we use
        # ground != on_ground proxy; peppi's `ground` gives the surface
        # the character is touching if any. For the adapter's purpose
        # (feeding the MIMIC encoder), off_stage matches what libmelee
        # would set — proxy via airborne + x-position beyond typical
        # stage edge. Since every MIMIC stage has a different edge
        # geometry, we defer to a per-stage geometry table later; for
        # now, approximate: off_stage iff airborne and |pos_x| > 65
        # (Final Destination has edges at ±65.5). Consumers who care
        # precisely (edgeguard tasks, future work) should use
        # libmelee's off_stage field via the libmelee adapter.
        off_stage = airborne and abs(float(post["position_x"][fi])) > 65.0

        facing = float(post["direction"][fi]) > 0  # +1 right, -1 left

        # Flags from the 5-byte state_flags bitfield. Bit positions
        # verified against the Slippi SPEC + libmelee's mapping:
        # state_flag_2 bit 4 = invulnerable (post-respawn / after hit).
        # state_flag_5 bit 0 = moonwalk warning (dash backward in
        # dashdance window).
        invulnerable = bool(post["state_flag_2"][fi] & (1 << 4))
        moonwalk = bool(post["state_flag_5"][fi] & (1 << 0))

        speed_air_x_self = float(post["vel_self_x_air"][fi])
        speed_ground_x_self = float(post["vel_self_x_ground"][fi])
        speed_y_self = float(post["vel_self_y"][fi])
        speed_x_attack = float(post["vel_knockback_x"][fi])
        speed_y_attack = float(post["vel_knockback_y"][fi])

        # hitstun: peppi exposes this via misc_as when state is a damage
        # state. Approximate from misc_as for v0.1.
        hitstun = float(post["misc_as"][fi])

        return PlayerState(
            character=self.player_characters[pi],
            port=self.player_ports[pi],
            position_x=float(post["position_x"][fi]),
            position_y=float(post["position_y"][fi]),
            percent=float(post["percent"][fi]),
            stock=int(post["stocks"][fi]),
            jumps_left=int(post["jumps"][fi]),
            speed_air_x_self=speed_air_x_self,
            speed_ground_x_self=speed_ground_x_self,
            speed_x_attack=speed_x_attack,
            speed_y_attack=speed_y_attack,
            speed_y_self=speed_y_self,
            hitlag_left=float(post["hitlag"][fi]),
            hitstun_frames_left=hitstun,
            shield_strength=float(post["shield"][fi]),
            on_ground=on_ground,
            off_stage=off_stage,
            facing=facing,
            invulnerable=invulnerable,
            moonwalkwarning=moonwalk,
            action=int(post["state"][fi]),
            l_cancel=int(post["l_cancel"][fi]),
            controller=ctrl,
        )


def _decode_controller(pre: dict, fi: int) -> ControllerInput:
    b = int(pre["buttons_physical"][fi])
    return ControllerInput(
        main_x=float(pre["joystick_x"][fi]),
        main_y=float(pre["joystick_y"][fi]),
        c_x=float(pre["cstick_x"][fi]),
        c_y=float(pre["cstick_y"][fi]),
        shoulder_l=float(pre["triggers_l"][fi]),
        shoulder_r=float(pre["triggers_r"][fi]),
        a_button=bool(b & _BIT_A),
        b_button=bool(b & _BIT_B),
        x_button=bool(b & _BIT_X),
        y_button=bool(b & _BIT_Y),
        z_button=bool(b & _BIT_Z),
        l_button=bool(b & _BIT_L),
        r_button=bool(b & _BIT_R),
        start_button=bool(b & _BIT_START),
        d_up=bool(b & _BIT_DPAD_UP),
        d_down=bool(b & _BIT_DPAD_DOWN),
        d_left=bool(b & _BIT_DPAD_LEFT),
        d_right=bool(b & _BIT_DPAD_RIGHT),
    )


# Public convenience: eager full-replay materialization. Mostly for
# tests; real code uses Replay directly.
def parse_replay(path: Path | str) -> List[GameState]:
    r = Replay(Path(path))
    return r.gamestate_range(0, r.num_frames)
