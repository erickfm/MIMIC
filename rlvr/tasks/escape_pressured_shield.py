"""Escape pressured shield task.

Skill: when the player is holding shield and has absorbed a recent hit that
dropped shield_strength significantly, leaving the shield visibly small
(< 30 out of 60 max = 50% of full), the preferred action is to leave
shield — release, roll, spotdodge, grab OoS, up-smash OoS, or jump OoS —
rather than continue holding shield and risk break.

Motivation: master-tier Fox data shows 79% of SHIELD_STUN exits return
to SHIELD, and 70% of shield-damage events are followed by stay-in-
shield. The BC model inherited that over-shielding pattern, which is
exploitable — the bot gets shield-broken by pressure that a less-
shield-happy policy would escape. RLVR on a narrow "pressured shield"
subset shifts the policy toward more-frequent escape without
disturbing general shielding behavior.

Tagger condition:
  1. Frame where player is in a SHIELD-family action state.
  2. shield_strength dropped by >= 8 in the last (1..3) frames —
     indicates a big hit was just absorbed.
  3. First actionable frame after the stun resolves (state exits
     SHIELD_STUN into SHIELD / SHIELD_START / SHIELD_REFLECT).
  4. shield_strength at that actionable frame is < 30 (small shield).
  5. Emit up to WINDOW_FRAMES (default 3) consecutive prompts while
     the player remains actionable-in-shield.

Verifier: 1.0 if the sampled controller produces an action that
transitions out of shield. Multiple valid escapes — see `_is_escape`.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np

from rlvr.state.gamestate import ControllerInput
from rlvr.tasks.base import FrameRow, Prompt, Task, Verifier
from rlvr.tasks.registry import register_task

TASK_ID = "escape_pressured_shield"

# libmelee Action enum values — kept as integers to avoid importing
# melee at tagger hot path.
SHIELD_START = 178
SHIELD = 179
SHIELD_RELEASE = 180
SHIELD_STUN = 181
SHIELD_REFLECT = 182
SHIELD_FAMILY = {SHIELD_START, SHIELD, SHIELD_RELEASE, SHIELD_STUN, SHIELD_REFLECT}
SHIELD_ACTIONABLE = {SHIELD_START, SHIELD, SHIELD_REFLECT}

# Task parameters (tune via constants, not CLI — they're part of the task
# definition stamped into events.parquet metadata via registry_hash).
SHIELD_MAX = 60.0
SMALL_SHIELD_THRESHOLD = 30.0           # 50% of max — visually small
DAMAGE_DELTA_THRESHOLD = 8.0            # absorbed hit >= this
DAMAGE_DELTA_LOOKBACK = 3               # compare shield[i] vs shield[i-k] for k in 1..3
WINDOW_FRAMES = 3                       # emit up to N actionable prompts per event
TRIG_PRESSED_THRESHOLD = 0.3
STICK_ESCAPE_THRESHOLD = 0.5


class EscapePressuredShieldTask:
    id = TASK_ID
    description = (
        f"Escape from pressured shield: player in SHIELD-family, "
        f"shield_strength dropped by >= {DAMAGE_DELTA_THRESHOLD} in the "
        f"last {DAMAGE_DELTA_LOOKBACK} frames, current strength "
        f"< {SMALL_SHIELD_THRESHOLD}. Valid escapes: release trigger, "
        f"grab (Z), jump (X/Y), OoS attack (A), spotdodge, roll."
    )

    def tag_frames(self, replay) -> Iterator[FrameRow]:
        replay_id = replay.path.stem
        stage = replay.stage
        for pi, char in enumerate(replay.player_characters):
            port = replay.player_ports[pi]
            state = replay._post[pi]["state"]
            shield = replay._post[pi]["shield"]
            n = len(state)

            i = 1
            while i < n:
                s = int(state[i])
                if s not in SHIELD_FAMILY:
                    i += 1
                    continue
                # Shield-damage event: shield dropped by >= delta vs any
                # of the last DAMAGE_DELTA_LOOKBACK frames.
                delta = 0.0
                for k in range(1, DAMAGE_DELTA_LOOKBACK + 1):
                    if i - k < 0:
                        break
                    d = float(shield[i - k]) - float(shield[i])
                    if d > delta:
                        delta = d
                if delta < DAMAGE_DELTA_THRESHOLD:
                    i += 1
                    continue

                # Walk forward from damage frame to first SHIELD_ACTIONABLE.
                j = i
                skip = False
                while j < n and int(state[j]) not in SHIELD_ACTIONABLE:
                    if int(state[j]) not in SHIELD_FAMILY:
                        skip = True
                        break
                    j += 1
                if skip or j >= n:
                    i = j + 1
                    continue

                # Small-shield condition at the decision frame.
                sh = float(shield[j])
                if sh >= SMALL_SHIELD_THRESHOLD:
                    # Advance past this event's resolution to avoid
                    # re-tagging the same stun on the next iteration.
                    i = j + 1
                    continue

                # Skip near-start-of-replay (no context for the encoder).
                if j < 180:
                    i = j + 1
                    continue

                # Emit up to WINDOW_FRAMES consecutive actionable frames.
                emitted = 0
                k = 0
                while k < WINDOW_FRAMES and (j + k) < n:
                    sk = int(state[j + k])
                    if sk not in SHIELD_ACTIONABLE:
                        break
                    abs_idx = j + k
                    frame_id = int(replay.frame_ids[abs_idx])
                    yield FrameRow(
                        replay_id=replay_id,
                        player_port=port,
                        frame_idx=frame_id,
                        task_id=self.id,
                        character=char,
                        stage=stage,
                        task_metadata={
                            "damage_frame_idx": int(replay.frame_ids[i]),
                            "decision_offset": int(k),
                            "shield_at_decision": float(sh),
                            "damage_delta": float(delta),
                        },
                    )
                    emitted += 1
                    k += 1

                # Advance past the emitted window so we don't double-tag
                # subsequent damage events within the same decision window.
                i = j + max(emitted, 1)


class EscapePressuredShieldVerifier:
    task_id = TASK_ID

    def __call__(self, prompt: Prompt, sampled_ctrl: ControllerInput) -> float:
        return _is_escape(sampled_ctrl)


def _is_escape(ctrl: ControllerInput) -> float:
    """Any action that would transition the player out of shield returns
    1.0; pure shield-holding with no escape intent returns 0.0."""
    trig_held = (
        ctrl.shoulder_l > TRIG_PRESSED_THRESHOLD
        or ctrl.shoulder_r > TRIG_PRESSED_THRESHOLD
        or ctrl.l_button
        or ctrl.r_button
    )
    # Trigger released -> shield drops -> escape.
    if not trig_held:
        return 1.0
    # Grab out of shield (Z), jump OoS (X/Y), up-smash OoS (A).
    if ctrl.z_button or ctrl.x_button or ctrl.y_button or ctrl.a_button:
        return 1.0
    # Spotdodge: trigger held + main stick firmly down.
    if ctrl.main_y < -STICK_ESCAPE_THRESHOLD:
        return 1.0
    # Roll: trigger held + main stick firmly sideways.
    if abs(ctrl.main_x) > STICK_ESCAPE_THRESHOLD:
        return 1.0
    return 0.0


register_task(EscapePressuredShieldTask(), EscapePressuredShieldVerifier())
