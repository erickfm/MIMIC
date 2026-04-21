"""L-cancel opportunity task.

Skill: press a trigger (L/R/Z, or analog shoulder > 0.3) in the 7-frame
window preceding an aerial landing, so the game engine reduces the
landing lag by ~50%.

Tagger: uses the game engine's ground-truth `post.l_cancel` column
(peppi), which flags each eligible landing frame with 1 (success) or
2 (failure). For each such landing at absolute peppi index `lf`, we
emit 7 prompts — one per frame in the window [lf-7, lf-1]. Skips the
opening-7-frames corner case (not enough history).

Verifier: pure function of the sampled action. Reward 1 if the sampled
controller's trigger state indicates an L-cancel press, else 0.

Why it's valid to grade only the action and not the future state: the
tagger has already certified via `post.l_cancel != 0` that this frame is
inside an eligible window. The verifier's job is to ask "given the
state context, does the policy emit a trigger press?" — exactly what
RLVR needs.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np

from rlvr.state.gamestate import ControllerInput
from rlvr.tasks.base import FrameRow, Prompt, Task, Verifier
from rlvr.tasks.registry import register_task

TASK_ID = "l_cancel_opportunity"
WINDOW_FRAMES = 7  # L-cancel window length in frames before landing
TRIGGER_ANALOG_THRESHOLD = 0.3


class LCancelTask:
    id = TASK_ID
    description = (
        "Press L, R, Z, or an analog shoulder > 0.3 within the 7-frame "
        "window preceding an aerial landing to execute an L-cancel. "
        "Tagged from peppi post.l_cancel ground-truth labels."
    )

    def tag_frames(self, replay) -> Iterator[FrameRow]:
        """Emit prompt rows for every frame inside an L-cancel window.

        Iterates each player independently. For a Fox ditto both ports
        emit rows (each as self with the other as opp)."""
        replay_id = replay.path.stem
        stage = replay.stage
        for pi, char in enumerate(replay.player_characters):
            port = replay.player_ports[pi]
            lc = replay.l_cancel_per_player(pi)
            # Indices of eligible landings. Skip if we can't form a full
            # 7-frame window (need at least WINDOW_FRAMES frames of
            # history).
            landing_idx = np.where(lc != 0)[0]
            for lf in landing_idx:
                if lf < WINDOW_FRAMES:
                    continue
                result = int(lc[lf])  # 1 or 2
                for off in range(-WINDOW_FRAMES, 0):
                    abs_idx = int(lf + off)
                    frame_id = int(replay.frame_ids[abs_idx])
                    aerial_state = int(replay._post[pi]["state"][abs_idx])
                    yield FrameRow(
                        replay_id=replay_id,
                        player_port=port,
                        frame_idx=frame_id,
                        task_id=self.id,
                        character=char,
                        stage=stage,
                        task_metadata={
                            "landing_frame_idx": int(replay.frame_ids[int(lf)]),
                            "offset_to_landing": int(off),
                            "replay_l_cancel_result": result,
                            "aerial_action_state": aerial_state,
                        },
                    )


class LCancelVerifier:
    task_id = TASK_ID

    def __call__(self, prompt: Prompt, sampled_ctrl: ControllerInput) -> float:
        return _l_cancel_pressed(sampled_ctrl)


def _l_cancel_pressed(ctrl: ControllerInput) -> float:
    """Core rule: trigger press within-window == L-cancel attempt."""
    if ctrl.l_button or ctrl.r_button or ctrl.z_button:
        return 1.0
    if ctrl.shoulder_l > TRIGGER_ANALOG_THRESHOLD:
        return 1.0
    if ctrl.shoulder_r > TRIGGER_ANALOG_THRESHOLD:
        return 1.0
    return 0.0


# Register at import time.
register_task(LCancelTask(), LCancelVerifier())
