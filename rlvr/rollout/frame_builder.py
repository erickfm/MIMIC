"""Build MIMIC model-input tensors from canonical RLVR GameState objects.

Adapts `tools/inference_utils.build_frame` to consume our RLVR dataclasses
instead of libmelee GameState objects. The output schema is bit-identical
to what the model was trained on; we just change the input producer.

Per-field numeric transforms are imported from inference_utils so we can't
drift from the shard-builder conventions.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from mimic.features import encode_controller_onehot_single
from rlvr.state.gamestate import ControllerInput, GameState, PlayerState
from tools.inference_utils import MIMIC_NUM_FULL, XFORM, build_mock_frame


def _numeric_13(ps: PlayerState, hal_features: dict, norm_stats: dict) -> List[float]:
    """13-col normalized numeric vector for one player. Mirrors
    tools/inference_utils._player_numeric_full but reads from our
    dataclass instead of a libmelee PlayerState."""
    raw = {
        "pos_x": ps.position_x,
        "pos_y": ps.position_y,
        "percent": ps.percent,
        "stock": float(ps.stock),
        "jumps_left": float(ps.jumps_left),
        "speed_air_x_self": ps.speed_air_x_self,
        "speed_ground_x_self": ps.speed_ground_x_self,
        "speed_x_attack": ps.speed_x_attack,
        "speed_y_attack": ps.speed_y_attack,
        "speed_y_self": ps.speed_y_self,
        "hitlag_left": ps.hitlag_left,
        "hitstun_left": ps.hitstun_frames_left,
        "shield_strength": ps.shield_strength,
    }
    out = []
    for f in MIMIC_NUM_FULL:
        if f in hal_features and f in XFORM:
            out.append(XFORM[f](raw[f], hal_features[f]))
        else:
            stats = norm_stats.get(f"self_{f}")
            if stats is None:
                out.append(0.0)
            else:
                mean, std = stats[0], stats[1]
                out.append((raw[f] - mean) / std if std != 0 else 0.0)
    return out


def _flags_5(ps: PlayerState) -> List[float]:
    return [
        float(ps.on_ground),
        float(ps.off_stage),
        float(ps.facing),
        float(ps.invulnerable),
        float(ps.moonwalkwarning),
    ]


def _controller_onehot(ctrl: ControllerInput, ctx: dict) -> np.ndarray:
    """One-hot encode a ControllerInput to MIMIC's 56-dim (with 7-combo)
    or 52-dim (with 5-combo) representation. Matches the shard-builder.

    The existing encoder expects:
      - main / c-stick x,y in the [0, 1] libmelee-controller range (not
        the [-1, 1] peppi range). We rescale here.
      - buttons as a dict keyed by BUTTON_* strings with 0/1 ints.
      - `l_shldr`, `r_shldr` as floats in [0, 1].
    """
    def _pm1_to_01(v: float) -> float:
        return (v + 1.0) * 0.5

    btns = {
        "BUTTON_A": int(ctrl.a_button),
        "BUTTON_B": int(ctrl.b_button),
        "BUTTON_X": int(ctrl.x_button),
        "BUTTON_Y": int(ctrl.y_button),
        "BUTTON_Z": int(ctrl.z_button),
        "BUTTON_L": int(ctrl.l_button),
        "BUTTON_R": int(ctrl.r_button),
    }
    return encode_controller_onehot_single(
        _pm1_to_01(ctrl.main_x), _pm1_to_01(ctrl.main_y),
        _pm1_to_01(ctrl.c_x), _pm1_to_01(ctrl.c_y),
        ctrl.shoulder_l, ctrl.shoulder_r,
        btns, ctx["combo_map"], ctx["n_combos"],
    )


def build_frame_from_gamestate(
    gs: GameState,
    self_port: int,
    prev_ctrl: Optional[ControllerInput],
    ctx: dict,
) -> Dict[str, torch.Tensor]:
    """Build one single-frame tensor dict, MIMIC encoder-compatible.

    `self_port` chooses which of the two players is ego. `prev_ctrl`
    is the controller the ego player sent on the PREVIOUS frame (or
    None, in which case we use a neutral fallback).
    """
    assert len(gs.players) == 2, f"expected 2 players, got {len(gs.players)}"
    by_port = {p.port: p for p in gs.players}
    self_ps = by_port[self_port]
    opp_port = next(p for p in by_port if p != self_port)
    opp_ps = by_port[opp_port]

    stage_idx = ctx["stage_map"].get(gs.stage, 0)

    def player_feats(ps):
        char_idx = ctx["char_map"].get(ps.character, 0)
        action_idx = ctx["action_map"].get(ps.action, 0)
        nums = _numeric_13(ps, ctx["hal_features"], ctx.get("norm_stats", {}))
        flags = _flags_5(ps)
        return char_idx, action_idx, nums, flags

    ego_char, ego_action, ego_nums, ego_flags = player_feats(self_ps)
    opp_char, opp_action, opp_nums, opp_flags = player_feats(opp_ps)

    ctrl = prev_ctrl if prev_ctrl is not None else ControllerInput.neutral()
    controller_oh = _controller_onehot(ctrl, ctx)

    return {
        "stage": torch.tensor([stage_idx], dtype=torch.long),
        "self_character": torch.tensor([ego_char], dtype=torch.long),
        "opp_character": torch.tensor([opp_char], dtype=torch.long),
        "self_action": torch.tensor([ego_action], dtype=torch.long),
        "opp_action": torch.tensor([opp_action], dtype=torch.long),
        "self_numeric": torch.tensor([ego_nums], dtype=torch.float32),
        "opp_numeric": torch.tensor([opp_nums], dtype=torch.float32),
        "self_flags": torch.tensor([ego_flags], dtype=torch.float32),
        "opp_flags": torch.tensor([opp_flags], dtype=torch.float32),
        "self_controller": torch.from_numpy(controller_oh).unsqueeze(0),
        "self_costume": torch.tensor([0], dtype=torch.long),
        "opp_costume": torch.tensor([0], dtype=torch.long),
        "self_port": torch.tensor([0], dtype=torch.long),
        "opp_port": torch.tensor([0], dtype=torch.long),
        "self_c_dir": torch.tensor([0], dtype=torch.long),
        "opp_c_dir": torch.tensor([0], dtype=torch.long),
    }


def build_context_batch(
    state_context: Tuple[GameState, ...],
    self_port: int,
    ctx: dict,
    context_length: int = 180,
) -> Dict[str, torch.Tensor]:
    """Build a T-frame tensor batch (T, feat_dim) for one prompt.

    Always returns exactly `context_length` frames. When state_context
    is shorter than context_length, the batch is prepended with
    HAL-style mock frames (same recipe as tools/inference_utils's
    PlayerState.push_frame prefill) so the model sees the deliberately-
    nonsensical filler it was trained to ignore. When longer, only the
    last `context_length` frames are used.

    `prev_ctrl` at each frame t is the ego player's controller at frame
    t-1 from the replay — this matches MIMIC's `self_controller`
    (previous-sent) input semantics.
    """
    if len(state_context) >= context_length:
        state_context = state_context[-context_length:]

    frames: list = []
    prev_ctrl: Optional[ControllerInput] = None
    for gs in state_context:
        f = build_frame_from_gamestate(gs, self_port, prev_ctrl, ctx)
        frames.append(f)
        by_port = {p.port: p for p in gs.players}
        prev_ctrl = by_port[self_port].controller

    # Prepend mock frames if the real context is short.
    n_missing = context_length - len(frames)
    if n_missing > 0:
        mock = build_mock_frame(ctx)
        frames = [mock] * n_missing + frames

    stacked: Dict[str, torch.Tensor] = {}
    for k in frames[0]:
        stacked[k] = torch.cat([f[k] for f in frames], dim=0)
    return stacked
