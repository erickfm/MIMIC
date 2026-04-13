"""Shared inference utilities for MIMIC model evaluation.

Used by play_vs_cpu.py, head_to_head.py, and play_netplay.py to ensure
consistent frame construction, model invocation, and controller decoding.
"""

import logging
from collections import deque
from pathlib import Path

import melee
import numpy as np
import torch
import torch.nn.functional as Fn

from mimic.model import FramePredictor, ModelConfig
from mimic.features import (
    HAL_STICK_CLUSTERS_37, HAL_CSTICK_CLUSTERS_9, HAL_SHOULDER_CLUSTERS_3,
    encode_controller_onehot_single, load_hal_norm, hal_normalize,
    load_controller_combos, get_enum_map, BTN7_N_CLASSES,
)

log = logging.getLogger("inference_utils")

ALL_ACTION_BUTTONS = [
    melee.enums.Button.BUTTON_A, melee.enums.Button.BUTTON_B,
    melee.enums.Button.BUTTON_X, melee.enums.Button.BUTTON_Y,
    melee.enums.Button.BUTTON_Z, melee.enums.Button.BUTTON_L,
    melee.enums.Button.BUTTON_R,
]
BTN_NAMES_5 = ["A", "B", "Jump", "Z", "NONE"]
BTN_NAMES_7 = ["A", "B", "Z", "JUMP", "TRIG", "A+TRIG", "NONE"]
BUTTONS_NO_SHOULDER_5 = [
    melee.enums.Button.BUTTON_A, melee.enums.Button.BUTTON_B,
    melee.enums.Button.BUTTON_X, melee.enums.Button.BUTTON_Z,
]

# Numeric features in shard column order (7-col hal_minimal)
MIMIC_NUM = ["pos_x", "pos_y", "percent", "stock", "jumps_left",
             "invuln_left", "shield_strength"]

# Transform functions matching hal_norm.json
def _norm(val, s):
    return 2 * (val - s["min"]) / (s["max"] - s["min"]) - 1 if s["max"] != s["min"] else 0.0

def _inv(val, s):
    return 2 * (s["max"] - val) / (s["max"] - s["min"]) - 1 if s["max"] != s["min"] else 0.0

def _std(val, s):
    return (val - s["mean"]) / s["std"] if s["std"] != 0 else 0.0

XFORM = {
    "percent": _norm, "stock": _norm, "jumps_left": _norm,
    "shield_strength": _inv, "pos_x": _std, "pos_y": _std,
}


def load_mimic_model(checkpoint_path, device):
    """Load a MIMIC FramePredictor (or HAL bare state dict) from checkpoint.

    Returns (model, cfg) with model in eval mode on device.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # HAL bare state dict — no "config" key, just raw weights with module. prefix
    if not isinstance(ckpt, dict) or "config" not in ckpt:
        sd = {k.removeprefix("module."): v for k, v in ckpt.items()}
        # Remap HAL key names → MIMIC FramePredictor key names
        remap = {}
        for k, v in sd.items():
            nk = k
            nk = nk.replace("stage_emb.", "encoder.stage_emb.")
            nk = nk.replace("character_emb.", "encoder.char_emb.")
            nk = nk.replace("action_emb.", "encoder.action_emb.")
            nk = nk.replace("transformer.proj_down.", "encoder.proj.")
            nk = nk.replace("transformer.drop.", "encoder.drop.")
            nk = nk.replace("transformer.ln_f.", "final_norm.")
            nk = nk.replace("transformer.h.", "blocks.")
            nk = nk.replace(".attn.", ".self_attn.")
            nk = nk.replace("shoulder_head.", "heads.shoulder_head.")
            nk = nk.replace("c_stick_head.", "heads.cdir_head.")
            nk = nk.replace("main_stick_head.", "heads.main_head.")
            nk = nk.replace("button_head.", "heads.btn_head.")
            remap[nk] = v
        sd = remap
        cfg = ModelConfig(
            d_model=512, nhead=8, num_layers=6, dim_feedforward=2048,
            dropout=0.2, max_seq_len=256, pos_enc="relpos",
            num_stages=6, num_characters=27, num_actions=396, num_c_dirs=9,
            hal_mode=True, hal_minimal_features=True, hal_controller_encoding=True,
            encoder_type="hal_flat", n_controller_combos=5, n_stick_clusters=37,
            no_self_inputs=False, no_opp_inputs=True,
        )
        model = FramePredictor(cfg)
        model.load_state_dict(sd)
        model.eval()
        model.to(device)
        return model, cfg

    cfg_dict = {k: v for k, v in ckpt["config"].items()
                if k in ModelConfig.__dataclass_fields__}
    cfg = ModelConfig(**cfg_dict)
    model = FramePredictor(cfg)
    sd = {k.removeprefix("_orig_mod."): v
          for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(sd)
    model.eval()
    model.to(device)
    return model, cfg


def load_inference_context(data_dir):
    """Load normalization stats, enum maps, and combo map from data dir.

    Returns dict with keys: hal_features, stage_map, char_map, action_map,
    combo_map, n_combos.
    """
    data_dir = Path(data_dir)
    hal_features = load_hal_norm(data_dir)
    _, combo_map, n_combos = load_controller_combos(data_dir)
    return {
        "hal_features": hal_features,
        "stage_map": get_enum_map("stage", {}),
        "char_map": get_enum_map("self_character", {}),
        "action_map": get_enum_map("self_action", {}),
        "combo_map": combo_map,
        "n_combos": n_combos,
    }


def build_frame(gs, prev_sent, ctx):
    """Build a MIMIC-format state dict from a gamestate.

    This produces tensors in the same format as training shards,
    suitable for passing directly to model(state_dict). The encoder's
    forward() handles all further normalization (flag scaling, reordering).

    Args:
        gs: melee.GameState
        prev_sent: dict of previous frame's sent controller values, or None
        ctx: dict from load_inference_context()

    Returns: dict of tensors (each with leading dim 1 for the time axis),
             or None if gamestate is invalid.
    """
    players = sorted(gs.players.items())
    if len(players) < 2:
        return None
    _, ps1 = players[0]
    _, ps2 = players[1]

    hal_features = ctx["hal_features"]
    stage_idx = ctx["stage_map"].get(gs.stage.value, 0)

    def player_feats(ps):
        char_idx = ctx["char_map"].get(ps.character.value, 0)
        action_idx = ctx["action_map"].get(ps.action.value, 0)

        raw_num = {
            "pos_x": float(ps.position.x),
            "pos_y": float(ps.position.y),
            "percent": float(ps.percent),
            "stock": float(ps.stock),
            "jumps_left": float(ps.jumps_left),
            "invuln_left": float(ps.invulnerability_left),
            "shield_strength": float(ps.shield_strength),
        }

        # Normalize numerics using same transforms as sharding
        nums = []
        for f in MIMIC_NUM:
            if f in hal_features and f in XFORM:
                nums.append(XFORM[f](raw_num[f], hal_features[f]))
            else:
                nums.append(0.0)

        # Flags: pass RAW 0/1 values. The encoder normalizes them (* 2.0 - 1.0).
        # Do NOT normalize here or they get double-normalized.
        flags = [float(ps.on_ground), 0.0, float(ps.facing),
                 float(ps.invulnerable), 0.0]

        return char_idx, action_idx, nums, flags

    ego_char, ego_action, ego_nums, ego_flags = player_feats(ps1)
    opp_char, opp_action, opp_nums, opp_flags = player_feats(ps2)

    # Controller from previous frame's sent values
    if prev_sent is not None:
        mx, my = prev_sent["main_x"], prev_sent["main_y"]
        cx, cy = prev_sent["c_x"], prev_sent["c_y"]
        ls, rs = prev_sent["l_shldr"], prev_sent["r_shldr"]
        btns = {b: prev_sent.get(f"btn_{b}", 0) for b in
                ["BUTTON_A", "BUTTON_B", "BUTTON_X", "BUTTON_Y",
                 "BUTTON_Z", "BUTTON_L", "BUTTON_R"]}
    else:
        mx, my, cx, cy, ls, rs = 0.5, 0.5, 0.5, 0.5, 0.0, 0.0
        btns = {}

    controller = encode_controller_onehot_single(
        mx, my, cx, cy, ls, rs, btns, ctx["combo_map"], ctx["n_combos"])

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
        "self_controller": torch.from_numpy(controller).unsqueeze(0),
        "self_costume": torch.tensor([0], dtype=torch.long),
        "opp_costume": torch.tensor([0], dtype=torch.long),
        "self_port": torch.tensor([0], dtype=torch.long),
        "opp_port": torch.tensor([0], dtype=torch.long),
        "self_c_dir": torch.tensor([0], dtype=torch.long),
        "opp_c_dir": torch.tensor([0], dtype=torch.long),
    }


def build_frame_p2(gs, prev_sent, ctx):
    """Like build_frame but from P2's perspective (ego=ps2, opp=ps1)."""
    players = sorted(gs.players.items())
    if len(players) < 2:
        return None
    _, ps1 = players[0]
    _, ps2 = players[1]

    hal_features = ctx["hal_features"]
    stage_idx = ctx["stage_map"].get(gs.stage.value, 0)

    def player_feats(ps):
        char_idx = ctx["char_map"].get(ps.character.value, 0)
        action_idx = ctx["action_map"].get(ps.action.value, 0)
        raw_num = {
            "pos_x": float(ps.position.x), "pos_y": float(ps.position.y),
            "percent": float(ps.percent), "stock": float(ps.stock),
            "jumps_left": float(ps.jumps_left),
            "invuln_left": float(ps.invulnerability_left),
            "shield_strength": float(ps.shield_strength),
        }
        nums = [XFORM[f](raw_num[f], hal_features[f])
                if f in hal_features and f in XFORM else 0.0
                for f in MIMIC_NUM]
        flags = [float(ps.on_ground), 0.0, float(ps.facing),
                 float(ps.invulnerable), 0.0]
        return char_idx, action_idx, nums, flags

    # P2 perspective: ego=ps2, opp=ps1
    ego_char, ego_action, ego_nums, ego_flags = player_feats(ps2)
    opp_char, opp_action, opp_nums, opp_flags = player_feats(ps1)

    if prev_sent is not None:
        mx, my = prev_sent["main_x"], prev_sent["main_y"]
        cx, cy = prev_sent["c_x"], prev_sent["c_y"]
        ls, rs = prev_sent["l_shldr"], prev_sent["r_shldr"]
        btns = {b: prev_sent.get(f"btn_{b}", 0) for b in
                ["BUTTON_A", "BUTTON_B", "BUTTON_X", "BUTTON_Y",
                 "BUTTON_Z", "BUTTON_L", "BUTTON_R"]}
    else:
        mx, my, cx, cy, ls, rs = 0.5, 0.5, 0.5, 0.5, 0.0, 0.0
        btns = {}

    controller = encode_controller_onehot_single(
        mx, my, cx, cy, ls, rs, btns, ctx["combo_map"], ctx["n_combos"])

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
        "self_controller": torch.from_numpy(controller).unsqueeze(0),
        "self_costume": torch.tensor([0], dtype=torch.long),
        "opp_costume": torch.tensor([0], dtype=torch.long),
        "self_port": torch.tensor([0], dtype=torch.long),
        "opp_port": torch.tensor([0], dtype=torch.long),
        "self_c_dir": torch.tensor([0], dtype=torch.long),
        "opp_c_dir": torch.tensor([0], dtype=torch.long),
    }


def build_mock_frame(ctx):
    """Build a HAL-style mock frame for context prefill.

    HAL fills its initial context with torch.ones for all raw features,
    then preprocesses them. This produces deliberately nonsensical values
    that the model learns to ignore/override quickly, rather than a
    plausible "standing still" state that would bias predictions.
    """
    hal_features = ctx["hal_features"]

    # Numerics: normalize raw value 1.0 through each feature's transform
    mock_nums = []
    for f in MIMIC_NUM:
        if f in hal_features and f in XFORM:
            mock_nums.append(XFORM[f](1.0, hal_features[f]))
        else:
            mock_nums.append(0.0)

    # Flags: raw 1.0 for all (encoder does *2-1 → 1.0)
    mock_flags = [1.0, 1.0, 1.0, 1.0, 1.0]

    # Controller: all buttons pressed, sticks at (1,1), full shoulder
    mock_btns = {"BUTTON_A": 1, "BUTTON_B": 1, "BUTTON_X": 1, "BUTTON_Y": 1,
                 "BUTTON_Z": 1, "BUTTON_L": 1, "BUTTON_R": 1}
    mock_ctrl = encode_controller_onehot_single(
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, mock_btns,
        ctx["combo_map"], ctx["n_combos"])

    return {
        "stage": torch.tensor([1], dtype=torch.long),
        "self_character": torch.tensor([1], dtype=torch.long),
        "opp_character": torch.tensor([1], dtype=torch.long),
        "self_action": torch.tensor([1], dtype=torch.long),
        "opp_action": torch.tensor([1], dtype=torch.long),
        "self_numeric": torch.tensor([mock_nums], dtype=torch.float32),
        "opp_numeric": torch.tensor([mock_nums], dtype=torch.float32),
        "self_flags": torch.tensor([mock_flags], dtype=torch.float32),
        "opp_flags": torch.tensor([mock_flags], dtype=torch.float32),
        "self_controller": torch.from_numpy(mock_ctrl).unsqueeze(0),
        "self_costume": torch.tensor([0], dtype=torch.long),
        "opp_costume": torch.tensor([0], dtype=torch.long),
        "self_port": torch.tensor([0], dtype=torch.long),
        "opp_port": torch.tensor([0], dtype=torch.long),
        "self_c_dir": torch.tensor([0], dtype=torch.long),
        "opp_c_dir": torch.tensor([0], dtype=torch.long),
    }


class PlayerState:
    """Manages context window and prediction for one player."""

    def __init__(self, model, seq_len, device, ctx=None):
        self.model = model
        self.seq_len = seq_len
        self.device = device
        self._ctx = ctx
        self._frame_cache = deque(maxlen=seq_len)
        self.prev_sent = None

    def push_frame(self, frame):
        if len(self._frame_cache) == 0:
            # Prefill with HAL-style mock frame (nonsensical values)
            if self._ctx is not None:
                mock = build_mock_frame(self._ctx)
            else:
                mock = {k: v.clone() for k, v in frame.items()}
            for _ in range(self.seq_len - 1):
                self._frame_cache.append({k: v.clone() for k, v in mock.items()})
        self._frame_cache.append(frame)

    def predict(self):
        frames = list(self._frame_cache)
        batch = {}
        for k in frames[0]:
            batch[k] = torch.cat([f[k] for f in frames], dim=0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.model(batch)


def decode_and_press(ctrl, preds, prev_sent, temperature=1.0):
    """Decode model predictions, press controller, return updated prev_sent.

    Works for both 5-class and 7-class button heads.

    Returns: (prev_sent dict, pressed list, btn_names list)
    """
    main_probs = Fn.softmax(preds["main_xy"][0, -1].float() / temperature, dim=-1)
    main_idx = int(torch.multinomial(main_probs, 1))
    mx = float(HAL_STICK_CLUSTERS_37[main_idx][0])
    my = float(HAL_STICK_CLUSTERS_37[main_idx][1])

    shldr_probs = Fn.softmax(preds["shoulder_val"][0, -1].float() / temperature, dim=-1)
    shldr_idx = int(torch.multinomial(shldr_probs, 1))
    shldr = [0.0, 0.4, 1.0][shldr_idx]

    n_cdir = preds["c_dir_logits"].size(-1)
    if n_cdir == 9:
        c_probs = Fn.softmax(preds["c_dir_logits"][0, -1].float() / temperature, dim=-1)
        c_idx = int(torch.multinomial(c_probs, 1))
        cx = float(HAL_CSTICK_CLUSTERS_9[c_idx][0])
        cy = float(HAL_CSTICK_CLUSTERS_9[c_idx][1])
    else:
        dir_idx = int(torch.argmax(preds["c_dir_logits"][0, -1]))
        _C_DIR = {0: (0.5, 0.5), 1: (0.5, 1.0), 2: (0.5, 0.0),
                  3: (0.0, 0.5), 4: (1.0, 0.5)}
        cx, cy = _C_DIR.get(dir_idx, (0.5, 0.5))

    btn_probs = Fn.softmax(preds["btn_logits"][0, -1].float() / temperature, dim=-1)
    btn_idx = int(torch.multinomial(btn_probs, 1))
    n_btn = btn_probs.size(-1)

    # Analog sticks
    ctrl.tilt_analog(melee.enums.Button.BUTTON_MAIN, mx, my)
    ctrl.tilt_analog(melee.enums.Button.BUTTON_C, cx, cy)

    # Release all buttons every frame
    for btn in ALL_ACTION_BUTTONS:
        ctrl.release_button(btn)

    # Shoulder (always sent)
    ctrl.press_shoulder(melee.enums.Button.BUTTON_L, shldr)

    # Buttons
    pressed = []
    if n_btn == 7:
        if btn_idx == 0:
            ctrl.press_button(melee.enums.Button.BUTTON_A); pressed.append("A")
        elif btn_idx == 1:
            ctrl.press_button(melee.enums.Button.BUTTON_B); pressed.append("B")
        elif btn_idx == 2:
            ctrl.press_button(melee.enums.Button.BUTTON_Z); pressed.append("Z")
        elif btn_idx == 3:
            ctrl.press_button(melee.enums.Button.BUTTON_X); pressed.append("JUMP")
        elif btn_idx == 4:
            ctrl.press_button(melee.enums.Button.BUTTON_L); pressed.append("TRIG")
        elif btn_idx == 5:
            ctrl.press_button(melee.enums.Button.BUTTON_A)
            ctrl.press_button(melee.enums.Button.BUTTON_L); pressed.append("A+TRIG")
        btn_names = BTN_NAMES_7
    else:
        if btn_idx < 4:
            btn = BUTTONS_NO_SHOULDER_5[btn_idx]
            ctrl.press_button(btn); pressed.append(btn.name)
        btn_names = BTN_NAMES_5

    ctrl.flush()

    # Build prev_sent for next frame's controller input encoding
    new_prev = {"main_x": mx, "main_y": my, "c_x": cx, "c_y": cy,
                "l_shldr": shldr, "r_shldr": 0.0}
    for b in ["BUTTON_A", "BUTTON_B", "BUTTON_X", "BUTTON_Y",
              "BUTTON_Z", "BUTTON_L", "BUTTON_R"]:
        new_prev[f"btn_{b}"] = 0
    for p in pressed:
        if p == "A": new_prev["btn_BUTTON_A"] = 1
        elif p == "B": new_prev["btn_BUTTON_B"] = 1
        elif p == "Z": new_prev["btn_BUTTON_Z"] = 1
        elif p == "JUMP": new_prev["btn_BUTTON_X"] = 1
        elif p == "TRIG": new_prev["btn_BUTTON_L"] = 1
        elif p == "A+TRIG":
            new_prev["btn_BUTTON_A"] = 1
            new_prev["btn_BUTTON_L"] = 1

    return new_prev, pressed, btn_names
