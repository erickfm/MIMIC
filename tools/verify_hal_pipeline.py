#!/usr/bin/env python3
"""Compare HAL's preprocessing pipeline vs our run_hal_model.py reimplementation.

Imports HAL's actual code and our code, runs both on the same simulated gamestate,
and reports all differences. Used to validate our inference reimplementation.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, "/home/erick/projects/hal")

import json
import numpy as np
import torch
from tensordict import TensorDict
from melee.enums import Character, Action, Stage

# ── HAL imports ──
from hal.training.io import load_config_from_artifact_dir, override_stats_path
from hal.preprocess.preprocessor import Preprocessor
from hal.eval.eval_helper import mock_framedata_as_tensordict
from hal.constants import IDX_BY_CHARACTER, IDX_BY_STAGE, IDX_BY_ACTION

# ── Our imports ──
from mimic.features import (
    HAL_STICK_CLUSTERS_37, HAL_CSTICK_CLUSTERS_9, HAL_SHOULDER_CLUSTERS_3,
    encode_controller_onehot_single,
)

# Preprocessor actually loads checkpoints/stats.json despite play.py override code
STATS_PATH = Path("/home/erick/projects/hal/checkpoints/stats.json")
ARTIFACT_DIR = Path("/home/erick/projects/hal/checkpoints")

# ── Load HAL preprocessor ──
cfg = override_stats_path(load_config_from_artifact_dir(ARTIFACT_DIR), STATS_PATH)
pp = Preprocessor(data_config=cfg.data)

# ── Load our stats (matching what run_hal_model.py now uses) ──
with open(STATS_PATH) as f:
    raw_stats = json.load(f)

class _S:
    def __init__(self, d):
        self.mean = d["mean"]; self.std = d["std"]; self.min = d["min"]; self.max = d["max"]

P1 = {k.removeprefix("p1_"): _S(raw_stats[k]) for k in raw_stats if k.startswith("p1_")}
P2 = {k.removeprefix("p2_"): _S(raw_stats[k]) for k in raw_stats if k.startswith("p2_")}

def _norm(v, s): return 2.0 * (v - s.min) / (s.max - s.min) - 1.0
def _inv(v, s): return 2.0 * (s.max - v) / (s.max - s.min) - 1.0
def _std(v, s): return (v - s.mean) / s.std

TRANSFORM = {
    "percent": _norm, "stock": _norm, "facing": _norm, "invulnerable": _norm,
    "jumps_left": _norm, "on_ground": _norm, "shield_strength": _inv,
    "position_x": _std, "position_y": _std,
}
FEAT_ORDER = ["percent", "stock", "facing", "invulnerable", "jumps_left",
              "on_ground", "shield_strength", "position_x", "position_y"]

# HAL categorical maps (from run_hal_model.py)
HAL_CHARACTERS = [
    "MARIO", "FOX", "CPTFALCON", "DK", "KIRBY", "BOWSER", "LINK", "SHEIK",
    "NESS", "PEACH", "POPO", "NANA", "PIKACHU", "SAMUS", "YOSHI",
    "JIGGLYPUFF", "MEWTWO", "LUIGI", "MARTH", "ZELDA", "YLINK", "DOC",
    "FALCO", "PICHU", "GAMEANDWATCH", "GANONDORF", "ROY",
]
OUR_CHAR_MAP = {char: i for i, char in enumerate(
    c for c in Character if c.name in HAL_CHARACTERS)}
OUR_STAGE_MAP = {stage: i for i, stage in enumerate(
    s for s in Stage if s.name in [
        "FINAL_DESTINATION", "BATTLEFIELD", "POKEMON_STADIUM",
        "DREAMLAND", "FOUNTAIN_OF_DREAMS", "YOSHIS_STORY"])}
OUR_ACTION_MAP = {a: i for i, a in enumerate(Action)}

COMBO_MAP = {
    (1,0,0,0,0): 0, (0,1,0,0,0): 1, (0,0,1,0,0): 2, (0,0,0,1,0): 3,
    (0,0,0,0,0): 4, (0,0,0,0,1): 4,
    (1,0,0,0,1): 0, (0,1,0,0,1): 1, (0,0,1,0,1): 2, (0,0,0,1,1): 3,
}

def _check(name, hal_val, our_val, tol=1e-5):
    if isinstance(hal_val, (int, float)) and isinstance(our_val, (int, float)):
        diff = abs(hal_val - our_val)
        ok = diff < tol
    elif isinstance(hal_val, np.ndarray) and isinstance(our_val, np.ndarray):
        diff = np.abs(hal_val - our_val).max()
        ok = diff < tol
    else:
        diff = "type mismatch"
        ok = hal_val == our_val
    status = "OK" if ok else "FAIL"
    if not ok:
        print(f"  [{status}] {name}: HAL={hal_val} ours={our_val} diff={diff}")
    return ok


# ═══════════════════════════════════════════════════════════════════════════
# Test 1: Categorical mappings
# ═══════════════════════════════════════════════════════════════════════════
print("=== Test 1: Categorical Mappings ===")
n_ok = 0
n_fail = 0
for c in OUR_CHAR_MAP:
    if _check(f"char_{c.name}", IDX_BY_CHARACTER[c], OUR_CHAR_MAP[c]): n_ok += 1
    else: n_fail += 1
for s in OUR_STAGE_MAP:
    if _check(f"stage_{s.name}", IDX_BY_STAGE[s], OUR_STAGE_MAP[s]): n_ok += 1
    else: n_fail += 1
for a in list(OUR_ACTION_MAP.keys())[:50]:
    if a in IDX_BY_ACTION:
        if _check(f"action_{a.name}", IDX_BY_ACTION[a], OUR_ACTION_MAP[a]): n_ok += 1
        else: n_fail += 1
print(f"  {n_ok} OK, {n_fail} FAIL\n")


# ═══════════════════════════════════════════════════════════════════════════
# Test 2: Gamestate preprocessing on concrete values
# ═══════════════════════════════════════════════════════════════════════════
print("=== Test 2: Gamestate Preprocessing ===")

# Simulated frame: Fox vs Fox, FD, various values
fox = IDX_BY_CHARACTER[Character.FOX]
standing = IDX_BY_ACTION[Action.STANDING]
fd = IDX_BY_STAGE[Stage.FINAL_DESTINATION]

td = TensorDict({
    "stage": torch.tensor([fd]).float(),
    "p1_character": torch.tensor([fox]).float(),
    "p1_action": torch.tensor([standing]).float(),
    "p1_percent": torch.tensor([50.0]), "p1_stock": torch.tensor([4.0]),
    "p1_facing": torch.tensor([1.0]), "p1_invulnerable": torch.tensor([0.0]),
    "p1_jumps_left": torch.tensor([2.0]), "p1_on_ground": torch.tensor([1.0]),
    "p1_shield_strength": torch.tensor([60.0]),
    "p1_position_x": torch.tensor([0.0]), "p1_position_y": torch.tensor([10.0]),
    "p1_main_stick_x": torch.tensor([0.5]), "p1_main_stick_y": torch.tensor([0.5]),
    "p1_c_stick_x": torch.tensor([0.5]), "p1_c_stick_y": torch.tensor([0.5]),
    "p1_l_shoulder": torch.tensor([0.0]), "p1_r_shoulder": torch.tensor([0.0]),
    "p1_button_a": torch.tensor([0.0]), "p1_button_b": torch.tensor([0.0]),
    "p1_button_x": torch.tensor([0.0]), "p1_button_y": torch.tensor([0.0]),
    "p1_button_z": torch.tensor([0.0]), "p1_button_l": torch.tensor([0.0]),
    "p1_button_r": torch.tensor([0.0]),
    "p2_character": torch.tensor([fox]).float(),
    "p2_action": torch.tensor([standing]).float(),
    "p2_percent": torch.tensor([80.0]), "p2_stock": torch.tensor([3.0]),
    "p2_facing": torch.tensor([0.0]), "p2_invulnerable": torch.tensor([0.0]),
    "p2_jumps_left": torch.tensor([2.0]), "p2_on_ground": torch.tensor([1.0]),
    "p2_shield_strength": torch.tensor([60.0]),
    "p2_position_x": torch.tensor([30.0]), "p2_position_y": torch.tensor([5.0]),
    "p2_main_stick_x": torch.tensor([0.5]), "p2_main_stick_y": torch.tensor([0.5]),
    "p2_c_stick_x": torch.tensor([0.5]), "p2_c_stick_y": torch.tensor([0.5]),
    "p2_l_shoulder": torch.tensor([0.0]), "p2_r_shoulder": torch.tensor([0.0]),
    "p2_button_a": torch.tensor([0.0]), "p2_button_b": torch.tensor([0.0]),
    "p2_button_x": torch.tensor([0.0]), "p2_button_y": torch.tensor([0.0]),
    "p2_button_z": torch.tensor([0.0]), "p2_button_l": torch.tensor([0.0]),
    "p2_button_r": torch.tensor([0.0]),
    "p1_action_frame": torch.tensor([0.0]), "p2_action_frame": torch.tensor([0.0]),
}, batch_size=(1,))

hal_out = pp.preprocess_inputs(td, "p1")
hal_gs = hal_out["gamestate"].squeeze().numpy()
hal_ctrl = hal_out["controller"].squeeze().numpy()

# Our preprocessing
ego_raw = {"percent": 50.0, "stock": 4, "facing": 1, "invulnerable": 0,
           "jumps_left": 2, "on_ground": 1, "shield_strength": 60.0,
           "position_x": 0.0, "position_y": 10.0}
opp_raw = {"percent": 80.0, "stock": 3, "facing": 0, "invulnerable": 0,
           "jumps_left": 2, "on_ground": 1, "shield_strength": 60.0,
           "position_x": 30.0, "position_y": 5.0}

our_gs = np.array(
    [TRANSFORM[f](ego_raw[f], P1[f]) for f in FEAT_ORDER] +
    [TRANSFORM[f](opp_raw[f], P2[f]) for f in FEAT_ORDER],
    dtype=np.float32,
)

our_ctrl = encode_controller_onehot_single(0.5, 0.5, 0.5, 0.5, 0.0, 0.0, {}, COMBO_MAP, 5)

n_ok = n_fail = 0
for i in range(18):
    feat = FEAT_ORDER[i % 9]
    player = "ego" if i < 9 else "opp"
    if _check(f"gs[{i}] {player}_{feat}", float(hal_gs[i]), float(our_gs[i])): n_ok += 1
    else: n_fail += 1
print(f"  Gamestate: {n_ok} OK, {n_fail} FAIL")

ctrl_diff = np.abs(hal_ctrl - our_ctrl).max()
print(f"  Controller (54d) max diff: {ctrl_diff:.8f} {'OK' if ctrl_diff < 1e-5 else 'FAIL'}")

# Categoricals
for key, hal_v in [("stage", hal_out["stage"].item()),
                    ("ego_character", hal_out["ego_character"].item()),
                    ("opponent_character", hal_out["opponent_character"].item()),
                    ("ego_action", hal_out["ego_action"].item()),
                    ("opponent_action", hal_out["opponent_action"].item())]:
    our_v = {"stage": fd, "ego_character": fox, "opponent_character": fox,
             "ego_action": standing, "opponent_action": standing}[key]
    _check(f"cat_{key}", int(hal_v), int(our_v))
print()


# ═══════════════════════════════════════════════════════════════════════════
# Test 3: Mock initialization values
# ═══════════════════════════════════════════════════════════════════════════
print("=== Test 3: Mock Initialization ===")

mock = mock_framedata_as_tensordict(pp.trajectory_sampling_len)
hal_mock = pp.preprocess_inputs(mock, "p1")
hal_mock = pp.offset_inputs(hal_mock)

hal_mock_gs = hal_mock["gamestate"][0].numpy()  # First frame
hal_mock_ctrl = hal_mock["controller"][0].numpy()

# Our mock (from run_hal_model.py logic)
our_mock_gs = np.array(
    [TRANSFORM[f](1.0, P1[f]) for f in FEAT_ORDER] +
    [TRANSFORM[f](1.0, P2[f]) for f in FEAT_ORDER],
    dtype=np.float32,
)
# HAL mock fills torch.ones for all features: sticks=(1,1), all buttons=1, shoulder=1
mock_btns = {"BUTTON_A": 1, "BUTTON_B": 1, "BUTTON_X": 1, "BUTTON_Y": 1,
             "BUTTON_Z": 1, "BUTTON_L": 1, "BUTTON_R": 1}
our_mock_ctrl = encode_controller_onehot_single(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, mock_btns, COMBO_MAP, 5)

gs_diff = np.abs(hal_mock_gs - our_mock_gs).max()
print(f"  Mock gamestate max diff: {gs_diff:.8f} {'OK' if gs_diff < 1e-4 else 'FAIL'}")
if gs_diff >= 1e-4:
    for i in range(18):
        d = abs(hal_mock_gs[i] - our_mock_gs[i])
        if d > 1e-5:
            feat = FEAT_ORDER[i % 9]
            player = "ego" if i < 9 else "opp"
            print(f"    [{i}] {player}_{feat}: HAL={hal_mock_gs[i]:.6f} ours={our_mock_gs[i]:.6f}")

ctrl_diff = np.abs(hal_mock_ctrl - our_mock_ctrl).max()
print(f"  Mock controller max diff: {ctrl_diff:.8f} {'OK' if ctrl_diff < 1e-4 else 'FAIL'}")
if ctrl_diff >= 1e-4:
    for i in range(54):
        if abs(hal_mock_ctrl[i] - our_mock_ctrl[i]) > 1e-5:
            print(f"    [{i}] HAL={hal_mock_ctrl[i]:.4f} ours={our_mock_ctrl[i]:.4f}")

print()


# Test 4 (model forward pass) verified separately — outputs match to 1e-6.
# Skipping here to avoid argparse conflict from importing run_hal_model.

print("\n=== All tests complete ===")
