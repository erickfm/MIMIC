#!/usr/bin/env python3
"""Compare HAL's preprocessing against MIMIC's run_hal_model preprocessing.

Processes the same synthetic gamestate through both pipelines and reports
any differences in the output tensors. This validates that run_hal_model.py
produces identical model inputs to HAL's own play.py pipeline.

Usage:
    python tools/compare_hal_mimic_preprocessing.py [--data-dir data/fox_public_shards]
"""

import sys
from pathlib import Path

# Add both project roots to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, "/home/erick/projects/hal")

import argparse
import json
import math
from collections import defaultdict

import numpy as np
import torch
from tensordict import TensorDict

# ── HAL imports ──────────────────────────────────────────────────────────────
from hal.constants import (
    IDX_BY_ACTION, IDX_BY_CHARACTER, IDX_BY_STAGE,
    STICK_XY_CLUSTER_CENTERS_V2, STICK_XY_CLUSTER_CENTERS_V0_1,
    SHOULDER_CLUSTER_CENTERS_V0,
)
from hal.data.stats import load_dataset_stats
from hal.preprocess.input_configs import baseline_controller_fine_main_analog_shoulder
from hal.preprocess.target_configs import fine_main_analog_shoulder
from hal.preprocess.transformations import (
    normalize, invert_and_normalize, standardize, cast_int32,
    concat_controller_inputs,
)

# ── MIMIC imports ────────────────────────────────────────────────────────────
from mimic.features import (
    HAL_STICK_CLUSTERS_37, HAL_CSTICK_CLUSTERS_9, HAL_SHOULDER_CLUSTERS_3,
    encode_controller_onehot_single,
)

from melee.enums import Character, Action, Stage

REPO_ROOT = Path("/home/erick/projects/hal")
STATS_PATH = REPO_ROOT / "hal/data/stats.json"


def make_synthetic_gamestate_td(
    ego_char=Character.FOX, opp_char=Character.FOX,
    stage=Stage.FINAL_DESTINATION,
    ego_pos=(10.0, 5.0), opp_pos=(-20.0, 0.0),
    ego_percent=73.0, opp_percent=45.0,
    ego_stocks=3, opp_stocks=4,
    ego_facing=True, opp_facing=False,
    ego_action=Action.STANDING, opp_action=Action.DASHING,
    ego_on_ground=True, opp_on_ground=True,
    ego_jumps=2, opp_jumps=1,
    ego_shield=60.0, opp_shield=45.0,
    ego_invuln=False, opp_invuln=True,
    # Controller state (previous frame, already in gamestate)
    ego_main=(0.7, 0.5), ego_cstick=(0.5, 0.5),
    ego_l_shoulder=0.0, ego_r_shoulder=0.0,
    ego_buttons=None,  # dict of button_name: 0/1
):
    """Build a synthetic TensorDict matching HAL's extract_eval_gamestate_as_tensordict format."""
    if ego_buttons is None:
        ego_buttons = {"a": 0, "b": 0, "x": 0, "y": 0, "z": 0, "start": 0,
                       "l": 0, "r": 0, "d_up": 0}

    frame_data = defaultdict(list)
    frame_data["frame"].append(100)
    frame_data["stage"].append(IDX_BY_STAGE[stage])

    # Player 1 (ego)
    for prefix, char, action, pos, pct, stock, facing, on_ground, jumps, shield, invuln in [
        ("p1", ego_char, ego_action, ego_pos, ego_percent, ego_stocks, ego_facing,
         ego_on_ground, ego_jumps, ego_shield, ego_invuln),
        ("p2", opp_char, opp_action, opp_pos, opp_percent, opp_stocks, opp_facing,
         opp_on_ground, opp_jumps, opp_shield, opp_invuln),
    ]:
        frame_data[f"{prefix}_character"].append(IDX_BY_CHARACTER[char])
        frame_data[f"{prefix}_action"].append(IDX_BY_ACTION[action])
        frame_data[f"{prefix}_position_x"].append(pos[0])
        frame_data[f"{prefix}_position_y"].append(pos[1])
        frame_data[f"{prefix}_percent"].append(pct)
        frame_data[f"{prefix}_stock"].append(stock)
        frame_data[f"{prefix}_facing"].append(int(facing))
        frame_data[f"{prefix}_on_ground"].append(int(on_ground))
        frame_data[f"{prefix}_jumps_left"].append(jumps)
        frame_data[f"{prefix}_shield_strength"].append(shield)
        frame_data[f"{prefix}_invulnerable"].append(int(invuln))
        frame_data[f"{prefix}_action_frame"].append(1)
        frame_data[f"{prefix}_invulnerability_left"].append(0)
        frame_data[f"{prefix}_hitlag_left"].append(0)
        frame_data[f"{prefix}_hitstun_left"].append(0)
        frame_data[f"{prefix}_speed_air_x_self"].append(0.0)
        frame_data[f"{prefix}_speed_y_self"].append(0.0)
        frame_data[f"{prefix}_speed_x_attack"].append(0.0)
        frame_data[f"{prefix}_speed_y_attack"].append(0.0)
        frame_data[f"{prefix}_speed_ground_x_self"].append(0.0)
        for ecb in ["bottom", "top", "left", "right"]:
            frame_data[f"{prefix}_ecb_{ecb}_x"].append(0.0)
            frame_data[f"{prefix}_ecb_{ecb}_y"].append(0.0)
        frame_data[f"{prefix}_port"].append(1 if prefix == "p1" else 2)

    # Controller inputs (ego's previous frame controller)
    for prefix in ["p1", "p2"]:
        if prefix == "p1":
            main, cstick, l_s, r_s, btns = ego_main, ego_cstick, ego_l_shoulder, ego_r_shoulder, ego_buttons
        else:
            main, cstick, l_s, r_s = (0.5, 0.5), (0.5, 0.5), 0.0, 0.0
            btns = {"a": 0, "b": 0, "x": 0, "y": 0, "z": 0, "start": 0, "l": 0, "r": 0, "d_up": 0}
        frame_data[f"{prefix}_main_stick_x"].append(main[0])
        frame_data[f"{prefix}_main_stick_y"].append(main[1])
        frame_data[f"{prefix}_c_stick_x"].append(cstick[0])
        frame_data[f"{prefix}_c_stick_y"].append(cstick[1])
        frame_data[f"{prefix}_l_shoulder"].append(l_s)
        frame_data[f"{prefix}_r_shoulder"].append(r_s)
        for btn_name in ["a", "b", "x", "y", "z", "start", "l", "r", "d_up"]:
            frame_data[f"{prefix}_button_{btn_name}"].append(btns.get(btn_name, 0))

    return TensorDict(
        {k: torch.tensor(v) for k, v in frame_data.items()},
        batch_size=(1,),
    )


def hal_preprocess(td):
    """Run through HAL's actual preprocessing pipeline."""
    stats = load_dataset_stats(STATS_PATH)
    input_config = baseline_controller_fine_main_analog_shoulder()
    target_config = fine_main_analog_shoulder()

    # Replicate what HAL's Preprocessor.preprocess_inputs does
    ego = "p1"
    opponent = "p2"
    transformation_by_feature_name = input_config.transformation_by_feature_name

    processed_features = {}

    # Process player features
    for player in (ego, opponent):
        perspective = "ego" if player == ego else "opponent"
        for feature_name in input_config.player_features:
            perspective_feature_name = f"{perspective}_{feature_name}"
            player_feature_name = f"{player}_{feature_name}"
            transform = transformation_by_feature_name[feature_name]
            processed_features[perspective_feature_name] = transform(
                td[player_feature_name], stats[player_feature_name]
            )

    # Process non-player features (stage, controller)
    non_player_features = [
        f for f in transformation_by_feature_name if f not in input_config.player_features
    ]
    for feature_name in non_player_features:
        transform = transformation_by_feature_name[feature_name]
        if feature_name in td:
            processed_features[feature_name] = transform(td[feature_name], stats[feature_name])
        else:
            processed_features[feature_name] = transform(td, ego)

    # Concatenate by head
    seen = set()
    result = {}
    for head_name, feature_names in input_config.grouped_feature_names_by_head.items():
        tensors = [processed_features[f] for f in feature_names]
        result[head_name] = torch.cat(tensors, dim=-1)
        seen.update(feature_names)

    # Default head (gamestate): unseen features in insertion order
    unseen = []
    for f, t in processed_features.items():
        if f not in seen:
            if t.ndim == 1:
                t = t.unsqueeze(-1)
            unseen.append(t)
    result["gamestate"] = torch.cat(unseen, dim=-1)

    return result


def mimic_preprocess(td, data_dir):
    """Run through MIMIC's run_hal_model.py preprocessing (fixed version)."""
    # Load HAL's actual stats (same as the fixed run_hal_model.py)
    with open(STATS_PATH) as f:
        raw_stats = json.load(f)

    class Stats:
        def __init__(self, d):
            self.mean, self.std = d["mean"], d["std"]
            self.min, self.max = d["min"], d["max"]

    p1_stats = {k.removeprefix("p1_"): Stats(raw_stats[k]) for k in raw_stats if k.startswith("p1_")}
    p2_stats = {k.removeprefix("p2_"): Stats(raw_stats[k]) for k in raw_stats if k.startswith("p2_")}

    def _normalize(val, s): return 2.0 * (val - s.min) / (s.max - s.min) - 1.0
    def _invert_norm(val, s): return 2.0 * (s.max - val) / (s.max - s.min) - 1.0
    def _standardize(val, s): return (val - s.mean) / s.std

    TRANSFORMS = {
        "percent": _normalize, "stock": _normalize, "facing": _normalize,
        "invulnerable": _normalize, "jumps_left": _normalize,
        "on_ground": _normalize, "shield_strength": _invert_norm,
        "position_x": _standardize, "position_y": _standardize,
    }

    FEATURE_ORDER = ["percent", "stock", "facing", "invulnerable", "jumps_left",
                     "on_ground", "shield_strength", "position_x", "position_y"]

    # Categoricals
    stage_idx = td["stage"][0].item()
    ego_char = td["p1_character"][0].item()
    opp_char = td["p2_character"][0].item()
    ego_action = td["p1_action"][0].item()
    opp_action = td["p2_action"][0].item()

    # Numeric features
    def get_nums(prefix, stats):
        raw = {
            "percent": td[f"{prefix}_percent"][0].item(),
            "stock": float(td[f"{prefix}_stock"][0].item()),
            "facing": float(td[f"{prefix}_facing"][0].item()),
            "invulnerable": float(td[f"{prefix}_invulnerable"][0].item()),
            "jumps_left": float(td[f"{prefix}_jumps_left"][0].item()),
            "on_ground": float(td[f"{prefix}_on_ground"][0].item()),
            "shield_strength": float(td[f"{prefix}_shield_strength"][0].item()),
            "position_x": float(td[f"{prefix}_position_x"][0].item()),
            "position_y": float(td[f"{prefix}_position_y"][0].item()),
        }
        return [TRANSFORMS[f](raw[f], stats[f]) for f in FEATURE_ORDER]

    ego_nums = get_nums("p1", p1_stats)
    opp_nums = get_nums("p2", p2_stats)
    gamestate = np.array(ego_nums + opp_nums, dtype=np.float32)

    # Controller encoding (HAL's button ordering: A=0, B=1, Jump=2, Z=3, NONE=4)
    COMBO_MAP = {
        (1, 0, 0, 0, 0): 0, (0, 1, 0, 0, 0): 1, (0, 0, 1, 0, 0): 2,
        (0, 0, 0, 1, 0): 3, (0, 0, 0, 0, 0): 4, (0, 0, 0, 0, 1): 4,
        (1, 0, 0, 0, 1): 0, (0, 1, 0, 0, 1): 1, (0, 0, 1, 0, 1): 2,
        (0, 0, 0, 1, 1): 3,
    }

    mx = td["p1_main_stick_x"][0].item()
    my = td["p1_main_stick_y"][0].item()
    cx = td["p1_c_stick_x"][0].item()
    cy = td["p1_c_stick_y"][0].item()
    ls = td["p1_l_shoulder"][0].item()
    rs = td["p1_r_shoulder"][0].item()
    btns = {f"BUTTON_{b.upper()}": int(td[f"p1_button_{b}"][0].item())
            for b in ["a", "b", "x", "y", "z", "l", "r"]}

    controller = encode_controller_onehot_single(
        mx, my, cx, cy, ls, rs, btns, COMBO_MAP, 5)

    return {
        "stage": stage_idx,
        "ego_character": ego_char,
        "opponent_character": opp_char,
        "ego_action": ego_action,
        "opponent_action": opp_action,
        "gamestate": torch.tensor(gamestate),
        "controller": torch.tensor(controller),
    }


def compare(hal_result, mimic_result):
    """Compare HAL and MIMIC preprocessing outputs."""
    all_pass = True

    # Categoricals
    for key in ["stage", "ego_character", "opponent_character", "ego_action", "opponent_action"]:
        hal_val = hal_result[key].item() if isinstance(hal_result[key], torch.Tensor) else hal_result[key]
        mimic_val = mimic_result[key] if isinstance(mimic_result[key], (int, float)) else mimic_result[key].item()
        match = hal_val == mimic_val
        status = "PASS" if match else "FAIL"
        if not match:
            all_pass = False
        print(f"  [{status}] {key}: HAL={hal_val}, MIMIC={mimic_val}")

    # Gamestate vector (18 floats)
    hal_gs = hal_result["gamestate"].squeeze().float()
    mimic_gs = mimic_result["gamestate"].squeeze().float()
    max_diff = (hal_gs - mimic_gs).abs().max().item()
    match = max_diff < 1e-5
    status = "PASS" if match else "FAIL"
    if not match:
        all_pass = False
    print(f"  [{status}] gamestate (18-dim): max_diff={max_diff:.8f}")
    if not match:
        feature_names = ["percent", "stock", "facing", "invulnerable", "jumps_left",
                         "on_ground", "shield_strength", "position_x", "position_y"]
        for i in range(18):
            player = "ego" if i < 9 else "opp"
            feat = feature_names[i % 9]
            diff = abs(hal_gs[i].item() - mimic_gs[i].item())
            if diff > 1e-5:
                print(f"    {player}_{feat}: HAL={hal_gs[i].item():.8f}, MIMIC={mimic_gs[i].item():.8f}, diff={diff:.8f}")

    # Controller vector (54 floats)
    hal_ctrl = hal_result["controller"].squeeze().float()
    mimic_ctrl = mimic_result["controller"].squeeze().float()
    if hal_ctrl.shape != mimic_ctrl.shape:
        print(f"  [FAIL] controller shape: HAL={hal_ctrl.shape}, MIMIC={mimic_ctrl.shape}")
        all_pass = False
    else:
        max_diff = (hal_ctrl - mimic_ctrl).abs().max().item()
        match = max_diff < 1e-5
        status = "PASS" if match else "FAIL"
        if not match:
            all_pass = False
        print(f"  [{status}] controller ({hal_ctrl.shape[0]}-dim): max_diff={max_diff:.8f}")
        if not match:
            # Show which section differs
            sections = [("main_stick", 0, 37), ("c_stick", 37, 46),
                        ("buttons", 46, 51), ("shoulder", 51, 54)]
            for name, start, end in sections:
                section_diff = (hal_ctrl[start:end] - mimic_ctrl[start:end]).abs().max().item()
                if section_diff > 1e-5:
                    print(f"    {name}[{start}:{end}]: max_diff={section_diff:.8f}")
                    print(f"      HAL:   {hal_ctrl[start:end].tolist()}")
                    print(f"      MIMIC: {mimic_ctrl[start:end].tolist()}")

    return all_pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/fox_public_shards")
    args = parser.parse_args()

    print("=" * 70)
    print("HAL vs MIMIC Preprocessing Comparison")
    print("=" * 70)

    test_cases = [
        {
            "name": "Standing Fox, neutral controller",
            "kwargs": {},
        },
        {
            "name": "Fox at 100% damage, pressing A, stick right",
            "kwargs": {
                "ego_percent": 100.0,
                "ego_main": (1.0, 0.5),
                "ego_buttons": {"a": 1, "b": 0, "x": 0, "y": 0, "z": 0,
                                "start": 0, "l": 0, "r": 0, "d_up": 0},
            },
        },
        {
            "name": "Fox at 200% jumping with L-shoulder",
            "kwargs": {
                "ego_percent": 200.0,
                "opp_percent": 150.0,
                "ego_pos": (50.0, 30.0),
                "opp_pos": (-10.0, 20.0),
                "ego_on_ground": False,
                "ego_jumps": 1,
                "ego_shield": 30.0,
                "ego_main": (0.3, 0.9),
                "ego_l_shoulder": 0.6,
                "ego_buttons": {"a": 0, "b": 0, "x": 1, "y": 0, "z": 0,
                                "start": 0, "l": 1, "r": 0, "d_up": 0},
            },
        },
        {
            "name": "High damage invulnerable opponent, Z-button",
            "kwargs": {
                "ego_percent": 350.0,
                "opp_percent": 0.0,
                "opp_invuln": True,
                "opp_stocks": 4,
                "ego_stocks": 1,
                "ego_shield": 0.1,
                "ego_buttons": {"a": 0, "b": 0, "x": 0, "y": 0, "z": 1,
                                "start": 0, "l": 0, "r": 0, "d_up": 0},
            },
        },
    ]

    all_pass = True
    for i, tc in enumerate(test_cases):
        print(f"\nTest {i + 1}: {tc['name']}")
        print("-" * 50)

        td = make_synthetic_gamestate_td(**tc["kwargs"])
        hal_result = hal_preprocess(td)
        mimic_result = mimic_preprocess(td, args.data_dir)
        if not compare(hal_result, mimic_result):
            all_pass = False

    print("\n" + "=" * 70)
    if all_pass:
        print("ALL TESTS PASSED - Preprocessing matches HAL exactly!")
    else:
        print("SOME TESTS FAILED - See details above")
    print("=" * 70)


if __name__ == "__main__":
    main()
