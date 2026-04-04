#!/usr/bin/env python3
"""Verify inference pipeline matches training pipeline.

Takes a .slp replay, processes it through BOTH:
  1. Training path: slp → DataFrame → preprocess_df → df_to_state_tensors → controller_offset → HAL encoding
  2. Inference path: slp → per-frame _process_one_row → HAL encoding

Compares every tensor value. Any mismatch is a bug.

Usage:
    python tools/verify_inference.py --slp /path/to/fox_game.slp --data-dir data/fox_public_shards
    python tools/verify_inference.py --slp /path/to/fox_game.slp --data-dir data/fox_public_shards --checkpoint checkpoints/hal-flat-5class_best.pt
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import math

import numpy as np
import pandas as pd
import torch
from melee import Console, stages
from melee.enums import Button, Menu

import mimic.features as F
from mimic.features import (
    load_cluster_centers, load_controller_combos,
    encode_controller_onehot, encode_controller_onehot_single,
    HAL_STICK_CLUSTERS_37, HAL_CSTICK_CLUSTERS_9,
)

BTN_ENUMS = [Button[name] for name in F.BTN]


def extract_raw_frames(slp_path: str, max_frames: int = 600):
    """Extract raw gamestate values from a .slp file, returning a list of row dicts."""
    console = Console(is_dolphin=False, path=slp_path, allow_old_version=True)
    console.connect()

    rows = []
    while True:
        gs = console.step()
        if gs is None:
            break
        if gs.menu_state != Menu.IN_GAME or gs.frame < 0:
            continue
        players = list(gs.players.items())
        if len(players) < 2:
            continue

        port1, ps1 = players[0]
        port2, ps2 = players[1]

        # Use p1-perspective: p1=self, p2=opp
        row = {"frame": gs.frame}

        # Stage
        stage = gs.stage
        row["stage"] = stage.value if stage else 0

        # Stage geometry
        if stage:
            row["blastzone_left"] = stages.BLASTZONES[stage][0]
            row["blastzone_right"] = stages.BLASTZONES[stage][1]
            row["blastzone_top"] = stages.BLASTZONES[stage][2]
            row["blastzone_bottom"] = stages.BLASTZONES[stage][3]
            row["stage_edge_left"] = -stages.EDGE_POSITION[stage]
            row["stage_edge_right"] = stages.EDGE_POSITION[stage]
            for name, func in [("left_platform", stages.left_platform_position),
                                ("right_platform", stages.right_platform_position)]:
                h, l, r = func(stage)
                row[f"{name}_height"] = h
                row[f"{name}_left"] = l
                row[f"{name}_right"] = r
            tp = stages.top_platform_position(stage)
            h, l, r = tp if tp[0] is not None else (float("nan"),) * 3
            row["top_platform_height"] = h
            row["top_platform_left"] = l
            row["top_platform_right"] = r
        if stage and stage.name == "YOSHIS_STORY":
            r0, r1, r2 = stages.randall_position(gs.frame)
        else:
            r0 = r1 = r2 = float("nan")
        row["randall_height"] = r0
        row["randall_left"] = r1
        row["randall_right"] = r2

        # Distance
        row["distance"] = math.hypot(
            float(ps1.position.x) - float(ps2.position.x),
            float(ps1.position.y) - float(ps2.position.y))

        # Players
        for prefix, ps, port in [("self", ps1, port1), ("opp", ps2, port2)]:
            row[f"{prefix}_port"] = port
            row[f"{prefix}_character"] = ps.character.value
            row[f"{prefix}_action"] = ps.action.value
            row[f"{prefix}_action_frame"] = ps.action_frame
            row[f"{prefix}_costume"] = ps.costume
            row[f"{prefix}_pos_x"] = float(ps.position.x)
            row[f"{prefix}_pos_y"] = float(ps.position.y)
            row[f"{prefix}_percent"] = float(ps.percent)
            row[f"{prefix}_stock"] = ps.stock
            row[f"{prefix}_jumps_left"] = ps.jumps_left
            row[f"{prefix}_shield_strength"] = float(ps.shield_strength)
            row[f"{prefix}_speed_air_x_self"] = float(ps.speed_air_x_self)
            row[f"{prefix}_speed_ground_x_self"] = float(ps.speed_ground_x_self)
            row[f"{prefix}_speed_x_attack"] = float(ps.speed_x_attack)
            row[f"{prefix}_speed_y_attack"] = float(ps.speed_y_attack)
            row[f"{prefix}_speed_y_self"] = float(ps.speed_y_self)
            row[f"{prefix}_hitlag_left"] = ps.hitlag_left
            row[f"{prefix}_hitstun_left"] = ps.hitstun_frames_left
            row[f"{prefix}_invuln_left"] = ps.invulnerability_left
            for part in ("bottom", "left", "right", "top"):
                ecb = getattr(ps, f"ecb_{part}")
                row[f"{prefix}_ecb_{part}_x"] = float(ecb[0])
                row[f"{prefix}_ecb_{part}_y"] = float(ecb[1])
            row[f"{prefix}_on_ground"] = float(ps.on_ground)
            row[f"{prefix}_off_stage"] = float(ps.off_stage)
            row[f"{prefix}_facing"] = float(ps.facing)
            row[f"{prefix}_invulnerable"] = float(ps.invulnerable)
            row[f"{prefix}_moonwalkwarning"] = float(ps.moonwalkwarning)
            row[f"{prefix}_main_x"] = ps.controller_state.main_stick[0]
            row[f"{prefix}_main_y"] = ps.controller_state.main_stick[1]
            row[f"{prefix}_c_x"] = ps.controller_state.c_stick[0]
            row[f"{prefix}_c_y"] = ps.controller_state.c_stick[1]
            row[f"{prefix}_l_shldr"] = ps.controller_state.l_shoulder
            row[f"{prefix}_r_shldr"] = ps.controller_state.r_shoulder
            for btn_enum in BTN_ENUMS:
                row[f"{prefix}_btn_{btn_enum.name}"] = float(
                    ps.controller_state.button.get(btn_enum, False))

            # Nana
            nana = ps.nana
            np_ = f"{prefix}_nana"
            if nana:
                row[f"{np_}_character"] = nana.character.value
                row[f"{np_}_action"] = nana.action.value
                row[f"{np_}_action_frame"] = nana.action_frame
                for btn_enum in BTN_ENUMS:
                    row[f"{np_}_btn_{btn_enum.name}"] = float(
                        nana.controller_state.button.get(btn_enum, False))
                row[f"{np_}_main_x"] = nana.controller_state.main_stick[0]
                row[f"{np_}_main_y"] = nana.controller_state.main_stick[1]
                row[f"{np_}_c_x"] = nana.controller_state.c_stick[0]
                row[f"{np_}_c_y"] = nana.controller_state.c_stick[1]
                row[f"{np_}_l_shldr"] = nana.controller_state.l_shoulder
                row[f"{np_}_r_shldr"] = nana.controller_state.r_shoulder
                row[f"{np_}_pos_x"] = float(nana.position.x)
                row[f"{np_}_pos_y"] = float(nana.position.y)
                row[f"{np_}_percent"] = float(nana.percent)
                row[f"{np_}_stock"] = nana.stock
                row[f"{np_}_jumps_left"] = nana.jumps_left
                row[f"{np_}_shield_strength"] = float(nana.shield_strength)
                row[f"{np_}_speed_air_x_self"] = float(nana.speed_air_x_self)
                row[f"{np_}_speed_ground_x_self"] = float(nana.speed_ground_x_self)
                row[f"{np_}_speed_x_attack"] = float(nana.speed_x_attack)
                row[f"{np_}_speed_y_attack"] = float(nana.speed_y_attack)
                row[f"{np_}_speed_y_self"] = float(nana.speed_y_self)
                row[f"{np_}_hitlag_left"] = nana.hitlag_left
                row[f"{np_}_hitstun_left"] = nana.hitstun_frames_left
                row[f"{np_}_invuln_left"] = nana.invulnerability_left
                for part in ("bottom", "left", "right", "top"):
                    ecb = getattr(nana, f"ecb_{part}")
                    row[f"{np_}_ecb_{part}_x"] = float(ecb[0])
                    row[f"{np_}_ecb_{part}_y"] = float(ecb[1])
                row[f"{np_}_on_ground"] = float(nana.on_ground)
                row[f"{np_}_off_stage"] = float(nana.off_stage)
                row[f"{np_}_facing"] = float(nana.facing)
                row[f"{np_}_invulnerable"] = float(nana.invulnerable)
                row[f"{np_}_moonwalkwarning"] = float(nana.moonwalkwarning)

        rows.append(row)
        if len(rows) >= max_frames:
            break

    print(f"  Extracted {len(rows)} frames from {slp_path}")
    return rows


def training_path(rows, norm_stats, cat_maps, hal_minimal, combo_map, n_combos):
    """Process raw frames through the training pipeline."""
    df = pd.DataFrame(rows)

    # 1. preprocess_df: c-stick encoding, fillna, categorical mapping, nana flags
    # Fill missing nana/projectile columns with defaults
    for p in ("self_nana", "opp_nana"):
        for ax in ("c_x", "c_y"):
            col = f"{p}_{ax}"
            if col not in df.columns:
                df[col] = 0.5
        for col_suffix in ["character", "action", "action_frame", "pos_x", "pos_y",
                            "percent", "stock", "jumps_left", "shield_strength",
                            "speed_air_x_self", "speed_ground_x_self",
                            "speed_x_attack", "speed_y_attack", "speed_y_self",
                            "hitlag_left", "hitstun_left", "invuln_left",
                            "on_ground", "off_stage", "facing", "invulnerable",
                            "moonwalkwarning"] + \
                           [f"ecb_{p}_{a}" for p in ("bottom","left","right","top") for a in ("x","y")] + \
                           [f"btn_{b}" for b in F.BTN] + \
                           ["main_x", "main_y", "l_shldr", "r_shldr"]:
            col = f"{p}_{col_suffix}"
            if col not in df.columns:
                df[col] = 0
    for j in range(8):
        for suffix in ["_owner", "_type", "_subtype", "_pos_x", "_pos_y",
                        "_speed_x", "_speed_y", "_frame"]:
            col = f"proj{j}{suffix}"
            if col not in df.columns:
                df[col] = 0
    fg = F.build_feature_groups(no_opp_inputs=False, no_self_inputs=False,
                                hal_minimal=hal_minimal)
    categorical_cols = F.get_categorical_cols(fg)
    df = F.preprocess_df(df, categorical_cols, dynamic_maps=cat_maps)

    # 2. Normalize numeric columns
    norm_cols = F.get_norm_cols(fg)
    for col in norm_cols:
        if col in df.columns and col in norm_stats:
            mean, std = norm_stats[col]
            df[col] = ((df[col].astype("float32") - mean) / std).clip(-10, 10)

    # 3. df_to_state_tensors
    state = F.df_to_state_tensors(df, fg)

    # 4. Controller offset: shift self controller by -1
    for ck in ("self_buttons", "self_analog", "self_c_dir"):
        if ck in state:
            orig = state[ck]
            shifted = torch.zeros_like(orig)
            shifted[1:] = orig[:-1]
            state[ck] = shifted

    # 5. HAL controller encoding
    if combo_map is not None:
        onehot = encode_controller_onehot(
            state["self_buttons"].numpy(),
            state["self_analog"].numpy(),
            state["self_c_dir"].numpy(),
            combo_map, n_combos,
            norm_stats=norm_stats,
        )
        state["self_controller"] = torch.from_numpy(onehot)
        del state["self_buttons"]
        del state["self_analog"]
        del state["self_c_dir"]

    return state


def inference_path(rows, norm_stats, cat_maps, hal_minimal, combo_map, n_combos):
    """Process raw frames through inference.py's _process_one_row logic."""
    # Build the same structures inference.py uses
    fg = F.build_feature_groups(no_opp_inputs=True, no_self_inputs=False,
                                hal_minimal=hal_minimal)
    categorical_cols = F.get_categorical_cols(fg)

    # Build enum maps (same as inference.py)
    enum_maps = {}
    for col in categorical_cols:
        enum_maps[col] = F.get_enum_map(col, cat_maps)

    # Build tensor layout
    tensor_layout = []
    for _, meta in F.walk_groups(fg, return_meta=True):
        entity = meta["entity"]
        ftype = meta["ftype"]
        cols = meta["cols"]
        key = f"{entity}_{ftype}" if entity != "global" else ftype
        tensor_layout.append((key, ftype, cols))

    frames = []
    prev_sent = None  # simulate _prev_sent feedback

    for row_raw in rows:
        r = dict(row_raw)

        # Simulate controller feedback from prev frame (like inference does)
        if prev_sent is not None:
            r["self_main_x"] = prev_sent["main_x"]
            r["self_main_y"] = prev_sent["main_y"]
            r["self_l_shldr"] = prev_sent["l_shldr"]
            r["self_r_shldr"] = prev_sent["r_shldr"]
            r["self_c_x"] = prev_sent["c_x"]
            r["self_c_y"] = prev_sent["c_y"]
            for b in F.BTN:
                r[f"self_btn_{b}"] = prev_sent.get(f"btn_{b}", 0)

        # C-stick encoding
        for p in ("self", "opp", "self_nana", "opp_nana"):
            cx = float(r.get(f"{p}_c_x", 0.5) or 0.5)
            cy = float(r.get(f"{p}_c_y", 0.5) or 0.5)
            dx, dy = cx - 0.5, cy - 0.5
            mag = math.hypot(dx, dy)
            if mag <= 0.15:
                r[f"{p}_c_dir"] = 0
            elif abs(dx) >= abs(dy):
                r[f"{p}_c_dir"] = 4 if dx > 0 else 3
            else:
                r[f"{p}_c_dir"] = 1 if dy > 0 else 2

        r.pop("startAt", None)

        sx, sy = float(r.get("self_pos_x", 0)), float(r.get("self_pos_y", 0))
        ox, oy = float(r.get("opp_pos_x", 0)), float(r.get("opp_pos_y", 0))
        r["distance"] = math.hypot(sx - ox, sy - oy)

        r["self_nana_present"] = 1.0 if r.get("self_nana_character", 0) and r["self_nana_character"] > 0 else 0.0
        r["opp_nana_present"] = 1.0 if r.get("opp_nana_character", 0) and r["opp_nana_character"] > 0 else 0.0

        # Capture raw controller values BEFORE normalization (for HAL encoding)
        if combo_map is not None:
            _raw_mx = float(r.get("self_main_x", 0.5) or 0.5)
            _raw_my = float(r.get("self_main_y", 0.5) or 0.5)
            _raw_cx = float(r.get("self_c_x", 0.5) or 0.5)
            _raw_cy = float(r.get("self_c_y", 0.5) or 0.5)
            _raw_ls = float(r.get("self_l_shldr", 0.0) or 0.0)
            _raw_rs = float(r.get("self_r_shldr", 0.0) or 0.0)
            _raw_btns = {b: int(r.get(f"self_btn_{b}", 0) or 0) for b in F.BTN}

        # Categorical mapping
        for col, m in enum_maps.items():
            raw = r.get(col, 0)
            if raw is None or (isinstance(raw, float) and not math.isfinite(raw)):
                raw = 0
            r[col] = m.get(raw, m.get(int(raw), 0))

        # Normalize
        for col, (mean, std) in norm_stats.items():
            if col in r:
                v = r[col]
                if v is None or (isinstance(v, float) and not math.isfinite(v)):
                    v = 0.0
                r[col] = max(-10.0, min(10.0, (float(v) - mean) / std))

        # Build tensors
        state = {}
        for key, ftype, cols in tensor_layout:
            if ftype == "categorical":
                for col in cols:
                    v = r.get(col, 0)
                    if v is None or (isinstance(v, float) and not math.isfinite(v)):
                        v = 0
                    state[col] = torch.tensor([int(v)], dtype=torch.long)
            else:
                vals = []
                for col in cols:
                    v = r.get(col, 0.0)
                    if v is None or (isinstance(v, float) and not math.isfinite(v)):
                        v = 0.0
                    vals.append(float(v))
                if len(vals) == 1:
                    state[key] = torch.tensor(vals, dtype=torch.float32)
                else:
                    state[key] = torch.tensor([vals], dtype=torch.float32)

        # HAL controller encoding
        if combo_map is not None:
            onehot = encode_controller_onehot_single(
                _raw_mx, _raw_my, _raw_cx, _raw_cy, _raw_ls, _raw_rs,
                _raw_btns, combo_map, n_combos,
            )
            state["self_controller"] = torch.from_numpy(onehot).unsqueeze(0)
            state.pop("self_analog", None)
            state.pop("self_c_dir", None)
            state.pop("self_buttons", None)

        # Simulate what inference would send (for next frame's feedback)
        # At real inference, the model predicts and we send those values.
        # Here we use the GROUND TRUTH controller values (what the player actually pressed).
        # This matches training where controller_offset gives frame i-1's actual inputs.
        prev_sent = {
            "main_x": float(row_raw.get("self_main_x", 0.5)),
            "main_y": float(row_raw.get("self_main_y", 0.5)),
            "l_shldr": float(row_raw.get("self_l_shldr", 0.0)),
            "r_shldr": float(row_raw.get("self_r_shldr", 0.0)),
            "c_x": float(row_raw.get("self_c_x", 0.5)),
            "c_y": float(row_raw.get("self_c_y", 0.5)),
        }
        for b in F.BTN:
            prev_sent[f"btn_{b}"] = int(row_raw.get(f"self_btn_{b}", 0))

        frames.append(state)

    # Stack into (T, ...) tensors
    keys = frames[0].keys()
    stacked = {}
    for k in keys:
        vals = [f[k] for f in frames]
        stacked[k] = torch.cat(vals, dim=0)

    return stacked


def compare(train_state, inf_state, combo_map):
    """Compare training and inference tensors, report mismatches."""
    # Keys that HALFlatEncoder actually uses
    hal_keys = ["stage", "self_character", "opp_character", "self_action", "opp_action",
                "self_numeric", "opp_numeric", "self_flags", "opp_flags"]
    if combo_map:
        hal_keys.append("self_controller")

    print(f"\n{'='*70}")
    print(f"  COMPARISON: Training path vs Inference path")
    print(f"{'='*70}")

    all_match = True
    for key in hal_keys:
        t_val = train_state.get(key)
        i_val = inf_state.get(key)

        if t_val is None:
            print(f"\n  {key}: MISSING from training path")
            all_match = False
            continue
        if i_val is None:
            print(f"\n  {key}: MISSING from inference path")
            all_match = False
            continue

        # Shapes might differ (training has all features, inference has hal_minimal)
        min_len = min(len(t_val), len(i_val))
        t_val = t_val[:min_len]
        i_val = i_val[:min_len]

        if t_val.shape != i_val.shape:
            print(f"\n  {key}: SHAPE MISMATCH training={list(t_val.shape)} inference={list(i_val.shape)}")
            # For numeric, compare the HAL-selected columns
            if "numeric" in key and t_val.shape[-1] == 22 and i_val.shape[-1] == 7:
                HAL_IDX = [0, 1, 2, 3, 4, 12, 13]
                t_val = t_val[..., HAL_IDX]
                print(f"    → Selected HAL indices from training: {list(t_val.shape)}")
            else:
                all_match = False
                continue

        if t_val.dtype != i_val.dtype:
            t_val = t_val.float()
            i_val = i_val.float()

        match = torch.allclose(t_val, i_val, atol=1e-4, rtol=1e-3)
        diff = (t_val.float() - i_val.float()).abs()
        n_mismatch = (diff > 1e-4).sum().item()
        max_diff = diff.max().item()

        status = "✓ MATCH" if match else f"✗ MISMATCH ({n_mismatch} elements, max_diff={max_diff:.6f})"
        print(f"\n  {key}: {status}")
        print(f"    shape={list(t_val.shape)} dtype={t_val.dtype}")
        print(f"    training range=[{t_val.min():.4f}, {t_val.max():.4f}]")
        print(f"    inference range=[{i_val.min():.4f}, {i_val.max():.4f}]")

        if not match and n_mismatch > 0:
            all_match = False
            # Show first few mismatches
            if t_val.dim() == 1:
                bad = (diff > 1e-4).nonzero(as_tuple=True)[0][:5]
                for idx in bad:
                    i = idx.item()
                    print(f"    frame {i}: training={t_val[i].item():.6f} inference={i_val[i].item():.6f}")
            elif t_val.dim() == 2:
                bad = (diff > 1e-4).nonzero()[:5]
                for row in bad:
                    i, j = row[0].item(), row[1].item()
                    print(f"    frame {i} col {j}: training={t_val[i,j].item():.6f} inference={i_val[i,j].item():.6f}")

    print(f"\n{'='*70}")
    print(f"  RESULT: {'ALL MATCH ✓' if all_match else 'MISMATCHES FOUND ✗'}")
    print(f"{'='*70}")
    return all_match


def main():
    parser = argparse.ArgumentParser(description="Verify inference matches training pipeline")
    parser.add_argument("--slp", type=str, required=True, help="Path to .slp replay file")
    parser.add_argument("--data-dir", type=str, required=True, help="Data dir with norm_stats.json, cat_maps.json")
    parser.add_argument("--max-frames", type=int, default=300, help="Max frames to compare")
    parser.add_argument("--hal-controller-encoding", action="store_true",
                        help="Compare with HAL controller encoding")
    parser.add_argument("--hal-minimal", action="store_true",
                        help="Use HAL minimal features")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Load metadata
    with open(data_dir / "norm_stats.json") as f:
        norm_stats = json.load(f)
    cat_maps = {}
    cm_path = data_dir / "cat_maps.json"
    if cm_path.exists():
        with open(cm_path) as f:
            raw = json.load(f)
            cat_maps = {col: {int(k): v for k, v in m.items()} for col, m in raw.items()}

    combo_map = None
    n_combos = 5
    if args.hal_controller_encoding:
        cc_path = data_dir / "controller_combos.json"
        if cc_path.exists():
            combos, combo_map = load_controller_combos(data_dir)
            n_combos = len(combos)
            print(f"  Loaded {n_combos} controller combos")

    print(f"Extracting frames from {args.slp} ...")
    rows = extract_raw_frames(args.slp, max_frames=args.max_frames)

    print(f"\nRunning training pipeline ...")
    train_state = training_path(rows, norm_stats, cat_maps, args.hal_minimal, combo_map, n_combos)

    print(f"Running inference pipeline ...")
    inf_state = inference_path(rows, norm_stats, cat_maps, args.hal_minimal, combo_map, n_combos)

    compare(train_state, inf_state, combo_map)


if __name__ == "__main__":
    main()
