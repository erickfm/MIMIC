#!/usr/bin/env python3
"""Validate that slp_to_shards.py produces identical output to the old pipeline.

Runs a .slp file through both:
  1. Old: .slp → extract.py → parquet → upload_dataset.py tensorize_game() → tensors
  2. New: .slp → slp_to_shards.py extract_replay() → tensors

Compares every tensor key with torch.allclose().

Usage:
    python tools/validate_pipeline.py --slp /path/to/file.slp --meta-dir data/wavedash
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import tempfile

import numpy as np
import pandas as pd
import torch

import mimic.features as F
from mimic.features import load_cluster_centers


def run_old_pipeline(slp_path: str, meta_dir: Path):
    """Run old pipeline: .slp → parquet → tensorize_game → tensors."""
    from tools._legacy_extract import (
        extract_player, extract_projectiles, extract_stage_static,
        preseed_nana, perspective,
    )
    from melee import Console, stages
    from melee.enums import Menu

    # Extract to DataFrame (like extract.py)
    console = Console(is_dolphin=False, path=slp_path, allow_old_version=True)
    console.connect()

    rows = []
    stage = None
    stage_static = None

    while True:
        gs = console.step()
        if gs is None:
            break
        if gs.menu_state != Menu.IN_GAME:
            continue

        if stage_static is None:
            stage = gs.stage
            stage_static = extract_stage_static(stage)

        if stage and stage.name == "YOSHIS_STORY":
            r0, r1, r2 = stages.randall_position(gs.frame)
        else:
            r0 = r1 = r2 = float("nan")

        row = {
            "frame": gs.frame,
            "distance": gs.distance,
            "stage": stage.value,
            **stage_static,
            "randall_height": r0, "randall_left": r1, "randall_right": r2,
        }

        for idx, (port, ps) in enumerate(gs.players.items()):
            pref = f"p{idx+1}_"
            row[f"{pref}port"] = port
            extract_player(row, pref, ps)
        extract_projectiles(row, gs.projectiles)
        preseed_nana(row)
        rows.append(row)

    if not rows:
        return None

    df_combined = pd.DataFrame(rows)
    float64_cols = df_combined.select_dtypes(include=["float64"]).columns
    for col in float64_cols:
        df_combined[col] = df_combined[col].astype("float32")

    df_p1 = perspective(df_combined, "p1_", "p2_")

    # Tensorize (like upload_dataset.py tensorize_game)
    with open(meta_dir / "norm_stats.json") as fh:
        norm_stats = json.load(fh)
    with open(meta_dir / "cat_maps.json") as fh:
        raw = json.load(fh)
        cat_maps = {col: {int(k): v for k, v in m.items()}
                    for col, m in raw.items()}

    stick_centers, shoulder_centers = load_cluster_centers(meta_dir)
    fg = F.build_feature_groups()
    categorical_cols = F.get_categorical_cols(fg)

    df = df_p1[df_p1["frame"] >= 0].reset_index(drop=True)
    if len(df) < 2:
        return None

    df = F.preprocess_df(df, categorical_cols, cat_maps)
    F.apply_normalization(df, norm_stats)
    states = F.df_to_state_tensors(df, fg)
    targets = F.build_targets_batch(df, norm_stats,
                                     stick_centers=stick_centers,
                                     shoulder_centers=shoulder_centers)

    return states, targets, len(df)


def run_new_pipeline(slp_path: str, meta_dir: Path):
    """Run new pipeline: .slp → extract_replay → tensors."""
    from tools.slp_to_shards import extract_replay, _load_prereqs

    schema, norm_stats, cat_maps, stick_centers, shoulder_centers = (
        _load_prereqs(meta_dir))

    result = extract_replay(slp_path, schema, norm_stats, cat_maps,
                            stick_centers, shoulder_centers)
    if result is None:
        return None

    # Return p1-perspective (first result)
    return result[0]


def compare(old, new, atol=1e-5):
    """Compare two (states, targets, n_frames) tuples."""
    old_states, old_targets, old_n = old
    new_states, new_targets, new_n = new

    issues = 0

    if old_n != new_n:
        print(f"  FRAME COUNT MISMATCH: old={old_n} new={new_n}")
        issues += 1
        return issues

    print(f"  Frames: {old_n}")

    # Compare states
    old_keys = sorted(old_states.keys())
    new_keys = sorted(new_states.keys())
    if old_keys != new_keys:
        print(f"  STATE KEY MISMATCH:")
        print(f"    Old only: {set(old_keys) - set(new_keys)}")
        print(f"    New only: {set(new_keys) - set(old_keys)}")
        issues += 1

    for key in sorted(set(old_keys) & set(new_keys)):
        ov, nv = old_states[key], new_states[key]
        if ov.shape != nv.shape:
            print(f"  SHAPE MISMATCH {key}: old={ov.shape} new={nv.shape}")
            issues += 1
            continue
        if ov.dtype != nv.dtype:
            print(f"  DTYPE MISMATCH {key}: old={ov.dtype} new={nv.dtype}")
            issues += 1
            continue

        if ov.dtype == torch.long:
            if not torch.equal(ov, nv):
                diff_count = (ov != nv).sum().item()
                first_diff = (ov != nv).nonzero(as_tuple=True)[0][0].item()
                print(f"  MISMATCH {key} (int64): {diff_count} diffs, "
                      f"first at frame {first_diff}: "
                      f"old={ov[first_diff].item()} new={nv[first_diff].item()}")
                issues += 1
        else:
            if not torch.allclose(ov, nv, atol=atol, rtol=0):
                diff = (ov - nv).abs()
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()
                n_bad = (diff > atol).sum().item()
                print(f"  MISMATCH {key} (float): max_diff={max_diff:.6e} "
                      f"mean_diff={mean_diff:.6e} n_exceed_atol={n_bad}")
                issues += 1

    # Compare targets
    old_tkeys = sorted(old_targets.keys())
    new_tkeys = sorted(new_targets.keys())
    if old_tkeys != new_tkeys:
        print(f"  TARGET KEY MISMATCH:")
        print(f"    Old only: {set(old_tkeys) - set(new_tkeys)}")
        print(f"    New only: {set(new_tkeys) - set(old_tkeys)}")
        issues += 1

    for key in sorted(set(old_tkeys) & set(new_tkeys)):
        ov, nv = old_targets[key], new_targets[key]
        if ov.shape != nv.shape:
            print(f"  TARGET SHAPE MISMATCH {key}: old={ov.shape} new={nv.shape}")
            issues += 1
            continue

        if ov.dtype == torch.long:
            if not torch.equal(ov, nv):
                diff_count = (ov != nv).sum().item()
                print(f"  TARGET MISMATCH {key} (int64): {diff_count} diffs")
                issues += 1
        else:
            if not torch.allclose(ov, nv, atol=atol, rtol=0):
                diff = (ov - nv).abs()
                max_diff = diff.max().item()
                print(f"  TARGET MISMATCH {key} (float): max_diff={max_diff:.6e}")
                issues += 1

    return issues


def main():
    parser = argparse.ArgumentParser(
        description="Validate new pipeline against old pipeline")
    parser.add_argument("--slp", type=str, required=True,
                        help="Path to .slp replay file")
    parser.add_argument("--meta-dir", type=str, required=True,
                        help="Directory with norm_stats.json, cat_maps.json, etc.")
    parser.add_argument("--atol", type=float, default=1e-5,
                        help="Absolute tolerance for float comparison")
    args = parser.parse_args()

    meta_dir = Path(args.meta_dir)

    print("=== Pipeline Validation ===")
    print(f"  .slp: {args.slp}")
    print(f"  meta: {meta_dir}")
    print()

    print("Running OLD pipeline (parquet path) ...")
    old = run_old_pipeline(args.slp, meta_dir)
    if old is None:
        print("  Old pipeline returned None (no valid frames)")
        return 1

    print("Running NEW pipeline (direct arrays) ...")
    new = run_new_pipeline(args.slp, meta_dir)
    if new is None:
        print("  New pipeline returned None (no valid frames)")
        return 1

    print()
    print("Comparing outputs ...")
    issues = compare(old, new, atol=args.atol)

    print()
    if issues == 0:
        print("PASS — all tensors match!")
        return 0
    else:
        print(f"FAIL — {issues} mismatches found")
        return 1


if __name__ == "__main__":
    sys.exit(main())
