#!/usr/bin/env python3
"""
closedloop_debug.py -- Frame-by-frame comparison of training vs inference tensors.

Follows Eric Gu's HAL methodology: "overfit on a single synthetic example,
and debug until training and closed loop eval data distributions perfectly match."

Phase 1 (offline, no emulator):
  Loads the training parquet, processes through BOTH the training pipeline
  (MeleeFrameDatasetWithDelay) and the inference pipeline (rows_to_state_seq),
  then compares tensors frame-by-frame to find preprocessing mismatches.

Phase 2 (online, with emulator):
  Reads inference logs (raw row dicts saved per-frame during a live run) and
  compares against the training parquet row-by-row.

Usage:
    # Phase 1: offline tensor comparison
    python3 closedloop_debug.py --data-dir data/wavedash --checkpoint checkpoints/wavedash-idle_best.pt

    # Phase 2: compare inference logs against training data
    python3 closedloop_debug.py --data-dir data/wavedash --checkpoint checkpoints/wavedash-idle_best.pt \
        --inference-log logs/diag/inf_rows_1.pkl
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

import features as F
from features import load_cluster_centers


def parse_args():
    p = argparse.ArgumentParser(description="Closed-loop distribution debug tool")
    p.add_argument("--data-dir", type=str, required=True,
                   help="Training data directory (parquet files)")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Model checkpoint (used for norm_stats/cat_maps/config)")
    p.add_argument("--inference-log", type=str, default=None,
                   help="Phase 2: path to pickled inference row dicts")
    p.add_argument("--window-idx", type=int, default=0,
                   help="Which training window to compare (default: 0 = first)")
    p.add_argument("--num-windows", type=int, default=5,
                   help="Number of windows to compare (default: 5)")
    p.add_argument("--tolerance", type=float, default=1e-5,
                   help="Absolute tolerance for float comparison")
    p.add_argument("--clusters-path", type=str, default=None,
                   help="Path to stick_clusters.json (default: data/full/stick_clusters.json)")
    return p.parse_args()


def load_norm_stats(data_dir: Path, checkpoint: str = None) -> Dict:
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        ns = ckpt.get("norm_stats", {})
        if ns:
            print(f"  norm_stats from checkpoint ({len(ns)} cols)")
            return ns

    ns_path = data_dir / "norm_stats.json"
    if ns_path.exists():
        with open(ns_path) as fh:
            ns = json.load(fh)
        print(f"  norm_stats from {ns_path} ({len(ns)} cols)")
        return ns

    return {}


def load_cat_maps(data_dir: Path, checkpoint: str = None) -> Dict:
    cm_path = data_dir / "cat_maps.json"
    if cm_path.exists():
        with open(cm_path) as fh:
            raw = json.load(fh)
        return {col: {int(k): v for k, v in m.items()} for col, m in raw.items()}
    return {}


def inference_pipeline(
    df_raw: pd.DataFrame,
    start: int, end: int,
    fg: Dict,
    categorical_cols: List[str],
    cat_maps: Dict,
    norm_stats: Dict,
) -> Dict[str, torch.Tensor]:
    """Run the inference preprocessing path on a slice of the raw parquet."""
    df_slice = df_raw.iloc[start:end].copy()

    df_slice = F.preprocess_df(df_slice, categorical_cols, cat_maps)
    F.apply_normalization(df_slice, norm_stats)

    missing_cats = [c for c in categorical_cols if c not in df_slice.columns]
    if missing_cats:
        df_slice = pd.concat([df_slice, pd.DataFrame(
            {c: 0 for c in missing_cats}, index=df_slice.index)], axis=1)

    numeric_missing = {}
    for _, meta in F.walk_groups(fg, return_meta=True):
        if meta["ftype"] != "categorical":
            for col in meta["cols"]:
                if col not in df_slice.columns:
                    numeric_missing[col] = 0.0
    if numeric_missing:
        df_slice = pd.concat([df_slice, pd.DataFrame(
            numeric_missing, index=df_slice.index)], axis=1)

    state_seq = F.df_to_state_tensors(df_slice, fg)
    return state_seq


def compare_tensors(
    train_tensors: Dict[str, torch.Tensor],
    inf_tensors: Dict[str, torch.Tensor],
    tol: float,
    window_label: str,
) -> List[Dict]:
    """Compare two tensor dicts element-by-element. Returns list of mismatches."""
    mismatches = []

    all_keys = set(train_tensors.keys()) | set(inf_tensors.keys())
    train_only = set(train_tensors.keys()) - set(inf_tensors.keys())
    inf_only = set(inf_tensors.keys()) - set(train_tensors.keys())

    if train_only:
        mismatches.append({
            "window": window_label,
            "key": str(train_only),
            "issue": "keys_only_in_training",
            "detail": f"Missing from inference: {train_only}",
        })
    if inf_only:
        mismatches.append({
            "window": window_label,
            "key": str(inf_only),
            "issue": "keys_only_in_inference",
            "detail": f"Missing from training: {inf_only}",
        })

    for key in sorted(all_keys):
        if key not in train_tensors or key not in inf_tensors:
            continue

        t_train = train_tensors[key]
        t_inf = inf_tensors[key]

        if t_train.shape != t_inf.shape:
            mismatches.append({
                "window": window_label,
                "key": key,
                "issue": "shape_mismatch",
                "detail": f"train={list(t_train.shape)} vs inf={list(t_inf.shape)}",
            })
            continue

        if t_train.dtype != t_inf.dtype:
            mismatches.append({
                "window": window_label,
                "key": key,
                "issue": "dtype_mismatch",
                "detail": f"train={t_train.dtype} vs inf={t_inf.dtype}",
            })

        if t_train.is_floating_point():
            diff = (t_train.float() - t_inf.float()).abs()
            max_diff = diff.max().item()
            if max_diff > tol:
                worst_idx = diff.argmax().item()
                flat_shape = t_train.shape
                if len(flat_shape) == 1:
                    pos = worst_idx
                elif len(flat_shape) == 2:
                    row = worst_idx // flat_shape[1]
                    col = worst_idx % flat_shape[1]
                    pos = (row, col)
                else:
                    pos = worst_idx

                mismatches.append({
                    "window": window_label,
                    "key": key,
                    "issue": "value_mismatch",
                    "detail": (f"max_diff={max_diff:.8f} at pos={pos}, "
                               f"train={t_train.flatten()[worst_idx].item():.8f}, "
                               f"inf={t_inf.flatten()[worst_idx].item():.8f}"),
                    "max_diff": max_diff,
                    "n_diverged": int((diff > tol).sum().item()),
                    "total_elements": int(t_train.numel()),
                })
        else:
            ne = (t_train != t_inf)
            if ne.any():
                first_idx = ne.flatten().nonzero(as_tuple=True)[0][0].item()
                mismatches.append({
                    "window": window_label,
                    "key": key,
                    "issue": "value_mismatch",
                    "detail": (f"{ne.sum().item()} mismatches, "
                               f"first at idx={first_idx}: "
                               f"train={t_train.flatten()[first_idx].item()}, "
                               f"inf={t_inf.flatten()[first_idx].item()}"),
                    "n_diverged": int(ne.sum().item()),
                    "total_elements": int(t_train.numel()),
                })

    return mismatches


def phase1_offline(args):
    """Compare training pipeline vs inference pipeline on the same parquet data."""
    data_dir = Path(args.data_dir)
    print("=" * 70)
    print("PHASE 1: Offline tensor comparison (training path vs inference path)")
    print("=" * 70)

    norm_stats = load_norm_stats(data_dir, args.checkpoint)
    cat_maps = load_cat_maps(data_dir, args.checkpoint)

    fg = F.build_feature_groups(no_opp_inputs=True)
    categorical_cols = F.get_categorical_cols(fg)

    print(f"\n  Building training dataset from {data_dir} ...")
    ds = MeleeFrameDatasetWithDelay(
        parquet_dir=str(data_dir),
        sequence_length=30,
        reaction_delay=1,
        split="train",
        norm_stats=norm_stats,
        no_opp_inputs=True,
    )
    print(f"  Dataset: {len(ds)} windows, {len(ds.files)} files")

    raw_dfs = {}
    for f in ds.files:
        raw_dfs[f] = pd.read_parquet(f)
        raw_dfs[f] = raw_dfs[f][raw_dfs[f]["frame"] >= 0].reset_index(drop=True)

    all_mismatches = []
    n_ok = 0

    for wi in range(args.window_idx, min(args.window_idx + args.num_windows, len(ds))):
        fpath, start_idx = ds.index_map[wi]
        W = ds.sequence_length

        train_state, train_target = ds[wi]

        if fpath in raw_dfs:
            raw_df = raw_dfs[fpath]
        else:
            raw_df = pd.read_parquet(fpath)
            raw_df = raw_df[raw_df["frame"] >= 0].reset_index(drop=True)

        inf_state = inference_pipeline(
            raw_df, start_idx, start_idx + W,
            fg, categorical_cols, cat_maps, norm_stats,
        )

        label = f"win={wi} file={fpath.name} start={start_idx}"
        mismatches = compare_tensors(train_state, inf_state, args.tolerance, label)

        if mismatches:
            all_mismatches.extend(mismatches)
        else:
            n_ok += 1

    print(f"\n{'=' * 70}")
    print(f"RESULTS: {args.num_windows} windows compared")
    print(f"  OK (exact match): {n_ok}")
    print(f"  Mismatches: {len(all_mismatches)}")
    print(f"{'=' * 70}")

    if all_mismatches:
        seen = set()
        for m in all_mismatches:
            sig = (m["key"], m["issue"])
            if sig in seen:
                continue
            seen.add(sig)
            print(f"\n  [{m['issue']}] key={m['key']}")
            print(f"    {m['detail']}")
            if "n_diverged" in m:
                print(f"    diverged: {m['n_diverged']}/{m['total_elements']} elements")
    else:
        print("\n  All tensors match perfectly between training and inference paths.")

    return len(all_mismatches)


def phase2_online(args):
    """Compare logged inference rows against training parquet."""
    data_dir = Path(args.data_dir)
    log_path = Path(args.inference_log)

    print("=" * 70)
    print("PHASE 2: Online comparison (inference logs vs training parquet)")
    print("=" * 70)

    with open(log_path, "rb") as fh:
        inf_rows = pickle.load(fh)
    print(f"  Loaded {len(inf_rows)} inference row dicts from {log_path}")

    parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"  ERROR: no parquet files in {data_dir}")
        return 1

    train_df = pd.read_parquet(parquet_files[0])
    train_df = train_df[train_df["frame"] >= 0].reset_index(drop=True)
    print(f"  Training data: {len(train_df)} frames from {parquet_files[0].name}")

    n_compare = min(len(inf_rows), len(train_df))
    print(f"  Comparing first {n_compare} frames")

    raw_cols = [
        "self_action", "self_action_frame",
        "self_position_x", "self_position_y",
        "self_percent", "self_stock",
        "self_facing", "self_on_ground",
        "self_jumps_left", "self_speed_ground_x_self",
        "self_main_x", "self_main_y",
        "self_l_shldr", "self_r_shldr",
        "self_btn_BUTTON_Y", "self_btn_BUTTON_L",
        "self_btn_BUTTON_A", "self_btn_BUTTON_B",
    ]

    first_divergence = None
    divergence_summary: Dict[str, List] = {}

    for i in range(n_compare):
        inf_row = inf_rows[i]
        train_row = train_df.iloc[i]

        for col in raw_cols:
            inf_val = inf_row.get(col)
            train_val = train_row.get(col)

            if inf_val is None or (hasattr(train_val, '__class__') and
                                   train_val.__class__.__name__ == 'NAType'):
                continue

            try:
                inf_f = float(inf_val)
                train_f = float(train_val)
            except (TypeError, ValueError):
                if inf_val != train_val:
                    if col not in divergence_summary:
                        divergence_summary[col] = []
                    if len(divergence_summary[col]) < 5:
                        divergence_summary[col].append(
                            (i, f"inf={inf_val} vs train={train_val}"))
                    if first_divergence is None:
                        first_divergence = (i, col, inf_val, train_val)
                continue

            diff = abs(inf_f - train_f)
            if diff > 0.001:
                if col not in divergence_summary:
                    divergence_summary[col] = []
                if len(divergence_summary[col]) < 5:
                    divergence_summary[col].append(
                        (i, f"inf={inf_f:.6f} vs train={train_f:.6f} (diff={diff:.6f})"))
                if first_divergence is None:
                    first_divergence = (i, col, inf_f, train_f)

    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")

    if first_divergence:
        frame, col, inf_v, train_v = first_divergence
        print(f"\n  FIRST DIVERGENCE at frame {frame}:")
        print(f"    Feature: {col}")
        print(f"    Inference: {inf_v}")
        print(f"    Training:  {train_v}")

    if divergence_summary:
        print(f"\n  {len(divergence_summary)} features diverge:")
        for col, examples in sorted(divergence_summary.items()):
            print(f"\n    {col}:")
            for frame_idx, detail in examples:
                print(f"      frame {frame_idx}: {detail}")
    else:
        print("\n  All compared features match perfectly.")

    return len(divergence_summary)


def main():
    args = parse_args()

    n_issues = phase1_offline(args)

    if args.inference_log:
        n_issues += phase2_online(args)

    if n_issues > 0:
        print(f"\nTotal unique issues found: {n_issues}")
        print("Fix these mismatches and re-run until clean.")
    else:
        print("\nNo mismatches found. Training and inference distributions match.")

    return 0 if n_issues == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
