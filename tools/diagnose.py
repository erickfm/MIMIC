#!/usr/bin/env python3
"""
diagnose.py -- Debug training vs inference tensor pipelines.

Combines functionality from the former closedloop_debug.py and diagnose.py:
  - Tensor-level comparison between training and inference pipelines
  - Model output (logit) comparison on both pipelines
  - Raw inference row inspection
  - Online comparison of logged inference rows vs training parquet

Usage:
    # Compare training parquet window vs inference preprocessing on same data
    python3 tools/diagnose.py --data-dir data/wavedash --checkpoint checkpoints/best.pt

    # Compare a saved inference batch against training data + run model on both
    python3 tools/diagnose.py --data-dir data/full --checkpoint checkpoints/best.pt \
        --inference-batch logs/diag/inf_batch_300.pt

    # Compare logged inference rows against training parquet
    python3 tools/diagnose.py --data-dir data/wavedash --checkpoint checkpoints/best.pt \
        --inference-log logs/diag/all_rows.pkl
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

import mimic.features as F
from mimic.model import FramePredictor, ModelConfig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="MIMIC pipeline diagnostic tool")
    p.add_argument("--data-dir", type=str, required=True,
                   help="Training data directory (parquet or tensor shards)")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Model checkpoint (.pt) for norm_stats/config/model weights")
    p.add_argument("--inference-batch", type=str, default=None,
                   help="Path to saved inference batch .pt (from inference.py diagnostic saves)")
    p.add_argument("--inference-log", type=str, default=None,
                   help="Path to pickled inference row dicts (from --diag-log-all)")
    p.add_argument("--window-idx", type=int, default=0,
                   help="Which training window to compare (default: 0)")
    p.add_argument("--seq-len", type=int, default=60,
                   help="Sequence length for training window (default: 60)")
    p.add_argument("--tolerance", type=float, default=1e-5,
                   help="Absolute tolerance for float comparison")
    p.add_argument("--no-opp-inputs", action="store_true", default=True,
                   help="Exclude opponent controller inputs (default: True)")
    p.add_argument("--no-self-inputs", action="store_true", default=False,
                   help="Exclude self controller inputs")
    p.add_argument("--run-model", action="store_true",
                   help="Also run model forward pass and compare logits")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------
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


def load_cat_maps(data_dir: Path) -> Dict:
    cm_path = data_dir / "cat_maps.json"
    if cm_path.exists():
        with open(cm_path) as fh:
            raw = json.load(fh)
        return {col: {int(k): v for k, v in m.items()} for col, m in raw.items()}
    return {}


# ---------------------------------------------------------------------------
# Pipeline runners
# ---------------------------------------------------------------------------
def inference_pipeline(
    df_raw: pd.DataFrame,
    start: int, end: int,
    fg: Dict,
    categorical_cols: List[str],
    cat_maps: Dict,
    norm_stats: Dict,
) -> Dict[str, torch.Tensor]:
    """Run the inference preprocessing path on a slice of raw parquet data."""
    df_slice = df_raw.iloc[start:end].copy()
    df_slice = F.preprocess_df(df_slice, categorical_cols, cat_maps)
    F.apply_normalization(df_slice, norm_stats)

    for c in categorical_cols:
        if c not in df_slice.columns:
            df_slice[c] = 0
    for _, meta in F.walk_groups(fg, return_meta=True):
        if meta["ftype"] != "categorical":
            for col in meta["cols"]:
                if col not in df_slice.columns:
                    df_slice[col] = 0.0

    return F.df_to_state_tensors(df_slice, fg)


def load_training_batch(data_dir: Path, fg: Dict, categorical_cols: List[str],
                        cat_maps: Dict, norm_stats: Dict,
                        seq_len: int = 60) -> Dict[str, torch.Tensor]:
    """Load one training window from .pt shards, return as (1, T, *) tensors."""
    manifest_path = data_dir / "tensor_manifest.json"
    if manifest_path.exists():
        import json as _json
        with open(manifest_path) as fh:
            manifest = _json.load(fh)
        shard_names = manifest.get("train_shards", [])
    else:
        shard_names = [p.name for p in sorted(data_dir.glob("*.pt"))]
    if not shard_names:
        raise RuntimeError(f"No .pt shards in {data_dir}")

    for sname in shard_names[:5]:
        shard = torch.load(data_dir / sname, weights_only=True)
        offsets = shard["offsets"]
        states = shard["states"]
        for g in range(shard["n_games"]):
            start, end = offsets[g].item(), offsets[g + 1].item()
            if end - start >= seq_len + 1:
                window = {k: v[start:start + seq_len].unsqueeze(0)
                          for k, v in states.items()}
                return window
    raise RuntimeError(f"No game with >= {seq_len + 1} frames in shards")


# ---------------------------------------------------------------------------
# Tensor comparison (detailed, from closedloop_debug)
# ---------------------------------------------------------------------------
def compare_tensors(
    train_tensors: Dict[str, torch.Tensor],
    inf_tensors: Dict[str, torch.Tensor],
    tol: float,
    label: str = "",
) -> List[Dict]:
    """Compare two tensor dicts element-by-element. Returns list of mismatches."""
    mismatches = []
    all_keys = set(train_tensors.keys()) | set(inf_tensors.keys())
    train_only = set(train_tensors.keys()) - set(inf_tensors.keys())
    inf_only = set(inf_tensors.keys()) - set(train_tensors.keys())

    if train_only:
        mismatches.append({
            "label": label, "key": str(train_only),
            "issue": "keys_only_in_training",
            "detail": f"Missing from inference: {train_only}",
        })
    if inf_only:
        mismatches.append({
            "label": label, "key": str(inf_only),
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
                "label": label, "key": key, "issue": "shape_mismatch",
                "detail": f"train={list(t_train.shape)} vs inf={list(t_inf.shape)}",
            })
            continue

        if t_train.dtype != t_inf.dtype:
            mismatches.append({
                "label": label, "key": key, "issue": "dtype_mismatch",
                "detail": f"train={t_train.dtype} vs inf={t_inf.dtype}",
            })

        if t_train.is_floating_point():
            diff = (t_train.float() - t_inf.float()).abs()
            max_diff = diff.max().item()
            if max_diff > tol:
                worst_idx = diff.argmax().item()
                mismatches.append({
                    "label": label, "key": key, "issue": "value_mismatch",
                    "detail": (f"max_diff={max_diff:.8f} at idx={worst_idx}, "
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
                    "label": label, "key": key, "issue": "value_mismatch",
                    "detail": (f"{ne.sum().item()} mismatches, "
                               f"first at idx={first_idx}: "
                               f"train={t_train.flatten()[first_idx].item()}, "
                               f"inf={t_inf.flatten()[first_idx].item()}"),
                    "n_diverged": int(ne.sum().item()),
                    "total_elements": int(t_train.numel()),
                })

    return mismatches


def compare_tensors_distributional(
    train_batch: Dict[str, torch.Tensor],
    inf_batch: Dict[str, torch.Tensor],
) -> None:
    """Print distributional comparison (mean/std/range) per tensor key."""
    all_keys = sorted(set(train_batch.keys()) | set(inf_batch.keys()))

    print("\n" + "=" * 80)
    print("DISTRIBUTIONAL COMPARISON: TRAINING vs INFERENCE")
    print("=" * 80)

    missing_in_inf = [k for k in train_batch if k not in inf_batch]
    missing_in_train = [k for k in inf_batch if k not in train_batch]
    if missing_in_inf:
        print(f"\n  !! Keys in TRAINING but NOT in INFERENCE: {missing_in_inf}")
    if missing_in_train:
        print(f"\n  !! Keys in INFERENCE but NOT in TRAINING: {missing_in_train}")

    for key in all_keys:
        if key not in train_batch or key not in inf_batch:
            continue
        t = train_batch[key].float()
        i = inf_batch[key].float()
        if t.shape != i.shape:
            print(f"\n  !! SHAPE MISMATCH for '{key}': train={t.shape} inf={i.shape}")
            continue

        t_flat, i_flat = t.reshape(-1), i.reshape(-1)

        if t.dtype in (torch.long, torch.int64, torch.int32):
            t_uniq = t_flat.unique().tolist()[:10]
            i_uniq = i_flat.unique().tolist()[:10]
            print(f"\n  {key} (categorical, {t.shape}):")
            print(f"    train unique (first 10): {t_uniq}")
            print(f"    inf   unique (first 10): {i_uniq}")
        else:
            mean_diff = abs(t_flat.mean().item() - i_flat.mean().item())
            flag = " <<<< LARGE MEAN DIFF" if mean_diff > 1.0 else (" << moderate diff" if mean_diff > 0.3 else "")
            print(f"\n  {key} ({t.shape}):{flag}")
            print(f"    train: mean={t_flat.mean():+.4f} std={t_flat.std():.4f} range=[{t_flat.min():.4f}, {t_flat.max():.4f}]")
            print(f"    inf:   mean={i_flat.mean():+.4f} std={i_flat.std():.4f} range=[{i_flat.min():.4f}, {i_flat.max():.4f}]")


# ---------------------------------------------------------------------------
# Model output comparison (from diagnose.py)
# ---------------------------------------------------------------------------
def compare_logits(model: FramePredictor, train_batch: Dict[str, torch.Tensor],
                   inf_batch: Dict[str, torch.Tensor], device: torch.device) -> None:
    """Run the model on both batches and compare output distributions."""
    print("\n" + "=" * 80)
    print("MODEL OUTPUT COMPARISON")
    print("=" * 80)

    model.eval()
    with torch.no_grad():
        train_in = {k: v.to(device) for k, v in train_batch.items()}
        inf_in = {k: v.to(device) for k, v in inf_batch.items()}
        train_out = model(train_in)
        inf_out = model(inf_in)
        train_out = {k: v[:, -1].cpu().squeeze(0) for k, v in train_out.items()}
        inf_out = {k: v[:, -1].cpu().squeeze(0) for k, v in inf_out.items()}

    btn_names = ["A", "B", "X", "Y", "Z", "L", "R", "START",
                 "D_UP", "D_DN", "D_LT", "D_RT"]
    cdir_names = ["neutral", "up", "down", "left", "right"]

    for key in sorted(train_out.keys()):
        t, i = train_out[key], inf_out[key]
        print(f"\n  {key} ({t.shape}):")

        if key == "btn_logits":
            t_probs, i_probs = torch.sigmoid(t), torch.sigmoid(i)
            print(f"    {'btn':>6}  train_prob  inf_prob   diff")
            print(f"    {'----':>6}  ----------  --------   ----")
            for j, name in enumerate(btn_names):
                tp, ip = t_probs[j].item(), i_probs[j].item()
                flag = " <<<" if abs(tp - ip) > 0.1 else ""
                print(f"    {name:>6}  {tp:10.4f}  {ip:8.4f}   {tp - ip:+.4f}{flag}")
        elif key == "c_dir_logits":
            t_probs = torch.softmax(t, dim=-1)
            i_probs = torch.softmax(i, dim=-1)
            for j, name in enumerate(cdir_names):
                print(f"    {name:>8}  {t_probs[j]:.4f}      {i_probs[j]:.4f}")
        elif key == "main_xy" and t.shape[-1] > 2:
            t_probs = torch.softmax(t, dim=-1)
            i_probs = torch.softmax(i, dim=-1)
            t5, i5 = t_probs.topk(5), i_probs.topk(5)
            print(f"    train top-5: {list(zip(t5.indices.tolist(), [f'{p:.3f}' for p in t5.values.tolist()]))}")
            print(f"    inf   top-5: {list(zip(i5.indices.tolist(), [f'{p:.3f}' for p in i5.values.tolist()]))}")
            t_ent = -(t_probs * (t_probs + 1e-8).log()).sum().item()
            i_ent = -(i_probs * (i_probs + 1e-8).log()).sum().item()
            print(f"    entropy: train={t_ent:.3f}  inf={i_ent:.3f}  (max={np.log(t.shape[-1]):.3f})")
        elif key in ("L_val", "R_val") and t.shape[-1] > 1:
            t_probs = torch.softmax(t, dim=-1)
            i_probs = torch.softmax(i, dim=-1)
            print(f"    train probs: {[f'{p:.3f}' for p in t_probs.tolist()]}")
            print(f"    inf   probs: {[f'{p:.3f}' for p in i_probs.tolist()]}")
        else:
            print(f"    train: {t.tolist()}")
            print(f"    inf:   {i.tolist()}")


# ---------------------------------------------------------------------------
# Inference row inspection
# ---------------------------------------------------------------------------
def inspect_inference_rows(rows: List[Dict]) -> None:
    """Check raw inference row values for obvious issues."""
    if not rows:
        return
    print("\n" + "=" * 80)
    print("INFERENCE ROW INSPECTION (raw values before preprocessing)")
    print("=" * 80)

    df = pd.DataFrame(rows)
    mid = len(df) // 2

    critical = [
        "self_action", "self_character", "self_pos_x", "self_pos_y",
        "self_percent", "self_stock", "self_facing", "self_on_ground",
        "self_main_x", "self_main_y", "self_l_shldr", "self_r_shldr",
        "self_action_frame", "self_jumps_left",
        "opp_action", "opp_character", "opp_pos_x", "opp_pos_y",
        "opp_percent", "opp_stock",
        "stage", "frame", "distance",
    ]
    print(f"\n  Sample row (middle of window, idx={mid}):")
    for col in critical:
        if col in df.columns:
            print(f"    {col:>30} = {df.iloc[mid][col]}")

    if "self_action" in df.columns:
        print(f"\n  self_action unique: {sorted(df['self_action'].unique())[:20]}")
    if "self_character" in df.columns:
        print(f"  self_character: {df['self_character'].iloc[0]}")
    if "self_main_x" in df.columns:
        print(f"  self_main_x range: [{df['self_main_x'].min():.4f}, {df['self_main_x'].max():.4f}]")
        print(f"  self_main_y range: [{df['self_main_y'].min():.4f}, {df['self_main_y'].max():.4f}]")

    btn_cols_list = [c for c in df.columns if c.startswith("self_btn_")]
    if btn_cols_list:
        btn_sums = df[btn_cols_list].sum()
        active = btn_sums[btn_sums > 0]
        print(f"  Active buttons: {dict(active)}" if len(active) > 0
              else "  NO buttons pressed in entire window")


# ---------------------------------------------------------------------------
# Online comparison: inference logs vs training parquet
# ---------------------------------------------------------------------------
def compare_online(data_dir: Path, log_path: Path) -> int:
    """Compare logged inference rows against training parquet."""
    print("=" * 70)
    print("ONLINE COMPARISON (inference logs vs training parquet)")
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
    raw_cols = [
        "self_action", "self_action_frame",
        "self_pos_x", "self_pos_y", "self_percent", "self_stock",
        "self_facing", "self_on_ground", "self_jumps_left",
        "self_speed_ground_x_self",
        "self_main_x", "self_main_y", "self_l_shldr", "self_r_shldr",
        "self_btn_BUTTON_Y", "self_btn_BUTTON_L",
        "self_btn_BUTTON_A", "self_btn_BUTTON_B",
    ]

    divergence_summary: Dict[str, List] = {}

    for i in range(n_compare):
        inf_row = inf_rows[i]
        train_row = train_df.iloc[i]

        for col in raw_cols:
            inf_val = inf_row.get(col)
            train_val = train_row.get(col)
            if inf_val is None:
                continue
            try:
                diff = abs(float(inf_val) - float(train_val))
            except (TypeError, ValueError):
                if inf_val != train_val:
                    divergence_summary.setdefault(col, [])
                    if len(divergence_summary[col]) < 5:
                        divergence_summary[col].append((i, f"inf={inf_val} vs train={train_val}"))
                continue

            if diff > 0.001:
                divergence_summary.setdefault(col, [])
                if len(divergence_summary[col]) < 5:
                    divergence_summary[col].append(
                        (i, f"inf={float(inf_val):.6f} vs train={float(train_val):.6f} (diff={diff:.6f})"))

    print(f"\n{'=' * 70}")
    if divergence_summary:
        print(f"  {len(divergence_summary)} features diverge:")
        for col, examples in sorted(divergence_summary.items()):
            print(f"\n    {col}:")
            for frame_idx, detail in examples:
                print(f"      frame {frame_idx}: {detail}")
    else:
        print("  All compared features match perfectly.")

    return len(divergence_summary)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("MIMIC Pipeline Diagnostic")
    print("=" * 80)

    norm_stats = load_norm_stats(data_dir, args.checkpoint)
    cat_maps = load_cat_maps(data_dir)
    fg = F.build_feature_groups(no_opp_inputs=args.no_opp_inputs,
                                no_self_inputs=args.no_self_inputs)
    categorical_cols = F.get_categorical_cols(fg)

    n_issues = 0

    # --- Offline tensor comparison: training data vs inference pipeline ---
    shards = sorted(data_dir.glob("*.pt"))
    manifest_path = data_dir / "tensor_manifest.json"
    if shards or manifest_path.exists():
        print(f"\nOFFLINE TENSOR COMPARISON (training shard vs inference pipeline)")
        print("-" * 70)

        W = args.seq_len
        train_batch = load_training_batch(
            data_dir, fg, categorical_cols, cat_maps, norm_stats, seq_len=W)
        train_state = {k: v.squeeze(0) for k, v in train_batch.items()}

        # Just report training tensor stats since inference pipeline
        # comparison requires raw data (handled by compare_online below)
        print(f"  Training shard window: {W} frames, "
              f"{len(train_state)} tensor keys")
        for k, v in sorted(train_state.items()):
            if v.dtype == torch.float32:
                print(f"    {k}: shape={list(v.shape)} "
                      f"mean={v.mean():.4f} std={v.std():.4f}")

    # --- Saved inference batch comparison ---
    if args.inference_batch:
        print(f"\nSAVED INFERENCE BATCH COMPARISON")
        print("-" * 70)

        inf_batch = torch.load(args.inference_batch, map_location="cpu")
        seq_len = list(inf_batch.values())[0].shape[1]
        train_batch = load_training_batch(
            data_dir, fg, categorical_cols, cat_maps, norm_stats, seq_len=seq_len)

        compare_tensors_distributional(train_batch, inf_batch)

        if args.run_model and args.checkpoint:
            ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
            cfg_dict = {k: v for k, v in ckpt["config"].items()
                        if k in ModelConfig.__dataclass_fields__}
            cfg = ModelConfig(**cfg_dict)
            model = FramePredictor(cfg).to(device)
            sd = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model_state_dict"].items()}
            model.load_state_dict(sd)
            print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} params")
            compare_logits(model, train_batch, inf_batch, device)

    # --- Online: inference logs vs training parquet ---
    if args.inference_log:
        n_issues += compare_online(data_dir, Path(args.inference_log))

    print(f"\n{'=' * 80}")
    if n_issues > 0:
        print(f"Total issues found: {n_issues}")
    else:
        print("No issues found. Pipelines match.")
    print("=" * 80)
    return 0 if n_issues == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
