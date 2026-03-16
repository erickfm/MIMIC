#!/usr/bin/env python3
"""
diagnose.py -- Compare training vs inference tensor pipelines.

Loads a saved inference batch and a training parquet, runs both through
the model, and reports any distributional mismatches that could explain
why inference produces near-zero button confidence.
"""

import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch

import features as F
from model import FramePredictor, ModelConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("data/full")
CKPT_PATH = Path("checkpoints/cls-ar-ctx180-40k-s2_step040000.pt")

# ── Load metadata ─────────────────────────────────────────────────────────
with open(DATA_DIR / "norm_stats.json") as fh:
    norm_stats = json.load(fh)
with open(DATA_DIR / "cat_maps.json") as fh:
    raw = json.load(fh)
    cat_maps = {col: {int(k): v for k, v in m.items()} for col, m in raw.items()}

fg = F.build_feature_groups(no_opp_inputs=True)
cat_cols = F.get_categorical_cols(fg)


def load_training_batch(seq_len: int = 180) -> Dict[str, torch.Tensor]:
    """Load one training parquet, preprocess, return a single window as (1, T, *) tensors."""
    parquets = sorted(DATA_DIR.glob("*.parquet"))
    for pq in parquets[:20]:
        df = pd.read_parquet(pq)
        df = df[df["frame"] >= 0].reset_index(drop=True)
        if len(df) >= seq_len + 1:
            break
    else:
        raise RuntimeError("No parquet with enough frames")

    df = F.preprocess_df(df, cat_cols, cat_maps)
    F.apply_normalization(df, norm_stats)
    window = df.iloc[:seq_len]
    state = F.df_to_state_tensors(window, fg)
    return {k: v.unsqueeze(0) for k, v in state.items()}


def load_inference_batch(call_num: int = 300) -> Dict[str, torch.Tensor]:
    """Load a saved inference batch from /tmp."""
    path = Path(f"/tmp/frame_inf_batch_{call_num}.pt")
    if not path.exists():
        raise FileNotFoundError(f"No saved batch at {path}")
    return torch.load(path, map_location="cpu")


def load_inference_rows(call_num: int = 300):
    """Load the raw row dicts that produced the inference batch."""
    path = Path(f"/tmp/frame_inf_rows_{call_num}.pkl")
    if not path.exists():
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)


def compare_tensors(train_batch: Dict[str, torch.Tensor],
                    inf_batch: Dict[str, torch.Tensor]) -> None:
    """Compare every tensor key between training and inference batches."""
    all_keys = sorted(set(train_batch.keys()) | set(inf_batch.keys()))

    print("\n" + "=" * 80)
    print("TENSOR COMPARISON: TRAINING vs INFERENCE")
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

        t_flat = t.reshape(-1)
        i_flat = i.reshape(-1)

        t_mean, t_std = t_flat.mean().item(), t_flat.std().item()
        i_mean, i_std = i_flat.mean().item(), i_flat.std().item()
        t_min, t_max = t_flat.min().item(), t_flat.max().item()
        i_min, i_max = i_flat.min().item(), i_flat.max().item()

        mean_diff = abs(t_mean - i_mean)
        std_diff = abs(t_std - i_std)

        flag = ""
        if mean_diff > 1.0:
            flag = " <<<< LARGE MEAN DIFF"
        elif mean_diff > 0.3:
            flag = " << moderate diff"

        if t.dtype in (torch.long, torch.int64, torch.int32):
            t_uniq = t_flat.unique().tolist()[:10]
            i_uniq = i_flat.unique().tolist()[:10]
            print(f"\n  {key} (categorical, {t.shape}):")
            print(f"    train unique (first 10): {t_uniq}")
            print(f"    inf   unique (first 10): {i_uniq}")
            if set(map(int, t_uniq)) != set(map(int, i_uniq)):
                print(f"    !! Different value sets")
        else:
            print(f"\n  {key} ({t.shape}):{flag}")
            print(f"    train: mean={t_mean:+.4f} std={t_std:.4f} range=[{t_min:.4f}, {t_max:.4f}]")
            print(f"    inf:   mean={i_mean:+.4f} std={i_std:.4f} range=[{i_min:.4f}, {i_max:.4f}]")

            if t.dim() >= 2 and t.shape[-1] > 1:
                n_cols = t.shape[-1]
                for c in range(min(n_cols, 30)):
                    tc = t[..., c].reshape(-1)
                    ic = i[..., c].reshape(-1)
                    cd = abs(tc.mean().item() - ic.mean().item())
                    if cd > 0.5:
                        print(f"      col {c}: train_mean={tc.mean():.4f} inf_mean={ic.mean():.4f} DIFF={cd:.4f} !!!")


def compare_logits(model: FramePredictor, train_batch: Dict[str, torch.Tensor],
                   inf_batch: Dict[str, torch.Tensor]) -> None:
    """Run the model on both batches and compare output distributions."""
    print("\n" + "=" * 80)
    print("MODEL OUTPUT COMPARISON")
    print("=" * 80)

    model.eval()
    with torch.no_grad():
        train_in = {k: v.to(DEVICE) for k, v in train_batch.items()}
        inf_in = {k: v.to(DEVICE) for k, v in inf_batch.items()}

        train_out = model(train_in)
        inf_out = model(inf_in)

        train_out = {k: v[:, -1].cpu().squeeze(0) for k, v in train_out.items()}
        inf_out = {k: v[:, -1].cpu().squeeze(0) for k, v in inf_out.items()}

    for key in sorted(train_out.keys()):
        t = train_out[key]
        i = inf_out[key]
        print(f"\n  {key} ({t.shape}):")

        if key == "btn_logits":
            btn_names = ["A", "B", "X", "Y", "Z", "L", "R", "START",
                         "D_UP", "D_DN", "D_LT", "D_RT"]
            t_probs = torch.sigmoid(t)
            i_probs = torch.sigmoid(i)
            print(f"    {'btn':>6}  train_prob  inf_prob   diff")
            print(f"    {'----':>6}  ----------  --------   ----")
            for j, name in enumerate(btn_names):
                tp = t_probs[j].item()
                ip = i_probs[j].item()
                flag = " <<<" if abs(tp - ip) > 0.1 else ""
                print(f"    {name:>6}  {tp:10.4f}  {ip:8.4f}   {tp - ip:+.4f}{flag}")

        elif key == "c_dir_logits":
            t_probs = torch.softmax(t, dim=-1)
            i_probs = torch.softmax(i, dim=-1)
            labels = ["neutral", "up", "down", "left", "right"]
            print(f"    {'dir':>8}  train_prob  inf_prob")
            for j, name in enumerate(labels):
                print(f"    {name:>8}  {t_probs[j]:.4f}      {i_probs[j]:.4f}")

        elif key == "main_xy":
            if t.shape[-1] > 2:
                t_probs = torch.softmax(t, dim=-1)
                i_probs = torch.softmax(i, dim=-1)
                t_top5 = t_probs.topk(5)
                i_top5 = i_probs.topk(5)
                print(f"    train top-5 clusters: {list(zip(t_top5.indices.tolist(), [f'{p:.3f}' for p in t_top5.values.tolist()]))}")
                print(f"    inf   top-5 clusters: {list(zip(i_top5.indices.tolist(), [f'{p:.3f}' for p in i_top5.values.tolist()]))}")
                t_ent = -(t_probs * (t_probs + 1e-8).log()).sum().item()
                i_ent = -(i_probs * (i_probs + 1e-8).log()).sum().item()
                print(f"    train entropy: {t_ent:.3f}  inf entropy: {i_ent:.3f}  (max={np.log(t.shape[-1]):.3f})")
            else:
                print(f"    train: {t.tolist()}")
                print(f"    inf:   {i.tolist()}")

        elif key in ("L_val", "R_val"):
            if t.shape[-1] > 1:
                t_probs = torch.softmax(t, dim=-1)
                i_probs = torch.softmax(i, dim=-1)
                print(f"    train probs: {[f'{p:.3f}' for p in t_probs.tolist()]}")
                print(f"    inf   probs: {[f'{p:.3f}' for p in i_probs.tolist()]}")
            else:
                print(f"    train: {t.item():.4f}  inf: {i.item():.4f}")


def inspect_inference_rows(rows):
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
            val = df.iloc[mid][col]
            print(f"    {col:>30} = {val}")

    action_vals = df["self_action"].unique()
    print(f"\n  self_action unique values in window: {sorted(action_vals)[:20]}")
    char_val = df["self_character"].iloc[0]
    print(f"  self_character: {char_val}")

    print(f"\n  self_main_x range: [{df['self_main_x'].min():.4f}, {df['self_main_x'].max():.4f}]")
    print(f"  self_main_y range: [{df['self_main_y'].min():.4f}, {df['self_main_y'].max():.4f}]")

    btn_cols_list = [c for c in df.columns if c.startswith("self_btn_")]
    if btn_cols_list:
        btn_sums = df[btn_cols_list].sum()
        active = btn_sums[btn_sums > 0]
        if len(active) > 0:
            print(f"  Active buttons in window: {dict(active)}")
        else:
            print(f"  NO buttons pressed in entire window")


def main():
    print("=" * 80)
    print("MIMIC Pipeline Diagnostic")
    print("=" * 80)

    call_num = 300
    for cn in [300, 600, 1200, 3, 2, 1]:
        if Path(f"/tmp/frame_inf_batch_{cn}.pt").exists():
            call_num = cn
            break

    print(f"\nUsing inference batch from call #{call_num}")
    print(f"Checkpoint: {CKPT_PATH}")
    print(f"Data dir: {DATA_DIR}")

    inf_batch = load_inference_batch(call_num)
    inf_rows = load_inference_rows(call_num)

    print(f"\nInference batch keys: {sorted(inf_batch.keys())}")
    for k, v in sorted(inf_batch.items()):
        print(f"  {k}: shape={v.shape} dtype={v.dtype}")

    if inf_rows:
        inspect_inference_rows(inf_rows)

    print("\nLoading training batch...")
    seq_len = list(inf_batch.values())[0].shape[1]
    train_batch = load_training_batch(seq_len)

    print(f"Training batch keys: {sorted(train_batch.keys())}")
    for k, v in sorted(train_batch.items()):
        print(f"  {k}: shape={v.shape} dtype={v.dtype}")

    compare_tensors(train_batch, inf_batch)

    print("\nLoading model...")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    cfg = ModelConfig(**ckpt["config"])
    model = FramePredictor(cfg).to(DEVICE)
    sd = ckpt["model_state_dict"]
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    compare_logits(model, train_batch, inf_batch)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
