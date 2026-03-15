#!/usr/bin/env python3
"""Offline inference diagnostics for MIMIC.

Loads a checkpoint and a sample of training parquet data, runs the model's
forward pass, and prints distributional statistics on every output head.
Helps diagnose whether the model produces sensible outputs or is degenerate.

Usage:
    python inference_diag.py --checkpoint checkpoints/noi_ctx180_65k_machB.pt
    python inference_diag.py --checkpoint checkpoints/noi_ctx180_65k_machB.pt --n-windows 200
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import features as F
from model import FramePredictor, ModelConfig

parser = argparse.ArgumentParser(description="Offline MIMIC inference diagnostics")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--data-dir", type=str, default="data/full",
                    help="Directory with parquet files")
parser.add_argument("--n-windows", type=int, default=100,
                    help="Number of random windows to sample")
parser.add_argument("--device", type=str, default="cuda")
args = parser.parse_args()

DEVICE = torch.device(args.device if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── Load checkpoint ──────────────────────────────────────────────────────
ckpt = torch.load(args.checkpoint, map_location=DEVICE)
cfg = ModelConfig(**ckpt["config"])
print(f"Config: d_model={cfg.d_model}, layers={cfg.num_layers}, "
      f"seq_len={cfg.max_seq_len}, no_opp_inputs={cfg.no_opp_inputs}, "
      f"stick_loss={cfg.stick_loss}")

model = FramePredictor(cfg).to(DEVICE)
sd = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model_state_dict"].items()}
model.load_state_dict(sd)
model.eval()
print(f"Loaded {args.checkpoint}")

# ── Norm stats + cat maps ────────────────────────────────────────────────
norm_stats = ckpt.get("norm_stats", {})
if not norm_stats:
    for d in [Path("data"), Path("data/subset"), Path("data/full")]:
        ns = d / "norm_stats.json"
        if ns.exists():
            with open(ns) as fh:
                norm_stats = json.load(fh)
            break
print(f"Norm stats: {len(norm_stats)} cols")

cat_maps = {}
for d in [Path("data"), Path("data/subset"), Path("data/full")]:
    cm = d / "cat_maps.json"
    if cm.exists():
        with open(cm) as fh:
            raw = json.load(fh)
            cat_maps = {col: {int(k): v for k, v in m.items()} for col, m in raw.items()}
        break
print(f"Cat maps: {len(cat_maps)} cols")

fg = F.build_feature_groups(no_opp_inputs=cfg.no_opp_inputs)
categorical_cols = F.get_categorical_cols(fg)

# ── Load data ────────────────────────────────────────────────────────────
parquets = sorted(glob.glob(f"{args.data_dir}/*.parquet"))
if not parquets:
    print(f"No parquets in {args.data_dir}"); sys.exit(1)

rng = np.random.default_rng(42)
selected = rng.choice(parquets, size=min(20, len(parquets)), replace=False)

all_dfs = []
for pf in selected:
    df = pd.read_parquet(pf)
    df = df[df["frame"] >= 0].reset_index(drop=True)
    if len(df) >= cfg.max_seq_len:
        all_dfs.append(df)
print(f"Loaded {len(all_dfs)} games from {len(selected)} files")

if not all_dfs:
    print("No games long enough for the window size"); sys.exit(1)

# ── Sample windows and run inference ─────────────────────────────────────
all_main_x, all_main_y = [], []
all_L, all_R = [], []
all_cdir = []
all_btn_probs = []
all_target_main_x, all_target_main_y = [], []

n = 0
for _ in range(args.n_windows * 5):
    if n >= args.n_windows:
        break
    game = all_dfs[rng.integers(len(all_dfs))]
    if len(game) < cfg.max_seq_len + 1:
        continue
    start = rng.integers(0, len(game) - cfg.max_seq_len)
    window = game.iloc[start:start + cfg.max_seq_len].copy()

    target_row = game.iloc[start + cfg.max_seq_len - 1]
    all_target_main_x.append(float(target_row["self_main_x"]))
    all_target_main_y.append(float(target_row["self_main_y"]))

    df = window.copy()
    df = F.preprocess_df(df, categorical_cols, cat_maps)
    F.apply_normalization(df, norm_stats)

    missing_cats = [c for c in categorical_cols if c not in df.columns]
    if missing_cats:
        df = pd.concat([df, pd.DataFrame({c: 0 for c in missing_cats},
                                         index=df.index)], axis=1)
    for _, meta in F.walk_groups(fg, return_meta=True):
        if meta["ftype"] != "categorical":
            for col in meta["cols"]:
                if col not in df.columns:
                    df[col] = 0.0

    state_seq = F.df_to_state_tensors(df, fg)
    batch = {k: v.unsqueeze(0).to(DEVICE) for k, v in state_seq.items()}

    with torch.no_grad():
        preds = model(batch)
        preds = {k: v[:, -1].cpu().squeeze(0) for k, v in preds.items()}

    main = torch.clamp(preds["main_xy"], 0, 1)
    all_main_x.append(main[0].item())
    all_main_y.append(main[1].item())
    all_L.append(torch.clamp(preds["L_val"], 0, 1).item())
    all_R.append(torch.clamp(preds["R_val"], 0, 1).item())
    all_cdir.append(int(torch.argmax(preds["c_dir_logits"])))
    all_btn_probs.append(torch.sigmoid(preds["btn_logits"]).numpy())
    n += 1

print(f"\n{'='*70}")
print(f"  INFERENCE DIAGNOSTICS  ({n} windows, seq_len={cfg.max_seq_len})")
print(f"{'='*70}\n")

# ── Main stick analysis ──────────────────────────────────────────────────
mx, my = np.array(all_main_x), np.array(all_main_y)
tmx, tmy = np.array(all_target_main_x), np.array(all_target_main_y)

print("MAIN STICK (predicted):")
print(f"  X: mean={mx.mean():.4f}  std={mx.std():.4f}  "
      f"min={mx.min():.4f}  max={mx.max():.4f}  "
      f"pct_near_neutral(0.4-0.6)={((mx > 0.4) & (mx < 0.6)).mean()*100:.1f}%")
print(f"  Y: mean={my.mean():.4f}  std={my.std():.4f}  "
      f"min={my.min():.4f}  max={my.max():.4f}  "
      f"pct_near_neutral(0.4-0.6)={((my > 0.4) & (my < 0.6)).mean()*100:.1f}%")
print(f"  X pct_left(<0.3)={((mx < 0.3)).mean()*100:.1f}%  "
      f"pct_right(>0.7)={((mx > 0.7)).mean()*100:.1f}%")

print("MAIN STICK (targets from data):")
print(f"  X: mean={tmx.mean():.4f}  std={tmx.std():.4f}  "
      f"min={tmx.min():.4f}  max={tmx.max():.4f}")
print(f"  Y: mean={tmy.mean():.4f}  std={tmy.std():.4f}  "
      f"min={tmy.min():.4f}  max={tmy.max():.4f}")

print(f"\n  Prediction vs target shift:")
print(f"    X bias: {(mx - tmx).mean():+.4f}  (positive = rightward)")
print(f"    Y bias: {(my - tmy).mean():+.4f}  (positive = upward)")
print(f"    X MAE:  {np.abs(mx - tmx).mean():.4f}")
print(f"    Y MAE:  {np.abs(my - tmy).mean():.4f}")

# ── L/R triggers ─────────────────────────────────────────────────────────
la, ra = np.array(all_L), np.array(all_R)
print(f"\nTRIGGERS:")
print(f"  L: mean={la.mean():.4f}  std={la.std():.4f}  "
      f"pct_active(>0.1)={((la > 0.1)).mean()*100:.1f}%")
print(f"  R: mean={ra.mean():.4f}  std={ra.std():.4f}  "
      f"pct_active(>0.1)={((ra > 0.1)).mean()*100:.1f}%")

# ── C-stick ──────────────────────────────────────────────────────────────
ca = np.array(all_cdir)
labels = {0: "neutral", 1: "up", 2: "down", 3: "left", 4: "right"}
print(f"\nC-STICK DIRECTION:")
for i in range(5):
    pct = (ca == i).mean() * 100
    print(f"  {labels[i]:>8s}: {pct:5.1f}%")

# ── Buttons ──────────────────────────────────────────────────────────────
bp = np.stack(all_btn_probs)
btn_names = [
    "A", "B", "X", "Y", "Z", "L", "R", "START",
    "D_UP", "D_DOWN", "D_LEFT", "D_RIGHT",
]
print(f"\nBUTTON PROBABILITIES (sigmoid):")
print(f"  {'Button':>8s}  {'mean':>6s}  {'std':>6s}  {'p>0.5':>6s}  {'max':>6s}")
for i, name in enumerate(btn_names):
    col = bp[:, i]
    print(f"  {name:>8s}  {col.mean():.4f}  {col.std():.4f}  "
          f"{(col > 0.5).mean()*100:5.1f}%  {col.max():.4f}")

# ── Degenerate output detection ──────────────────────────────────────────
print(f"\n{'='*70}")
print("  HEALTH CHECKS")
print(f"{'='*70}\n")

issues = []
if mx.std() < 0.05:
    issues.append(f"STUCK STICK X: std={mx.std():.4f}, model outputs near-constant X")
if my.std() < 0.05:
    issues.append(f"STUCK STICK Y: std={my.std():.4f}, model outputs near-constant Y")
if ((mx > 0.4) & (mx < 0.6)).mean() > 0.9:
    issues.append(f"NEUTRAL BIAS X: {((mx > 0.4) & (mx < 0.6)).mean()*100:.0f}% near neutral")
if ((my > 0.4) & (my < 0.6)).mean() > 0.9:
    issues.append(f"NEUTRAL BIAS Y: {((my > 0.4) & (my < 0.6)).mean()*100:.0f}% near neutral")
if (ca == 0).mean() > 0.98:
    issues.append(f"C-STICK ALWAYS NEUTRAL: {(ca == 0).mean()*100:.0f}%")
if bp.max() < 0.3:
    issues.append(f"DEAD BUTTONS: max prob across all buttons is {bp.max():.3f}")
if abs((mx - tmx).mean()) > 0.1:
    issues.append(f"LARGE X BIAS: mean shift = {(mx - tmx).mean():+.3f}")
if abs((my - tmy).mean()) > 0.1:
    issues.append(f"LARGE Y BIAS: mean shift = {(my - tmy).mean():+.3f}")
if np.abs(mx - tmx).mean() > 0.2:
    issues.append(f"HIGH X MAE: {np.abs(mx - tmx).mean():.3f} (model is inaccurate)")
if np.abs(my - tmy).mean() > 0.2:
    issues.append(f"HIGH Y MAE: {np.abs(my - tmy).mean():.3f} (model is inaccurate)")

if issues:
    print("ISSUES FOUND:")
    for iss in issues:
        print(f"  [!] {iss}")
else:
    print("  No obvious degenerate outputs detected.")

# ── Raw model output (pre-clamp) for debugging ──────────────────────────
print(f"\n{'='*70}")
print("  RAW OUTPUT (pre-clamp, last 10 windows)")
print(f"{'='*70}\n")

for game in all_dfs[:1]:
    if len(game) < cfg.max_seq_len + 1:
        continue
    window = game.iloc[:cfg.max_seq_len].copy()
    df = window.copy()
    df = F.preprocess_df(df, categorical_cols, cat_maps)
    F.apply_normalization(df, norm_stats)
    missing_cats = [c for c in categorical_cols if c not in df.columns]
    if missing_cats:
        df = pd.concat([df, pd.DataFrame({c: 0 for c in missing_cats},
                                         index=df.index)], axis=1)
    for _, meta in F.walk_groups(fg, return_meta=True):
        if meta["ftype"] != "categorical":
            for col in meta["cols"]:
                if col not in df.columns:
                    df[col] = 0.0
    state_seq = F.df_to_state_tensors(df, fg)
    batch = {k: v.unsqueeze(0).to(DEVICE) for k, v in state_seq.items()}
    with torch.no_grad():
        preds = model(batch)
        preds = {k: v[:, -1].cpu().squeeze(0) for k, v in preds.items()}
    print(f"  main_xy (RAW): {preds['main_xy'].tolist()}")
    print(f"  main_xy (clamped): {torch.clamp(preds['main_xy'], 0, 1).tolist()}")
    print(f"  L_val (RAW): {preds['L_val'].item():.4f}")
    print(f"  R_val (RAW): {preds['R_val'].item():.4f}")
    print(f"  c_dir_logits: {preds['c_dir_logits'].tolist()}")
    print(f"  c_dir softmax: {torch.softmax(preds['c_dir_logits'], -1).tolist()}")
    print(f"  btn_logits: {preds['btn_logits'].tolist()}")
    print(f"  btn_probs:  {torch.sigmoid(preds['btn_logits']).tolist()}")
    break

print(f"\nDone. Sampled {n} windows from {len(all_dfs)} games.")
