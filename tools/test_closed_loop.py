#!/usr/bin/env python3
"""
test_closed_loop.py -- Offline inference test for the closed-loop model.

Loads a checkpoint trained on the closed-loop (Falco-FD) data, feeds
training data windows through the model, and checks whether the outputs
match ground truth. This validates the pipeline end-to-end without
needing Dolphin.

Also compares button confidence levels between the closed-loop model
and the original full-data model.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch

import mimic.features as F
from mimic.model import FramePredictor, ModelConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CL_DATA = Path("data/closed_loop")
CL_CKPT = Path("checkpoints/closed-loop-overfit_step005000.pt")
FULL_CKPT = Path("checkpoints/cls-ar-ctx180-40k-s2_step040000.pt")

with open(CL_DATA / "norm_stats.json") as fh:
    cl_norm_stats = json.load(fh)
with open(CL_DATA / "cat_maps.json") as fh:
    raw = json.load(fh)
    cl_cat_maps = {col: {int(k): v for k, v in m.items()} for col, m in raw.items()}
with open(CL_DATA / "stick_clusters.json") as fh:
    cl_raw = json.load(fh)
    stick_centers = np.array(cl_raw["stick_centers"], dtype=np.float32)
    shoulder_centers = np.array(cl_raw["shoulder_centers"], dtype=np.float32)

fg = F.build_feature_groups(no_opp_inputs=True)
cat_cols = F.get_categorical_cols(fg)


def load_model(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    cfg = ModelConfig(**ckpt["config"])
    model = FramePredictor(cfg).to(DEVICE)
    sd = ckpt["model_state_dict"]
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    return model, cfg, ckpt


def load_batch(parquet_path: Path, norm_stats, cat_maps_dict, seq_len=60):
    df = pd.read_parquet(parquet_path)
    df = df[df["frame"] >= 0].reset_index(drop=True)
    df = F.preprocess_df(df, cat_cols, cat_maps_dict)

    targets_raw = F.build_targets_batch(df, norm_stats,
                                        stick_centers=stick_centers,
                                        shoulder_centers=shoulder_centers)

    F.apply_normalization(df, norm_stats)
    state = F.df_to_state_tensors(df, fg)

    mid = len(df) // 2
    start = max(0, mid - seq_len)
    end = start + seq_len

    window = {k: v[start:end].unsqueeze(0) for k, v in state.items()}
    target = {k: v[start:end] for k, v in targets_raw.items()}
    return window, target, end - 1


def decode_cluster(logits, centers):
    idx = int(torch.argmax(logits))
    return idx, centers[idx]


def run_test(model, cfg, norm_stats, cat_maps_dict, label: str):
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")

    parquets = sorted(CL_DATA.glob("*.parquet"))[:5]
    btn_names = ["A", "B", "X", "Y", "Z", "L", "R", "START",
                 "D_UP", "D_DN", "D_LT", "D_RT"]

    seq_len = cfg.max_seq_len

    all_btn_probs = []
    all_btn_gt = []
    stick_correct = 0
    stick_total = 0
    shoulder_correct = 0
    shoulder_total = 0

    for pq in parquets:
        batch, targets, target_frame = load_batch(pq, norm_stats, cat_maps_dict, seq_len)

        with torch.no_grad():
            inp = {k: v.to(DEVICE) for k, v in batch.items()}
            out = model(inp)
            pred = {k: v[:, -1].cpu().squeeze(0) for k, v in out.items()}

        gt_btns = targets["btns"][seq_len - 1]
        gt_main_cluster = targets["main_cluster"][seq_len - 1].item()
        gt_l_bin = targets["l_bin"][seq_len - 1].item()
        gt_r_bin = targets["r_bin"][seq_len - 1].item()
        gt_cdir = torch.argmax(targets["c_dir"][seq_len - 1]).item()

        btn_probs = torch.sigmoid(pred["btn_logits"])
        main_idx, main_center = decode_cluster(pred["main_xy"], stick_centers)
        l_idx, l_center = decode_cluster(pred["L_val"], shoulder_centers)
        r_idx, r_center = decode_cluster(pred["R_val"], shoulder_centers)
        cdir_idx = int(torch.argmax(pred["c_dir_logits"]))

        all_btn_probs.append(btn_probs)
        all_btn_gt.append(gt_btns)

        if main_idx == gt_main_cluster:
            stick_correct += 1
        stick_total += 1

        if l_idx == gt_l_bin:
            shoulder_correct += 1
        if r_idx == gt_r_bin:
            shoulder_correct += 1
        shoulder_total += 2

        gt_main_center = stick_centers[gt_main_cluster]
        print(f"\n  {pq.name}:")
        print(f"    Stick: pred={main_idx}({main_center[0]:.3f},{main_center[1]:.3f}) "
              f"gt={gt_main_cluster}({gt_main_center[0]:.3f},{gt_main_center[1]:.3f}) "
              f"{'OK' if main_idx == gt_main_cluster else 'MISS'}")
        print(f"    Shoulder L: pred={l_idx}({l_center:.3f}) gt={gt_l_bin}({shoulder_centers[gt_l_bin]:.3f}) "
              f"{'OK' if l_idx == gt_l_bin else 'MISS'}")
        print(f"    Shoulder R: pred={r_idx}({r_center:.3f}) gt={gt_r_bin}({shoulder_centers[gt_r_bin]:.3f}) "
              f"{'OK' if r_idx == gt_r_bin else 'MISS'}")
        print(f"    C-dir: pred={cdir_idx} gt={gt_cdir} {'OK' if cdir_idx == gt_cdir else 'MISS'}")

        active_btns = [(btn_names[i], btn_probs[i].item(), gt_btns[i].item())
                       for i in range(12) if btn_probs[i].item() > 0.1 or gt_btns[i].item() > 0.5]
        if active_btns:
            print(f"    Active buttons: " +
                  " ".join(f"{n}={p:.2f}(gt={g:.0f})" for n, p, g in active_btns))
        else:
            gt_active = [btn_names[i] for i in range(12) if gt_btns[i].item() > 0.5]
            print(f"    No buttons > 0.1 confidence. GT active: {gt_active if gt_active else 'none'}")

    all_probs = torch.stack(all_btn_probs)
    all_gt = torch.stack(all_btn_gt)

    mean_probs = all_probs.mean(dim=0)
    print(f"\n  SUMMARY:")
    print(f"    Stick accuracy: {stick_correct}/{stick_total} = {100*stick_correct/max(stick_total,1):.1f}%")
    print(f"    Shoulder accuracy: {shoulder_correct}/{shoulder_total} = {100*shoulder_correct/max(shoulder_total,1):.1f}%")
    print(f"    Mean button probs: " +
          " ".join(f"{btn_names[i]}={mean_probs[i]:.3f}" for i in range(12)))

    top3_idx = mean_probs.topk(3).indices
    print(f"    Top-3 avg button confidence: " +
          " ".join(f"{btn_names[i]}={mean_probs[i]:.3f}" for i in top3_idx))

    preds = (all_probs > 0.5).float()
    tp = (preds * all_gt).sum().item()
    fp = (preds * (1 - all_gt)).sum().item()
    fn = ((1 - preds) * all_gt).sum().item()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    print(f"    Button F1 (5 samples): {100*f1:.1f}% (P={100*precision:.1f}% R={100*recall:.1f}%)")


def main():
    print("=" * 70)
    print("CLOSED-LOOP INFERENCE TEST")
    print("=" * 70)

    print(f"\nClosed-loop checkpoint: {CL_CKPT}")
    cl_model, cl_cfg, _ = load_model(CL_CKPT)
    print(f"  {sum(p.numel() for p in cl_model.parameters()):,} params, seq_len={cl_cfg.max_seq_len}")

    run_test(cl_model, cl_cfg, cl_norm_stats, cl_cat_maps, "CLOSED-LOOP MODEL (trained on 20 Falco-FD games)")

    if FULL_CKPT.exists():
        print(f"\nFull-data checkpoint: {FULL_CKPT}")

        with open(Path("data/full/norm_stats.json")) as fh:
            full_norm = json.load(fh)
        with open(Path("data/full/cat_maps.json")) as fh:
            raw = json.load(fh)
            full_cats = {col: {int(k): v for k, v in m.items()} for col, m in raw.items()}

        full_model, full_cfg, _ = load_model(FULL_CKPT)
        print(f"  {sum(p.numel() for p in full_model.parameters()):,} params, seq_len={full_cfg.max_seq_len}")
        run_test(full_model, full_cfg, full_norm, full_cats, "FULL-DATA MODEL (for comparison)")

    print(f"\n{'=' * 70}")
    print("DONE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
