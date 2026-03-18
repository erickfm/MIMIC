#!/usr/bin/env python3
"""
Comprehensive offline evaluation of the closed-loop model.

Runs the model on ALL windows from the closed-loop data and computes
aggregate accuracy metrics, stratified by whether the ground truth
has active button presses. This tells us if the model can be confident
about buttons at the RIGHT times.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from typing import Dict

import numpy as np
import pandas as pd
import torch

import mimic.features as F
from mimic.model import FramePredictor, ModelConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CL_DATA = Path("data/closed_loop")
CL_CKPT = Path("checkpoints/closed-loop-overfit_step005000.pt")
FULL_CKPT = Path("checkpoints/cls-ar-ctx180-40k-s2_step040000.pt")


def load_model_and_meta(ckpt_path: Path, data_dir: Path):
    with open(data_dir / "norm_stats.json") as fh:
        norm_stats = json.load(fh)
    with open(data_dir / "cat_maps.json") as fh:
        raw = json.load(fh)
        cat_maps = {col: {int(k): v for k, v in m.items()} for col, m in raw.items()}
    with open(data_dir / "stick_clusters.json") as fh:
        sc = json.load(fh)
        stick_centers = np.array(sc["stick_centers"], dtype=np.float32)
        shoulder_centers = np.array(sc["shoulder_centers"], dtype=np.float32)

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    cfg = ModelConfig(**ckpt["config"])
    model = FramePredictor(cfg).to(DEVICE)
    sd = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(sd)
    model.eval()

    return model, cfg, norm_stats, cat_maps, stick_centers, shoulder_centers


def evaluate_model(model, cfg, norm_stats, cat_maps, stick_centers, shoulder_centers, label: str):
    fg = F.build_feature_groups(no_opp_inputs=cfg.no_opp_inputs)
    cat_cols = F.get_categorical_cols(fg)
    seq_len = cfg.max_seq_len
    batch_size = 32

    parquets = sorted(CL_DATA.glob("*.parquet"))

    all_main_correct = []
    all_shoulder_correct = []
    all_cdir_correct = []
    all_btn_tp = 0
    all_btn_fp = 0
    all_btn_fn = 0
    active_btn_tp = 0
    active_btn_fp = 0
    active_btn_fn = 0
    active_count = 0
    inactive_count = 0
    btn_confidence_when_active = []
    btn_confidence_when_inactive = []

    btn_names = ["A", "B", "X", "Y", "Z", "L", "R", "START",
                 "D_UP", "D_DN", "D_LT", "D_RT"]

    n_windows_eval = 0
    for pq in parquets:
        df = pd.read_parquet(pq)
        df = df[df["frame"] >= 0].reset_index(drop=True)
        if len(df) < seq_len + 1:
            continue

        df = F.preprocess_df(df, cat_cols, cat_maps)
        targets = F.build_targets_batch(df, norm_stats,
                                        stick_centers=stick_centers,
                                        shoulder_centers=shoulder_centers)
        F.apply_normalization(df, norm_stats)
        state = F.df_to_state_tensors(df, fg)

        n_frames = len(df)
        starts = list(range(0, n_frames - seq_len, seq_len // 2))[:20]

        for start in starts:
            end = start + seq_len
            window = {k: v[start:end].unsqueeze(0) for k, v in state.items()}
            target_frame = end - 1

            gt_main = targets["main_cluster"][target_frame].item()
            gt_l = targets["l_bin"][target_frame].item()
            gt_r = targets["r_bin"][target_frame].item()
            gt_cdir = torch.argmax(targets["c_dir"][target_frame]).item()
            gt_btns = targets["btns"][target_frame]

            with torch.no_grad():
                inp = {k: v.to(DEVICE) for k, v in window.items()}
                out = model(inp)
                pred = {k: v[:, -1].cpu().squeeze(0) for k, v in out.items()}

            pred_main = int(torch.argmax(pred["main_xy"]))
            pred_l = int(torch.argmax(pred["L_val"]))
            pred_r = int(torch.argmax(pred["R_val"]))
            pred_cdir = int(torch.argmax(pred["c_dir_logits"]))
            btn_probs = torch.sigmoid(pred["btn_logits"])
            pred_btns = (btn_probs > 0.5).float()

            all_main_correct.append(int(pred_main == gt_main))
            all_shoulder_correct.append(int(pred_l == gt_l))
            all_shoulder_correct.append(int(pred_r == gt_r))
            all_cdir_correct.append(int(pred_cdir == gt_cdir))

            tp = (pred_btns * gt_btns).sum().item()
            fp = (pred_btns * (1 - gt_btns)).sum().item()
            fn = ((1 - pred_btns) * gt_btns).sum().item()
            all_btn_tp += tp
            all_btn_fp += fp
            all_btn_fn += fn

            any_active = gt_btns.sum().item() > 0
            if any_active:
                active_count += 1
                active_btn_tp += tp
                active_btn_fp += fp
                active_btn_fn += fn
                btn_confidence_when_active.append(btn_probs.max().item())
            else:
                inactive_count += 1
                btn_confidence_when_inactive.append(btn_probs.max().item())

            n_windows_eval += 1

    print(f"\n{'=' * 70}")
    print(f"  {label} -- {n_windows_eval} windows from {len(parquets)} games")
    print(f"{'=' * 70}")

    main_acc = 100 * np.mean(all_main_correct)
    shoulder_acc = 100 * np.mean(all_shoulder_correct)
    cdir_acc = 100 * np.mean(all_cdir_correct)

    print(f"\n  STICK CLUSTER ACCURACY:  {main_acc:.1f}%")
    print(f"  SHOULDER BIN ACCURACY:   {shoulder_acc:.1f}%")
    print(f"  C-DIR ACCURACY:          {cdir_acc:.1f}%")

    prec = all_btn_tp / max(all_btn_tp + all_btn_fp, 1)
    recall = all_btn_tp / max(all_btn_tp + all_btn_fn, 1)
    f1 = 2 * prec * recall / max(prec + recall, 1e-8)
    print(f"\n  BUTTON F1 (all frames):  {100*f1:.1f}%  (P={100*prec:.1f}% R={100*recall:.1f}%)")
    print(f"    Frames with buttons: {active_count} ({100*active_count/max(active_count+inactive_count,1):.1f}%)")
    print(f"    Frames without: {inactive_count}")

    if active_count > 0:
        act_prec = active_btn_tp / max(active_btn_tp + active_btn_fp, 1)
        act_recall = active_btn_tp / max(active_btn_tp + active_btn_fn, 1)
        act_f1 = 2 * act_prec * act_recall / max(act_prec + act_recall, 1e-8)
        print(f"\n  BUTTON F1 (active frames only): {100*act_f1:.1f}%  (P={100*act_prec:.1f}% R={100*act_recall:.1f}%)")
        avg_conf_active = np.mean(btn_confidence_when_active)
        print(f"  Avg max btn confidence when buttons ACTIVE: {avg_conf_active:.3f}")

    if inactive_count > 0:
        avg_conf_inactive = np.mean(btn_confidence_when_inactive)
        print(f"  Avg max btn confidence when buttons INACTIVE: {avg_conf_inactive:.3f}")


def main():
    print("Comprehensive closed-loop evaluation")
    print("=" * 70)

    print("\nLoading closed-loop model...")
    cl = load_model_and_meta(CL_CKPT, CL_DATA)
    evaluate_model(*cl, "CLOSED-LOOP MODEL (overfit on 20 Falco-FD games)")

    if FULL_CKPT.exists():
        print("\nLoading full-data model...")
        full = load_model_and_meta(FULL_CKPT, Path("data/full"))
        evaluate_model(*full, "FULL-DATA MODEL (for comparison)")

    print(f"\n{'=' * 70}")
    print("DONE")


if __name__ == "__main__":
    main()
