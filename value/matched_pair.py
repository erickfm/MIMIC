"""Matched-pair discovery: find features that predict V *conditional* on
the macro state (stocks + percent).

Approach:
  1. Run the existing Markov baseline on ~50K random val frames.
  2. Bin frames by (self_stock_int, opp_stock_int, self_pct_bucket,
     opp_pct_bucket). Within each bin, frames have ~identical macro state.
  3. Within each bin, find the frames with the highest predicted V and
     the lowest predicted V — both are "self looks like they're winning
     vs losing" judgments the model makes *given the same macro state*.
  4. Compute, for each non-macro feature, the mean difference between
     high-V and low-V frames within each bin. Aggregate across bins.

The features with the largest within-bin |Δ(feature)| between high-V and
low-V frames are the features V is using to make conditional judgments —
discovery candidates that are NOT just stocks/percent.

This complements the gate-ranking approach: gates rank features by how
much the model attends to them globally; this ranks features by how
much they discriminate "good states from bad states given the macro."
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from value.analyze import NUMERIC_COLS, FLAG_COLS
from value.analyze_ckpt import load_value_model, sample_for_inference, \
    run_inference, BC_ENCODER_KEYS
from value.dataset import STOCK_COL, PERCENT_COL


def stock_bucket(s: float) -> int:
    """Normalized stock in {-1, -0.5, 0, 0.5, 1} → int 0..4."""
    return int(round((s + 1.0) * 2))


def percent_bucket(p: float) -> int:
    """Normalized percent into 5 buckets across the seen range."""
    # mimic_norm uses standardize for percent, so values are roughly
    # mean=0 std=1 with tails. Buckets at -1.5, -0.5, 0.5, 1.5 standard
    # deviations.
    if p < -1.5:
        return 0
    if p < -0.5:
        return 1
    if p < 0.5:
        return 2
    if p < 1.5:
        return 3
    return 4


def feature_dict_per_frame(feats: dict, idx: int) -> Dict[str, float]:
    """Extract named scalar features for one frame (idx into the sample arrays)."""
    out = {}
    for side in ("self", "opp"):
        for j, c in enumerate(NUMERIC_COLS):
            out[f"{side}_{c}"] = float(feats[f"{side}_numeric"][idx, j])
        for j, c in enumerate(FLAG_COLS):
            out[f"{side}_{c}"] = float(feats[f"{side}_flags"][idx, j]) * 2.0 - 1.0
        out[f"{side}_action_elapsed"] = float(feats[f"{side}_action_elapsed"][idx])
        # Action ID (categorical — treat as just an int per frame for now)
        out[f"{side}_action_id"] = int(feats[f"{side}_action"][idx])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-dir", default="data/fox_all_v2")
    p.add_argument("--split", default="val")
    p.add_argument("--n-samples", type=int, default=100000)
    p.add_argument("--min-bin-n", type=int, default=200,
                   help="Skip macro bins with fewer than N samples")
    p.add_argument("--quantile", type=float, default=0.1,
                   help="High-V = top Q quantile within bin; Low-V = bottom Q")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt_args, is_windowed = load_value_model(Path(args.ckpt), device)

    if is_windowed:
        from value.dataset import VALUE_ENCODER_KEYS
        state_keys = VALUE_ENCODER_KEYS
    else:
        state_keys = BC_ENCODER_KEYS

    # Sample frames matching training distribution + run inference
    W = 60 if is_windowed else 1
    batches, y, pos, trivial_X = sample_for_inference(
        Path(args.data_dir), args.split, args.n_samples, W,
        is_windowed, state_keys, seed=0,
    )
    print(f"got {len(batches)} samples; running inference...")
    logits = run_inference(model, batches, state_keys, device, is_windowed)
    print(f"inference done.")

    # Also extract feature scalars at the *last frame* of each window
    # (the prediction's reference frame).
    print(f"extracting per-frame features...")
    feats_collected: Dict[str, np.ndarray] = {}
    # Re-pull the last-frame features from each sampled batch (each batch
    # is a single sample whose state dict has keys with shape (W, ...))
    self_num = []
    opp_num = []
    self_flg = []
    opp_flg = []
    self_action = []
    opp_action = []
    self_ae = []
    opp_ae = []
    for s in batches:
        self_num.append(s["self_numeric"][-1].numpy())
        opp_num.append(s["opp_numeric"][-1].numpy())
        self_flg.append(s["self_flags"][-1].numpy())
        opp_flg.append(s["opp_flags"][-1].numpy())
        self_action.append(int(s["self_action"][-1].item()))
        opp_action.append(int(s["opp_action"][-1].item()))
        if "self_action_elapsed" in s:
            self_ae.append(float(s["self_action_elapsed"][-1].item()))
            opp_ae.append(float(s["opp_action_elapsed"][-1].item()))
        else:
            self_ae.append(0.0); opp_ae.append(0.0)
    feats = {
        "self_numeric": np.stack(self_num),
        "opp_numeric": np.stack(opp_num),
        "self_flags": np.stack(self_flg),
        "opp_flags": np.stack(opp_flg),
        "self_action": np.array(self_action),
        "opp_action": np.array(opp_action),
        "self_action_elapsed": np.array(self_ae),
        "opp_action_elapsed": np.array(opp_ae),
    }

    # Bucket by macro state
    print(f"bucketing by macro state...")
    bins: Dict[tuple, List[int]] = defaultdict(list)
    for i in range(len(batches)):
        ss = stock_bucket(feats["self_numeric"][i, STOCK_COL])
        os = stock_bucket(feats["opp_numeric"][i, STOCK_COL])
        sp = percent_bucket(feats["self_numeric"][i, PERCENT_COL])
        op = percent_bucket(feats["opp_numeric"][i, PERCENT_COL])
        bins[(ss, os, sp, op)].append(i)

    print(f"got {len(bins)} macro bins; "
          f"keeping bins with >= {args.min_bin_n} samples")
    bins = {k: v for k, v in bins.items() if len(v) >= args.min_bin_n}
    print(f"after filter: {len(bins)} bins, "
          f"total samples in retained bins: {sum(len(v) for v in bins.values())}")

    # For each bin, compute high-V vs low-V quantile groups and per-feature diff
    print(f"computing per-bin matched-pair feature differences...")
    feature_diffs = defaultdict(list)  # feature_name -> list of (within_bin_diff, bin_n)

    # All scalar features
    scalar_feats = []
    for side in ("self", "opp"):
        for j, c in enumerate(NUMERIC_COLS):
            scalar_feats.append((f"{side}_{c}", "numeric", side, j))
        for j, c in enumerate(FLAG_COLS):
            scalar_feats.append((f"{side}_{c}", "flags", side, j))
        scalar_feats.append((f"{side}_action_elapsed", "action_elapsed", side, None))

    for macro_key, idxs in bins.items():
        bin_logits = logits[idxs]
        n = len(idxs)
        # Top/bottom quantile within bin
        n_q = max(10, int(n * args.quantile))
        order = np.argsort(bin_logits)
        low_idx = [idxs[i] for i in order[:n_q]]
        high_idx = [idxs[i] for i in order[-n_q:]]
        bin_logit_gap = float(bin_logits[order[-n_q:]].mean() - bin_logits[order[:n_q]].mean())

        # For each feature, compute mean(high) - mean(low)
        for name, kind, side, j in scalar_feats:
            if kind == "numeric":
                hi = feats[f"{side}_numeric"][high_idx, j].mean()
                lo = feats[f"{side}_numeric"][low_idx, j].mean()
            elif kind == "flags":
                hi = feats[f"{side}_flags"][high_idx, j].mean() * 2.0 - 1.0
                lo = feats[f"{side}_flags"][low_idx, j].mean() * 2.0 - 1.0
            elif kind == "action_elapsed":
                hi = feats[f"{side}_action_elapsed"][high_idx].mean()
                lo = feats[f"{side}_action_elapsed"][low_idx].mean()
            feature_diffs[name].append({
                "bin": macro_key, "n": n, "diff": float(hi - lo),
                "logit_gap": bin_logit_gap,
            })

    # Compute per-feature std over ALL samples — used to z-score the Δ so
    # features on different scales (m/s vs normalized vs binary) are
    # comparable. Without this, raw-units features dominate trivially.
    feature_std: Dict[str, float] = {}
    for name, kind, side, j in scalar_feats:
        if kind == "numeric":
            v = feats[f"{side}_numeric"][:, j]
        elif kind == "flags":
            v = feats[f"{side}_flags"][:, j] * 2.0 - 1.0
        else:
            v = feats[f"{side}_action_elapsed"]
        feature_std[name] = float(np.std(v)) + 1e-9

    # Aggregate: for each feature, compute n-weighted mean of within-bin diff
    print(f"aggregating...")
    ranking = []
    for name, entries in feature_diffs.items():
        total_n = sum(e["n"] for e in entries)
        weighted_diff = sum(e["diff"] * e["n"] for e in entries) / total_n
        weighted_abs = sum(abs(e["diff"]) * e["n"] for e in entries) / total_n
        std = feature_std[name]
        # z-scored difference: how many population stds does the
        # high-V vs low-V Δ cover? Comparable across feature scales.
        z_diff = weighted_diff / std
        ranking.append({
            "feature": name,
            "weighted_diff": float(weighted_diff),
            "weighted_abs_diff": float(weighted_abs),
            "feature_std": float(std),
            "z_diff": float(z_diff),
            "n_bins": len(entries),
            "n_total": int(total_n),
        })
    ranking.sort(key=lambda x: -abs(x["z_diff"]))

    print()
    print(f"top 25 features by within-macro-bin |z-Δ|(high-V) - (low-V):")
    print(f"  {'feature':<28} {'z_Δ':>10} {'Δ (raw)':>12} "
          f"{'feat_std':>10} {'n_bins':>7}")
    for r in ranking[:25]:
        print(f"  {r['feature']:<28} {r['z_diff']:>+10.4f} "
              f"{r['weighted_diff']:>+12.4f} "
              f"{r['feature_std']:>10.4f} "
              f"{r['n_bins']:>7}")

    print()
    print(f"bottom 10 features (smallest |z-Δ|):")
    for r in ranking[-10:]:
        print(f"  {r['feature']:<28} {r['z_diff']:>+10.4f} "
              f"{r['weighted_diff']:>+12.4f} "
              f"{r['feature_std']:>10.4f} "
              f"{r['n_bins']:>7}")

    # Categorical action analysis: within macro-matched bins, which action
    # states appear more frequently in high-V vs low-V frames?
    # Build action -> {high_count, low_count, total_count}
    print()
    print(f"action-state analysis (within macro-matched bins):")
    action_stats = defaultdict(lambda: {"hi": 0, "lo": 0, "tot": 0})
    for macro_key, idxs in bins.items():
        bin_logits = logits[idxs]
        n = len(idxs)
        n_q = max(10, int(n * args.quantile))
        order = np.argsort(bin_logits)
        low_idx = [idxs[i] for i in order[:n_q]]
        high_idx = [idxs[i] for i in order[-n_q:]]
        # opp_action distribution
        for ix in high_idx:
            key = ("opp_action", int(feats["opp_action"][ix]))
            action_stats[key]["hi"] += 1
        for ix in low_idx:
            key = ("opp_action", int(feats["opp_action"][ix]))
            action_stats[key]["lo"] += 1
        for ix in idxs:
            key = ("opp_action", int(feats["opp_action"][ix]))
            action_stats[key]["tot"] += 1
        # self_action distribution
        for ix in high_idx:
            key = ("self_action", int(feats["self_action"][ix]))
            action_stats[key]["hi"] += 1
        for ix in low_idx:
            key = ("self_action", int(feats["self_action"][ix]))
            action_stats[key]["lo"] += 1
        for ix in idxs:
            key = ("self_action", int(feats["self_action"][ix]))
            action_stats[key]["tot"] += 1

    # Rank actions by (P(action | high-V) - P(action | low-V)) ratio
    # Filter to actions with total occurrences >= 100 (statistical power)
    total_hi = sum(s["hi"] for s in action_stats.values()) // 2  # /2 because each frame contributes to both self and opp
    total_lo = sum(s["lo"] for s in action_stats.values()) // 2
    action_diffs = []
    for (side_name, action_id), s in action_stats.items():
        if s["tot"] < 100:
            continue
        # Compute P(action | high) and P(action | low) for this side
        # We need per-side totals — recompute
        p_hi = s["hi"] / max(1, total_hi)
        p_lo = s["lo"] / max(1, total_lo)
        action_diffs.append({
            "side": side_name,
            "action_id": int(action_id),
            "p_high_V": float(p_hi),
            "p_low_V": float(p_lo),
            "p_diff": float(p_hi - p_lo),
            "log_ratio": float(np.log((p_hi + 1e-9) / (p_lo + 1e-9))),
            "n_total": int(s["tot"]),
        })
    action_diffs.sort(key=lambda x: -abs(x["log_ratio"]))

    # Also try to resolve action IDs to names if cat_maps available
    action_name_map = {}
    try:
        with open("data/fox_all_v2/cat_maps.json") as f:
            cat_maps = json.load(f)
        # cat_maps["action"] maps action_name -> id; we need the inverse
        if "action" in cat_maps:
            action_name_map = {v: k for k, v in cat_maps["action"].items()}
    except Exception:
        pass

    print(f"  top 25 actions by |log P(high)/P(low)| (filter n>=100):")
    print(f"  {'side':<12} {'a_id':>5} {'name':<32} "
          f"{'P(hi)':>8} {'P(lo)':>8} {'log_r':>8} {'n':>6}")
    for r in action_diffs[:25]:
        name = action_name_map.get(r["action_id"], f"act_{r['action_id']}")
        print(f"  {r['side']:<12} {r['action_id']:>5} {name:<32} "
              f"{r['p_high_V']:>8.4f} {r['p_low_V']:>8.4f} "
              f"{r['log_ratio']:>+8.3f} {r['n_total']:>6}")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "ckpt": str(args.ckpt),
            "n_samples": len(batches),
            "n_bins_kept": len(bins),
            "ranking": ranking,
            "action_ranking": action_diffs[:100],
        }, indent=2, default=str))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
