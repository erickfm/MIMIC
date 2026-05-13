"""Check matched-pair findings robustness across stock-differential strata.

Question: do the top discriminative features hold in close games (|stock_diff|
≤ 1) AND blowouts (|stock_diff| ≥ 2)? If yes, they're robust VR candidates.
If features only matter in one regime, they're context-dependent and need
to be conditioned on game phase.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from value.analyze import NUMERIC_COLS, FLAG_COLS
from value.analyze_ckpt import load_value_model, sample_for_inference, \
    run_inference, BC_ENCODER_KEYS
from value.dataset import STOCK_COL
from value.matched_pair import stock_bucket, percent_bucket


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-dir", default="data/fox_all_v2")
    p.add_argument("--n-samples", type=int, default=80000)
    p.add_argument("--quantile", type=float, default=0.15)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt_args, is_windowed = load_value_model(Path(args.ckpt), device)

    if is_windowed:
        from value.dataset import VALUE_ENCODER_KEYS
        state_keys = VALUE_ENCODER_KEYS
    else:
        state_keys = BC_ENCODER_KEYS

    W = 60 if is_windowed else 1
    print(f"sampling {args.n_samples}...")
    batches, y, pos, trivial_X = sample_for_inference(
        Path(args.data_dir), "val", args.n_samples, W,
        is_windowed, state_keys, seed=42,
    )
    print(f"got {len(batches)} samples; running inference...")
    logits = run_inference(model, batches, state_keys, device, is_windowed)

    # Extract features
    self_num = np.stack([b["self_numeric"][-1].numpy() for b in batches])
    opp_num = np.stack([b["opp_numeric"][-1].numpy() for b in batches])
    self_flg = np.stack([b["self_flags"][-1].numpy() for b in batches])
    opp_flg = np.stack([b["opp_flags"][-1].numpy() for b in batches])

    # Stock differential = self_stock_bucket - opp_stock_bucket
    self_sb = np.array([stock_bucket(v) for v in self_num[:, STOCK_COL]])
    opp_sb = np.array([stock_bucket(v) for v in opp_num[:, STOCK_COL]])
    stock_diff = self_sb - opp_sb

    # 3 strata: |diff| <= 1 (close), |diff| == 2 (medium), |diff| >= 3 (blowout)
    strata = {
        "close (|diff|≤1)": np.abs(stock_diff) <= 1,
        "medium (|diff|=2)": np.abs(stock_diff) == 2,
        "blowout (|diff|≥3)": np.abs(stock_diff) >= 3,
    }

    scalar_feats = []
    for side in ("self", "opp"):
        for j, c in enumerate(NUMERIC_COLS):
            scalar_feats.append((f"{side}_{c}", "numeric", side, j))
        for j, c in enumerate(FLAG_COLS):
            scalar_feats.append((f"{side}_{c}", "flags", side, j))

    # Per-stratum, run the matched-pair within macro bins
    stratum_rankings = {}
    for stratum_name, stratum_mask in strata.items():
        print(f"\n=== stratum: {stratum_name}  n={stratum_mask.sum()} ===")
        s_logits = logits[stratum_mask]
        s_self_num = self_num[stratum_mask]
        s_opp_num = opp_num[stratum_mask]
        s_self_flg = self_flg[stratum_mask]
        s_opp_flg = opp_flg[stratum_mask]

        # Bin by macro state within stratum
        bins = defaultdict(list)
        for i in range(len(s_logits)):
            ss = stock_bucket(s_self_num[i, STOCK_COL])
            os = stock_bucket(s_opp_num[i, STOCK_COL])
            sp = percent_bucket(s_self_num[i, 2])
            op = percent_bucket(s_opp_num[i, 2])
            bins[(ss, os, sp, op)].append(i)
        bins = {k: v for k, v in bins.items() if len(v) >= 100}
        print(f"  {len(bins)} macro bins after filter (>=100 samples)")
        if not bins:
            continue

        # Per-feature aggregate diff
        feature_diffs = defaultdict(list)
        for macro_key, idxs in bins.items():
            bin_logits = s_logits[idxs]
            n = len(idxs)
            n_q = max(10, int(n * args.quantile))
            order = np.argsort(bin_logits)
            low_idx = [idxs[i] for i in order[:n_q]]
            high_idx = [idxs[i] for i in order[-n_q:]]

            for name, kind, side, j in scalar_feats:
                arr = s_self_num if side == "self" else s_opp_num
                flg_arr = s_self_flg if side == "self" else s_opp_flg
                if kind == "numeric":
                    hi = arr[high_idx, j].mean()
                    lo = arr[low_idx, j].mean()
                elif kind == "flags":
                    hi = flg_arr[high_idx, j].mean() * 2.0 - 1.0
                    lo = flg_arr[low_idx, j].mean() * 2.0 - 1.0
                feature_diffs[name].append((float(hi - lo), n))

        # Compute z-Δ per feature using global std (so values are comparable across strata)
        feature_std = {}
        for name, kind, side, j in scalar_feats:
            if kind == "numeric":
                v = (self_num if side == "self" else opp_num)[:, j]
            else:
                v = (self_flg if side == "self" else opp_flg)[:, j] * 2.0 - 1.0
            feature_std[name] = float(np.std(v)) + 1e-9

        ranking = []
        for name, entries in feature_diffs.items():
            total_n = sum(e[1] for e in entries)
            wd = sum(e[0] * e[1] for e in entries) / total_n
            std = feature_std[name]
            ranking.append({
                "feature": name,
                "z_diff": float(wd / std),
                "raw_diff": float(wd),
                "n_total": int(total_n),
            })
        ranking.sort(key=lambda x: -abs(x["z_diff"]))

        print(f"  top 12 features by |z_diff|:")
        for r in ranking[:12]:
            print(f"    {r['feature']:<28} {r['z_diff']:>+7.3f}  raw={r['raw_diff']:>+8.3f}")
        stratum_rankings[stratum_name] = ranking

    if args.out:
        Path(args.out).write_text(json.dumps({
            "ckpt": str(args.ckpt),
            "stratum_rankings": stratum_rankings,
        }, indent=2, default=str))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
