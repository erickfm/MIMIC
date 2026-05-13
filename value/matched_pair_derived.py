"""Matched-pair discovery with engineered (derived) features.

Same as value/matched_pair.py but adds the 37 derived features from
value/derived_features.py (combo state, neutral indicators, cumulative
counts, frame timers) to the comparison.

For each sample, we compute the derived feature value at the last frame
of the window. Then within each macro-matched bin (same stocks+percent
buckets), we compute the z-normalized Δ between high-V and low-V groups
for every feature, raw + derived.

Discovery output: ranked list with engineered features competing for
attention alongside raw shard features.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from value.analyze import NUMERIC_COLS, FLAG_COLS
from value.analyze_ckpt import load_value_model, run_inference, BC_ENCODER_KEYS
from value.dataset import STOCK_COL, PERCENT_COL, compute_game_outcomes
from value.derived_features import (
    compute_derived_features, derived_feature_names,
)
from value.matched_pair import stock_bucket, percent_bucket


def sample_with_derived(
    data_dir: Path,
    split: str,
    n_target: int,
    window: int,
    is_windowed: bool,
    state_keys: list,
    seed: int = 0,
):
    """Sample N frames + their derived features. Computes derived per shard."""
    with open(data_dir / "tensor_manifest.json") as f:
        manifest = json.load(f)
    shard_files = [data_dir / n for n in manifest[f"{split}_shards"]]
    rng = np.random.default_rng(seed)
    W = window if is_windowed else 1
    n_per_shard = max(1, n_target // len(shard_files))

    batches = []
    outcomes_all = []
    positions_all = []
    trivial_X_all = []
    derived_keys = derived_feature_names()
    derived_at_last_frame = {k: [] for k in derived_keys}
    # Also collect raw shard features at last frame for matched-pair comparison
    self_num_at_last = []
    opp_num_at_last = []
    self_flg_at_last = []
    opp_flg_at_last = []
    self_action_at_last = []
    opp_action_at_last = []

    for shard_path in shard_files:
        shard = torch.load(shard_path, map_location="cpu",
                           weights_only=False, mmap=True)
        states = shard["states"]
        offsets = shard["offsets"]
        n_games = int(shard["n_games"])
        if n_games == 0:
            continue

        self_stock = states["self_numeric"][:, STOCK_COL]
        opp_stock = states["opp_numeric"][:, STOCK_COL]
        self_pct = states["self_numeric"][:, PERCENT_COL]
        opp_pct = states["opp_numeric"][:, PERCENT_COL]
        gi = compute_game_outcomes(
            self_stock, opp_stock, offsets, n_games,
            self_pct_col=self_pct, opp_pct_col=opp_pct,
        )

        # Compute derived features once for the entire shard
        derived = compute_derived_features(states, offsets, n_games)

        for _ in range(n_per_shard):
            g = int(rng.integers(0, n_games))
            s, e, outcome = gi[g]
            length = e - s
            if length < W + 1:
                continue
            offset = int(rng.integers(0, length - W + 1))
            abs_start = s + offset
            sample = {}
            for k in state_keys:
                if k not in states:
                    sample = None
                    break
                sample[k] = states[k][abs_start: abs_start + W]
            if sample is None:
                continue
            batches.append(sample)
            outcomes_all.append(outcome)
            last_abs = abs_start + W - 1
            positions_all.append((offset + W - 1) / length)
            trivial_X_all.append(np.array([
                self_stock[last_abs].item(), opp_stock[last_abs].item(),
                self_pct[last_abs].item(), opp_pct[last_abs].item(),
            ]))
            # Raw features at last frame
            self_num_at_last.append(states["self_numeric"][last_abs].numpy())
            opp_num_at_last.append(states["opp_numeric"][last_abs].numpy())
            self_flg_at_last.append(states["self_flags"][last_abs].numpy())
            opp_flg_at_last.append(states["opp_flags"][last_abs].numpy())
            self_action_at_last.append(int(states["self_action"][last_abs].item()))
            opp_action_at_last.append(int(states["opp_action"][last_abs].item()))
            # Derived features at last frame
            for k in derived_keys:
                derived_at_last_frame[k].append(float(derived[k][last_abs]))

    feats = {
        "self_numeric": np.stack(self_num_at_last),
        "opp_numeric": np.stack(opp_num_at_last),
        "self_flags": np.stack(self_flg_at_last),
        "opp_flags": np.stack(opp_flg_at_last),
        "self_action": np.array(self_action_at_last),
        "opp_action": np.array(opp_action_at_last),
    }
    derived_np = {k: np.array(v, dtype=np.float32)
                  for k, v in derived_at_last_frame.items()}
    return (batches, np.array(outcomes_all), np.array(positions_all),
            np.stack(trivial_X_all), feats, derived_np)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-dir", default="data/fox_all_v2")
    p.add_argument("--n-samples", type=int, default=50000)
    p.add_argument("--min-bin-n", type=int, default=100)
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
    print(f"sampling {args.n_samples} frames + computing derived features per shard...")
    batches, y, pos, trivial_X, feats, derived = sample_with_derived(
        Path(args.data_dir), "val", args.n_samples, W,
        is_windowed, state_keys, seed=0,
    )
    print(f"got {len(batches)} samples")
    print(f"running V(s) inference...")
    logits = run_inference(model, batches, state_keys, device, is_windowed)
    print("inference done.")

    # Bucket by macro state
    bins = defaultdict(list)
    for i in range(len(batches)):
        ss = stock_bucket(feats["self_numeric"][i, STOCK_COL])
        os = stock_bucket(feats["opp_numeric"][i, STOCK_COL])
        sp = percent_bucket(feats["self_numeric"][i, PERCENT_COL])
        op = percent_bucket(feats["opp_numeric"][i, PERCENT_COL])
        bins[(ss, os, sp, op)].append(i)
    bins = {k: v for k, v in bins.items() if len(v) >= args.min_bin_n}
    print(f"{len(bins)} macro bins kept (>= {args.min_bin_n} each); "
          f"total samples in retained bins: {sum(len(v) for v in bins.values())}")

    # Build the full feature list: raw + derived
    scalar_feats = []
    for side in ("self", "opp"):
        for j, c in enumerate(NUMERIC_COLS):
            scalar_feats.append((f"{side}_{c}", "numeric", side, j))
        for j, c in enumerate(FLAG_COLS):
            scalar_feats.append((f"{side}_{c}", "flags", side, j))
    # Add all derived features
    for name in derived_feature_names():
        scalar_feats.append((name, "derived", None, None))

    # Compute per-feature std for z-normalization
    feature_std = {}
    for name, kind, side, j in scalar_feats:
        if kind == "numeric":
            v = feats[f"{side}_numeric"][:, j]
        elif kind == "flags":
            v = feats[f"{side}_flags"][:, j] * 2.0 - 1.0
        elif kind == "derived":
            v = derived[name]
        feature_std[name] = float(np.std(v)) + 1e-9

    # Per-bin matched-pair Δ aggregation
    feature_diffs = defaultdict(list)
    for macro_key, idxs in bins.items():
        bin_logits = logits[idxs]
        n = len(idxs)
        n_q = max(10, int(n * args.quantile))
        order = np.argsort(bin_logits)
        low_idx = [idxs[i] for i in order[:n_q]]
        high_idx = [idxs[i] for i in order[-n_q:]]

        for name, kind, side, j in scalar_feats:
            if kind == "numeric":
                hi = feats[f"{side}_numeric"][high_idx, j].mean()
                lo = feats[f"{side}_numeric"][low_idx, j].mean()
            elif kind == "flags":
                hi = feats[f"{side}_flags"][high_idx, j].mean() * 2.0 - 1.0
                lo = feats[f"{side}_flags"][low_idx, j].mean() * 2.0 - 1.0
            elif kind == "derived":
                hi = derived[name][high_idx].mean()
                lo = derived[name][low_idx].mean()
            feature_diffs[name].append((float(hi - lo), n))

    # Aggregate to ranking
    ranking = []
    for name, entries in feature_diffs.items():
        total_n = sum(e[1] for e in entries)
        wd = sum(e[0] * e[1] for e in entries) / total_n
        std = feature_std[name]
        z = wd / std
        is_derived = any(name == d for d in derived_feature_names())
        ranking.append({
            "feature": name,
            "kind": "derived" if is_derived else "raw",
            "z_diff": float(z),
            "raw_diff": float(wd),
            "feature_std": float(std),
            "n_total": int(total_n),
        })
    ranking.sort(key=lambda x: -abs(x["z_diff"]))

    print()
    print(f"top 30 features by |z_diff| (derived AND raw competing):")
    print(f"  {'feature':<35} {'kind':<8} {'z_diff':>9} {'raw_Δ':>10} "
          f"{'std':>10}")
    for r in ranking[:30]:
        print(f"  {r['feature']:<35} {r['kind']:<8} {r['z_diff']:>+9.4f} "
              f"{r['raw_diff']:>+10.4f} {r['feature_std']:>10.4f}")

    print()
    print(f"top 15 DERIVED features only:")
    derived_only = [r for r in ranking if r["kind"] == "derived"]
    for r in derived_only[:15]:
        print(f"  {r['feature']:<35} {r['z_diff']:>+9.4f} "
              f"{r['raw_diff']:>+10.4f}")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "ckpt": str(args.ckpt),
            "n_samples": int(len(batches)),
            "n_bins_kept": len(bins),
            "ranking": ranking,
        }, indent=2, default=str))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
