"""Pre-tensorize a parquet dataset into .pt shards for fast DDP training.

Usage:
    python3 tensorize.py --data-dir data/wavedash_v2 --n-train-shards 64 --n-val-shards 8
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd
import torch

import features as F
from features import load_cluster_centers


def tensorize(
    data_dir: str,
    n_train_shards: int = 64,
    n_val_shards: int = 8,
    seq_len: int = 60,
    reaction_delay: int = 1,
    val_frac: float = 0.1,
    no_opp_inputs: bool = False,
    no_self_inputs: bool = False,
    seed: int = 42,
):
    data_dir = Path(data_dir)

    with open(data_dir / "norm_stats.json") as f:
        norm_stats = json.load(f)
    with open(data_dir / "cat_maps.json") as f:
        raw = json.load(f)
        cat_maps = {col: {int(k): v for k, v in m.items()} for col, m in raw.items()}
    with open(data_dir / "file_index.json") as f:
        file_index = json.load(f)

    stick_centers, shoulder_centers = load_cluster_centers(data_dir)
    fg = F.build_feature_groups(no_opp_inputs=no_opp_inputs, no_self_inputs=no_self_inputs)
    categorical_cols = F.get_categorical_cols(fg)

    all_names = sorted(file_index.keys())
    rng = random.Random(seed)
    rng.shuffle(all_names)
    n_val = int(len(all_names) * val_frac)
    val_names = all_names[:n_val] if n_val > 0 else []
    train_names = all_names[n_val:] if n_val > 0 else all_names

    W, R = seq_len, reaction_delay

    def process_files(names):
        windows = []
        for name in names:
            path = data_dir / name
            df = pd.read_parquet(path)
            df = df[df["frame"] >= 0].reset_index(drop=True)
            if len(df) < 2:
                continue
            df = F.preprocess_df(df, categorical_cols, cat_maps)
            F.apply_normalization(df, norm_stats)
            state = F.df_to_state_tensors(df, fg)
            targets = F.build_targets_batch(
                df, norm_stats,
                stick_centers=stick_centers,
                shoulder_centers=shoulder_centers,
            )
            n_frames = len(df)
            max_start = n_frames - W - R
            for s in range(max_start + 1):
                ws = {k: v[s : s + W] for k, v in state.items()}
                wt = {k: v[s + R : s + W + R] for k, v in targets.items()}
                windows.append((ws, wt))
        return windows

    def save_shards(windows, prefix, n_shards):
        rng_s = random.Random(seed)
        rng_s.shuffle(windows)
        buckets = [[] for _ in range(n_shards)]
        for i, w in enumerate(windows):
            buckets[i % n_shards].append(w)

        shard_files = []
        for k in range(n_shards):
            if not buckets[k]:
                continue
            states, targets = {}, {}
            for key in buckets[k][0][0]:
                states[key] = torch.stack([w[0][key] for w in buckets[k]])
            for key in buckets[k][0][1]:
                targets[key] = torch.stack([w[1][key] for w in buckets[k]])
            fname = f"{prefix}_shard_{k:03d}.pt"
            torch.save({"states": states, "targets": targets, "n": len(buckets[k])}, data_dir / fname)
            shard_files.append(fname)
            print(f"  {fname}: {len(buckets[k])} windows")
        return shard_files

    print(f"Processing {len(train_names)} train file(s)...")
    train_windows = process_files(train_names)
    print(f"  {len(train_windows)} train windows")

    print(f"Processing {len(val_names)} val file(s)...")
    val_windows = process_files(val_names)
    if not val_windows and train_windows:
        print("  0 val files -> reusing train windows for val (overfit mode)")
        val_windows = train_windows
    print(f"  {len(val_windows)} val windows")

    print(f"\nSaving {n_train_shards} train shards...")
    train_files = save_shards(train_windows, "train", n_train_shards)
    print(f"Saving {n_val_shards} val shards...")
    val_files = save_shards(val_windows, "val", n_val_shards)

    meta = {
        "train_shards": train_files,
        "val_shards": val_files,
        "n_train_windows": len(train_windows),
        "n_val_windows": len(val_windows),
        "seq_len": seq_len,
        "reaction_delay": reaction_delay,
        "no_opp_inputs": no_opp_inputs,
        "no_self_inputs": no_self_inputs,
    }
    with open(data_dir / "tensor_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    train_size = sum((data_dir / f).stat().st_size for f in train_files)
    val_size = sum((data_dir / f).stat().st_size for f in val_files)
    print(f"\nDone. Train: {len(train_files)} shards ({train_size / 1e6:.1f} MB), "
          f"Val: {len(val_files)} shards ({val_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Pre-tensorize dataset for fast DDP training")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--n-train-shards", type=int, default=64)
    p.add_argument("--n-val-shards", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=60)
    p.add_argument("--reaction-delay", type=int, default=1)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--no-opp-inputs", action="store_true")
    p.add_argument("--no-self-inputs", action="store_true")
    args = p.parse_args()
    tensorize(**vars(args))
