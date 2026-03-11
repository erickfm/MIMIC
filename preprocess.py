#!/usr/bin/env python3
# preprocess.py
# ---------------------------------------------------------------------------
# Metadata-only preprocessing: streams raw parquets to compute three JSON
# files that the StreamingMeleeDataset needs at training time.
#
# No .pt files are produced -- the dataset reads raw parquets on-the-fly.
#
# Usage:
#   python3 preprocess.py --data-dir data/subset
#   python3 preprocess.py --data-dir data/full
# ---------------------------------------------------------------------------

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import pandas as pd

import features as F


def main(data_dir: str) -> None:
    src = Path(data_dir)
    parquets = sorted(src.glob("*.parquet"))
    if not parquets:
        raise RuntimeError(f"No .parquet files in {src}")
    print(f"Found {len(parquets)} parquet files in {src}")

    fg = F.build_feature_groups()
    categorical_cols = F.get_categorical_cols(fg)
    norm_cols = F.get_norm_cols(fg)
    t0 = time.time()

    # -- Pass 1: scan dynamic categoricals (ports, costumes) -----------------
    print("Pass 1: scanning categorical values ...", end=" ", flush=True)
    cat_maps = F.build_categorical_mappings_streaming(parquets, categorical_cols)
    with open(src / "cat_maps.json", "w") as fh:
        json.dump(cat_maps, fh)
    print(f"done. ({len(cat_maps)} dynamic columns, {time.time()-t0:.0f}s)")

    # -- Pass 2: preprocess each file, accumulate norm stats, record sizes ---
    print("Pass 2: computing normalization stats + file index ...", flush=True)
    col_sum: Dict[str, float] = {}
    col_sq:  Dict[str, float] = {}
    col_n:   Dict[str, int]   = {}
    file_index: Dict[str, int] = {}
    t1 = time.time()

    for i, pf in enumerate(parquets, 1):
        df = pd.read_parquet(pf)
        df = df[df["frame"] >= 0].reset_index(drop=True)
        n_frames = len(df)
        if n_frames < 2:
            continue

        df = F.preprocess_df(df, categorical_cols, cat_maps)
        F.update_norm_accumulators(df, norm_cols, col_sum, col_sq, col_n)
        file_index[pf.name] = n_frames

        if i % 500 == 0 or i == len(parquets):
            print(f"  [{i}/{len(parquets)}] ({time.time()-t1:.0f}s)", flush=True)

    norm_stats = F.finalize_norm_stats(norm_cols, col_sum, col_sq, col_n)

    with open(src / "norm_stats.json", "w") as fh:
        json.dump(norm_stats, fh)
    with open(src / "file_index.json", "w") as fh:
        json.dump(file_index, fh)

    elapsed = time.time() - t0
    print(f"\nDone. {len(file_index)} files indexed in {src}  ({elapsed:.0f}s total)")
    print(f"  norm_stats.json  ({len(norm_stats)} columns)")
    print(f"  cat_maps.json    ({len(cat_maps)} dynamic columns)")
    print(f"  file_index.json  ({len(file_index)} files)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute training metadata from raw parquets")
    parser.add_argument("--data-dir", required=True, help="Directory containing .parquet files")
    args = parser.parse_args()
    main(args.data_dir)
