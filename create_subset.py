#!/usr/bin/env python3
"""Create a symlinked data subset directory for controlled data scaling experiments.

Reads file_index.json from the source data directory, randomly selects N% of files
with a fixed seed for reproducibility, and creates a target directory containing:
  - Symlinks to the selected parquet files
  - Copies of shared JSON metadata (cat_maps, norm_stats, stick_clusters)
  - A filtered file_index.json containing only the subset files

Usage:
    python3 create_subset.py --src data/full --pct 10 --dst data/full_10pct
    python3 create_subset.py --src data/full --pct 25 --dst data/full_25pct
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Create symlinked data subset")
    parser.add_argument("--src", type=str, required=True,
                        help="Source data directory (e.g. data/full)")
    parser.add_argument("--dst", type=str, required=True,
                        help="Destination directory for subset (e.g. data/full_10pct)")
    parser.add_argument("--pct", type=float, required=True,
                        help="Percentage of files to include (e.g. 10, 25, 50, 75)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    pct = args.pct / 100.0

    index_path = src / "file_index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"No file_index.json in {src}")

    with open(index_path) as f:
        file_index = json.load(f)

    # file_index is {filename: num_windows}
    all_filenames = sorted(file_index.keys())
    n_total = len(all_filenames)
    n_select = max(1, int(n_total * pct))

    rng = random.Random(args.seed)
    selected_names = sorted(rng.sample(all_filenames, n_select))

    print(f"Source: {src}")
    print(f"Destination: {dst}")
    print(f"Selecting {n_select}/{n_total} files ({args.pct}%, seed={args.seed})")

    dst.mkdir(parents=True, exist_ok=True)

    n_linked = 0
    for filename in selected_names:
        src_file = src / filename
        dst_file = dst / filename
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        if dst_file.exists() or dst_file.is_symlink():
            dst_file.unlink()
        os.symlink(src_file, dst_file)
        n_linked += 1

    subset_index = {name: file_index[name] for name in selected_names}

    with open(dst / "file_index.json", "w") as f:
        json.dump(subset_index, f)

    for meta_file in ("cat_maps.json", "norm_stats.json", "stick_clusters.json"):
        src_meta = src / meta_file
        if src_meta.exists():
            shutil.copy2(src_meta, dst / meta_file)
            print(f"  Copied {meta_file}")

    total_windows = sum(subset_index.values())
    print(f"Done: {n_linked} symlinks, {total_windows:,} windows")
    print(f"Subset directory: {dst}")


if __name__ == "__main__":
    main()
