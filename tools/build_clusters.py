#!/usr/bin/env python3
"""Build stick cluster centers and shoulder bins from training data.

Samples parquet files, runs k-means++ on joint (main_x, main_y) positions,
clusters 1D shoulder trigger values, and saves results to stick_clusters.json.
Also generates a visualization for manual review.
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def sample_data(data_dir: Path, n_files: int = 1000, seed: int = 42):
    """Sample stick and shoulder values from random parquet files."""
    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        raise RuntimeError(f"No .parquet files in {data_dir}")

    rng = random.Random(seed)
    sample_files = rng.sample(files, min(n_files, len(files)))

    stick_xy = []
    shoulder_vals = []

    for i, f in enumerate(sample_files):
        df = pd.read_parquet(f, columns=[
            "self_main_x", "self_main_y", "self_l_shldr", "self_r_shldr",
        ])
        df = df.dropna()
        stick_xy.append(df[["self_main_x", "self_main_y"]].values.astype(np.float32))
        shoulder_vals.append(df[["self_l_shldr", "self_r_shldr"]].values.astype(np.float32).ravel())

        if (i + 1) % 200 == 0:
            print(f"  Loaded {i + 1}/{len(sample_files)} files ...", flush=True)

    stick_xy = np.concatenate(stick_xy)
    shoulder_vals = np.concatenate(shoulder_vals)

    print(f"  Total stick samples: {len(stick_xy):,}")
    print(f"  Total shoulder samples: {len(shoulder_vals):,}")
    return stick_xy, shoulder_vals


def build_stick_clusters(stick_xy: np.ndarray, n_clusters: int = 30,
                         seed: int = 42) -> np.ndarray:
    """K-means++ on joint (x, y) stick positions."""
    print(f"\nRunning k-means++ with {n_clusters} clusters on {len(stick_xy):,} points ...")
    km = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10,
                random_state=seed, max_iter=300)
    km.fit(stick_xy)
    centers = km.cluster_centers_
    print(f"  Inertia: {km.inertia_:.4f}")
    return centers


def build_shoulder_bins(shoulder_vals: np.ndarray, n_bins: int = 4,
                        seed: int = 42) -> np.ndarray:
    """K-means on 1D shoulder trigger values."""
    print(f"\nRunning k-means++ with {n_bins} bins on {len(shoulder_vals):,} shoulder values ...")
    vals_2d = shoulder_vals.reshape(-1, 1)
    km = KMeans(n_clusters=n_bins, init="k-means++", n_init=10,
                random_state=seed, max_iter=300)
    km.fit(vals_2d)
    centers = np.sort(km.cluster_centers_.ravel())
    print(f"  Shoulder bin centers: {centers.tolist()}")
    return centers


def visualize(stick_centers: np.ndarray, shoulder_centers: np.ndarray,
              stick_xy_sample: np.ndarray, out_path: Path):
    """Generate cluster visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.set_title(f"Main Stick Clusters (n={len(stick_centers)})")
    rng = np.random.RandomState(42)
    idx = rng.choice(len(stick_xy_sample), min(50000, len(stick_xy_sample)), replace=False)
    ax.scatter(stick_xy_sample[idx, 0], stick_xy_sample[idx, 1],
               alpha=0.02, s=1, c="gray", label="data")
    ax.scatter(stick_centers[:, 0], stick_centers[:, 1],
               c="red", s=80, marker="x", linewidths=2, zorder=10, label="cluster centers")
    for i, (cx, cy) in enumerate(stick_centers):
        ax.annotate(str(i), (cx, cy), fontsize=7, ha="center", va="bottom",
                    xytext=(0, 4), textcoords="offset points", color="red")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("main_x (0=left, 0.5=neutral, 1=right)")
    ax.set_ylabel("main_y (0=down, 0.5=neutral, 1=up)")
    ax.set_aspect("equal")
    ax.axhline(0.5, color="lightblue", lw=0.5, ls="--")
    ax.axvline(0.5, color="lightblue", lw=0.5, ls="--")
    circle = plt.Circle((0.5, 0.5), 0.5, fill=False, color="lightblue", lw=0.5, ls="--")
    ax.add_patch(circle)
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[1]
    ax.set_title(f"Shoulder Bins (n={len(shoulder_centers)})")
    ax.hist(np.clip(stick_xy_sample[:50000, 0], 0, 1), bins=50,
            alpha=0.0, label="_skip")
    shoulder_sample = np.random.RandomState(42).choice(
        np.concatenate([stick_xy_sample[:, 0]]),  # placeholder, we'll use actual shoulder data
        min(50000, len(stick_xy_sample)), replace=False)
    ax.barh(range(len(shoulder_centers)), [1]*len(shoulder_centers),
            left=[c - 0.02 for c in shoulder_centers], height=0.4, color="red", alpha=0.7)
    for i, c in enumerate(shoulder_centers):
        ax.text(c, i, f"  {c:.3f}", va="center", fontsize=9)
    ax.set_xlabel("Trigger value")
    ax.set_ylabel("Bin index")
    ax.set_xlim(-0.05, 1.05)
    ax.set_title(f"Shoulder Trigger Bins (n={len(shoulder_centers)})")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nVisualization saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Build stick cluster centers from training data")
    parser.add_argument("--data-dir", type=str, default="./data/full")
    parser.add_argument("--n-stick-clusters", type=int, default=30)
    parser.add_argument("--n-shoulder-bins", type=int, default=4)
    parser.add_argument("--n-files", type=int, default=1000,
                        help="Number of parquet files to sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None,
                        help="Output JSON path (default: <data-dir>/stick_clusters.json)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_path = Path(args.out) if args.out else data_dir / "stick_clusters.json"

    print("=== Building Stick Clusters ===")
    stick_xy, shoulder_vals = sample_data(data_dir, n_files=args.n_files, seed=args.seed)

    stick_centers = build_stick_clusters(stick_xy, n_clusters=args.n_stick_clusters, seed=args.seed)
    shoulder_centers = build_shoulder_bins(shoulder_vals, n_bins=args.n_shoulder_bins, seed=args.seed)

    result = {
        "stick_centers": [[round(float(x), 6), round(float(y), 6)]
                          for x, y in stick_centers],
        "shoulder_centers": [round(float(c), 6) for c in shoulder_centers],
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved clusters to {out_path}")

    print("\nStick cluster centers:")
    for i, (x, y) in enumerate(result["stick_centers"]):
        print(f"  [{i:2d}] x={x:.4f}  y={y:.4f}")

    viz_path = out_path.parent / "stick_clusters.png"
    visualize(stick_centers, shoulder_centers, stick_xy, viz_path)


if __name__ == "__main__":
    main()
