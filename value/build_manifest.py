"""Build tensor_manifest.json for V(s) training over fox_all_v2 shards.

The shards in data/fox_all_v2/ were written by the WM pipeline which doesn't
emit a manifest or val split. This script:
  1. Scans train_shard_*.pt files
  2. Applies a deterministic seeded shuffle and splits 10% val / 90% train
  3. Reads each shard's offsets array to count games/frames (mmap'd, fast)
  4. Writes tensor_manifest.json matching the schema used by
     fox_master_v2/tensor_manifest.json.

Games are disjoint within a shard by construction in slp_to_shards.py, so
shard-level holdout is leakage-free.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from tqdm import tqdm


def shard_stats(path: Path) -> tuple[int, int]:
    """Return (n_games, n_frames) for a shard. Uses mmap to avoid full load."""
    shard = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    n_games = int(shard["n_games"])
    offsets = shard["offsets"]
    n_frames = int(offsets[-1].item()) if offsets.numel() > 0 else 0
    return n_games, n_frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/fox_all_v2")
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=None,
                   help="Defaults to {data_dir}/tensor_manifest.json")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out = Path(args.output) if args.output else data_dir / "tensor_manifest.json"

    all_shards = sorted(p.name for p in data_dir.glob("train_shard_*.pt"))
    if not all_shards:
        raise SystemExit(f"No train_shard_*.pt files found in {data_dir}")

    rng = random.Random(args.seed)
    shuffled = list(all_shards)
    rng.shuffle(shuffled)
    n_val = max(1, round(args.val_frac * len(shuffled)))
    val_shards = sorted(shuffled[:n_val])
    train_shards = sorted(shuffled[n_val:])

    print(f"total shards: {len(all_shards)}")
    print(f"val: {len(val_shards)}  train: {len(train_shards)}  "
          f"(val_frac={len(val_shards)/len(all_shards):.3f})")

    n_train_games = n_val_games = 0
    n_train_frames = n_val_frames = 0
    for name in tqdm(train_shards, desc="train shards"):
        g, f = shard_stats(data_dir / name)
        n_train_games += g
        n_train_frames += f
    for name in tqdm(val_shards, desc="val shards"):
        g, f = shard_stats(data_dir / name)
        n_val_games += g
        n_val_frames += f

    manifest = {
        "train_shards": train_shards,
        "val_shards": val_shards,
        "n_train_games": n_train_games,
        "n_val_games": n_val_games,
        "n_train_frames": n_train_frames,
        "n_val_frames": n_val_frames,
        "val_frac": args.val_frac,
        "seed": args.seed,
    }
    out.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out}")
    print(f"  train: {n_train_games:,} games  {n_train_frames:,} frames")
    print(f"  val:   {n_val_games:,} games  {n_val_frames:,} frames")


if __name__ == "__main__":
    main()
