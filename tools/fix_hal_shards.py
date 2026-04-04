#!/usr/bin/env python3
"""Fix HAL-normalized shards to use HAL's actual stats.json values.

Corrections applied:
1. Numeric features: undo old hal_norm.json transforms, apply HAL's correct transforms
   - pos_x, pos_y (standardize): fix mean/std
   - percent (normalize): fix max (236 → 362 for p1, 421 for p2)
   - shield_strength (invert_normalize): fix min (0.0 → 0.024 for p1)
2. Controller encoding: permute button section [NONE,A,B,Jump,Z] → [A,B,Jump,Z,NONE]
3. Flags: normalize [0,1] → [-1,1] (HAL's normalize transform on binary features)

Usage:
    python tools/fix_hal_shards.py /path/to/shard_dir [--dry-run]
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import numpy as np

# ── Old stats (from hal_norm.json — WRONG, Fox-only data) ──────────────────
OLD_STATS = {
    "pos_x":            {"transform": "standardize", "mean": 1.1179643869400024, "std": 56.67056655883789},
    "pos_y":            {"transform": "standardize", "mean": 11.491534233093262, "std": 32.57835388183594},
    "percent":          {"transform": "normalize",   "min": 0.0, "max": 236.0},
    "stock":            {"transform": "normalize",   "min": 0, "max": 4},
    "jumps_left":       {"transform": "normalize",   "min": 0, "max": 6},
    "invuln_left":      {"transform": "normalize",   "min": 0, "max": 0},
    "shield_strength":  {"transform": "invert_normalize", "min": 0.0, "max": 60.0},
    # These have identical min/max in both old and new — no correction needed:
    # on_ground, facing, invulnerable: normalize with min=0, max=1
}

# ── New stats (from HAL's actual stats.json) ─────────────────────────────────
HAL_STATS_PATH = Path("/home/erick/projects/hal/hal/data/stats.json")


def load_hal_stats():
    with open(HAL_STATS_PATH) as f:
        raw = json.load(f)
    p1 = {k.removeprefix("p1_"): raw[k] for k in raw if k.startswith("p1_")}
    p2 = {k.removeprefix("p2_"): raw[k] for k in raw if k.startswith("p2_")}
    return p1, p2


def invert_old(val, old):
    """Invert old normalization to recover raw value."""
    t = old["transform"]
    if t == "standardize":
        return val * old["std"] + old["mean"]
    elif t == "normalize":
        if old["max"] == old["min"]:
            return torch.zeros_like(val)
        return (val + 1) * (old["max"] - old["min"]) / 2 + old["min"]
    elif t == "invert_normalize":
        if old["max"] == old["min"]:
            return torch.full_like(val, old["max"])
        return old["max"] - (val + 1) * (old["max"] - old["min"]) / 2
    return val


def apply_new(raw, new_stats, transform_type):
    """Apply correct HAL normalization."""
    if transform_type == "standardize":
        std = new_stats["std"]
        if std == 0:
            return torch.zeros_like(raw)
        return (raw - new_stats["mean"]) / std
    elif transform_type == "normalize":
        denom = new_stats["max"] - new_stats["min"]
        if denom == 0:
            return torch.zeros_like(raw)
        return 2 * (raw - new_stats["min"]) / denom - 1
    elif transform_type == "invert_normalize":
        denom = new_stats["max"] - new_stats["min"]
        if denom == 0:
            return torch.zeros_like(raw)
        return 2 * (new_stats["max"] - raw) / denom - 1
    return raw


# Numeric column indices (22-dim self_numeric / opp_numeric)
#  0: pos_x, 1: pos_y, 2: percent, 3: stock, 4: jumps_left,
# 12: invuln_left, 13: shield_strength
FEATURES_TO_FIX = {
    0:  ("pos_x",           "position_x"),
    1:  ("pos_y",           "position_y"),
    2:  ("percent",         "percent"),
    13: ("shield_strength", "shield_strength"),
    # stock (idx 3), jumps_left (idx 4): min/max identical, no fix needed
    # invuln_left (idx 12): always 0, degenerate, no fix needed
}

# Flag indices (5-dim self_flags / opp_flags)
# 0: on_ground, 1: off_stage, 2: facing, 3: invulnerable, 4: moonwalkwarning
# HAL uses: on_ground(0), facing(2), invulnerable(3) — all need [0,1]→[-1,1]
HAL_FLAG_INDICES = [0, 2, 3]  # on_ground, facing, invulnerable


def fix_numeric(tensor, old_stats_map, new_stats_dict):
    """Fix numeric features in-place. tensor shape: [N, 22]."""
    for col_idx, (old_key, new_key) in FEATURES_TO_FIX.items():
        old = old_stats_map[old_key]
        new = new_stats_dict[new_key]
        col = tensor[:, col_idx]
        raw = invert_old(col, old)
        tensor[:, col_idx] = apply_new(raw, new, old["transform"])


def fix_flags(tensor):
    """Normalize flag columns [0,1] → [-1,1]. tensor shape: [N, 5]."""
    for idx in HAL_FLAG_INDICES:
        tensor[:, idx] = 2 * tensor[:, idx] - 1


def fix_controller_buttons(tensor):
    """Permute button section of controller one-hot.
    Old ordering: [NONE=0, A=1, B=2, Jump=3, Z=4]
    New ordering: [A=0, B=1, Jump=2, Z=3, NONE=4]
    Button section is indices 46:51 (after main_stick=37, c_stick=9).
    tensor shape: [N, 54]."""
    btn = tensor[:, 46:51].clone()
    # old[0]=NONE→new[4], old[1]=A→new[0], old[2]=B→new[1],
    # old[3]=Jump→new[2], old[4]=Z→new[3]
    tensor[:, 46] = btn[:, 1]  # A
    tensor[:, 47] = btn[:, 2]  # B
    tensor[:, 48] = btn[:, 3]  # Jump
    tensor[:, 49] = btn[:, 4]  # Z
    tensor[:, 50] = btn[:, 0]  # NONE


def fix_shard(shard_path, p1_stats, p2_stats, dry_run=False):
    """Fix a single shard file."""
    shard = torch.load(shard_path, map_location="cpu", weights_only=False)
    states = shard["states"]

    n_frames = states["self_numeric"].shape[0]

    # 1. Fix self_numeric (uses p1 stats)
    fix_numeric(states["self_numeric"], OLD_STATS, p1_stats)

    # 2. Fix opp_numeric (uses p2 stats)
    fix_numeric(states["opp_numeric"], OLD_STATS, p2_stats)

    # 3. Fix flags [0,1] → [-1,1]
    fix_flags(states["self_flags"])
    fix_flags(states["opp_flags"])

    # 4. Fix controller button ordering
    if "self_controller" in states:
        fix_controller_buttons(states["self_controller"])

    if not dry_run:
        torch.save(shard, shard_path)

    return n_frames


def verify_shard(shard_path, p1_stats):
    """Quick verification of a fixed shard."""
    shard = torch.load(shard_path, map_location="cpu", weights_only=False)
    states = shard["states"]
    num = states["self_numeric"]
    flags = states["self_flags"]

    print(f"  pos_x:    mean={num[:,0].mean():.4f}  std={num[:,0].std():.4f}")
    print(f"  percent:  min={num[:,2].min():.4f}  max={num[:,2].max():.4f}  mean={num[:,2].mean():.4f}")
    print(f"  shield:   min={num[:,13].min():.4f}  max={num[:,13].max():.4f}")
    print(f"  flags:    unique={flags[:1000].unique().tolist()}")

    if "self_controller" in states:
        ctrl = states["self_controller"]
        btn = ctrl[:, 46:51]
        argmax_dist = dict(zip(*[v.tolist() for v in btn.argmax(dim=1).unique(return_counts=True)]))
        total = sum(argmax_dist.values())
        names = ["A", "B", "Jump", "Z", "NONE"]
        parts = [f"{names[i]}={argmax_dist.get(i, 0)/total*100:.1f}%" for i in range(5)]
        print(f"  btn dist: {' '.join(parts)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("shard_dir", help="Directory containing .pt shards")
    parser.add_argument("--dry-run", action="store_true", help="Don't write files")
    parser.add_argument("--verify", action="store_true", help="Verify after fixing")
    args = parser.parse_args()

    shard_dir = Path(args.shard_dir)
    shard_files = sorted(shard_dir.glob("*.pt"))
    if not shard_files:
        print(f"No .pt files found in {shard_dir}")
        sys.exit(1)

    print(f"Loading HAL stats from {HAL_STATS_PATH}")
    p1_stats, p2_stats = load_hal_stats()

    # Show what will change
    print(f"\nCorrections to apply ({len(shard_files)} shards):")
    print(f"  pos_x standardize:   mean {OLD_STATS['pos_x']['mean']:.3f}→{p1_stats['position_x']['mean']:.3f} (p1)")
    print(f"  pos_y standardize:   mean {OLD_STATS['pos_y']['mean']:.3f}→{p1_stats['position_y']['mean']:.3f} (p1)")
    print(f"  percent normalize:   max {OLD_STATS['percent']['max']}→{p1_stats['percent']['max']} (p1)")
    print(f"  shield invert_norm:  min {OLD_STATS['shield_strength']['min']}→{p1_stats['shield_strength']['min']:.4f} (p1)")
    print(f"  flags:               [0,1] → [-1,1]")
    print(f"  button ordering:     [NONE,A,B,Jump,Z] → [A,B,Jump,Z,NONE]")
    if args.dry_run:
        print("\n  DRY RUN — no files will be modified")

    total_frames = 0
    for i, shard_path in enumerate(shard_files):
        print(f"\n[{i+1}/{len(shard_files)}] {shard_path.name} ...", end=" ", flush=True)
        n = fix_shard(shard_path, p1_stats, p2_stats, dry_run=args.dry_run)
        total_frames += n
        print(f"{n:,} frames {'(dry run)' if args.dry_run else 'fixed'}")

        if args.verify and not args.dry_run:
            verify_shard(shard_path, p1_stats)

    print(f"\nDone. {total_frames:,} total frames across {len(shard_files)} shards.")

    if args.verify and not args.dry_run:
        print("\nVerifying first shard:")
        verify_shard(shard_files[0], p1_stats)


if __name__ == "__main__":
    main()
