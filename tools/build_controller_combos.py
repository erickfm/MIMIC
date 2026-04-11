#!/usr/bin/env python3
"""Discover unique button combos from tensor shards for HAL-style controller encoding.

Scans all train + val shards, collapses X|Y → Jump and L|R → Shoulder,
and writes controller_combos.json with the full set of observed combos.

Usage:
    python tools/build_controller_combos.py --data-dir data/fox_public_shards
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import torch
from collections import Counter


LOGICAL_BUTTONS = ["A", "B", "Jump", "Z", "Shoulder"]


def collapse_buttons(btns: torch.Tensor) -> torch.Tensor:
    """Collapse 12-dim raw buttons to 5-dim logical buttons.

    Input cols: A=0, B=1, X=2, Y=3, Z=4, L=5, R=6, START=7, D_UP=8, D_DOWN=9, D_LEFT=10, D_RIGHT=11
    Output cols: A=0, B=1, Jump(X|Y)=2, Z=3, Shoulder(L|R)=4
    """
    a = btns[:, 0] > 0.5
    b = btns[:, 1] > 0.5
    jump = (btns[:, 2] > 0.5) | (btns[:, 3] > 0.5)
    z = btns[:, 4] > 0.5
    shoulder = (btns[:, 5] > 0.5) | (btns[:, 6] > 0.5)
    return torch.stack([a, b, jump, z, shoulder], dim=-1).int()


def scan_shards(data_dir: Path) -> Counter:
    """Scan all shards and count button combo frequencies."""
    combo_counts: Counter = Counter()

    # Support both manifest (per-game) and meta (pre-windowed) formats
    manifest_path = data_dir / "tensor_manifest.json"
    meta_path = data_dir / "tensor_meta.json"

    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        shard_names = manifest["train_shards"] + manifest["val_shards"]
    elif meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        shard_names = meta.get("train_shards", []) + meta.get("val_shards", [])
    else:
        # Fall back to globbing
        shard_names = [p.name for p in sorted(data_dir.glob("*.pt"))]

    if not shard_names:
        raise RuntimeError(f"No shards found in {data_dir}")

    total_frames = 0
    for i, name in enumerate(shard_names):
        path = data_dir / name
        if not path.exists():
            continue
        shard = torch.load(path, weights_only=True)
        btns = shard["states"]["self_buttons"]  # (N, T, 12) or (N, 12)
        if btns.dim() == 3:
            btns = btns.reshape(-1, 12)

        collapsed = collapse_buttons(btns)  # (N, 5) int

        # Batch count using unique
        unique, counts = collapsed.unique(dim=0, return_counts=True)
        for combo, count in zip(unique.tolist(), counts.tolist()):
            combo_counts[tuple(combo)] += count

        total_frames += len(btns)
        if (i + 1) % 20 == 0:
            print(f"  [{i + 1}/{len(shard_names)}] shards scanned, "
                  f"{total_frames:,} frames ...", flush=True)

    print(f"  Scanned {total_frames:,} frames across {len(shard_names)} shards")
    return combo_counts


def main():
    parser = argparse.ArgumentParser(
        description="Discover button combos from tensor shards")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory with tensor shards")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    print("=== Building Controller Combos ===")
    combo_counts = scan_shards(data_dir)

    total = sum(combo_counts.values())

    # Sort by frequency, NONE always first
    none_combo = (0, 0, 0, 0, 0)
    sorted_combos = sorted(combo_counts.items(),
                           key=lambda x: (x[0] != none_combo, -x[1]))

    combos = [list(c) for c, _ in sorted_combos]
    counts = [cnt for _, cnt in sorted_combos]

    result = {
        "button_names": LOGICAL_BUTTONS,
        "combos": combos,
        "counts": counts,
        "n_combos": len(combos),
    }

    out_path = data_dir / "controller_combos.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved {len(combos)} combos to {out_path}")
    print(f"\n{'Combo':<35} {'Count':>12} {'Pct':>8}")
    print("-" * 58)
    for combo, count in zip(combos, counts):
        pressed = [LOGICAL_BUTTONS[i] for i, v in enumerate(combo) if v]
        label = "+".join(pressed) if pressed else "NONE"
        pct = 100 * count / total
        print(f"{label:<35} {count:>12,} {pct:>7.3f}%")


if __name__ == "__main__":
    main()
