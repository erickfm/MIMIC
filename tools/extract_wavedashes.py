#!/usr/bin/env python3
"""Extract wavedash sequences from existing shards into a new overfit-test shard.

A wavedash is identified by the LANDING_SPECIAL action (43) following a
KNEE_BEND (24). We extract a 256-frame window ending ~10 frames after the
wavedash landing, so the model sees the full setup → execution → landing.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import glob
import torch
import shutil

W = 256
TAIL = 10  # frames after LANDING_SPECIAL onset to include in window


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dir", default="data/falco_v2")
    parser.add_argument("--dst-dir", default="data/falco_wavedash")
    parser.add_argument("--max-windows", type=int, default=2000)
    parser.add_argument("--shards-to-scan", type=int, default=10)
    args = parser.parse_args()

    src = Path(args.src_dir)
    dst = Path(args.dst_dir)
    dst.mkdir(parents=True, exist_ok=True)

    shards = sorted(glob.glob(str(src / "train_shard_*.pt")))[:args.shards_to_scan]
    print(f"Scanning {len(shards)} shards from {src}")

    # Find all wavedash onsets (LANDING_SPECIAL after not-LANDING_SPECIAL)
    windows_states = []
    windows_targets = []
    n_found = 0

    for sp in shards:
        if n_found >= args.max_windows:
            break
        s = torch.load(sp, map_location='cpu', weights_only=False)
        states = s['states']
        targets = s['targets']
        action = states['self_action']
        n = len(action)

        btns = targets['btns_single']  # 7-class button targets
        for i in range(1, n):
            if n_found >= args.max_windows:
                break
            if action[i].item() != 43 or action[i-1].item() == 43:
                continue
            # Onset of LANDING_SPECIAL at frame i
            end = min(i + TAIL, n)
            start = end - W
            if start < 0:
                continue
            # Verify wavedash setup is inside window:
            # there must be a KNEE_BEND (action 24) within the last 30 frames
            # before the LANDING_SPECIAL onset, with a JUMP button press
            # preceding it.
            setup_ok = False
            for j in range(max(start, i - 30), i):
                if action[j].item() == 24:
                    # Check for JUMP button target in the 5 frames leading to KNEE_BEND
                    for k in range(max(start, j - 5), j + 1):
                        if btns[k].item() == 3:  # JUMP class
                            setup_ok = True
                            break
                    if setup_ok:
                        break
            if not setup_ok:
                continue
            # Extract window
            w_states = {k: v[start:end].clone() for k, v in states.items()}
            w_targets = {k: v[start:end].clone() for k, v in targets.items()}
            windows_states.append(w_states)
            windows_targets.append(w_targets)
            n_found += 1

        print(f"  {Path(sp).name}: {n_found} wavedashes so far")

    print(f"\nTotal wavedash windows extracted: {n_found}")

    # Concatenate windows into a single shard (each window is one "game")
    # The shard format expects: states (dict of [total_frames, ...]), targets (same), offsets, n_games
    offsets = [0]
    for i in range(len(windows_states)):
        offsets.append(offsets[-1] + W)
    offsets = torch.tensor(offsets, dtype=torch.int64)

    states_cat = {}
    for k in windows_states[0]:
        states_cat[k] = torch.cat([w[k] for w in windows_states], dim=0)
    targets_cat = {}
    for k in windows_targets[0]:
        targets_cat[k] = torch.cat([w[k] for w in windows_targets], dim=0)

    # Save as a single train shard
    shard_data = {
        "states": states_cat,
        "targets": targets_cat,
        "offsets": offsets,
        "n_games": len(windows_states),
    }
    out = dst / "train_shard_000.pt"
    torch.save(shard_data, out)
    print(f"Saved {out}: {len(windows_states)} wavedash windows, "
          f"{offsets[-1].item():,} total frames "
          f"({out.stat().st_size / 1e9:.2f} GB)")

    # Also save a small val shard (take 10% of the windows)
    n_val = max(1, len(windows_states) // 10)
    val_states = {k: v[:n_val * W].clone() for k, v in states_cat.items()}
    val_targets = {k: v[:n_val * W].clone() for k, v in targets_cat.items()}
    val_offsets = offsets[:n_val + 1].clone()
    val_shard = {
        "states": val_states,
        "targets": val_targets,
        "offsets": val_offsets,
        "n_games": n_val,
    }
    out_val = dst / "val_shard_000.pt"
    torch.save(val_shard, out_val)
    print(f"Saved {out_val}: {n_val} val windows")

    # Copy metadata from source
    for f in ["mimic_norm.json", "hal_norm.json", "controller_combos.json",
              "cat_maps.json", "stick_clusters.json", "norm_stats.json"]:
        src_f = src / f
        if src_f.exists():
            shutil.copy(src_f, dst / f)
            print(f"  copied {f}")


if __name__ == "__main__":
    main()
