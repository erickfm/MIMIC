#!/usr/bin/env python3
"""Reshard .pt shard files into smaller chunks.

Reads existing shards (with states/targets/offsets/n_games structure),
splits games across more, smaller shards, and writes a new tensor_manifest.json.

Usage:
    python tools/reshard.py --src data/fox_hal_local --dst data/fox_hal_800m --target-mb 800
"""

import argparse
import json
import shutil
from pathlib import Path

import torch


def reshard(src_dir: Path, dst_dir: Path, target_bytes: int):
    with open(src_dir / "tensor_manifest.json") as f:
        manifest = json.load(f)

    dst_dir.mkdir(parents=True, exist_ok=True)

    # Copy metadata files (include both new mimic_norm.json and legacy hal_norm.json)
    for meta_file in ["norm_stats.json", "cat_maps.json", "stick_clusters.json",
                       "controller_combos.json", "mimic_norm.json", "hal_norm.json",
                       "tensor_manifest.json"]:
        src_path = src_dir / meta_file
        if src_path.exists():
            shutil.copy2(src_path, dst_dir / meta_file)

    for split in ["train", "val"]:
        shard_names = manifest.get(f"{split}_shards", [])
        if not shard_names:
            continue

        print(f"\n=== {split}: {len(shard_names)} source shards ===")

        # Estimate bytes_per_frame from first shard
        first_shard = torch.load(src_dir / shard_names[0], weights_only=True)
        first_offsets = first_shard["offsets"]
        n_frames_first = first_offsets[-1].item()
        bpf = sum(v.nelement() * v.element_size() for v in first_shard["states"].values())
        bpf += sum(v.nelement() * v.element_size() for v in first_shard["targets"].values())
        bytes_per_frame = bpf / n_frames_first
        del first_shard

        # Stream through source shards one at a time, accumulate games, flush when full
        new_shard_names = []
        shard_idx = 0
        current_games = []  # list of (states_dict, targets_dict)
        current_frames = 0
        total_games = 0

        def write_shard(games, idx):
            state_keys = list(games[0][0].keys())
            target_keys = list(games[0][1].keys())
            cat_states = {k: torch.cat([g[0][k] for g in games]) for k in state_keys}
            cat_targets = {k: torch.cat([g[1][k] for g in games]) for k in target_keys}
            offsets = [0]
            for g_states, _ in games:
                n = next(iter(g_states.values())).shape[0]
                offsets.append(offsets[-1] + n)
            shard_data = {
                "states": cat_states,
                "targets": cat_targets,
                "offsets": torch.tensor(offsets, dtype=torch.long),
                "n_games": len(games),
            }
            name = f"{split}_shard_{idx:04d}.pt"
            torch.save(shard_data, dst_dir / name)
            total_frames = offsets[-1]
            size_mb = (dst_dir / name).stat().st_size / 1e6
            print(f"  Wrote {name}: {len(games)} games, {total_frames} frames, {size_mb:.0f} MB")
            return name

        for sname in shard_names:
            path = src_dir / sname
            print(f"  Reading {sname}...", end="", flush=True)
            shard = torch.load(path, weights_only=True)
            offsets = shard["offsets"]
            n_games = shard["n_games"]
            states = shard["states"]
            targets = shard["targets"]
            print(f" {n_games} games, {offsets[-1].item()} frames")

            for g in range(n_games):
                start = offsets[g].item()
                end = offsets[g + 1].item()
                n_frames = end - start
                game_states = {k: v[start:end].clone() for k, v in states.items()}
                game_targets = {k: v[start:end].clone() for k, v in targets.items()}

                projected_bytes = (current_frames + n_frames) * bytes_per_frame
                if current_games and projected_bytes > target_bytes:
                    name = write_shard(current_games, shard_idx)
                    new_shard_names.append(name)
                    shard_idx += 1
                    current_games = []
                    current_frames = 0

                current_games.append((game_states, game_targets))
                current_frames += n_frames
                total_games += 1

            del shard, states, targets  # free source shard memory

        if current_games:
            name = write_shard(current_games, shard_idx)
            new_shard_names.append(name)

        print(f"  {split}: {total_games} games → {len(new_shard_names)} shards")
        manifest[f"{split}_shards"] = new_shard_names

    # Write new manifest
    with open(dst_dir / "tensor_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone! {dst_dir / 'tensor_manifest.json'} written.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reshard .pt files into smaller chunks")
    parser.add_argument("--src", type=str, required=True, help="Source data directory")
    parser.add_argument("--dst", type=str, required=True, help="Destination data directory")
    parser.add_argument("--target-mb", type=int, default=800, help="Target shard size in MB")
    args = parser.parse_args()

    reshard(Path(args.src), Path(args.dst), args.target_mb * 1024 * 1024)
