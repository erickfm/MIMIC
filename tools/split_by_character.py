#!/usr/bin/env python3
"""Split a HuggingFace tensor shard dataset by character.

Downloads existing per-game tensor shards one at a time, splits each shard's
games by self_character (majority character for Sheik/Zelda transforms),
accumulates per-character buffers, and streams reorganized shards to a new
HuggingFace repo with character subdirectories.

Output structure:
    {dest-repo}/
    ├── characters.json
    ├── FALCO/
    │   ├── tensor_manifest.json
    │   ├── norm_stats.json, cat_maps.json, stick_clusters.json
    │   ├── train_shard_000.pt
    │   └── val_shard_000.pt
    ├── FOX/
    │   └── ...

Usage:
    # Full run (download, split, upload)
    python3 tools/split_by_character.py \\
        --source-repo erickfm/mimic-melee \\
        --dest-repo erickfm/mimic-melee-by-character \\
        --clean

    # Dry run (no upload)
    python3 tools/split_by_character.py \\
        --source-repo erickfm/mimic-melee-subset \\
        --dest-repo erickfm/mimic-melee-subset-by-character \\
        --no-upload

    # Resume after crash
    python3 tools/split_by_character.py \\
        --source-repo erickfm/mimic-melee \\
        --dest-repo erickfm/mimic-melee-by-character
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import os as _os
_os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import torch
from huggingface_hub import HfApi, hf_hub_download, upload_large_folder
from melee.enums import Character

# ── Character mapping ─────────────────────────────────────────────────────────

# Dense index → directory name
IDX_TO_DIR_NAME = {i: c.name for i, c in enumerate(Character)}
IDX_TO_DIR_NAME[10] = "ICE_CLIMBERS"  # POPO → ICE_CLIMBERS for clarity

DEFAULT_SKIP = {
    "WIREFRAME_MALE", "WIREFRAME_FEMALE", "GIGA_BOWSER",
    "SANDBAG", "UNKNOWN_CHARACTER",
}

METADATA_FILES = ["norm_stats.json", "cat_maps.json", "stick_clusters.json"]


# ── Shard splitting ───────────────────────────────────────────────────────────

def get_majority_character(char_tensor):
    """Return the dense character index that appears most often."""
    unique_chars, counts = char_tensor.unique(return_counts=True)
    return unique_chars[counts.argmax()].item()


def split_shard_by_character(shard, skip_indices: Set[int]):
    """Split a loaded shard into per-character game lists.

    Returns {char_idx: [(game_states, game_targets, n_frames), ...]}
    """
    offsets = shard["offsets"]
    n_games = shard["n_games"]
    states = shard["states"]
    targets = shard["targets"]

    by_char = defaultdict(list)
    for g in range(n_games):
        start = offsets[g].item()
        end = offsets[g + 1].item()
        n_frames = end - start
        if n_frames < 2:
            continue

        char_idx = get_majority_character(states["self_character"][start:end])
        if char_idx in skip_indices:
            continue

        # .clone() so we can free the source shard
        game_states = {k: v[start:end].clone() for k, v in states.items()}
        game_targets = {k: v[start:end].clone() for k, v in targets.items()}
        by_char[char_idx].append((game_states, game_targets, n_frames))

    return by_char


# ── Shard flushing (adapted from upload_dataset.py) ──────────────────────────

def flush_shard(buf_states, buf_targets, buf_offsets, prefix, shard_idx, out_dir):
    """Concatenate buffered games and save a shard .pt file."""
    offsets = torch.tensor(buf_offsets, dtype=torch.int64)
    states = {k: torch.cat([s[k] for s in buf_states], dim=0)
              for k in buf_states[0]}
    targets = {k: torch.cat([t[k] for t in buf_targets], dim=0)
               for k in buf_targets[0]}

    fname = f"{prefix}_shard_{shard_idx:03d}.pt"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "states": states,
        "targets": targets,
        "offsets": offsets,
        "n_games": len(buf_states),
    }, out_dir / fname)

    total_frames = offsets[-1].item()
    size_bytes = (out_dir / fname).stat().st_size
    print(f"    {out_dir.name}/{fname}: {len(buf_states)} games, "
          f"{total_frames:,} frames ({size_bytes / 1e9:.2f} GB)", flush=True)
    return fname, len(buf_states), total_frames, size_bytes


# ── Per-character buffer management ───────────────────────────────────────────

class CharacterBuffers:
    """Manages per-character accumulation buffers with byte tracking."""

    def __init__(self, flush_bytes: int):
        self.flush_bytes = flush_bytes
        # {char_idx: {"states": [...], "targets": [...], "offsets": [0,...], "bytes": int}}
        self.buffers: Dict[int, dict] = {}
        # {char_idx: {"train_shard_idx": int, "val_shard_idx": int}}
        self.shard_indices: Dict[int, Dict[str, int]] = {}
        # {char_idx: {"train": [(fname, n_games, n_frames, size_bytes), ...], "val": [...]}}
        self.results: Dict[int, Dict[str, list]] = {}

    def add_game(self, char_idx: int, game_states, game_targets, n_frames: int):
        if char_idx not in self.buffers:
            self.buffers[char_idx] = {
                "states": [], "targets": [], "offsets": [0], "bytes": 0,
            }
        if char_idx not in self.shard_indices:
            self.shard_indices[char_idx] = {"train": 0, "val": 0}
        if char_idx not in self.results:
            self.results[char_idx] = {"train": [], "val": []}

        buf = self.buffers[char_idx]
        buf["states"].append(game_states)
        buf["targets"].append(game_targets)
        buf["offsets"].append(buf["offsets"][-1] + n_frames)

        frame_bytes = sum(v.nelement() * v.element_size()
                          for v in game_states.values())
        frame_bytes += sum(v.nelement() * v.element_size()
                           for v in game_targets.values())
        buf["bytes"] += frame_bytes

    def should_flush(self, char_idx: int) -> bool:
        buf = self.buffers.get(char_idx)
        return buf is not None and buf["bytes"] >= self.flush_bytes

    def has_data(self, char_idx: int) -> bool:
        buf = self.buffers.get(char_idx)
        return buf is not None and len(buf["states"]) > 0

    def flush(self, char_idx: int, split: str, staging_dir: Path):
        """Flush one character's buffer to a shard file. Returns shard info."""
        buf = self.buffers[char_idx]
        char_name = IDX_TO_DIR_NAME[char_idx]
        shard_idx = self.shard_indices[char_idx][split]

        char_dir = staging_dir / char_name
        info = flush_shard(
            buf["states"], buf["targets"], buf["offsets"],
            split, shard_idx, char_dir,
        )
        self.shard_indices[char_idx][split] += 1
        self.results[char_idx][split].append(info)

        # Reset buffer
        buf["states"] = []
        buf["targets"] = []
        buf["offsets"] = [0]
        buf["bytes"] = 0
        return info

    def active_chars(self) -> List[int]:
        return [c for c in self.buffers if self.has_data(c)]

    def restore_progress(self, progress: dict):
        """Restore shard indices and results from a progress checkpoint."""
        for char_name, indices in progress.get("shard_indices", {}).items():
            # Find char_idx from name
            for idx, name in IDX_TO_DIR_NAME.items():
                if name == char_name:
                    self.shard_indices[idx] = indices
                    break
        for char_name, results in progress.get("results", {}).items():
            for idx, name in IDX_TO_DIR_NAME.items():
                if name == char_name:
                    self.results[idx] = results
                    break

    def save_progress(self) -> dict:
        """Serialize shard indices and results for checkpointing."""
        return {
            "shard_indices": {
                IDX_TO_DIR_NAME[c]: idx
                for c, idx in self.shard_indices.items()
            },
            "results": {
                IDX_TO_DIR_NAME[c]: res
                for c, res in self.results.items()
            },
        }


# ── Progress tracking ─────────────────────────────────────────────────────────

def load_progress(staging_dir: Path) -> dict:
    path = staging_dir / "progress.json"
    if path.exists():
        with open(path) as fh:
            return json.load(fh)
    return {"completed_shards": [], "shard_indices": {}, "results": {}}


def save_progress(staging_dir: Path, progress: dict):
    staging_dir.mkdir(parents=True, exist_ok=True)
    with open(staging_dir / "progress.json", "w") as fh:
        json.dump(progress, fh, indent=2)


# ── Main processing ──────────────────────────────────────────────────────────

def process_split(
    split: str,
    shard_list: List[str],
    source_repo: str,
    staging_dir: Path,
    buffers: CharacterBuffers,
    progress: dict,
):
    """Process all shards for one split (train or val). Writes to disk only."""
    completed = set(progress["completed_shards"])
    t_split = time.time()

    for i, shard_name in enumerate(shard_list):
        shard_key = f"{split}:{shard_name}"
        if shard_key in completed:
            print(f"  [{split} {i + 1}/{len(shard_list)}] Skipping {shard_name} "
                  f"(already processed)", flush=True)
            continue

        elapsed = time.time() - t_split
        rate = max(i, 1) / max(elapsed, 1)
        remaining = len(shard_list) - i
        eta_min = remaining / max(rate, 0.001) / 60

        print(f"\n  [{split} {i + 1}/{len(shard_list)}] {shard_name} "
              f"(ETA {eta_min:.0f}m)", flush=True)

        # Download
        t0 = time.time()
        local_path = hf_hub_download(
            source_repo, shard_name,
            repo_type="dataset",
            local_dir=staging_dir / "_download",
        )
        local_path = Path(local_path)
        dl_time = time.time() - t0
        dl_size = local_path.stat().st_size / 1e9
        print(f"    Downloaded ({dl_size:.2f} GB in {dl_time:.1f}s)", flush=True)

        # Load into memory and delete local source file
        shard = torch.load(local_path, weights_only=True)
        local_path.unlink(missing_ok=True)

        # Split by character
        skip_indices = {
            idx for idx, name in IDX_TO_DIR_NAME.items()
            if name in _skip_set
        }
        by_char = split_shard_by_character(shard, skip_indices)
        del shard

        n_games_total = sum(len(games) for games in by_char.values())
        chars_found = [IDX_TO_DIR_NAME[c] for c in sorted(by_char.keys())]
        print(f"    Split into {n_games_total} games across "
              f"{len(by_char)} characters: {', '.join(chars_found)}", flush=True)

        # Add to buffers and conditionally flush
        for char_idx, games in by_char.items():
            for game_states, game_targets, n_frames in games:
                buffers.add_game(char_idx, game_states, game_targets, n_frames)

            if buffers.should_flush(char_idx):
                buffers.flush(char_idx, split, staging_dir)

        # Mark this source shard as completed
        progress["completed_shards"].append(shard_key)
        progress.update(buffers.save_progress())
        save_progress(staging_dir, progress)

    # Flush all remaining buffers for this split
    for char_idx in list(buffers.buffers.keys()):
        if buffers.has_data(char_idx):
            buffers.flush(char_idx, split, staging_dir)

    # Save final progress for this split
    progress.update(buffers.save_progress())
    save_progress(staging_dir, progress)


# ── Manifest and metadata upload ──────────────────────────────────────────────

def build_manifests(
    buffers: CharacterBuffers,
    staging_dir: Path,
    source_manifest: dict,
):
    """Build per-character tensor_manifest.json files to disk."""
    print("\n=== Building per-character manifests ===")
    for char_idx, char_results in sorted(buffers.results.items()):
        char_name = IDX_TO_DIR_NAME[char_idx]
        train_infos = char_results.get("train", [])
        val_infos = char_results.get("val", [])

        manifest = {
            "train_shards": [r[0] for r in train_infos],
            "val_shards": [r[0] for r in val_infos],
            "n_train_games": sum(r[1] for r in train_infos),
            "n_val_games": sum(r[1] for r in val_infos),
            "n_train_frames": sum(r[2] for r in train_infos),
            "n_val_frames": sum(r[2] for r in val_infos),
            "val_frac": source_manifest.get("val_frac", 0.1),
            "seed": source_manifest.get("seed", 42),
        }

        char_dir = staging_dir / char_name
        char_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = char_dir / "tensor_manifest.json"
        with open(manifest_path, "w") as fh:
            json.dump(manifest, fh, indent=2)

        print(f"  {char_name}: {manifest['n_train_games']} train games, "
              f"{manifest['n_val_games']} val games, "
              f"{manifest['n_train_frames']:,} train frames", flush=True)


def stage_metadata_per_char(
    buffers: CharacterBuffers,
    metadata_dir: Path,
    staging_dir: Path,
):
    """Copy shared metadata files into each character subdirectory."""
    print("\n=== Staging metadata to character directories ===")
    for char_idx in sorted(buffers.results.keys()):
        char_name = IDX_TO_DIR_NAME[char_idx]
        char_dir = staging_dir / char_name
        char_dir.mkdir(parents=True, exist_ok=True)
        for name in METADATA_FILES:
            src = metadata_dir / name
            if not src.exists():
                continue
            shutil.copy2(src, char_dir / name)
        print(f"  {char_name}: metadata staged", flush=True)


def build_characters_json(
    buffers: CharacterBuffers,
    staging_dir: Path,
):
    """Build root characters.json index."""
    print("\n=== Building characters.json ===")
    characters = {}
    for char_idx, char_results in sorted(buffers.results.items()):
        char_name = IDX_TO_DIR_NAME[char_idx]
        train_infos = char_results.get("train", [])
        val_infos = char_results.get("val", [])

        characters[char_name] = {
            "dense_index": char_idx,
            "n_train_games": sum(r[1] for r in train_infos),
            "n_val_games": sum(r[1] for r in val_infos),
            "n_train_frames": sum(r[2] for r in train_infos),
            "n_val_frames": sum(r[2] for r in val_infos),
            "n_train_shards": len(train_infos),
            "n_val_shards": len(val_infos),
            "total_size_bytes": (sum(r[3] for r in train_infos)
                                 + sum(r[3] for r in val_infos)),
        }

    chars_path = staging_dir / "characters.json"
    with open(chars_path, "w") as fh:
        json.dump(characters, fh, indent=2)

    # Print summary
    total_games = sum(c["n_train_games"] + c["n_val_games"]
                      for c in characters.values())
    total_frames = sum(c["n_train_frames"] + c["n_val_frames"]
                       for c in characters.values())
    total_bytes = sum(c["total_size_bytes"] for c in characters.values())
    print(f"\n  {len(characters)} characters, {total_games:,} total games, "
          f"{total_frames:,} total frames, {total_bytes / 1e12:.2f} TB")
    print()
    for name, info in sorted(characters.items(),
                              key=lambda x: -x[1]["n_train_games"]):
        pct = (info["n_train_frames"] + info["n_val_frames"]) / max(total_frames, 1) * 100
        print(f"  {name:20s}  {info['n_train_games']:>7,} games  "
              f"{info['n_train_frames']:>12,} frames  ({pct:.1f}%)")

    print(f"\n  Wrote characters.json")


# ── CLI ───────────────────────────────────────────────────────────────────────

# Module-level so process_split can access it (set in main)
_skip_set: Set[str] = DEFAULT_SKIP


def main():
    global _skip_set

    parser = argparse.ArgumentParser(
        description="Split HuggingFace tensor shard dataset by character")
    parser.add_argument("--source-repo", default="erickfm/mimic-melee",
                        help="Source HF dataset repo (default: erickfm/mimic-melee)")
    parser.add_argument("--dest-repo", default="erickfm/mimic-melee-by-character",
                        help="Destination HF dataset repo")
    parser.add_argument("--staging-dir", default="data/char_split_staging",
                        help="Local staging directory for temp files")
    parser.add_argument("--flush-gb", type=float, default=1.0,
                        help="Flush per-character buffer at this size in GB (default: 1)")
    parser.add_argument("--clean", action="store_true",
                        help="Delete existing dest repo before uploading")
    parser.add_argument("--skip-characters", type=str,
                        default=",".join(sorted(DEFAULT_SKIP)),
                        help="Comma-separated character names to skip")
    parser.add_argument("--no-upload", action="store_true",
                        help="Split only, skip upload (for testing)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Process shards but don't upload or save (for testing)")
    args = parser.parse_args()

    _skip_set = set(args.skip_characters.split(",")) if args.skip_characters else set()
    staging_dir = Path(args.staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    flush_bytes = int(args.flush_gb * 1e9)

    print(f"Source repo:  {args.source_repo}")
    print(f"Dest repo:    {args.dest_repo}")
    print(f"Staging dir:  {staging_dir}")
    print(f"Flush size:   {args.flush_gb} GB")
    print(f"Skip chars:   {_skip_set}")

    # ── Initialize HF API ─────────────────────────────────────────────────
    api = None
    if not args.no_upload and not args.dry_run:
        api = HfApi()
        if args.clean:
            try:
                api.delete_repo(repo_id=args.dest_repo, repo_type="dataset")
                print(f"Deleted old repo {args.dest_repo}")
            except Exception:
                pass
        api.create_repo(repo_id=args.dest_repo, repo_type="dataset",
                        exist_ok=True)

    # ── Download source manifest + metadata ───────────────────────────────
    print("\n=== Downloading source manifest and metadata ===")
    metadata_dir = staging_dir / "_metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = hf_hub_download(
        args.source_repo, "tensor_manifest.json",
        repo_type="dataset", local_dir=metadata_dir,
    )
    with open(manifest_path) as fh:
        source_manifest = json.load(fh)

    for name in METADATA_FILES:
        hf_hub_download(
            args.source_repo, name,
            repo_type="dataset", local_dir=metadata_dir,
        )
        print(f"  Downloaded {name}")

    train_shards = source_manifest["train_shards"]
    val_shards = source_manifest["val_shards"]
    print(f"\n  Source: {len(train_shards)} train shards, "
          f"{len(val_shards)} val shards")
    print(f"  Total: {source_manifest.get('n_train_games', '?')} train games, "
          f"{source_manifest.get('n_val_games', '?')} val games")

    # ── Load or initialize progress ───────────────────────────────────────
    progress = load_progress(staging_dir)
    buffers = CharacterBuffers(flush_bytes)
    buffers.restore_progress(progress)

    if progress["completed_shards"]:
        print(f"\n  Resuming: {len(progress['completed_shards'])} shards "
              f"already processed")

    # ── Process train and val splits ──────────────────────────────────────
    t0 = time.time()

    for split, shard_list in [("train", train_shards), ("val", val_shards)]:
        print(f"\n{'=' * 60}")
        print(f"=== Processing {len(shard_list)} {split} shards ===")
        print(f"{'=' * 60}")

        process_split(
            split, shard_list, args.source_repo,
            staging_dir, buffers, progress,
        )

    # ── Build manifests + metadata ───────────────────────────────────────
    build_manifests(buffers, staging_dir, source_manifest)
    stage_metadata_per_char(buffers, metadata_dir, staging_dir)
    build_characters_json(buffers, staging_dir)

    # Clean up internal dirs before upload
    for d in ["_download", "_metadata"]:
        p = staging_dir / d
        if p.exists():
            shutil.rmtree(p)
    progress_path = staging_dir / "progress.json"
    if progress_path.exists():
        progress_path.unlink()

    # ── Bulk upload ───────────────────────────────────────────────────────
    if not args.no_upload and not args.dry_run and api is not None:
        print(f"\n{'=' * 60}")
        print(f"=== Uploading to {args.dest_repo} ===")
        print(f"{'=' * 60}")
        upload_large_folder(
            folder_path=str(staging_dir),
            repo_id=args.dest_repo,
            repo_type="dataset",
        )
        print(f"\n  Upload complete.")

    elapsed = time.time() - t0
    h, m = divmod(int(elapsed), 3600)
    m, s = divmod(m, 60)
    print(f"\n=== Done in {h}h {m}m {s}s ===")


if __name__ == "__main__":
    main()
