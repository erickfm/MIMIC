#!/usr/bin/env python3
"""Single-pass .slp → rank/character organized .pt shard pipeline.

Tensorizes ranked .slp replays and routes each perspective into
{rank}/{character}/ subdirectories for bespoke subset downloads.

Each perspective is routed by:
  - rank: parsed from filename (e.g. "master-diamond-{hash}.slp")
  - character: majority self_character from the tensorized game

Output structure:
    {repo}/
    ├── norm_stats.json, cat_maps.json, stick_clusters.json
    ├── dataset_index.json
    ├── master/
    │   ├── characters.json
    │   ├── FOX/
    │   │   ├── tensor_manifest.json
    │   │   ├── norm_stats.json, cat_maps.json, stick_clusters.json
    │   │   ├── train_shard_000.pt
    │   │   └── val_shard_000.pt
    │   └── FALCO/ ...
    ├── diamond/ ...
    └── platinum/ ...

Usage:
    python tools/slp_to_ranked_shards.py \\
        --slp-dir /data/ranked-anonymized \\
        --meta-dir data/ranked_meta \\
        --repo erickfm/mimic-melee-ranked \\
        --flush-gb 1.0 --stream --workers 8
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import os as _os
_os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import resource as _resource
_soft, _hard = _resource.getrlimit(_resource.RLIMIT_NOFILE)
_resource.setrlimit(_resource.RLIMIT_NOFILE, (min(_hard, 65536), _hard))

import argparse
import json
import logging
import random
import shutil
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Set

import torch
from melee.enums import Character

# Reuse from existing tools
from tools.slp_to_shards import (
    extract_replay,
    _load_prereqs,
    _make_result_iter,
    _numpy_to_torch,
    METADATA_FILES,
)
from tools.split_by_character import (
    CharacterBuffers,
    IDX_TO_DIR_NAME,
    DEFAULT_SKIP,
    flush_shard,
    get_majority_character,
)

log = logging.getLogger(__name__)

RANKS = ["master", "diamond", "platinum"]


# --- Rank parsing ------------------------------------------------------------

def _parse_ranks(slp_path: str):
    """Parse (p1_rank, p2_rank) from filename like 'master-diamond-{hash}.slp'."""
    name = Path(slp_path).stem
    parts = name.split("-")
    return parts[0], parts[1]


# --- File discovery + splitting ----------------------------------------------

def _find_and_split_files(slp_dir: str, val_frac: float, seed: int):
    """Find .slp files, split into train/val."""
    slp_dir = Path(slp_dir)
    files = sorted(str(f) for f in slp_dir.iterdir() if f.suffix.lower() == ".slp")

    rng = random.Random(seed)
    shuffled = list(files)
    rng.shuffle(shuffled)
    n_val = int(len(shuffled) * val_frac)

    return {
        "train": sorted(shuffled[n_val:]),
        "val": sorted(shuffled[:n_val]) if n_val > 0 else [],
    }


# --- Upload ------------------------------------------------------------------

def _upload_shard(api, repo_id, local_path, path_in_repo):
    """Upload a shard file and delete local copy. Retries on rate limit."""
    t0 = time.time()
    for attempt in range(5):
        try:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="dataset",
            )
            break
        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                wait = 60 * (attempt + 1)
                print(f"      Rate limited, waiting {wait}s...", flush=True)
                time.sleep(wait)
            else:
                raise
    dt = time.time() - t0
    size_gb = local_path.stat().st_size / 1e9
    local_path.unlink()
    return path_in_repo, dt, size_gb


# --- Progress tracking -------------------------------------------------------

def _save_progress(staging_dir, n_files_done, rank_buffers):
    """Save checkpoint for resume."""
    state = {
        "n_files_done": n_files_done,
        "ranks": {},
    }
    for rank, buffers in rank_buffers.items():
        state["ranks"][rank] = buffers.save_progress()

    path = staging_dir / "_progress.json"
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f)
    tmp.rename(path)


def _load_progress(staging_dir):
    """Load checkpoint, or return None."""
    path = staging_dir / "_progress.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# --- Index files -------------------------------------------------------------

def _build_characters_json(buffers, staging_dir):
    """Build characters.json index for one rank tier."""
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
    return characters


def _build_per_char_manifests(buffers, staging_dir, val_frac, seed):
    """Build tensor_manifest.json in each character subdirectory."""
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
            "val_frac": val_frac,
            "seed": seed,
        }

        char_dir = staging_dir / char_name
        char_dir.mkdir(parents=True, exist_ok=True)
        with open(char_dir / "tensor_manifest.json", "w") as fh:
            json.dump(manifest, fh, indent=2)


def _stage_metadata_per_char(buffers, meta_dir, staging_dir):
    """Copy shared metadata into each character subdirectory."""
    for char_idx in sorted(buffers.results.keys()):
        char_name = IDX_TO_DIR_NAME[char_idx]
        char_dir = staging_dir / char_name
        char_dir.mkdir(parents=True, exist_ok=True)
        for name in METADATA_FILES:
            src = meta_dir / name
            if src.exists():
                shutil.copy2(src, char_dir / name)


# --- Worker override for multiprocessing ------------------------------------

# For multiprocessing, we need the worker to return rank info alongside
# the tensorized result. We override the worker to include the path.

import multiprocessing as mp
from contextlib import contextmanager
from tools.slp_to_shards import _init_worker, _W

def _worker_fn_ranked(slp_path: str):
    """Worker that returns (slp_path, result) so main process can parse rank."""
    result = extract_replay(
        slp_path, _W["schema"], _W["norm_stats"], _W["cat_maps"],
        _W["stick_centers"], _W["shoulder_centers"],
    )
    if result is None:
        return (slp_path, None)
    return (slp_path, [
        ({k: v.numpy() for k, v in states.items()},
         {k: v.numpy() for k, v in targets.items()},
         n_frames)
        for states, targets, n_frames in result
    ])


def _numpy_to_torch_ranked(raw):
    """Convert (path, numpy_result) back to (path, torch_result)."""
    path, result = raw
    if result is None:
        return path, None
    return path, [
        ({k: torch.from_numpy(v) for k, v in states.items()},
         {k: torch.from_numpy(v) for k, v in targets.items()},
         n_frames)
        for states, targets, n_frames in result
    ]


@contextmanager
def _make_ranked_result_iter(slp_files, schema, norm_stats, cat_maps,
                              stick_centers, shoulder_centers, n_workers):
    """Yield (slp_path, result) tuples."""
    if n_workers > 1:
        pool = mp.Pool(
            n_workers,
            initializer=_init_worker,
            initargs=(schema, norm_stats, cat_maps,
                      stick_centers, shoulder_centers),
            maxtasksperchild=500,
        )
        try:
            yield pool.imap_unordered(_worker_fn_ranked, slp_files, chunksize=1)
        finally:
            pool.close()
            pool.join()
    else:
        def _serial():
            for path in slp_files:
                result = extract_replay(
                    path, schema, norm_stats, cat_maps,
                    stick_centers, shoulder_centers,
                )
                yield (path, result)
        yield _serial()


# Rewrite process_split to use _make_ranked_result_iter properly
def process_split_v2(
    split, split_files, staging_dir, rank_buffers, skip_indices,
    schema, norm_stats, cat_maps, stick_centers, shoulder_centers,
    n_workers, api, repo_id, stream, resume_n_files=0,
):
    """Process one split (train or val) of .slp files."""
    total_files = len(split_files)
    files_to_process = split_files[resume_n_files:] if resume_n_files else split_files
    if resume_n_files:
        print(f"    Resuming from file {resume_n_files}, "
              f"{len(files_to_process)} remaining", flush=True)

    upload_pool = ThreadPoolExecutor(max_workers=1) if stream else None
    pending_upload = None

    def _wait_pending():
        nonlocal pending_upload
        if pending_upload is not None:
            path_in_repo, dt, size_gb = pending_upload.result()
            print(f"      Uploaded {path_in_repo} ({size_gb:.2f} GB in {dt:.1f}s)",
                  flush=True)
            pending_upload = None

    def _flush_char(rank, char_idx):
        nonlocal pending_upload
        buffers = rank_buffers[rank]
        rank_dir = staging_dir / rank

        _wait_pending()
        info = buffers.flush(char_idx, split, rank_dir)
        fname = info[0]
        char_name = IDX_TO_DIR_NAME[char_idx]

        if stream and api:
            local_path = rank_dir / char_name / fname
            path_in_repo = f"{rank}/{char_name}/{fname}"
            pending_upload = upload_pool.submit(
                _upload_shard, api, repo_id, local_path, path_in_repo,
            )

    n_processed = 0
    n_games = 0
    t_start = time.time()

    with _make_ranked_result_iter(
        files_to_process, schema, norm_stats, cat_maps,
        stick_centers, shoulder_centers, n_workers,
    ) as result_iter:
        for raw in result_iter:
            if n_workers > 1:
                slp_path, result = _numpy_to_torch_ranked(raw)
            else:
                slp_path, result = raw

            n_processed += 1

            if result is None:
                continue

            p1_rank, p2_rank = _parse_ranks(slp_path)
            ranks_for_perspective = [p1_rank, p2_rank]

            for i, (states, targets, n_frames) in enumerate(result):
                rank = ranks_for_perspective[i]
                if rank not in rank_buffers:
                    continue

                char_idx = get_majority_character(states["self_character"])
                if char_idx in skip_indices:
                    continue

                rank_buffers[rank].add_game(char_idx, states, targets, n_frames)
                n_games += 1

                if rank_buffers[rank].should_flush(char_idx):
                    _flush_char(rank, char_idx)

            if n_processed % 500 == 0:
                elapsed = time.time() - t_start
                rate = n_processed / max(elapsed, 1)
                eta = (len(files_to_process) - n_processed) / max(rate, 1)
                done = n_processed + resume_n_files
                print(f"      [{done}/{total_files}] {split} files, "
                      f"{n_games} games  ({rate:.0f} files/s, "
                      f"ETA {eta/60:.0f}m)", flush=True)
                # Save progress periodically
                _save_progress(staging_dir, done, rank_buffers)

    # Flush remaining buffers
    for rank in RANKS:
        if rank in rank_buffers:
            for char_idx in list(rank_buffers[rank].buffers.keys()):
                if rank_buffers[rank].has_data(char_idx):
                    _flush_char(rank, char_idx)

    _wait_pending()
    if upload_pool:
        upload_pool.shutdown()

    total_done = n_processed + resume_n_files
    _save_progress(staging_dir, total_done, rank_buffers)


# --- Main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Tensorize ranked .slp replays into rank/character shard tree")
    parser.add_argument("--slp-dir", required=True,
                        help="Directory containing ranked .slp files")
    parser.add_argument("--meta-dir", required=True,
                        help="Directory with norm_stats.json, cat_maps.json, "
                             "stick_clusters.json")
    parser.add_argument("--repo", required=True,
                        help="HuggingFace repo (e.g. erickfm/mimic-melee-ranked)")
    parser.add_argument("--staging-dir", default=None,
                        help="Local staging directory (default: {slp-dir}/../ranked_staging)")
    parser.add_argument("--flush-gb", type=float, default=1.0,
                        help="Flush per-character buffer at this size in GB")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stream", action="store_true",
                        help="Upload each shard immediately after creation")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--clean", action="store_true",
                        help="Delete existing HF repo first")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers (default: 1)")
    parser.add_argument("--no-upload", action="store_true",
                        help="Tensorize only, skip upload")
    args = parser.parse_args()

    if args.resume and args.clean:
        parser.error("--resume and --clean are mutually exclusive")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    meta_dir = Path(args.meta_dir)
    staging_dir = (Path(args.staging_dir) if args.staging_dir
                   else Path(args.slp_dir).parent / "ranked_staging")
    staging_dir.mkdir(parents=True, exist_ok=True)
    flush_bytes = int(args.flush_gb * 1e9)
    n_workers = args.workers

    skip_indices = {
        idx for idx, name in IDX_TO_DIR_NAME.items()
        if name in DEFAULT_SKIP
    }

    # --- HF setup ---
    api = None
    if args.stream and not args.no_upload:
        from huggingface_hub import HfApi
        api = HfApi()
        if args.clean:
            try:
                api.delete_repo(repo_id=args.repo, repo_type="dataset")
                print(f"Deleted old repo {args.repo}")
            except Exception:
                pass
        api.create_repo(repo_id=args.repo, repo_type="dataset", exist_ok=True)

        # Upload shared metadata to repo root
        for name in METADATA_FILES:
            src = meta_dir / name
            if src.exists():
                api.upload_file(
                    path_or_fileobj=str(src),
                    path_in_repo=name,
                    repo_id=args.repo,
                    repo_type="dataset",
                )
                print(f"  Uploaded root {name}")

    # --- Load prereqs ---
    schema, norm_stats, cat_maps, stick_centers, shoulder_centers = (
        _load_prereqs(meta_dir))

    # --- Find and split files ---
    print(f"\nScanning {args.slp_dir} ...")
    splits = _find_and_split_files(args.slp_dir, args.val_frac, args.seed)
    print(f"  Found {len(splits['train'])} train + {len(splits['val'])} val files")

    # --- Initialize per-rank buffers ---
    rank_buffers = {rank: CharacterBuffers(flush_bytes) for rank in RANKS}

    # --- Resume ---
    progress = _load_progress(staging_dir) if args.resume else None
    resume_n_files = {"train": 0, "val": 0}
    if progress:
        for rank, rank_state in progress.get("ranks", {}).items():
            if rank in rank_buffers:
                rank_buffers[rank].restore_progress(rank_state)
        n_done = progress.get("n_files_done", 0)
        n_train = len(splits["train"])
        if n_done >= n_train:
            resume_n_files["train"] = n_train
            resume_n_files["val"] = n_done - n_train
        else:
            resume_n_files["train"] = n_done
        print(f"  Resuming: {n_done} files already processed")

    # --- Process ---
    t0 = time.time()

    for split in ["train", "val"]:
        split_files = splits[split]
        if not split_files:
            continue
        skip = resume_n_files[split]
        if skip >= len(split_files):
            print(f"\n=== {split} split already complete ===")
            continue

        print(f"\n=== Processing {len(split_files)} {split} files "
              f"({n_workers} workers) ===")

        process_split_v2(
            split, split_files, staging_dir, rank_buffers, skip_indices,
            schema, norm_stats, cat_maps, stick_centers, shoulder_centers,
            n_workers, api, args.repo, args.stream, resume_n_files=skip,
        )

    # --- Build index files ---
    print("\n=== Building index files ===")
    dataset_index = {"ranks": {}}

    for rank in RANKS:
        buffers = rank_buffers[rank]
        if not buffers.results:
            continue

        rank_dir = staging_dir / rank
        rank_dir.mkdir(parents=True, exist_ok=True)

        # Per-rank characters.json
        characters = _build_characters_json(buffers, rank_dir)

        # Per-character tensor_manifest.json + metadata
        _build_per_char_manifests(buffers, rank_dir, args.val_frac, args.seed)
        _stage_metadata_per_char(buffers, meta_dir, rank_dir)

        # Aggregate for dataset_index
        total_games = sum(c["n_train_games"] + c["n_val_games"]
                          for c in characters.values())
        total_frames = sum(c["n_train_frames"] + c["n_val_frames"]
                           for c in characters.values())
        total_bytes = sum(c["total_size_bytes"] for c in characters.values())

        dataset_index["ranks"][rank] = {
            "n_characters": len(characters),
            "n_games": total_games,
            "n_frames": total_frames,
            "total_size_bytes": total_bytes,
        }

        print(f"\n  {rank}: {len(characters)} characters, "
              f"{total_games:,} games, {total_frames:,} frames")
        for name, info in sorted(characters.items(),
                                  key=lambda x: -x[1]["n_train_games"]):
            pct = (info["n_train_frames"] + info["n_val_frames"]) / max(total_frames, 1) * 100
            print(f"    {name:20s}  {info['n_train_games']:>7,} games  "
                  f"{info['n_train_frames']:>12,} frames  ({pct:.1f}%)")

    # Write root dataset_index.json
    with open(staging_dir / "dataset_index.json", "w") as fh:
        json.dump(dataset_index, fh, indent=2)
    print(f"\n  Wrote dataset_index.json")

    # --- Upload index files ---
    if api and not args.no_upload:
        print("\n=== Uploading index + metadata files ===")
        # Upload dataset_index.json
        api.upload_file(
            path_or_fileobj=str(staging_dir / "dataset_index.json"),
            path_in_repo="dataset_index.json",
            repo_id=args.repo, repo_type="dataset",
        )
        # Upload per-rank characters.json + per-char manifests + metadata
        for rank in RANKS:
            rank_dir = staging_dir / rank
            if not rank_dir.exists():
                continue
            chars_json = rank_dir / "characters.json"
            if chars_json.exists():
                api.upload_file(
                    path_or_fileobj=str(chars_json),
                    path_in_repo=f"{rank}/characters.json",
                    repo_id=args.repo, repo_type="dataset",
                )
            for char_dir in sorted(rank_dir.iterdir()):
                if not char_dir.is_dir():
                    continue
                for meta_file in ["tensor_manifest.json"] + METADATA_FILES:
                    fp = char_dir / meta_file
                    if fp.exists():
                        api.upload_file(
                            path_or_fileobj=str(fp),
                            path_in_repo=f"{rank}/{char_dir.name}/{meta_file}",
                            repo_id=args.repo, repo_type="dataset",
                        )
        print("  Upload complete.")

    elapsed = time.time() - t0
    h, m = divmod(int(elapsed), 3600)
    m, s = divmod(m, 60)
    print(f"\n=== Done in {h}h {m}m {s}s ===")


if __name__ == "__main__":
    main()
