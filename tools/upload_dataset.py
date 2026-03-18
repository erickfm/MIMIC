#!/usr/bin/env python3
"""Pretokenize a parquet dataset into tensor shards and upload to HuggingFace.

Each shard stores multiple games concatenated along the time axis with an
offsets array for game boundaries.  All parquet reading, preprocessing,
normalization, and target building happen here -- the resulting shards are
ready for training with zero overhead at load time.

Prerequisites (already done for existing datasets):
    Data dir must contain norm_stats.json, cat_maps.json, and
    stick_clusters.json (produced by the legacy preprocess.py and
    build_clusters.py scripts).

Usage:
    # Tensorize + upload (streaming, multiprocessing)
    python3 upload_dataset.py --data-dir data/full --repo erickfm/mimic-melee --stream --workers 64

    # Tensorize only (no upload)
    python3 upload_dataset.py --data-dir data/full --repo erickfm/mimic-melee --no-upload

    # Upload a previously staged directory
    python3 upload_dataset.py --staging-dir data/full_upload --repo erickfm/mimic-melee --upload-only
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
import multiprocessing as mp
import os
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from huggingface_hub import HfApi, upload_large_folder

import mimic.features as F
from mimic.features import load_cluster_centers

METADATA_FILES = ["norm_stats.json", "cat_maps.json", "stick_clusters.json"]

# ── Worker pool state (set by initializer, used by _worker_tensorize) ────────

_W = {}


def _init_worker(categorical_cols, cat_maps, norm_stats, fg,
                 stick_centers, shoulder_centers):
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(hard, 65536), hard))
    _W["categorical_cols"] = categorical_cols
    _W["cat_maps"] = cat_maps
    _W["norm_stats"] = norm_stats
    _W["fg"] = fg
    _W["stick_centers"] = stick_centers
    _W["shoulder_centers"] = shoulder_centers


def _worker_tensorize(path: Path):
    """Per-worker entry point — reads one parquet and returns tensorized game."""
    return tensorize_game(
        path, _W["categorical_cols"], _W["cat_maps"], _W["norm_stats"],
        _W["fg"], _W["stick_centers"], _W["shoulder_centers"],
    )


# ── Core functions ───────────────────────────────────────────────────────────

def tensorize_game(
    path: Path,
    categorical_cols: List[str],
    cat_maps: Dict[str, Dict[int, int]],
    norm_stats: Dict[str, Tuple[float, float]],
    fg: Dict,
    stick_centers: np.ndarray | None,
    shoulder_centers: np.ndarray | None,
):
    """Read one parquet, preprocess, return (states, targets, n_frames) or None."""
    df = pd.read_parquet(path)
    df = df[df["frame"] >= 0].reset_index(drop=True)
    if len(df) < 2:
        return None

    df = F.preprocess_df(df, categorical_cols, cat_maps)
    F.apply_normalization(df, norm_stats)
    states = F.df_to_state_tensors(df, fg)
    targets = F.build_targets_batch(
        df, norm_stats,
        stick_centers=stick_centers,
        shoulder_centers=shoulder_centers,
    )
    return states, targets, len(df)


def flush_shard(buf_states, buf_targets, buf_offsets, prefix, shard_idx, out_dir):
    """Concatenate buffered games and save a shard .pt file."""
    offsets = torch.tensor(buf_offsets, dtype=torch.int64)
    states = {k: torch.cat([s[k] for s in buf_states], dim=0)
              for k in buf_states[0]}
    targets = {k: torch.cat([t[k] for t in buf_targets], dim=0)
               for k in buf_targets[0]}

    fname = f"{prefix}_shard_{shard_idx:03d}.pt"
    torch.save({
        "states": states,
        "targets": targets,
        "offsets": offsets,
        "n_games": len(buf_states),
    }, out_dir / fname)

    total_frames = offsets[-1].item()
    size_bytes = (out_dir / fname).stat().st_size
    print(f"    {fname}: {len(buf_states)} games, {total_frames:,} frames "
          f"({size_bytes / 1e9:.2f} GB)", flush=True)
    return fname, len(buf_states), total_frames, size_bytes


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_prereqs(data_dir: Path):
    with open(data_dir / "norm_stats.json") as fh:
        norm_stats = json.load(fh)
    with open(data_dir / "cat_maps.json") as fh:
        raw = json.load(fh)
        cat_maps = {col: {int(k): v for k, v in m.items()} for col, m in raw.items()}

    stick_centers, shoulder_centers = load_cluster_centers(data_dir)
    fg = F.build_feature_groups()
    categorical_cols = F.get_categorical_cols(fg)
    return norm_stats, cat_maps, stick_centers, shoulder_centers, fg, categorical_cols


def _get_split_parquets(data_dir: Path, val_frac: float, seed: int) -> Dict[str, List[Path]]:
    parquets = sorted(data_dir.glob("*.parquet"))
    if not parquets:
        raise RuntimeError(f"No .parquet files in {data_dir}")

    rng = random.Random(seed)
    names = [p.name for p in parquets]
    rng.shuffle(names)
    n_val = int(len(names) * val_frac)
    val_names = set(names[:n_val]) if n_val > 0 else set()
    train_names = set(names[n_val:]) if n_val > 0 else set(names)

    return {
        "train": [data_dir / n for n in sorted(train_names)],
        "val": [data_dir / n for n in sorted(val_names)],
    }


def _build_manifest(results, val_frac, seed):
    train_shards = [r[0] for r in results["train"]]
    val_shards = [r[0] for r in results["val"]]
    return {
        "train_shards": train_shards,
        "val_shards": val_shards,
        "n_train_games": sum(r[1] for r in results["train"]),
        "n_val_games": sum(r[1] for r in results["val"]),
        "n_train_frames": sum(r[2] for r in results["train"]),
        "n_val_frames": sum(r[2] for r in results["val"]),
        "val_frac": val_frac,
        "seed": seed,
    }


def _print_summary(manifest):
    print(f"\n  Manifest: {len(manifest['train_shards'])} train shards "
          f"({manifest['n_train_games']} games, {manifest['n_train_frames']:,} frames), "
          f"{len(manifest['val_shards'])} val shards "
          f"({manifest['n_val_games']} games, {manifest['n_val_frames']:,} frames)")


def stage_metadata(data_dir: Path, staging_dir: Path) -> None:
    staging_dir.mkdir(parents=True, exist_ok=True)
    for name in METADATA_FILES:
        src = data_dir / name
        if src.exists():
            shutil.copy2(src, staging_dir / name)
            print(f"  Copied {name}")


# ── Upload functions ─────────────────────────────────────────────────────────

def _upload_shard(api, repo_id, local_path, fname):
    """Upload a single shard file and delete it locally. Returns (fname, elapsed, size)."""
    t0 = time.time()
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=fname,
        repo_id=repo_id,
        repo_type="dataset",
    )
    dt = time.time() - t0
    size_gb = local_path.stat().st_size / 1e9
    local_path.unlink()
    return fname, dt, size_gb


def do_upload(staging_dir: Path, repo_id: str, clean: bool = False) -> None:
    """Upload using upload_large_folder (per-file commits, resumable)."""
    api = HfApi()
    if clean:
        try:
            api.delete_repo(repo_id=repo_id, repo_type="dataset")
            print(f"  Deleted old repo {repo_id}")
        except Exception:
            pass
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    print(f"  Uploading to https://huggingface.co/datasets/{repo_id} ...")

    upload_large_folder(
        folder_path=str(staging_dir),
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"  Upload complete.")


# ── Batch mode (all shards to disk, then upload) ────────────────────────────

def create_tensor_shards(
    data_dir: Path,
    staging_dir: Path,
    shard_gb: float,
    val_frac: float,
    seed: int,
    n_workers: int = 0,
) -> Dict:
    """Tensorize all parquets into per-game concatenated shards.

    Returns manifest dict for tensor_manifest.json.
    """
    norm_stats, cat_maps, stick_centers, shoulder_centers, fg, categorical_cols = _load_prereqs(data_dir)
    parquets = _get_split_parquets(data_dir, val_frac, seed)

    staging_dir.mkdir(parents=True, exist_ok=True)
    shard_bytes = int(shard_gb * 1e9)

    results = {"train": [], "val": []}
    for split, split_parquets in parquets.items():
        if not split_parquets:
            continue
        print(f"\n  Tensorizing {len(split_parquets)} {split} games "
              f"({n_workers} workers) ...")
        shard_infos = _tensorize_split(
            split_parquets, split, staging_dir, shard_bytes,
            categorical_cols, cat_maps, norm_stats, fg,
            stick_centers, shoulder_centers, n_workers,
        )
        results[split] = shard_infos

    manifest = _build_manifest(results, val_frac, seed)
    with open(staging_dir / "tensor_manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)

    _print_summary(manifest)
    return manifest


def _tensorize_split(
    split_parquets, split, out_dir, shard_bytes,
    categorical_cols, cat_maps, norm_stats, fg,
    stick_centers, shoulder_centers, n_workers,
):
    """Tensorize a split with optional multiprocessing."""
    buf_states: List = []
    buf_targets: List = []
    buf_offsets = [0]
    buf_bytes = 0
    shard_idx = 0
    n_processed = 0
    infos = []

    with _make_result_iter(
        split_parquets, categorical_cols, cat_maps, norm_stats, fg,
        stick_centers, shoulder_centers, n_workers,
    ) as result_iter:
        for result in result_iter:
            if result is None:
                n_processed += 1
                continue

            states, targets, n_frames = result
            buf_states.append(states)
            buf_targets.append(targets)
            buf_offsets.append(buf_offsets[-1] + n_frames)

            frame_bytes = sum(v.nelement() * v.element_size() for v in states.values())
            frame_bytes += sum(v.nelement() * v.element_size() for v in targets.values())
            buf_bytes += frame_bytes
            n_processed += 1

            if buf_bytes >= shard_bytes:
                info = flush_shard(buf_states, buf_targets, buf_offsets, split, shard_idx, out_dir)
                infos.append(info)
                shard_idx += 1
                buf_states, buf_targets, buf_offsets, buf_bytes = [], [], [0], 0

            if n_processed % 2000 == 0:
                print(f"      [{n_processed}/{len(split_parquets)}] games ...", flush=True)

        if buf_states:
            info = flush_shard(buf_states, buf_targets, buf_offsets, split, shard_idx, out_dir)
            infos.append(info)

    return infos


# ── Streaming mode (tensorize → upload → delete, pipelined) ─────────────────

def create_and_stream_upload(
    data_dir: Path,
    staging_dir: Path,
    shard_gb: float,
    val_frac: float,
    seed: int,
    repo_id: str,
    clean: bool,
    n_workers: int = 0,
) -> Dict:
    """Tensorize with multiprocessing, upload each shard async, delete immediately.

    Pipeline: workers tensorize games → main thread accumulates shards →
    background thread uploads completed shards. Disk stays bounded at ~2 * shard_gb.
    """
    api = HfApi()
    if clean:
        try:
            api.delete_repo(repo_id=repo_id, repo_type="dataset")
            print(f"  Deleted old repo {repo_id}")
        except Exception:
            pass
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

    norm_stats, cat_maps, stick_centers, shoulder_centers, fg, categorical_cols = _load_prereqs(data_dir)
    parquets = _get_split_parquets(data_dir, val_frac, seed)

    staging_dir.mkdir(parents=True, exist_ok=True)
    shard_bytes = int(shard_gb * 1e9)

    print(f"\n=== Uploading metadata ===")
    stage_metadata(data_dir, staging_dir)
    for name in METADATA_FILES:
        src = staging_dir / name
        if src.exists():
            api.upload_file(
                path_or_fileobj=str(src),
                path_in_repo=name,
                repo_id=repo_id,
                repo_type="dataset",
            )
            print(f"  Uploaded {name}")

    results = {"train": [], "val": []}
    for split, split_parquets in parquets.items():
        if not split_parquets:
            continue
        print(f"\n=== Tensorizing + streaming {len(split_parquets)} {split} games "
              f"({n_workers} workers) ===")
        shard_infos = _tensorize_split_streaming(
            split_parquets, split, staging_dir, shard_bytes,
            categorical_cols, cat_maps, norm_stats, fg,
            stick_centers, shoulder_centers, api, repo_id,
            n_workers,
        )
        results[split] = shard_infos

    manifest = _build_manifest(results, val_frac, seed)
    manifest_path = staging_dir / "tensor_manifest.json"
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    api.upload_file(
        path_or_fileobj=str(manifest_path),
        path_in_repo="tensor_manifest.json",
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"\n  Uploaded tensor_manifest.json")
    _print_summary(manifest)
    return manifest


def _tensorize_split_streaming(
    split_parquets, split, out_dir, shard_bytes,
    categorical_cols, cat_maps, norm_stats, fg,
    stick_centers, shoulder_centers, api, repo_id,
    n_workers,
):
    """Tensorize with multiprocessing + pipelined async upload."""
    buf_states: List = []
    buf_targets: List = []
    buf_offsets = [0]
    buf_bytes = 0
    shard_idx = 0
    n_processed = 0
    infos = []

    upload_pool = ThreadPoolExecutor(max_workers=1)
    pending_upload = None

    def _wait_pending():
        nonlocal pending_upload
        if pending_upload is not None:
            fname, dt, size_gb = pending_upload.result()
            print(f"      Uploaded {fname} ({size_gb:.2f} GB in {dt:.1f}s, "
                  f"{size_gb/max(dt,0.01):.2f} GB/s)", flush=True)
            pending_upload = None

    def _flush_and_submit():
        nonlocal buf_states, buf_targets, buf_offsets, buf_bytes, shard_idx
        _wait_pending()
        info = flush_shard(buf_states, buf_targets, buf_offsets, split, shard_idx, out_dir)
        fname = info[0]
        local_path = out_dir / fname
        pending_upload_ref = upload_pool.submit(
            _upload_shard, api, repo_id, local_path, fname,
        )
        nonlocal pending_upload
        pending_upload = pending_upload_ref
        infos.append(info)
        shard_idx += 1
        buf_states, buf_targets, buf_offsets, buf_bytes = [], [], [0], 0

    with _make_result_iter(
        split_parquets, categorical_cols, cat_maps, norm_stats, fg,
        stick_centers, shoulder_centers, n_workers,
    ) as result_iter:
        t_split = time.time()
        for result in result_iter:
            if result is None:
                n_processed += 1
                continue

            states, targets, n_frames = result
            buf_states.append(states)
            buf_targets.append(targets)
            buf_offsets.append(buf_offsets[-1] + n_frames)

            frame_bytes = sum(v.nelement() * v.element_size() for v in states.values())
            frame_bytes += sum(v.nelement() * v.element_size() for v in targets.values())
            buf_bytes += frame_bytes
            n_processed += 1

            if buf_bytes >= shard_bytes:
                _flush_and_submit()

            if n_processed % 2000 == 0:
                elapsed = time.time() - t_split
                rate = n_processed / elapsed
                eta = (len(split_parquets) - n_processed) / max(rate, 1)
                print(f"      [{n_processed}/{len(split_parquets)}] games  "
                      f"({rate:.0f} games/s, ETA {eta/60:.0f}m)", flush=True)

        if buf_states:
            _flush_and_submit()

    _wait_pending()
    upload_pool.shutdown()
    return infos


# ── Shared iterator factory ──────────────────────────────────────────────────

@contextmanager
def _make_result_iter(
    split_parquets, categorical_cols, cat_maps, norm_stats, fg,
    stick_centers, shoulder_centers, n_workers,
):
    """Yield an iterator of tensorize_game results (multiprocessing or serial)."""
    if n_workers > 1:
        pool = mp.Pool(
            n_workers,
            initializer=_init_worker,
            initargs=(categorical_cols, cat_maps, norm_stats, fg,
                      stick_centers, shoulder_centers),
        )
        try:
            yield pool.imap_unordered(_worker_tensorize, split_parquets, chunksize=32)
        finally:
            pool.close()
            pool.join()
    else:
        def _serial():
            for path in split_parquets:
                yield tensorize_game(
                    path, categorical_cols, cat_maps, norm_stats, fg,
                    stick_centers, shoulder_centers,
                )
        yield _serial()


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pretokenize parquet dataset into tensor shards and upload to HuggingFace")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory with parquets + metadata JSONs")
    parser.add_argument("--repo", required=True,
                        help="HuggingFace repo ID (e.g. erickfm/mimic-melee)")
    parser.add_argument("--shard-gb", type=float, default=4.0,
                        help="Target shard size in GB (default: 4)")
    parser.add_argument("--staging-dir", default=None,
                        help="Staging directory (default: <data-dir>_upload)")
    parser.add_argument("--keep-staging", action="store_true",
                        help="Don't delete staging dir after upload")
    parser.add_argument("--val-frac", type=float, default=0.1,
                        help="Fraction of games for validation (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-upload", action="store_true",
                        help="Tensorize only, skip upload")
    parser.add_argument("--upload-only", action="store_true",
                        help="Upload an existing staging directory (skip tensorization)")
    parser.add_argument("--clean", action="store_true",
                        help="Delete existing HF repo before uploading (fresh start)")
    parser.add_argument("--stream", action="store_true",
                        help="Upload each shard immediately after creation (low disk usage)")
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of parallel workers for tensorization (0=auto)")
    args = parser.parse_args()

    n_workers = args.workers if args.workers > 0 else min(os.cpu_count() or 4, 64)

    if args.upload_only:
        if not args.staging_dir:
            parser.error("--upload-only requires --staging-dir")
        staging_dir = Path(args.staging_dir)
    else:
        if not args.data_dir:
            parser.error("--data-dir is required unless --upload-only is used")
        data_dir = Path(args.data_dir)
        staging_dir = Path(args.staging_dir) if args.staging_dir else data_dir.parent / f"{data_dir.name}_upload"

    t0 = time.time()
    print(f"Workers: {n_workers}")

    if args.upload_only:
        print(f"\n=== Uploading staged directory to HuggingFace ===")
        print(f"  Repo: {args.repo}")
        do_upload(staging_dir, args.repo, clean=args.clean)
    elif args.stream and not args.no_upload:
        print(f"\n=== Streaming tensorize + upload ===")
        create_and_stream_upload(
            data_dir, staging_dir, args.shard_gb,
            args.val_frac, args.seed, args.repo, args.clean,
            n_workers,
        )
    else:
        print(f"\n=== Step 1: Creating tensor shards ===")
        manifest = create_tensor_shards(data_dir, staging_dir, args.shard_gb,
                                        args.val_frac, args.seed, n_workers)
        print(f"\n=== Step 2: Staging metadata ===")
        stage_metadata(data_dir, staging_dir)

        if not args.no_upload:
            print(f"\n=== Step 3: Uploading to HuggingFace ===")
            print(f"  Repo: {args.repo}")
            do_upload(staging_dir, args.repo, clean=args.clean)

    if not args.keep_staging and staging_dir.exists():
        print(f"\n=== Cleaning up staging directory ===")
        shutil.rmtree(staging_dir)
        print(f"  Removed {staging_dir}")

    elapsed = time.time() - t0
    h, m = divmod(int(elapsed), 3600)
    m, s = divmod(m, 60)
    print(f"\n=== Done in {h}h {m}m {s}s ===")


if __name__ == "__main__":
    main()
