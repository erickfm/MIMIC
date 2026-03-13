#!/usr/bin/env python3
"""Package a parquet dataset into tar shards and upload to HuggingFace.

Supports two-stage upload: tar shards first, metadata JSONs added later
(useful when preprocess.py is still running).

Usage:
    # Full upload (tars + metadata in one shot)
    python3 upload_dataset.py --data-dir data/full --repo erickfm/frame-melee

    # Stage 1: create tars + upload (skip metadata if not ready)
    python3 upload_dataset.py --data-dir data/full --repo erickfm/frame-melee --skip-metadata

    # Stage 2: upload just the metadata JSONs
    python3 upload_dataset.py --data-dir data/full --repo erickfm/frame-melee --metadata-only
"""

import argparse
import shutil
import tarfile
import time
from pathlib import Path

from huggingface_hub import HfApi, upload_large_folder


METADATA_FILES = ["norm_stats.json", "cat_maps.json", "file_index.json"]


def create_shards(data_dir: Path, staging_dir: Path, shard_gb: float) -> list[Path]:
    """Split parquets into tar shards of roughly `shard_gb` GB each."""
    parquets = sorted(data_dir.glob("*.parquet"))
    if not parquets:
        raise RuntimeError(f"No parquet files in {data_dir}")

    sizes = [(p, p.stat().st_size) for p in parquets]
    total_bytes = sum(s for _, s in sizes)
    shard_bytes = int(shard_gb * 1e9)
    n_shards = max(1, (total_bytes + shard_bytes - 1) // shard_bytes)

    print(f"  {len(parquets)} parquets, {total_bytes / 1e9:.1f} GB total")
    print(f"  Target: {n_shards} shards of ~{shard_gb} GB each")

    staging_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = []
    shard_idx = 0
    current_size = 0

    tar_path = staging_dir / f"shard_{shard_idx:03d}.tar"
    tar = tarfile.open(tar_path, "w")
    shard_paths.append(tar_path)

    for i, (pf, sz) in enumerate(sizes):
        if current_size > 0 and current_size + sz > shard_bytes:
            tar.close()
            print(f"    shard_{shard_idx:03d}.tar  ({current_size / 1e9:.2f} GB)")
            shard_idx += 1
            tar_path = staging_dir / f"shard_{shard_idx:03d}.tar"
            tar = tarfile.open(tar_path, "w")
            shard_paths.append(tar_path)
            current_size = 0

        tar.add(str(pf), arcname=pf.name)
        current_size += sz

        if (i + 1) % 5000 == 0:
            print(f"    [{i+1}/{len(sizes)}] files packed ...", flush=True)

    tar.close()
    print(f"    shard_{shard_idx:03d}.tar  ({current_size / 1e9:.2f} GB)")

    return shard_paths


def stage_metadata(data_dir: Path, staging_dir: Path) -> None:
    """Copy JSON metadata files into the staging directory."""
    staging_dir.mkdir(parents=True, exist_ok=True)
    for name in METADATA_FILES:
        src = data_dir / name
        if not src.exists():
            raise FileNotFoundError(
                f"Missing {src} -- run preprocess.py --data-dir {data_dir} first"
            )
        shutil.copy2(src, staging_dir / name)
        print(f"  Copied {name}")


def do_upload(staging_dir: Path, repo_id: str) -> None:
    """Upload using upload_large_folder (per-file commits, resumable)."""
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    print(f"  Uploading to https://huggingface.co/datasets/{repo_id} ...")
    print(f"  Using upload_large_folder (per-file commits, resumable)")

    upload_large_folder(
        folder_path=str(staging_dir),
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"  Upload complete.")


def main():
    parser = argparse.ArgumentParser(description="Package and upload dataset to HuggingFace")
    parser.add_argument("--data-dir", required=True, help="Directory with parquets + metadata JSONs")
    parser.add_argument("--repo", required=True, help="HuggingFace repo ID (e.g. erickfm/frame-melee)")
    parser.add_argument("--shard-gb", type=float, default=4.0, help="Target shard size in GB (default: 4)")
    parser.add_argument("--staging-dir", default=None, help="Staging directory (default: <data-dir>_upload)")
    parser.add_argument("--keep-staging", action="store_true", help="Don't delete staging dir after upload")
    parser.add_argument("--skip-metadata", action="store_true",
                        help="Upload tar shards only, skip metadata JSONs (add them later with --metadata-only)")
    parser.add_argument("--metadata-only", action="store_true",
                        help="Upload only the 3 metadata JSONs (second stage)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    staging_dir = Path(args.staging_dir) if args.staging_dir else data_dir.parent / f"{data_dir.name}_upload"

    t0 = time.time()

    if args.metadata_only:
        print(f"\n=== Uploading metadata only ===")
        stage_metadata(data_dir, staging_dir)
        do_upload(staging_dir, args.repo)
        if not args.keep_staging:
            shutil.rmtree(staging_dir)
    else:
        print(f"\n=== Step 1: Creating tar shards ===")
        shard_paths = create_shards(data_dir, staging_dir, args.shard_gb)

        if not args.skip_metadata:
            print(f"\n=== Step 2: Staging metadata ===")
            stage_metadata(data_dir, staging_dir)

        print(f"\n=== Step {'2' if args.skip_metadata else '3'}: Uploading to HuggingFace ===")
        print(f"  Repo: {args.repo}")
        print(f"  Contents: {len(shard_paths)} shards" +
              ("" if args.skip_metadata else f" + {len(METADATA_FILES)} metadata files"))
        do_upload(staging_dir, args.repo)

        if not args.keep_staging:
            print(f"\n=== Cleaning up staging directory ===")
            shutil.rmtree(staging_dir)
            print(f"  Removed {staging_dir}")

    elapsed = time.time() - t0
    h, m = divmod(int(elapsed), 3600)
    m, s = divmod(m, 60)
    print(f"\n=== Done in {h}h {m}m {s}s ===")


if __name__ == "__main__":
    main()
