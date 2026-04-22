#!/usr/bin/env python3
"""Shard and upload one ranked archive to erickfm/melee-ranked-replays.

Extracts a .7z/.zip archive of ranked .slp replays, parses each header with
peppi_py to get both player characters, groups replays into (CHAR, rank_pair)
buckets (ZELDA/SHEIK -> ZELDA_SHEIK, POPO/NANA -> ICE_CLIMBERS, each replay
routed into both players' buckets), tars each bucket, and uploads to
{CHAR}/{CHAR}_{combo}_a{N}.tar.gz. Writes and uploads metadata_a{N}.json.
Skips buckets whose path already exists on HF — cheap resume.

Usage:
    python tools/shard_and_upload_ranked.py \
        --archive /home/erick/Documents/melee/ranked-anonymized-1-116248.7z \
        --archive-id 1 \
        --workdir /home/erick/Documents/melee/staging_a1
"""
import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import peppi_py
from huggingface_hub import HfApi
from melee import Character

# huggingface_hub uploads have no default socket timeout. Without this, a
# stalled TCP session can wedge the script for hours. 10 min is long enough
# for a slow but alive upload; anything longer should trigger a retry.
socket.setdefaulttimeout(600)

REPO = "erickfm/melee-ranked-replays"
RANK_RE = re.compile(r"^(platinum|diamond|master)-(platinum|diamond|master)-[0-9a-f]+\.slp$")

BAD_CHARS = {"WIREFRAME_MALE", "WIREFRAME_FEMALE", "GIGA_BOWSER", "SANDBAG", "UNKNOWN_CHARACTER"}
CHAR_NAME = {c.value: c.name for c in Character}
CHAR_NAME[Character.ZELDA.value] = "ZELDA_SHEIK"
CHAR_NAME[Character.SHEIK.value] = "ZELDA_SHEIK"
CHAR_NAME[Character.POPO.value] = "ICE_CLIMBERS"
CHAR_NAME[Character.NANA.value] = "ICE_CLIMBERS"


def _parse_one(slp_path: str):
    try:
        g = peppi_py.read_slippi(slp_path, skip_frames=True)
        players = [p for p in g.start.players if p is not None]
        if len(players) != 2:
            return (slp_path, None, None, f"n_players={len(players)}")
        chars = []
        for p in players:
            name = CHAR_NAME.get(p.character)
            if name is None or name in BAD_CHARS:
                return (slp_path, None, None, f"bad_char:{p.character}")
            chars.append(name)
        return (slp_path, chars[0], chars[1], None)
    except Exception as e:
        return (slp_path, None, None, f"err:{type(e).__name__}:{e}")


def _parse_rank(fn: str):
    m = RANK_RE.match(fn)
    return f"{m.group(1)}-{m.group(2)}" if m else None


def extract_archive(archive: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".7z":
        cmd = ["7z", "x", "-y", f"-o{dst}", str(archive)]
    elif archive.suffix == ".zip":
        cmd = ["unzip", "-q", "-o", "-d", str(dst), str(archive)]
    else:
        raise ValueError(f"unsupported archive: {archive}")
    print(f"Extracting {archive.name} -> {dst}", flush=True)
    t0 = time.time()
    subprocess.check_call(cmd)
    print(f"  extracted in {time.time() - t0:.0f}s", flush=True)


def find_slp_files(root: Path):
    return sorted(str(p) for p in root.rglob("*.slp"))


def _gunzip_one(src: str):
    import gzip
    dst = src[:-3]  # strip .gz
    try:
        with gzip.open(src, "rb") as gf, open(dst, "wb") as out:
            shutil.copyfileobj(gf, out, length=1 << 20)
        os.remove(src)
        return (src, None)
    except Exception as e:
        return (src, f"{type(e).__name__}:{e}")


def decompress_slp_gz(root: Path, jobs: int):
    gz_files = sorted(str(p) for p in root.rglob("*.slp.gz"))
    if not gz_files:
        return
    print(f"Decompressing {len(gz_files)} .slp.gz files", flush=True)
    t0 = time.time()
    bad = 0
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for i, (src, err) in enumerate(ex.map(_gunzip_one, gz_files, chunksize=64)):
            if err:
                bad += 1
                if bad <= 5:
                    print(f"  gunzip failed: {src}: {err}", flush=True)
            if (i + 1) % 5000 == 0:
                print(f"  gunzipped {i + 1}/{len(gz_files)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"Decompressed in {time.time() - t0:.0f}s ({bad} failures)", flush=True)


def already_uploaded(api: HfApi):
    try:
        return set(api.list_repo_files(REPO, repo_type="dataset"))
    except Exception:
        return set()


def upload_file(api: HfApi, local: Path, path_in_repo: str):
    for attempt in range(5):
        try:
            api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=path_in_repo,
                repo_id=REPO,
                repo_type="dataset",
                commit_message=f"add {path_in_repo}",
            )
            return
        except Exception as e:
            wait = min(300, 2 ** attempt * 10)
            print(f"  upload {path_in_repo} failed ({e}); retry in {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"upload failed after 5 attempts: {path_in_repo}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True, type=Path)
    ap.add_argument("--archive-id", required=True, type=str)
    ap.add_argument("--workdir", required=True, type=Path)
    ap.add_argument("--extract-dir", type=Path, default=None)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--skip-cleanup", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="debug: only process first N slp")
    ap.add_argument("--only-chars", type=str, default="",
                    help="comma-separated character names; skip buckets for other chars")
    ap.add_argument("--refresh-before-upload", action="store_true",
                    help="re-fetch HF file list before each bucket upload (for parallel workers)")
    ap.add_argument("--skip-metadata-upload", action="store_true",
                    help="don't upload metadata_aN.json (leave it to the main worker)")
    args = ap.parse_args()

    only_chars = set(c.strip() for c in args.only_chars.split(",") if c.strip()) if args.only_chars else None

    args.workdir.mkdir(parents=True, exist_ok=True)
    extract_dir = args.extract_dir or (args.workdir / "extracted")

    if not args.skip_extract:
        if extract_dir.exists() and any(extract_dir.iterdir()):
            print(f"Extract dir {extract_dir} is non-empty; pass --skip-extract to reuse.", file=sys.stderr)
            sys.exit(1)
        extract_archive(args.archive, extract_dir)

    # Some archives (a4/a5/a6) contain individually gzipped .slp.gz files; decompress in place.
    decompress_slp_gz(extract_dir, args.jobs)

    slp_files = find_slp_files(extract_dir)
    print(f"Found {len(slp_files)} .slp files")
    if not slp_files:
        print(f"ERROR: no .slp files found under {extract_dir}", file=sys.stderr)
        sys.exit(1)
    if args.limit:
        slp_files = slp_files[: args.limit]
        print(f"  limited to {len(slp_files)} for debug")

    buckets: dict[tuple[str, str], list[str]] = defaultdict(list)
    metadata = []
    bad = 0
    bad_reasons: dict[str, int] = defaultdict(int)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futures = {ex.submit(_parse_one, p): p for p in slp_files}
        for i, f in enumerate(as_completed(futures)):
            path, c1, c2, err = f.result()
            fn = os.path.basename(path)
            rank = _parse_rank(fn)
            if err or rank is None or c1 is None or c2 is None:
                bad += 1
                reason = err or ("bad_filename" if rank is None else "no_chars")
                bad_reasons[reason.split(":", 1)[0]] += 1
                continue
            buckets[(c1, rank)].append(path)
            if c1 != c2:
                buckets[(c2, rank)].append(path)
            metadata.append({"filename": fn, "p1": c1, "p2": c2, "rank": rank, "archive": args.archive_id})
            if (i + 1) % 5000 == 0:
                print(f"  parsed {i + 1}/{len(slp_files)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"Parsed {len(metadata)} good, {bad} bad in {time.time() - t0:.0f}s")
    if bad_reasons:
        print(f"  bad breakdown: {dict(bad_reasons)}")
    print(f"Buckets: {len(buckets)}")

    meta_path = args.workdir / f"metadata_a{args.archive_id}.json"
    meta_path.write_text(json.dumps(metadata))
    print(f"Wrote {meta_path} ({len(metadata)} entries)")

    if args.dry_run:
        for (char, rank), files in sorted(buckets.items()):
            print(f"  {char}_{rank}: {len(files)} files")
        return

    api = HfApi()
    already = already_uploaded(api)
    tar_dir = args.workdir / "tars"
    tar_dir.mkdir(exist_ok=True)

    bucket_items = sorted(buckets.items())
    if only_chars is not None:
        bucket_items = [b for b in bucket_items if b[0][0] in only_chars]
        print(f"Filtered to {len(bucket_items)} buckets via --only-chars={sorted(only_chars)}")
    for idx, ((char, rank), files) in enumerate(bucket_items, 1):
        tar_name = f"{char}_{rank}_a{args.archive_id}.tar.gz"
        path_in_repo = f"{char}/{tar_name}"
        print(f"[{idx}/{len(bucket_items)}] {tar_name}: {len(files)} files", flush=True)
        if args.refresh_before_upload:
            already = already_uploaded(api)
        if path_in_repo in already:
            print(f"  skip (already uploaded)")
            continue
        local_tar = tar_dir / tar_name
        t1 = time.time()
        with tarfile.open(local_tar, "w:gz", compresslevel=6) as t:
            for src in files:
                t.add(src, arcname=os.path.basename(src))
        size_mb = local_tar.stat().st_size / 1e6
        print(f"  tarred in {time.time() - t1:.0f}s -> {size_mb:.1f} MB", flush=True)
        t2 = time.time()
        upload_file(api, local_tar, path_in_repo)
        print(f"  uploaded in {time.time() - t2:.0f}s", flush=True)
        local_tar.unlink()

    if args.skip_metadata_upload:
        print("Skipping metadata upload (--skip-metadata-upload)")
    else:
        print(f"Uploading metadata_a{args.archive_id}.json")
        upload_file(api, meta_path, f"metadata/metadata_a{args.archive_id}.json")

    if not args.skip_cleanup and only_chars is None:
        print(f"Removing {extract_dir}")
        shutil.rmtree(extract_dir)
        if tar_dir.exists() and not any(tar_dir.iterdir()):
            tar_dir.rmdir()
    print("Done.")


if __name__ == "__main__":
    main()
