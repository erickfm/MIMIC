#!/usr/bin/env python3
"""Download Fox .slp files from HuggingFace dataset.

Downloads files batch by batch to manage rate limiting.
Resumes from where it left off.
"""
import os
import time
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

REPO = "erickfm/slippi-public-dataset-v3.7"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "fox_slp" / "FOX"
BATCH_DIRS = ["FOX/batch_00", "FOX/batch_01", "FOX/batch_02", "FOX/batch_03", "FOX/batch_04", "FOX/batch_05"]

api = HfApi()
OUT_DIR.mkdir(parents=True, exist_ok=True)

total_downloaded = 0
total_skipped = 0

for batch_dir in BATCH_DIRS:
    print(f"\n=== Listing {batch_dir} ===")
    try:
        files = list(api.list_repo_tree(REPO, repo_type="dataset", path_in_repo=batch_dir))
    except Exception as e:
        print(f"  ERROR listing: {e}")
        print("  Waiting 60s for rate limit...")
        time.sleep(60)
        files = list(api.list_repo_tree(REPO, repo_type="dataset", path_in_repo=batch_dir))

    slp_files = [f for f in files if hasattr(f, 'size') and f.path.endswith('.slp')]
    print(f"  Found {len(slp_files)} .slp files")

    for i, f in enumerate(slp_files):
        fname = Path(f.path).name
        out_path = OUT_DIR / fname
        if out_path.exists() and out_path.stat().st_size > 100:
            total_skipped += 1
            continue

        try:
            local = hf_hub_download(
                REPO, f.path, repo_type="dataset",
                local_dir=str(OUT_DIR.parent),
            )
            total_downloaded += 1
        except Exception as e:
            if "429" in str(e):
                print(f"  Rate limited at file {i}, waiting 120s...")
                time.sleep(120)
                try:
                    local = hf_hub_download(
                        REPO, f.path, repo_type="dataset",
                        local_dir=str(OUT_DIR.parent),
                    )
                    total_downloaded += 1
                except Exception as e2:
                    print(f"  Failed again: {e2}")
            else:
                print(f"  ERROR: {e}")

        if (total_downloaded + total_skipped) % 500 == 0:
            print(f"  Progress: {total_downloaded} downloaded, {total_skipped} skipped", flush=True)

    print(f"  Batch done: {total_downloaded} total downloaded, {total_skipped} skipped")

# Also get top-level FOX/*.slp files
print(f"\n=== Listing FOX/ (top-level) ===")
try:
    files = list(api.list_repo_tree(REPO, repo_type="dataset", path_in_repo="FOX"))
    slp_files = [f for f in files if hasattr(f, 'size') and f.path.endswith('.slp')]
    print(f"  Found {len(slp_files)} .slp files")

    for i, f in enumerate(slp_files):
        fname = Path(f.path).name
        out_path = OUT_DIR / fname
        if out_path.exists() and out_path.stat().st_size > 100:
            total_skipped += 1
            continue

        try:
            local = hf_hub_download(
                REPO, f.path, repo_type="dataset",
                local_dir=str(OUT_DIR.parent),
            )
            total_downloaded += 1
        except Exception as e:
            if "429" in str(e):
                print(f"  Rate limited at file {i}, waiting 120s...")
                time.sleep(120)
            else:
                print(f"  ERROR: {e}")

        if (total_downloaded + total_skipped) % 500 == 0:
            print(f"  Progress: {total_downloaded} downloaded, {total_skipped} skipped", flush=True)
except Exception as e:
    print(f"  ERROR: {e}")

print(f"\n=== Done: {total_downloaded} downloaded, {total_skipped} skipped ===")
