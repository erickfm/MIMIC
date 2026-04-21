"""Corpus scanner: replay directory -> events.parquet.

For each .slp in --slp-dir, parse with peppi, run each registered
task's tag_frames, and write emitted FrameRows to a Parquet file.
Parallelized over files with ProcessPoolExecutor. Supports character
filtering (e.g. --character FOX) and incremental updates.

CLI:
    python -m rlvr.tagger.scan \\
        --slp-dir data/fox_ranked_slp \\
        --out data/rlvr/events_v0.1.parquet \\
        --task l_cancel_opportunity \\
        --character FOX \\
        --workers 8

Incremental mode: if --out already exists, only process replays whose
replay_id is not already present in the output. The event table
appends rather than overwriting.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Set

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Registering all built-in tasks. Must import BEFORE accessing registry.
import rlvr.tasks  # noqa: F401
from rlvr.state.gamestate import SCHEMA_VERSION
from rlvr.state.peppi_adapter import Replay
from rlvr.tasks.base import FrameRow
from rlvr.tasks.registry import get_task, list_tasks, registry_hash


def _rows_to_records(rows: List[FrameRow]) -> List[dict]:
    """Convert FrameRow dataclasses to dicts for Parquet write."""
    out = []
    for r in rows:
        out.append({
            "replay_id": r.replay_id,
            "player_port": r.player_port,
            "frame_idx": r.frame_idx,
            "task_id": r.task_id,
            "character": r.character,
            "stage": r.stage,
            "task_metadata": json.dumps(r.task_metadata),
        })
    return out


def _process_one(args):
    """Worker: parse one .slp, run all requested tasks, return records."""
    path, task_ids, character_filter_libm = args
    try:
        replay = Replay(Path(path))
    except Exception as e:
        return (path, None, f"parse_error:{type(e).__name__}:{e}")

    # Character filter: drop if none of the replay's players match.
    if character_filter_libm is not None:
        if character_filter_libm not in replay.player_characters:
            return (path, [], None)

    records: List[dict] = []
    for tid in task_ids:
        task = get_task(tid)
        rows = list(task.tag_frames(replay))
        if character_filter_libm is not None:
            # Keep only rows for the matching-character self-player.
            rows = [r for r in rows if r.character == character_filter_libm]
        records.extend(_rows_to_records(rows))
    return (path, records, None)


def scan_corpus(
    slp_dir: Path,
    out_path: Path,
    task_ids: List[str],
    character_filter: Optional[str] = None,
    workers: int = 8,
    limit: Optional[int] = None,
    verbose: bool = True,
) -> dict:
    """Scan a directory of .slp files and write events.parquet.

    Returns a metrics dict.
    """
    slp_dir = Path(slp_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Map character name to libmelee enum value
    character_filter_libm: Optional[int] = None
    if character_filter is not None:
        from melee import Character
        character_filter_libm = Character[character_filter].value

    # Collect .slp files
    files = sorted(slp_dir.rglob("*.slp"))
    if limit is not None:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"no .slp files found under {slp_dir}")

    # Incremental: skip already-processed replay_ids.
    already_processed: Set[str] = set()
    if out_path.exists():
        existing_table = pq.read_table(out_path, columns=["replay_id"])
        already_processed = set(existing_table["replay_id"].to_pylist())
        if verbose:
            print(f"[scan] out exists: {len(already_processed)} replay_ids already processed; appending")

    remaining = [f for f in files if f.stem not in already_processed]
    if verbose:
        print(f"[scan] {len(files)} .slp files found; {len(remaining)} to process")

    if not remaining:
        return {"processed": 0, "skipped": len(files), "errors": 0, "rows_written": 0}

    # Process
    task_ids_arg = list(task_ids)
    args_iter = [(str(f), task_ids_arg, character_filter_libm) for f in remaining]

    n_errors = 0
    n_rows = 0
    all_records: List[dict] = []
    t0 = time.time()
    if workers <= 1:
        for a in args_iter:
            path, records, err = _process_one(a)
            if err:
                n_errors += 1
                continue
            all_records.extend(records)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_process_one, a) for a in args_iter]
            for i, fut in enumerate(as_completed(futs)):
                path, records, err = fut.result()
                if err:
                    n_errors += 1
                    if verbose and n_errors <= 5:
                        print(f"[scan] error on {path}: {err}")
                    continue
                if records:
                    all_records.extend(records)
                if verbose and (i + 1) % 100 == 0:
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed
                    print(f"[scan] {i+1}/{len(futs)} done ({rate:.1f}/s), {len(all_records)} rows")
    n_rows = len(all_records)

    if all_records:
        df_new = pd.DataFrame(all_records)
        if out_path.exists():
            df_old = pd.read_parquet(out_path)
            df = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df = df_new
        table = pa.Table.from_pandas(df)
        # Stamp artifact metadata
        existing_meta = table.schema.metadata or {}
        meta = dict(existing_meta)
        meta.update({
            b"rlvr_schema_version": SCHEMA_VERSION.encode(),
            b"rlvr_registry_hash": registry_hash().encode(),
            b"rlvr_tasks": json.dumps(list_tasks()).encode(),
        })
        table = table.replace_schema_metadata(meta)
        pq.write_table(table, out_path)

    elapsed = time.time() - t0
    metrics = {
        "processed": len(remaining) - n_errors,
        "errors": n_errors,
        "rows_written": n_rows,
        "elapsed_sec": round(elapsed, 2),
        "out_path": str(out_path),
    }
    if verbose:
        print(f"[scan] done. {metrics}")
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--slp-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--task",
        action="append",
        default=None,
        help="Task ID(s) to run; repeatable. Default: all registered.",
    )
    ap.add_argument(
        "--character",
        default=None,
        help="libmelee Character name (e.g. FOX). Filters to replays "
             "containing this character on at least one port; only "
             "rows for that port are kept.",
    )
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 4))
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process first N .slp files (debugging).")
    args = ap.parse_args()

    task_ids = args.task or list_tasks()
    if not task_ids:
        print("no tasks registered", file=sys.stderr)
        sys.exit(2)

    metrics = scan_corpus(
        slp_dir=args.slp_dir,
        out_path=args.out,
        task_ids=task_ids,
        character_filter=args.character,
        workers=args.workers,
        limit=args.limit,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
