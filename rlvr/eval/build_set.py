"""Build a frozen eval set from events.parquet.

Strategy:
  1. Split replay_ids into train/eval by a deterministic hash seed
     (default 10% eval). All prompts in the eval replays are eligible.
  2. Restrict to eval replay_ids.
  3. Dedupe to at most one prompt per (replay_id, landing_frame_idx,
     player_port) — i.e. one per aerial attack, not 7 per aerial — so
     eval-set passes aren't inflated by within-window correlation.
  4. Stratify by aerial_action_state to match natural distribution.
  5. Sample N (default 500) prompts with a fixed seed.
  6. Write eval_v0.1.parquet with version metadata.

Once written, the file is immutable. New versions get new filenames
(eval_v0.2.parquet, etc.). The eval loader asserts schema version
compatibility.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Optional, Set

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from rlvr.state.gamestate import SCHEMA_VERSION
from rlvr.tasks.registry import registry_hash

EVAL_SET_VERSION = "v0.1"


def split_replay_ids(replay_ids: list, eval_frac: float = 0.1,
                     split_seed: str = "rlvr-eval-v0.1") -> Set[str]:
    """Deterministically partition replay_ids. Returns the eval set.

    Uses SHA-256 of (seed, replay_id) mod 10000 to decide membership.
    Same split_seed -> same partition across machines / time.
    """
    eval_ids = set()
    thresh = int(eval_frac * 10000)
    for rid in replay_ids:
        h = hashlib.sha256(f"{split_seed}::{rid}".encode()).hexdigest()
        bucket = int(h[:8], 16) % 10000
        if bucket < thresh:
            eval_ids.add(rid)
    return eval_ids


def build_eval_set(
    events_path: Path,
    out_path: Path,
    n: int = 500,
    eval_frac: float = 0.1,
    split_seed: str = "rlvr-eval-v0.1",
    sample_seed: int = 12345,
    at_offset: int = -4,
    task_id: Optional[str] = None,
) -> dict:
    """Build and write the eval set. Returns a metrics dict.

    `task_id` scopes the eval set to a single task. If None, all tasks
    in the events parquet are included. The L-cancel path applies an
    `offset_to_landing == at_offset` filter and stratifies by
    `aerial_action_state`; other tasks get uniform random sampling
    (no stratification) within the eval-replay pool.
    """
    events = pd.read_parquet(events_path)
    md = events["task_metadata"].apply(json.loads)

    if task_id is not None:
        events = events[events["task_id"] == task_id].reset_index(drop=True)
        md = md.loc[events.index]

    all_rids = sorted(events["replay_id"].unique().tolist())
    eval_rids = split_replay_ids(all_rids, eval_frac, split_seed)
    df = events[events["replay_id"].isin(eval_rids)].copy().reset_index(drop=True)
    md = md.loc[df.index]

    rng = np.random.default_rng(sample_seed)
    stratification: dict = {}

    if task_id == "l_cancel_opportunity":
        # Legacy stratified path.
        df["offset_to_landing"] = md.apply(lambda m: m["offset_to_landing"])
        df["aerial_action_state"] = md.apply(lambda m: m["aerial_action_state"])
        df = df[df["offset_to_landing"] == at_offset].copy()

        action_counts = df["aerial_action_state"].value_counts()
        total_pool = len(df)
        if total_pool < n:
            n = total_pool
            print(f"[warn] eval pool ({total_pool}) smaller than n; using all")
        quotas = {}
        assigned = 0
        for action, count in action_counts.items():
            q = int(round(n * count / total_pool))
            quotas[action] = q
            assigned += q
        if assigned != n:
            top = action_counts.index[0]
            quotas[top] += (n - assigned)

        selected_idx = []
        for action, q in quotas.items():
            pool = df[df["aerial_action_state"] == action]
            q = min(q, len(pool))
            idx = rng.choice(pool.index.values, size=q, replace=False)
            selected_idx.extend(idx.tolist())
        stratification = {str(k): int(v) for k, v in quotas.items()}
    else:
        # Uniform random sample for other tasks.
        total_pool = len(df)
        if total_pool < n:
            n = total_pool
            print(f"[warn] eval pool ({total_pool}) smaller than n; using all")
        selected_idx = rng.choice(df.index.values, size=n, replace=False).tolist()
        stratification = {"uniform": int(n)}

    eval_df = df.loc[selected_idx].copy().reset_index(drop=True)
    out_cols = ["replay_id", "player_port", "frame_idx", "task_id",
                "character", "stage", "task_metadata"]
    eval_df = eval_df[out_cols]
    eval_df["eval_set_version"] = EVAL_SET_VERSION

    table = pa.Table.from_pandas(eval_df)
    meta = dict(table.schema.metadata or {})
    meta.update({
        b"rlvr_schema_version": SCHEMA_VERSION.encode(),
        b"rlvr_registry_hash": registry_hash().encode(),
        b"rlvr_eval_set_version": EVAL_SET_VERSION.encode(),
        b"rlvr_frozen_at": dt.datetime.now(dt.timezone.utc).isoformat().encode(),
        b"rlvr_split_seed": split_seed.encode(),
        b"rlvr_sample_seed": str(sample_seed).encode(),
        b"rlvr_at_offset": str(at_offset).encode(),
        b"rlvr_stratification": json.dumps(stratification).encode(),
        b"rlvr_eval_replay_ids": json.dumps(sorted(list(eval_rids))).encode(),
    })
    table = table.replace_schema_metadata(meta)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path)

    return {
        "eval_set_version": EVAL_SET_VERSION,
        "n_prompts": len(eval_df),
        "n_eval_replays": len(eval_rids),
        "n_total_replays": len(all_rids),
        "stratification": stratification,
        "out_path": str(out_path),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--events", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--eval-frac", type=float, default=0.1)
    ap.add_argument("--split-seed", default="rlvr-eval-v0.1")
    ap.add_argument("--sample-seed", type=int, default=12345)
    ap.add_argument("--at-offset", type=int, default=-4,
                    help="which offset-to-landing frame to keep per aerial (l_cancel only)")
    ap.add_argument("--task", default=None,
                    help="filter to a single task_id; enables per-task sampling strategy")
    args = ap.parse_args()
    metrics = build_eval_set(
        events_path=args.events, out_path=args.out,
        n=args.n, eval_frac=args.eval_frac,
        split_seed=args.split_seed, sample_seed=args.sample_seed,
        at_offset=args.at_offset, task_id=args.task,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
