"""Mined-state sampler: pull a batch of Prompts from events.parquet.

The sampler does three things:
  1. Query events.parquet via DuckDB with optional filters (character,
     stage, replay_id-exclusion set, l_cancel_result bucket).
  2. LRU-cache parsed Replay objects so we don't re-parse the same
     replay for every prompt drawn from it.
  3. Materialize the T-frame state_context window for each drawn row.

Returns a list of `Prompt` objects ready for the rollout harness.

Context window: T=180 frames (MIMIC's block size). For prompts near the
start of a replay, the context is clamped to what's available — the
rollout harness / encoder pads/slices on its own.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import duckdb

from rlvr.state.peppi_adapter import Replay
from rlvr.tasks.base import Prompt

# Match MIMIC's block size (see mimic/model.py MODEL_PRESETS['mimic']).
CONTEXT_LENGTH = 180


@lru_cache(maxsize=64)
def _load_replay(path: str) -> Replay:
    return Replay(Path(path))


def sample_states(
    events_path: Path | str,
    slp_dir: Path | str,
    task_id: str,
    n: int,
    seed: int = 0,
    character: Optional[int] = None,
    stage: Optional[int] = None,
    exclude_replay_ids: Optional[Set[str]] = None,
    l_cancel_result: Optional[int] = None,
    context_length: int = CONTEXT_LENGTH,
) -> List[Prompt]:
    """Sample n prompts for a task, with optional filters.

    Args:
        events_path: path to the events.parquet written by the tagger.
        slp_dir: directory containing the .slp files the parquet
            references (replay_id = .slp filename stem).
        task_id: which task to sample for.
        n: batch size.
        seed: RNG seed (DuckDB's SELECT ... USING SAMPLE is reproducible
            given a seed).
        character, stage: libmelee enum-value filters.
        exclude_replay_ids: set of replay_id strings to never return
            (eval-set holdout).
        l_cancel_result: optional filter on the replay_l_cancel_result
            field in task_metadata (1 or 2).
        context_length: number of frames of history to materialize for
            each prompt.

    Returns:
        list of Prompt, length n (or fewer if not enough rows match).
    """
    events_path = Path(events_path)
    slp_dir = Path(slp_dir)
    exclude = exclude_replay_ids or set()

    con = duckdb.connect(":memory:")
    # DuckDB can query parquet directly via parquet_scan().
    con.execute(
        f"CREATE VIEW events AS SELECT * FROM parquet_scan('{events_path}')"
    )

    where_clauses = [f"task_id = '{task_id}'"]
    if character is not None:
        where_clauses.append(f"character = {int(character)}")
    if stage is not None:
        where_clauses.append(f"stage = {int(stage)}")
    if l_cancel_result is not None:
        # metadata is stringified JSON; use regex for a targeted match
        # without parsing every row.
        where_clauses.append(
            f"task_metadata LIKE '%\"replay_l_cancel_result\": {int(l_cancel_result)}%'"
        )
    if exclude:
        # Parameterized IN-list
        excl_list = ",".join(f"'{r}'" for r in exclude)
        where_clauses.append(f"replay_id NOT IN ({excl_list})")
    where = " AND ".join(where_clauses)

    # Apply filters first, then sample from the filtered pool. Using
    # a subquery guarantees SAMPLE runs over the filtered set rather
    # than the full table. ORDER BY random_seed gives reproducible
    # ordering before the LIMIT.
    sql = (
        f"WITH filtered AS ("
        f"  SELECT replay_id, player_port, frame_idx, character, stage, "
        f"  task_metadata FROM events WHERE {where}"
        f") "
        f"SELECT * FROM filtered "
        f"ORDER BY hash(replay_id || CAST(frame_idx AS VARCHAR) || "
        f"              CAST(player_port AS VARCHAR) || CAST({int(seed)} AS VARCHAR)) "
        f"LIMIT {int(n)}"
    )
    rows = con.execute(sql).fetchall()
    con.close()

    prompts: List[Prompt] = []
    for replay_id, port, frame_id, char, stg, md_json in rows:
        md = json.loads(md_json)
        slp_path = slp_dir / f"{replay_id}.slp"
        if not slp_path.exists():
            # Corpus / events mismatch — skip silently.
            continue
        replay = _load_replay(str(slp_path))
        ctx = _materialize_context(replay, int(frame_id), context_length)
        prompts.append(Prompt(
            task_id=task_id,
            replay_id=replay_id,
            player_port=int(port),
            frame_idx=int(frame_id),
            state_context=ctx,
            task_metadata=md,
        ))
    return prompts


def _materialize_context(
    replay: Replay,
    frame_id: int,
    context_length: int,
) -> tuple:
    """Materialize up to context_length GameStates ending at frame_id
    (inclusive). If the replay doesn't have enough history, return a
    shorter tuple — the encoder/rollout handles the clamp."""
    import numpy as np
    # Find the dedup-space index for this frame_id.
    idx = int(np.searchsorted(replay.frame_ids, frame_id))
    if idx >= replay.num_frames or int(replay.frame_ids[idx]) != frame_id:
        return ()
    start = max(0, idx - context_length + 1)
    return tuple(replay.gamestate_at(i) for i in range(start, idx + 1))


def clear_cache() -> None:
    """Drop the LRU replay cache. Call between epochs if memory pressure."""
    _load_replay.cache_clear()
