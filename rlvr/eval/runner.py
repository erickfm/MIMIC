"""Evaluate a checkpoint on a frozen eval set, produce a JSON report.

Loads the eval parquet (with its metadata), streams prompts through the
rollout harness at a fixed temperature (default greedy), applies the
verifier, aggregates per-task pass rate + 95% CI (Wilson score).

Report is diffable: sorted keys, canonical float format, no timestamps
inside per-task dict (moves to top-level `generated_at`).

CLI:
    python -m rlvr.eval.runner \\
        --ckpt checkpoints/fox-20260420-baseline-33k.pt \\
        --eval-set data/rlvr/eval_v0.1.parquet \\
        --data-dir hf_checkpoints/fox \\
        --slp-dir data/fox_ranked_slp \\
        --out reports/bc_fox.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

import rlvr.tasks  # noqa: F401  — register tasks
from rlvr.rollout import rollout
from rlvr.sampler.mined import _materialize_context, _load_replay
from rlvr.state.gamestate import SCHEMA_VERSION
from rlvr.tasks.base import Prompt
from rlvr.tasks.registry import get_verifier, registry_hash


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score 95% CI for a proportion. Preferred over normal
    approximation for small n or near-boundary p."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + (z ** 2) / n
    center = (p + (z ** 2) / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + (z ** 2) / (4 * n ** 2))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _build_prompts(df: pd.DataFrame, slp_dir: Path, context_length: int = 180):
    """Materialize Prompt objects from the eval dataframe."""
    prompts = []
    for _, row in df.iterrows():
        slp_path = slp_dir / f"{row['replay_id']}.slp"
        if not slp_path.exists():
            continue
        replay = _load_replay(str(slp_path))
        ctx = _materialize_context(replay, int(row["frame_idx"]), context_length)
        prompts.append(Prompt(
            task_id=row["task_id"],
            replay_id=row["replay_id"],
            player_port=int(row["player_port"]),
            frame_idx=int(row["frame_idx"]),
            state_context=ctx,
            task_metadata=json.loads(row["task_metadata"]),
        ))
    return prompts


def run_eval(
    eval_set_path: Path,
    slp_dir: Path,
    ckpt_path: Optional[Path] = None,
    data_dir: Optional[Path] = None,
    model=None,
    ctx: Optional[dict] = None,
    out_path: Optional[Path] = None,
    n_per_prompt: int = 1,
    temperature: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    device: str = "cuda",
) -> dict:
    """Run eval. Either pass (ckpt_path + data_dir) to load from disk, or
    (model + ctx) to use a live in-training model. For training-loop
    invocation prefer the latter to avoid reloading the base ckpt."""
    from tools.inference_utils import load_inference_context, load_mimic_model

    # Load eval set + its metadata
    table = pq.read_table(eval_set_path)
    parquet_meta = table.schema.metadata or {}
    eval_version = parquet_meta.get(b"rlvr_eval_set_version", b"?").decode()
    df = table.to_pandas()

    if model is None:
        assert ckpt_path is not None and data_dir is not None, (
            "either model+ctx or ckpt_path+data_dir must be provided"
        )
        model, cfg = load_mimic_model(str(ckpt_path), device)
        ctx = load_inference_context(data_dir)
    assert ctx is not None, "ctx required when passing a live model"

    # Dummy ref model — KL unused during eval.
    ref_model = model
    model.eval()

    t0 = time.time()

    # Group eval prompts by task_id
    per_task_results: dict = {}
    for task_id, sub in df.groupby("task_id"):
        verifier = get_verifier(task_id)
        n_rows = len(sub)
        k_success = 0
        n_scored = 0
        for start in range(0, n_rows, batch_size):
            chunk = sub.iloc[start:start + batch_size]
            prompts = _build_prompts(chunk, Path(slp_dir))
            if not prompts:
                continue
            with torch.no_grad():
                rb = rollout(
                    model, ref_model, prompts,
                    n_per_prompt=n_per_prompt,
                    ctx=ctx, temperature=temperature,
                    seed=seed + start, device=device,
                )
            for i, prompt in enumerate(rb.prompts):
                # With n_per_prompt samples, each prompt gets N rollouts.
                # For eval we report per-sample pass rate (each rollout
                # counts). With N=1 + greedy, it's one pass/fail per
                # prompt.
                for j in range(n_per_prompt):
                    ctrl = rb.sampled_ctrls[i * n_per_prompt + j]
                    reward = verifier(prompt, ctrl)
                    if reward > 0.5:
                        k_success += 1
                    n_scored += 1

        pass_rate = k_success / n_scored if n_scored else 0.0
        ci_lo, ci_hi = _wilson_ci(k_success, n_scored)
        per_task_results[task_id] = {
            "pass_rate": round(pass_rate, 6),
            "n": n_scored,
            "successes": k_success,
            "ci_95_low": round(ci_lo, 6),
            "ci_95_high": round(ci_hi, 6),
        }

    runtime = time.time() - t0
    report = {
        "ckpt": str(ckpt_path) if ckpt_path is not None else "<in-memory>",
        "eval_set_path": str(eval_set_path),
        "eval_set_version": eval_version,
        "schema_version": SCHEMA_VERSION,
        "task_registry_hash": registry_hash(),
        "temperature": temperature,
        "n_per_prompt": n_per_prompt,
        "seed": seed,
        "device": device,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "runtime_sec": round(runtime, 2),
        "per_task": per_task_results,
    }

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--eval-set", required=True, type=Path)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--slp-dir", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--n-per-prompt", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    report = run_eval(
        eval_set_path=args.eval_set,
        slp_dir=args.slp_dir,
        ckpt_path=args.ckpt,
        data_dir=args.data_dir,
        out_path=args.out,
        n_per_prompt=args.n_per_prompt,
        temperature=args.temperature,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
