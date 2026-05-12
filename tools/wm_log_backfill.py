#!/usr/bin/env python3
"""Replay a `train_wm.py` stdout log into a fresh WandB run with proper step alignment.

Use when a training run was launched with the old (step-less)
`wandb_run.log(...)` calls — the wandb default-step axis ends up at
~1-per-N train-steps, which makes the run incomparable to other runs
whose default axis matches real training step.

This tool re-emits the training + val metrics to a new wandb run
passing `step=step` on every call, producing a run whose default
axis IS the real training step.

    python3 tools/wm_log_backfill.py \
        --log /tmp/fox_wm_oppsym.log \
        --run-name fox-wm-20260424-oppsym-baseline-fixed \
        --project mimic-wm \
        --tags baseline fox oppinputs symmetric
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple


# `step 13050/32768  total 0.3970  act_s 0.057  act_o 0.063  num_s 0.132 ... lr 1.97e-04  1345 sam/s`
TRAIN_RE = re.compile(
    r"^step\s+(?P<step>\d+)/(?P<total>\d+)\s+"
    r"total\s+(?P<total_loss>[-\d.]+)\s+"
    r"act_s\s+(?P<act_s>[-\d.]+)\s+act_o\s+(?P<act_o>[-\d.]+)\s+"
    r"num_s\s+(?P<num_s>[-\d.]+)\s+num_o\s+(?P<num_o>[-\d.]+)\s+"
    r"flg_s\s+(?P<flg_s>[-\d.]+)\s+flg_o\s+(?P<flg_o>[-\d.]+)"
    r"(?:\s+elp_s\s+(?P<elp_s>[-\d.]+)\s+elp_o\s+(?P<elp_o>[-\d.]+))?"
    r"\s+lr\s+(?P<lr>[-\d.eE+]+)\s+"
    r"(?P<sps>[-\d.]+)\s+sam/s"
)

VAL_HEADER_RE = re.compile(r"^running val @ step (?P<step>\d+)")
VAL_METRICS_RE = re.compile(r"(\w+)=([-\d.]+(?:[eE][-+]?\d+)?)")


def parse_log(log_path: Path) -> Tuple[List[Dict], List[Dict]]:
    train_records: List[Dict] = []
    val_records: List[Dict] = []
    pending_val_step: int | None = None

    with open(log_path) as fh:
        for line in fh:
            line = line.rstrip("\n")

            m = TRAIN_RE.match(line)
            if m:
                d = m.groupdict()
                rec = {
                    "step": int(d["step"]),
                    "total": float(d["total_loss"]),
                    "action_self": float(d["act_s"]),
                    "action_opp": float(d["act_o"]),
                    "numeric_self": float(d["num_s"]),
                    "numeric_opp": float(d["num_o"]),
                    "flags_self": float(d["flg_s"]),
                    "flags_opp": float(d["flg_o"]),
                    "lr": float(d["lr"]),
                    "samples_per_sec": float(d["sps"]),
                }
                if d.get("elp_s") is not None:
                    rec["action_elapsed_self"] = float(d["elp_s"])
                    rec["action_elapsed_opp"] = float(d["elp_o"])
                train_records.append(rec)
                continue

            h = VAL_HEADER_RE.match(line.strip())
            if h:
                pending_val_step = int(h.group("step"))
                continue

            if pending_val_step is not None and "val_loss=" in line:
                pairs = VAL_METRICS_RE.findall(line)
                val = {"step": pending_val_step}
                for k, v in pairs:
                    try:
                        val[k] = float(v)
                    except ValueError:
                        continue
                val_records.append(val)
                pending_val_step = None
                continue

    return train_records, val_records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--project", default="mimic-wm")
    ap.add_argument("--tags", nargs="*", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.log.exists():
        print(f"log not found: {args.log}", file=sys.stderr)
        sys.exit(1)

    train_records, val_records = parse_log(args.log)
    print(f"parsed {len(train_records)} train records, "
          f"{len(val_records)} val records")
    if train_records:
        last = train_records[-1]
        print(f"  last train step: {last['step']}  total={last['total']:.4f}")
    if val_records:
        last = val_records[-1]
        print(f"  last val step:   {last['step']}  val_loss={last.get('val_loss'):.4f}")

    if args.dry_run:
        return

    import wandb
    run = wandb.init(
        project=args.project, name=args.run_name,
        tags=args.tags or [],
        config={"source": "backfilled-from-log-with-step",
                "log_path": str(args.log), "run_name": args.run_name},
    )
    print(f"wandb run: {run.name}  url={run.url}")

    # Merge + order by step. Train logged first at a given step so val
    # (same step) dedupes on top, matching the live ordering.
    events: List[Tuple[int, Dict]] = []
    for r in train_records:
        events.append((r["step"],
                       {f"train/{k}": v for k, v in r.items() if k != "step"}))
    for r in val_records:
        events.append((r["step"],
                       {k: v for k, v in r.items() if k != "step"}))
    events.sort(key=lambda x: x[0])

    for step, payload in events:
        wandb.log(payload, step=step)

    run.finish()
    print(f"logged {len(train_records)} train + {len(val_records)} val points")


if __name__ == "__main__":
    main()
