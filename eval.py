#!/usr/bin/env python3
"""eval.py -- Evaluate checkpoints with new metrics (btn_f1, cdir_active_acc, etc.)

Loads each checkpoint, builds val dataset from subset data, and computes
metrics over MAX_VAL_BATCHES batches.  Outputs a TSV table.
"""

import sys
import os
import math
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast

from model import FramePredictor, ModelConfig
from dataset import MeleeFrameDatasetWithDelay
from train import compute_loss, collate_fn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_DTYPE = torch.bfloat16
MAX_VAL_BATCHES = 200
BATCH_SIZE = 128
NUM_WORKERS = 8


def load_checkpoint(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw_cfg = ckpt["config"]

    cfg_fields = {f.name for f in ModelConfig.__dataclass_fields__.values()}
    filtered = {k: v for k, v in raw_cfg.items() if k in cfg_fields}
    filtered.setdefault("no_opp_inputs", False)
    filtered.setdefault("btn_loss", "bce")
    cfg = ModelConfig(**filtered)

    model = FramePredictor(cfg).to(DEVICE)
    sd = ckpt["model_state_dict"]
    cleaned = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(cleaned)
    model.eval()

    norm_stats = ckpt.get("norm_stats", {})
    step = ckpt.get("global_step", 0)
    return model, cfg, norm_stats, step


def build_val_dataset(data_dir: str, seq_len: int, norm_stats: dict,
                      no_opp_inputs: bool):
    ds = MeleeFrameDatasetWithDelay(
        parquet_dir=data_dir,
        sequence_length=seq_len,
        reaction_delay=1,
        split="val",
        norm_stats=norm_stats,
        no_opp_inputs=no_opp_inputs,
    )
    return ds


def evaluate(model, cfg, val_dl, max_batches=200):
    keys = ["total", "loss_main", "loss_l", "loss_r", "loss_cdir", "loss_btn",
            "cdir_acc", "btn_acc", "btn_f1", "btn_precision", "btn_recall",
            "cdir_active_acc"]
    sums = {k: 0.0 for k in keys}
    ct = 0

    with torch.no_grad():
        for vs, vt in val_dl:
            for k, v in vs.items():
                vs[k] = v.to(DEVICE, non_blocking=True)
            for k, v in vt.items():
                vt[k] = v.to(DEVICE, non_blocking=True)
            with autocast("cuda", dtype=AMP_DTYPE):
                preds = model(vs)
            metrics, task_losses = compute_loss(
                preds, vt,
                stick_loss_type=cfg.stick_loss,
                stick_bins=cfg.stick_bins,
                btn_loss_type=cfg.btn_loss,
                delta_targets=cfg.delta_targets,
                state=vs,
            )
            batch_total = sum(t.item() for t in task_losses)
            if math.isfinite(batch_total):
                sums["total"] += batch_total
                for mk in keys[1:]:
                    sums[mk] += metrics[mk]
                ct += 1
            if ct >= max_batches:
                break

    if ct == 0:
        return None
    return {k: sums[k] / ct for k in keys}


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoints")
    parser.add_argument("checkpoints", nargs="+", help="Checkpoint paths")
    parser.add_argument("--data-dir", default="./data/subset")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-batches", type=int, default=MAX_VAL_BATCHES)
    parser.add_argument("--tsv", action="store_true", help="Output TSV format")
    args = parser.parse_args()

    max_batches = args.max_batches
    batch_size = args.batch_size

    header = ["checkpoint", "step", "seq_len", "no_opp", "val/total",
              "btn_f1", "btn_prec", "btn_rec", "cdir_active", "cdir_acc", "btn_acc"]
    sep = "\t" if args.tsv else "  "
    print(sep.join(header))

    dataset_cache = {}

    for ckpt_path in args.checkpoints:
        name = Path(ckpt_path).stem
        try:
            model, cfg, norm_stats, step = load_checkpoint(ckpt_path)
        except Exception as e:
            print(f"SKIP {name}: {e}", file=sys.stderr)
            continue

        cache_key = (cfg.max_seq_len, cfg.no_opp_inputs)
        if cache_key not in dataset_cache:
            ds = build_val_dataset(args.data_dir, cfg.max_seq_len,
                                   norm_stats, cfg.no_opp_inputs)
            dl = DataLoader(
                ds, batch_size=batch_size, shuffle=False,
                num_workers=NUM_WORKERS, collate_fn=collate_fn,
                drop_last=True, pin_memory=True,
            )
            dataset_cache[cache_key] = dl
        else:
            dl = dataset_cache[cache_key]

        results = evaluate(model, cfg, dl, max_batches)
        if results is None:
            print(f"SKIP {name}: no valid batches", file=sys.stderr)
            continue

        row = [
            name,
            str(step),
            str(cfg.max_seq_len),
            "Y" if cfg.no_opp_inputs else "N",
            f"{results['total']:.4f}",
            f"{results['btn_f1']:.4f}",
            f"{results['btn_precision']:.4f}",
            f"{results['btn_recall']:.4f}",
            f"{results['cdir_active_acc']:.4f}",
            f"{results['cdir_acc']:.4f}",
            f"{results['btn_acc']:.4f}",
        ]
        print(sep.join(row), flush=True)

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
