#!/usr/bin/env python3
"""Smoke-test the WorldModel: one real batch, forward + loss + metrics.

Verifies shapes and loss magnitudes before committing to a long training run.
Runs on CPU or GPU; prints timings and parameter count.

    python3 tools/wm_smoke.py --data-dir data/fox_master_v2 --batch-size 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mimic.wm_dataset import WorldModelDataset
from mimic.wm_losses import WMLossWeights, compute_wm_loss, compute_wm_metrics

# Reuse train_wm.build_model to exercise the same construction path.
from tools.train_wm import build_model, collate


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--model", default="mimic-wm")
    ap.add_argument("--seq-len", type=int, default=180)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--discretize-counters", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    print(f"device={device}")

    combos_path = Path(args.data_dir) / "controller_combos.json"
    with open(combos_path) as fh:
        combo_map = json.load(fh)
    if isinstance(combo_map, dict) and "combos" in combo_map:
        n_combos = len(combo_map["combos"])
    elif isinstance(combo_map, dict):
        n_combos = combo_map.get("n_combos", len(combo_map))
    else:
        n_combos = len(combo_map)
    print(f"n_controller_combos={n_combos}")

    ds = WorldModelDataset(
        args.data_dir, sequence_length=args.seq_len, split="train",
        distributed=False, windows_per_game=1,
    )
    print(f"shard schema: n_numeric={ds.n_numeric}  n_flags={ds.n_flags}")

    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=0,
                    collate_fn=collate)

    model, cfg, head_dims = build_model(
        args.model, args.seq_len, n_combos,
        shard_n_numeric=ds.n_numeric, shard_n_flags=ds.n_flags,
        discretize_counters=args.discretize_counters,
    )
    model = model.to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"WorldModel params: {params / 1e6:.2f}M  "
          f"(d_model={cfg.d_model}, layers={cfg.num_layers}, heads={cfg.nhead})")
    print(f"head dims: n_numeric={head_dims['n_numeric']}  "
          f"n_flags={head_dims['n_flags']}")

    it = iter(dl)
    state, next_ctrl, target = next(it)
    print(f"\nbatch shapes:")
    for k, v in state.items():
        print(f"  state.{k}: {tuple(v.shape)}  {v.dtype}")
    for k, v in next_ctrl.items():
        print(f"  next_ctrl.{k}: {tuple(v.shape)}  {v.dtype}")
    for k, v in target.items():
        print(f"  target.{k}: {tuple(v.shape)}  {v.dtype}")

    state = {k: v.to(device) for k, v in state.items()}
    next_ctrl = {k: v.to(device) for k, v in next_ctrl.items()}
    target = {k: v.to(device) for k, v in target.items()}
    frames = {**state, **next_ctrl}

    # Warm-up pass (torch.compile / cudnn auto-tune).
    model.eval()
    with torch.no_grad():
        _ = model(frames)

    # Timed forward + loss.
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        preds = model(frames)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_fwd = time.time() - t0

    print(f"\nforward: {t_fwd * 1000:.1f} ms  (batch={args.batch_size}, "
          f"T={args.seq_len})")
    for k, v in preds.items():
        print(f"  pred.{k}: {tuple(v.shape)}")

    losses = compute_wm_loss(preds, target, WMLossWeights())
    print(f"\nlosses (untrained, expect high action CE, near-zero numeric MSE):")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")

    metrics = compute_wm_metrics(preds, target)
    print(f"\nmetrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Backward pass sanity.
    model.train()
    preds = model(frames)
    loss = compute_wm_loss(preds, target, WMLossWeights())["total"]
    loss.backward()
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    print(f"\nbackward ok. {len(grad_norms)} params with grads. "
          f"grad-norm mean={sum(grad_norms) / len(grad_norms):.4f}")

    print("\nsmoke test PASSED.")


if __name__ == "__main__":
    main()
