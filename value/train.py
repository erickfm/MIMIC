"""Windowed V(s) trainer over fox_all_v2.

Single-GPU driver. BCEWithLogitsLoss against game-outcome label, AdamW
with cosine LR (no warmup, matching MIMIC convention), bf16 AMP. Effective
batch 512 via gradient accumulation. Early-stops when val_loss flatlines
across N consecutive val checks, otherwise caps at --max-steps.

Usage:
    python -m value.train \
        --data-dir data/fox_all_v2 \
        --run-name v-fox-windowed-$(date -u +%Y%m%d) \
        --batch-size 128 --grad-accum 4 --max-steps 200000

Smoke (200 steps, batch 32, no wandb):
    python -m value.train --smoke
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from value.dataset import FoxValueWindowedDataset, collate_windows, VALUE_ENCODER_KEYS
from value.model import WindowedValueModel, MarkovValueModel


# BC encoder's input subset — used by MarkovValueModel.
BC_ENCODER_KEYS = [
    "stage",
    "self_character", "opp_character",
    "self_action", "opp_action",
    "self_numeric", "opp_numeric",
    "self_flags", "opp_flags",
    "self_controller",
]


def cosine_lr(step: int, max_steps: int, base_lr: float, min_lr: float) -> float:
    if step >= max_steps:
        return min_lr
    t = step / max(1, max_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t))


def _to_device(state: dict, y: torch.Tensor, device: torch.device):
    state = {k: v.to(device, non_blocking=True) for k, v in state.items()}
    y = y.to(device, non_blocking=True)
    return state, y


@torch.no_grad()
def evaluate(model, val_loader, device, max_batches: int):
    model.eval()
    total_loss_w = 0.0
    total_n = 0
    total_correct = 0
    total_non_draw = 0
    total_pred_pos = 0
    for i, (state, y) in enumerate(val_loader):
        if i >= max_batches:
            break
        state, y = _to_device(state, y, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            logits = model(state)  # (B,)
        loss = F.binary_cross_entropy_with_logits(
            logits.float(), y, reduction="mean")
        total_loss_w += loss.item() * y.numel()
        total_n += y.numel()
        pred = (logits > 0).float()
        non_draw = (y != 0.5)
        if non_draw.any():
            total_correct += (pred[non_draw] == y[non_draw]).sum().item()
            total_non_draw += int(non_draw.sum().item())
        total_pred_pos += int(pred.sum().item())
    model.train()
    return {
        "val/loss": total_loss_w / max(1, total_n),
        "val/acc": total_correct / max(1, total_non_draw),
        "val/pred_pos_rate": total_pred_pos / max(1, total_n),
        "val/n": total_n,
        "val/non_draw_n": total_non_draw,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/fox_all_v2")
    p.add_argument("--run-name", default="v-fox-windowed-dev")
    p.add_argument("--window", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--grad-accum", type=int, default=4,
                   help="Gradient accumulation steps (effective batch = batch * grad_accum)")
    p.add_argument("--max-steps", type=int, default=200000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--cosine-min-lr", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=0.01)
    # Model
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--dim-feedforward", type=int, default=2048)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--head-hidden", type=int, default=512)
    p.add_argument("--head-layers", type=int, default=2)
    p.add_argument("--input-gate-l1", type=float, default=0.0)
    p.add_argument("--model-type", choices=["windowed", "markov"],
                   default="windowed",
                   help="windowed = full-feature transformer; "
                        "markov = single-frame MLP on BC feature subset.")
    # Data
    p.add_argument("--windows-per-shard", type=int, default=4096)
    p.add_argument("--num-workers", type=int, default=2)
    # Position-filter: apply to TRAIN sampling only. Val stays unrestricted
    # so we can see how the model generalizes outside its training range.
    p.add_argument("--min-position", type=float, default=0.0,
                   help="Train only on windows whose last frame is at >=N of game length")
    p.add_argument("--max-position", type=float, default=1.0,
                   help="Train only on windows whose last frame is at <=N of game length")
    # Eval / logging
    p.add_argument("--val-every", type=int, default=1000)
    p.add_argument("--val-batches", type=int, default=64)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--early-stop-patience", type=int, default=5,
                   help="Stop if val_loss didn't improve over N consecutive val checks")
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.smoke:
        args.max_steps = 200
        args.val_every = 100
        args.val_batches = 8
        args.log_every = 10
        args.batch_size = 32
        args.grad_accum = 1
        args.windows_per_shard = 256
        args.num_workers = 0
        args.early_stop_patience = 999
        args.no_wandb = True

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eff_batch = args.batch_size * args.grad_accum

    if args.model_type == "markov":
        # Markov: force window=1, BC feature subset
        if args.window != 1:
            print(f"[markov] forcing window=1 (was {args.window})")
            args.window = 1
        state_keys = BC_ENCODER_KEYS
    else:
        state_keys = VALUE_ENCODER_KEYS

    # Datasets
    train_ds = FoxValueWindowedDataset(
        data_dir=args.data_dir, split="train",
        window=args.window,
        windows_per_shard_visit=args.windows_per_shard,
        state_keys=state_keys,
        seed=args.seed,
        world_size=1, rank=0, distributed=False,
        min_position=args.min_position, max_position=args.max_position,
    )
    val_ds = FoxValueWindowedDataset(
        data_dir=args.data_dir, split="val",
        window=args.window,
        windows_per_shard_visit=max(256, args.windows_per_shard // 4),
        state_keys=state_keys,
        seed=args.seed + 1,
        world_size=1, rank=0, distributed=False,
        # Val: full-range so we measure generalization across all positions
        min_position=0.0, max_position=1.0,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=args.num_workers,
        collate_fn=collate_windows, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, num_workers=max(0, args.num_workers // 2),
        collate_fn=collate_windows, pin_memory=True,
        persistent_workers=args.num_workers > 1,
    )

    # Model
    if args.model_type == "markov":
        model = MarkovValueModel(
            d_model=args.d_model,
            dropout=args.dropout,
            head_hidden=args.head_hidden,
            head_layers=args.head_layers,
            use_input_gate=args.input_gate_l1 > 0,
        ).to(device)
    else:
        model = WindowedValueModel(
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            head_hidden=args.head_hidden,
            head_layers=args.head_layers,
            use_input_gate=args.input_gate_l1 > 0,
        ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    # MimicFlatEncoder (markov) exposes its input dim via _input_dim, not input_dim
    if hasattr(model.encoder, "input_dim"):
        in_dim = model.encoder.input_dim
    else:
        in_dim = getattr(model.encoder, "_input_dim", "?")
    print(f"model: {n_params/1e6:.2f}M params  type={args.model_type}  "
          f"d_model={args.d_model}  "
          f"layers={getattr(args, 'num_layers', 0) if args.model_type=='windowed' else 0}  "
          f"input_dim={in_dim}")
    print(f"effective batch = {args.batch_size} × {args.grad_accum} = {eff_batch}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project="MIMIC-value",
                name=args.run_name,
                config={**vars(args),
                        "n_params": n_params,
                        "effective_batch": eff_batch,
                        "input_dim": model.encoder.input_dim},
            )
        except Exception as exc:
            print(f"wandb disabled: {exc}")
            use_wandb = False

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    val_history: list[float] = []

    model.train()
    step = 0
    micro_step = 0  # micro-batches accumulated within current optimizer step
    t_start = time.time()
    running_loss = 0.0
    running_count = 0
    train_iter = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)
    stopped_early = False

    while step < args.max_steps:
        try:
            state, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            state, y = next(train_iter)

        state, y = _to_device(state, y, device)

        # LR schedule applies on optimizer-step boundaries
        lr = cosine_lr(step, args.max_steps, args.lr, args.cosine_min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            logits = model(state)
        loss = F.binary_cross_entropy_with_logits(logits.float(), y)
        if args.input_gate_l1 > 0:
            loss = loss + args.input_gate_l1 * model.encoder.gate_l1_penalty()
        (loss / args.grad_accum).backward()
        micro_step += 1

        if micro_step >= args.grad_accum:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            micro_step = 0
            step += 1
            running_loss += loss.item()
            running_count += 1

            if step % args.log_every == 0:
                elapsed = time.time() - t_start
                fps = (step * eff_batch * args.window) / max(1e-6, elapsed)
                avg_loss = running_loss / running_count
                print(f"step {step:6d}/{args.max_steps}  loss={avg_loss:.4f}  "
                      f"lr={lr:.2e}  fps={fps:.0f}")
                if use_wandb:
                    import wandb
                    wandb.log({"train/loss": avg_loss, "train/lr": lr,
                               "train/fps": fps}, step=step)
                running_loss = 0.0
                running_count = 0

            if step % args.val_every == 0 or step == args.max_steps:
                val = evaluate(model, val_loader, device, args.val_batches)
                print(f"  [val] loss={val['val/loss']:.4f}  "
                      f"acc={val['val/acc']:.3f}  "
                      f"pred_pos={val['val/pred_pos_rate']:.3f}  "
                      f"n={val['val/n']}")
                if use_wandb:
                    import wandb
                    wandb.log(val, step=step)
                val_history.append(val["val/loss"])
                if val["val/loss"] < best_val:
                    best_val = val["val/loss"]
                    torch.save({
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": step,
                        "val_loss": val["val/loss"],
                        "args": vars(args),
                    }, ckpt_dir / f"{args.run_name}_best.pt")
                    print(f"  -> saved best (val_loss={best_val:.4f})")

                # Early stop: no improvement over the last `patience` checks.
                # Concretely: best val_loss must be OLDER than the patience
                # window — if any of the last N vals achieved (or matched) the
                # global best, we keep going.
                if len(val_history) > args.early_stop_patience:
                    recent = val_history[-args.early_stop_patience:]
                    if min(recent) > best_val + 1e-4:
                        print(f"  early stop: val_loss didn't improve over "
                              f"last {args.early_stop_patience} checks "
                              f"(best={best_val:.4f}, recent_min={min(recent):.4f})")
                        stopped_early = True
                        break

    elapsed = time.time() - t_start
    print(f"done. best val_loss={best_val:.4f}  steps={step}  "
          f"elapsed={elapsed:.0f}s  early_stop={stopped_early}")

    # Dump gate report if input-gate was enabled
    if args.input_gate_l1 > 0:
        report = model.encoder.gate_report()  # sorted ascending (most-pruned first)
        # Reverse to show most-attended-to first (descending)
        report_desc = list(reversed(report))
        out_path = ckpt_dir / f"{args.run_name}_gate_report.json"
        out_path.write_text(__import__('json').dumps({
            "run_name": args.run_name,
            "best_val_loss": best_val,
            "steps": step,
            "input_gate_l1": args.input_gate_l1,
            "n_features": len(report),
            "gate_report_descending": [
                {"feature": n, "gate_value": float(v)} for n, v in report_desc
            ],
        }, indent=2))
        print(f"wrote gate report: {out_path}")
        print(f"top 20 features by gate value:")
        for n, v in report_desc[:20]:
            print(f"  {v:.4f}  {n}")
        print(f"bottom 10 features by gate value (most pruned):")
        for n, v in report[:10]:
            print(f"  {v:.4f}  {n}")


if __name__ == "__main__":
    main()
