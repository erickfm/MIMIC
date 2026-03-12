#!/usr/bin/env python3
# train.py  --  FRAME training with BF16 AMP, torch.compile, epoch/step-based loop
#
# NOTE: overfit_log*.txt files in this repo are from intentional overfit runs
# on a small fsmash-only subset.  Near-zero main/l/r/btn losses in those logs
# are expected -- the model learned the dominant neutral/no-press pattern on
# that narrow data.  They do NOT indicate a bug.

import sys
import os
import math
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as nn_utils
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from torch.amp import autocast
from pathlib import Path

import wandb

from dataset import MeleeFrameDatasetWithDelay, StreamingMeleeDataset
from model   import FramePredictor, ModelConfig, MODEL_PRESETS

# -----------------------------------------------------------------------------
# 1. Config
# -----------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE      = 200
LEARNING_RATE   = 1e-3
WEIGHT_DECAY    = 1e-2
NUM_WORKERS     = 8
SEQUENCE_LENGTH = 60
REACTION_DELAY  = 1
DATA_DIR        = "./data/subset"

GRAD_CLIP_NORM      = 1.0
TASK_NAMES          = ["main", "l", "r", "cdir", "btn"]

FOCAL_GAMMA         = 2.0
LABEL_SMOOTHING     = 0.1

AMP_DTYPE = torch.bfloat16

MAX_VAL_BATCHES    = 100

# Intervals derived from max_steps at runtime (see _compute_intervals)
WARMUP_FRAC        = 0.01     # 1% warmup
LOG_FRAC           = 0.005    # log every ~0.5%
VAL_FRAC           = 0.05     # validate every ~5%
CKPT_FRAC          = 0.05     # checkpoint every ~5%

# -----------------------------------------------------------------------------
# 1b. Speed settings
# -----------------------------------------------------------------------------
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

# -----------------------------------------------------------------------------
# 2. Collate
# -----------------------------------------------------------------------------
def collate_fn(batch):
    batch_state, batch_target = {}, {}
    for k in batch[0][0]:
        batch_state[k]  = torch.stack([item[0][k] for item in batch], 0)
    for k in batch[0][1]:
        batch_target[k] = torch.stack([item[1][k] for item in batch], 0)
    return batch_state, batch_target

# -----------------------------------------------------------------------------
# 3. Loss
# -----------------------------------------------------------------------------
_mse = nn.MSELoss()
_bce = nn.BCEWithLogitsLoss()

def focal_loss(logits, targets, gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTHING):
    n_classes = logits.shape[-1]
    with torch.no_grad():
        smooth = label_smoothing / n_classes
        targets_smooth = torch.full_like(logits, smooth)
        targets_smooth.scatter_(1, targets.unsqueeze(1), 1.0 - label_smoothing + smooth)

    log_p = F.log_softmax(logits, dim=-1)
    p     = log_p.exp()

    p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
    focal_weight = (1.0 - p_t).pow(gamma)

    ce = -(targets_smooth * log_p).sum(dim=-1)
    loss = focal_weight * ce
    return loss.mean()


def compute_loss(preds, targets):
    main_pred = preds["main_xy"].float()
    l_pred    = preds["L_val"].squeeze(-1).float()
    r_pred    = preds["R_val"].squeeze(-1).float()
    c_logits  = preds["c_dir_logits"].float()
    btn_pred  = preds["btn_logits"].float()

    main_tgt = torch.stack([targets["main_x"], targets["main_y"]], dim=-1)
    l_tgt    = targets["l_shldr"]
    r_tgt    = targets["r_shldr"]
    cdir_tgt = targets["c_dir"].long()
    btn_tgt  = targets.get("btns", targets.get("btns_float")).float()

    loss_main = _mse(main_pred, main_tgt)
    loss_l    = _mse(l_pred, l_tgt)
    loss_r    = _mse(r_pred, r_tgt)

    cdir_classes = cdir_tgt.argmax(dim=-1)
    c_logits_flat = c_logits.reshape(-1, c_logits.size(-1))
    cdir_flat     = cdir_classes.reshape(-1)
    loss_cdir = focal_loss(c_logits_flat, cdir_flat)
    loss_btn  = _bce(btn_pred, btn_tgt)

    cdir_pred = c_logits_flat.argmax(dim=-1)
    cdir_acc  = (cdir_pred == cdir_flat).float().mean().item()
    btn_acc   = ((torch.sigmoid(btn_pred) > 0.5) == (btn_tgt > 0.5)).float().mean().item()

    metrics = {
        "loss_main": loss_main.item(),
        "loss_l":    loss_l.item(),
        "loss_r":    loss_r.item(),
        "loss_cdir": loss_cdir.item(),
        "loss_btn":  loss_btn.item(),
        "cdir_acc":  cdir_acc,
        "btn_acc":   btn_acc,
    }
    task_losses = (loss_main, loss_l, loss_r, loss_cdir, loss_btn)
    return metrics, task_losses

# -----------------------------------------------------------------------------
# 4. Helpers
# -----------------------------------------------------------------------------
def _compute_intervals(max_steps):
    """Derive logging/checkpoint/warmup intervals from total steps."""
    return dict(
        log_interval  = max(int(max_steps * LOG_FRAC),  1),
        val_interval  = max(int(max_steps * VAL_FRAC),  50),
        ckpt_interval = max(int(max_steps * CKPT_FRAC), 50),
        warmup_steps  = max(int(max_steps * WARMUP_FRAC), 10),
    )


def get_datasets(data_dir):
    p = Path(data_dir)
    has_metadata = all((p / f).exists() for f in
                       ("norm_stats.json", "cat_maps.json", "file_index.json"))

    if has_metadata:
        print(f"  Using streaming dataset from {data_dir}", flush=True)
        train_ds = StreamingMeleeDataset(
            data_dir=data_dir,
            sequence_length=SEQUENCE_LENGTH,
            reaction_delay=REACTION_DELAY,
            split="train",
        )
        val_ds = StreamingMeleeDataset(
            data_dir=data_dir,
            sequence_length=SEQUENCE_LENGTH,
            reaction_delay=REACTION_DELAY,
            split="val",
        )
    else:
        print(f"  No metadata found, using raw-parquet dataset (slow startup)", flush=True)
        train_ds = MeleeFrameDatasetWithDelay(
            parquet_dir=data_dir,
            sequence_length=SEQUENCE_LENGTH,
            reaction_delay=REACTION_DELAY,
            split="train",
        )
        val_ds = MeleeFrameDatasetWithDelay(
            parquet_dir=data_dir,
            sequence_length=SEQUENCE_LENGTH,
            reaction_delay=REACTION_DELAY,
            split="val",
            norm_stats=train_ds.norm_stats,
        )
    return train_ds, val_ds

def get_dataloader(ds, shuffle=True):
    is_iterable = isinstance(ds, IterableDataset)
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=(shuffle and not is_iterable),
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True,
    )

def get_model(compile_model=True, model_preset=None):
    overrides = MODEL_PRESETS.get(model_preset, {}) if model_preset else {}
    cfg = ModelConfig(max_seq_len=SEQUENCE_LENGTH, **overrides)
    model = FramePredictor(cfg).to(DEVICE)
    if compile_model:
        model = torch.compile(model)
    return model, cfg

def infinite_loader(dl):
    """Yield batches forever, cycling through epochs."""
    while True:
        for batch in dl:
            yield batch

# -----------------------------------------------------------------------------
# 5. Training loop (step-based)
# -----------------------------------------------------------------------------
def train(epochs: int = None, max_steps: int = None, data_dir: str = DATA_DIR,
          debug: bool = False, resume: str = None, compile_model: bool = True,
          model_preset: str = None, lr: float = None, run_name: str = None):
    if debug:
        torch.autograd.set_detect_anomaly(True)

    print(f"Loading dataset from {data_dir} ...", flush=True)
    ds, val_ds = get_datasets(data_dir)
    n_train = len(ds)
    n_val   = len(val_ds)
    n_train_games = getattr(ds, "n_games", len(getattr(ds, "files", [])))
    n_val_games   = getattr(val_ds, "n_games", len(getattr(val_ds, "files", [])))
    print(f"  Train: {n_train:,} windows from {n_train_games} games", flush=True)
    print(f"  Val:   {n_val:,} windows from {n_val_games} games", flush=True)

    dl     = get_dataloader(ds)
    val_dl = get_dataloader(val_ds, shuffle=False)

    est_batches = max(n_train // BATCH_SIZE, 1)
    if max_steps is None:
        if epochs is None:
            epochs = 1
        max_steps = epochs * est_batches
        print(f"  {epochs} epoch(s) x {est_batches} batches = {max_steps:,} steps", flush=True)
    else:
        epoch_frac = max_steps / est_batches
        print(f"  {max_steps:,} steps (~{epoch_frac:.2f} epochs)", flush=True)

    intervals = _compute_intervals(max_steps)
    log_interval  = intervals["log_interval"]
    val_interval  = intervals["val_interval"]
    ckpt_interval = intervals["ckpt_interval"]
    warmup_steps  = intervals["warmup_steps"]

    actual_lr = lr or LEARNING_RATE

    print(f"  Compiling model: {compile_model}  preset: {model_preset or 'base'}", flush=True)
    model, cfg = get_model(compile_model=compile_model, model_preset=model_preset)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} params on {DEVICE}  (AMP={AMP_DTYPE}, LR={actual_lr})", flush=True)
    print(f"  Intervals: log={log_interval}  val={ckpt_interval}  "
          f"ckpt={ckpt_interval}  warmup={warmup_steps}", flush=True)

    loss_weights = torch.ones(len(TASK_NAMES), device=DEVICE)

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if n.endswith("bias") or "norm" in n.lower()
                  else decay).append(p)
    optimiser = optim.AdamW(
        [{"params": decay,    "weight_decay": WEIGHT_DECAY},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=actual_lr,
        betas=(0.9, 0.999),
        fused=True,
    )

    warmup  = optim.lr_scheduler.LinearLR(
        optimiser, start_factor=0.01, total_iters=warmup_steps)
    cosine  = optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=max(max_steps - warmup_steps, 1))
    scheduler = optim.lr_scheduler.SequentialLR(
        optimiser, [warmup, cosine], milestones=[warmup_steps])

    start_step = 0
    if resume:
        ckpt = torch.load(resume, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        optimiser.load_state_dict(ckpt['optimizer_state_dict'])
        if ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_step = ckpt.get('global_step', 0)
        print(f"Resumed from {resume}, starting at step {start_step}")

    wandb.init(
        project="FRAME",
        entity="erickfm",
        name=run_name,
        config=dict(
            batch_size=BATCH_SIZE,
            learning_rate=actual_lr,
            max_steps=max_steps,
            warmup_steps=warmup_steps,
            log_interval=log_interval,
            val_interval=val_interval,
            ckpt_interval=ckpt_interval,
            num_workers=NUM_WORKERS,
            sequence_length=SEQUENCE_LENGTH,
            reaction_delay=REACTION_DELAY,
            amp_dtype=str(AMP_DTYPE),
            compiled=compile_model,
            model_preset=model_preset or "base",
            n_params=n_params,
            **cfg.__dict__,
        ),
    )

    loader = infinite_loader(dl)
    skip_ct = 0
    t0 = time.time()

    _AVG_KEYS = ["total", "loss_main", "loss_l", "loss_r", "loss_cdir",
                 "loss_btn", "cdir_acc", "btn_acc", "grad_norm"]
    _run_sums = {k: 0.0 for k in _AVG_KEYS}
    _run_ct = 0

    if start_step and not isinstance(ds, IterableDataset):
        for _ in range(start_step):
            next(loader)

    print(f"\n=== Training {max_steps} steps ===", flush=True)
    for step in range(start_step + 1, max_steps + 1):
        model.train()
        state, target = next(loader)

        for k, v in state.items():
            state[k] = v.to(DEVICE, non_blocking=True)
        for k, v in target.items():
            target[k] = v.to(DEVICE, non_blocking=True)

        with autocast("cuda", dtype=AMP_DTYPE):
            preds = model(state)

        metrics, task_losses = compute_loss(preds, target)
        loss_vec = torch.stack(task_losses)
        total_loss = (loss_weights * loss_vec).sum()

        if not torch.isfinite(total_loss):
            skip_ct += 1
            print(f"  [skip] step {step} -- non-finite loss ({skip_ct} total)", flush=True)
            scheduler.step()
            continue

        optimiser.zero_grad(set_to_none=True)
        total_loss.backward()

        grad_norm = nn_utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM).item()
        has_inf_grad = not math.isfinite(grad_norm)
        if has_inf_grad:
            skip_ct += 1
            print(f"  [skip] step {step} -- inf grad norm ({skip_ct} total)", flush=True)
            optimiser.zero_grad(set_to_none=True)
            scheduler.step()
            continue

        optimiser.step()
        scheduler.step()

        _run_sums["total"]     += total_loss.item()
        _run_sums["grad_norm"] += grad_norm
        for _mk in ("loss_main", "loss_l", "loss_r", "loss_cdir",
                     "loss_btn", "cdir_acc", "btn_acc"):
            _run_sums[_mk] += metrics[_mk]
        _run_ct += 1

        wandb.log({
            "train/total":     total_loss.item(),
            "train/main":      metrics["loss_main"],
            "train/l":         metrics["loss_l"],
            "train/r":         metrics["loss_r"],
            "train/cdir":      metrics["loss_cdir"],
            "train/btn":       metrics["loss_btn"],
            "train/cdir_acc":  metrics["cdir_acc"],
            "train/btn_acc":   metrics["btn_acc"],
            "train/grad_norm": grad_norm,
            "train/lr":        scheduler.get_last_lr()[0],
        }, step=step)

        if step % log_interval == 0 or step == 1:
            elapsed = time.time() - t0
            sps = step / elapsed

            if _run_ct > 0:
                avg_log = {f"avg/{k}": _run_sums[k] / _run_ct for k in _AVG_KEYS}
                avg_log["perf/step_per_sec"]    = sps
                avg_log["perf/samples_per_sec"] = sps * BATCH_SIZE
                wandb.log(avg_log, step=step)
                _run_sums = {k: 0.0 for k in _AVG_KEYS}
                _run_ct = 0

            print(
                f"[{step:5d}/{max_steps}] "
                f"total={total_loss.item():.4f} "
                f"main={metrics['loss_main']:.5f} "
                f"l={metrics['loss_l']:.5f} "
                f"r={metrics['loss_r']:.5f} "
                f"cdir={metrics['loss_cdir']:.4f} "
                f"btn={metrics['loss_btn']:.5f} "
                f"cacc={metrics['cdir_acc']:.1%} "
                f"bacc={metrics['btn_acc']:.1%} "
                f"gnorm={grad_norm:.2f} "
                f"lr={scheduler.get_last_lr()[0]:.2e} "
                f"({sps:.1f} step/s)",
                flush=True,
            )

        # Validation
        if step % val_interval == 0 or step == max_steps:
            model.eval()
            _VAL_KEYS = ["total", "loss_main", "loss_l", "loss_r",
                         "loss_cdir", "loss_btn", "cdir_acc", "btn_acc"]
            val_sums = {k: 0.0 for k in _VAL_KEYS}
            val_ct = 0
            with torch.no_grad():
                for vs, vt in val_dl:
                    for k, v in vs.items():
                        vs[k] = v.to(DEVICE, non_blocking=True)
                    for k, v in vt.items():
                        vt[k] = v.to(DEVICE, non_blocking=True)
                    with autocast("cuda", dtype=AMP_DTYPE):
                        vpreds = model(vs)
                    vm, vtl = compute_loss(vpreds, vt)
                    batch_total = sum(t.item() for t in vtl)
                    if math.isfinite(batch_total):
                        val_sums["total"] += batch_total
                        for _vk in ("loss_main", "loss_l", "loss_r",
                                     "loss_cdir", "loss_btn", "cdir_acc", "btn_acc"):
                            val_sums[_vk] += vm[_vk]
                        val_ct += 1
                    if val_ct >= MAX_VAL_BATCHES:
                        break

            if val_ct > 0:
                val_avg = {f"val/{k}": val_sums[k] / val_ct for k in _VAL_KEYS}
                wandb.log(val_avg, step=step)
                print(f"  -- val total={val_avg['val/total']:.4f}  "
                      f"cdir_acc={val_avg['val/cdir_acc']:.1%}  "
                      f"btn_acc={val_avg['val/btn_acc']:.1%}  "
                      f"(batches={val_ct})", flush=True)
            else:
                print("  -- val: no valid batches", flush=True)

        # Checkpoint
        if step % ckpt_interval == 0 or step == max_steps:
            os.makedirs("checkpoints", exist_ok=True)
            ckpt_path = f"checkpoints/step_{step:06d}.pt"
            torch.save({
                "global_step":          step,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimiser.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss_weights":         loss_weights.cpu(),
                "norm_stats":           ds.norm_stats,
                "config":               cfg.__dict__,
            }, ckpt_path)
            print(f"  -- saved {ckpt_path}", flush=True)

    elapsed = time.time() - t0
    skip_msg = f"  skipped={skip_ct}" if skip_ct else ""
    epoch_frac = max_steps / max(est_batches, 1)
    print(f"\nDone. {max_steps:,} steps ({epoch_frac:.2f} epochs) in {elapsed:.0f}s "
          f"({max_steps/elapsed:.1f} step/s){skip_msg}", flush=True)
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FRAME training")
    parser.add_argument("--epochs",     type=int, default=None,
                        help="Number of epochs (default: 1 if --max-steps not set)")
    parser.add_argument("--max-steps",  type=int, default=None,
                        help="Override: train for exactly this many steps")
    parser.add_argument("--data-dir",   type=str, default=DATA_DIR)
    parser.add_argument("--debug",      action="store_true")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile (debug)")
    parser.add_argument("--resume",     type=str, default=None,
                        help="Path to checkpoint (.pt) to resume from")
    parser.add_argument("--model",      type=str, default=None,
                        choices=list(MODEL_PRESETS.keys()),
                        help="Model size preset (default: base)")
    parser.add_argument("--lr",         type=float, default=None,
                        help="Learning rate override (default: 1e-3)")
    parser.add_argument("--run-name",   type=str, default=None,
                        help="Wandb run name (auto-generated if not set)")
    args = parser.parse_args()
    train(
        epochs=args.epochs,
        max_steps=args.max_steps,
        data_dir=args.data_dir,
        debug=args.debug or bool(os.getenv("DEBUG", "")),
        resume=args.resume,
        compile_model=not args.no_compile,
        model_preset=args.model,
        lr=args.lr,
        run_name=args.run_name,
    )
