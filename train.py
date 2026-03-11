#!/usr/bin/env python3
# train.py  —  FRAME training with focal loss, cosine LR, and fp16 AMP

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
from torch.utils.data import DataLoader

# —— AMP ————————————————————————————————————————————————————————
from torch.amp import autocast
from torch.cuda.amp import GradScaler

import wandb

from dataset import MeleeFrameDatasetWithDelay
from model   import FramePredictor, ModelConfig

# ─────────────────────────────────────────────────────────────────────────────
# 1. Config
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE      = 192
NUM_EPOCHS      = 100
LEARNING_RATE   = 1e-3
WEIGHT_DECAY    = 1e-2            # AdamW (weights only)
NUM_WORKERS     = 8
SEQUENCE_LENGTH = 60
REACTION_DELAY  = 1
DATA_DIR        = "./data/subset"

GRAD_CLIP_NORM      = 1.0
TASK_NAMES          = ["main", "l", "r", "cdir", "btn"]

# —— Focal loss + label smoothing for c-dir ————————————————————————————
FOCAL_GAMMA         = 2.0       # down-weight easy examples
LABEL_SMOOTHING     = 0.1       # soften hard targets

USE_AMP         = False  # disabled — fp16 causes NaN overflow on real data

# ─────────────────────────────────────────────────────────────────────────────
# 2. Collate
# ─────────────────────────────────────────────────────────────────────────────
def collate_fn(batch):
    batch_state, batch_target = {}, {}
    for k in batch[0][0]:
        batch_state[k]  = torch.stack([item[0][k] for item in batch], 0)
    for k in batch[0][1]:
        batch_target[k] = torch.stack([item[1][k] for item in batch], 0)
    return batch_state, batch_target

# ─────────────────────────────────────────────────────────────────────────────
# 3. Loss (with per-term safety)
# ─────────────────────────────────────────────────────────────────────────────
_mse = nn.MSELoss()
_bce = nn.BCEWithLogitsLoss()

def focal_loss(logits, targets, gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTHING):
    """Focal loss with label smoothing for multi-class classification."""
    n_classes = logits.shape[-1]
    # label smoothing: hard targets → soft targets
    with torch.no_grad():
        smooth = label_smoothing / n_classes
        targets_smooth = torch.full_like(logits, smooth)
        targets_smooth.scatter_(1, targets.unsqueeze(1), 1.0 - label_smoothing + smooth)

    log_p = F.log_softmax(logits, dim=-1)
    p     = log_p.exp()

    # focal modulation by p_t (true-class probability) per sample
    p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)   # (B,)
    focal_weight = (1.0 - p_t).pow(gamma)                # (B,)

    ce = -(targets_smooth * log_p).sum(dim=-1)            # (B,)
    loss = focal_weight * ce
    return loss.mean()

def safe_loss(fn, pred, tgt, name):
    out = fn(pred, tgt)
    return out

def compute_loss(preds, targets):
    # —— predictions (cast to fp32 so dtypes match targets) ————————
    main_pred = preds["main_xy"].float()
    l_pred    = preds["L_val"].squeeze(-1).float()
    r_pred    = preds["R_val"].squeeze(-1).float()
    c_logits  = preds["c_dir_logits"].float()
    btn_pred  = preds["btn_logits"].float()

    # —— targets ————————————————————————————————————————————————
    main_tgt = torch.stack([targets["main_x"], targets["main_y"]], dim=-1)
    l_tgt    = targets["l_shldr"]
    r_tgt    = targets["r_shldr"]
    cdir_tgt = targets["c_dir"].long()
    btn_tgt  = targets.get("btns", targets.get("btns_float")).float()

    # —— per-head losses ————————————————————————————————————————
    loss_main = safe_loss(_mse, main_pred, main_tgt, "main_xy")
    loss_l    = safe_loss(_mse, l_pred,    l_tgt,    "L_val")
    loss_r    = safe_loss(_mse, r_pred,    r_tgt,    "R_val")
    cdir_classes = cdir_tgt.argmax(dim=-1)
    loss_cdir = focal_loss(c_logits, cdir_classes)
    loss_btn  = safe_loss(_bce, btn_pred,  btn_tgt, "btns")

    # —— c-dir accuracy ——————————————————————————————————————————
    cdir_pred = c_logits.argmax(dim=-1)
    cdir_acc  = (cdir_pred == cdir_classes).float().mean().item()

    metrics = {
        "loss_main": loss_main.item(),
        "loss_l":    loss_l.item(),
        "loss_r":    loss_r.item(),
        "loss_cdir": loss_cdir.item(),
        "loss_btn":  loss_btn.item(),
        "cdir_acc":  cdir_acc,
    }
    task_losses = (loss_main, loss_l, loss_r, loss_cdir, loss_btn)
    return metrics, task_losses

# ─────────────────────────────────────────────────────────────────────────────
# 4. Helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_datasets():
    train_ds = MeleeFrameDatasetWithDelay(
        parquet_dir=DATA_DIR,
        sequence_length=SEQUENCE_LENGTH,
        reaction_delay=REACTION_DELAY,
        split="train",
    )
    val_ds = MeleeFrameDatasetWithDelay(
        parquet_dir=DATA_DIR,
        sequence_length=SEQUENCE_LENGTH,
        reaction_delay=REACTION_DELAY,
        split="val",
        norm_stats=train_ds.norm_stats,
    )
    return train_ds, val_ds

def get_dataloader(ds, shuffle=True):
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=True,
    )

def get_model():
    cfg   = ModelConfig(max_seq_len=SEQUENCE_LENGTH)
    model = FramePredictor(cfg).to(DEVICE)
    return model, cfg

# ─────────────────────────────────────────────────────────────────────────────
# 5. Training loop
# ─────────────────────────────────────────────────────────────────────────────
def train(debug: bool = False, resume: str = None):
    if debug:
        torch.autograd.set_detect_anomaly(True)

    print(f"Loading dataset from {DATA_DIR} ...", flush=True)
    ds, val_ds = get_datasets()
    print(f"  Train: {len(ds):,} windows from {len(ds.files)} files", flush=True)
    print(f"  Val:   {len(val_ds):,} windows from {len(val_ds.files)} files", flush=True)
    dl     = get_dataloader(ds)
    val_dl = get_dataloader(val_ds, shuffle=False)
    model, cfg = get_model()
    print(f"  Model: {sum(p.numel() for p in model.parameters()):,} params on {DEVICE}", flush=True)

    # —— Fixed task weights (no learnable weights — they collapse to 0) ——
    loss_weights = torch.ones(len(TASK_NAMES), device=DEVICE)

    # —— AdamW param-groups (bias/Norm excluded from decay) ————————
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if n.endswith("bias") or "norm" in n.lower()
                  else decay).append(p)

    optimiser = optim.AdamW(
        [{"params": decay,      "weight_decay": WEIGHT_DECAY},
         {"params": no_decay,   "weight_decay": 0.0}],
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
    )

    cosine    = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=NUM_EPOCHS)
    warmup    = optim.lr_scheduler.LinearLR(optimiser, start_factor=0.01, total_iters=3)
    scheduler = optim.lr_scheduler.SequentialLR(optimiser, [warmup, cosine], milestones=[3])
    scaler    = GradScaler(enabled=USE_AMP)

    # —— optional resume ————————————————————————————————————————
    start_epoch = 1
    if resume:
        ckpt = torch.load(resume, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        optimiser.load_state_dict(ckpt['optimizer_state_dict'])
        if USE_AMP and ckpt.get("scaler_state_dict"):
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        if ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt['epoch'] + 1
        print(f"Resumed from {resume}, starting at epoch {start_epoch}")

    wandb.init(
        project="FRAME",
        entity="erickfm",
        config=dict(
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            epochs=NUM_EPOCHS,
            num_workers=NUM_WORKERS,
            sequence_length=SEQUENCE_LENGTH,
            reaction_delay=REACTION_DELAY,
            amp=USE_AMP,
            **cfg.__dict__,
        ),
    )

    global_step = 0
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        model.train()
        epoch_loss, batch_ct, skip_ct = 0.0, 0, 0
        t0 = time.time()
        print(f"\n=== Epoch {epoch}/{NUM_EPOCHS} ===", flush=True)

        for i, (state, target) in enumerate(dl, 1):
            # —— move to device & sanity-check inputs ——————————————
            for k, v in state.items():
                state[k] = v.to(DEVICE, non_blocking=True)
                if debug and not torch.isfinite(state[k]).all():
                    raise RuntimeError(f"Non-finite in state['{k}']")
            for k, v in target.items():
                target[k] = v.to(DEVICE, non_blocking=True)
                if debug and not torch.isfinite(target[k]).all():
                    raise RuntimeError(f"Non-finite in target['{k}']")

            # —— forward (fp16 autocast) ——————————————————————————
            with autocast("cuda", enabled=USE_AMP):
                preds = model(state)

            if debug:
                for name, t in preds.items():
                    if not torch.isfinite(t).all():
                        raise RuntimeError(f"Non-finite in output '{name}'")

            # —— losses ————————————————————————————————————————————
            metrics, task_losses = compute_loss(preds, target)
            loss_vec = torch.stack(task_losses)
            total_loss = (loss_weights * loss_vec).sum()

            if not torch.isfinite(total_loss):
                skip_ct += 1
                print(f"⚠️  Skipping batch {i} — non-finite loss ({skip_ct} total)", flush=True)
                continue

            # —— first-batch grad stats ————————————————————————————
            if i == 1 and epoch == start_epoch:
                grads = torch.autograd.grad(total_loss,
                                            model.parameters(),
                                            retain_graph=True)
                print("=== GRADIENT STATS FOR FIRST BATCH ===")
                for (pname, p), g in zip(model.named_parameters(), grads):
                    if g is None:
                        print(f"{pname:40s} | no grad")
                    else:
                        print(f"{pname:40s} | norm={g.norm().item():8.3f}"
                              f"  nan={int(torch.isnan(g).sum())}"
                              f"  inf={int(torch.isinf(g).sum())}")

            # —— backward / step with scaler ————————————————
            optimiser.zero_grad(set_to_none=True)
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimiser)
            nn_utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimiser)
            scaler.update()

            # —— logging ——————————————————————————————————————
            global_step += 1
            epoch_loss += total_loss.item()
            batch_ct   += 1
            wandb.log(
                dict(step=global_step,
                     total=total_loss.item(),
                     avg_loss=epoch_loss / batch_ct,
                     lr=scheduler.get_last_lr()[0],
                     epoch=epoch,
                     **metrics),
                step=global_step,
            )

            if i % 250 == 0:
                print(
                    f"[{i:05d}] total={total_loss.item():.4f} "
                    f"main={metrics['loss_main']:.3f} l={metrics['loss_l']:.3f} "
                    f"r={metrics['loss_r']:.3f} cdir={metrics['loss_cdir']:.3f} "
                    f"btn={metrics['loss_btn']:.3f} "
                    f"cdir_acc={metrics['cdir_acc']:.1%}",
                    flush=True,
                )

        # —— Validation ————————————————————————————————————————
        model.eval()
        val_loss, val_acc, val_ct = 0.0, 0.0, 0
        with torch.no_grad():
            for state, target in val_dl:
                for k, v in state.items():
                    state[k] = v.to(DEVICE, non_blocking=True)
                for k, v in target.items():
                    target[k] = v.to(DEVICE, non_blocking=True)
                preds = model(state)
                metrics, task_losses = compute_loss(preds, target)
                loss = sum(task_losses).item()
                if math.isfinite(loss):
                    val_loss += loss
                    val_acc  += metrics["cdir_acc"]
                    val_ct   += 1

        val_avg  = val_loss / max(val_ct, 1)
        val_cacc = val_acc / max(val_ct, 1)

        # —— LR step & checkpoint ——————————————————————————————
        scheduler.step()
        elapsed = time.time() - t0
        avg = epoch_loss / max(batch_ct, 1)
        skip_msg = f"  skipped={skip_ct}" if skip_ct else ""
        print(f"Epoch {epoch} done. train={avg:.4f} val={val_avg:.4f} "
              f"val_cdir_acc={val_cacc:.1%}  "
              f"batches={batch_ct}  time={elapsed:.0f}s{skip_msg}",
              flush=True)
        os.makedirs("checkpoints", exist_ok=True)
        ckpt_path = f"checkpoints/epoch_{epoch:02d}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict":      model.state_dict(),
            "optimizer_state_dict":  optimiser.state_dict(),
            "scheduler_state_dict":  scheduler.state_dict(),
            "loss_weights":          loss_weights.cpu(),
            "scaler_state_dict":     scaler.state_dict(),
            "norm_stats":            ds.norm_stats,
            "config":                cfg.__dict__,
        }, ckpt_path)
        print("Saved checkpoint →", ckpt_path)
        wandb.log(dict(val_loss=val_avg, val_cdir_acc=val_cacc), step=global_step)

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FRAME training")
    parser.add_argument("--debug",  action="store_true", help="Verbose sanity checks")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint (.pt) to resume from")
    args  = parser.parse_args()
    debug = args.debug or bool(os.getenv("DEBUG", ""))
    train(debug=debug, resume=args.resume)
