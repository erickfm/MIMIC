#!/usr/bin/env python3
# train.py  —  FRAME training with strict finiteness checks
#             + AdamW, loss rebalancing, cosine LR, and fp16 AMP

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as nn_utils
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

BATCH_SIZE      = 64
NUM_EPOCHS      = 100
LEARNING_RATE   = 1e-3
WEIGHT_DECAY    = 1e-2            # AdamW (weights only)
NUM_WORKERS     = 4
SEQUENCE_LENGTH = 60
REACTION_DELAY  = 1
DATA_DIR        = "./data"

GRAD_CLIP_NORM      = 1.0
REBALANCE_ALPHA     = 1.0
TASK_NAMES          = ["main", "l", "r", "cdir", "btn"]

USE_AMP         = torch.cuda.is_available()  # fp16 only if CUDA

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
_ce  = nn.CrossEntropyLoss(reduction='mean')
_bce = nn.BCEWithLogitsLoss()

def safe_loss(fn, pred, tgt, name):
    out = fn(pred, tgt)
    if not torch.isfinite(out):
        raise RuntimeError(f"❌ {name} produced non-finite value: {out}")
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
    loss_cdir = safe_loss(_ce,  c_logits,  cdir_tgt.argmax(dim=-1), "c_dir")
    loss_btn  = safe_loss(_bce, btn_pred,  btn_tgt, "btns")

    # —— c-dir accuracy & prediction distribution ————————————————
    cdir_true = cdir_tgt.argmax(dim=-1)
    cdir_pred = c_logits.argmax(dim=-1)
    cdir_acc  = (cdir_pred == cdir_true).float().mean().item()

    # count predictions per class (0=neutral, 1=up, 2=down, 3=left, 4=right)
    pred_counts = torch.bincount(cdir_pred, minlength=5).tolist()
    tgt_counts  = torch.bincount(cdir_true, minlength=5).tolist()

    metrics = {
        "loss_main": loss_main.item(),
        "loss_l":    loss_l.item(),
        "loss_r":    loss_r.item(),
        "loss_cdir": loss_cdir.item(),
        "loss_btn":  loss_btn.item(),
        "cdir_acc":  cdir_acc,
        "cdir_pred_neutral": pred_counts[0],
        "cdir_pred_left":    pred_counts[3],
        "cdir_pred_right":   pred_counts[4],
        "cdir_tgt_neutral":  tgt_counts[0],
        "cdir_tgt_left":     tgt_counts[3],
        "cdir_tgt_right":    tgt_counts[4],
    }
    task_losses = (loss_main, loss_l, loss_r, loss_cdir, loss_btn)
    return metrics, task_losses

# ─────────────────────────────────────────────────────────────────────────────
# 4. Helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_dataset():
    return MeleeFrameDatasetWithDelay(
        parquet_dir=DATA_DIR,
        sequence_length=SEQUENCE_LENGTH,
        reaction_delay=REACTION_DELAY,
    )

def get_dataloader(ds):
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
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

    ds        = get_dataset()
    dl        = get_dataloader(ds)
    model, cfg = get_model()

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

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=NUM_EPOCHS)
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
        epoch_loss, batch_ct = 0.0, 0
        print(f"\n=== Epoch {epoch}/{NUM_EPOCHS} ===")

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
            wandb.log(
                dict(step=global_step,
                     total=total_loss.item(),
                     lr=scheduler.get_last_lr()[0],
                     **metrics),
                step=global_step,
            )
            epoch_loss += total_loss.item()
            batch_ct   += 1

            if i % 25 == 0:
                print(
                    f"[{i:04d}] total={total_loss.item():.4f} "
                    f"main={metrics['loss_main']:.3f} l={metrics['loss_l']:.3f} "
                    f"r={metrics['loss_r']:.3f} cdir={metrics['loss_cdir']:.3f} "
                    f"btn={metrics['loss_btn']:.3f} "
                    f"cdir_acc={metrics['cdir_acc']:.1%}"
                )

        # —— LR step & checkpoint ——————————————————————————————
        scheduler.step()
        avg = epoch_loss / max(batch_ct, 1)
        print(f"Epoch {epoch} done. Avg loss={avg:.4f}")
        os.makedirs("checkpoints", exist_ok=True)
        ckpt_path = f"checkpoints/epoch_{epoch:02d}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict":      model.state_dict(),
            "optimizer_state_dict":  optimiser.state_dict(),
            "scheduler_state_dict":  scheduler.state_dict(),
            "loss_weights":          loss_weights.cpu(),
            "scaler_state_dict":     scaler.state_dict(),
            "config":                cfg.__dict__,
        }, ckpt_path)
        print("Saved checkpoint →", ckpt_path)
        wandb.log(dict(epoch=epoch, avg_loss=avg), step=global_step)

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FRAME training")
    parser.add_argument("--debug",  action="store_true", help="Verbose sanity checks")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint (.pt) to resume from")
    args  = parser.parse_args()
    debug = args.debug or bool(os.getenv("DEBUG", ""))
    train(debug=debug, resume=args.resume)
