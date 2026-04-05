#!/usr/bin/env python3
# train.py  --  MIMIC training with BF16 AMP, torch.compile, epoch/step-based loop
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
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as nn_utils
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast
from pathlib import Path

import wandb

from mimic.dataset import StreamingMeleeDataset
from mimic.model   import FramePredictor, ModelConfig, MODEL_PRESETS

# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------
_is_main = True  # rank 0 or single-GPU; set at start of train()

def _log(*args, **kwargs):
    """Print only on the main process."""
    if _is_main:
        kwargs.setdefault("flush", True)
        print(*args, **kwargs)

# -----------------------------------------------------------------------------
# 1. Config
# -----------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE      = 256
LEARNING_RATE   = 5e-5
WEIGHT_DECAY    = 1e-2
NUM_WORKERS     = 8
SEQUENCE_LENGTH = 60
REACTION_DELAY  = 1
DATA_DIR        = "./data/full"
MAX_SAMPLES     = 2_000_000

GRAD_CLIP_NORM      = 1.0
TASK_NAMES          = ["main", "l", "r", "cdir", "btn"]

FOCAL_GAMMA         = 2.0
LABEL_SMOOTHING     = 0.1

_focal_gamma = FOCAL_GAMMA
_label_smoothing = LABEL_SMOOTHING
_btn_focal_gamma = FOCAL_GAMMA

AMP_DTYPE = torch.bfloat16

MAX_VAL_BATCHES    = 100

# Intervals derived from max_steps at runtime (see _compute_intervals)
WARMUP_FRAC        = 0.05     # 5% warmup
LOG_FRAC           = 0.001    # log every ~0.1%
VAL_FRAC           = 0.01     # validate every ~1%
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
_bce = nn.BCEWithLogitsLoss()

def focal_loss(logits, targets, gamma=None, label_smoothing=None):
    if gamma is None: gamma = _focal_gamma
    if label_smoothing is None: label_smoothing = _label_smoothing
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


def focal_bce(pred, target, gamma=None):
    if gamma is None: gamma = _btn_focal_gamma
    """Focal binary cross-entropy for multi-label classification.

    Downweights easy-to-classify examples (the vast majority of unpressed
    buttons) and focuses gradient on uncertain / rare presses.
    """
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    p = torch.sigmoid(pred)
    pt = p * target + (1 - p) * (1 - target)
    return ((1 - pt).pow(gamma) * bce).mean()


def _multiclass_prf(pred_idx, tgt_idx, n_classes):
    """Macro-averaged precision, recall, F1 for multiclass predictions."""
    tp = torch.zeros(n_classes, device=pred_idx.device)
    fp = torch.zeros(n_classes, device=pred_idx.device)
    fn = torch.zeros(n_classes, device=pred_idx.device)

    correct = pred_idx == tgt_idx
    wrong = ~correct
    tp.scatter_add_(0, tgt_idx[correct], torch.ones_like(tgt_idx[correct], dtype=torch.float))
    fp.scatter_add_(0, pred_idx[wrong], torch.ones_like(pred_idx[wrong], dtype=torch.float))
    fn.scatter_add_(0, tgt_idx[wrong], torch.ones_like(tgt_idx[wrong], dtype=torch.float))

    support = (tp + fn) > 0
    prec = torch.where(tp + fp > 0, tp / (tp + fp), torch.zeros_like(tp))
    rec = torch.where(support, tp / (tp + fn), torch.zeros_like(tp))
    f1 = torch.where(prec + rec > 0, 2 * prec * rec / (prec + rec), torch.zeros_like(prec))

    if support.any():
        return f1[support].mean().item(), prec[support].mean().item(), rec[support].mean().item()
    return 0.0, 0.0, 0.0


def _multi_hot_to_single_label(btn_tgt):
    """Convert multi-hot (B, 12) button targets to single-label (B,) with 5 classes.
    Classes: A=0, B=1, jump(X|Y)=2, Z=3, NO_BUTTON=4.
    Priority: A > B > jump > Z > no_button."""
    a = btn_tgt[..., 0] > 0.5
    b = btn_tgt[..., 1] > 0.5
    jump = (btn_tgt[..., 2] > 0.5) | (btn_tgt[..., 3] > 0.5)
    z = btn_tgt[..., 4] > 0.5
    # Default = NO_BUTTON (4)
    single = torch.full(btn_tgt.shape[:-1], 4, device=btn_tgt.device, dtype=torch.long)
    single[z] = 3
    single[jump] = 2
    single[b] = 1
    single[a] = 0
    return single


# Module-level state for combo encoding (set by train() when --hal-controller-encoding)
_combo_map: dict = None
_combo_lookup: torch.Tensor = None  # vectorized lookup table: combo_int → class_idx


def _build_combo_lookup(combo_map: dict, device: torch.device = None) -> torch.Tensor:
    """Build a vectorized lookup table from combo_map.
    Encodes (a,b,jump,z,shoulder) as int: a*16 + b*8 + jump*4 + z*2 + shoulder.
    Returns tensor of shape (32,) mapping combo_int → class_idx."""
    table = torch.zeros(32, dtype=torch.long, device=device)
    for combo, idx in combo_map.items():
        combo_int = combo[0]*16 + combo[1]*8 + combo[2]*4 + combo[3]*2 + combo[4]
        table[combo_int] = idx
    return table


def _multi_hot_to_combo_label(btn_tgt):
    """Convert multi-hot (B, T, 12) button targets to combo class indices.
    Uses vectorized lookup — no per-frame Python loop."""
    shape = btn_tgt.shape[:-1]
    flat = btn_tgt.reshape(-1, 12)
    # Collapse: X|Y→Jump, L|R→Shoulder
    a = (flat[:, 0] > 0.5).long()
    b = (flat[:, 1] > 0.5).long()
    jump = ((flat[:, 2] > 0.5) | (flat[:, 3] > 0.5)).long()
    z = (flat[:, 4] > 0.5).long()
    shoulder = ((flat[:, 5] > 0.5) | (flat[:, 6] > 0.5)).long()
    # Encode as integer and lookup
    combo_int = a * 16 + b * 8 + jump * 4 + z * 2 + shoulder
    global _combo_lookup
    if _combo_lookup is None or _combo_lookup.device != combo_int.device:
        _combo_lookup = _build_combo_lookup(_combo_map, device=combo_int.device)
    return _combo_lookup[combo_int].reshape(shape)


def _combine_shoulder_targets(l_bin, r_bin, shoulder_centers):
    """Combine separate L/R shoulder bin targets into max(L,R) mapped to 3 classes.
    HAL uses centers [0.0, 0.4, 1.0]."""
    # Decode bin indices back to analog values using our centers, take max, re-quantize
    l_vals = shoulder_centers[l_bin]
    r_vals = shoulder_centers[r_bin]
    combined = torch.max(l_vals, r_vals)
    # Map to nearest of [0.0, 0.4, 1.0]
    hal_centers = torch.tensor([0.0, 0.4, 1.0], device=combined.device)
    dists = (combined.unsqueeze(-1) - hal_centers).abs()
    return dists.argmin(dim=-1)


# 5-class c_dir → 9-cluster mapping: neutral→0, up→4, down→3, left→2, right→1
_CDIR_5_TO_9 = torch.tensor([0, 4, 3, 2, 1], dtype=torch.long)


def _compute_loss_hal(preds, targets, shoulder_centers=None, n_cdir=5):
    """HAL-style loss: plain CE on all heads, single-label buttons, combined shoulder."""
    main_pred = preds["main_xy"].float()
    shldr_pred = preds["shoulder_val"].float()
    c_logits = preds["c_dir_logits"].float()
    btn_pred = preds["btn_logits"].float()  # (B, T, 5) single-label

    main_tgt = targets["main_cluster"].long()
    l_bin_tgt = targets["l_bin"].long()
    r_bin_tgt = targets["r_bin"].long()
    cdir_tgt = targets["c_dir"].long()
    btn_tgt = targets.get("btns", targets.get("btns_float")).float()

    n_main = main_pred.size(-1)
    n_shldr = shldr_pred.size(-1)

    # Main stick
    loss_main = F.cross_entropy(main_pred.reshape(-1, n_main), main_tgt.reshape(-1))

    # Combined shoulder: max(L,R) → 3-class
    if shoulder_centers is not None:
        shldr_tgt = _combine_shoulder_targets(l_bin_tgt.reshape(-1), r_bin_tgt.reshape(-1), shoulder_centers)
    else:
        shldr_tgt = l_bin_tgt.reshape(-1)  # fallback
    loss_shldr = F.cross_entropy(shldr_pred.reshape(-1, n_shldr), shldr_tgt)

    # C-dir (remap 5-class → 9-cluster if needed)
    cdir_classes = cdir_tgt.argmax(dim=-1)
    if n_cdir == 9:
        cdir_classes = _CDIR_5_TO_9.to(cdir_classes.device)[cdir_classes]
    loss_cdir = F.cross_entropy(c_logits.reshape(-1, c_logits.size(-1)), cdir_classes.reshape(-1))

    # Single-label buttons (combo-based if controller encoding, else priority-based)
    if _combo_map is not None:
        btn_single = _multi_hot_to_combo_label(btn_tgt)
    else:
        btn_single = _multi_hot_to_single_label(btn_tgt)
    loss_btn = F.cross_entropy(btn_pred.reshape(-1, btn_pred.size(-1)), btn_single.reshape(-1))

    # Metrics
    main_flat = main_pred.reshape(-1, n_main)
    main_tgt_flat = main_tgt.reshape(-1)
    main_pred_idx = main_flat.argmax(-1)
    main_top1 = (main_pred_idx == main_tgt_flat).float().mean().item()
    main_f1, main_prec, main_rec = _multiclass_prf(main_pred_idx, main_tgt_flat, n_main)

    shldr_pred_idx = shldr_pred.reshape(-1, n_shldr).argmax(-1)
    shldr_f1, shldr_prec, shldr_rec = _multiclass_prf(shldr_pred_idx, shldr_tgt, n_shldr)
    shldr_top1 = (shldr_pred_idx == shldr_tgt).float().mean().item()

    cdir_flat = cdir_classes.reshape(-1)
    cdir_pred = c_logits.reshape(-1, c_logits.size(-1)).argmax(-1)
    cdir_acc = (cdir_pred == cdir_flat).float().mean().item()
    cdir_f1, cdir_prec, cdir_rec = _multiclass_prf(cdir_pred, cdir_flat, c_logits.size(-1))

    btn_flat = btn_single.reshape(-1)
    btn_pred_idx = btn_pred.reshape(-1, btn_pred.size(-1)).argmax(-1)
    btn_acc = (btn_pred_idx == btn_flat).float().mean().item()
    btn_f1, btn_prec, btn_rec = _multiclass_prf(btn_pred_idx, btn_flat, btn_pred.size(-1))

    cdir_active_mask = cdir_flat != 0
    cdir_active_acc = (cdir_pred[cdir_active_mask] == cdir_flat[cdir_active_mask]).float().mean().item() if cdir_active_mask.any() else 0.0

    metrics = {
        "loss_main": loss_main.item(),
        "loss_l":    loss_shldr.item(),  # reuse key for compatibility
        "loss_r":    0.0,
        "loss_cdir": loss_cdir.item(),
        "loss_btn":  loss_btn.item(),
        "cdir_acc":  cdir_acc,
        "btn_acc":   btn_acc,
        "btn_f1":       btn_f1,
        "btn_precision": btn_prec,
        "btn_recall":    btn_rec,
        "main_f1":       main_f1,
        "main_precision": main_prec,
        "main_recall":    main_rec,
        "shldr_f1":       shldr_f1,
        "shldr_precision": shldr_prec,
        "shldr_recall":    shldr_rec,
        "cdir_f1":       cdir_f1,
        "cdir_precision": cdir_prec,
        "cdir_recall":    cdir_rec,
        "cdir_active_acc": cdir_active_acc,
        "main_top1_acc":    main_top1,
        "shoulder_top1_acc": shldr_top1,
    }
    task_losses = (loss_main, loss_shldr, torch.tensor(0.0, device=loss_main.device), loss_cdir, loss_btn)
    return metrics, task_losses


def compute_loss(preds, targets, btn_loss_type="focal", plain_ce=False, hal_mode=False,
                  shoulder_centers=None, n_cdir=5):
    if hal_mode:
        return _compute_loss_hal(preds, targets, shoulder_centers, n_cdir=n_cdir)

    c_logits = preds["c_dir_logits"].float()
    btn_pred = preds["btn_logits"].float()

    cdir_tgt = targets["c_dir"].long()
    btn_tgt  = targets.get("btns", targets.get("btns_float")).float()

    main_pred = preds["main_xy"].float()
    l_pred = preds["L_val"].float()
    r_pred = preds["R_val"].float()
    n_main = main_pred.size(-1)
    n_shldr = l_pred.size(-1)

    main_cluster_tgt = targets["main_cluster"].long()
    l_bin_tgt = targets["l_bin"].long()
    r_bin_tgt = targets["r_bin"].long()

    _ce = F.cross_entropy if plain_ce else focal_loss
    loss_main = _ce(main_pred.reshape(-1, n_main),
                    main_cluster_tgt.reshape(-1))
    loss_l = _ce(l_pred.reshape(-1, n_shldr),
                 l_bin_tgt.reshape(-1))
    loss_r = _ce(r_pred.reshape(-1, n_shldr),
                 r_bin_tgt.reshape(-1))

    main_pred_flat = main_pred.reshape(-1, n_main)
    main_tgt_flat = main_cluster_tgt.reshape(-1)
    main_pred_idx = main_pred_flat.argmax(-1)
    main_top1 = (main_pred_idx == main_tgt_flat).float().mean().item()
    main_f1, main_prec, main_rec = _multiclass_prf(main_pred_idx, main_tgt_flat, n_main)

    l_pred_flat = l_pred.reshape(-1, n_shldr)
    l_tgt_flat = l_bin_tgt.reshape(-1)
    r_pred_flat = r_pred.reshape(-1, n_shldr)
    r_tgt_flat = r_bin_tgt.reshape(-1)
    l_pred_idx = l_pred_flat.argmax(-1)
    r_pred_idx = r_pred_flat.argmax(-1)
    l_top1 = (l_pred_idx == l_tgt_flat).float().mean().item()
    r_top1 = (r_pred_idx == r_tgt_flat).float().mean().item()
    shldr_pred_idx = torch.cat([l_pred_idx, r_pred_idx])
    shldr_tgt_idx = torch.cat([l_tgt_flat, r_tgt_flat])
    shldr_f1, shldr_prec, shldr_rec = _multiclass_prf(shldr_pred_idx, shldr_tgt_idx, n_shldr)

    cdir_classes = cdir_tgt.argmax(dim=-1)
    c_logits_flat = c_logits.reshape(-1, c_logits.size(-1))
    cdir_flat     = cdir_classes.reshape(-1)
    loss_cdir = _ce(c_logits_flat, cdir_flat)
    # Only compute button loss on active buttons (first 7: A,B,X,Y,Z,L,R)
    active_btn_pred = btn_pred[..., :7]
    active_btn_tgt = btn_tgt[..., :7]
    if plain_ce:
        loss_btn = _bce(active_btn_pred, active_btn_tgt)
    else:
        loss_btn = focal_bce(active_btn_pred, active_btn_tgt) if btn_loss_type == "focal" else _bce(active_btn_pred, active_btn_tgt)

    cdir_pred = c_logits_flat.argmax(dim=-1)
    cdir_acc  = (cdir_pred == cdir_flat).float().mean().item()
    cdir_f1, cdir_prec, cdir_rec = _multiclass_prf(cdir_pred, cdir_flat, c_logits.size(-1))

    btn_prob = torch.sigmoid(active_btn_pred)
    btn_hat  = btn_prob > 0.5
    btn_ref  = active_btn_tgt > 0.5
    btn_acc  = (btn_hat == btn_ref).float().mean().item()

    tp = (btn_hat & btn_ref).sum().float()
    fp = (btn_hat & ~btn_ref).sum().float()
    fn = (~btn_hat & btn_ref).sum().float()
    btn_precision = (tp / (tp + fp)).item() if (tp + fp) > 0 else 0.0
    btn_recall    = (tp / (tp + fn)).item() if (tp + fn) > 0 else 0.0
    btn_f1 = (2 * btn_precision * btn_recall / (btn_precision + btn_recall)
              if (btn_precision + btn_recall) > 0 else 0.0)

    cdir_active_mask = cdir_flat != 0
    if cdir_active_mask.any():
        cdir_active_acc = (cdir_pred[cdir_active_mask] == cdir_flat[cdir_active_mask]).float().mean().item()
    else:
        cdir_active_acc = 0.0

    metrics = {
        "loss_main": loss_main.item(),
        "loss_l":    loss_l.item(),
        "loss_r":    loss_r.item(),
        "loss_cdir": loss_cdir.item(),
        "loss_btn":  loss_btn.item(),
        "cdir_acc":  cdir_acc,
        "btn_acc":   btn_acc,
        "btn_f1":       btn_f1,
        "btn_precision": btn_precision,
        "btn_recall":    btn_recall,
        "main_f1":       main_f1,
        "main_precision": main_prec,
        "main_recall":    main_rec,
        "shldr_f1":       shldr_f1,
        "shldr_precision": shldr_prec,
        "shldr_recall":    shldr_rec,
        "cdir_f1":       cdir_f1,
        "cdir_precision": cdir_prec,
        "cdir_recall":    cdir_rec,
        "cdir_active_acc": cdir_active_acc,
        "main_top1_acc":    main_top1,
        "shoulder_top1_acc": (l_top1 + r_top1) / 2,
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


def get_datasets(data_dir, no_opp_inputs=False, no_self_inputs=True,
                  rank=0, world_size=1, controller_offset=False,
                  hal_controller_encoding=False,
                  controller_combo_map=None, n_controller_combos=5):
    _log(f"  Using streaming dataset from {data_dir}")
    ctrl_kw = dict(
        hal_controller_encoding=hal_controller_encoding,
        controller_combo_map=controller_combo_map,
        n_controller_combos=n_controller_combos,
    )
    train_ds = StreamingMeleeDataset(
        data_dir=data_dir,
        sequence_length=SEQUENCE_LENGTH,
        reaction_delay=REACTION_DELAY,
        split="train",
        rank=rank,
        world_size=world_size,
        controller_offset=controller_offset,
        **ctrl_kw,
    )
    val_ds = StreamingMeleeDataset(
        data_dir=data_dir,
        sequence_length=SEQUENCE_LENGTH,
        reaction_delay=REACTION_DELAY,
        split="val",
        rank=rank,
        world_size=world_size,
        controller_offset=controller_offset,
        **ctrl_kw,
    )
    return train_ds, val_ds

def get_dataloader(ds, shuffle=True, persistent=True, sampler=None):
    is_iterable = isinstance(ds, IterableDataset)
    nw = NUM_WORKERS if persistent else 0
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=(shuffle and not is_iterable and sampler is None),
        sampler=sampler,
        num_workers=nw,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=(nw > 0),
        persistent_workers=(persistent and nw > 0),
    )

def get_model(compile_model=True, model_preset=None, num_layers_override=None,
              encoder_type=None, d_intra=None, dropout_override=None,
              k_query=None, intra_layers=None, scaled_emb=False,
              pos_enc=None, attn_variant=None, n_kv_heads=None,
              btn_loss=None, no_opp_inputs=True, no_self_inputs=True,
              n_stick_clusters=None, n_shoulder_bins=None,
              n_heads_override=None, hal_mode=False, lean_features=False,
              hal_minimal_features=False,
              hal_controller_encoding=False, n_controller_combos=5):
    overrides = MODEL_PRESETS.get(model_preset, {}) if model_preset else {}
    if num_layers_override:
        overrides["num_layers"] = num_layers_override
    if encoder_type:
        overrides["encoder_type"] = encoder_type
    if d_intra is not None:
        overrides["d_intra"] = d_intra
    if dropout_override is not None:
        overrides["dropout"] = dropout_override
    if k_query is not None:
        overrides["k_query"] = k_query
    if intra_layers is not None:
        overrides["encoder_nlayers"] = intra_layers
    if scaled_emb:
        overrides["scaled_emb"] = True
    if pos_enc:
        overrides["pos_enc"] = pos_enc
    if attn_variant:
        overrides["attn_variant"] = attn_variant
    if n_kv_heads:
        overrides["n_kv_heads"] = n_kv_heads
    if btn_loss:
        overrides["btn_loss"] = btn_loss
    overrides["no_opp_inputs"] = no_opp_inputs
    overrides["no_self_inputs"] = no_self_inputs
    if n_heads_override:
        overrides["nhead"] = n_heads_override
    if hal_mode:
        overrides["hal_mode"] = True
    if lean_features:
        overrides["lean_features"] = True
    if hal_minimal_features:
        overrides["hal_minimal_features"] = True
    if hal_controller_encoding:
        overrides["hal_controller_encoding"] = True
        overrides["n_controller_combos"] = n_controller_combos
    if n_stick_clusters is not None:
        overrides["n_stick_clusters"] = n_stick_clusters
    if n_shoulder_bins is not None:
        overrides["n_shoulder_bins"] = n_shoulder_bins
    seq_len = overrides.pop("max_seq_len", SEQUENCE_LENGTH)
    cfg = ModelConfig(max_seq_len=seq_len, **overrides)
    model = FramePredictor(cfg).to(DEVICE)
    if compile_model:
        model = torch.compile(model)
    return model, cfg

def infinite_loader(dl, sampler=None):
    """Yield batches forever, cycling through epochs."""
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in dl:
            yield batch
        epoch += 1

# -----------------------------------------------------------------------------
# 5. Training loop (step-based)
# -----------------------------------------------------------------------------
def _auto_run_name(preset: str, lr: float, seq_len: int, extra: dict = None) -> str:
    """Generate a descriptive run name from config."""
    lr_str = f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e")
    parts = [preset or "base", f"lr{lr_str}"]
    if seq_len != 60:
        parts.append(f"{seq_len}f")
    if extra:
        for k, v in extra.items():
            parts.append(f"{k}{v}")
    return "-".join(parts)


def train(epochs: int = None, max_steps: int = None, max_samples: int = MAX_SAMPLES,
          data_dir: str = DATA_DIR,
          debug: bool = False, resume: str = None, compile_model: bool = True,
          model_preset: str = None, lr: float = None, run_name: str = None,
          wandb_tags: list = None, wandb_group: str = None,
          num_layers_override: int = None, seq_len_override: int = None,
          batch_size_override: int = None,
          encoder_type: str = None, d_intra: int = None,
          dropout_override: float = None, k_query: int = None,
          intra_layers: int = None, scaled_emb: bool = False,
          pos_enc: str = None, attn_variant: str = None, n_kv_heads: int = None,
          btn_loss: str = None,
          no_opp_inputs: bool = True,
          no_self_inputs: bool = False,
          clusters_path: str = None,
          target_val_f1: float = None,
          max_wall_time: float = None,
          val_frac_override: float = None,
          warmup_steps_override: int = None,
          weight_decay_override: float = None,
          scale_lr: bool = False,
          grad_accum_steps: int = 1,
          nccl_timeout: int = 1800,
          grad_clip_norm: float = None,
          no_load_optim: bool = False,
          si_drop_start: float = None, si_drop_end: float = None,
          si_drop_max: float = 1.0,
          plain_ce: bool = False,
          n_heads_override: int = None,
          hal_mode: bool = False,
          no_amp: bool = False,
          cosine_min_lr: float = None,
          lean_features: bool = False,
          no_warmup: bool = False,
          stick_clusters: str = None,
          hal_minimal_features: bool = False,
          reaction_delay_override: int = None,
          controller_offset: bool = False,
          hal_controller_encoding: bool = False):
    if debug:
        torch.autograd.set_detect_anomaly(True)

    # --- Distributed setup (auto-detect torchrun via RANK env var) ---
    global _is_main, DEVICE
    is_distributed = "RANK" in os.environ
    if is_distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        DEVICE = torch.device(f"cuda:{local_rank}")
        from datetime import timedelta
        dist.init_process_group("nccl", device_id=DEVICE,
                                timeout=timedelta(seconds=nccl_timeout))
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        rank, local_rank, world_size = 0, 0, 1
    _is_main = (rank == 0)
    if not _is_main:
        os.environ["WANDB_MODE"] = "disabled"

    global SEQUENCE_LENGTH, BATCH_SIZE, VAL_FRAC, WEIGHT_DECAY, GRAD_CLIP_NORM, AMP_DTYPE, REACTION_DELAY
    if reaction_delay_override is not None:
        REACTION_DELAY = reaction_delay_override
    if seq_len_override:
        SEQUENCE_LENGTH = seq_len_override
    elif model_preset == "hal":
        SEQUENCE_LENGTH = 256
    if batch_size_override:
        BATCH_SIZE = batch_size_override
    if val_frac_override is not None:
        VAL_FRAC = val_frac_override
    if weight_decay_override is not None:
        WEIGHT_DECAY = weight_decay_override
    if no_amp:
        AMP_DTYPE = torch.float32
    if grad_clip_norm is not None:
        GRAD_CLIP_NORM = grad_clip_norm

    eff_batch = BATCH_SIZE * world_size * grad_accum_steps
    if is_distributed:
        _log(f"DDP: {world_size} GPUs (rank {rank}, local_rank {local_rank}), "
             f"per-GPU batch={BATCH_SIZE}, accum={grad_accum_steps}, effective batch={eff_batch}")
    elif grad_accum_steps > 1:
        _log(f"Gradient accumulation: {grad_accum_steps} steps, effective batch={eff_batch}")
    if eff_batch > 512:
        _log(f"  WARNING: eff batch {eff_batch} > 512 — known to plateau on full Melee data")

    # --- Controller combo loading ---
    global _combo_map
    _combo_map = None
    _n_combos = 5  # default (HAL's A, B, Jump, Z, NONE)
    if hal_controller_encoding:
        from mimic.features import load_controller_combos
        combos_path = Path(data_dir) / "controller_combos.json"
        if combos_path.exists():
            combos, _combo_map = load_controller_combos(data_dir)
            _n_combos = len(combos)
            _log(f"  Controller encoding: {_n_combos} button combos from {combos_path}")
        else:
            raise RuntimeError(
                f"--hal-controller-encoding requires controller_combos.json in {data_dir}. "
                f"Run: python tools/build_controller_combos.py --data-dir {data_dir}")

    _log(f"Loading dataset from {data_dir} ...")
    ds, val_ds = get_datasets(data_dir, no_opp_inputs=no_opp_inputs,
                              no_self_inputs=no_self_inputs,
                              rank=rank, world_size=world_size,
                              controller_offset=controller_offset,
                              hal_controller_encoding=hal_controller_encoding,
                              controller_combo_map=_combo_map,
                              n_controller_combos=_n_combos)
    n_train = len(ds)
    n_val   = len(val_ds)
    n_train_games = getattr(ds, "n_games", len(getattr(ds, "files", [])))
    n_val_games   = getattr(val_ds, "n_games", len(getattr(val_ds, "files", [])))
    _log(f"  Train: {n_train:,} windows from {n_train_games} games")
    _log(f"  Val:   {n_val:,} windows from {n_val_games} games")

    train_sampler, val_sampler = None, None
    if is_distributed and not isinstance(ds, IterableDataset):
        train_sampler = DistributedSampler(ds, shuffle=True)
        val_sampler = DistributedSampler(val_ds, shuffle=False)

    dl     = get_dataloader(ds, sampler=train_sampler)
    val_dl = get_dataloader(val_ds, shuffle=False, persistent=False,
                            sampler=val_sampler)

    est_batches = max(n_train // BATCH_SIZE, 1)
    if max_samples and max_steps is None:
        max_steps = max_samples // BATCH_SIZE
        _log(f"  {max_samples:,} samples / bs {BATCH_SIZE} = {max_steps:,} steps")
    elif max_steps is None:
        if epochs is None:
            epochs = 1
        max_steps = epochs * est_batches
        _log(f"  {epochs} epoch(s) x {est_batches} batches = {max_steps:,} steps")
    else:
        epoch_frac = max_steps / est_batches
        _log(f"  {max_steps:,} steps (~{epoch_frac:.2f} epochs)")

    intervals = _compute_intervals(max_steps)
    log_interval  = intervals["log_interval"]
    val_interval  = intervals["val_interval"]
    ckpt_interval = intervals["ckpt_interval"]
    warmup_steps  = intervals["warmup_steps"]
    if no_warmup:
        warmup_steps = 0
    elif warmup_steps_override is not None:
        warmup_steps = warmup_steps_override

    actual_lr = lr or LEARNING_RATE
    if scale_lr and world_size > 1:
        actual_lr *= world_size
        _log(f"  LR scaled by {world_size}x: {lr or LEARNING_RATE} → {actual_lr}")

    _log(f"  Compiling model: {compile_model}  preset: {model_preset or 'base'}")
    from mimic.features import load_cluster_centers
    _cp = Path(clusters_path) if clusters_path else None
    stick_centers_np, shoulder_centers_np = load_cluster_centers(
        data_dir=Path(data_dir), clusters_path=_cp, stick_clusters=stick_clusters)
    if stick_centers_np is None:
        raise RuntimeError(
            f"No stick_clusters.json found "
            f"(checked: {clusters_path}, {data_dir}, data/full/)")
    n_stick_clusters = len(stick_centers_np)
    if hal_minimal_features and hal_mode:
        # Force 3-class shoulder output, but keep data's centers for decoding
        _shoulder_centers_for_decode = torch.tensor(shoulder_centers_np, dtype=torch.float32, device=DEVICE)
        n_shoulder_bins = 3
    else:
        _shoulder_centers_for_decode = None
        n_shoulder_bins = len(shoulder_centers_np)
    _shoulder_centers = torch.tensor(shoulder_centers_np if not hal_minimal_features else [0.0, 0.4, 1.0],
                                     dtype=torch.float32, device=DEVICE)
    _stick_centers = torch.tensor(stick_centers_np, dtype=torch.float32, device=DEVICE)
    _recluster_sticks = stick_clusters is not None  # re-assign if using non-data clusters
    _log(f"  Clusters: {n_stick_clusters} stick, {n_shoulder_bins} shoulder"
         f"{'  (re-clustering at runtime)' if _recluster_sticks else ''}")

    model, cfg = get_model(compile_model=compile_model, model_preset=model_preset,
                           num_layers_override=num_layers_override,
                           encoder_type=encoder_type, d_intra=d_intra,
                           dropout_override=dropout_override, k_query=k_query,
                           intra_layers=intra_layers, scaled_emb=scaled_emb,
                           pos_enc=pos_enc, attn_variant=attn_variant,
                           n_kv_heads=n_kv_heads,
                           btn_loss=btn_loss, no_opp_inputs=no_opp_inputs,
                           no_self_inputs=no_self_inputs,
                           n_stick_clusters=n_stick_clusters,
                           n_shoulder_bins=n_shoulder_bins,
                           n_heads_override=n_heads_override,
                           hal_mode=hal_mode,
                           lean_features=lean_features,
                           hal_minimal_features=hal_minimal_features,
                           hal_controller_encoding=hal_controller_encoding,
                           n_controller_combos=_n_combos)
    n_params = sum(p.numel() for p in model.parameters())
    _log(f"  Model: {n_params:,} params on {DEVICE}  (AMP={AMP_DTYPE}, LR={actual_lr})")
    _log(f"  Intervals: log={log_interval}  val={ckpt_interval}  "
         f"ckpt={ckpt_interval}  warmup={warmup_steps}")

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

    _eta_min = cosine_min_lr if cosine_min_lr is not None else 0.0
    if warmup_steps > 0:
        warmup  = optim.lr_scheduler.LinearLR(
            optimiser, start_factor=0.01, total_iters=warmup_steps)
        cosine  = optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=max(max_steps - warmup_steps, 1), eta_min=_eta_min)
        scheduler = optim.lr_scheduler.SequentialLR(
            optimiser, [warmup, cosine], milestones=[warmup_steps])
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=max_steps, eta_min=_eta_min)

    start_step = 0
    if resume:
        ckpt = torch.load(resume, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        if not no_load_optim:
            optimiser.load_state_dict(ckpt['optimizer_state_dict'])
            if ckpt.get("scheduler_state_dict"):
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_step = ckpt.get('global_step', 0)
        _log(f"Resumed from {resume}, starting at step {start_step}"
             + (f" (fresh optimizer, lr={actual_lr})" if no_load_optim else ""))

    # --- DDP wrapping (after compile, optimizer, and checkpoint load) ---
    if is_distributed:
        model = DDP(model, device_ids=[local_rank],
                    find_unused_parameters=True)
    _raw_model = model.module if is_distributed else model

    # -- Self-input curriculum schedule --
    _si_schedule = None
    if si_drop_start is not None and si_drop_end is not None:
        si_start_step = int(si_drop_start * max_steps)
        si_end_step = int(si_drop_end * max_steps)
        def _si_drop_prob(step):
            if step < si_start_step:
                return 0.0
            if step >= si_end_step:
                return si_drop_max
            return si_drop_max * (step - si_start_step) / (si_end_step - si_start_step)
        _si_schedule = _si_drop_prob
        _log(f"SI curriculum: drop 0→{si_drop_max} over steps {si_start_step}→{si_end_step}")

    if not run_name:
        extra = {}
        if num_layers_override:
            extra["L"] = num_layers_override
        if encoder_type and encoder_type != "default":
            extra["enc"] = encoder_type
        run_name = _auto_run_name(model_preset, actual_lr, SEQUENCE_LENGTH, extra)

    wandb.init(
        project="MIMIC",
        entity="erickfm",
        name=run_name,
        group=wandb_group,
        tags=wandb_tags or [],
        config=dict(
            batch_size=BATCH_SIZE,
            effective_batch_size=BATCH_SIZE * world_size,
            world_size=world_size,
            learning_rate=actual_lr,
            max_steps=max_steps,
            total_samples=max_steps * BATCH_SIZE,
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
            si_drop_start=si_drop_start,
            si_drop_end=si_drop_end,
            si_drop_max=si_drop_max,
            **cfg.__dict__,
        ),
    )
    wandb.run.summary["n_params"] = n_params

    loader = infinite_loader(dl, sampler=train_sampler)
    skip_ct = 0
    t0 = time.time()

    _AVG_KEYS = ["total", "loss_main", "loss_l", "loss_r", "loss_cdir",
                 "loss_btn", "cdir_acc", "btn_acc", "btn_f1",
                 "btn_precision", "btn_recall",
                 "main_f1", "main_precision", "main_recall",
                 "shldr_f1", "shldr_precision", "shldr_recall",
                 "cdir_f1", "cdir_precision", "cdir_recall",
                 "cdir_active_acc",
                 "main_top1_acc", "shoulder_top1_acc", "grad_norm"]
    _run_sums = {k: 0.0 for k in _AVG_KEYS}
    _run_ct = 0

    if start_step and not isinstance(ds, IterableDataset):
        for _ in range(start_step):
            next(loader)

    _oom_retries = 0
    best_val_f1 = -1.0
    _log(f"\n=== Training {max_steps} steps ===")
    _target_hit = False
    for step in range(start_step + 1, max_steps + 1):
        if max_wall_time and (time.time() - t0) > max_wall_time:
            _log(f"WALL TIME LIMIT ({max_wall_time}s) exceeded at step {step}")
            break

        model.train()
        if _si_schedule:
            _raw_model.encoder.si_drop_prob = _si_schedule(step)
        optimiser.zero_grad(set_to_none=True)

        # -- Gradient accumulation loop --
        accum_loss = 0.0
        accum_metrics = None
        skip_step = False
        for _micro in range(grad_accum_steps):
            # Skip DDP gradient sync on all but last micro-batch
            maybe_no_sync = (model.no_sync() if is_distributed and _micro < grad_accum_steps - 1
                             else nullcontext())
            state, target = next(loader)
            for k, v in state.items():
                state[k] = v.to(DEVICE, non_blocking=True)
            for k, v in target.items():
                target[k] = v.to(DEVICE, non_blocking=True)

            try:
                # Re-assign stick clusters if using non-data clusters (e.g. hal37)
                if _recluster_sticks and "main_x" in target:
                    xy = torch.stack([target["main_x"], target["main_y"]], dim=-1)
                    dists = torch.cdist(xy.reshape(-1, 2), _stick_centers)
                    target["main_cluster"] = dists.argmin(dim=-1).reshape(xy.shape[:-1])

                btn_tgt = target.get("btns", target.get("btns_float"))
                with autocast("cuda", dtype=AMP_DTYPE):
                    preds = model(state, btn_targets=btn_tgt if not hal_mode else None)
                metrics, task_losses = compute_loss(
                    preds, target, btn_loss_type=cfg.btn_loss, plain_ce=plain_ce,
                    hal_mode=hal_mode, shoulder_centers=_shoulder_centers_for_decode if _shoulder_centers_for_decode is not None else _shoulder_centers,
                    n_cdir=cfg.num_c_dirs)
            except torch.cuda.OutOfMemoryError:
                mem = torch.cuda.max_memory_allocated() / 1e9
                _log(f"OOM on rank {rank}, device {DEVICE}: {mem:.1f}GB peak")
                if is_distributed:
                    raise RuntimeError(
                        f"OOM on rank {rank} ({mem:.1f}GB peak). "
                        f"Reduce --batch-size or add --grad-accum-steps.")
                torch.cuda.empty_cache()
                _oom_retries += 1
                if _oom_retries <= 3:
                    BATCH_SIZE = max(BATCH_SIZE // 2, 1)
                    _log(f"  [OOM] Halving batch size to {BATCH_SIZE}, retry #{_oom_retries}")
                    dl = get_dataloader(ds, shuffle=True)
                    loader = infinite_loader(dl)
                    skip_step = True
                    break
                else:
                    raise RuntimeError(f"OOM after {_oom_retries} retries (batch_size={BATCH_SIZE})")

            loss_vec = torch.stack(task_losses)
            micro_loss = (loss_weights * loss_vec).sum() / grad_accum_steps

            if not torch.isfinite(micro_loss):
                skip_step = True
                skip_ct += 1
                _log(f"  [skip] step {step} -- non-finite loss ({skip_ct} total)")
                break

            with maybe_no_sync:
                micro_loss.backward()
            accum_loss += micro_loss.item()

            # Accumulate metrics (use last micro-batch for per-step logging)
            accum_metrics = metrics

        if skip_step:
            optimiser.zero_grad(set_to_none=True)
            scheduler.step()
            continue

        grad_norm = nn_utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM).item()
        has_inf_grad = not math.isfinite(grad_norm)
        if has_inf_grad:
            skip_ct += 1
            _log(f"  [skip] step {step} -- inf grad norm ({skip_ct} total)")
            optimiser.zero_grad(set_to_none=True)
            scheduler.step()
            continue

        optimiser.step()
        scheduler.step()

        total_loss_val = accum_loss  # already averaged across micro-batches
        metrics = accum_metrics
        _run_sums["total"]     += total_loss_val
        _run_sums["grad_norm"] += grad_norm
        for _mk in _AVG_KEYS:
            if _mk not in ("total", "grad_norm"):
                _run_sums[_mk] += metrics[_mk]
        _run_ct += 1

        if step % log_interval == 0 or step == 1:
            elapsed = time.time() - t0
            sps = step / elapsed

            if _run_ct > 0:
                avg_log = {f"avg/{k}": _run_sums[k] / _run_ct for k in _AVG_KEYS}
                avg_log["perf/step_per_sec"]    = sps
                avg_log["perf/samples_per_sec"] = sps * BATCH_SIZE * world_size * grad_accum_steps
                if _si_schedule:
                    avg_log["curriculum/si_drop_prob"] = _raw_model.encoder.si_drop_prob
                wandb.log(avg_log, step=step)
                _run_sums = {k: 0.0 for k in _AVG_KEYS}
                _run_ct = 0

            _log(
                f"[{step:5d}/{max_steps}] "
                f"total={total_loss_val:.4f} "
                f"main={metrics['loss_main']:.5f} "
                f"l={metrics['loss_l']:.5f} "
                f"r={metrics['loss_r']:.5f} "
                f"cdir={metrics['loss_cdir']:.4f} "
                f"btn={metrics['loss_btn']:.5f} "
                f"bf1={metrics['btn_f1']:.1%} "
                f"bP={metrics['btn_precision']:.1%} bR={metrics['btn_recall']:.1%} "
                f"mf1={metrics['main_f1']:.1%} "
                f"sf1={metrics['shldr_f1']:.1%} "
                f"cf1={metrics['cdir_f1']:.1%} "
                f"gnorm={grad_norm:.2f} "
                f"lr={scheduler.get_last_lr()[0]:.2e} "
                + (f"si_drop={_raw_model.encoder.si_drop_prob:.2f} " if _si_schedule else "")
                + f"({sps:.1f} step/s)",
            )

        # Validation
        if step % val_interval == 0 or step == max_steps:
            model.eval()
            if _si_schedule:
                _saved_si_prob = _raw_model.encoder.si_drop_prob
                _raw_model.encoder.si_drop_prob = 0.0
            _VAL_KEYS = ["total", "loss_main", "loss_l", "loss_r",
                         "loss_cdir", "loss_btn", "cdir_acc", "btn_acc",
                         "btn_f1", "btn_precision", "btn_recall",
                         "main_f1", "main_precision", "main_recall",
                         "shldr_f1", "shldr_precision", "shldr_recall",
                         "cdir_f1", "cdir_precision", "cdir_recall",
                         "cdir_active_acc",
                         "main_top1_acc", "shoulder_top1_acc"]
            val_sums = {k: 0.0 for k in _VAL_KEYS}
            val_ct = 0
            with torch.no_grad():
                for vs, vt in val_dl:
                    for k, v in vs.items():
                        vs[k] = v.to(DEVICE, non_blocking=True)
                    for k, v in vt.items():
                        vt[k] = v.to(DEVICE, non_blocking=True)
                    if _recluster_sticks and "main_x" in vt:
                        xy = torch.stack([vt["main_x"], vt["main_y"]], dim=-1)
                        dists = torch.cdist(xy.reshape(-1, 2), _stick_centers)
                        vt["main_cluster"] = dists.argmin(dim=-1).reshape(xy.shape[:-1])
                    with autocast("cuda", dtype=AMP_DTYPE):
                        vpreds = model(vs)
                    vm, vtl = compute_loss(vpreds, vt,
                                           btn_loss_type=cfg.btn_loss, plain_ce=plain_ce,
                                           hal_mode=hal_mode, shoulder_centers=_shoulder_centers_for_decode if _shoulder_centers_for_decode is not None else _shoulder_centers,
                                           n_cdir=cfg.num_c_dirs)
                    batch_total = sum(t.item() for t in vtl)
                    if math.isfinite(batch_total):
                        val_sums["total"] += batch_total
                        for _vk in _VAL_KEYS:
                            if _vk != "total":
                                val_sums[_vk] += vm[_vk]
                        val_ct += 1
                    if val_ct >= MAX_VAL_BATCHES:
                        break

            if val_ct > 0:
                val_avg = {f"val/{k}": val_sums[k] / val_ct for k in _VAL_KEYS}
                wandb.log(val_avg, step=step)
                _log(f"  -- val total={val_avg['val/total']:.4f}  "
                     f"bf1={val_avg['val/btn_f1']:.1%} "
                     f"bP={val_avg['val/btn_precision']:.1%} bR={val_avg['val/btn_recall']:.1%}  "
                     f"mf1={val_avg['val/main_f1']:.1%} "
                     f"sf1={val_avg['val/shldr_f1']:.1%} "
                     f"cf1={val_avg['val/cdir_f1']:.1%}  "
                     f"(batches={val_ct})")
                cur_val_f1 = val_avg.get('val/btn_f1', 0.0)
                if cur_val_f1 > best_val_f1:
                    best_val_f1 = cur_val_f1
                    if _is_main:
                        os.makedirs("checkpoints", exist_ok=True)
                        best_path = f"checkpoints/{run_name}_best.pt"
                        best_ckpt = {
                            "global_step":          step,
                            "model_state_dict":     _raw_model.state_dict(),
                            "optimizer_state_dict": optimiser.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "loss_weights":         loss_weights.cpu(),
                            "norm_stats":           ds.norm_stats,
                            "config":               cfg.__dict__,
                        }
                        if stick_centers_np is not None:
                            best_ckpt["stick_centers"] = stick_centers_np.tolist()
                        if shoulder_centers_np is not None:
                            best_ckpt["shoulder_centers"] = shoulder_centers_np.tolist()
                        if hal_controller_encoding and _combo_map:
                            best_ckpt["controller_combos"] = [list(c) for c in sorted(_combo_map.keys(), key=_combo_map.get)]
                        torch.save(best_ckpt, best_path)
                        _log(f"  -- new best val f1={cur_val_f1:.1%} → {best_path}")
                if target_val_f1 and cur_val_f1 >= target_val_f1:
                    _elapsed = time.time() - t0
                    _log(f"TARGET REACHED: val_f1={cur_val_f1:.4f} at step {step} in {_elapsed:.1f}s")
                    wandb.log({"target_reached_step": step, "target_reached_wall_s": _elapsed}, step=step)
                    _target_hit = True
                    break
            else:
                _log("  -- val: no valid batches")
            if is_distributed:
                dist.barrier()

            # Restore si_drop_prob after normal val
            if _si_schedule:
                _raw_model.encoder.si_drop_prob = _saved_si_prob

            # Dual validation: eval with all self-inputs dropped
            if _si_schedule and _saved_si_prob > 0:
                _raw_model.encoder.si_drop_prob = 1.0
                val_nsi_sums = {k: 0.0 for k in _VAL_KEYS}
                val_nsi_ct = 0
                with torch.no_grad():
                    for vs, vt in val_dl:
                        for k, v in vs.items():
                            vs[k] = v.to(DEVICE, non_blocking=True)
                        for k, v in vt.items():
                            vt[k] = v.to(DEVICE, non_blocking=True)
                        if _recluster_sticks and "main_x" in vt:
                            xy = torch.stack([vt["main_x"], vt["main_y"]], dim=-1)
                            dists = torch.cdist(xy.reshape(-1, 2), _stick_centers)
                            vt["main_cluster"] = dists.argmin(dim=-1).reshape(xy.shape[:-1])
                        with autocast("cuda", dtype=AMP_DTYPE):
                            vpreds = model(vs)
                        vm, vtl = compute_loss(vpreds, vt,
                                               btn_loss_type=cfg.btn_loss, plain_ce=plain_ce)
                        batch_total = sum(t.item() for t in vtl)
                        if math.isfinite(batch_total):
                            val_nsi_sums["total"] += batch_total
                            for _vk in _VAL_KEYS:
                                if _vk != "total":
                                    val_nsi_sums[_vk] += vm[_vk]
                            val_nsi_ct += 1
                        if val_nsi_ct >= MAX_VAL_BATCHES:
                            break
                if val_nsi_ct > 0:
                    val_nsi_avg = {f"val_nsi/{k}": val_nsi_sums[k] / val_nsi_ct for k in _VAL_KEYS}
                    wandb.log(val_nsi_avg, step=step)
                    _log(f"  -- val_nsi bf1={val_nsi_avg['val_nsi/btn_f1']:.1%} "
                         f"mf1={val_nsi_avg['val_nsi/main_f1']:.1%} "
                         f"cf1={val_nsi_avg['val_nsi/cdir_f1']:.1%}")
                _raw_model.encoder.si_drop_prob = _saved_si_prob

        # Checkpoint
        if step % ckpt_interval == 0 or step == max_steps:
            if _is_main:
                os.makedirs("checkpoints", exist_ok=True)
                ckpt_path = f"checkpoints/{run_name}_step{step:06d}.pt"
                ckpt_data = {
                    "global_step":          step,
                    "model_state_dict":     _raw_model.state_dict(),
                    "optimizer_state_dict": optimiser.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss_weights":         loss_weights.cpu(),
                    "norm_stats":           ds.norm_stats,
                    "config":               cfg.__dict__,
                }
                if stick_centers_np is not None:
                    ckpt_data["stick_centers"] = stick_centers_np.tolist()
                if shoulder_centers_np is not None:
                    ckpt_data["shoulder_centers"] = shoulder_centers_np.tolist()
                if hal_controller_encoding and _combo_map:
                    ckpt_data["controller_combos"] = [list(c) for c in sorted(_combo_map.keys(), key=_combo_map.get)]
                torch.save(ckpt_data, ckpt_path)
                _log(f"  -- saved {ckpt_path}")
            if is_distributed:

                dist.barrier()

    elapsed = time.time() - t0
    skip_msg = f"  skipped={skip_ct}" if skip_ct else ""
    epoch_frac = step / max(est_batches, 1)
    _log(f"\nDone. {step:,}/{max_steps:,} steps ({epoch_frac:.2f} epochs) in {elapsed:.0f}s "
         f"({step/elapsed:.1f} step/s){skip_msg}")
    if target_val_f1 and not _target_hit:
        _log(f"TARGET NOT REACHED: best val_f1={best_val_f1:.4f} (target={target_val_f1:.4f})")
    wandb.finish()
    if is_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MIMIC training")
    parser.add_argument("--epochs",     type=int, default=None,
                        help="Number of epochs (default: 1 if --max-steps not set)")
    parser.add_argument("--max-steps",  type=int, default=None,
                        help="Override: train for exactly this many steps")
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES,
                        help="Train on exactly this many samples (computes steps from batch size, default: 2M)")
    parser.add_argument("--data-dir",   type=str, default=DATA_DIR)
    parser.add_argument("--debug",      action="store_true")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile (debug)")
    parser.add_argument("--resume",     type=str, default=None,
                        help="Path to checkpoint (.pt) to resume from")
    parser.add_argument("--model",      type=str, default="medium",
                        choices=list(MODEL_PRESETS.keys()),
                        help="Model size preset (default: medium ~32M params)")
    parser.add_argument("--lr",         type=float, default=None,
                        help="Learning rate override (default: 5e-5)")
    parser.add_argument("--run-name",   type=str, default=None,
                        help="Wandb run name (auto-generated if not set)")
    parser.add_argument("--wandb-tags", type=str, default=None,
                        help="Comma-separated wandb tags (e.g. 'lr-sweep,small')")
    parser.add_argument("--wandb-group", type=str, default=None,
                        help="Wandb group for related runs (e.g. 'sweep-v1')")
    parser.add_argument("--num-layers", type=int, default=None,
                        help="Override number of transformer layers")
    parser.add_argument("--seq-len",    type=int, default=None,
                        help="Override sequence length (default: 60)")
    parser.add_argument("--reaction-delay", type=int, default=None,
                        help="Reaction delay in frames (default: 1, HAL uses 0)")
    parser.add_argument("--controller-offset", action="store_true",
                        help="Shift self-controller input by -1 frame (HAL-style: see prev frame's press)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size (default: 200)")
    parser.add_argument("--encoder",   type=str, default="hybrid16",
                        choices=["default", "flat", "composite8", "hybrid16", "hal_flat"],
                        help="Frame encoder variant (default: hybrid16)")
    parser.add_argument("--d-intra",   type=int, default=None,
                        help="Intra-frame encoder width (default: 256)")
    parser.add_argument("--dropout",   type=float, default=None,
                        help="Dropout rate (default: 0.1)")
    parser.add_argument("--k-query",   type=int, default=None,
                        help="Number of query tokens in intra-frame attention")
    parser.add_argument("--intra-layers", type=int, default=None,
                        help="Number of intra-frame attention layers (0=no attn)")
    parser.add_argument("--scaled-emb", action="store_true",
                        help="Scale embedding dim by vocab size (emb = max(16, card^0.25*16))")
    parser.add_argument("--pos-enc",   type=str, default=None,
                        choices=["learned", "rope", "sinusoidal", "alibi"],
                        help="Positional encoding type")
    parser.add_argument("--attn-variant", type=str, default=None,
                        choices=["standard", "sliding", "gqa"],
                        help="Attention variant")
    parser.add_argument("--n-kv-heads", type=int, default=None,
                        help="Number of KV heads for GQA (default: nhead)")
    parser.add_argument("--clusters-path", type=str, default="data/full/stick_clusters.json",
                        help="Path to stick_clusters.json")
    parser.add_argument("--btn-loss", type=str, default=None,
                        choices=["bce", "focal"],
                        help="Button loss type (default: focal)")
    parser.add_argument("--opp-inputs", action="store_true",
                        help="Include opponent controller inputs (default: excluded)")
    parser.add_argument("--self-inputs", action="store_true",
                        help="Include self controller inputs (default: excluded)")
    parser.add_argument("--weight-decay", type=float, default=None,
                        help="Weight decay (default: 1e-2)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--focal-gamma", type=float, default=FOCAL_GAMMA,
                        help="Focal CE gamma for sticks/shoulders/cdir (default: 2.0)")
    parser.add_argument("--btn-focal-gamma", type=float, default=None,
                        help="Focal BCE gamma for buttons (default: same as --focal-gamma)")
    parser.add_argument("--label-smoothing", type=float, default=LABEL_SMOOTHING,
                        help="Label smoothing (default: 0.1, set 0 to disable)")
    parser.add_argument("--target-val-f1", type=float, default=None,
                        help="Stop training when val btn_f1 >= this value (e.g. 0.985)")
    parser.add_argument("--max-wall-time", type=float, default=None,
                        help="Hard wall-time limit in seconds; stop if exceeded")
    parser.add_argument("--val-frac", type=float, default=None,
                        help="Override validation frequency as fraction of max_steps (default: 0.05)")
    parser.add_argument("--warmup-steps", type=int, default=None,
                        help="Override warmup steps (default: 1%% of max_steps)")
    parser.add_argument("--scale-lr", action="store_true",
                        help="Scale LR by world_size for DDP (linear scaling rule)")
    parser.add_argument("--grad-accum-steps", type=int, default=1,
                        help="Gradient accumulation steps (default: 1, eff_batch = bs * gpus * accum)")
    parser.add_argument("--nccl-timeout", type=int, default=1800,
                        help="NCCL timeout in seconds (default: 1800, use 3600 for large models)")
    parser.add_argument("--grad-clip-norm", type=float, default=None,
                        help="Gradient clipping norm (default: 1.0)")
    parser.add_argument("--no-load-optim", action="store_true",
                        help="Skip loading optimizer/scheduler state on resume (fresh LR)")
    parser.add_argument("--plain-ce", action="store_true",
                        help="Use plain cross-entropy instead of focal loss for all heads")
    parser.add_argument("--n-heads", type=int, default=None,
                        help="Override number of attention heads")
    parser.add_argument("--lean-features", action="store_true",
                        help="Drop nana/projectile features to match HAL's lean input set")
    parser.add_argument("--hal-mode", action="store_true",
                        help="HAL-exact mode: single-label buttons, combined shoulder, LN heads, plain CE")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable AMP (train in FP32)")
    parser.add_argument("--cosine-min-lr", type=float, default=None,
                        help="Minimum LR for cosine schedule (default: 0, HAL uses 1e-6)")
    parser.add_argument("--no-warmup", action="store_true",
                        help="Disable LR warmup (start at full LR)")
    parser.add_argument("--stick-clusters", type=str, default=None,
                        help="Stick cluster set: 'hal37' for HAL's 37 hand-designed clusters")
    parser.add_argument("--hal-minimal-features", action="store_true",
                        help="Use HAL's minimal input set (drop ECB, speeds, hitlag/hitstun from numeric)")
    parser.add_argument("--hal-controller-encoding", action="store_true",
                        help="Use HAL-style one-hot controller feedback encoding (requires controller_combos.json)")
    parser.add_argument("--si-drop-start", type=float, default=None,
                        help="Fraction of training where SI dropout begins")
    parser.add_argument("--si-drop-end", type=float, default=None,
                        help="Fraction of training where SI dropout reaches max")
    parser.add_argument("--si-drop-max", type=float, default=1.0,
                        help="Max SI drop probability (default: 1.0)")
    args = parser.parse_args()

    import train as _self_module
    _self_module._focal_gamma = args.focal_gamma
    _self_module._label_smoothing = args.label_smoothing
    _self_module._btn_focal_gamma = args.btn_focal_gamma if args.btn_focal_gamma is not None else args.focal_gamma

    if args.seed is not None:
        import random
        import numpy as np
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    tags = [t.strip() for t in args.wandb_tags.split(",")] if args.wandb_tags else None
    train(
        epochs=args.epochs,
        max_steps=args.max_steps,
        max_samples=args.max_samples,
        data_dir=args.data_dir,
        debug=args.debug or bool(os.getenv("DEBUG", "")),
        resume=args.resume,
        compile_model=not args.no_compile,
        model_preset=args.model,
        lr=args.lr,
        run_name=args.run_name,
        wandb_tags=tags,
        wandb_group=args.wandb_group,
        num_layers_override=args.num_layers,
        seq_len_override=args.seq_len,
        batch_size_override=args.batch_size,
        encoder_type=args.encoder,
        d_intra=args.d_intra,
        dropout_override=args.dropout,
        k_query=args.k_query,
        intra_layers=args.intra_layers,
        scaled_emb=args.scaled_emb,
        pos_enc=args.pos_enc,
        attn_variant=args.attn_variant,
        n_kv_heads=args.n_kv_heads,
        btn_loss=args.btn_loss,
        no_opp_inputs=not args.opp_inputs,
        no_self_inputs=not args.self_inputs,
        clusters_path=args.clusters_path,
        target_val_f1=args.target_val_f1,
        max_wall_time=args.max_wall_time,
        val_frac_override=args.val_frac,
        warmup_steps_override=args.warmup_steps,
        weight_decay_override=args.weight_decay,
        scale_lr=args.scale_lr,
        grad_accum_steps=args.grad_accum_steps,
        nccl_timeout=args.nccl_timeout,
        grad_clip_norm=args.grad_clip_norm,
        no_load_optim=args.no_load_optim,
        si_drop_start=args.si_drop_start,
        si_drop_end=args.si_drop_end,
        si_drop_max=args.si_drop_max,
        plain_ce=args.plain_ce,
        n_heads_override=args.n_heads,
        hal_mode=args.hal_mode,
        no_amp=args.no_amp,
        cosine_min_lr=args.cosine_min_lr,
        lean_features=args.lean_features,
        no_warmup=args.no_warmup,
        stick_clusters=args.stick_clusters,
        hal_minimal_features=args.hal_minimal_features,
        reaction_delay_override=args.reaction_delay,
        controller_offset=args.controller_offset,
        hal_controller_encoding=args.hal_controller_encoding,
    )
