#!/usr/bin/env python3
# train.py  --  MIMIC training with BF16 AMP, torch.compile, epoch/step-based loop
#
# NOTE: overfit_log*.txt files in this repo are from intentional overfit runs
# on a small fsmash-only subset.  Near-zero main/l/r/btn losses in those logs
# are expected -- the model learned the dominant neutral/no-press pattern on
# that narrow data.  They do NOT indicate a bug.

import sys
import os
# Load .env so WANDB_API_KEY / HF_TOKEN / HUGGING_FACE_HUB_TOKEN flow through
# to wandb.init() and huggingface_hub without a separate login step.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass
import math
import time
import argparse
from collections import deque
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
LEARNING_RATE   = 3e-4
WEIGHT_DECAY    = 1e-2
NUM_WORKERS     = int(os.environ.get("MIMIC_NUM_WORKERS", "8"))
SEQUENCE_LENGTH = 180
REACTION_DELAY  = 0     # v2 shards bake in the next-frame target; 1 only for legacy leaked shards
DATA_DIR        = "./data/full"
MAX_SAMPLES     = 2_000_000

GRAD_CLIP_NORM      = 1.0
TASK_NAMES          = ["main", "l", "r", "cdir", "btn"]

AMP_DTYPE = torch.bfloat16

MAX_VAL_BATCHES    = 100

# Intervals derived from max_steps at runtime (see _compute_intervals)
WARMUP_FRAC        = 0.05     # 5% warmup
LOG_FRAC           = 0.001    # log every ~0.1%
VAL_FRAC           = 0.025    # validate every ~2.5% (was 1% before 2026-10-22; --val-frac 0.01)
CKPT_FRAC          = 0.05     # checkpoint every ~5%

# -----------------------------------------------------------------------------
# 1b. Speed settings
# -----------------------------------------------------------------------------
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
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


# Module-level state for combo encoding (set by train() when --mimic-controller-encoding)
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


def _cond_head_classes(targets, shoulder_centers, n_cdir):
    """(shoulder, c_dir, main) target classes, exactly as compute_loss derives its CE
    targets: what --cond-heads conditions the later heads on in training."""
    l_bin, r_bin = targets["l_bin"].long(), targets["r_bin"].long()
    shldr = (_combine_shoulder_targets(l_bin, r_bin, shoulder_centers)
             if shoulder_centers is not None else l_bin)
    cdir_tgt = targets["c_dir"].long()
    cdir = cdir_tgt.argmax(dim=-1)
    if n_cdir == 9 and cdir_tgt.size(-1) == 5:
        cdir = _CDIR_5_TO_9.to(cdir.device)[cdir]
    return (shldr, cdir, targets["main_cluster"].long())


def compute_loss(preds, targets, shoulder_centers=None, n_cdir=5, lite=False):
    """The mimic loss: plain CE on all heads, single-label buttons, combined shoulder."""
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

    # C-dir: if targets are already 9-class (from new shards), use directly; else remap 5→9
    cdir_classes = cdir_tgt.argmax(dim=-1)
    if n_cdir == 9 and cdir_tgt.size(-1) == 5:
        cdir_classes = _CDIR_5_TO_9.to(cdir_classes.device)[cdir_classes]
    loss_cdir = F.cross_entropy(c_logits.reshape(-1, c_logits.size(-1)), cdir_classes.reshape(-1))

    # Single-label buttons: use pre-computed early_release labels if available
    btn_single_pre = targets.get("btns_single")
    if btn_single_pre is not None:
        btn_single = btn_single_pre.long()
    elif _combo_map is not None:
        btn_single = _multi_hot_to_combo_label(btn_tgt)
    else:
        raise ValueError("no btns_single targets and no controller_combos.json combo map")
    loss_btn = F.cross_entropy(btn_pred.reshape(-1, btn_pred.size(-1)), btn_single.reshape(-1))
    task_losses = (loss_main, loss_shldr, torch.tensor(0.0, device=loss_main.device), loss_cdir, loss_btn)
    if lite:
        # --lean-metrics on a non-log step: only the loss terms, still on the GPU (no syncs).
        metrics = {"loss_main": loss_main.detach(), "loss_l": loss_shldr.detach(), "loss_r": 0.0,
                   "loss_cdir": loss_cdir.detach(), "loss_btn": loss_btn.detach()}
        return metrics, task_losses

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
                  rank=0, world_size=1,
                  hal_controller_encoding=False,
                  controller_combo_map=None, n_controller_combos=5,
                  character_filter=None):
    _log(f"  Using streaming dataset from {data_dir}")
    if character_filter is not None:
        _log(f"  Character filter: self_character={character_filter}")
    ctrl_kw = dict(
        hal_controller_encoding=hal_controller_encoding,
        controller_combo_map=controller_combo_map,
        n_controller_combos=n_controller_combos,
    )
    # Keys the HAL encoder actually reads — skip slicing/collating the rest
    _HAL_STATE_KEYS = [
        "stage", "self_character", "opp_character", "self_action", "opp_action",
        "self_numeric", "opp_numeric", "self_flags", "opp_flags", "self_controller",
        "self_port", "opp_port",
    ]
    train_ds = StreamingMeleeDataset(
        data_dir=data_dir,
        sequence_length=SEQUENCE_LENGTH,
        reaction_delay=REACTION_DELAY,
        split="train",
        rank=rank,
        world_size=world_size,
        character_filter=character_filter,
        **ctrl_kw,
    )
    train_ds._state_keys = _HAL_STATE_KEYS
    val_ds = StreamingMeleeDataset(
        data_dir=data_dir,
        sequence_length=SEQUENCE_LENGTH,
        reaction_delay=REACTION_DELAY,
        split="val",
        rank=rank,
        world_size=world_size,
        distributed=False,  # all ranks see all val data — prevents DDP deadlock
        character_filter=character_filter,
        **ctrl_kw,
    )
    val_ds._state_keys = _HAL_STATE_KEYS
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
        prefetch_factor=4 if nw > 0 else None,
    )

def get_model(compile_model=True, model_preset=None, num_layers_override=None,
              encoder_type=None, dropout_override=None,
              no_opp_inputs=True, no_self_inputs=True,
              n_stick_clusters=None,
              n_heads_override=None,
              hal_minimal_features=False,
              hal_controller_encoding=False, n_controller_combos=5,
              use_input_gate=False, cond_heads=False, relpos_impl="fp32", attn_window=0):
    overrides = MODEL_PRESETS.get(model_preset, {}) if model_preset else {}
    if num_layers_override:
        overrides["num_layers"] = num_layers_override
    if encoder_type:
        overrides["encoder_type"] = encoder_type
    if dropout_override is not None:
        overrides["dropout"] = dropout_override
    overrides["no_opp_inputs"] = no_opp_inputs
    overrides["no_self_inputs"] = no_self_inputs
    if n_heads_override:
        overrides["nhead"] = n_heads_override
    overrides["mimic_mode"] = True   # the only mode
    if hal_minimal_features:
        overrides["hal_minimal_features"] = True
    if hal_controller_encoding:
        overrides["hal_controller_encoding"] = True
        overrides["n_controller_combos"] = n_controller_combos
    if n_stick_clusters is not None:
        overrides["n_stick_clusters"] = n_stick_clusters
    if use_input_gate:
        overrides["use_input_gate"] = True
    if cond_heads:
        overrides["cond_heads"] = True
    if relpos_impl != "fp32":
        overrides["relpos_impl"] = relpos_impl
    if attn_window:
        overrides["attn_window"] = attn_window
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


_ARCH_FLAGS = {"cond_heads": "--cond-heads / --no-cond-heads", "attn_window": "--attn-window N"}


def _arch_mismatch(ck_cfg, cfg):
    """Architecture fields that differ between a checkpoint's config and this run's (same parameter
    shapes, so load_state_dict alone wouldn't notice). None/False/0 all mean off."""
    ck_cfg = ck_cfg or {}
    return [(k, ck_cfg.get(k) or 0, getattr(cfg, k, 0) or 0) for k in _ARCH_FLAGS
            if (ck_cfg.get(k) or 0) != (getattr(cfg, k, 0) or 0)]


def _arch_error(path, diffs, what):
    return ValueError(f"{path} differs from this run's flags ({what}): " + "; ".join(
        f"{k}: checkpoint {a!r} vs run {b!r} ({_ARCH_FLAGS[k]})" for k, a, b in diffs)
        + ". Pass matching flags (cond-heads became the default on 2026-10-22).")


def train(epochs: int = None, max_steps: int = None, max_samples: int = MAX_SAMPLES,
          data_dir: str = DATA_DIR,
          debug: bool = False, resume: str = None, compile_model: bool = True,
          model_preset: str = None, lr: float = None, run_name: str = None,
          wandb_tags: list = None, wandb_group: str = None,
          num_layers_override: int = None, seq_len_override: int = None,
          batch_size_override: int = None,
          encoder_type: str = None,
          dropout_override: float = None,
          no_opp_inputs: bool = True,
          no_self_inputs: bool = False,
          clusters_path: str = None,
          max_wall_time: float = None,
          val_frac_override: float = None,
          warmup_steps_override: int = None,
          weight_decay_override: float = None,
          grad_accum_steps: int = 1,
          nccl_timeout: int = 1800,
          grad_clip_norm: float = None,
          no_load_optim: bool = False,
          warm_restart: bool = False,
          eval_only: str = None,
          n_heads_override: int = None,
          no_amp: bool = False,
          cosine_min_lr: float = None,
          no_warmup: bool = False,
          stick_clusters: str = None,
          hal_minimal_features: bool = False,
          reaction_delay_override: int = None,
          hal_controller_encoding: bool = False,
          character_filter: int = None,
          input_gate_l1: float = 0.0,
          cond_heads: bool = True,
          relpos_impl: str = "fp32",
          attn_window: int = 0,
          distill_from: str = None,
          distill_alpha: float = 1.0,
          lean_metrics: bool = False,
          optimizer: str = "muon",
          muon_lr: float = 0.02,
          save_steps: str = None):
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
    elif model_preset and (
        model_preset in MODEL_PRESETS
        and MODEL_PRESETS[model_preset].get("max_seq_len") is not None
    ):
        SEQUENCE_LENGTH = MODEL_PRESETS[model_preset]["max_seq_len"]
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
            combos, _combo_map, _n_combos = load_controller_combos(data_dir)
            _log(f"  Controller encoding: {_n_combos} button combos from {combos_path}")
        else:
            raise RuntimeError(
                f"--mimic-controller-encoding requires controller_combos.json in {data_dir}. "
                f"Copy from an existing HAL-normalized data dir (e.g. data/fox_hal_local/).")

    _log(f"Loading dataset from {data_dir} ...")
    ds, val_ds = get_datasets(data_dir, no_opp_inputs=no_opp_inputs,
                              no_self_inputs=no_self_inputs,
                              rank=rank, world_size=world_size,
                              hal_controller_encoding=hal_controller_encoding,
                              controller_combo_map=_combo_map,
                              n_controller_combos=_n_combos,
                              character_filter=character_filter)
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
        eff_bs = BATCH_SIZE * world_size * grad_accum_steps
        max_steps = max_samples // eff_bs
        _log(f"  {max_samples:,} samples / eff_bs {eff_bs} ({BATCH_SIZE}×{world_size}gpu×{grad_accum_steps}accum) = {max_steps:,} steps")
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
    if hal_minimal_features:
        # Force 3-class shoulder output, but keep data's centers for decoding
        _shoulder_centers_for_decode = torch.tensor(shoulder_centers_np, dtype=torch.float32, device=DEVICE)
        n_shoulder_bins = 3
    else:
        _shoulder_centers_for_decode = None
        n_shoulder_bins = len(shoulder_centers_np)
    def _cc(t):   # --cond-heads: true earlier-head classes to condition on (None otherwise)
        if not cond_heads:
            return None
        sc = _shoulder_centers_for_decode if _shoulder_centers_for_decode is not None else _shoulder_centers
        return _cond_head_classes(t, sc, cfg.num_c_dirs)
    _shoulder_centers = torch.tensor(shoulder_centers_np if not hal_minimal_features else [0.0, 0.4, 1.0],
                                     dtype=torch.float32, device=DEVICE)
    _stick_centers = torch.tensor(stick_centers_np, dtype=torch.float32, device=DEVICE)
    _recluster_sticks = stick_clusters is not None  # re-assign if using non-data clusters
    _log(f"  Clusters: {n_stick_clusters} stick, {n_shoulder_bins} shoulder"
         f"{'  (re-clustering at runtime)' if _recluster_sticks else ''}")

    model, cfg = get_model(compile_model=compile_model, model_preset=model_preset,
                           num_layers_override=num_layers_override,
                           encoder_type=encoder_type,
                           dropout_override=dropout_override,
                           no_opp_inputs=no_opp_inputs,
                           no_self_inputs=no_self_inputs,
                           n_stick_clusters=n_stick_clusters,
                           n_heads_override=n_heads_override,
                           hal_minimal_features=hal_minimal_features,
                           hal_controller_encoding=hal_controller_encoding,
                           n_controller_combos=_n_combos,
                           use_input_gate=(input_gate_l1 > 0),
                           cond_heads=cond_heads, relpos_impl=relpos_impl,
                           attn_window=attn_window)
    n_params = sum(p.numel() for p in model.parameters())
    _log(f"  Model: {n_params:,} params on {DEVICE}  (AMP={AMP_DTYPE}, LR={actual_lr})")
    _log(f"  Intervals: log={log_interval}  val={val_interval}  "
         f"ckpt={ckpt_interval}  warmup={warmup_steps}")

    loss_weights = torch.ones(len(TASK_NAMES), device=DEVICE)

    if optimizer == "adamw":
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
    elif optimizer == "muon":
        # Muon on the hidden 2D weight matrices; aux-AdamW on embeddings, the
        # relpos tables, the output heads, and all norms/biases (Muon convention).
        # SingleDeviceMuonWithAuxAdam is one optimizer with standard param_groups,
        # so the existing LR scheduler / step / clip / save paths are untouched.
        from mimic.muon import SingleDeviceMuonWithAuxAdam
        _emb_names = set()
        for _mn, _mod in model.named_modules():
            if isinstance(_mod, torch.nn.Embedding):
                for _pn, _ in _mod.named_parameters(recurse=False):
                    _emb_names.add(f"{_mn}.{_pn}" if _mn else _pn)

        def _is_matrix(n, p):
            # Names are `_orig_mod.`-prefixed when torch.compile is on; strip it
            # so the heads/embeds/relpos exclusions match in both cases.
            base = n[len("_orig_mod."):] if n.startswith("_orig_mod.") else n
            return (p.ndim == 2 and n not in _emb_names
                    and not base.endswith(".Er") and not base.endswith(".queries")
                    and not base.startswith("heads."))

        muon_p, decay, no_decay = [], [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if _is_matrix(n, p):
                muon_p.append(p)
            elif n.endswith("bias") or "norm" in n.lower():
                no_decay.append(p)
            else:
                decay.append(p)
        # Muon group: weight_decay=0 (canonical — the orthogonalized, spectral-
        # norm-bounded update makes heavy decay redundant; LR-coupling would make
        # WEIGHT_DECAY ~60x stronger here than on the AdamW group).
        optimiser = SingleDeviceMuonWithAuxAdam([
            dict(params=muon_p, use_muon=True, lr=muon_lr, momentum=0.95,
                 weight_decay=0.0),
            dict(params=decay, use_muon=False, lr=actual_lr, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=WEIGHT_DECAY),
            dict(params=no_decay, use_muon=False, lr=actual_lr, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0.0),
        ])
        _log(f"  Muon: {len(muon_p)} hidden matrices @ lr={muon_lr}; "
             f"aux-AdamW: {len(decay)+len(no_decay)} tensors @ lr={actual_lr}")
    else:
        raise ValueError(f"unknown --optimizer {optimizer!r}")

    _eta_min = cosine_min_lr if cosine_min_lr is not None else 1e-6

    def _build_cosine(n):
        return optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(n, 1), eta_min=_eta_min)

    def _make_scheduler():   # optional linear warmup, then cosine to eta_min at max_steps
        if warmup_steps > 0:
            warmup  = optim.lr_scheduler.LinearLR(
                optimiser, start_factor=0.01, total_iters=warmup_steps)
            cosine  = _build_cosine(max(max_steps - warmup_steps, 1))
            return optim.lr_scheduler.SequentialLR(
                optimiser, [warmup, cosine], milestones=[warmup_steps])
        return _build_cosine(max_steps)
    scheduler = _make_scheduler()

    start_step = 0
    if resume:
        ckpt = torch.load(resume, map_location=DEVICE)
        _diffs = _arch_mismatch(ckpt.get("config"), cfg)
        if _diffs and not warm_restart:   # warm starts may switch architecture on purpose
            raise _arch_error(resume, _diffs, "resume; use --warm-restart to start a new run from "
                                              "these weights with a different architecture")
        model.load_state_dict(ckpt['model_state_dict'])
        if not no_load_optim:
            _base_lrs = [g['lr'] for g in optimiser.param_groups]   # this run's per-group base LRs
            try:
                optimiser.load_state_dict(ckpt['optimizer_state_dict'])
            except ValueError as e:
                raise ValueError(
                    f"optimizer state in {resume} doesn't match --optimizer {optimizer}. Runs from "
                    f"before 2026-10-22 used AdamW: pass --optimizer adamw to resume them (or "
                    f"--no-load-optim to start a fresh optimizer).") from e
            # warm-restart: keep Adam momentum but reset the LR schedule (fresh
            # cosine over the full new horizon from step 0). CosineAnnealingLR
            # updates recurrently from the optimizer's CURRENT lr, which
            # load_state_dict just set to the checkpoint's (low, end-of-cosine)
            # value — so we must reset lr back to the base AND rebuild the
            # scheduler, else the "fresh" cosine continues from ~eta_min.
            if warm_restart:   # per group: Muon's matrices keep --muon-lr, aux groups --lr
                for g, lr0 in zip(optimiser.param_groups, _base_lrs):
                    g['lr'] = lr0
                    g['initial_lr'] = lr0
                scheduler = _make_scheduler()
            elif ckpt.get("scheduler_state_dict"):
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_step = 0 if warm_restart else ckpt.get('global_step', 0)
        _log(f"Resumed from {resume}, starting at step {start_step}"
             + (" (warm restart: Adam kept, LR schedule reset)" if warm_restart else "")
             + (f" (fresh optimizer, lr={actual_lr})" if no_load_optim else ""))

    # --- DDP wrapping (after compile, optimizer, and checkpoint load) ---
    if is_distributed:
        model = DDP(model, device_ids=[local_rank])
    _raw_model = model.module if is_distributed else model

    # --- Distillation (--distill-from): a frozen teacher's per-head distributions as soft
    # targets; loss += alpha * sum over heads of KL(teacher || student), on every position. ---
    _teacher = None
    if distill_from:
        import dataclasses
        _tck = torch.load(distill_from, map_location="cpu", weights_only=False)
        _tf = {f.name for f in dataclasses.fields(ModelConfig)}
        _teacher = FramePredictor(ModelConfig(**{k: v for k, v in _tck["config"].items() if k in _tf})).to(DEVICE)
        _teacher.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in _tck["model_state_dict"].items()})
        _teacher.eval().requires_grad_(False)
        _teacher_cond = bool(getattr(_teacher.cfg, "cond_heads", False))
        if _teacher_cond != bool(cond_heads):
            _log(f"  WARNING: teacher cond_heads={_teacher_cond}, student cond_heads={bool(cond_heads)}: "
                 f"the per-head KL compares differently conditioned distributions")
        if compile_model:
            _teacher = torch.compile(_teacher)
        del _tck
        _log(f"  Distilling from {distill_from} (alpha={distill_alpha})")



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
    _run_key_ct = {}
    # Rolling window of train total loss for val/train ratio. 100 steps of
    # smoothing is enough to damp batch noise without lagging behind the
    # current training regime — short enough that the ratio still moves
    # when val/train diverge (overfitting signal).
    _train_loss_window = deque(maxlen=100)

    if start_step and not isinstance(ds, IterableDataset):
        for _ in range(start_step):
            next(loader)

    _oom_retries = 0
    best_val_f1 = -1.0
    best_val_loss = float("inf")
    best_val_loss_step = 0
    evals_since_improve = 0
    _log(f"\n=== Training {max_steps} steps ===")
    # --- Eval-only: load a checkpoint, score it with the EXACT training val loop
    # (same compute_loss / flags / MAX_VAL_BATCHES), print val/total, and exit. ---
    if eval_only:
        _ec = torch.load(eval_only, map_location=DEVICE, weights_only=False)
        if isinstance(_ec, dict) and "config" in _ec:
            _diffs = _arch_mismatch(_ec["config"], cfg)
            if _diffs:   # the model is built from the CLI flags: a mismatch would score the wrong arch
                raise _arch_error(eval_only, _diffs, "eval-only")
        _sd = _ec["model_state_dict"] if isinstance(_ec, dict) and "model_state_dict" in _ec else _ec
        _msd = _raw_model.state_dict()
        _want = any(k.startswith("_orig_mod.") for k in _msd)
        _have = any(k.startswith("_orig_mod.") for k in _sd)
        if _want and not _have:
            _sd = {f"_orig_mod.{k}": v for k, v in _sd.items()}
        elif _have and not _want:
            _sd = {k[len("_orig_mod."):]: v for k, v in _sd.items()}
        _raw_model.load_state_dict(_sd)
        model.eval()
        _EK = ["total", "main_f1", "btn_f1", "shldr_f1", "cdir_f1"]
        _s = {k: 0.0 for k in _EK}; _c = 0
        with torch.no_grad():
            for vs, vt in val_dl:
                for k, v in vs.items(): vs[k] = v.to(DEVICE, non_blocking=True)
                for k, v in vt.items(): vt[k] = v.to(DEVICE, non_blocking=True)
                if _recluster_sticks and "main_x" in vt:
                    xy = torch.stack([vt["main_x"], vt["main_y"]], dim=-1)
                    dists = torch.cdist(xy.reshape(-1, 2), _stick_centers)
                    vt["main_cluster"] = dists.argmin(dim=-1).reshape(xy.shape[:-1])
                with autocast("cuda", dtype=AMP_DTYPE):
                    vpreds = _raw_model(vs, cond_classes=_cc(vt))
                vm, vtl = compute_loss(vpreds, vt,
                                       shoulder_centers=_shoulder_centers_for_decode if _shoulder_centers_for_decode is not None else _shoulder_centers,
                                       n_cdir=cfg.num_c_dirs)
                _bt = sum(t.item() for t in vtl)
                if math.isfinite(_bt):
                    _s["total"] += _bt
                    for _k in _EK:
                        if _k != "total": _s[_k] += vm[_k]
                    _c += 1
                if _c >= MAX_VAL_BATCHES: break
        _res = {k: _s[k] / max(_c, 1) for k in _EK}
        print(f"EVAL_ONLY ckpt={eval_only} val/total={_res['total']:.4f} "
              f"main_f1={_res['main_f1']:.4f} btn_f1={_res['btn_f1']:.4f} batches={_c}", flush=True)
        _log(f"EVAL_ONLY {os.path.basename(eval_only)}: val/total={_res['total']:.4f} (batches={_c})")
        return

    # --save-steps: weights-only snapshots at the listed optimizer steps (0 = the initial weights),
    # named _snap so the _step* resume/averaging globs never pick them up.
    _save_steps = {int(x) for x in save_steps.split(",") if x.strip()} if save_steps else set()

    def _save_snapshot(step):
        if not _is_main:
            return
        os.makedirs("checkpoints", exist_ok=True)
        _p = f"checkpoints/{run_name}_snap{step:06d}.pt"
        c = {
            "global_step":      step,
            "model_state_dict": _raw_model.state_dict(),
            "loss_weights":     loss_weights.cpu(),
            "norm_stats":       ds.norm_stats,
            "config":           {**cfg.__dict__, "model_preset": model_preset, "run_name": run_name},
        }
        if stick_centers_np is not None:
            c["stick_centers"] = stick_centers_np.tolist()
        if shoulder_centers_np is not None:
            c["shoulder_centers"] = shoulder_centers_np.tolist()
        if hal_controller_encoding and _combo_map:
            c["controller_combos"] = [list(c_) for c_ in sorted(_combo_map.keys(), key=_combo_map.get)]
        torch.save(c, _p)
        _log(f"  -- snapshot {_p}")

    if start_step == 0 and 0 in _save_steps:
        _save_snapshot(0)

    for step in range(start_step + 1, max_steps + 1):
        if max_wall_time and (time.time() - t0) > max_wall_time:
            _log(f"WALL TIME LIMIT ({max_wall_time}s) exceeded at step {step}")
            break

        model.train()
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

                with autocast("cuda", dtype=AMP_DTYPE):
                    preds = model(state, cond_classes=_cc(target))
                metrics, task_losses = compute_loss(
                    preds, target,
                    lite=lean_metrics and not (step % log_interval == 0 or step == 1),
                    shoulder_centers=_shoulder_centers_for_decode if _shoulder_centers_for_decode is not None else _shoulder_centers,
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
            if _teacher is not None:
                with torch.no_grad(), autocast("cuda", dtype=AMP_DTYPE):
                    # a cond teacher conditions on the true earlier-head classes (not its own samples)
                    t_preds = _teacher(state, cond_classes=_cond_head_classes(
                        target, _shoulder_centers_for_decode if _shoulder_centers_for_decode is not None
                        else _shoulder_centers, cfg.num_c_dirs) if _teacher_cond else None)
                kd = sum(F.kl_div(F.log_softmax(preds[k].float(), -1), F.log_softmax(t_preds[k].float(), -1),
                                  log_target=True, reduction="none").sum(-1).mean()
                         for k in ("main_xy", "shoulder_val", "c_dir_logits", "btn_logits"))
                micro_loss = micro_loss + distill_alpha * kd / grad_accum_steps

            # L1 penalty on encoder input gate (learned feature importance).
            # Divide by grad_accum_steps so the effective penalty matches the
            # configured lambda regardless of accumulation.
            if input_gate_l1 > 0:
                micro_loss = micro_loss + (input_gate_l1 / grad_accum_steps) * \
                             _raw_model.encoder.gate_l1_penalty()

            if not torch.isfinite(micro_loss):
                skip_step = True
                skip_ct += 1
                _log(f"  [skip] step {step} -- non-finite loss ({skip_ct} total)")
                break

            with maybe_no_sync:
                micro_loss.backward()
            accum_loss += micro_loss.detach()

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

        total_loss_val = accum_loss.item() if isinstance(accum_loss, torch.Tensor) else accum_loss
        if math.isfinite(total_loss_val):
            _train_loss_window.append(total_loss_val)
        metrics = accum_metrics
        _run_sums["total"]     += total_loss_val
        _run_sums["grad_norm"] += grad_norm
        for _mk in _AVG_KEYS:
            if _mk not in ("total", "grad_norm") and _mk in metrics:
                _run_sums[_mk] += metrics[_mk]
                _run_key_ct[_mk] = _run_key_ct.get(_mk, 0) + 1
        _run_ct += 1

        if step % log_interval == 0 or step == 1:
            elapsed = time.time() - t0
            sps = step / elapsed

            if _run_ct > 0:
                # Per-key counts: under --lean-metrics the F1/accuracy keys are only computed on
                # log steps, the loss terms every step (summed on the GPU, read here).
                avg_log = {f"avg/{k}": float(_run_sums[k]) / (_run_ct if k in ("total", "grad_norm")
                                                               else max(1, _run_key_ct.get(k, 0)))
                           for k in _AVG_KEYS}
                avg_log["perf/step_per_sec"]    = sps
                avg_log["perf/samples_per_sec"] = sps * BATCH_SIZE * world_size * grad_accum_steps
                avg_log["train/lr"] = optimiser.param_groups[0]["lr"]
                if input_gate_l1 > 0 and hasattr(_raw_model.encoder, "input_gate_logits"):
                    g = torch.sigmoid(_raw_model.encoder.input_gate_logits).detach()
                    avg_log["gate/mean"]       = g.mean().item()
                    avg_log["gate/min"]        = g.min().item()
                    avg_log["gate/max"]        = g.max().item()
                    avg_log["gate/frac_below_0.1"] = (g < 0.1).float().mean().item()
                wandb.log(avg_log, step=step)
                _run_sums = {k: 0.0 for k in _AVG_KEYS}
                _run_ct = 0
                _run_key_ct = {}

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
                + f"({sps:.1f} step/s)",
            )

        # Validation
        if step % val_interval == 0 or step == max_steps:
            model.eval()
            _VAL_KEYS = ["total", "loss_main", "loss_l", "loss_r",
                         "loss_cdir", "loss_btn", "cdir_acc", "btn_acc",
                         "btn_f1", "btn_precision", "btn_recall",
                         "main_f1", "main_precision", "main_recall",
                         "shldr_f1", "shldr_precision", "shldr_recall",
                         "cdir_f1", "cdir_precision", "cdir_recall",
                         "cdir_active_acc",
                         "main_top1_acc", "shoulder_top1_acc"]

            # One val pass over up to MAX_VAL_BATCHES on _raw_model's current weights.
            def _val_pass():
                _sums = {k: 0.0 for k in _VAL_KEYS}
                _ct = 0
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
                            vpreds = _raw_model(vs, cond_classes=_cc(vt))  # unwrapped — no DDP collectives during eval
                        vm, vtl = compute_loss(vpreds, vt,
                                               shoulder_centers=_shoulder_centers_for_decode if _shoulder_centers_for_decode is not None else _shoulder_centers,
                                               n_cdir=cfg.num_c_dirs)
                        batch_total = sum(t.item() for t in vtl)
                        if math.isfinite(batch_total):
                            _sums["total"] += batch_total
                            for _vk in _VAL_KEYS:
                                if _vk != "total":
                                    _sums[_vk] += vm[_vk]
                            _ct += 1
                        if _ct >= MAX_VAL_BATCHES:
                            break
                return _sums, _ct

            def _build_best_ckpt():
                _cfg_save = {**cfg.__dict__, "model_preset": model_preset, "run_name": run_name}
                c = {
                    "global_step":          step,
                    "model_state_dict":     _raw_model.state_dict(),
                    "optimizer_state_dict": optimiser.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss_weights":         loss_weights.cpu(),
                    "norm_stats":           ds.norm_stats,
                    "config":               _cfg_save,
                }
                if stick_centers_np is not None:
                    c["stick_centers"] = stick_centers_np.tolist()
                if shoulder_centers_np is not None:
                    c["shoulder_centers"] = shoulder_centers_np.tolist()
                if hal_controller_encoding and _combo_map:
                    c["controller_combos"] = [list(c_) for c_ in sorted(_combo_map.keys(), key=_combo_map.get)]
                return c

            # --- Primary val on RAW (training) weights ---
            val_sums, val_ct = _val_pass()
            if val_ct > 0:
                val_avg = {f"val/{k}": val_sums[k] / val_ct for k in _VAL_KEYS}
                # Generalization gap: val_total / rolling_train_total.
                # > 1.0 is normal; trending up over time = overfitting.
                if _train_loss_window:
                    _train_rolling = sum(_train_loss_window) / len(_train_loss_window)
                    val_avg["train/rolling_total"] = _train_rolling
                    val_avg["val/train_ratio"] = val_avg["val/total"] / _train_rolling if _train_rolling > 0 else float("inf")
                else:
                    _train_rolling = float("nan")
                _cur_v = val_avg.get('val/total', float("inf"))
                _next_n = 0 if _cur_v < best_val_loss else evals_since_improve + 1
                val_avg["val/evals_since_improve"] = _next_n
                val_avg["val/best_val_loss_step"] = best_val_loss_step
                wandb.log(val_avg, step=step)
                _ratio_str = ""
                if _train_loss_window:
                    _ratio_str = f" train_roll={_train_rolling:.4f} v/t={val_avg['val/train_ratio']:.3f}"
                _log(f"  -- val total={val_avg['val/total']:.4f}  "
                     f"bf1={val_avg['val/btn_f1']:.1%} "
                     f"bP={val_avg['val/btn_precision']:.1%} bR={val_avg['val/btn_recall']:.1%}  "
                     f"mf1={val_avg['val/main_f1']:.1%} "
                     f"sf1={val_avg['val/shldr_f1']:.1%} "
                     f"cf1={val_avg['val/cdir_f1']:.1%}  "
                     f"(batches={val_ct})"
                     + _ratio_str)
                cur_val_f1 = val_avg.get('val/btn_f1', 0.0)
                cur_val_loss = val_avg.get('val/total', float("inf"))
                if cur_val_f1 > best_val_f1:
                    best_val_f1 = cur_val_f1
                    if _is_main:
                        os.makedirs("checkpoints", exist_ok=True)
                        best_path = f"checkpoints/{run_name}_best.pt"
                        torch.save(_build_best_ckpt(), best_path)
                        _log(f"  -- new best val f1={cur_val_f1:.1%} → {best_path}")
                if cur_val_loss < best_val_loss:
                    best_val_loss = cur_val_loss
                    best_val_loss_step = step
                    evals_since_improve = 0
                    if _is_main:
                        os.makedirs("checkpoints", exist_ok=True)
                        best_loss_path = f"checkpoints/{run_name}_bestloss.pt"
                        torch.save(_build_best_ckpt(), best_loss_path)
                        _log(f"  -- new best val loss={cur_val_loss:.4f} → {best_loss_path}")
                elif math.isfinite(cur_val_loss):
                    evals_since_improve += 1
            else:
                _log("  -- val: no valid batches")
            if is_distributed:
                dist.barrier()

        if step in _save_steps:
            _save_snapshot(step)

        # Checkpoint
        if step % ckpt_interval == 0 or step == max_steps:
            if _is_main:
                os.makedirs("checkpoints", exist_ok=True)
                ckpt_path = f"checkpoints/{run_name}_step{step:06d}.pt"
                _cfg_save = {**cfg.__dict__, "model_preset": model_preset, "run_name": run_name}
                ckpt_data = {
                    "global_step":          step,
                    "model_state_dict":     _raw_model.state_dict(),
                    "optimizer_state_dict": optimiser.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss_weights":         loss_weights.cpu(),
                    "norm_stats":           ds.norm_stats,
                    "config":               _cfg_save,
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
    if math.isfinite(best_val_loss):
        _log(f"Best val_loss={best_val_loss:.4f} @ step {best_val_loss_step}.")

    # Dump input-gate importance ranking if gate was enabled (main rank only).
    if input_gate_l1 > 0 and _is_main and hasattr(_raw_model.encoder, "input_gate_logits"):
        import json
        report = _raw_model.encoder.gate_report()
        gate_path = f"checkpoints/{run_name}_gate_report.json"
        with open(gate_path, "w") as f:
            json.dump(
                {"input_gate_l1": input_gate_l1,
                 "step": step,
                 "ranked_ascending": [[n, float(v)] for n, v in report]},
                f, indent=2,
            )
        _log(f"Wrote input-gate report → {gate_path}")
        # Print the 15 most pruned + 10 most kept for quick eyeball.
        _log("Lowest (most pruned) gates:")
        for n, v in report[:15]:
            _log(f"  {v:6.3f}  {n}")
        _log("Highest (most kept) gates:")
        for n, v in report[-10:]:
            _log(f"  {v:6.3f}  {n}")

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
    parser.add_argument("--model",      type=str, default="mimic",
                        choices=list(MODEL_PRESETS.keys()),
                        help="Model preset (default: mimic, the ~20M production model)")
    parser.add_argument("--lr",         type=float, default=None,
                        help="Learning rate (default: 3e-4; the aux-AdamW groups under Muon)")
    parser.add_argument("--run-name",   type=str, default=None,
                        help="Wandb run name (auto-generated if not set)")
    parser.add_argument("--wandb-tags", type=str, default=None,
                        help="Comma-separated wandb tags (e.g. 'lr-sweep,small')")
    parser.add_argument("--wandb-group", type=str, default=None,
                        help="Wandb group for related runs (e.g. 'sweep-v1')")
    parser.add_argument("--num-layers", type=int, default=None,
                        help="Override number of transformer layers")
    parser.add_argument("--cond-heads", action=argparse.BooleanOptionalAction, default=True,
                        help="Condition each mimic head on the one-hot class of the earlier heads "
                             "(true class in training, sampled at inference) instead of their "
                             "detached logits, so the heads model the joint distribution. On by "
                             "default since 2026-10-22; --no-cond-heads for the older heads (and to "
                             "resume runs trained without it).")
    parser.add_argument("--relpos-impl", choices=["fp32", "sdpa", "flex"], default="fp32",
                        help="Training kernel for the relpos attention (speed only; eval and "
                             "inference always use fp32). flex drops attention dropout.")
    parser.add_argument("--attn-window", type=int, default=0,
                        help="Sliding-window relpos attention: each layer sees only the last N frames "
                             "(0 = full causal). Pair with a longer --seq-len; stream with mimic/kv_cache.py.")
    parser.add_argument("--distill-from", type=str, default=None,
                        help="Checkpoint of a frozen teacher; adds alpha * KL(teacher || student) per head.")
    parser.add_argument("--distill-alpha", type=float, default=1.0)
    parser.add_argument("--save-steps", type=str, default=None,
                        help="Comma-separated optimizer steps at which to save a weights-only "
                             "snapshot checkpoints/{run}_snapNNNNNN.pt (0 = initial weights).")
    parser.add_argument("--lean-metrics", action=argparse.BooleanOptionalAction, default=True,
                        help="Compute F1/accuracy metrics only on log steps and keep the loss terms "
                             "on the GPU between them (fewer GPU->CPU syncs per step). Logged F1s "
                             "become log-step snapshots; training is unchanged. On by default since "
                             "2026-10-22; --no-lean-metrics restores per-step metrics.")
    parser.add_argument("--input-gate-l1", type=float, default=0.0,
                        help="L1 penalty on per-input-column sigmoid gate in "
                             "the encoder's projection. >0 enables the gate; "
                             "trained gate values = feature importance. "
                             "Typical values: 1e-3 to 1e-2.")
    parser.add_argument("--seq-len",    type=int, default=None,
                        help="Override sequence length (default: 60)")
    parser.add_argument("--reaction-delay", type=int, default=None,
                        help="Reaction delay in frames (default: 0, correct for v2 shards)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size (default: 200)")
    parser.add_argument("--encoder",   type=str, default="mimic_flat",
                        choices=["mimic_flat", "hal_flat"],
                        help="Frame encoder (mimic_flat; hal_flat is its pre-rename alias)")
    parser.add_argument("--dropout",   type=float, default=None,
                        help="Dropout rate (default: 0.1)")
    parser.add_argument("--clusters-path", type=str, default="data/full/stick_clusters.json",
                        help="Path to stick_clusters.json")
    parser.add_argument("--opp-inputs", action="store_true",
                        help="Include opponent controller inputs (default: excluded)")
    parser.add_argument("--self-inputs", action=argparse.BooleanOptionalAction, default=True,
                        help="Include self controller inputs (default: on; required even on v2 shards)")
    parser.add_argument("--weight-decay", type=float, default=None,
                        help="Weight decay (default: 1e-2)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--max-wall-time", type=float, default=None,
                        help="Hard wall-time limit in seconds; stop if exceeded")
    parser.add_argument("--val-frac", type=float, default=None,
                        help="Override validation frequency as fraction of max_steps (default: 0.025)")
    parser.add_argument("--warmup-steps", type=int, default=None,
                        help="Override warmup steps (default: 1%% of max_steps)")
    parser.add_argument("--grad-accum-steps", type=int, default=1,
                        help="Gradient accumulation steps (default: 1, eff_batch = bs * gpus * accum)")
    parser.add_argument("--nccl-timeout", type=int, default=1800,
                        help="NCCL timeout in seconds (default: 1800, use 3600 for large models)")
    parser.add_argument("--grad-clip-norm", type=float, default=None,
                        help="Gradient clipping norm (default: 1.0)")
    parser.add_argument("--no-load-optim", action="store_true",
                        help="Skip loading optimizer/scheduler state on resume (fresh LR)")
    parser.add_argument("--warm-restart", action="store_true",
                        help="On resume: keep Adam optimizer state but reset the LR "
                             "schedule + step counter (fresh cosine from the warm weights)")
    parser.add_argument("--eval-only", default=None,
                        help="Path to a checkpoint: score it with the exact training "
                             "val loop (val/total), print EVAL_ONLY ..., and exit. "
                             "For scoring averaged/loaded checkpoints reproducibly.")
    parser.add_argument("--optimizer", choices=["adamw", "muon"],
                        default="muon",
                        help="Optimizer for hidden matrices. 'muon' (default since 2026-10-22; "
                             "docs/research-notes-2026-10-22.md) runs Muon on "
                             "2D weight matrices + aux-AdamW on embeddings/heads/"
                             "norms/biases. 'adamw' is the pre-2026-10-22 default: use it to "
                             "reproduce or resume older runs.")
    parser.add_argument("--muon-lr", type=float, default=0.02,
                        help="LR for the Muon group (2D hidden matrices) when "
                             "--optimizer muon; the aux-AdamW group keeps --lr.")
    parser.add_argument("--plain-ce", action="store_true",
                        help="No-op: the mimic loss is always plain cross-entropy (kept so old "
                             "commands parse)")
    parser.add_argument("--n-heads", type=int, default=None,
                        help="Override number of attention heads")
    parser.add_argument("--mimic-mode", dest="mimic_mode", action="store_true",
                        help="No-op: mimic mode (single-label buttons, combined shoulder, LN heads, "
                             "plain CE) is the only mode (kept so old commands parse)")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable AMP (train in FP32)")
    parser.add_argument("--cosine-min-lr", type=float, default=None,
                        help="Minimum LR for the cosine schedule (default: 1e-6)")
    parser.add_argument("--no-warmup", action="store_true",
                        help="No LR warmup: the default since 2026-10-22 (kept so old commands parse)")
    parser.add_argument("--warmup", action="store_true",
                        help="Enable the linear LR warmup (5%% of steps, or --warmup-steps)")
    parser.add_argument("--stick-clusters", type=str, default="hal37",
                        help="Stick cluster set: 'hal37' (default) for the 37 hand-designed clusters")
    parser.add_argument("--mimic-minimal-features",
                        dest="mimic_minimal_features", action="store_true",
                        help="Minimal input set (drop ECB, speeds, hitlag/hitstun from numeric)")
    parser.add_argument("--mimic-controller-encoding",
                        dest="mimic_controller_encoding", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="One-hot controller feedback encoding (requires controller_combos.json)")
    parser.add_argument("--character-filter", type=int, default=None,
                        help="Only train on games where self_character matches this index (e.g. 1 for Fox)")
    args = parser.parse_args()

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
        dropout_override=args.dropout,
        no_opp_inputs=not args.opp_inputs,
        no_self_inputs=not args.self_inputs,
        clusters_path=args.clusters_path,
        max_wall_time=args.max_wall_time,
        val_frac_override=args.val_frac,
        warmup_steps_override=args.warmup_steps,
        weight_decay_override=args.weight_decay,
        grad_accum_steps=args.grad_accum_steps,
        nccl_timeout=args.nccl_timeout,
        grad_clip_norm=args.grad_clip_norm,
        no_load_optim=args.no_load_optim,
        warm_restart=args.warm_restart,
        eval_only=args.eval_only,
        n_heads_override=args.n_heads,
        no_amp=args.no_amp,
        cosine_min_lr=args.cosine_min_lr,
        no_warmup=not args.warmup,
        stick_clusters=args.stick_clusters,
        hal_minimal_features=args.mimic_minimal_features,
        reaction_delay_override=args.reaction_delay,
        hal_controller_encoding=args.mimic_controller_encoding,
        character_filter=args.character_filter,
        input_gate_l1=args.input_gate_l1,
        cond_heads=args.cond_heads,
        relpos_impl=args.relpos_impl,
        attn_window=args.attn_window,
        distill_from=args.distill_from,
        distill_alpha=args.distill_alpha,
        save_steps=args.save_steps,
        lean_metrics=args.lean_metrics,
        optimizer=args.optimizer,
        muon_lr=args.muon_lr,
    )
