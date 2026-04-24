#!/usr/bin/env python3
"""World-model training entry point.

Slim standalone loop — the BC `train.py` is baked around controller
prediction (5 losses, focal CE, stick-cluster rebinning, etc.) and
branching every one of those paths for WM was going to be churn. This
script shares `get_model` + `ModelConfig` with BC but runs its own loop
with `WorldModelDataset` + `compute_wm_loss`.

Single-GPU / single-node by default. Extend to DDP later if needed.

Example:
    python3 tools/train_wm.py \
        --model mimic-wm --data-dir data/fox_all_v2 \
        --lr 3e-4 --batch-size 64 --grad-accum-steps 8 \
        --max-samples 16777216 \
        --run-name fox-wm-$(date +%Y%m%d)-v1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mimic.model import ModelConfig, MODEL_PRESETS
from mimic.world_model import WorldModel
from mimic.wm_dataset import WorldModelDataset
from mimic.wm_losses import WMLossWeights, compute_wm_loss, compute_wm_metrics


# -----------------------------------------------------------------------------
# DDP helpers (mirror train.py's pattern so torchrun --nproc_per_node=N works).
# -----------------------------------------------------------------------------
def _ddp_setup() -> tuple[int, int, int, bool]:
    """Initialize process group from torchrun env vars.

    Returns (rank, world_size, local_rank, is_distributed). When launched
    without torchrun, returns (0, 1, 0, False) and does not init the
    process group.
    """
    if "RANK" not in os.environ:
        return 0, 1, 0, False
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank, True


def _ddp_cleanup(is_distributed: bool) -> None:
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mimic-wm",
                    choices=list(MODEL_PRESETS.keys()),
                    help="MODEL_PRESETS key; must set wm_mode=True.")
    ap.add_argument("--data-dir", required=True,
                    help="Per-game shard dir (has tensor_manifest.json).")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--cosine-min-lr", type=float, default=1e-6)
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--grad-accum-steps", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=180)
    ap.add_argument("--max-samples", type=int, default=16_777_216)
    ap.add_argument("--val-every", type=int, default=500,
                    help="Run val every N steps.")
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the model.")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--grad-clip-norm", type=float, default=1.0)
    ap.add_argument("--character-filter", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    # Loss weights
    ap.add_argument("--w-action", type=float, default=1.0)
    ap.add_argument("--w-numeric", type=float, default=1.0)
    ap.add_argument("--w-flags", type=float, default=0.5)
    ap.add_argument("--numeric-loss", default="huber", choices=["huber", "mse"],
                    help="Numeric regression loss. Huber (SmoothL1) handles "
                         "the heavy tails on z-scored velocity cols better.")
    ap.add_argument("--huber-delta", type=float, default=1.0)
    ap.add_argument("--w-action-elapsed", type=float, default=0.25,
                    help="Weight on action_elapsed regression (per side).")
    ap.add_argument("--no-action-elapsed", action="store_true",
                    help="Disable the action_elapsed head + target.")
    # W&B
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="mimic-wm")
    ap.add_argument("--wandb-tags", nargs="*", default=None)
    return ap.parse_args()


# -----------------------------------------------------------------------------
# Model + dataset construction
# -----------------------------------------------------------------------------
def build_model(model_preset: str, seq_len: int,
                n_controller_combos: int,
                shard_n_numeric: int,
                shard_n_flags: int,
                predict_action_elapsed: bool = True,
                ) -> tuple[WorldModel, ModelConfig, dict]:
    """Build the WM. Schema kwargs let us train on both shard layouts:

    - 13-col fullfeat (post-2026-04-20 schema drop): pass n_numeric=13,
      n_flags=5. Encoder uses the full path.
    - 22-col legacy (pre-2026-04-20): pass n_numeric=22, n_flags=5. Encoder
      uses the minimal path (reduces to 9 internally), heads predict the
      full 22+5 shard columns so targets and preds align.

    Returns (model, cfg, head_dims) where head_dims={"n_numeric","n_flags"}
    is what the loss should target after slicing.
    """
    overrides = dict(MODEL_PRESETS[model_preset])
    if not overrides.get("wm_mode", False):
        raise ValueError(
            f"Preset {model_preset!r} does not have wm_mode=True. "
            f"Use 'mimic-wm' or add a wm_mode preset."
        )
    overrides.pop("max_seq_len", None)
    overrides["mimic_mode"] = True
    overrides["mimic_controller_encoding"] = True
    # Encoder path: full (13-col) or minimal (22/13/7-col, reduces to 9).
    use_minimal = shard_n_numeric != 13
    overrides["mimic_minimal_features"] = use_minimal
    overrides["n_controller_combos"] = n_controller_combos
    overrides["no_self_inputs"] = False
    overrides["no_opp_inputs"] = True
    cfg = ModelConfig(max_seq_len=seq_len, **overrides)

    # Heads predict the shard's native column layout so MSE/BCE line up
    # with the raw targets from wm_dataset. When use_minimal=True and the
    # shard is 22-col, we still predict all 22 + 5 (not 6 + 3) — simpler,
    # and the tail cols the encoder ignores are easy signal for the head.
    head_n_numeric = shard_n_numeric
    head_n_flags = shard_n_flags
    # Temporarily clobber the heads by constructing a WorldModel then
    # swapping in correctly-sized heads. We can't pass these through
    # ModelConfig without adding fields, and WorldModelHeads accepts the
    # dims directly. Same d_model / num_actions from cfg.
    model = WorldModel(cfg)
    from mimic.world_model import WorldModelHeads
    model.heads = WorldModelHeads(
        d_model=cfg.d_model,
        num_actions=cfg.num_actions,
        n_numeric=head_n_numeric,
        n_flags=head_n_flags,
        predict_action_elapsed=predict_action_elapsed,
    )
    return model, cfg, {"n_numeric": head_n_numeric, "n_flags": head_n_flags,
                         "predict_action_elapsed": predict_action_elapsed}


def collate(batch):
    """Stack a list of 3-tuples into batched dicts."""
    states, next_ctrls, targets = zip(*batch)

    def stack(dicts):
        keys = dicts[0].keys()
        return {k: torch.stack([d[k] for d in dicts], dim=0) for k in keys}

    return stack(states), stack(next_ctrls), stack(targets)


# -----------------------------------------------------------------------------
# Train loop
# -----------------------------------------------------------------------------
def make_lr_lambda(warmup_steps: int, total_steps: int,
                   min_lr_ratio: float):
    """Warmup → cosine decay to min_lr_ratio * base_lr."""
    def fn(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine
    return fn


def run_val(model: WorldModel, dl: DataLoader, device: torch.device,
            weights: WMLossWeights, max_batches: int = 50
            ) -> Dict[str, float]:
    model.eval()
    loss_sum = 0.0
    metrics_sum: Dict[str, float] = {}
    n = 0
    with torch.no_grad():
        for i, batch in enumerate(dl):
            if i >= max_batches:
                break
            state, next_ctrl, target = batch
            state = {k: v.to(device, non_blocking=True) for k, v in state.items()}
            next_ctrl = {k: v.to(device, non_blocking=True) for k, v in next_ctrl.items()}
            target = {k: v.to(device, non_blocking=True) for k, v in target.items()}
            frames = {**state, **next_ctrl}
            preds = model(frames)
            losses = compute_wm_loss(preds, target, weights)
            loss_sum += losses["total"].item()
            m = compute_wm_metrics(preds, target)
            for k, v in m.items():
                metrics_sum[k] = metrics_sum.get(k, 0.0) + v
            n += 1
    model.train()
    if n == 0:
        return {}
    out = {"val_loss": loss_sum / n}
    out.update({f"val_{k}": v / n for k, v in metrics_sum.items()})
    return out


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    # --- DDP bring-up
    rank, world_size, local_rank, is_distributed = _ddp_setup()
    is_main = rank == 0
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    def log(msg: str) -> None:
        if is_main:
            print(msg, flush=True)

    log(f"device={device}  rank={rank}/{world_size}  "
        f"amp={'off' if args.no_amp else 'bf16'}  compile={args.compile}")

    # --- Load controller_combos from data dir to size the model correctly.
    combos_path = Path(args.data_dir) / "controller_combos.json"
    with open(combos_path) as fh:
        combo_map = json.load(fh)
    if isinstance(combo_map, dict) and "combos" in combo_map:
        n_combos = len(combo_map["combos"])
    elif isinstance(combo_map, dict):
        n_combos = combo_map.get("n_combos", len(combo_map))
    else:
        n_combos = len(combo_map)
    log(f"n_controller_combos={n_combos}")

    # Probe one shard to surface the schema and route encoder/heads.
    _probe_ds = WorldModelDataset(
        args.data_dir, sequence_length=args.seq_len, split="train",
        character_filter=args.character_filter, distributed=False,
        windows_per_game=1,
    )
    shard_n_numeric = _probe_ds.n_numeric
    shard_n_flags = _probe_ds.n_flags
    log(f"shard schema: n_numeric={shard_n_numeric}  n_flags={shard_n_flags}  "
        f"→ encoder minimal={shard_n_numeric != 13}")

    model, cfg, head_dims = build_model(
        args.model, args.seq_len, n_combos,
        shard_n_numeric=shard_n_numeric,
        shard_n_flags=shard_n_flags,
        predict_action_elapsed=not args.no_action_elapsed,
    )
    model = model.to(device)
    if args.compile:
        model = torch.compile(model)
    if is_distributed:
        model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None,
                    find_unused_parameters=False)
    params = sum(p.numel() for p in model.parameters())
    log(f"WorldModel params: {params / 1e6:.2f}M  "
        f"(d_model={cfg.d_model}, layers={cfg.num_layers}, heads={cfg.nhead})")

    # --- Datasets (rank-aware)
    train_ds = WorldModelDataset(
        args.data_dir, sequence_length=args.seq_len, split="train",
        character_filter=args.character_filter,
        rank=rank, world_size=world_size, distributed=is_distributed,
    )
    val_ds = WorldModelDataset(
        args.data_dir, sequence_length=args.seq_len, split="val",
        character_filter=args.character_filter,
        rank=rank, world_size=world_size, distributed=is_distributed,
        windows_per_game=10,
    )
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None,
        collate_fn=collate,
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size, num_workers=max(1, args.num_workers // 2),
        pin_memory=True, persistent_workers=False,
        collate_fn=collate,
    )

    # --- Optimizer + LR schedule.
    # Effective batch = local_batch × world_size × grad_accum, matching the
    # BC train.py formula. Do NOT divide-by-world-size elsewhere.
    effective_batch = args.batch_size * world_size * args.grad_accum_steps
    total_steps = args.max_samples // effective_batch
    log(f"effective batch size: {effective_batch}  total_steps: {total_steps}")

    # Unwrap DDP for optimizer parameter collection — AdamW doesn't care,
    # but this keeps state dicts clean.
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    warmup = 0 if args.no_warmup else args.warmup_steps
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, make_lr_lambda(warmup, total_steps, args.cosine_min_lr / args.lr)
    )

    weights = WMLossWeights(
        action_self=args.w_action, action_opp=args.w_action,
        numeric_self=args.w_numeric, numeric_opp=args.w_numeric,
        flags_self=args.w_flags, flags_opp=args.w_flags,
        action_elapsed_self=args.w_action_elapsed,
        action_elapsed_opp=args.w_action_elapsed,
        numeric_loss=args.numeric_loss, huber_delta=args.huber_delta,
    )

    # --- W&B (rank-0 only)
    wandb_run = None
    if args.wandb and is_main:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project, name=args.run_name,
            tags=args.wandb_tags or [],
            config={**vars(args), "total_steps": total_steps, "params_M": params / 1e6,
                    "world_size": world_size, "effective_batch": effective_batch,
                    "shard_n_numeric": shard_n_numeric, "shard_n_flags": shard_n_flags},
        )

    # --- Train loop
    if is_main:
        Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    best_path = Path(args.ckpt_dir) / f"{args.run_name}_best.pt"

    train_iter = iter(train_dl)
    step = 0
    t0 = time.time()
    model.train()
    amp_dtype = None if args.no_amp else torch.bfloat16

    while step < total_steps:
        opt.zero_grad(set_to_none=True)
        loss_log: Dict[str, float] = {}
        for accum in range(args.grad_accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dl)
                batch = next(train_iter)

            state, next_ctrl, target = batch
            state = {k: v.to(device, non_blocking=True) for k, v in state.items()}
            next_ctrl = {k: v.to(device, non_blocking=True) for k, v in next_ctrl.items()}
            target = {k: v.to(device, non_blocking=True) for k, v in target.items()}
            frames = {**state, **next_ctrl}

            autocast_cm = (
                torch.autocast(device_type="cuda", dtype=amp_dtype)
                if amp_dtype is not None else _nullctx()
            )
            with autocast_cm:
                preds = model(frames)
                losses = compute_wm_loss(preds, target, weights)
                loss = losses["total"] / args.grad_accum_steps

            loss.backward()
            for k, v in losses.items():
                loss_log[k] = loss_log.get(k, 0.0) + v.item() / args.grad_accum_steps

        if args.grad_clip_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        opt.step()
        sched.step()
        step += 1

        if step % 50 == 0:
            # All-reduce loss for accurate global reporting
            if is_distributed:
                loss_log = _reduce_scalars(loss_log, world_size)
            if is_main:
                dt = time.time() - t0
                sps = (step * effective_batch) / dt
                lr = opt.param_groups[0]["lr"]
                extras = ""
                if "action_elapsed_self" in loss_log:
                    extras = (
                        f"  elp_s {loss_log['action_elapsed_self']:.3f}"
                        f"  elp_o {loss_log['action_elapsed_opp']:.3f}"
                    )
                print(f"step {step}/{total_steps}  total {loss_log['total']:.4f}  "
                      f"act_s {loss_log['action_self']:.3f}  act_o {loss_log['action_opp']:.3f}  "
                      f"num_s {loss_log['numeric_self']:.3f}  num_o {loss_log['numeric_opp']:.3f}  "
                      f"flg_s {loss_log['flags_self']:.3f}  flg_o {loss_log['flags_opp']:.3f}"
                      f"{extras}  lr {lr:.2e}  {sps:.0f} sam/s", flush=True)
                if wandb_run is not None:
                    wandb_run.log({
                        "train/step": step,
                        "train/lr": lr,
                        "train/samples_per_sec": sps,
                        **{f"train/{k}": v for k, v in loss_log.items()},
                    })

        if step % args.val_every == 0:
            log(f"running val @ step {step}...")
            val = run_val(model, val_dl, device, weights)
            if is_distributed:
                val = _reduce_scalars(val, world_size)
            if is_main:
                print("  " + "  ".join(f"{k}={v:.4f}" for k, v in val.items()), flush=True)
                if wandb_run is not None:
                    wandb_run.log({"step": step, **val})
                vl = val.get("val_loss", float("inf"))
                if vl < best_val:
                    best_val = vl
                    _save_ckpt(best_path, model, opt, sched, cfg, step, val)
                    print(f"  new best val_loss={vl:.4f}, saved {best_path}",
                          flush=True)

        if step % args.save_every == 0 and is_main:
            path = Path(args.ckpt_dir) / f"{args.run_name}_step{step}.pt"
            _save_ckpt(path, model, opt, sched, cfg, step, {})
            print(f"  checkpoint: {path}", flush=True)

    log(f"done. best val_loss={best_val:.4f}  best path={best_path}")
    if wandb_run is not None:
        wandb_run.finish()
    _ddp_cleanup(is_distributed)


def _reduce_scalars(d: Dict[str, float], world_size: int) -> Dict[str, float]:
    """All-reduce-mean a dict of Python floats across ranks."""
    if not d:
        return d
    keys = list(d.keys())
    vals = torch.tensor([float(d[k]) for k in keys],
                        device="cuda" if torch.cuda.is_available() else "cpu")
    dist.all_reduce(vals, op=dist.ReduceOp.SUM)
    vals /= world_size
    return {k: vals[i].item() for i, k in enumerate(keys)}


def _unwrap(model):
    """Peel off DDP / torch.compile wrappers to get the bare WorldModel."""
    m = model.module if hasattr(model, "module") else model
    m = m._orig_mod if hasattr(m, "_orig_mod") else m
    return m


def _save_ckpt(path, model, opt, sched, cfg, step, val):
    sd = _unwrap(model).state_dict()
    torch.save({
        "global_step": step,
        "model_state_dict": sd,
        "optimizer_state_dict": opt.state_dict(),
        "scheduler_state_dict": sched.state_dict(),
        "config": cfg.__dict__ | {"wm_mode": True},
        "val": val,
    }, path)


class _nullctx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


if __name__ == "__main__":
    main()
