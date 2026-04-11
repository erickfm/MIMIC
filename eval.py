#!/usr/bin/env python3
"""eval.py -- Diagnostic evaluation of MIMIC checkpoints.

Supports both standard (multi-label buttons) and HAL mode (single-label combos).

Usage:
    python eval.py checkpoints/my_best.pt --data-dir data/fox_public_shards
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn
from torch.amp import autocast
from torch.utils.data import DataLoader

from mimic.dataset import StreamingMeleeDataset
from mimic.features import load_cluster_centers, load_controller_combos
from mimic.model import FramePredictor, ModelConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_DTYPE = torch.bfloat16

BUTTON_NAMES_12 = [
    "A", "B", "X", "Y", "Z", "L", "R", "START",
    "D_UP", "D_DOWN", "D_LEFT", "D_RIGHT",
]


def collate_fn(batch):
    batch_state, batch_target = {}, {}
    for k in batch[0][0]:
        batch_state[k] = torch.stack([item[0][k] for item in batch], 0)
    for k in batch[0][1]:
        batch_target[k] = torch.stack([item[1][k] for item in batch], 0)
    return batch_state, batch_target


def _multiclass_prf(pred_idx, tgt_idx, n_classes):
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
    """Priority-based: A > B > jump > Z > NONE."""
    a = btn_tgt[..., 0] > 0.5
    b = btn_tgt[..., 1] > 0.5
    jump = (btn_tgt[..., 2] > 0.5) | (btn_tgt[..., 3] > 0.5)
    z = btn_tgt[..., 4] > 0.5
    single = torch.full(btn_tgt.shape[:-1], 4, device=btn_tgt.device, dtype=torch.long)
    single[z] = 3
    single[jump] = 2
    single[b] = 1
    single[a] = 0
    return single


def _multi_hot_to_combo_label(btn_tgt, combo_lookup):
    """Vectorized combo lookup."""
    flat = btn_tgt.reshape(-1, 12)
    a = (flat[:, 0] > 0.5).long()
    b = (flat[:, 1] > 0.5).long()
    jump = ((flat[:, 2] > 0.5) | (flat[:, 3] > 0.5)).long()
    z = (flat[:, 4] > 0.5).long()
    shoulder = ((flat[:, 5] > 0.5) | (flat[:, 6] > 0.5)).long()
    combo_int = a * 16 + b * 8 + jump * 4 + z * 2 + shoulder
    return combo_lookup.to(combo_int.device)[combo_int].reshape(btn_tgt.shape[:-1])


def _build_combo_lookup(combo_map):
    table = torch.zeros(32, dtype=torch.long)
    for combo, idx in combo_map.items():
        combo_int = combo[0]*16 + combo[1]*8 + combo[2]*4 + combo[3]*2 + combo[4]
        table[combo_int] = idx
    return table


def _combine_shoulder_targets(l_bin, r_bin, shoulder_centers):
    l_vals = shoulder_centers[l_bin]
    r_vals = shoulder_centers[r_bin]
    combined = torch.max(l_vals, r_vals)
    hal_centers = torch.tensor([0.0, 0.4, 1.0], device=combined.device)
    dists = (combined.unsqueeze(-1) - hal_centers).abs()
    return dists.argmin(dim=-1)


def stick_diagnostics(main_logits, main_tgt, stick_centers):
    probs = Fn.softmax(main_logits.float(), dim=-1)
    pred_idx = probs.argmax(dim=-1)
    top1_conf = probs.max(dim=-1).values
    entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)

    centers = torch.tensor(stick_centers, device=main_logits.device, dtype=torch.float32)
    neutral_mask_pred = torch.zeros(pred_idx.shape, dtype=torch.bool, device=pred_idx.device)
    neutral_mask_gt = torch.zeros(main_tgt.shape, dtype=torch.bool, device=main_tgt.device)
    for i, c in enumerate(centers):
        dist = ((c[0] - 0.5)**2 + (c[1] - 0.5)**2).sqrt()
        if dist < 0.05:
            neutral_mask_pred |= (pred_idx == i)
            neutral_mask_gt |= (main_tgt == i)

    non_neutral_gt = ~neutral_mask_gt
    non_neutral_acc = (pred_idx[non_neutral_gt] == main_tgt[non_neutral_gt]).float().mean().item() if non_neutral_gt.any() else 0.0

    return {
        "pred_neutral_rate": neutral_mask_pred.float().mean().item(),
        "gt_neutral_rate": neutral_mask_gt.float().mean().item(),
        "entropy_mean": entropy.mean().item(),
        "top1_confidence": top1_conf.mean().item(),
        "non_neutral_top1_acc": non_neutral_acc,
    }


def evaluate(checkpoint_path: str, data_dir: str, max_batches: int = 500,
             batch_size: int = 256, num_workers: int = 8):
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    cfg_dict = ckpt["config"]
    cfg = ModelConfig(**{k: v for k, v in cfg_dict.items() if k in ModelConfig.__dataclass_fields__})

    hal_mode = cfg.hal_mode
    hal_ctrl = getattr(cfg, 'hal_controller_encoding', False)
    n_combos = getattr(cfg, 'n_controller_combos', 5)
    seq_len = cfg.max_seq_len
    rd = cfg_dict.get("reaction_delay", 1)

    print(f"  Config: d_model={cfg.d_model} layers={cfg.num_layers} "
          f"encoder={cfg.encoder_type} seq_len={seq_len}")
    print(f"  HAL mode={hal_mode}  controller_encoding={hal_ctrl}  n_combos={n_combos}")

    model = FramePredictor(cfg).to(DEVICE)
    sd = ckpt["model_state_dict"]
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} params on {DEVICE}")

    stick_centers = np.array(ckpt.get("stick_centers", []), dtype=np.float32)
    if len(stick_centers) == 0:
        sc_np, _ = load_cluster_centers(data_dir=Path(data_dir))
        stick_centers = sc_np

    shoulder_centers = np.array(ckpt.get("shoulder_centers", []), dtype=np.float32)
    if len(shoulder_centers) == 0:
        _, sh_np = load_cluster_centers(data_dir=Path(data_dir))
        shoulder_centers = sh_np
    shoulder_centers_t = torch.tensor(shoulder_centers, device=DEVICE, dtype=torch.float32) if shoulder_centers is not None else None

    # Controller combo support
    combo_map = None
    combo_lookup = None
    combo_names = []
    if hal_ctrl:
        if "controller_combos" in ckpt:
            combos = [tuple(c) for c in ckpt["controller_combos"]]
            combo_map = {c: i for i, c in enumerate(combos)}
        else:
            combos_path = Path(data_dir) / "controller_combos.json"
            if combos_path.exists():
                combos, combo_map, _ = load_controller_combos(data_dir)
            else:
                print(f"  WARNING: hal_controller_encoding but no combos found")
        if combo_map:
            n_combos = len(combo_map)
            combo_lookup = _build_combo_lookup(combo_map)
            NAMES = ["A", "B", "Jump", "Z", "Shoulder"]
            combo_names = []
            for combo in sorted(combo_map.keys(), key=combo_map.get):
                pressed = [NAMES[j] for j, v in enumerate(combo) if v]
                combo_names.append("+".join(pressed) if pressed else "NONE")
            print(f"  Loaded {n_combos} button combos")

    # Dataset
    controller_offset = cfg_dict.get("controller_offset", False)
    ds_kwargs = dict(
        data_dir=data_dir, sequence_length=seq_len,
        reaction_delay=0 if hal_mode else 1, split="val",
        controller_offset=controller_offset,
    )
    if hal_ctrl and combo_map:
        ds_kwargs["hal_controller_encoding"] = True
        ds_kwargs["controller_combo_map"] = combo_map
        ds_kwargs["n_controller_combos"] = n_combos

    val_ds = StreamingMeleeDataset(**ds_kwargs)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=collate_fn,
                        drop_last=True, pin_memory=True)

    print(f"\nRunning eval on up to {max_batches} batches (bs={batch_size}) ...")
    t0 = time.time()

    # Accumulators
    loss_sums = {"total": 0.0, "main": 0.0, "shoulder": 0.0, "cdir": 0.0, "btn": 0.0}
    metric_sums = {"btn_f1": 0.0, "btn_prec": 0.0, "btn_rec": 0.0,
                   "main_f1": 0.0, "shldr_f1": 0.0, "cdir_f1": 0.0,
                   "main_top1": 0.0, "cdir_acc": 0.0}
    stick_diag_sums = {"pred_neutral_rate": 0.0, "gt_neutral_rate": 0.0,
                       "entropy_mean": 0.0, "top1_confidence": 0.0,
                       "non_neutral_top1_acc": 0.0}

    # Per-class button accumulators (for breakdown)
    if hal_mode:
        btn_class_tp = torch.zeros(n_combos)
        btn_class_fp = torch.zeros(n_combos)
        btn_class_fn = torch.zeros(n_combos)
        btn_class_count = torch.zeros(n_combos)

    n_batches = 0

    with torch.no_grad():
        for vs, vt in val_dl:
            for k, v in vs.items():
                vs[k] = v.to(DEVICE, non_blocking=True)
            for k, v in vt.items():
                vt[k] = v.to(DEVICE, non_blocking=True)

            with autocast("cuda", dtype=AMP_DTYPE):
                vpreds = model(vs)

            preds_f = {k: v.float() for k, v in vpreds.items()}
            main_pred = preds_f["main_xy"]
            c_logits = preds_f["c_dir_logits"]
            btn_pred = preds_f["btn_logits"]

            main_tgt = vt["main_cluster"].long()
            cdir_tgt = vt["c_dir"].long()
            btn_tgt = vt.get("btns", vt.get("btns_float")).float()

            n_main = main_pred.size(-1)

            # Main stick loss + metrics
            loss_main = Fn.cross_entropy(main_pred.reshape(-1, n_main), main_tgt.reshape(-1))
            main_pred_idx = main_pred.reshape(-1, n_main).argmax(-1)
            main_tgt_flat = main_tgt.reshape(-1)
            main_top1 = (main_pred_idx == main_tgt_flat).float().mean().item()
            main_f1, _, _ = _multiclass_prf(main_pred_idx, main_tgt_flat, n_main)

            # C-dir loss + metrics
            cdir_classes = cdir_tgt.argmax(dim=-1) if cdir_tgt.dim() > 2 or (cdir_tgt.dim() == 2 and cdir_tgt.shape[-1] > 1) else cdir_tgt
            if cdir_classes.dim() > 1 and cdir_classes.shape[-1] > 1:
                cdir_classes = cdir_classes.argmax(dim=-1)
            loss_cdir = Fn.cross_entropy(c_logits.reshape(-1, c_logits.size(-1)), cdir_classes.reshape(-1))
            cdir_pred = c_logits.reshape(-1, c_logits.size(-1)).argmax(-1)
            cdir_flat = cdir_classes.reshape(-1)
            cdir_acc = (cdir_pred == cdir_flat).float().mean().item()
            cdir_f1, _, _ = _multiclass_prf(cdir_pred, cdir_flat, c_logits.size(-1))

            if hal_mode:
                # Combined shoulder
                shldr_pred = preds_f["shoulder_val"]
                l_bin_tgt = vt["l_bin"].long()
                r_bin_tgt = vt["r_bin"].long()
                shldr_tgt = _combine_shoulder_targets(l_bin_tgt, r_bin_tgt, shoulder_centers_t)
                loss_shldr = Fn.cross_entropy(shldr_pred.reshape(-1, shldr_pred.size(-1)), shldr_tgt.reshape(-1))
                shldr_pred_idx = shldr_pred.reshape(-1, shldr_pred.size(-1)).argmax(-1)
                shldr_f1, _, _ = _multiclass_prf(shldr_pred_idx, shldr_tgt.reshape(-1), shldr_pred.size(-1))

                # Button: single-label (combo or 5-class)
                if hal_ctrl and combo_lookup is not None:
                    btn_labels = _multi_hot_to_combo_label(btn_tgt, combo_lookup)
                else:
                    btn_labels = _multi_hot_to_single_label(btn_tgt)
                loss_btn = Fn.cross_entropy(btn_pred.reshape(-1, btn_pred.size(-1)), btn_labels.reshape(-1))
                btn_pred_idx = btn_pred.reshape(-1, btn_pred.size(-1)).argmax(-1)
                btn_tgt_flat = btn_labels.reshape(-1)
                btn_f1, btn_prec, btn_rec = _multiclass_prf(btn_pred_idx, btn_tgt_flat, btn_pred.size(-1))

                # Per-class accumulation
                correct = btn_pred_idx == btn_tgt_flat
                wrong = ~correct
                btn_class_tp.scatter_add_(0, btn_tgt_flat[correct].cpu(), torch.ones(correct.sum().item()))
                btn_class_fp.scatter_add_(0, btn_pred_idx[wrong].cpu(), torch.ones(wrong.sum().item()))
                btn_class_fn.scatter_add_(0, btn_tgt_flat[wrong].cpu(), torch.ones(wrong.sum().item()))
                for c in range(n_combos):
                    btn_class_count[c] += (btn_tgt_flat == c).sum().item()

                loss_total = loss_main.item() + loss_shldr.item() + loss_cdir.item() + loss_btn.item()
                loss_sums["shoulder"] += loss_shldr.item()
            else:
                # Standard: separate L/R, multi-label buttons
                l_pred = preds_f["L_val"]
                r_pred = preds_f["R_val"]
                l_bin_tgt = vt["l_bin"].long()
                r_bin_tgt = vt["r_bin"].long()
                n_shldr = l_pred.size(-1)
                loss_l = Fn.cross_entropy(l_pred.reshape(-1, n_shldr), l_bin_tgt.reshape(-1))
                loss_r = Fn.cross_entropy(r_pred.reshape(-1, n_shldr), r_bin_tgt.reshape(-1))
                loss_shldr_val = loss_l.item() + loss_r.item()

                shldr_pred_idx = torch.cat([l_pred.reshape(-1, n_shldr).argmax(-1),
                                            r_pred.reshape(-1, n_shldr).argmax(-1)])
                shldr_tgt_idx = torch.cat([l_bin_tgt.reshape(-1), r_bin_tgt.reshape(-1)])
                shldr_f1, _, _ = _multiclass_prf(shldr_pred_idx, shldr_tgt_idx, n_shldr)

                btn_prob = torch.sigmoid(btn_pred)
                btn_hat = (btn_prob > 0.5)
                btn_ref = (btn_tgt > 0.5)
                tp = (btn_hat & btn_ref).sum().float()
                fp = (btn_hat & ~btn_ref).sum().float()
                fn = (~btn_hat & btn_ref).sum().float()
                btn_prec = (tp / (tp + fp)).item() if (tp + fp) > 0 else 0.0
                btn_rec = (tp / (tp + fn)).item() if (tp + fn) > 0 else 0.0
                btn_f1 = 2 * btn_prec * btn_rec / (btn_prec + btn_rec) if (btn_prec + btn_rec) > 0 else 0.0
                loss_btn = Fn.binary_cross_entropy_with_logits(btn_pred, btn_tgt).item()

                loss_total = loss_main.item() + loss_shldr_val + loss_cdir.item() + loss_btn
                loss_sums["shoulder"] += loss_shldr_val

            if not math.isfinite(loss_total):
                continue

            loss_sums["total"] += loss_total
            loss_sums["main"] += loss_main.item()
            loss_sums["cdir"] += loss_cdir.item()
            loss_sums["btn"] += loss_btn if isinstance(loss_btn, float) else loss_btn.item()

            metric_sums["btn_f1"] += btn_f1
            metric_sums["btn_prec"] += btn_prec
            metric_sums["btn_rec"] += btn_rec
            metric_sums["main_f1"] += main_f1
            metric_sums["shldr_f1"] += shldr_f1
            metric_sums["cdir_f1"] += cdir_f1
            metric_sums["main_top1"] += main_top1
            metric_sums["cdir_acc"] += cdir_acc

            sd = stick_diagnostics(main_pred.reshape(-1, n_main), main_tgt.reshape(-1), stick_centers)
            for k in stick_diag_sums:
                stick_diag_sums[k] += sd[k]

            n_batches += 1
            if n_batches % 50 == 0:
                print(f"  {n_batches}/{max_batches} batches ...", flush=True)
            if n_batches >= max_batches:
                break

    elapsed = time.time() - t0
    print(f"\nEval done: {n_batches} batches in {elapsed:.1f}s")

    if n_batches == 0:
        print("ERROR: No valid batches")
        return

    # Averages
    avg_loss = {k: v / n_batches for k, v in loss_sums.items()}
    avg_m = {k: v / n_batches for k, v in metric_sums.items()}
    stick_avg = {k: v / n_batches for k, v in stick_diag_sums.items()}

    # Report
    ckpt_name = Path(checkpoint_path).stem
    step = ckpt.get("global_step", "?")
    print(f"\n{'='*70}")
    print(f"  EVAL REPORT: {ckpt_name}  (step {step})")
    print(f"{'='*70}")

    print(f"\n--- Losses ---")
    print(f"  total:    {avg_loss['total']:.4f}")
    print(f"  main:     {avg_loss['main']:.4f}")
    print(f"  shoulder: {avg_loss['shoulder']:.4f}")
    print(f"  cdir:     {avg_loss['cdir']:.4f}")
    print(f"  btn:      {avg_loss['btn']:.4f}")

    print(f"\n--- Metrics ---")
    print(f"  btn_f1:    {avg_m['btn_f1']:.1%}  (P={avg_m['btn_prec']:.1%} R={avg_m['btn_rec']:.1%})")
    print(f"  main_f1:   {avg_m['main_f1']:.1%}")
    print(f"  shldr_f1:  {avg_m['shldr_f1']:.1%}")
    print(f"  cdir_f1:   {avg_m['cdir_f1']:.1%}")
    print(f"  main_top1: {avg_m['main_top1']:.1%}")
    print(f"  cdir_acc:  {avg_m['cdir_acc']:.1%}")

    # Per-class button breakdown (HAL mode)
    if hal_mode and combo_names:
        print(f"\n--- Button Combo Breakdown ---")
        print(f"  {'Combo':<25} {'F1':>6}  {'Prec':>6}  {'Rec':>6}  {'GT%':>8}")
        print(f"  {'─'*25} {'─'*6}  {'─'*6}  {'─'*6}  {'─'*8}")
        total_gt = btn_class_count.sum().item()
        for i, name in enumerate(combo_names):
            tp = btn_class_tp[i].item()
            fp = btn_class_fp[i].item()
            fn = btn_class_fn[i].item()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            gt_pct = btn_class_count[i].item() / total_gt if total_gt > 0 else 0.0
            print(f"  {name:<25} {f1:5.1%}  {prec:5.1%}  {rec:5.1%}  {gt_pct:7.2%}")

    print(f"\n--- Stick Diagnostics ---")
    print(f"  Pred neutral rate:     {stick_avg['pred_neutral_rate']:.1%}")
    print(f"  GT neutral rate:       {stick_avg['gt_neutral_rate']:.1%}")
    ratio = stick_avg['pred_neutral_rate'] / max(stick_avg['gt_neutral_rate'], 1e-6)
    print(f"  Neutral ratio (P/GT):  {ratio:.2f}x {'(LAZY)' if ratio > 1.2 else '(ok)'}")
    print(f"  Stick entropy (mean):  {stick_avg['entropy_mean']:.3f}")
    print(f"  Stick top-1 conf:      {stick_avg['top1_confidence']:.1%}")
    print(f"  Non-neutral top-1 acc: {stick_avg['non_neutral_top1_acc']:.1%}")

    print(f"\n{'='*70}")

    return {"losses": avg_loss, "metrics": avg_m, "stick": stick_avg}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnostic eval for MIMIC checkpoints")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint .pt file")
    parser.add_argument("--data-dir", type=str, default="./data/full",
                        help="Data directory (default: ./data/full)")
    parser.add_argument("--max-batches", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    evaluate(args.checkpoint, args.data_dir, args.max_batches,
             args.batch_size, args.num_workers)
