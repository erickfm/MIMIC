#!/usr/bin/env python3
"""eval.py -- Comprehensive diagnostic evaluation of MIMIC checkpoints.

Loads a checkpoint, runs the full validation set, and reports:
  - Standard metrics (btn_f1, main_f1, shldr_f1, cdir_f1, losses)
  - Per-button F1/precision/recall breakdown
  - Stick diagnostics (neutral rate, entropy, top-1 confidence)
  - Transition-frame accuracy (accuracy on frames where ground truth changes)
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader

from dataset import StreamingMeleeDataset, MeleeFrameDatasetWithDelay, _load_cluster_centers
from model import FramePredictor, ModelConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_DTYPE = torch.bfloat16

BUTTON_NAMES = [
    "A", "B", "X", "Y", "Z", "L", "R", "START",
    "D_UP", "D_DOWN", "D_LEFT", "D_RIGHT",
]
CRITICAL_BUTTONS = {"A", "B", "X", "Y", "L", "R"}

# ── Collate ──────────────────────────────────────────────────────────────────

def collate_fn(batch):
    batch_state, batch_target = {}, {}
    for k in batch[0][0]:
        batch_state[k] = torch.stack([item[0][k] for item in batch], 0)
    for k in batch[0][1]:
        batch_target[k] = torch.stack([item[1][k] for item in batch], 0)
    return batch_state, batch_target


# ── Loss helpers (copied from train.py to keep eval standalone) ──────────────

_focal_gamma = 2.0
_label_smoothing = 0.1
_btn_focal_gamma = 2.0


def focal_loss(logits, targets, gamma=None, label_smoothing=None):
    if gamma is None:
        gamma = _focal_gamma
    if label_smoothing is None:
        label_smoothing = _label_smoothing
    n_classes = logits.shape[-1]
    with torch.no_grad():
        smooth = label_smoothing / n_classes
        targets_smooth = torch.full_like(logits, smooth)
        targets_smooth.scatter_(1, targets.unsqueeze(1), 1.0 - label_smoothing + smooth)
    log_p = F.log_softmax(logits, dim=-1)
    p = log_p.exp()
    p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
    focal_weight = (1.0 - p_t).pow(gamma)
    ce = -(targets_smooth * log_p).sum(dim=-1)
    return (focal_weight * ce).mean()


def focal_bce(pred, target, gamma=None):
    if gamma is None:
        gamma = _btn_focal_gamma
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    p = torch.sigmoid(pred)
    pt = p * target + (1 - p) * (1 - target)
    return ((1 - pt).pow(gamma) * bce).mean()


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


# ── Per-button metrics ───────────────────────────────────────────────────────

def per_button_prf(btn_hat, btn_ref):
    """Returns dict of {button_name: (f1, prec, rec)} for each of the 12 buttons."""
    results = {}
    for i, name in enumerate(BUTTON_NAMES):
        pred_i = btn_hat[:, i]
        ref_i = btn_ref[:, i]
        tp = (pred_i & ref_i).sum().float()
        fp = (pred_i & ~ref_i).sum().float()
        fn = (~pred_i & ref_i).sum().float()
        prec = (tp / (tp + fp)).item() if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn)).item() if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        press_rate = ref_i.float().mean().item()
        results[name] = {"f1": f1, "prec": prec, "rec": rec, "gt_rate": press_rate}
    return results


# ── Stick diagnostics ────────────────────────────────────────────────────────

def stick_diagnostics(main_logits, main_tgt, stick_centers):
    """Compute stick-specific diagnostic metrics.

    Returns dict with:
      - pred_neutral_rate: fraction of predictions near (0.5, 0.5)
      - gt_neutral_rate: fraction of ground truth near (0.5, 0.5)
      - entropy_mean: mean entropy of softmax distribution
      - top1_confidence: mean probability of argmax cluster
      - non_neutral_top1_acc: accuracy on frames where GT is NOT neutral
    """
    probs = F.softmax(main_logits.float(), dim=-1)
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
    if non_neutral_gt.any():
        non_neutral_acc = (pred_idx[non_neutral_gt] == main_tgt[non_neutral_gt]).float().mean().item()
    else:
        non_neutral_acc = 0.0

    return {
        "pred_neutral_rate": neutral_mask_pred.float().mean().item(),
        "gt_neutral_rate": neutral_mask_gt.float().mean().item(),
        "entropy_mean": entropy.mean().item(),
        "top1_confidence": top1_conf.mean().item(),
        "non_neutral_top1_acc": non_neutral_acc,
    }


# ── Transition accuracy ─────────────────────────────────────────────────────

def transition_metrics(btn_hat, btn_ref, main_pred_idx, main_tgt,
                       prev_btn_ref, prev_main_tgt):
    """Accuracy on frames where the ground truth changes from the previous frame.

    btn_hat/btn_ref: (B, T, 12) bools
    main_pred_idx/main_tgt: (B, T) ints
    prev_*: from the previous frame in the sequence (T-1, shifted)
    """
    results = {}

    btn_changed = (btn_ref != prev_btn_ref).any(dim=-1)
    if btn_changed.any():
        btn_match = (btn_hat == btn_ref).all(dim=-1)
        results["btn_transition_acc"] = btn_match[btn_changed].float().mean().item()
        results["btn_steady_acc"] = btn_match[~btn_changed].float().mean().item() if (~btn_changed).any() else 0.0
        results["btn_transition_frac"] = btn_changed.float().mean().item()
    else:
        results["btn_transition_acc"] = 0.0
        results["btn_steady_acc"] = 0.0
        results["btn_transition_frac"] = 0.0

    stick_changed = (main_tgt != prev_main_tgt)
    if stick_changed.any():
        stick_match = (main_pred_idx == main_tgt)
        results["stick_transition_acc"] = stick_match[stick_changed].float().mean().item()
        results["stick_steady_acc"] = stick_match[~stick_changed].float().mean().item() if (~stick_changed).any() else 0.0
        results["stick_transition_frac"] = stick_changed.float().mean().item()
    else:
        results["stick_transition_acc"] = 0.0
        results["stick_steady_acc"] = 0.0
        results["stick_transition_frac"] = 0.0

    return results


# ── Main eval loop ───────────────────────────────────────────────────────────

def evaluate(checkpoint_path: str, data_dir: str, max_batches: int = 500,
             batch_size: int = 256, num_workers: int = 8):
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    cfg_dict = ckpt["config"]
    cfg = ModelConfig(**{k: v for k, v in cfg_dict.items() if k in ModelConfig.__dataclass_fields__})

    seq_len = cfg.max_seq_len
    print(f"  Config: d_model={cfg.d_model} nhead={cfg.nhead} layers={cfg.num_layers} "
          f"seq_len={seq_len} dropout={cfg.dropout} pos_enc={cfg.pos_enc}")
    print(f"  Clusters: {cfg.n_stick_clusters} stick, {cfg.n_shoulder_bins} shoulder")

    model = FramePredictor(cfg).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} params on {DEVICE}")

    stick_centers = np.array(ckpt.get("stick_centers", []), dtype=np.float32)
    if len(stick_centers) == 0:
        cp = Path(data_dir) / "stick_clusters.json"
        sc_np, _ = _load_cluster_centers(data_dir=Path(data_dir), clusters_path=cp)
        stick_centers = sc_np

    step = ckpt.get("global_step", "?")
    print(f"  Checkpoint step: {step}")

    # Dataset
    import train as _train_mod
    _train_mod.SEQUENCE_LENGTH = seq_len
    _train_mod.BATCH_SIZE = batch_size

    p = Path(data_dir)
    has_metadata = all((p / f).exists() for f in
                       ("norm_stats.json", "cat_maps.json", "file_index.json"))
    nsi = getattr(cfg, 'no_self_inputs', False)
    if has_metadata:
        val_ds = StreamingMeleeDataset(
            data_dir=data_dir, sequence_length=seq_len,
            reaction_delay=1, split="val", no_opp_inputs=cfg.no_opp_inputs,
            no_self_inputs=nsi)
    else:
        from dataset import MeleeFrameDatasetWithDelay
        val_ds = MeleeFrameDatasetWithDelay(
            parquet_dir=data_dir, sequence_length=seq_len,
            reaction_delay=1, split="val", no_opp_inputs=cfg.no_opp_inputs,
            no_self_inputs=nsi)

    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=collate_fn,
                        drop_last=True, pin_memory=True)

    print(f"\nRunning eval on {max_batches} batches (bs={batch_size}) ...")
    t0 = time.time()

    # Accumulators for standard metrics
    std_keys = ["total", "loss_main", "loss_l", "loss_r", "loss_cdir", "loss_btn",
                "cdir_acc", "btn_acc", "btn_f1", "btn_precision", "btn_recall",
                "main_f1", "main_precision", "main_recall",
                "shldr_f1", "shldr_precision", "shldr_recall",
                "cdir_f1", "cdir_precision", "cdir_recall",
                "cdir_active_acc", "main_top1_acc", "shoulder_top1_acc"]
    sums = {k: 0.0 for k in std_keys}

    # Per-button accumulators
    all_btn_hat = []
    all_btn_ref = []

    # Stick diagnostic accumulators
    stick_diag_sums = {"pred_neutral_rate": 0.0, "gt_neutral_rate": 0.0,
                       "entropy_mean": 0.0, "top1_confidence": 0.0,
                       "non_neutral_top1_acc": 0.0}

    # Transition accumulators
    trans_sums = {"btn_transition_acc": 0.0, "btn_steady_acc": 0.0,
                  "btn_transition_frac": 0.0,
                  "stick_transition_acc": 0.0, "stick_steady_acc": 0.0,
                  "stick_transition_frac": 0.0}
    trans_ct = 0

    n_batches = 0

    with torch.no_grad():
        for vs, vt in val_dl:
            for k, v in vs.items():
                vs[k] = v.to(DEVICE, non_blocking=True)
            for k, v in vt.items():
                vt[k] = v.to(DEVICE, non_blocking=True)

            with autocast("cuda", dtype=AMP_DTYPE):
                vpreds = model(vs)

            # Standard metrics
            preds_f = {k: v.float() for k, v in vpreds.items()}
            main_pred = preds_f["main_xy"]
            l_pred = preds_f["L_val"]
            r_pred = preds_f["R_val"]
            c_logits = preds_f["c_dir_logits"]
            btn_pred = preds_f["btn_logits"]

            main_cluster_tgt = vt["main_cluster"].long()
            l_bin_tgt = vt["l_bin"].long()
            r_bin_tgt = vt["r_bin"].long()
            cdir_tgt = vt["c_dir"].long()
            btn_tgt = vt.get("btns", vt.get("btns_float")).float()

            n_main = main_pred.size(-1)
            n_shldr = l_pred.size(-1)

            loss_main = focal_loss(main_pred.reshape(-1, n_main), main_cluster_tgt.reshape(-1))
            loss_l = focal_loss(l_pred.reshape(-1, n_shldr), l_bin_tgt.reshape(-1))
            loss_r = focal_loss(r_pred.reshape(-1, n_shldr), r_bin_tgt.reshape(-1))

            cdir_classes = cdir_tgt.argmax(dim=-1)
            c_logits_flat = c_logits.reshape(-1, c_logits.size(-1))
            cdir_flat = cdir_classes.reshape(-1)
            loss_cdir = focal_loss(c_logits_flat, cdir_flat)
            loss_btn = focal_bce(btn_pred, btn_tgt)

            batch_total = loss_main.item() + loss_l.item() + loss_r.item() + loss_cdir.item() + loss_btn.item()
            if not math.isfinite(batch_total):
                continue

            main_pred_flat = main_pred.reshape(-1, n_main)
            main_tgt_flat = main_cluster_tgt.reshape(-1)
            main_pred_idx = main_pred_flat.argmax(-1)
            main_top1 = (main_pred_idx == main_tgt_flat).float().mean().item()
            main_f1, main_prec, main_rec = _multiclass_prf(main_pred_idx, main_tgt_flat, n_main)

            l_pred_idx = l_pred.reshape(-1, n_shldr).argmax(-1)
            r_pred_idx = r_pred.reshape(-1, n_shldr).argmax(-1)
            l_tgt_flat = l_bin_tgt.reshape(-1)
            r_tgt_flat = r_bin_tgt.reshape(-1)
            l_top1 = (l_pred_idx == l_tgt_flat).float().mean().item()
            r_top1 = (r_pred_idx == r_tgt_flat).float().mean().item()
            shldr_pred_idx = torch.cat([l_pred_idx, r_pred_idx])
            shldr_tgt_idx = torch.cat([l_tgt_flat, r_tgt_flat])
            shldr_f1, shldr_prec, shldr_rec = _multiclass_prf(shldr_pred_idx, shldr_tgt_idx, n_shldr)

            cdir_pred = c_logits_flat.argmax(dim=-1)
            cdir_acc = (cdir_pred == cdir_flat).float().mean().item()
            cdir_f1, cdir_prec, cdir_rec = _multiclass_prf(cdir_pred, cdir_flat, c_logits.size(-1))

            btn_prob = torch.sigmoid(btn_pred)
            btn_hat_2d = (btn_prob > 0.5)
            btn_ref_2d = (btn_tgt > 0.5)
            btn_acc = (btn_hat_2d == btn_ref_2d).float().mean().item()

            tp = (btn_hat_2d & btn_ref_2d).sum().float()
            fp = (btn_hat_2d & ~btn_ref_2d).sum().float()
            fn = (~btn_hat_2d & btn_ref_2d).sum().float()
            btn_precision = (tp / (tp + fp)).item() if (tp + fp) > 0 else 0.0
            btn_recall = (tp / (tp + fn)).item() if (tp + fn) > 0 else 0.0
            btn_f1 = 2 * btn_precision * btn_recall / (btn_precision + btn_recall) if (btn_precision + btn_recall) > 0 else 0.0

            cdir_active_mask = cdir_flat != 0
            cdir_active_acc = (cdir_pred[cdir_active_mask] == cdir_flat[cdir_active_mask]).float().mean().item() if cdir_active_mask.any() else 0.0

            metrics = {
                "total": batch_total, "loss_main": loss_main.item(), "loss_l": loss_l.item(),
                "loss_r": loss_r.item(), "loss_cdir": loss_cdir.item(), "loss_btn": loss_btn.item(),
                "cdir_acc": cdir_acc, "btn_acc": btn_acc, "btn_f1": btn_f1,
                "btn_precision": btn_precision, "btn_recall": btn_recall,
                "main_f1": main_f1, "main_precision": main_prec, "main_recall": main_rec,
                "shldr_f1": shldr_f1, "shldr_precision": shldr_prec, "shldr_recall": shldr_rec,
                "cdir_f1": cdir_f1, "cdir_precision": cdir_prec, "cdir_recall": cdir_rec,
                "cdir_active_acc": cdir_active_acc, "main_top1_acc": main_top1,
                "shoulder_top1_acc": (l_top1 + r_top1) / 2,
            }
            for k in std_keys:
                sums[k] += metrics[k]

            # Per-button: flatten (B, T, 12) -> (B*T, 12)
            all_btn_hat.append(btn_hat_2d.reshape(-1, 12).cpu())
            all_btn_ref.append(btn_ref_2d.reshape(-1, 12).cpu())

            # Stick diagnostics
            sd = stick_diagnostics(main_pred.reshape(-1, n_main),
                                   main_cluster_tgt.reshape(-1),
                                   stick_centers)
            for k in stick_diag_sums:
                stick_diag_sums[k] += sd[k]

            # Transition metrics: use seq dim (B, T) -- compare frame t to frame t-1
            B, T = main_cluster_tgt.shape[:2]
            if T > 1:
                btn_hat_3d = btn_hat_2d.reshape(B, T, 12) if btn_hat_2d.dim() == 3 else btn_hat_2d
                btn_ref_3d = btn_ref_2d.reshape(B, T, 12) if btn_ref_2d.dim() == 3 else btn_ref_2d
                main_pred_2d = main_pred.reshape(B, T, n_main).argmax(-1)
                main_tgt_2d = main_cluster_tgt.reshape(B, T) if main_cluster_tgt.dim() >= 2 else main_cluster_tgt

                if btn_hat_3d.dim() == 3 and btn_hat_3d.shape[1] == T:
                    tm = transition_metrics(
                        btn_hat_3d[:, 1:], btn_ref_3d[:, 1:],
                        main_pred_2d[:, 1:], main_tgt_2d[:, 1:],
                        btn_ref_3d[:, :-1], main_tgt_2d[:, :-1])
                    for k in trans_sums:
                        trans_sums[k] += tm[k]
                    trans_ct += 1

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

    # Average standard metrics
    avg = {k: sums[k] / n_batches for k in std_keys}

    # Per-button metrics
    all_btn_hat_t = torch.cat(all_btn_hat, dim=0)
    all_btn_ref_t = torch.cat(all_btn_ref, dim=0)
    btn_breakdown = per_button_prf(all_btn_hat_t, all_btn_ref_t)

    # Average stick diagnostics
    stick_avg = {k: stick_diag_sums[k] / n_batches for k in stick_diag_sums}

    # Average transition metrics
    if trans_ct > 0:
        trans_avg = {k: trans_sums[k] / trans_ct for k in trans_sums}
    else:
        trans_avg = trans_sums

    # ── Report ───────────────────────────────────────────────────────────────
    ckpt_name = Path(checkpoint_path).stem
    print(f"\n{'='*70}")
    print(f"  EVAL REPORT: {ckpt_name}  (step {step})")
    print(f"{'='*70}")

    print(f"\n--- Standard Metrics ---")
    print(f"  total_loss:     {avg['total']:.4f}")
    print(f"  loss_main:      {avg['loss_main']:.4f}")
    print(f"  loss_btn:       {avg['loss_btn']:.4f}")
    print(f"  loss_l/r:       {avg['loss_l']:.4f} / {avg['loss_r']:.4f}")
    print(f"  loss_cdir:      {avg['loss_cdir']:.4f}")
    print(f"")
    print(f"  btn_f1:         {avg['btn_f1']:.1%}  (P={avg['btn_precision']:.1%} R={avg['btn_recall']:.1%})")
    print(f"  main_f1:        {avg['main_f1']:.1%}  (P={avg['main_precision']:.1%} R={avg['main_recall']:.1%})")
    print(f"  shldr_f1:       {avg['shldr_f1']:.1%}  (P={avg['shldr_precision']:.1%} R={avg['shldr_recall']:.1%})")
    print(f"  cdir_f1:        {avg['cdir_f1']:.1%}  (P={avg['cdir_precision']:.1%} R={avg['cdir_recall']:.1%})")
    print(f"  main_top1_acc:  {avg['main_top1_acc']:.1%}")
    print(f"  shldr_top1_acc: {avg['shoulder_top1_acc']:.1%}")
    print(f"  cdir_acc:       {avg['cdir_acc']:.1%}  (active: {avg['cdir_active_acc']:.1%})")

    print(f"\n--- Per-Button Breakdown ---")
    print(f"  {'Button':>8}  {'F1':>6}  {'Prec':>6}  {'Rec':>6}  {'GT rate':>8}  {'Category'}")
    print(f"  {'─'*8}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*8}  {'─'*10}")
    for name in BUTTON_NAMES:
        d = btn_breakdown[name]
        cat = "CRITICAL" if name in CRITICAL_BUTTONS else ("niche" if name == "Z" else "irrelevant")
        print(f"  {name:>8}  {d['f1']:5.1%}  {d['prec']:5.1%}  {d['rec']:5.1%}  {d['gt_rate']:7.3%}  {cat}")

    crit_f1s = [btn_breakdown[n]["f1"] for n in CRITICAL_BUTTONS]
    print(f"\n  Critical-button mean F1: {sum(crit_f1s)/len(crit_f1s):.1%}")

    print(f"\n--- Stick Diagnostics ---")
    print(f"  Predicted neutral rate: {stick_avg['pred_neutral_rate']:.1%}")
    print(f"  Ground-truth neutral rate: {stick_avg['gt_neutral_rate']:.1%}")
    ratio = stick_avg['pred_neutral_rate'] / max(stick_avg['gt_neutral_rate'], 1e-6)
    print(f"  Neutral ratio (pred/gt): {ratio:.2f}x {'(LAZY)' if ratio > 1.2 else '(ok)'}")
    print(f"  Stick entropy (mean):    {stick_avg['entropy_mean']:.3f}")
    print(f"  Stick top-1 confidence:  {stick_avg['top1_confidence']:.1%}")
    print(f"  Non-neutral top-1 acc:   {stick_avg['non_neutral_top1_acc']:.1%}")

    print(f"\n--- Transition Accuracy ---")
    print(f"  Button transitions:  {trans_avg['btn_transition_frac']:.1%} of frames")
    print(f"    Transition acc:    {trans_avg['btn_transition_acc']:.1%}")
    print(f"    Steady-state acc:  {trans_avg['btn_steady_acc']:.1%}")
    gap = trans_avg['btn_steady_acc'] - trans_avg['btn_transition_acc']
    print(f"    Gap:               {gap:+.1%} {'(transitions much harder)' if gap > 0.05 else ''}")
    print(f"  Stick transitions:   {trans_avg['stick_transition_frac']:.1%} of frames")
    print(f"    Transition acc:    {trans_avg['stick_transition_acc']:.1%}")
    print(f"    Steady-state acc:  {trans_avg['stick_steady_acc']:.1%}")
    gap_s = trans_avg['stick_steady_acc'] - trans_avg['stick_transition_acc']
    print(f"    Gap:               {gap_s:+.1%} {'(transitions much harder)' if gap_s > 0.05 else ''}")

    print(f"\n{'='*70}")

    return {"standard": avg, "per_button": btn_breakdown,
            "stick": stick_avg, "transitions": trans_avg}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnostic eval for MIMIC checkpoints")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint .pt file")
    parser.add_argument("--data-dir", type=str, default="./data/full",
                        help="Data directory (default: ./data/full)")
    parser.add_argument("--max-batches", type=int, default=500,
                        help="Max validation batches to evaluate (default: 500)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size (default: 256)")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="DataLoader workers (default: 8)")
    args = parser.parse_args()

    evaluate(args.checkpoint, args.data_dir, args.max_batches,
             args.batch_size, args.num_workers)
