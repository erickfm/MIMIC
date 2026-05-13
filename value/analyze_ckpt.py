"""Diagnostics on an existing V(s) checkpoint.

Given a checkpoint, compute:
  - Per-position val loss (binned by frame_offset / game_length).
  - Per-position val acc, and pred_pos_rate.
  - Comparison to the trivial-LR ceiling per bin (computed inline).
  - Logit-magnitude histogram.
  - Reliability diagram (calibration).

The headline number we care about: does the NN beat the trivial-LR
ceiling in MID-game bins, or only in the LATE-game bin where stocks+
percent already work? If only late, the NN is just non-linearly
combining macro features and isn't a discovery tool. If mid too, the NN
is finding non-trivial signal there.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from value.dataset import compute_game_outcomes, STOCK_COL, PERCENT_COL


# Match the model loaded — these are the keys MimicFlatEncoder reads (BC subset)
BC_ENCODER_KEYS = [
    "self_numeric", "opp_numeric", "self_flags", "opp_flags",
    "self_character", "opp_character", "self_action", "opp_action",
    "stage", "self_controller",
]


def load_value_model(ckpt_path: Path, device: torch.device):
    """Build the matching ValueModel/MLP variant from the saved args."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    print(f"checkpoint args: model class inferred from keys")
    state = ckpt["model"]
    # Detect which model class to use by inspecting state_dict keys
    if any(k.startswith("blocks.") for k in state.keys()):
        # Windowed transformer
        from value.model import WindowedValueModel
        model = WindowedValueModel(
            d_model=args.get("d_model", 512),
            nhead=args.get("nhead", 8),
            num_layers=args.get("num_layers", 6),
            dim_feedforward=args.get("dim_feedforward", 2048),
            dropout=args.get("dropout", 0.1),
            head_hidden=args.get("head_hidden", 512),
            head_layers=args.get("head_layers", 2),
            use_input_gate=args.get("input_gate_l1", 0.0) > 0,
        ).to(device)
        is_windowed = True
    elif any(k.startswith("encoder.proj.") for k in state.keys()) and not any(
            k.startswith("blocks.") for k in state.keys()):
        # The old Markov MLP — encoder=MimicFlatEncoder + head MLP
        # Use the V1 ValueModel from a backup if available, or reconstruct.
        # The old value/model.py has been overwritten; reconstruct the architecture.
        from mimic.frame_encoder import MimicFlatEncoder
        import torch.nn as nn

        class _OldMLPModel(nn.Module):
            def __init__(self, args):
                super().__init__()
                self.encoder = MimicFlatEncoder(
                    d_model=args.get("d_model", 512),
                    dropout=args.get("dropout", 0.1),
                    num_stages=6, num_characters=27, num_actions=396,
                    mimic_minimal_features=False,
                    mimic_controller_encoding=True,
                    n_controller_combos=7,
                )
                d_model = args.get("d_model", 512)
                head_hidden = args.get("head_hidden", 512)
                head_layers = args.get("head_layers", 2)
                dropout = args.get("dropout", 0.1)
                layers = [nn.LayerNorm(d_model)]
                in_dim = d_model
                for _ in range(head_layers):
                    layers += [nn.Linear(in_dim, head_hidden), nn.GELU(),
                               nn.Dropout(dropout)]
                    in_dim = head_hidden
                layers.append(nn.Linear(in_dim, 1))
                self.head = nn.Sequential(*layers)

            def forward(self, state):
                h = self.encoder(state)
                logits = self.head(h).squeeze(-1)  # (B, T)
                return logits

        model = _OldMLPModel(args).to(device)
        is_windowed = False
    else:
        raise ValueError(f"Unknown checkpoint architecture: {list(state.keys())[:5]}")

    # Strict load — fail loudly if keys mismatch
    msg = model.load_state_dict(state, strict=True)
    print(f"loaded {ckpt_path.name}: step={ckpt.get('step')} val={ckpt.get('val_loss'):.4f}")
    return model, args, is_windowed


def sample_for_inference(
    data_dir: Path,
    split: str,
    n_target: int,
    window: int,
    is_windowed: bool,
    state_keys: List[str],
    seed: int = 0,
):
    """Sample frames for inference. Matches train sampling.

    For Markov models, window=1 (single frame). For windowed, returns
    W-frame slices.
    """
    with open(data_dir / "tensor_manifest.json") as f:
        manifest = json.load(f)
    shard_files = [data_dir / n for n in manifest[f"{split}_shards"]]
    rng = np.random.default_rng(seed)
    W = window if is_windowed else 1

    batches: List[dict] = []
    outcomes_all: List[float] = []
    positions_all: List[float] = []
    trivial_X_all: List[np.ndarray] = []

    n_per_shard = max(1, n_target // len(shard_files))

    for shard_path in shard_files:
        shard = torch.load(shard_path, map_location="cpu",
                           weights_only=False, mmap=True)
        states = shard["states"]
        offsets = shard["offsets"]
        n_games = int(shard["n_games"])
        if n_games == 0:
            continue

        self_stock = states["self_numeric"][:, STOCK_COL]
        opp_stock = states["opp_numeric"][:, STOCK_COL]
        self_pct = states["self_numeric"][:, PERCENT_COL]
        opp_pct = states["opp_numeric"][:, PERCENT_COL]
        gi = compute_game_outcomes(
            self_stock, opp_stock, offsets, n_games,
            self_pct_col=self_pct, opp_pct_col=opp_pct,
        )

        for _ in range(n_per_shard):
            g = int(rng.integers(0, n_games))
            s, e, outcome = gi[g]
            length = e - s
            if length < W + 1:
                continue
            max_start = length - W
            offset = int(rng.integers(0, max_start + 1))
            abs_start = s + offset
            # Build per-sample dict
            sample = {}
            for k in state_keys:
                if k not in states:
                    sample = None
                    break
                v = states[k]
                sample[k] = v[abs_start: abs_start + W]
            if sample is None:
                continue
            batches.append(sample)
            outcomes_all.append(outcome)
            # "Position" = position of the *last* frame of the window in the game
            position = (offset + W - 1) / length
            positions_all.append(position)
            # Trivial features at the last frame of the window
            last_abs = abs_start + W - 1
            trivial_X_all.append(np.array([
                self_stock[last_abs].item(),
                opp_stock[last_abs].item(),
                self_pct[last_abs].item(),
                opp_pct[last_abs].item(),
            ]))

    return batches, np.array(outcomes_all), np.array(positions_all), \
           np.stack(trivial_X_all)


def collate_value(batch_list, state_keys):
    """Stack a list of single-sample dicts into a batched dict."""
    out = {}
    for k in state_keys:
        out[k] = torch.stack([b[k] for b in batch_list], dim=0)
    return out


@torch.no_grad()
def run_inference(model, batches, state_keys, device, is_windowed: bool,
                  batch_size: int = 256) -> np.ndarray:
    """Returns (N,) logits."""
    logits_all: List[float] = []
    model.eval()
    for i in range(0, len(batches), batch_size):
        chunk = batches[i: i + batch_size]
        state = collate_value(chunk, state_keys)
        state = {k: v.to(device, non_blocking=True) for k, v in state.items()}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            logit = model(state)
        # Old Markov MLP returns (B, T) with T=1. Windowed returns (B,).
        if logit.dim() > 1:
            logit = logit[:, -1]
        logits_all.extend(logit.float().cpu().tolist())
    return np.array(logits_all)


def per_position_table(
    logits: np.ndarray, y: np.ndarray, position: np.ndarray,
    trivial_X: np.ndarray, n_bins: int = 10,
) -> List[dict]:
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (position >= lo) & (position < hi if i < n_bins - 1 else position <= hi)
        if mask.sum() < 200:
            continue
        non_draw = y[mask] != 0.5
        ym = y[mask][non_draw].astype(int)
        # NN per-bin metrics
        probs = 1.0 / (1.0 + np.exp(-logits[mask][non_draw]))
        nn_bce = log_loss(ym, probs, labels=[0, 1])
        nn_acc = ((probs > 0.5).astype(int) == ym).mean()
        nn_pred_pos = float((probs > 0.5).mean())
        # Trivial LR ceiling per bin
        Xb = trivial_X[mask][non_draw]
        # 80/20 split within bin
        n = len(Xb)
        rng = np.random.default_rng(0)
        idx = rng.permutation(n)
        n_tr = int(n * 0.8)
        lr = LogisticRegression(C=1.0, max_iter=1000)
        lr.fit(Xb[idx[:n_tr]], ym[idx[:n_tr]])
        lr_p = lr.predict_proba(Xb[idx[n_tr:]])[:, 1]
        lr_bce = log_loss(ym[idx[n_tr:]], lr_p, labels=[0, 1])
        lr_acc = (lr.predict(Xb[idx[n_tr:]]) == ym[idx[n_tr:]]).mean()
        rows.append({
            "bin": f"[{lo:.2f}, {hi:.2f})",
            "n": int(mask.sum()),
            "class_pos_rate": float(ym.mean()),
            "nn_bce": float(nn_bce),
            "nn_acc": float(nn_acc),
            "nn_pred_pos": nn_pred_pos,
            "lr_bce": float(lr_bce),
            "lr_acc": float(lr_acc),
            "nn_minus_lr_bce": float(nn_bce - lr_bce),
        })
    return rows


def logit_histogram(logits: np.ndarray, y: np.ndarray, bins: int = 20) -> dict:
    bin_edges = np.linspace(-8, 8, bins + 1)
    h, _ = np.histogram(logits.clip(-8, 8), bins=bin_edges)
    counts = h.tolist()
    edges = bin_edges.tolist()
    # Also pred_pos histogram split by class
    pos = y == 1.0
    neg = y == 0.0
    return {
        "edges": edges,
        "all_counts": counts,
        "pos_counts": np.histogram(logits[pos].clip(-8, 8), bins=bin_edges)[0].tolist(),
        "neg_counts": np.histogram(logits[neg].clip(-8, 8), bins=bin_edges)[0].tolist(),
        "mean": float(logits.mean()),
        "std": float(logits.std()),
        "min": float(logits.min()), "max": float(logits.max()),
    }


def reliability_diagram(logits: np.ndarray, y: np.ndarray,
                        n_bins: int = 10) -> dict:
    probs = 1.0 / (1.0 + np.exp(-logits))
    non_draw = y != 0.5
    p = probs[non_draw]; yd = y[non_draw].astype(int)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < n_bins - 1 else p <= edges[i + 1])
        if m.sum() < 20:
            rows.append({"bin": f"[{edges[i]:.1f},{edges[i+1]:.1f})", "n": int(m.sum()),
                         "mean_pred": None, "empirical": None})
            continue
        rows.append({
            "bin": f"[{edges[i]:.1f},{edges[i+1]:.1f})",
            "n": int(m.sum()),
            "mean_pred": float(p[m].mean()),
            "empirical": float(yd[m].mean()),
        })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-dir", default="data/fox_all_v2")
    p.add_argument("--split", default="val")
    p.add_argument("--n-samples", type=int, default=50000)
    p.add_argument("--window", type=int, default=60)
    p.add_argument("--out", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt_args, is_windowed = load_value_model(Path(args.ckpt), device)
    print(f"is_windowed: {is_windowed}")

    # Pick the right key set
    if is_windowed:
        from value.dataset import VALUE_ENCODER_KEYS
        state_keys = VALUE_ENCODER_KEYS
    else:
        state_keys = BC_ENCODER_KEYS

    print(f"sampling {args.n_samples} {args.split}-set frames...")
    W = args.window if is_windowed else 1
    batches, y, pos, trivial_X = sample_for_inference(
        Path(args.data_dir), args.split, args.n_samples, W,
        is_windowed, state_keys, args.seed,
    )
    print(f"got {len(batches)} samples")

    print("running inference...")
    logits = run_inference(model, batches, state_keys, device, is_windowed)
    probs = 1.0 / (1.0 + np.exp(-logits))

    # Overall metrics
    non_draw = y != 0.5
    overall_bce = log_loss(y[non_draw].astype(int),
                           probs[non_draw], labels=[0, 1])
    overall_acc = ((probs[non_draw] > 0.5).astype(int) == y[non_draw].astype(int)).mean()
    print(f"overall  val_bce={overall_bce:.4f}  val_acc={overall_acc:.3f}  "
          f"pred_pos={(probs > 0.5).mean():.3f}  "
          f"n={non_draw.sum()}")
    print()

    # Per-position table
    print(f"per-position breakdown (NN vs trivial LR within bin):")
    print(f"  {'bin':<14} {'n':>6} {'pos_rate':>8} "
          f"{'lr_BCE':>9} {'nn_BCE':>9} {'Δ(nn-lr)':>10} "
          f"{'lr_acc':>7} {'nn_acc':>7} {'nn_pp':>7}")
    rows = per_position_table(logits, y, pos, trivial_X)
    for r in rows:
        print(f"  {r['bin']:<14} {r['n']:>6} {r['class_pos_rate']:>8.3f} "
              f"{r['lr_bce']:>9.4f} {r['nn_bce']:>9.4f} "
              f"{r['nn_minus_lr_bce']:>+10.4f} "
              f"{r['lr_acc']:>7.3f} {r['nn_acc']:>7.3f} {r['nn_pred_pos']:>7.3f}")
    print()

    # Calibration
    print(f"reliability diagram (predicted prob vs empirical rate):")
    rel = reliability_diagram(logits, y)
    for r in rel:
        if r["mean_pred"] is None:
            print(f"  {r['bin']:<12} n={r['n']:>5}  (insufficient)")
        else:
            print(f"  {r['bin']:<12} n={r['n']:>5}  "
                  f"pred={r['mean_pred']:.3f}  empirical={r['empirical']:.3f}")
    print()

    # Logit histogram
    hist = logit_histogram(logits, y)
    print(f"logit distribution:  mean={hist['mean']:+.3f}  "
          f"std={hist['std']:.3f}  range=[{hist['min']:+.2f}, {hist['max']:+.2f}]")

    if args.out:
        payload = {
            "ckpt": str(args.ckpt),
            "is_windowed": is_windowed,
            "n_samples": int(non_draw.sum()),
            "overall_bce": float(overall_bce),
            "overall_acc": float(overall_acc),
            "overall_pred_pos": float((probs > 0.5).mean()),
            "per_position": rows,
            "reliability": rel,
            "logit_hist": hist,
        }
        Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
