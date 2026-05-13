"""Cheap, compute-light diagnostics for V(s) discovery.

What this answers (without training another NN):

1. Trivial-features ceiling — fit a logistic regression on
   {self_stock, opp_stock, self_percent, opp_percent} and report BCE.
   This is the floor V(s) needs to beat to be doing anything non-trivial.

2. Wider-features ceiling — same but on all 36 per-player numeric+flag
   features. The bound V(s) needs to push past to find non-stock-percent
   signal.

3. Per-position breakdown — bin frames by where they sit in their game
   (offset/length quantile) and compute per-bin LR ceiling + class
   balance. Tells us where the predictable signal lives.

4. Per-feature ranking — fit a 1-feature LR for each named scalar.
   Surfaces what features individually carry signal.

5. Existing-checkpoint diagnostics — given a value/* checkpoint, compute
   per-position val loss + a logit-magnitude histogram + a reliability
   diagram. Tells us where the model loses, and how calibrated it is.

Sampling matches the V(s) trainer: uniform over games × uniform over
frame-position within game. So the ceiling numbers are directly
comparable to the NN val_loss numbers.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from value.dataset import compute_game_outcomes, STOCK_COL, PERCENT_COL


# Named scalar features in BC's per-player view (self_numeric is 13 cols)
NUMERIC_COLS = [
    "pos_x", "pos_y", "percent", "stock", "jumps_left",
    "speed_air_x_self", "speed_ground_x_self", "speed_x_attack",
    "speed_y_attack", "speed_y_self", "hitlag_left", "hitstun_left",
    "shield_strength",
]
FLAG_COLS = ["on_ground", "off_stage", "facing", "invulnerable",
             "moonwalkwarning"]


def sample_frames(
    data_dir: Path,
    split: str,
    n_target: int,
    seed: int = 0,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Sample frames matching the NN training distribution.

    Returns:
        feats: dict of (N, ...) numpy arrays per shard key
        outcomes: (N,) float in {0.0, 0.5, 1.0}
        pos_in_game: (N,) float in [0, 1], frame_offset / game_length
    """
    with open(data_dir / "tensor_manifest.json") as f:
        manifest = json.load(f)
    shard_files = [data_dir / n for n in manifest[f"{split}_shards"]]
    rng = np.random.default_rng(seed)

    # Choose how many samples per shard (uniform-per-shard).
    n_per_shard = max(1, n_target // len(shard_files))

    # Collect samples
    out_self_num: List[np.ndarray] = []
    out_opp_num: List[np.ndarray] = []
    out_self_flg: List[np.ndarray] = []
    out_opp_flg: List[np.ndarray] = []
    out_self_action: List[np.ndarray] = []
    out_opp_action: List[np.ndarray] = []
    out_self_ae: List[np.ndarray] = []
    out_opp_ae: List[np.ndarray] = []
    out_outcomes: List[float] = []
    out_position: List[float] = []

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

        # Uniform-per-game × uniform-within-game (matches FoxValueWindowedDataset).
        for _ in range(n_per_shard):
            g = int(rng.integers(0, n_games))
            s, e, outcome = gi[g]
            length = e - s
            if length < 2:
                continue
            offset = int(rng.integers(0, length))
            absf = s + offset

            out_self_num.append(states["self_numeric"][absf].numpy())
            out_opp_num.append(states["opp_numeric"][absf].numpy())
            out_self_flg.append(states["self_flags"][absf].numpy())
            out_opp_flg.append(states["opp_flags"][absf].numpy())
            out_self_action.append(np.array([int(states["self_action"][absf].item())]))
            out_opp_action.append(np.array([int(states["opp_action"][absf].item())]))
            out_self_ae.append(np.array([float(states["self_action_elapsed"][absf].item())]))
            out_opp_ae.append(np.array([float(states["opp_action_elapsed"][absf].item())]))
            out_outcomes.append(outcome)
            out_position.append(offset / length)

    feats = {
        "self_numeric": np.stack(out_self_num),      # (N, 13)
        "opp_numeric": np.stack(out_opp_num),        # (N, 13)
        "self_flags": np.stack(out_self_flg),        # (N, 5)
        "opp_flags": np.stack(out_opp_flg),          # (N, 5)
        "self_action": np.concatenate(out_self_action),   # (N,)
        "opp_action": np.concatenate(out_opp_action),     # (N,)
        "self_action_elapsed": np.concatenate(out_self_ae),  # (N,)
        "opp_action_elapsed": np.concatenate(out_opp_ae),
    }
    return feats, np.array(out_outcomes, dtype=np.float64), \
           np.array(out_position, dtype=np.float64)


def fit_and_score(X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None,
                  y_val: np.ndarray = None, C: float = 1.0) -> Dict[str, float]:
    """Fit LR; return train BCE + val BCE (if val split given) + accuracy."""
    # Drop draws (rare, BCE undefined on {0.5} target with binary loss)
    keep = y != 0.5
    Xk, yk = X[keep], y[keep].astype(int)
    if X_val is None:
        # 80/20 split
        n = len(Xk)
        rng = np.random.default_rng(0)
        idx = rng.permutation(n)
        n_tr = int(n * 0.8)
        Xt, yt = Xk[idx[:n_tr]], yk[idx[:n_tr]]
        Xv, yv = Xk[idx[n_tr:]], yk[idx[n_tr:]]
    else:
        kv = y_val != 0.5
        Xt, yt = Xk, yk
        Xv, yv = X_val[kv], y_val[kv].astype(int)

    model = LogisticRegression(C=C, max_iter=1000, solver="lbfgs")
    model.fit(Xt, yt)

    train_bce = log_loss(yt, model.predict_proba(Xt)[:, 1], labels=[0, 1])
    val_bce = log_loss(yv, model.predict_proba(Xv)[:, 1], labels=[0, 1])
    val_acc = (model.predict(Xv) == yv).mean()

    return {
        "n_train": len(Xt), "n_val": len(Xv),
        "train_bce": float(train_bce),
        "val_bce": float(val_bce),
        "val_acc": float(val_acc),
        "model_coef": model.coef_[0],
        "model_intercept": float(model.intercept_[0]),
    }


def feature_list(feats: Dict[str, np.ndarray]) -> List[Tuple[str, np.ndarray]]:
    """Decompose feats into named single-scalar columns."""
    out: List[Tuple[str, np.ndarray]] = []
    for side in ("self", "opp"):
        for i, c in enumerate(NUMERIC_COLS):
            out.append((f"{side}_{c}", feats[f"{side}_numeric"][:, i]))
        for i, c in enumerate(FLAG_COLS):
            out.append((f"{side}_{c}", feats[f"{side}_flags"][:, i].astype(float) * 2.0 - 1.0))
        out.append((f"{side}_action_elapsed", feats[f"{side}_action_elapsed"]))
    return out


def stack_feat_array(feat_pairs: List[Tuple[str, np.ndarray]]) -> Tuple[np.ndarray, List[str]]:
    cols = [p[1].reshape(-1, 1) for p in feat_pairs]
    names = [p[0] for p in feat_pairs]
    X = np.concatenate(cols, axis=1)
    return X, names


def ceiling_trivial(feats, y) -> Dict[str, float]:
    """LR on {self_stock, opp_stock, self_percent, opp_percent} only."""
    X = np.stack([
        feats["self_numeric"][:, STOCK_COL],
        feats["opp_numeric"][:, STOCK_COL],
        feats["self_numeric"][:, PERCENT_COL],
        feats["opp_numeric"][:, PERCENT_COL],
    ], axis=1)
    r = fit_and_score(X, y)
    return {
        "name": "trivial (4 features: stocks + percent)",
        **{k: v for k, v in r.items() if not isinstance(v, np.ndarray)},
        "coefs": {n: float(c) for n, c in zip(
            ["self_stock", "opp_stock", "self_percent", "opp_percent"],
            r["model_coef"])},
        "intercept": r["model_intercept"],
    }


def ceiling_wide(feats, y) -> Dict[str, float]:
    """LR on all named per-player numeric + flag + action_elapsed scalars."""
    pairs = feature_list(feats)
    X, names = stack_feat_array(pairs)
    r = fit_and_score(X, y)
    coefs = sorted(zip(names, r["model_coef"]),
                   key=lambda x: -abs(x[1]))[:20]
    return {
        "name": f"wide ({X.shape[1]} features)",
        **{k: v for k, v in r.items() if not isinstance(v, np.ndarray)},
        "top_20_coefs_abs": [(n, float(c)) for n, c in coefs],
    }


def per_feature_ranking(feats, y) -> List[Dict[str, float]]:
    """Per single-feature LR, ranked by val BCE (lower = better predictor)."""
    pairs = feature_list(feats)
    results = []
    for name, col in pairs:
        X = col.reshape(-1, 1)
        try:
            r = fit_and_score(X, y)
        except Exception:
            continue
        results.append({
            "feature": name,
            "val_bce": r["val_bce"],
            "val_acc": r["val_acc"],
            "coef": float(r["model_coef"][0]),
            "intercept": r["model_intercept"],
        })
    results.sort(key=lambda x: x["val_bce"])
    return results


def per_position_breakdown(feats, y, position, n_bins: int = 10) -> List[Dict]:
    """Bin samples by frame_in_game_pct; report trivial-LR BCE per bin."""
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (position >= lo) & (position < hi if i < n_bins - 1 else position <= hi)
        if mask.sum() < 200:
            continue
        # Class balance in this bin
        non_draw = y[mask] != 0.5
        pos_rate = float((y[mask][non_draw] == 1.0).mean())
        # Trivial-features LR on this bin alone
        Xb = np.stack([
            feats["self_numeric"][mask, STOCK_COL],
            feats["opp_numeric"][mask, STOCK_COL],
            feats["self_numeric"][mask, PERCENT_COL],
            feats["opp_numeric"][mask, PERCENT_COL],
        ], axis=1)
        r = fit_and_score(Xb, y[mask])
        # Constant-prediction baseline (predict mean)
        p_const = float(y[mask][non_draw].mean())
        const_bce = -p_const * math.log(max(1e-9, p_const)) \
                    - (1 - p_const) * math.log(max(1e-9, 1 - p_const))
        rows.append({
            "bin": f"[{lo:.2f}, {hi:.2f})",
            "n": int(mask.sum()),
            "class_pos_rate": pos_rate,
            "const_bce": float(const_bce),
            "trivial_lr_bce": r["val_bce"],
            "trivial_lr_acc": r["val_acc"],
        })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/fox_all_v2")
    p.add_argument("--split", default="val")
    p.add_argument("--n-samples", type=int, default=200000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None,
                   help="JSON output path. If omitted, just print.")
    args = p.parse_args()

    print(f"sampling {args.n_samples} frames from {args.split} shards "
          f"({args.data_dir}, seed={args.seed})...")
    feats, y, pos = sample_frames(
        Path(args.data_dir), args.split, args.n_samples, args.seed,
    )
    print(f"got {len(y)} samples  "
          f"(wins: {(y==1.0).sum()}  losses: {(y==0.0).sum()}  draws: {(y==0.5).sum()})")
    print()

    print("=" * 70)
    print("1. Trivial-features ceiling")
    print("=" * 70)
    trivial = ceiling_trivial(feats, y)
    print(f"  val BCE: {trivial['val_bce']:.4f}   "
          f"val acc: {trivial['val_acc']:.3f}   "
          f"train BCE: {trivial['train_bce']:.4f}")
    print(f"  coefs: {trivial['coefs']}")
    print(f"  intercept: {trivial['intercept']:+.3f}")
    print()

    print("=" * 70)
    print("2. Wide-features ceiling (all named scalar features)")
    print("=" * 70)
    wide = ceiling_wide(feats, y)
    print(f"  val BCE: {wide['val_bce']:.4f}   "
          f"val acc: {wide['val_acc']:.3f}   "
          f"train BCE: {wide['train_bce']:.4f}   "
          f"n_features: {wide['name'].split('(')[1].split()[0]}")
    print(f"  top 20 |coef|:")
    for n, c in wide["top_20_coefs_abs"]:
        print(f"    {c:+.3f}  {n}")
    print()

    print("=" * 70)
    print("3. Per-feature ranking (single-feature LR)")
    print("=" * 70)
    per_feat = per_feature_ranking(feats, y)
    print(f"  top 20 features (lowest val BCE = best single predictor):")
    for r in per_feat[:20]:
        print(f"    BCE {r['val_bce']:.4f}  acc {r['val_acc']:.3f}  "
              f"coef {r['coef']:+.3f}  {r['feature']}")
    print(f"  bottom 5 features (highest val BCE = worst):")
    for r in per_feat[-5:]:
        print(f"    BCE {r['val_bce']:.4f}  acc {r['val_acc']:.3f}  "
              f"coef {r['coef']:+.3f}  {r['feature']}")
    print()

    print("=" * 70)
    print("4. Per-frame-position breakdown")
    print("=" * 70)
    per_pos = per_position_breakdown(feats, y, pos)
    print(f"  {'bin':<14} {'n':>7} {'pos_rate':>10} "
          f"{'const_BCE':>10} {'triv_BCE':>10} {'triv_acc':>10}")
    for r in per_pos:
        print(f"  {r['bin']:<14} {r['n']:>7} "
              f"{r['class_pos_rate']:>10.3f} "
              f"{r['const_bce']:>10.4f} "
              f"{r['trivial_lr_bce']:>10.4f} "
              f"{r['trivial_lr_acc']:>10.3f}")
    print()

    if args.out:
        payload = {
            "n_samples": len(y),
            "class_balance": {
                "wins": int((y == 1.0).sum()),
                "losses": int((y == 0.0).sum()),
                "draws": int((y == 0.5).sum()),
            },
            "trivial_ceiling": trivial,
            "wide_ceiling": wide,
            "per_feature_ranking": per_feat,
            "per_position_breakdown": per_pos,
        }
        Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
