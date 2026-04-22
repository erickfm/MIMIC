#!/usr/bin/env python3
"""K-step autoregressive rollout evaluation for the world model.

Starting from a real state at frame `t`, feed the model its own predicted
state back in as input for step t+1, t+2, ... while using the *real*
controllers from both players as conditioning (teacher-forced on actions,
not state). This is the canonical measure of how well the WM can replace
Dolphin for RL rollouts.

Metrics at K ∈ {1, 8, 60, 180}:
- position L2 drift (in normalized units; de-normalize if needed downstream)
- self_action top-1 agreement
- opp_action top-1 agreement
- self_flags agreement (fraction of 5 flags matched)
- "diverged" rate (on_ground flipped permanently by K)

Usage:
    python tools/wm_rollout.py --checkpoint checkpoints/fox-wm-*_best.pt \
                               --data-dir data/fox_all_v2 \
                               --n-starts 200 --ks 1,8,60,180
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mimic.model import ModelConfig
from mimic.world_model import WorldModel


STATE_INPUT_KEYS = (
    "stage", "self_character", "opp_character",
    "self_action", "opp_action",
    "self_numeric", "opp_numeric",
    "self_flags", "opp_flags",
    "self_controller",
)
NEXT_CTRL_KEYS_SHARD = (
    "self_controller",  # → next_self_controller
    "opp_buttons",       # → next_opp_buttons
    "opp_analog",        # → next_opp_analog
    "opp_c_dir",         # → next_opp_c_dir
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--ks", default="1,8,60,180",
                    help="Comma-separated rollout horizons to measure.")
    ap.add_argument("--n-starts", type=int, default=200,
                    help="Number of random start frames to sample.")
    ap.add_argument("--context-len", type=int, default=180,
                    help="Size of rolling history window fed to the model.")
    ap.add_argument("--output", default=None,
                    help="JSON output path (default: reports/wm_rollout_*.json).")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def load_model(ckpt_path: Path) -> Tuple[WorldModel, ModelConfig, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = dict(ckpt["config"])
    cfg_dict.pop("wm_mode", None)  # re-set below
    cfg = ModelConfig(**{k: v for k, v in cfg_dict.items()
                         if k in ModelConfig.__dataclass_fields__})
    cfg.wm_mode = True
    model = WorldModel(cfg).to(device)
    sd = ckpt["model_state_dict"]
    # Strip torch.compile's _orig_mod prefix if present.
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model, cfg, device


def sample_starts(data_dir: Path, n_starts: int, ctx: int, horizon_max: int,
                  seed: int) -> List[Tuple[Path, int, int]]:
    """Pick random (shard, game_start, offset_in_game) tuples."""
    rng = random.Random(seed)
    shards = sorted(data_dir.glob("val_shard_*.pt"))
    if not shards:
        shards = sorted(data_dir.glob("train_shard_*.pt"))[:5]
    picks = []
    need = ctx + horizon_max + 1
    while len(picks) < n_starts:
        path = rng.choice(shards)
        shard = torch.load(path, weights_only=True, mmap=True)
        offsets = shard["offsets"]
        n_games = shard["n_games"]
        for _ in range(20):
            g = rng.randrange(n_games)
            g_start = offsets[g].item()
            g_end = offsets[g + 1].item()
            room = (g_end - g_start) - need
            if room <= 0:
                continue
            off = rng.randint(0, room)
            picks.append((path, g_start, off))
            if len(picks) >= n_starts:
                break
    return picks


def state_slice(states: dict, t_abs: int) -> Dict[str, torch.Tensor]:
    """Extract model-input fields at absolute frame t (no batch/seq dims)."""
    out = {}
    for k in STATE_INPUT_KEYS:
        out[k] = states[k][t_abs].clone()
    for k in NEXT_CTRL_KEYS_SHARD:
        out[k] = states[k][t_abs].clone()  # real opp ctrl at this frame
    return out


def push_window(window: Dict[str, torch.Tensor], new_frame: Dict[str, torch.Tensor]) -> None:
    """Roll window left by one; place new_frame at the last position. In-place."""
    for k, buf in window.items():
        buf[:-1] = buf[1:].clone()
        buf[-1] = new_frame[k]


@torch.no_grad()
def rollout_one(
    model: WorldModel,
    states: dict,
    start: int,
    ctx: int,
    horizons: List[int],
    device: torch.device,
) -> Dict[int, Dict[str, float]]:
    """Run one rollout from `start` for max(horizons) steps.

    Returns per-K metrics:
      pos_err (L2), action_self_eq, action_opp_eq, flags_self_eq
    """
    H = max(horizons)
    needed = ctx + H + 1

    # Seed the history window with real state up through frame `start`.
    window: Dict[str, torch.Tensor] = {}
    t0 = start
    t1 = start + ctx   # end-exclusive: [t0, t1) = ctx frames of real history
    for k in STATE_INPUT_KEYS:
        window[k] = states[k][t0:t1].clone().to(device)

    # Rolling rollout. At step i (1-indexed), we predict state[t1 + i - 1 → t1 + i].
    # Conditioning controller at that position is the real one from the shard.
    per_k_stats: Dict[int, Dict[str, float]] = {}
    horizon_set = set(horizons)

    for step in range(1, H + 1):
        tgt_idx = t1 + step - 1  # the frame we're predicting into
        # Build conditioning at t+1 from real shard.
        next_self_ctrl = states["self_controller"][tgt_idx].to(device)
        next_opp_btns = states["opp_buttons"][tgt_idx].to(device)
        next_opp_analog = states["opp_analog"][tgt_idx].to(device)
        next_opp_cdir = states["opp_c_dir"][tgt_idx].to(device)

        # Forward: (B=1, T=ctx, ...)
        frames = {k: v.unsqueeze(0) for k, v in window.items()}
        T = frames["stage"].shape[1]
        frames["next_self_controller"] = next_self_ctrl.expand(1, T, -1).clone()
        frames["next_opp_buttons"] = next_opp_btns.expand(1, T, -1).clone()
        frames["next_opp_analog"] = next_opp_analog.expand(1, T, -1).clone()
        frames["next_opp_c_dir"] = next_opp_cdir.expand(1, T).clone()

        preds = model(frames)

        # Use last-position prediction as the new frame.
        pred_last = {k: v[0, -1] for k, v in preds.items()}
        pred_self_action = pred_last["self_action_logits"].argmax().item()
        pred_opp_action = pred_last["opp_action_logits"].argmax().item()
        pred_self_num = pred_last["self_numeric_pred"]
        pred_opp_num = pred_last["opp_numeric_pred"]
        pred_self_flags = (pred_last["self_flags_logits"] > 0).float()
        pred_opp_flags = (pred_last["opp_flags_logits"] > 0).float()

        # Fabricate the new frame to roll into the window (using predicted state
        # but keeping static fields and the next-frame self_controller as real).
        new_frame = {
            "stage": states["stage"][tgt_idx].to(device),
            "self_character": states["self_character"][tgt_idx].to(device),
            "opp_character": states["opp_character"][tgt_idx].to(device),
            "self_action": torch.tensor(pred_self_action, dtype=torch.long, device=device),
            "opp_action": torch.tensor(pred_opp_action, dtype=torch.long, device=device),
            "self_numeric": pred_self_num,
            "opp_numeric": pred_opp_num,
            "self_flags": pred_self_flags,
            "opp_flags": pred_opp_flags,
            # In the window we feed the self_controller at *current* position —
            # that's the controller that produced the state we just predicted.
            "self_controller": next_self_ctrl,
        }
        push_window(window, new_frame)

        if step in horizon_set:
            real = {
                "self_action": states["self_action"][tgt_idx].item(),
                "opp_action": states["opp_action"][tgt_idx].item(),
                "self_numeric": states["self_numeric"][tgt_idx].to(device),
                "self_flags": states["self_flags"][tgt_idx].to(device),
            }
            # position error uses first two numeric dims (pos_x, pos_y) in
            # normalized space.
            pos_err = torch.linalg.vector_norm(
                pred_self_num[:2] - real["self_numeric"][:2]
            ).item()
            per_k_stats[step] = {
                "pos_err": pos_err,
                "action_self_eq": float(pred_self_action == real["self_action"]),
                "action_opp_eq": float(pred_opp_action == real["opp_action"]),
                "flags_self_eq": (pred_self_flags == real["self_flags"]).float().mean().item(),
            }

    return per_k_stats


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    ks = sorted(int(x) for x in args.ks.split(","))
    data_dir = Path(args.data_dir)
    ckpt_path = Path(args.checkpoint)
    out_path = Path(args.output) if args.output else (
        Path("reports") / f"wm_rollout_{ckpt_path.stem}.json"
    )

    print(f"loading {ckpt_path}")
    model, cfg, device = load_model(ckpt_path)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  cfg: d_model={cfg.d_model}, layers={cfg.num_layers}, "
          f"params={n_params / 1e6:.1f}M")

    print(f"sampling {args.n_starts} start points ...")
    picks = sample_starts(data_dir, args.n_starts, args.context_len,
                          max(ks), args.seed)

    per_k_accum: Dict[int, Dict[str, List[float]]] = {k: defaultdict(list) for k in ks}
    for i, (path, g_start, off) in enumerate(picks):
        shard = torch.load(path, weights_only=True, mmap=True)
        states = shard["states"]
        frame_stats = rollout_one(
            model, states, start=g_start + off,
            ctx=args.context_len, horizons=ks, device=device,
        )
        for k, d in frame_stats.items():
            for name, val in d.items():
                per_k_accum[k][name].append(val)
        if (i + 1) % 25 == 0:
            print(f"  rolled {i + 1}/{len(picks)} starts")

    # Aggregate.
    report = {"checkpoint": str(ckpt_path), "n_starts": len(picks), "by_k": {}}
    for k in ks:
        agg = {}
        for name, vals in per_k_accum[k].items():
            t = torch.tensor(vals)
            agg[f"{name}_mean"] = t.mean().item()
            agg[f"{name}_median"] = t.median().item()
        report["by_k"][str(k)] = agg
        print(f"K={k:3d}  "
              f"pos_err={agg['pos_err_mean']:.4f} (med {agg['pos_err_median']:.4f})  "
              f"act_self={agg['action_self_eq_mean']:.3f}  "
              f"act_opp={agg['action_opp_eq_mean']:.3f}  "
              f"flags={agg['flags_self_eq_mean']:.3f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
