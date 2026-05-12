#!/usr/bin/env python3
"""Visualize WM rollouts vs ground truth as a side-by-side animation.

Minimal Melee renderer — draws stage outline + two characters as colored
dots with facing triangles, action state label, percent, stock. Enough
to eyeball whether the WM's predictions look physically plausible or
drift visibly into cursedness.

Usage:
    python3 tools/wm_viz.py \\
        --checkpoint checkpoints/fox-wm-20260424-baseline-32k.pt \\
        --data-dir data/fox_all_v2 \\
        --context-len 180 --rollout-frames 120 \\
        --output reports/wm_viz.mp4
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch
from melee.enums import Action, Character, Stage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mimic.world_model import WorldModel
from mimic.model import ModelConfig
from tools.wm_rollout import load_model, STATE_INPUT_KEYS


# ---------------------------------------------------------------------------
# Stage geometry (Melee world coordinates, main platform at y=0)
# Values from SSBM internal data; y=0 is the ground, blastzones vary.
# We draw only the platforms + a dashed blastzone box. Good enough for
# visual sanity, not combat-ready.
# ---------------------------------------------------------------------------
STAGE_GEOM = {
    "FINAL_DESTINATION": {
        "main": (-85.56, 85.56, 0.0),
        "platforms": [],
        "blast": (-246, 246, -140, 188),
    },
    "BATTLEFIELD": {
        "main": (-68.4, 68.4, 0.0),
        "platforms": [(-57.6, -20.0, 27.2), (20.0, 57.6, 27.2), (-18.8, 18.8, 54.4)],
        "blast": (-224, 224, -108, 200),
    },
    "DREAMLAND": {
        "main": (-77.3, 77.3, 0.0),
        "platforms": [(-61.4, -31.7, 30.14), (31.7, 61.4, 30.14), (-16.2, 16.2, 51.4)],
        "blast": (-255, 255, -123, 250),
    },
    "YOSHIS_STORY": {
        "main": (-54.1, 54.1, 0.0),
        "platforms": [(-59.4, -28.0, 23.5), (28.0, 59.4, 23.5), (-15.75, 15.75, 42.0)],
        "blast": (-175, 173, -91, 168),
    },
    "POKEMON_STADIUM": {
        "main": (-87.75, 87.75, 0.0),
        "platforms": [(-55, -25, 25), (25, 55, 25)],
        "blast": (-230, 230, -111, 180),
    },
    "FOUNTAIN_OF_DREAMS": {
        "main": (-63.4, 63.4, 0.0),
        "platforms": [(-49.5, -21, 16), (21, 49.5, 16), (-14.25, 14.25, 42.0)],
        "blast": (-200, 200, -147, 200),
    },
}


def draw_stage(ax, stage_enum_name: str, title: str) -> None:
    ax.clear()
    geom = STAGE_GEOM.get(stage_enum_name)
    if geom is None:
        ax.set_xlim(-250, 250)
        ax.set_ylim(-150, 200)
        ax.text(0, 0, f"unknown stage {stage_enum_name}", ha="center")
    else:
        bx0, bx1, by0, by1 = geom["blast"]
        ax.add_patch(plt.Rectangle((bx0, by0), bx1 - bx0, by1 - by0,
                                   fill=False, ls="--", ec="gray", lw=0.5))
        # Main platform
        x0, x1, y = geom["main"]
        ax.plot([x0, x1], [y, y], "k-", lw=4)
        # Floating platforms
        for px0, px1, py in geom["platforms"]:
            ax.plot([px0, px1], [py, py], "k-", lw=2)
        ax.set_xlim(bx0 * 1.05, bx1 * 1.05)
        ax.set_ylim(by0 * 1.05, by1 * 1.1)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.2)


def draw_char(ax, x, y, facing, color, label_text):
    """Draw a character as a colored circle with a facing triangle."""
    ax.scatter([x], [y], c=color, s=180, zorder=5, edgecolors="black", lw=0.5)
    # Facing triangle (small arrow next to the dot)
    dx = 6 if facing > 0 else -6
    ax.annotate("", xy=(x + dx, y), xytext=(x, y),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
                zorder=4)
    ax.text(x, y + 11, label_text, ha="center", va="bottom",
            color=color, fontsize=7, zorder=6,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1))


# ---------------------------------------------------------------------------
# De-normalization (z-score for pos) and enum lookups.
# ---------------------------------------------------------------------------
ACTION_NAMES = {a.value: a.name for a in Action}
CHAR_NAMES = {c.value: c.name for c in Character}
# Shard stage enum → name. The STAGE_MAP in cat_maps.py is raw→compact
# with NO_STAGE=0 and 1..6 as tournament stages. Invert it.
from mimic.cat_maps import STAGE_MAP
_SHARD_STAGE_TO_ENUM = {compact: raw for raw, compact in STAGE_MAP.items()}
_ENUM_TO_NAME = {s.value: s.name for s in Stage}


def shard_stage_to_name(shard_stage: int) -> str:
    raw = _SHARD_STAGE_TO_ENUM.get(int(shard_stage), 25)
    return _ENUM_TO_NAME.get(raw, f"stage_{raw}")


def load_norm(data_dir: Path) -> Dict:
    """Load normalization params — pos_x/pos_y transforms for de-norming."""
    for name in ("hal_norm.json", "mimic_norm.json"):
        p = data_dir / name
        if p.exists():
            with open(p) as fh:
                return json.load(fh)["features"]
    raise RuntimeError(f"no hal_norm / mimic_norm in {data_dir}")


def denorm_pos(x_norm: float, params: Dict) -> float:
    """De-normalize one axis. Handles standardize (the common case)."""
    t = params["transform"]
    if t == "standardize":
        return x_norm * params["std"] + params["mean"]
    if t == "normalize":
        rng = params["max"] - params["min"]
        return (x_norm + 1) * rng / 2 + params["min"]
    return x_norm  # shouldn't happen for pos


# ---------------------------------------------------------------------------
# Rollout (reuses the core from wm_rollout.py but also returns the full
# per-step trajectory, not just summary stats).
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_rollout_and_gt(
    model: WorldModel,
    states: dict,
    start: int,
    ctx: int,
    horizon: int,
    device: torch.device,
) -> Tuple[List[dict], List[dict]]:
    """Return (gt_frames, pred_frames), each a list of dicts per rollout step
    with keys: stage_name, chars, p1_pos, p1_action, p1_facing, p1_percent,
    p1_stock, p2_*. Positions are in NORMALIZED shard coords.
    """
    t0 = start
    t1 = start + ctx
    window: Dict[str, torch.Tensor] = {}
    want_opp_ctrl = getattr(model.encoder, "_include_opp_ctrl", False)
    keys = list(STATE_INPUT_KEYS)
    if want_opp_ctrl and "opp_controller" in states:
        keys.append("opp_controller")
    for k in keys:
        window[k] = states[k][t0:t1].clone().to(device)

    gt_frames: List[dict] = []
    pred_frames: List[dict] = []

    # Keep the denorm-able views of the full window so the starting frame is
    # identical between gt and pred (rollout step 0).
    for step in range(horizon):
        tgt_idx = t1 + step  # absolute frame we're predicting into
        # Conditioning: real controllers at tgt_idx
        next_self_ctrl = states["self_controller"][tgt_idx].to(device)
        next_opp_btns = states["opp_buttons"][tgt_idx].to(device)
        next_opp_analog = states["opp_analog"][tgt_idx].to(device)
        next_opp_cdir = states["opp_c_dir"][tgt_idx].to(device)

        frames = {k: v.unsqueeze(0) for k, v in window.items()}
        T = frames["stage"].shape[1]
        frames["next_self_controller"] = next_self_ctrl.expand(1, T, -1).clone()
        frames["next_opp_buttons"] = next_opp_btns.expand(1, T, -1).clone()
        frames["next_opp_analog"] = next_opp_analog.expand(1, T, -1).clone()
        frames["next_opp_c_dir"] = next_opp_cdir.expand(1, T).clone()

        preds = model(frames)
        pred_last = {k: v[0, -1] for k, v in preds.items()}
        pred_self_action = pred_last["self_action_logits"].argmax().item()
        pred_opp_action = pred_last["opp_action_logits"].argmax().item()
        pred_self_num = pred_last["self_numeric_pred"]
        pred_opp_num = pred_last["opp_numeric_pred"]
        pred_self_flags = (pred_last["self_flags_logits"] > 0).float()
        pred_opp_flags = (pred_last["opp_flags_logits"] > 0).float()

        # GT for this frame (index tgt_idx in the shard).
        stage_shard = int(states["stage"][tgt_idx].item())
        p1_char = int(states["self_character"][tgt_idx].item())
        p2_char = int(states["opp_character"][tgt_idx].item())

        gt_sn = states["self_numeric"][tgt_idx]
        gt_on = states["opp_numeric"][tgt_idx]
        gt_sf = states["self_flags"][tgt_idx]
        gt_of = states["opp_flags"][tgt_idx]
        gt_frames.append(dict(
            step=step, stage_shard=stage_shard,
            p1_char=p1_char, p2_char=p2_char,
            p1_pos=(gt_sn[0].item(), gt_sn[1].item()),
            p1_action=int(states["self_action"][tgt_idx].item()),
            p1_facing=float(gt_sf[2].item()),
            p1_percent_norm=float(gt_sn[2].item()),
            p1_stock_norm=float(gt_sn[3].item()),
            p2_pos=(gt_on[0].item(), gt_on[1].item()),
            p2_action=int(states["opp_action"][tgt_idx].item()),
            p2_facing=float(gt_of[2].item()),
            p2_percent_norm=float(gt_on[2].item()),
            p2_stock_norm=float(gt_on[3].item()),
        ))
        pred_frames.append(dict(
            step=step, stage_shard=stage_shard,
            p1_char=p1_char, p2_char=p2_char,
            p1_pos=(pred_self_num[0].item(), pred_self_num[1].item()),
            p1_action=pred_self_action,
            p1_facing=float(pred_self_flags[2].item()),
            p1_percent_norm=float(pred_self_num[2].item()),
            p1_stock_norm=float(pred_self_num[3].item()),
            p2_pos=(pred_opp_num[0].item(), pred_opp_num[1].item()),
            p2_action=pred_opp_action,
            p2_facing=float(pred_opp_flags[2].item()),
            p2_percent_norm=float(pred_opp_num[2].item()),
            p2_stock_norm=float(pred_opp_num[3].item()),
        ))

        # Push prediction back into the window for the next step.
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
            "self_controller": next_self_ctrl,
        }
        if want_opp_ctrl:
            new_frame["opp_controller"] = states["opp_controller"][tgt_idx].to(device)
        for k, buf in window.items():
            buf[:-1] = buf[1:].clone()
            buf[-1] = new_frame[k]

    return gt_frames, pred_frames


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def render_frame(ax, frame: dict, norm: Dict, title: str) -> None:
    stage_name = shard_stage_to_name(frame["stage_shard"])
    draw_stage(ax, stage_name, title)

    p1_x = denorm_pos(frame["p1_pos"][0], norm["pos_x"])
    p1_y = denorm_pos(frame["p1_pos"][1], norm["pos_y"])
    p2_x = denorm_pos(frame["p2_pos"][0], norm["pos_x"])
    p2_y = denorm_pos(frame["p2_pos"][1], norm["pos_y"])

    # Denormalize percent/stock from the normalize transform in hal_norm.
    # percent: min=0, max=236 → raw = (norm+1)/2 * 236
    # stock:   min=0, max=4   → raw = (norm+1)/2 * 4
    p1_pct = int(max(0, (frame["p1_percent_norm"] + 1) * 0.5 * 236))
    p2_pct = int(max(0, (frame["p2_percent_norm"] + 1) * 0.5 * 236))
    p1_stk = int(round(max(0, (frame["p1_stock_norm"] + 1) * 0.5 * 4)))
    p2_stk = int(round(max(0, (frame["p2_stock_norm"] + 1) * 0.5 * 4)))
    p1_ch = CHAR_NAMES.get(frame["p1_char"], f"c{frame['p1_char']}")
    p2_ch = CHAR_NAMES.get(frame["p2_char"], f"c{frame['p2_char']}")
    p1_act = ACTION_NAMES.get(frame["p1_action"], f"#{frame['p1_action']}")
    p2_act = ACTION_NAMES.get(frame["p2_action"], f"#{frame['p2_action']}")

    draw_char(ax, p1_x, p1_y, frame["p1_facing"], "tab:blue",
              f"{p1_ch}\n{p1_act}\n{p1_pct}% × {p1_stk}")
    draw_char(ax, p2_x, p2_y, frame["p2_facing"], "tab:red",
              f"{p2_ch}\n{p2_act}\n{p2_pct}% × {p2_stk}")


def pick_game(data_dir: Path, rng: random.Random,
              min_frames: int) -> Tuple[Path, int, int]:
    """Pick a shard + (game_start, game_end) with enough frames."""
    shards = sorted(data_dir.glob("val_shard_*.pt"))
    if not shards:
        shards = sorted(data_dir.glob("train_shard_*.pt"))[:10]
    rng.shuffle(shards)
    for path in shards:
        shard = torch.load(path, weights_only=True, mmap=True)
        offsets = shard["offsets"]
        n = shard["n_games"]
        games = [(offsets[g].item(), offsets[g + 1].item())
                 for g in range(n)]
        games = [(s, e) for s, e in games if e - s >= min_frames]
        if games:
            gs, ge = rng.choice(games)
            return path, gs, ge
    raise RuntimeError("no game long enough")




# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--context-len", type=int, default=180)
    ap.add_argument("--rollout-frames", type=int, default=120)
    ap.add_argument("--output", default=None, help="MP4 path (default: reports/...)")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    data_dir = Path(args.data_dir)
    norm = load_norm(data_dir)

    print(f"loading {args.checkpoint}")
    model, cfg, device = load_model(Path(args.checkpoint))
    print(f"  params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    need = args.context_len + args.rollout_frames + 10
    path, gs, ge = pick_game(data_dir, rng, need)
    print(f"game: {path.name} [{gs}..{ge}] ({ge - gs} frames)")
    shard = torch.load(path, weights_only=True, mmap=True)
    states = shard["states"]

    start = gs + rng.randint(0, (ge - gs) - need)
    print(f"rollout from abs frame {start} (game-relative {start - gs})")

    print("running rollout...")
    gt, pred = run_rollout_and_gt(
        model, states, start=start,
        ctx=args.context_len, horizon=args.rollout_frames,
        device=device,
    )

    # --- Render ---
    out_path = Path(args.output or f"reports/wm_viz_{Path(args.checkpoint).stem}.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    def update(i: int):
        render_frame(axes[0], gt[i], norm, f"ground truth  (K={i + 1})")
        render_frame(axes[1], pred[i], norm, f"WM prediction  (K={i + 1})")
        # Mark drift arrow between GT and pred positions (both panels)
        for ax in axes:
            ax.plot([denorm_pos(gt[i]["p1_pos"][0], norm["pos_x"]),
                     denorm_pos(pred[i]["p1_pos"][0], norm["pos_x"])],
                    [denorm_pos(gt[i]["p1_pos"][1], norm["pos_y"]),
                     denorm_pos(pred[i]["p1_pos"][1], norm["pos_y"])],
                    "b:", alpha=0.3, lw=0.8)
        return []

    ani = animation.FuncAnimation(
        fig, update, frames=args.rollout_frames, interval=1000 // args.fps,
        blit=False,
    )

    print(f"writing {out_path} ({args.fps} fps, {args.rollout_frames} frames)...")
    writer = animation.FFMpegWriter(fps=args.fps, bitrate=2400)
    ani.save(str(out_path), writer=writer)
    print(f"done: {out_path}")


if __name__ == "__main__":
    main()
