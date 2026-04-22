#!/usr/bin/env python3
"""World-model alignment audit.

Loads a per-game shard and prints state[i], controllers[i+1], state[i+1]
for a handful of frames where self_action actually changes. Goal: confirm
that `target_state = states[k][i+1]` gives the right causal alignment for
world-model training (no extra shift needed, because v2 shards already
store post-frame gamestate aligned with the controller that produced it,
so shifting input-side by +1 step gives us "what buttons produced the
next state").

Usage:
    python tools/wm_audit.py --shard data/fox_master_v2/train_shard_000.pt
    python tools/wm_audit.py --data-dir data/fox_master_v2 --n-examples 8
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from melee.enums import Action

IDX_TO_ACTION = {i: a.name for i, a in enumerate(Action)}


def action_name(idx: int) -> str:
    return IDX_TO_ACTION.get(idx, f"#{idx}")


def decode_ctrl56(ctrl: torch.Tensor) -> str:
    """Decode the 56-dim one-hot: 37 main + 9 cstick + 7 combo + 3 shoulder.

    Returns short summary like 'main=23 cstick=0 combo=JUMP shoulder=0'.
    """
    main = ctrl[:37].argmax().item()
    cstick = ctrl[37:46].argmax().item()
    combo = ctrl[46:53].argmax().item()
    shoulder = ctrl[53:56].argmax().item()
    combo_names = ["A", "B", "Z", "JUMP", "TRIG", "A_TRIG", "NONE"]
    return (f"main={main:2d} cstick={cstick} "
            f"combo={combo_names[combo]:<6s} shoulder={shoulder}")


def opp_ctrl_summary(
    buttons: torch.Tensor, analog: torch.Tensor, cdir: int
) -> str:
    btn_active = buttons.nonzero(as_tuple=True)[0].tolist()
    main_x, main_y, lshld, rshld = analog.tolist()
    return (f"btns={btn_active} main=({main_x:+.2f},{main_y:+.2f}) "
            f"shld=({lshld:.2f},{rshld:.2f}) cdir={cdir}")


def pick_shard(data_dir: Path) -> Path:
    shards = sorted(data_dir.glob("train_shard_*.pt"))
    if not shards:
        raise SystemExit(f"no train shards in {data_dir}")
    return random.choice(shards)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=Path,
                    help="exact shard .pt (overrides --data-dir)")
    ap.add_argument("--data-dir", type=Path, default=Path("data/fox_master_v2"))
    ap.add_argument("--n-examples", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    shard_path = args.shard or pick_shard(args.data_dir)
    print(f"loading {shard_path}")
    shard = torch.load(shard_path, weights_only=False, mmap=True)
    states = shard["states"]
    offsets = shard["offsets"]
    n_games = shard["n_games"]

    # Pick a random long-enough game
    candidates = []
    for g in range(n_games):
        start = offsets[g].item()
        end = offsets[g + 1].item()
        if end - start > 500:
            candidates.append((start, end))
    if not candidates:
        raise SystemExit("no games long enough for audit")
    game_start, game_end = random.choice(candidates)
    print(f"game {game_start}..{game_end} ({game_end - game_start} frames)")

    # Pre-compute action changes within this game
    self_act = states["self_action"][game_start:game_end]
    opp_act = states["opp_action"][game_start:game_end]
    changes = (self_act[1:] != self_act[:-1]).nonzero(as_tuple=True)[0].tolist()
    if len(changes) < args.n_examples:
        print(f"WARN: only {len(changes)} self_action changes in this game")
    random.shuffle(changes)

    picked = changes[: args.n_examples]
    for k, local_i in enumerate(picked):
        i = game_start + local_i   # absolute frame index (post-frame state i)
        print()
        print(f"=== example {k + 1}/{len(picked)}  (abs {i} → {i + 1}) ===")
        # State at i
        sa_i = states["self_action"][i].item()
        oa_i = states["opp_action"][i].item()
        pos_i = states["self_numeric"][i][:2].tolist()   # pos_x, pos_y
        sf_i = states["self_flags"][i].tolist()
        print(f"  state[i]:   self_action={action_name(sa_i)}  opp_action={action_name(oa_i)}")
        print(f"              self_pos=({pos_i[0]:+.3f},{pos_i[1]:+.3f})  "
              f"self_flags(on_grd,off_stg,facing,inv,mw)={sf_i}")
        # Conditioning at i+1  (what's pressed between post-frame i and post-frame i+1)
        ctrl_i1 = states["self_controller"][i + 1]
        opp_b_i1 = states["opp_buttons"][i + 1]
        opp_a_i1 = states["opp_analog"][i + 1]
        opp_cd_i1 = states["opp_c_dir"][i + 1].item()
        print(f"  ctrl[i+1] (WM input):")
        print(f"    self: {decode_ctrl56(ctrl_i1)}")
        print(f"    opp:  {opp_ctrl_summary(opp_b_i1, opp_a_i1, opp_cd_i1)}")
        # State at i+1 (prediction target)
        sa_i1 = states["self_action"][i + 1].item()
        oa_i1 = states["opp_action"][i + 1].item()
        pos_i1 = states["self_numeric"][i + 1][:2].tolist()
        sf_i1 = states["self_flags"][i + 1].tolist()
        d_pos = (pos_i1[0] - pos_i[0], pos_i1[1] - pos_i[1])
        d_action = "SAME" if sa_i == sa_i1 else f"{action_name(sa_i)} → {action_name(sa_i1)}"
        print(f"  state[i+1] (target):  self_action={action_name(sa_i1)}  "
              f"opp_action={action_name(oa_i1)}")
        print(f"                         self_pos=({pos_i1[0]:+.3f},{pos_i1[1]:+.3f})  "
              f"delta=({d_pos[0]:+.4f},{d_pos[1]:+.4f})")
        print(f"  self_action change: {d_action}")

    print()
    print("Audit done. Check that each action transition is consistent with the")
    print("ctrl[i+1] conditioning (e.g., combo=JUMP should cause action →")
    print("KNEE_BEND or a jumpsquat variant on the next frame).")


if __name__ == "__main__":
    main()
