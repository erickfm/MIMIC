"""Live DolphinActor smoke for ComboExtendOnlineTask.

Boots one Dolphin via libmelee, loads the production Fox BC checkpoint
as the policy (no training, no ref-model diff), drives a series of
matches against a CPU-level-9 Fox on Final Destination, collects
episodes via the combo_extend task, and prints reward distributions.

This is the live-actor counterpart to manual_smoke_combo_extend.py —
that one walks shard tensors offline through the task's state machine;
this one walks Dolphin's live state through the same machine. If both
produce similar episode-shape distributions, the actor wiring works.

Run:
    DISPLAY=:99 python -m rlvr.tests.manual_smoke_combo_extend_actor \\
        --base-ckpt hf_checkpoints/fox/model.pt \\
        --data-dir  hf_checkpoints/fox \\
        --n-episodes 30
"""
from __future__ import annotations

import argparse
import collections
import statistics
import sys
from pathlib import Path

import torch

from rlvr.online.dolphin_actor import ActorConfig, DolphinActor
from rlvr.online.tasks.combo_extend_online import ComboExtendOnlineTask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", required=True, type=Path)
    ap.add_argument("--data-dir", required=True, type=Path,
                    help="Directory containing cat_maps.json, controller_combos.json, "
                         "stick_clusters.json, mimic_norm.json (matches base_ckpt's config)")
    ap.add_argument("--dolphin-path",
                    default="emulator/squashfs-root/usr/bin/dolphin-emu",
                    type=Path)
    ap.add_argument("--iso-path", default="melee.iso", type=Path)
    ap.add_argument("--n-episodes", type=int, default=30)
    ap.add_argument("--cpu-character", default="FOX")
    ap.add_argument("--stage", default="FINAL_DESTINATION")
    ap.add_argument("--cpu-level", type=int, default=9)
    ap.add_argument("--gfx-backend", default="Vulkan",
                    help="GPU backend. Use 'Null' with --enable-ffw.")
    ap.add_argument("--use-exi-inputs", action="store_true",
                    help="EXI input injection (required for --enable-ffw)")
    ap.add_argument("--enable-ffw", action="store_true",
                    help="Run Dolphin unlimited-speed (needs Exi-AI build)")
    ap.add_argument("--replay-dir", type=Path,
                    default=Path("replays_online_smoke"))
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--self-port", type=int, default=1)
    args = ap.parse_args()

    from tools.inference_utils import load_inference_context, load_mimic_model

    print(f"loading BC checkpoint: {args.base_ckpt}")
    model, cfg = load_mimic_model(str(args.base_ckpt), args.device)
    # Use model as its own ref for smoke (no PPO update, no KL).
    ref_model = model
    print(f"  model_preset={getattr(cfg, 'model_preset', '?')}, "
          f"d_model={cfg.d_model}, max_seq_len={cfg.max_seq_len}, "
          f"n_combos={getattr(cfg, 'n_controller_combos', '?')}")

    print(f"loading inference context from: {args.data_dir}")
    ctx = load_inference_context(args.data_dir)

    task = ComboExtendOnlineTask(self_port=args.self_port)
    args.replay_dir.mkdir(parents=True, exist_ok=True)

    actor_cfg = ActorConfig(
        dolphin_path=str(args.dolphin_path),
        iso_path=str(args.iso_path),
        character="FOX",
        cpu_character=args.cpu_character,
        cpu_level=args.cpu_level,
        stage=args.stage,
        temperature=args.temperature,
        gfx_backend=args.gfx_backend,
        use_exi_inputs=args.use_exi_inputs,
        enable_ffw=args.enable_ffw,
        disable_audio=args.enable_ffw,
        replay_dir=str(args.replay_dir),
    )
    print(f"booting DolphinActor: stage={args.stage} cpu_level={args.cpu_level} "
          f"gfx={args.gfx_backend}")
    actor = DolphinActor(
        cfg=actor_cfg, task=task,
        model=model, ref_model=ref_model, ctx=ctx,
        device=args.device, model_seq_len=cfg.max_seq_len,
        self_port=args.self_port,
    )
    actor.start()
    print(f"actor started; collecting {args.n_episodes} episodes...")

    episodes = actor.collect(n_episodes=args.n_episodes)
    print(f"collected {len(episodes)} episodes")
    if not episodes:
        print("NO EPISODES collected. Likely actor wiring / Dolphin issue.")
        sys.exit(1)

    # Summarize
    rewards = [ep.terminal_reward for ep in episodes]
    lengths = [len(ep.frames) for ep in episodes]
    results = collections.Counter(ep.metadata.get("result", "?") for ep in episodes)

    nan_count = sum(1 for r in rewards if r != r)
    print()
    print(f"=== Summary ===")
    print(f"terminal_reward:")
    print(f"  min={min(rewards):.3f}  max={max(rewards):.3f}  "
          f"mean={statistics.mean(rewards):.3f}  NaN={nan_count}")
    n_zero = sum(1 for r in rewards if r == 0.0)
    n_small = sum(1 for r in rewards if 0 < r < 0.4)
    n_large = sum(1 for r in rewards if 0.4 <= r < 0.99)
    n_max = sum(1 for r in rewards if r >= 0.99)
    print(f"  bucket counts:")
    print(f"    reward = 0     : {n_zero}  (sub-threshold)")
    print(f"    0 < r < 0.4    : {n_small}")
    print(f"    0.4 <= r < 0.99: {n_large}")
    print(f"    r >= 0.99      : {n_max}  (stock confirms + 100%+ damage)")
    print()
    print(f"episode lengths (frames):")
    print(f"  min={min(lengths)}  median={statistics.median(lengths)}  "
          f"max={max(lengths)}  mean={statistics.mean(lengths):.1f}")
    print()
    print(f"result types:")
    for k, v in results.most_common():
        print(f"  {k}: {v}")

    # Sanity gates
    print()
    print(f"=== Pass criteria ===")
    p1 = nan_count == 0
    p2 = max(lengths) < 600
    p3 = min(lengths) >= 1
    p4 = len(episodes) >= 5  # at least some episodes collected
    print(f"  no NaN rewards: {'PASS' if p1 else 'FAIL'}")
    print(f"  no runaway episodes: {'PASS' if p2 else f'FAIL (max={max(lengths)})'}")
    print(f"  no zero-length: {'PASS' if p3 else f'FAIL (min={min(lengths)})'}")
    print(f"  at least 5 episodes: {'PASS' if p4 else f'FAIL ({len(episodes)})'}")


if __name__ == "__main__":
    main()
