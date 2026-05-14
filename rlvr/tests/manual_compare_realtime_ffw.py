"""Side-by-side timing: realtime vs FFW actor collection.

Boots a DolphinActor in two configurations against the same task and
the same target episode count, then prints frames/sec, episodes/sec,
and the realtime/FFW speedup ratio.

Run:
    DISPLAY=:99 python -m rlvr.tests.manual_compare_realtime_ffw \\
        --base-ckpt hf_checkpoints/fox/model.pt \\
        --data-dir hf_checkpoints/fox \\
        --n-episodes 10
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from rlvr.online.dolphin_actor import ActorConfig, DolphinActor
from rlvr.online.tasks.combo_extend_online import ComboExtendOnlineTask


def collect_and_time(
    cfg: ActorConfig,
    model,
    ref_model,
    ctx,
    device: str,
    model_seq_len: int,
    n_episodes: int,
    label: str,
):
    task = ComboExtendOnlineTask(self_port=1)
    Path(cfg.replay_dir).mkdir(parents=True, exist_ok=True)
    actor = DolphinActor(
        cfg=cfg, task=task,
        model=model, ref_model=ref_model, ctx=ctx,
        device=device, model_seq_len=model_seq_len,
        self_port=1,
    )

    boot_start = time.time()
    actor.start()
    boot_time = time.time() - boot_start
    print(f"[{label}] booted in {boot_time:.1f}s; collecting {n_episodes}...")

    collect_start = time.time()
    episodes = actor.collect(n_episodes=n_episodes)
    collect_time = time.time() - collect_start

    actor.stop()

    total_frames = sum(len(e.frames) for e in episodes)
    n = len(episodes)
    eps_per_sec = n / collect_time if collect_time > 0 else 0.0
    frames_per_sec = total_frames / collect_time if collect_time > 0 else 0.0

    print(f"[{label}] collected={n}  total_frames={total_frames}  "
          f"wall={collect_time:.1f}s  fps={frames_per_sec:.0f}  "
          f"ep/s={eps_per_sec:.2f}")
    return {
        "label": label,
        "boot_time": boot_time,
        "collect_time": collect_time,
        "n_episodes": n,
        "total_frames": total_frames,
        "eps_per_sec": eps_per_sec,
        "frames_per_sec": frames_per_sec,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", required=True, type=Path)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--realtime-dolphin",
                    default="emulator/squashfs-root/usr/bin/dolphin-emu",
                    type=Path)
    ap.add_argument("--ffw-dolphin",
                    default="emulator_ffw/squashfs-root/usr/bin/dolphin-emu",
                    type=Path)
    ap.add_argument("--iso-path", default="melee.iso", type=Path)
    ap.add_argument("--n-episodes", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()

    from tools.inference_utils import load_inference_context, load_mimic_model

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"loading model from {args.base_ckpt}")
    model, mcfg = load_mimic_model(str(args.base_ckpt), device)
    ref_model = model
    ctx = load_inference_context(args.data_dir)

    # Realtime config
    realtime_cfg = ActorConfig(
        dolphin_path=str(args.realtime_dolphin),
        iso_path=str(args.iso_path),
        character="FOX", cpu_character="FOX", cpu_level=9,
        stage="FINAL_DESTINATION",
        temperature=args.temperature,
        gfx_backend="Vulkan",
        disable_audio=False,
        use_exi_inputs=False,
        enable_ffw=False,
        replay_dir="replays_compare_realtime",
    )
    # FFW config (blocking input, default)
    ffw_cfg = ActorConfig(
        dolphin_path=str(args.ffw_dolphin),
        iso_path=str(args.iso_path),
        character="FOX", cpu_character="FOX", cpu_level=9,
        stage="FINAL_DESTINATION",
        temperature=args.temperature,
        gfx_backend="Null",
        disable_audio=True,
        use_exi_inputs=True,
        enable_ffw=True,
        blocking_input=True,
        polling_mode=False,
        replay_dir="replays_compare_ffw",
    )
    # FFW + polling (try to remove the Python-loop bottleneck)
    ffw_poll_cfg = ActorConfig(
        dolphin_path=str(args.ffw_dolphin),
        iso_path=str(args.iso_path),
        character="FOX", cpu_character="FOX", cpu_level=9,
        stage="FINAL_DESTINATION",
        temperature=args.temperature,
        gfx_backend="Null",
        disable_audio=True,
        use_exi_inputs=True,
        enable_ffw=True,
        blocking_input=False,
        polling_mode=True,
        replay_dir="replays_compare_ffw_poll",
    )

    print()
    print(f"=== Realtime (Vulkan, audio on, no FFW) ===")
    rt = collect_and_time(realtime_cfg, model, ref_model, ctx,
                          device, mcfg.max_seq_len, args.n_episodes, "RT")

    print()
    print(f"=== FFW blocking (Exi-AI, Null gfx, FFW on, blocking_input) ===")
    ffw = collect_and_time(ffw_cfg, model, ref_model, ctx,
                           device, mcfg.max_seq_len, args.n_episodes, "FFW")

    print()
    print(f"=== FFW polling (Exi-AI, Null gfx, FFW on, polling_mode) ===")
    ffw_p = collect_and_time(ffw_poll_cfg, model, ref_model, ctx,
                             device, mcfg.max_seq_len, args.n_episodes, "FFW-P")

    print()
    print(f"=== Comparison ===")
    print(f"{'metric':<22}  {'realtime':>10}  {'ffw-block':>10}  {'ffw-poll':>10}")
    print(f"{'-'*22}  {'-'*10}  {'-'*10}  {'-'*10}")
    def row(name, key, fmt):
        rv = rt[key]; fv = ffw[key]; pv = ffw_p[key]
        print(f"{name:<22}  {fv if False else rv:{fmt}}  {fv:{fmt}}  {pv:{fmt}}")
    row("wall time (s)",    "collect_time",   ">10.1f")
    row("frames/sec",       "frames_per_sec", ">10.0f")
    row("episodes/sec",     "eps_per_sec",    ">10.3f")
    row("episodes collected","n_episodes",    ">10d")
    row("frames collected", "total_frames",   ">10d")
    print()
    print(f"speedup vs realtime (frames/sec):  "
          f"ffw-block={ffw['frames_per_sec']/max(1e-6,rt['frames_per_sec']):.2f}x  "
          f"ffw-poll={ffw_p['frames_per_sec']/max(1e-6,rt['frames_per_sec']):.2f}x")
    print(f"ffw effective speed vs game-60Hz:  "
          f"ffw-block={ffw['frames_per_sec']/60:.1f}x  "
          f"ffw-poll={ffw_p['frames_per_sec']/60:.1f}x")


if __name__ == "__main__":
    main()
