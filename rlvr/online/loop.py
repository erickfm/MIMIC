"""Online RL training driver.

Loop:
  collect M episodes from Dolphin actor ->
  PPO update on those episodes ->
  checkpoint + eval periodically ->
  repeat until max steps.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

import torch

from rlvr.online.dolphin_actor import ActorConfig, DolphinActor
from rlvr.online.ppo import OnlinePPOConfig, ppo_update


log = logging.getLogger("rlvr.online.loop")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  [%(levelname)s]  %(message)s")


def _build_task(task_id: str, self_port: int):
    if task_id == "l_cancel_online":
        from rlvr.online.tasks.l_cancel_online import LCancelOnlineTask
        return LCancelOnlineTask(self_port=self_port)
    if task_id == "shield_escape_online":
        from rlvr.online.tasks.shield_escape_online import ShieldEscapeOnlineTask
        return ShieldEscapeOnlineTask(self_port=self_port)
    raise ValueError(f"unknown online task: {task_id}")


def train(
    base_ckpt: Path,
    data_dir: Path,
    dolphin_path: Path,
    iso_path: Path,
    task_id: str,
    run_name: str,
    episodes_per_update: int = 32,
    lr: float = 1e-6,
    temperature: float = 1.0,
    clip_eps: float = 0.2,
    kl_beta: float = 0.01,
    max_updates: int = 100,
    checkpoint_every: int = 10,
    checkpoint_dir: Path = Path("checkpoints"),
    device: str = "cuda",
    self_port: int = 1,
    cpu_character: str = "FOX",
    stage: str = "FINAL_DESTINATION",
    cpu_level: int = 9,
    gfx_backend: str = "Vulkan",
    replay_dir: Optional[Path] = None,
    use_wandb: bool = False,
    seed: int = 0,
) -> None:
    from tools.inference_utils import load_inference_context, load_mimic_model

    torch.manual_seed(seed)

    model, cfg = load_mimic_model(str(base_ckpt), device)
    ref_model, _ = load_mimic_model(str(base_ckpt), device)
    for p in ref_model.parameters():
        p.requires_grad_(False)
    ref_model.eval()
    model.train()

    ctx = load_inference_context(data_dir)
    task = _build_task(task_id, self_port=self_port)

    actor_cfg = ActorConfig(
        dolphin_path=str(dolphin_path),
        iso_path=str(iso_path),
        character="FOX",
        cpu_character=cpu_character,
        cpu_level=cpu_level,
        stage=stage,
        temperature=temperature,
        gfx_backend=gfx_backend,
        replay_dir=str(replay_dir) if replay_dir else None,
    )
    actor = DolphinActor(
        cfg=actor_cfg, task=task,
        model=model, ref_model=ref_model, ctx=ctx,
        device=device, model_seq_len=cfg.max_seq_len,
        self_port=self_port,
    )
    actor.start()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    ppo_cfg = OnlinePPOConfig(clip_eps=clip_eps, kl_beta=kl_beta)

    wandb_run = None
    if use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(project="MIMIC-RLVR-online", name=run_name,
                                   config={"lr": lr, "temperature": temperature,
                                           "clip_eps": clip_eps, "kl_beta": kl_beta,
                                           "episodes_per_update": episodes_per_update,
                                           "task_id": task_id})
        except Exception as e:
            log.warning("wandb disabled: %s", e)

    t0 = time.time()
    try:
        for update in range(1, max_updates + 1):
            t_collect = time.time()
            episodes = actor.collect(n_episodes=episodes_per_update)
            t_collect = time.time() - t_collect

            # Tally rewards for logging before we hit PPO (which detaches).
            import math
            valid = [ep for ep in episodes
                     if not math.isnan(ep.terminal_reward)]
            result_counts = {}
            for ep in valid:
                r = ep.metadata.get("result", "?")
                result_counts[r] = result_counts.get(r, 0) + 1

            if not valid:
                log.warning("update %d: no valid episodes collected", update)
                continue

            t_ppo = time.time()
            metrics = ppo_update(model, valid, optimizer, ppo_cfg, device=device)
            t_ppo = time.time() - t_ppo

            log.info(
                "update=%d collected=%d valid=%d "
                "reward=%.3f kl=%.4f clip_frac=%.2f results=%s "
                "t_collect=%.1fs t_ppo=%.1fs",
                update, len(episodes), len(valid),
                metrics["reward_mean"], metrics["kl"], metrics["clip_frac"],
                result_counts, t_collect, t_ppo,
            )
            if wandb_run is not None:
                wandb_run.log({
                    "train/loss": metrics["loss"],
                    "train/kl": metrics["kl"],
                    "train/clip_frac": metrics["clip_frac"],
                    "train/reward_mean": metrics["reward_mean"],
                    "train/n_episodes": metrics["n_episodes"],
                    "train/n_frames": metrics["n_frames"],
                    "train/grad_norm": metrics["grad_norm"],
                    "train/advantage_std": metrics["advantage_std"],
                    "train/update": update,
                    **{f"train/result_{k}": v for k, v in result_counts.items()},
                }, step=update)

            if checkpoint_every > 0 and update % checkpoint_every == 0:
                ck = checkpoint_dir / f"{run_name}_update{update:04d}.pt"
                _save_ckpt(ck, model, optimizer, cfg_snapshot, update, task_id)
                log.info("saved %s", ck)
    finally:
        actor.stop()
        if wandb_run is not None:
            wandb_run.finish()

    final = checkpoint_dir / f"{run_name}_final.pt"
    _save_ckpt(final, model, optimizer, cfg_snapshot, max_updates, task_id)
    log.info("done. final: %s  total_elapsed=%.1fs", final, time.time() - t0)


def _save_ckpt(path, model, optimizer, cfg_snapshot, update, task_id):
    """Save in the format `tools.inference_utils.load_mimic_model` expects:
    `config` must be a dict whose keys cover ModelConfig fields. Without
    it the loader falls into the legacy HAL bare-state-dict path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": cfg_snapshot,
        "update": update,
        "task_id": task_id,
    }, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base-ckpt", required=True, type=Path)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--dolphin-path", default="emulator/squashfs-root/usr/bin/dolphin-emu", type=Path)
    ap.add_argument("--iso-path", default="melee.iso", type=Path)
    ap.add_argument("--task", required=True,
                    choices=["l_cancel_online", "shield_escape_online"])
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--episodes-per-update", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--kl-beta", type=float, default=0.01)
    ap.add_argument("--max-updates", type=int, default=100)
    ap.add_argument("--checkpoint-every", type=int, default=10)
    ap.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    ap.add_argument("--cpu-character", default="FOX")
    ap.add_argument("--stage", default="FINAL_DESTINATION")
    ap.add_argument("--cpu-level", type=int, default=9)
    ap.add_argument("--self-port", type=int, default=1)
    ap.add_argument("--gfx-backend", default="Vulkan")
    ap.add_argument("--replay-dir", type=Path, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    train(
        base_ckpt=args.base_ckpt, data_dir=args.data_dir,
        dolphin_path=args.dolphin_path, iso_path=args.iso_path,
        task_id=args.task, run_name=args.run_name,
        episodes_per_update=args.episodes_per_update,
        lr=args.lr, temperature=args.temperature,
        clip_eps=args.clip_eps, kl_beta=args.kl_beta,
        max_updates=args.max_updates,
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device, self_port=args.self_port,
        cpu_character=args.cpu_character, stage=args.stage,
        cpu_level=args.cpu_level,
        gfx_backend=args.gfx_backend,
        replay_dir=args.replay_dir,
        use_wandb=args.wandb, seed=args.seed,
    )


if __name__ == "__main__":
    main()
