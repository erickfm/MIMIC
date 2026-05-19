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


def _build_task(task_id, vrs, vr_weights, self_port: int):
    """Build the OnlineTask. Either a `CompositeVRTask` over `vrs` (the
    VR suite — whole-match episode) or a legacy single scenario task."""
    if vrs:
        from rlvr.online.vr import (
            DEFAULT_VR_WEIGHTS, VR_REGISTRY, CompositeVRTask,
        )
        modules = []
        for vid in vrs:
            if vid not in VR_REGISTRY:
                raise ValueError(
                    f"unknown VR '{vid}'; known: {sorted(VR_REGISTRY)}")
            modules.append(VR_REGISTRY[vid](self_port=self_port))
        # --vr-weights overrides; otherwise use the pre-seeded defaults.
        weights = (list(vr_weights) if vr_weights
                   else [DEFAULT_VR_WEIGHTS.get(v, 1.0) for v in vrs])
        if len(weights) != len(modules):
            raise ValueError(
                f"--vr-weights ({len(weights)}) must match "
                f"--vrs ({len(modules)})")
        if "low_percent_kill" in vrs and "stock_delta" not in vrs:
            log.warning("low_percent_kill is in --vrs without stock_delta; "
                        "it is a bonus layered on stock-delta and should "
                        "not run standalone")
        return CompositeVRTask(modules, weights, self_port=self_port)
    if task_id == "l_cancel_online":
        from rlvr.online.tasks.l_cancel_online import LCancelOnlineTask
        return LCancelOnlineTask(self_port=self_port)
    if task_id == "shield_escape_online":
        from rlvr.online.tasks.shield_escape_online import ShieldEscapeOnlineTask
        return ShieldEscapeOnlineTask(self_port=self_port)
    raise ValueError(f"unknown online task: {task_id}")


def train(
    base_ckpt: Path,
    ref_ckpt: Optional[Path],
    data_dir: Path,
    dolphin_path: Path,
    iso_path: Path,
    task_id: Optional[str],
    run_name: str,
    episodes_per_update: int = 32,
    lr: float = 1e-6,
    temperature: float = 1.0,
    clip_eps: float = 0.2,
    kl_beta: float = 0.01,
    gamma: float = 0.998,
    vrs: Optional[list] = None,
    vr_weights: Optional[list] = None,
    max_updates: int = 100,
    checkpoint_every: int = 10,
    checkpoint_dir: Path = Path("checkpoints"),
    device: str = "cuda",
    self_port: int = 1,
    cpu_character: str = "FOX",
    stage: str = "FINAL_DESTINATION",
    cpu_level: int = 9,
    gfx_backend: str = "Vulkan",
    use_exi_inputs: bool = False,
    enable_ffw: bool = False,
    opponent_ckpt: Optional[Path] = None,
    opponent_data_dir: Optional[Path] = None,
    opponent_temperature: float = 1.0,
    replay_dir: Optional[Path] = None,
    log_file: Optional[Path] = None,
    use_wandb: bool = False,
    seed: int = 0,
) -> None:
    from tools.inference_utils import load_inference_context, load_mimic_model

    # If --log-file was passed, also tee Python logging to that file
    # (in addition to whatever stdout/stderr the shell is doing). The
    # HUD subprocess tails this file.
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        _fh = logging.FileHandler(str(log_file), mode="w")
        _fh.setLevel(logging.INFO)
        _fh.setFormatter(logging.Formatter(
            "%(asctime)s  [%(levelname)s]  %(message)s"))
        logging.getLogger().addHandler(_fh)
        log.info("logging to file: %s", log_file)

    torch.manual_seed(seed)

    model, cfg = load_mimic_model(str(base_ckpt), device)
    # Reference model (for PPO's KL penalty) defaults to the same ckpt
    # as base, but can be explicitly different. Useful for resuming
    # training from a mid-run checkpoint while still anchoring KL
    # against the original BC baseline (so the bot can't drift further
    # from BC than the kl_beta budget originally allowed).
    _ref_path = ref_ckpt if ref_ckpt is not None else base_ckpt
    ref_model, _ = load_mimic_model(str(_ref_path), device)
    log.info("base_ckpt=%s  ref_ckpt=%s", base_ckpt, _ref_path)
    # Snapshot the model config for checkpoint saves. Stored as a dict
    # so tools.inference_utils.load_mimic_model can reconstruct the
    # ModelConfig without falling into the legacy HAL bare-state-dict
    # branch.
    from dataclasses import asdict
    try:
        model_cfg_snapshot = asdict(cfg)
    except TypeError:
        # Not a dataclass — store as the raw object.
        model_cfg_snapshot = cfg
    for p in ref_model.parameters():
        p.requires_grad_(False)
    ref_model.eval()
    # eval() not train(): the policy is forwarded deterministically for
    # both rollout and the PPO re-forward. Dropout during rollout would
    # handicap the trainee vs the eval-mode opponent and make logprobs
    # stochastic vs the eval-mode ref; exploration comes from action
    # sampling (temperature), not dropout. eval() does not stop gradients.
    model.eval()

    ctx = load_inference_context(data_dir)
    task = _build_task(task_id, vrs, vr_weights, self_port=self_port)
    # Use the actual task id (composite "a+b+c", or the scenario id) for
    # checkpoint provenance and logging.
    task_id = task.id
    log.info("task=%s", task_id)

    actor_cfg = ActorConfig(
        dolphin_path=str(dolphin_path),
        iso_path=str(iso_path),
        character="FOX",
        cpu_character=cpu_character,
        cpu_level=cpu_level,
        stage=stage,
        temperature=temperature,
        # The VR suite runs as one whole-match episode (CompositeVRTask).
        whole_match_episode=bool(vrs),
        gfx_backend=gfx_backend,
        # FFW: needs both use_exi_inputs and enable_ffw, plus the
        # emulator_ffw/ Exi-AI build (not emulator/).
        use_exi_inputs=use_exi_inputs,
        enable_ffw=enable_ffw,
        # Audio is noisy in headless FFW runs; disable when FFW is on.
        disable_audio=enable_ffw,
        opponent_ckpt=str(opponent_ckpt) if opponent_ckpt else None,
        opponent_data_dir=str(opponent_data_dir) if opponent_data_dir else None,
        opponent_temperature=opponent_temperature,
        replay_dir=str(replay_dir) if replay_dir else None,
    )
    actor = DolphinActor(
        cfg=actor_cfg, task=task,
        model=model, ref_model=ref_model, ctx=ctx,
        device=device, model_seq_len=cfg.max_seq_len,
        self_port=self_port,
    )
    actor.start()

    # Auto-launch the live web HUD on every RLVR run when --log-file
    # is set. The HUD is a small HTTP+SSE server (rlvr.eval.training_web)
    # that tails the log and serves http://localhost:8765 — the
    # browser opens it directly, and OBS can use Browser Source.
    # Independent of Dolphin's gfx_backend: FFW headless training
    # still gets the HUD (you just won't see the game itself, only
    # the dashboard).
    import os as _os
    import subprocess as _sp
    import sys as _sys
    hud_proc = None
    if log_file is not None:
        # Don't try to auto-open a browser when running fully headless
        # (no DISPLAY available, e.g. remote training box). The server
        # still serves; user can open the URL from any machine.
        no_open = (gfx_backend == "Null"
                   or not _os.environ.get("DISPLAY"))
        cmd = [_sys.executable, "-m", "rlvr.eval.training_web.server",
               "--log", str(log_file), "--port", "8765",
               "--max-updates", str(max_updates)]
        if no_open:
            cmd.append("--no-open")
        try:
            hud_proc = _sp.Popen(
                cmd,
                env={**_os.environ,
                     "DISPLAY": _os.environ.get("DISPLAY", ":0")},
                stdout=_sp.DEVNULL, stderr=_sp.DEVNULL,
            )
            log.info("HUD web server launched (pid=%d). "
                     "Open http://localhost:8765 — log=%s",
                     hud_proc.pid, log_file)
        except Exception as e:
            log.warning("could not launch HUD: %s", e)
    else:
        log.info("HUD skipped — pass --log-file <path> to enable.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    ppo_cfg = OnlinePPOConfig(clip_eps=clip_eps, kl_beta=kl_beta, gamma=gamma)

    wandb_run = None
    if use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(project="MIMIC-RLVR-online", name=run_name,
                                   config={"lr": lr, "temperature": temperature,
                                           "clip_eps": clip_eps, "kl_beta": kl_beta,
                                           "episodes_per_update": episodes_per_update,
                                           "gamma": gamma,
                                           "task_id": task_id})
        except Exception as e:
            log.warning("wandb disabled: %s", e)

    t0 = time.time()
    try:
        for update in range(1, max_updates + 1):
            t_collect = time.time()
            episodes = actor.collect(n_episodes=episodes_per_update)
            t_collect = time.time() - t_collect

            # Keep Dolphin's enet connection warm across PPO + checkpointing.
            # Nothing else pumps console.step() between collect() calls, and
            # the FFW slippstream peer drops after ~20 s of silence. The
            # keepalive thread is stopped (and joined) before the next
            # collect() so the main thread regains sole console ownership.
            actor.start_keepalive()
            try:
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
                metrics = ppo_update(model, valid, optimizer, ppo_cfg,
                                     device=device, ref_model=ref_model)
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
                    _save_ckpt(ck, model, optimizer, model_cfg_snapshot, update, task_id)
                    log.info("saved %s", ck)
            finally:
                actor.stop_keepalive()
    finally:
        actor.stop()
        if hud_proc is not None:
            try:
                hud_proc.terminate()
                hud_proc.wait(timeout=2)
            except Exception:
                try:
                    hud_proc.kill()
                except Exception:
                    pass
        if wandb_run is not None:
            wandb_run.finish()

    final = checkpoint_dir / f"{run_name}_final.pt"
    _save_ckpt(final, model, optimizer, model_cfg_snapshot, max_updates, task_id)
    log.info("done. final: %s  total_elapsed=%.1fs", final, time.time() - t0)


def _save_ckpt(path, model, optimizer, model_cfg_snapshot, update, task_id):
    """Save in the format `tools.inference_utils.load_mimic_model` expects:
    `config` must be a dict whose keys cover ModelConfig fields. Without
    it the loader falls into the legacy HAL bare-state-dict path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": model_cfg_snapshot,
        "update": update,
        "task_id": task_id,
    }, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base-ckpt", required=True, type=Path)
    ap.add_argument("--ref-ckpt", type=Path, default=None,
                    help="Reference checkpoint for PPO's KL penalty. "
                         "Defaults to --base-ckpt. Use a different one "
                         "when resuming from a mid-run checkpoint but "
                         "still want KL anchored to the original "
                         "(usually BC baseline).")
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--dolphin-path", default="emulator/squashfs-root/usr/bin/dolphin-emu", type=Path)
    ap.add_argument("--iso-path", default="melee.iso", type=Path)
    ap.add_argument("--task", default=None,
                    choices=["l_cancel_online", "shield_escape_online"],
                    help="Legacy single scenario task. Mutually exclusive "
                         "with --vrs.")
    ap.add_argument("--vrs", nargs="+", default=None, metavar="VR",
                    help="Verifiable-reward ids run together as one "
                         "whole-match CompositeVRTask, e.g. "
                         "`--vrs stock_delta damage_delta`. Mutually "
                         "exclusive with --task.")
    ap.add_argument("--vr-weights", nargs="+", type=float, default=None,
                    metavar="W",
                    help="Per-VR weights, parallel to --vrs (default 1.0 "
                         "each).")
    ap.add_argument("--gamma", type=float, default=0.998,
                    help="PPO return discount. Whole-match episodes need "
                         "gamma<1 (default 0.998, ~8s credit horizon).")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--episodes-per-update", type=int, default=None,
                    help="Episodes per PPO update (matches, for --vrs). "
                         "Default 6 for --vrs, 32 for --task.")
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
    ap.add_argument("--gfx-backend", default="Vulkan",
                    help="GPU backend. Use 'Null' for FFW headless training.")
    ap.add_argument("--use-exi-inputs", action="store_true",
                    help="Use EXI input injection (required for --enable-ffw). "
                         "Needs the emulator_ffw/ Exi-AI build, not emulator/.")
    ap.add_argument("--enable-ffw", action="store_true",
                    help="Run Dolphin at unlimited speed (FFW). Requires "
                         "--use-exi-inputs and the Exi-AI emulator at "
                         "emulator_ffw/squashfs-root/usr/bin/dolphin-emu.")
    ap.add_argument("--opponent-ckpt", type=Path, default=None,
                    help="Frozen bot opponent on the other port. Without "
                         "this flag the opponent is CPU-9 (legacy).")
    ap.add_argument("--opponent-data-dir", type=Path, default=None,
                    help="Inference-context dir for the opponent. Defaults "
                         "to --data-dir.")
    ap.add_argument("--opponent-temperature", type=float, default=1.0,
                    help="Sampling temperature for the opponent's policy. "
                         "1.0 mirrors the trainee.")
    ap.add_argument("--replay-dir", type=Path, default=None)
    ap.add_argument("--log-file", type=Path, default=None,
                    help="Tee Python logging to this file (in addition to "
                         "stdout/stderr). Required for auto-HUD launch.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if (args.task is None) == (args.vrs is None):
        ap.error("exactly one of --task / --vrs is required")
    epu = args.episodes_per_update
    if epu is None:
        epu = 6 if args.vrs else 32
    train(
        base_ckpt=args.base_ckpt, ref_ckpt=args.ref_ckpt,
        data_dir=args.data_dir,
        dolphin_path=args.dolphin_path, iso_path=args.iso_path,
        task_id=args.task, vrs=args.vrs, vr_weights=args.vr_weights,
        run_name=args.run_name,
        episodes_per_update=epu, gamma=args.gamma,
        lr=args.lr, temperature=args.temperature,
        clip_eps=args.clip_eps, kl_beta=args.kl_beta,
        max_updates=args.max_updates,
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device, self_port=args.self_port,
        cpu_character=args.cpu_character, stage=args.stage,
        cpu_level=args.cpu_level,
        gfx_backend=args.gfx_backend,
        use_exi_inputs=args.use_exi_inputs,
        enable_ffw=args.enable_ffw,
        opponent_ckpt=args.opponent_ckpt,
        opponent_data_dir=args.opponent_data_dir,
        opponent_temperature=args.opponent_temperature,
        replay_dir=args.replay_dir,
        log_file=args.log_file,
        use_wandb=args.wandb, seed=args.seed,
    )


if __name__ == "__main__":
    main()
