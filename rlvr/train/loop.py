"""RLVR training driver.

Each step:
  1. Sample B prompts from the mined-state pool (excluding eval holdout).
  2. Rollout N samples per prompt with the current policy. Capture
     log-probs under theta (grad-enabled) and under the frozen ref
     (no grad) at sampling time.
  3. Reward = verifier(prompt, sampled_ctrl). Shape (B*N,).
  4. Recompute log-probs under theta with fresh forward pass (old
     log-probs are the detached snapshot from step 2; this fresh pass
     is where PPO's "current policy" log-probs come from).
     *Short-circuit note*: single-step on-policy GRPO collapses old
     == current on the first gradient step, so ratio = 1. For >1 inner
     epochs we'd diverge; for now we do one PPO epoch per rollout and
     drop the recompute.
  5. Compute GRPO loss + KL + step optimizer.
  6. Every --eval-every steps, run the frozen eval set.

Checkpoints save: model state, optimizer state, config, metadata for
version compat.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow.parquet as pq
import torch

import rlvr.tasks  # noqa: F401 — register tasks
from rlvr.eval.runner import run_eval
from rlvr.rollout import rollout
from rlvr.sampler.mined import sample_states
from rlvr.state.gamestate import SCHEMA_VERSION
from rlvr.tasks.registry import get_verifier, registry_hash
from rlvr.train.grpo import GRPOConfig, grpo_loss


def _load_eval_holdout_ids(eval_set_path: Path) -> set:
    meta = pq.read_metadata(eval_set_path).metadata or {}
    blob = meta.get(b"rlvr_eval_replay_ids")
    if blob is None:
        return set()
    return set(json.loads(blob.decode()))


def _save_checkpoint(path: Path, model, optimizer, config: dict, step: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "rlvr_schema_version": SCHEMA_VERSION,
        "rlvr_registry_hash": registry_hash(),
        "step": step,
    }, path)


def train(
    base_ckpt: Path,
    data_dir: Path,
    events_path: Path,
    slp_dir: Path,
    eval_set_path: Path,
    task_id: str,
    run_name: str,
    prompts_per_step: int = 32,
    rollouts_per_prompt: int = 8,
    lr: float = 1e-6,
    temperature: float = 1.0,
    clip_eps: float = 0.2,
    kl_beta: float = 0.01,
    max_steps: int = 2000,
    eval_every: int = 200,
    checkpoint_every: int = 500,
    checkpoint_dir: Path = Path("checkpoints"),
    device: str = "cuda",
    l_cancel_result_filter: Optional[int] = None,
    use_wandb: bool = True,
    seed: int = 0,
) -> None:
    from tools.inference_utils import load_inference_context, load_mimic_model

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load model (trainable) + frozen reference copy from same ckpt.
    model, cfg = load_mimic_model(str(base_ckpt), device)
    ref_model, _ = load_mimic_model(str(base_ckpt), device)
    for p in ref_model.parameters():
        p.requires_grad_(False)
    ref_model.eval()
    model.train()

    ctx = load_inference_context(data_dir)
    verifier = get_verifier(task_id)
    eval_holdout_ids = _load_eval_holdout_ids(eval_set_path)

    # Config snapshot stored with every checkpoint
    cfg_snapshot = {
        **{k: getattr(cfg, k) for k in cfg.__dataclass_fields__},
        "run_name": run_name,
        "task_id": task_id,
        "lr": lr,
        "prompts_per_step": prompts_per_step,
        "rollouts_per_prompt": rollouts_per_prompt,
        "temperature": temperature,
        "clip_eps": clip_eps,
        "kl_beta": kl_beta,
        "seed": seed,
    }

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    grpo_cfg = GRPOConfig(clip_eps=clip_eps, kl_beta=kl_beta)

    wandb_run = None
    if use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project="MIMIC-RLVR", name=run_name, config=cfg_snapshot,
            )
        except Exception as e:
            print(f"[wandb] disabled: {e}")
            wandb_run = None

    print(f"[train] starting run {run_name!r} for {max_steps} steps")
    t0 = time.time()
    for step in range(1, max_steps + 1):
        # 1. Sample prompts (with eval-holdout excluded)
        prompts = sample_states(
            events_path=events_path,
            slp_dir=slp_dir,
            task_id=task_id,
            n=prompts_per_step,
            seed=seed * 10_000 + step,
            exclude_replay_ids=eval_holdout_ids,
            l_cancel_result=l_cancel_result_filter,
        )
        if len(prompts) < prompts_per_step:
            print(f"[train] step {step}: only {len(prompts)} prompts; skipping")
            continue

        # 2. Rollout
        model.train()
        rb = rollout(
            model, ref_model, prompts,
            n_per_prompt=rollouts_per_prompt,
            ctx=ctx, temperature=temperature,
            seed=seed * 1_000_000 + step, device=device,
        )

        # 3. Rewards (verifier per rollout)
        rewards = torch.tensor(
            [verifier(rb.prompts[i // rollouts_per_prompt], c)
             for i, c in enumerate(rb.sampled_ctrls)],
            dtype=torch.float32, device=device,
        )

        # 4. Single-epoch on-policy: logprobs_old == logprobs_theta at
        #    rollout time. Detach a snapshot for ratio computation.
        logprobs_old = rb.logprobs_theta.detach()

        # 5. GRPO loss + step
        out = grpo_loss(
            logprobs_theta=rb.logprobs_theta,
            logprobs_old=logprobs_old,
            logprobs_ref=rb.logprobs_ref,
            rewards=rewards,
            group_size=rollouts_per_prompt,
            cfg=grpo_cfg,
        )
        optimizer.zero_grad()
        out["loss"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # 6. Log
        metrics = {
            "train/loss": out["loss"].item(),
            "train/pg_loss": out["pg_loss"].item(),
            "train/kl": out["kl"].item(),
            "train/reward_mean": out["reward_mean"].item(),
            "train/advantage_mean": out["advantage_mean"].item(),
            "train/advantage_std": out["advantage_std"].item(),
            "train/ratio_mean": out["ratio_mean"].item(),
            "train/clip_frac": out["clip_frac"].item(),
            "train/grad_norm": float(grad_norm),
            "train/step": step,
            "train/elapsed_sec": time.time() - t0,
        }
        if step % 10 == 0 or step == 1:
            print(
                f"[train] step={step} loss={metrics['train/loss']:.4f} "
                f"r={metrics['train/reward_mean']:.3f} "
                f"kl={metrics['train/kl']:.5f} "
                f"clip_frac={metrics['train/clip_frac']:.2f}"
            )
        if wandb_run is not None:
            wandb_run.log(metrics, step=step)

        # 7. Periodic eval (uses the live model, not the on-disk ckpt)
        if eval_every > 0 and step % eval_every == 0:
            model.eval()
            with torch.no_grad():
                report = run_eval(
                    eval_set_path=eval_set_path,
                    slp_dir=slp_dir,
                    model=model,
                    ctx=ctx,
                    device=device,
                )
            model.train()
            # Log per-task pass rates
            eval_metrics = {}
            for tid, rep in report["per_task"].items():
                eval_metrics[f"eval/{tid}_pass_rate"] = rep["pass_rate"]
                eval_metrics[f"eval/{tid}_n"] = rep["n"]
            print(f"[train] step={step} eval: {eval_metrics}")
            if wandb_run is not None:
                wandb_run.log(eval_metrics, step=step)

        # 8. Periodic checkpoint
        if checkpoint_every > 0 and step % checkpoint_every == 0:
            ckpt_path = checkpoint_dir / f"{run_name}_step{step:06d}.pt"
            _save_checkpoint(ckpt_path, model, optimizer, cfg_snapshot, step)
            print(f"[train] step={step} saved {ckpt_path}")

    # Final checkpoint
    final_path = checkpoint_dir / f"{run_name}_final.pt"
    _save_checkpoint(final_path, model, optimizer, cfg_snapshot, max_steps)
    print(f"[train] done. final ckpt: {final_path}")
    if wandb_run is not None:
        wandb_run.finish()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base-ckpt", required=True, type=Path)
    ap.add_argument("--data-dir", required=True, type=Path,
                    help="dir with mimic_norm.json, norm_stats.json, etc.")
    ap.add_argument("--events", required=True, type=Path)
    ap.add_argument("--slp-dir", required=True, type=Path)
    ap.add_argument("--eval-set", required=True, type=Path)
    ap.add_argument("--task", default="l_cancel_opportunity")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--prompts-per-step", type=int, default=32)
    ap.add_argument("--rollouts-per-prompt", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--kl-beta", type=float, default=0.01)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--checkpoint-every", type=int, default=500)
    ap.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--l-cancel-result-filter", type=int, default=None,
                    help="Only sample prompts where replay_l_cancel_result == this (1 or 2).")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    train(
        base_ckpt=args.base_ckpt,
        data_dir=args.data_dir,
        events_path=args.events,
        slp_dir=args.slp_dir,
        eval_set_path=args.eval_set,
        task_id=args.task,
        run_name=args.run_name,
        prompts_per_step=args.prompts_per_step,
        rollouts_per_prompt=args.rollouts_per_prompt,
        lr=args.lr,
        temperature=args.temperature,
        clip_eps=args.clip_eps,
        kl_beta=args.kl_beta,
        max_steps=args.max_steps,
        eval_every=args.eval_every,
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        l_cancel_result_filter=args.l_cancel_result_filter,
        use_wandb=not args.no_wandb,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
