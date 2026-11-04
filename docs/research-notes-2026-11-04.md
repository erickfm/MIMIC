# 2026-11-04 — bf16 PPO update (timing only), Phillip's RL vs ours, batch/LR literature

Owner decisions today: L is dropped, stick with M. The bf16 update below is timed only; its effect on
win rate is untested.

## bf16 PPO update: `tools/sim_rl.py --ppo-amp`

Where M RL time went (`fox-M-bc2-mlr01-s1.rl2-s1`, 4090, `rl-v2`): 131 s per update = 51 s rollout +
80 s update; 100 updates = 219 min. The rollout already ran in bf16; the update ran fp32, uncompiled,
minibatch 64.

Changes (uncommitted at time of writing):
- `--ppo-amp` (default off, so `rl-v2` is unchanged): bf16 autocast for the update's forwards — the
  "old" log-prob scoring pass, the PPO epochs and the global anchor (reference and learner). Old
  log-probs are scored with the same setting as the epochs, so the ratio still starts at 1. Attention
  math stays fp32 inside the model, as in BC training.
- Full-causal models: each update's minibatches are collated onto the GPU once and reused by the
  scoring pass and every epoch (previously re-collated every epoch). Identical tensors.
- `ppo_update` keeps metric sums on the device (no host sync per minibatch).
- The log gains `score_s` (anchor prep + collate + old log-prob scoring) and `epochs_s`.
- `rlvr/tests/test_rl_context.py` passes (re-scoring exact to ~1e-7).

Timing: M base (`runs/fox-M-bc2-mlr01-s1/policy`), `rl-v2` args, 3 updates per config, seed 1. Values
from updates 2–3 (update 1 includes warmup).

| Config | update_s | score_s | epochs_s (4 epochs) | rollout_s |
|---|---|---|---|---|
| Original run (10-29 log, updates 93–100) | 80 | – | – | 51 |
| fp32, collate once (`--no-ppo-amp`) | 83 | 6.4 | 77 | 52 |
| bf16, minibatch 64 (`--ppo-amp`) | **52** | 4.0 | 48 | 52 |
| bf16, minibatch 256 | 55 | 4.4 | 50 | 52 |

- bf16 is the whole gain. Collating once and dropping the syncs gave nothing measurable (83 vs 80 s is
  within run-to-run variation).
- Per update 131 → ~104 s; 100 updates ~219 → ~173 min (estimated from these per-update times).
- Anchor KL (7e-5 → 5e-4 over updates 1–3), clip fraction (≤0.001) and grad norm (0.13–0.24) were in
  the same range across all three configs. Three updates say nothing about final strength.
- The update now runs ~250k frame-positions/s (16,384 × 180 × 4 in 48 s) vs roughly 400k for M BC on
  the same GPU. `torch.compile` on the update's forward is the next untried candidate.
- The Triton "out of resource" autotune errors in these logs come from compiling the frozen opponent
  for the rollout; they were present before and are harmless.
- Logs: `/tmp/claude-1000/-home-erick-projects-MIMIC/3ed2ba82-bc12-494b-93b0-2325720d6b95/scratchpad/ppo_amp/` (not kept).

To check before using it: one full 100-update M run with `--ppo-amp`, h2h vs
`fox-M-bc2-mlr01-s1.rl2-s1`.

Other wall-time ideas, not tried:
- Two GPUs: rollout on one and update on the other with a one-update-stale policy; wall time per
  update becomes the max of the two instead of the sum. Little gain on one GPU (both stages GPU-bound).
- Rollout (2,560 frames/s on 64 envs) is GPU-bound on re-reading 180 frames per frame for learner and
  opponent. The KV-cache speedup only applies to stream models (sunset). More envs per forward or
  compiling the learner might help a little; unmeasured.
- Fewer PPO epochs is not free: clipping almost never binds, so the 4 epochs act as 4 small gradient
  steps, and 1 epoch at 4× lr was clearly worse (153–231, 10-27).

## Phillip (slippi-ai) RL vs ours

Read from the slippi-ai source, local clone at `5ada33e` (2026-10-20, = origin today). The repo has
default configs and launch scripts (`slippi_ai/jax/rl/scripts/launch_rl.py`, `scripts/rl_example.sh`),
not the settings used for the released agents.

Reward (`slippi_ai/reward.py`, `RewardConfig.default`, used by the JAX learner):

| Term | Phillip | Ours (`rl-v2`) |
|---|---|---|
| Stocks | ±1 per death | same |
| Damage | 0.01 per % (100% = 1 stock) | 0.04 per % (25% = 1 stock) |
| Bad ledge grab | −0.02 per ledge grab while the opponent is toward center and not invulnerable | none |
| Offstage stalling | −0.2 per second >30 units offstage while the opponent is onstage | none |
| Approaching | +0.001 × velocity toward the opponent, per frame | none |
| Zero-sum | yes ("no collusion") | yes |

Algorithm (`slippi_ai/jax/rl/learner.py`, `launch_rl.py`):

| | Phillip | Ours |
|---|---|---|
| Opponent | self-play: current learner, both sides trained | frozen BC base, learner side only |
| Baseline | separate value network (value cost 1, 8 burn-in epochs) | batch mean, no critic |
| Advantage | λ=1 (MC return − value); value targets λ=0; reward half-life 8 s (γ ≈ 0.9986) | γ 0.99 (~1.2 s half-life), horizon 5 s |
| Trust region | log-ratio clip ±0.01, + 0.3 × KL(behavior‖policy), abort if mean KL > 1e-4 | ratio clip ±0.2 (rarely binds) |
| KL to BC | 0.01 forward + 0.01 reverse, every trained frame | 0.01 anchor, 512 random frames per update |
| Optimizer | Adam 3e-5, 2 epochs, policy-gradient weight 3 | AdamW 1e-5, 4 epochs |
| Frames trained | all (LSTM) | 1 in 8 learner frames |
| Scale | 2,048 sim envs × 80 frames per step, up to 20k steps (≈3.3B frames per side) | 64 envs, ~13M learner frames |
| Reaction delay | 18 frames | 0 |

They also have their own batched sim (`slippi_ai/sim_env`, added 2026-05), plus newer two-agent
(`train_two`), Nash-RL and Q-learning variants.

## Batch size and LR in the LLM literature (for a future M batch/LR test)

Directional, mostly AdamW single-epoch pretraining; read from abstracts and summaries.
- Optimal and critical batch size grow with data size, not model size: Zhang et al. 2024
  (arXiv 2410.21676), Bergsma et al. 2025 "Power Lines" (2505.13738), Step Law (2503.04715).
- Optimal LR falls with model size (Step Law, DeepSeek LLM 2401.02954, Porian et al. 2406.19146) and,
  at fixed batch, with training length (Bjorck et al. 2409.19913, holds under µP).
- Critical batch size grows during a run; batch ramp-up reached slightly better loss with 43% fewer
  steps on OLMo 1B (Merrill et al. 2505.23971).
- Small batches train fine if Adam β2's half-life is kept fixed in tokens; gradient accumulation on one
  device is wasteful (Marek et al. 2507.07101).
- Optimal AdamW weight decay scales linearly with batch size (Power Lines).
- Muon keeps its data-efficiency edge past AdamW's critical batch size (Essential AI 2505.02222).

Our setting differs: many epochs over 46k games, highly correlated frames within a 180-frame window,
and Muon. At M only the optimizer has been tuned (Muon LR). Batch, dropout, weight decay, data and
steps were copied from S. On one 4090 a bigger batch can't cut wall time (M already runs 384 × 2
accumulation); it only can with more GPUs. Proposed test (not run): M at effective batch 1,536 for 28k
steps (same samples as `bc-M-v3`), h2h vs `fox-M-bc2-mlr01-s1`.
