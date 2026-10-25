# 2026-10-25 — RL small Fox beats both non-RL small Foxes; history use; whole-game training

Night of 10-24 → 10-25, small models only, one seed per variant (owner: fewer seeds, accept weaker
significance), nothing promoted. Scripts: `checkpoints/smallfox/queue_{f,h,i,j}.sh`; every result line
is in `checkpoints/smallfox/results.txt`. All h2h in the native sim, bf16, Fox dittos, `replays_doc_rt`
templates, 192 games at seed 41 unless noted.

## RL is not off-policy (`tools/rl_offpolicy_check.py`)

Runs VecSim exactly as `sim_rl` does, keeps the sampling logits from inside the CUDA graph, and
re-scores every sampled decision in fp32 with `ppo.head_logits` (12,139 decisions per case).

| Case | KL(rollout ‖ fp32), full | stream |
|---|---|---|
| Base weights | 1.4e-5 | 3.3e-5 |
| RL weights | 1.5e-5 | 3.2e-5 |
| After a weight swap + `refresh()` | 1.5e-5 | 3.1e-5 |

Ratios p1–p99 0.975–1.029, none outside the 0.2 clip. Not the reason stream RL fails.

## Proxy 2x2 (full/stream × independent/cond heads), `data/mastfox_c9_half`, 1,900 steps

Last-frame loss, seed 1, vs the original proxy (full, independent heads, 0.7780): full+cond −0.0305,
stream 1,440 independent −0.0036, stream 1,440 cond −0.0350, stream 1,100 cond −0.0343, stream 720
cond −0.0351. RL screens (S2 recipe, 30 updates): full+indep 61.2% / 59.8% (1,152 games each, earlier),
full+cond 93–99 (48.4%, p = 0.72; a 480-game seed-51 confirmation 228–252), stream+indep 109–83
(56.8%, p = 0.071), stream+cond 108–84 (56.2%, p = 0.097). Cond-heads are not clearly the cause.

## S-full 56k: the matched full-attention base

`fox-20261024-S-full56k` (`checkpoints/smallfox/base_full56k`): same data (`data/fox_full_hf/fox`,
647-shard manifest), batch 768, 56k steps, cond-heads, seed 1 — S-stream 56k's recipe with full
attention.

- Last-frame loss: full56k 0.6877, stream56k 0.6805 (stream −0.0072 ± 0.0026).
- h2h: stream56k vs full56k **77–115** (stream 40.1% [33.4, 47.2], p = 0.007, stock margin −0.39).

Stream has the lower loss and loses the games.

## RL on S-full 56k: stock 1 + damage 1, 30 updates

| Run | Change from the S2 recipe | vs full56k (base) | vs stream56k |
|---|---|---|---|
| SF1 | — (4,096 decisions per update) | 114–78 (59.4%, p = 0.011) | 123–69 (64.1%) |
| SF2 | 16,384 decisions per update | **121–71 (63.0%, p < 0.001)** | 130–62 (67.7%) |
| SF3 | 16,384, lr 2e-5 | 120–72 (62.5%, p = 0.001) | 136–56 (70.8%) |
| SF4 | 16,384, no critic | 119–73 (62.0%, p = 0.001) | 133–59 (69.3%) |

- Every RL model beats both non-RL small Foxes. First RL gain on a strong small base.
- Training reward, per-minute deltas averaged over the first vs last 10 updates (single updates are
  too noisy to read): SF1 stock −0.06 → −0.01, damage +0.10 → +0.15 (nearly flat); SF2/SF3/SF4 stock
  −0.12/−0.00/−0.14 → +0.15/+0.17/+0.08, damage +0.29/+0.30/+0.06 → +0.84/+0.82/+0.73. Read: at
  4,096 decisions the PPO gradient on a strong base is mostly noise; 4× the batch halves it. KL to
  the base 0.003–0.007 per decision.
- lr 2e-5 and dropping the critic didn't change the win rate vs the base (within 1 point).
- Checkpoints `checkpoints/smallfox/SF*/eval_final/`. Not checked in Dolphin, not played vs fox-master.

## How much history the stream model uses (`eval_last_frame.py name=ckpt@K`)

stream56k (receptive field 1,075), last-frame loss with history cut to K frames:

| K | 1,075 | 900 | 720 | 360 | 180 | 60 |
|---|---|---|---|---|---|---|
| Δ loss | — | 0.0000 | 0.0000 | +0.0013 | +0.0068 | +0.0310 |

Identical to 4 decimals down to 720 frames (12 s). At 180 frames (the full model's context) it
scores 0.6873, the same as full56k (0.6877): the stream model's loss edge comes entirely from history
past 3 s. Proxy stream models (c1440, c720) are also unchanged at 720.

## Whole-game training (`--whole-games-from`, commit eea064b)

Proxy setup (S, stream, cond-heads, 138,240 frames per step, 1,900 steps, tail-SWA), last-frame loss
vs c1440 (0.7430):

| Model | Δ |
|---|---|
| c720 | −0.0001 ± 0.0020 |
| whole games from step 0 | +0.0145 ± 0.0022 |
| 720 chunks for 70%, then whole games | −0.0027 ± 0.0020 |
| full attention (180) | +0.0045 ± 0.0021 |

Whole games from the start are clearly worse; short-to-long is within noise of chunks. Untested
guess for the whole-games loss: each step holds far fewer distinct games than 96 chunks drawn from 96 games, so the
gradient is more correlated, and length-bucket padding wastes some of the frame budget.
