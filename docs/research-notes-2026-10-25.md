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

# Afternoon: stream sunset, packed stream RL update, LLM-architecture sweep

Owner-approved, one seed each, nothing promoted. Scripts: `checkpoints/smallfox/queue_k_{rl,w60}.sh`,
`checkpoints/fastproxy/arch_sweep.sh`. Result lines: `checkpoints/smallfox/results.txt` (SS8 lines
copied from the box) and `checkpoints/fastproxy/results.tsv`.

## Can stream catch full? Two interventions, both no

Hypothesis tested (exposure bias / copycat): stream conditions on 12 s of its own off-distribution
history, full on 3 s, so stream loses in closed loop despite lower loss.

**(1) RL on stream56k with SF2's recipe (SS8, 16,384 decisions per update, 30 updates):**

| Matchup | Record | SS8 win rate | p |
|---|---|---|---|
| SS8 vs stream56k (base) | 103–89 | 53.6% | 0.35 |
| SS8 vs full56k | 93–99 | 48.4% | 0.72 |
| SS8 vs SF2 | 74–118 | 38.5% | 0.002 |

**(2) Shorter receptive field: `fox-20261025-S-w60-56k`** (`checkpoints/smallfox/base_w60`), window 60
(355 frames, ~6 s), otherwise stream56k's recipe. Last-frame loss 0.6897 (stream56k 0.6805, full56k
0.6877). h2h: w60 vs full56k 74–118 (38.5%, p = 0.002); w60 vs stream56k 98–94. RL on it (SW1, SF2's
recipe): vs w60 94–98, vs full56k 94–98, vs SF2 74–118, vs SS8 86–106.

Neither closes the gap, so the exposure-bias hypothesis is not supported. Owner sunset stream
(2026-10-25). Open, untested: stream trains on 96 chunks per step where full trains on 768 windows
(same frames, 8x fewer independent sequences); the whole-game result above points the same way.

## Packed RL update for sliding-window models (commit 718f619)

The PPO update re-scored each sampled decision from its own 1,075-frame context, though decisions ~8
frames apart share ~99% of it. `ppo.pack_streams` stitches one slot's overlapping contexts into one
sequence (only when the overlapping frames are identical) and one causal forward scores all of its
decisions. Exact: logits within ~1e-7 in `rlvr/tests/test_rl_context.py`; on an identical rollout
update 1 matched the per-decision path to 4-5 digits (KL 5.4407e-5 vs 5.4408e-5, grad norm 0.290826
vs 0.290818). PPO step 303 s -> 15 s at 16,384 decisions. On by default for window models
(`--no-packed-update` to disable). Moot for now with stream sunset.

Ran SS8 on a rented vast.ai box (2x RTX 5090) for ~1.5 h; sim rebuilt there was physics-exact 6/6.
Without the packed update the 5090 was barely faster than the shared 4090 (update not GPU-bound).

## LLM-architecture sweep on full S (commit d5917d0, `train.py --arch key=value`)

`tools/fast_proxy.sh NAME S 1900 1 --cond-heads --arch ...`; seed gap ~0.0007. Control 0.7662, 271 s.

| Variant (`--arch`) | Δ val | Run time |
|---|---|---|
| `attn_gate=1` (Qwen gated attention, elementwise; +7% params) | **−0.0062** | +9% |
| `canon=1` (Canon convs, k=4, before attn and MLP) | −0.0013 | +33% |
| `use_swiglu=1` | −0.0011 | +16% |
| `encoder_type=mimic_entity` (opp + self token per frame) | −0.0009 | +158% |
| `qk_norm=1` | −0.0003 | +10% |
| `logit_softcap=30` | −0.0001 | +7% |
| `zero_init_proj=1` | 0.0000 | −1% |
| `use_rmsnorm=1` | +0.0004 | +6% |
| `value_residual=1` (ResFormer) | +0.0006 | +8% |
| `d_model=192,nhead=3,dim_feedforward=768,num_layers=10` (deep-thin, 0.98x params) | +0.0012 | +26% |
| `unet_skips=1` | +0.0025 | +6% |
| `mlp_act=relu2` | +0.0046 | +3% |
| modded-nanogpt bundle (qk_norm, relu2, zero_init, softcap 30, value_residual, unet) | +0.0053 | +14% |

Run times include the fixed eval/averaging overhead and are single runs.

**Gated attention in play (proxy scale):** fp3-gate vs fp3-ctrl, 96-game sim h2h, seed 41:
**62–34** (64.6% [54.6, 73.4], p = 0.006, stock margin +0.51; 31–17 in each port order). Untested at
S 56k. Owner: per-head gating not wanted.

All variants are full-causal only (refused with `--attn-window`), causal (checked), and fall back to
the plain forward in `fast_policy.forward_last` (correct, slower sim inference; the gate could get a
fast path).
