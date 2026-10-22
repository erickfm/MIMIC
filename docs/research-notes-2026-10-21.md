# 2026-10-21 — cond-heads beats the champion in the sim; L-cancel RL in the sim; h2h harness bias; trainer crash

## TL;DR

- **Cond-heads (`--cond-heads`, champion recipe otherwise) beats `control`
  (same code and recipe, no flag) 611–445 in the native sim** (57.9%, 95% CI
  54.9–60.8%, p ≈ 4×10⁻⁷, 1,056 bf16 games on the fixed harness). Against the
  fox-master champion it's 610–446 (57.8%, fp32). Val loss on a fixed held-out
  set: 0.6803 → 0.6519 (−4.2%).
- **RoPE (`mimic-rope-matched`) loses to relpos on val loss, and the gap grows
  with training.** It was worse than `control` at all 62 matched evals, by 1.3%
  at step 174k. Stopped by the owner at step 175k; RoPE is dropped. See
  "RoPE vs relpos" below for how strong this is.
- **The old `sim_h2h` harness was biased.** An identical-model control gave
  225–175 (p = 0.014). Fixed in `27bda08`. The fixed harness's control gives
  248–232 (p = 0.49). Every h2h number from before the fix is unconfirmed.
- **L-cancel RL in the sim raises L-cancel from ~91% to 94–97%, but no
  checkpoint has a strength gain over BC.** The one apparent win (B2 u25, 57% on
  the old harness) came out 247–233 (51.5%, p = 0.55) on the fixed harness.
  Pushing to ~97% cost strength even with a KL anchor. As a test that the RL
  loop learns what it's rewarded for, it works.
- **The trainer's intermittent CUDA fault is a cuBLAS SGEMM launched with a bad
  address after CUDA-graph rollouts.** `CUBLAS_WORKSPACE_CONFIG=:0:0` stops it
  (0 crashes in 34 updates vs one every ~5–10). Fixed in `ecaebaf`.

## Native-sim h2h harness: bias found and fixed

`tools/sim_h2h.py` before `27bda08`:

- A's side came from env parity, and env `e` uses template `e % 6`. With an
  even number of templates (6), each template was only ever played in one
  orientation.
- The report kept the first N games to finish across 48 concurrent envs, which
  favours short games.

Identical-model controls (BC fox-master vs itself, bf16):

| harness | seed | games | A–B | A win rate | p |
|---|---|---|---|---|---|
| old | 0 | 300 | 151–149 | 50.3% | 0.95 |
| old | 11 | 400 | 225–175 | 56.2% | 0.014 |
| old | 12 | 400 | 201–199 | 50.2% | 0.96 |
| **fixed** | 11 | 480 | 248–232 | 51.7% [47.2, 56.1] | 0.49 |

On seed 11, A led in 5 of 6 templates and on both sides of the board. Seed 12
showed no advantage for the first-listed policy. So the 56% is most likely
chance on a confounded design, not a systematic ordering effect. The fixed
harness removes both confounds anyway.

The fixed harness (`27bda08`):

- every template is played equally often in both orientations (envs rounded to
  a multiple of 2 × templates);
- each env contributes the same number of games;
- the games run in two passes with the policies registered in opposite orders;
- the report includes by-order and by-template/side records.

Because every env plays the same number of games, `--games 400` gives 480 games
and `--games 1000` gives 1,056.

Lesson: one 50% control on one seed did not show the rig was unbiased. Run a
same-model control on the seed and harness actually used, and read the
breakdowns.

## Cond-heads vs champion and vs control

Cond-heads is `fox-20261020-condheads`, averaged over its last 5 checkpoints
(tail-SWA) on the vast pod; the champion is `hf_eval/fox-master` (`AVG_c9nm`).
Both ran in fp32, the Dolphin inference numerics.

| harness | games | cond–champ | cond win rate | p | stock margin |
|---|---|---|---|---|---|
| old (seed 21) | 1,000 | 545–454 (1 draw) | 54.6% [51.5, 57.6] | 0.004 | +0.20 |
| **fixed** (seed 21) | 1,056 | **610–446** | **57.8% [54.8, 60.7]** | 5×10⁻⁷ | +0.33 |

**Against `control`** (`fox-20261020-control`: same code, recipe, data and seed,
no `--cond-heads`), fixed harness, bf16, 1,056 games: **611–445, 57.9%
[54.9, 60.8], p ≈ 4×10⁻⁷**, stock margin +0.27. The record was 304–224 with
cond-heads listed first and 307–221 with control first. Cond-heads deals about
22 more damage per game than it takes (difference of total damage dealt, from
the recorded games). This is the clean A/B: the gain comes from the flag.

Val loss on a fixed held-out set (`train.py --eval-only --seed 0`, batch 768,
100 batches from the 3 local val shards): control **0.6803**, cond-heads
**0.6519** (−0.0284, −4.2%). In-training val curves show the gap at 3.6–4.0%
from 1% of training (step 2.8k) to the end.

On the fixed harness, cond-heads won 292–236 when listed first and 318–210 when
the champion was listed first. It led in 10 of 12 template/side cells; the
other two were 45–43 and 43–45. L-cancel was 91.2% for both models, so the gain
comes from somewhere else.

Caveats:

- Not yet checked in Dolphin. Needed before promoting.
- Sim only. One Dolphin check of the unannealed step-28k checkpoint (2 games)
  showed nothing broken: L-cancel 77% (n=44), center-stick 42%.

**Precision.** Production Dolphin inference is fp32: `inference_utils.py`,
`play.py`, `play_netplay.py` and `discord_bot.py` contain no autocast or bf16.
The two cond-heads runs above used fp32 to match it. bf16 and fp32 play even in
the sim (144–155, 48.2%, p = 0.56, 2026-10-20). From here on, sim h2h evals use
bf16, which is `sim_h2h`'s default and about 2× faster. That includes cond-heads
vs `control`.

## L-cancel RL in the native sim

Trainer: `tools/sim_rl.py` (`ecaebaf`).

- **Rollouts:** the learner plays Fox dittos against frozen BC across 64 envs,
  with ports alternated. The learner runs bf16 inside a CUDA graph that reads its
  live weights.
- **Episodes and reward:** the `l_cancel_online` definitions
  (`rlvr/sim/lcancel.py`), fed by a per-frame info array from the workers.
- **Update:** PPO via `ppo_update`, 4 epochs, with old log-probs recomputed in
  fp32.

Defaults: lr 3e-5, kl_beta 0.003, 192 episodes per update.

Arms (all on fox-master; screens are 75 games vs BC, seed 7, **old harness**):

| arm | change | outcome |
|---|---|---|
| A0 | baseline | L-cancel 89%→97–100%; KL to BC 0.05 → 1.5 by u40. Screens 55% (u5) → 0% (u30–u45). Aerial landings per 75 games fell from ~2,300 to ~400 (it stops doing aerials). Crashed at u50. |
| B1 | `--misses-per-update 12` (batch grows ~1/miss-rate) | slower collapse: 52% (u5) → 12% (u30); L-cancel 98.8% at u30 |
| B2 | B1 + lr 1e-5 | 48–59% through u30; L-cancel ~94–95% |
| B3 | B1 + kl_beta 0.03 | 37–61%, 33% at u30; L-cancel 98.3% |
| G1 | B1 + `--global-kl-beta 0.05` (exact 4-head KL to BC on 512 random learner frames) | 55% → 32–40% from u20; L-cancel 98.3% at u30 |
| G2 | B1 + `--global-kl-beta 0.5` | 39–55% through u30, no downward trend; L-cancel 94–95% |
| G2L | G2 run to 60 updates | 39–60% at u35–u60 (75 games each); L-cancel 95.5–97.4% |

400-game finals vs BC (seed 11, **old harness, unconfirmed**):

| checkpoint | RL–BC | win rate | p | L-cancel RL / BC |
|---|---|---|---|---|
| B2 u25 | 228–172 | 57.0% | 0.006 | 94.5 / 90.7 |
| B2 u25, replication (seed 31) | 206–194 | 51.5% | 0.58 | 94.1 / 91.3 |
| B2 u30 | 201–199 | 50.2% | 0.96 | 95.0 / 91.3 |
| G1 u15 | 200–200 | 50.0% | 1.00 | 94.6 / 90.8 |
| G2 u25 | 207–193 | 51.7% | 0.52 | 94.7 / 91.1 |
| B3 u15 | 182–218 | 45.5% | 0.08 | 94.2 / 91.1 |
| G2L u60 | 179–221 | 44.8% | 0.04 | 97.1 / 90.9 |
| BC + 0.5·(G2L u60 − BC) | 214–186 | 53.5% | 0.18 | 94.7 / 90.9 |

The last row is a halfway weight blend of BC and G2L u60 (WiSE-FT).

Fixed-harness rerun (480 games vs BC, bf16): **B2 u25 247–233, 51.5% [47.0, 55.9],
p = 0.55.** The blend and G2L u60 reruns were stopped as low value once this
came in.

Reading:

- At ~94–95% L-cancel the policy plays about even with BC.
- B2 u25's 57% didn't replicate on seed 31, and seed 11 is the seed where the
  old harness's identical-model control came out 56%.
- Pushing to ~97% cost strength even with the 0.5 anchor. Without it, the
  policy collapsed (A0, B1).
- The blend kept about half of G2L's L-cancel gain and lost its strength deficit.

This matches CLAUDE.md's "error consequence is non-uniform" point:
L-cancel is worth little to win rate, so a pure L-cancel reward can only cost
strength once it changes behaviour beyond timing.

As a test of the RL loop itself, it worked: the reward rose within 10–30
updates, the KL penalties behaved as designed, and the unanchored runs showed
the expected reward hacking.

## RoPE vs relpos (`mimic-rope-matched`): dropped

`fox-20261020-ropematched` is `control` with only the position encoding changed.
`mimic-rope-matched` has the same block, projections and dropout sites as `mimic`,
with RoPE on Q/K and a fused SDPA kernel instead of Shaw relative-position
attention. Same data, seed, batch, schedule and steps.

In-training val loss at matched steps (each eval draws its own val sample, so
single points wobble; the trend across evals is what counts):

| step | control | ropematched | rope − control |
|---|---|---|---|
| 2.8k (1%) | 0.8051 | 0.8066 | +0.0015 (+0.2%) |
| 14k (5%) | 0.7543 | 0.7561 | +0.0018 (+0.2%) |
| 28k (10%) | 0.7361 | 0.7406 | +0.0045 (+0.6%) |
| 56k (20%) | 0.7245 | 0.7323 | +0.0078 (+1.1%) |
| 112k (40%) | 0.7056 | 0.7180 | +0.0124 (+1.8%) |
| 174k (62%) | 0.7006 | 0.7098 | +0.0092 (+1.3%) |

**How strong the evidence is:**

- **Consistent.** RoPE was worse at all 62 matched evals. After 20% of
  training it was worse at 43 of 43, by +0.0096 on average (1.3%).
- **Large relative to measurement jitter.** The eval-to-eval wobble of a fixed
  difference between two runs (control minus cond-heads, after 20%) has an SD of
  0.0014, so the RoPE gap is about 7× that.
- **Larger than seed noise, but only one seed.** The June study put the spread
  between seeds at full size at about 0.004 (0.5%), so the gap is about 2.4× that.
  Each arm is a single training seed.
- **Not run to the end.** It was stopped at step 175k of 280k, before most of
  the cosine LR decay. The final gap could shrink or grow; at the last eval it
  was still about 1.3%.
- **No h2h.** Val loss only; no averaged checkpoint was made.
- **It agrees with history.** Earlier RoPE presets also underperformed. The
  10-20c memo found those comparisons confounded (seq 60 vs 256); this one isn't.
- **It's a gradual effect.** The gap is near zero for the first 5% of training
  and opens over time. Short proxy runs would have called RoPE a tie.

Conclusion: RoPE as implemented is worse than Shaw relpos for this model, by
about 1–2% val loss. The evidence is strong enough to stop spending pod time on
it, but not a final number. What RoPE buys is speed: 11.2 vs 8.4 steps/s in
training (+33%), and fused-kernel inference (about 2×, measured 2026-10-20).
Relpos stays.

## Proxy setup for fast research iteration (calibration in progress)

The goal: a small local training run (RTX 4090) that ranks changes the way the
9-hour pod runs do, with clear single-seed signal.

Setup:
- Data: 16 of 342 master-Fox shards (`data/mastfox_c9_half`, about 5%).
- Recipe: champion recipe, eff batch 768, scaled down.
- Scoring: tail-SWA of the last 5 checkpoints, then fixed-set val loss.
- Script: `tools/proxy_run.sh`.

Three equal-wall-clock cells: 5M × 13k steps, 11M × 9k, 20M × 6.5k.

| Model | Baseline val | Cond-heads val | Gain |
|---|---|---|---|
| 5M, 13k steps (40 min) | 0.7807 | 0.7472 | −4.3% |
| 11M, 9k steps (50 min) | 0.7747 | 0.7427 | −4.1% |
| 20M full run (9 h, pod) | 0.6803 | 0.6519 | −4.2% |

So far:

- **The proxies reproduce the full-size gain in relative terms.**
- **The 5M gain shows no in-game edge.** A 96-game sim match between the 5M
  pair was a tie (44–52; damage +0.8 ± 9.6). At this size the val gain doesn't
  show in play, so h2h sanity checks need a bigger proxy.
- **Pending:** the 20M proxy cell, and second baseline seeds for the noise
  floor.

Whole-game h2h metrics all need about 300 games for an 8-point edge:

| Metric | Games needed (cond-heads vs control) |
|---|---|
| Win/loss | ~310 |
| Stock margin | ~390 |
| Damage dealt minus taken | ~280 |

## Trainer crash: cuBLAS SGEMM bad address after CUDA-graph rollouts

**Symptom.** `sim_rl.py` died intermittently with "CUDA error: misaligned
address" or "an illegal memory access". The error always surfaced at the first
`loss.item()` in `ppo_update`. It happened about once per 5–10 updates while
other GPU jobs ran, and less often alone: B1/B2/B3/G1/G2 did 150 updates
without a crash.

**Diagnosis:**

1. **Kernel log.** `journalctl -k` showed Xid 13 ("Out Of Range Address" /
   "Misaligned Address") for every crash, plus Xid 43 naming the trainer's PID.
   The ESR words were identical across all five crashes (same instruction), so
   it is software, not hardware.
2. **Freed memory ruled out.** `torch.cuda.memory_stats()` showed 0 alloc
   retries and no `cudaFree` during training (`num_device_free` fixed at 37 from
   startup).
3. **Stress test.** A standalone test with graph-captured rollouts and the real
   `ppo_update`, but constant fake features, ran 41 updates without a crash.
4. **memcheck.** A real update saved just before `ppo_update` and replayed
   outside the sim under `compute-sanitizer --tool memcheck` (caching allocator
   off) gave 0 errors. The update's own kernels are clean; the fault needs the
   rollout context.
5. **Core dump.** A GPU core dump (`CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1`) caught
   it on the first update. To read it, take the faulting grid's function entry
   from `.cudbg.gridtbl` and match it against the symbol tables of the dump's
   relocated module images (`.cudbg.relfimg`). Result: cuBLAS
   `cutlass_80_simt_sgemm_128x256_8x4_nt_align1` at instruction offset 0xd0,
   the only grid running. So an fp32 GEMM in the PPO update was launched with a
   bad pointer.
6. **Fix test.** With `CUBLAS_WORKSPACE_CONFIG=:0:0`, 34 updates and 0 crashes
   under the same contention; the default setting crashed on update 1 in the
   core-dump run. Updates were no slower (7.3 vs 9.9 ms/frame).

**Mechanism: inferred, not proven.** The CUDA graphs and eager training share
cuBLAS's per-stream workspace. PyTorch's current `CublasHandlePool.cpp` carries
a warning that captured kernels keep their workspace address, so a graph must
not share scratch memory with eager work.

**Misleading signals:** `CUDA_LAUNCH_BLOCKING=1` and extra syncs before the
update also suppressed it (17–20 clean updates), which looks like a fix but is
only timing.

`sim_rl.py` now sets the variable before torch loads (`ecaebaf`). `--resume`
plus saved optimizer state (`opt_uNNNN.pt`) were added along the way.

## Pod queue (vast, 2×5090)

- `fox-20261020-condheads` finished 13:31 UTC; `fox-20261020-control` finished
  22:45 UTC. Both averaged checkpoints were pulled.
- `fox-20261020-ropematched` was stopped by the owner at step 175k (03:08 UTC
  10-22); see above. The pod queue was restarted as `run_queue3.sh` (wdfix only).
- `fox-20261020-wdfix` started 03:08 UTC 10-22 at 8.4 steps/s, ETA ~12:20 UTC.
  Auto top-up is on, so the balance isn't a constraint.

## Next

- `wdfix` vs `control`: fixed-set val loss, then a sim h2h, when it lands.
- Cond-heads in Dolphin before any promotion.
- Proxy calibration (local 4090) to set up a fast research loop: see the
  proxy notes when the calibration finishes.

## Files

- `checkpoints/simrl_sweep_20261021/`: RL arms, `ladder.txt` (screens),
  `finals.txt`, `sweep.log`
- `checkpoints/h2h_rerun_20261021/results.txt`: fixed-harness control and reruns
- `checkpoints/box_20261020/eval/condheads_avg/`: old-harness cond-heads h2h
