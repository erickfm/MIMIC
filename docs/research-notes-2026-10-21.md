# 2026-10-21 — cond-heads beats the champion in the sim; L-cancel RL in the sim; h2h harness bias; trainer crash

## TL;DR

- **Cond-heads (`--cond-heads`, champion recipe otherwise) beats the fox-master
  champion 610–446 in the native sim** (57.8%, 95% CI 54.8–60.7%, p = 5×10⁻⁷,
  1,056 fp32 games on the fixed harness). The win holds with either policy
  listed first, and L-cancel is equal (91.2% both). The clean A/B against the
  same-code `control` run is pending (pod, ~22:50 UTC).
- **The old `sim_h2h` harness was biased.** An identical-model control gave
  225–175 (p = 0.014). Fixed in `27bda08`. The fixed harness's control gives
  248–232 (p = 0.49). Every h2h number from before the fix is unconfirmed.
- **L-cancel RL in the sim raises L-cancel from ~91% to 94–97%, but no
  checkpoint has a confirmed strength gain over BC.** Pushing to ~97% cost
  strength even with a KL anchor. The fixed-harness reruns are pending.
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

## Cond-heads vs champion

Cond-heads is `fox-20261020-condheads`, averaged over its last 5 checkpoints
(tail-SWA) on the vast pod; the champion is `hf_eval/fox-master` (`AVG_c9nm`).
Both ran in fp32, the Dolphin inference numerics.

| harness | games | cond–champ | cond win rate | p | stock margin |
|---|---|---|---|---|---|
| old (seed 21) | 1,000 | 545–454 (1 draw) | 54.6% [51.5, 57.6] | 0.004 | +0.20 |
| **fixed** (seed 21) | 1,056 | **610–446** | **57.8% [54.8, 60.7]** | 5×10⁻⁷ | +0.33 |

On the fixed harness, cond-heads won 292–236 when listed first and 318–210 when
the champion was listed first. It led in 10 of 12 template/side cells; the
other two were 45–43 and 43–45. L-cancel was 91.2% for both models, so the gain
comes from somewhere else.

Caveats:

- The champion was trained in July on the same recipe and shards but older
  code. `fox-20261020-control` (same code, no flag) is the clean comparison.
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

Reading, pending the fixed-harness reruns:

- At ~94–95% L-cancel the policy plays about even with BC.
- B2 u25's 57% didn't replicate on seed 31, and seed 11 is the seed where the
  old harness's identical-model control came out 56%.
- Pushing to ~97% cost strength even with the 0.5 anchor. Without it, the
  policy collapsed (A0, B1).
- The blend kept about half of G2L's L-cancel gain and lost its strength deficit.

This matches CLAUDE.md's "error consequence is non-uniform" point:
L-cancel is worth little to win rate, so a pure L-cancel reward can only cost
strength once it changes behaviour beyond timing.

Reruns on the fixed harness (B2 u25, the blend and G2L u60 vs BC, 480 games each)
are running: `checkpoints/h2h_rerun_20261021/results.txt`.

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

- `fox-20261020-condheads` finished 13:31 UTC; its averaged checkpoint was pulled.
- `fox-20261020-control` is at step 226k of 280k, ETA ~22:50 UTC.
- Then `ropematched` (~07:40 UTC 10-22) and `wdfix`. `wdfix` won't finish
  before the balance runs out (~14:45 UTC 10-22) without a top-up.

## Next

- Cond-heads vs `control` on the fixed harness (1,000+ bf16 games) when the
  control checkpoint lands.
- Read the fixed-harness RL reruns. If nothing beats BC, L-cancel-only RL is
  done as a strength lever. Its value is the pipeline, which is now fast,
  faithful and stable, for a consequence-weighted reward.

## Files

- `checkpoints/simrl_sweep_20261021/`: RL arms, `ladder.txt` (screens),
  `finals.txt`, `sweep.log`
- `checkpoints/h2h_rerun_20261021/results.txt`: fixed-harness control and reruns
- `checkpoints/box_20261020/eval/condheads_avg/`: old-harness cond-heads h2h
