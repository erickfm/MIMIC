# 2026-10-22 — faster/cheaper variants of the current arch: Muon, sliding-window + KV cache, step budget

## TL;DR

All local (RTX 4090), single seed, 20M `mimic` proxy (6.5k steps, 5% of master-Fox; the
"C cell" of `tools/proxy_run.sh`), unless noted. Δ = last-frame loss vs the AdamW fp32
baseline `p1-C-base-s1`, paired over 20,000 held-out frames (see "Metrics").

- **Muon (`--optimizer muon --muon-lr 0.02`) is a clear win.** Δ −0.0127 ± 0.0025; fixed-subset
  val 0.7622 vs 0.7749/0.7757 (AdamW seeds 1/2). Muon over 3,500 steps (full schedule) beats
  AdamW over 6,500 (Δ −0.0037 ± 0.0022, val 0.7710): ≥1.8× fewer steps for ~4% more time per
  step. lr 0.04 was worse than AdamW (early-killed at step 2,145); SOAP at lr 3e-3 was far worse
  (early-killed at step 1,300).
- **Sliding-window relpos attention (`--attn-window 180`, trained on 720-frame chunks) ties the
  current model, and enables an exact streaming KV cache (`mimic/kv_cache.py`).** Δ +0.0007 ±
  0.0026 from scratch; −0.0008 ± 0.0010 when the champion is warm-started into it for 2k steps.
  Streaming equals the full forward to 2.5e-7. Per-frame inference, idle GPU, CUDA graphs:
  13.3 → 0.61 ms at batch 64 bf16 (22×), 0.53 → 0.33 ms at batch 1 bf16.
- **The two stack.** Muon + sliding window: Δ −0.0198 ± 0.0031 (Muon alone −0.0185 on the same
  late-frame targets); at 3,500 steps, Δ −0.0137 ± 0.0028 vs 6,500-step AdamW.
- **Step budget: the last ~25–50% of a production run buys little in play.** Native-sim h2h,
  480 games each, raw mid-run checkpoints of `fox-20261020-control` vs its tail-SWA final:
  step 210k 233–247 (48.5%, p = 0.55), step 140k 223–257 (46.5%, p = 0.13). Neither checkpoint
  had LR decay or averaging.
- **Distillation only helps with a teacher that saw more data.** From the champion: Δ −0.0158
  (still −0.010 at equal wall-clock). From a same-data, same-size teacher: Δ −0.0041 but ~30%
  slower per step, so it loses at equal wall-clock. Not a production-recipe speedup.
- **A 2k-step warm start from the champion is a ~3× faster, cleaner research proxy for
  shape-compatible changes.** Cond-heads: −0.0227 ± 0.0015 warm (2k steps) vs −0.0311 ± 0.0027
  from scratch (6.5k steps). Can't test optimizer/init changes.
- **Falsified:** temporal patching (2-frame patches: +0.012 at step 1,430, only ~2% faster),
  multi-token prediction (Δ +0.0010, 5% slower), window 90 (Δ +0.0078), bf16 attention math
  (diverged at ~step 1k), FlexAttention without attention dropout (small cost, ~+0.0025).

## Setup

- **Proxy:** `tools/proxy_run.sh NAME mimic data/mastfox_c9_half 6500 SEED [flags]` — the
  production recipe (eff batch 768, AdamW 3e-4, no warmup, cosine to 1e-6) on 16 of 342
  master-Fox shards; tail-SWA of the last 5 checkpoints; fixed-subset val (seed 0, 100 batches).
  `PROXY_BS` / `PROXY_GA` / `PROXY_EVAL_BS` now override the batch (used for 720-frame chunks:
  96 × 2 chunks = the same 138k tokens per step), and `--max-samples` follows the effective
  batch (it was hardcoded to 768 × steps).
- **Single seed.** Seed-to-seed noise: the two AdamW seeds' in-training val (every 65 steps)
  differ by sd 0.002, max 0.005 after step 1,300; fixed-subset val 0.7749 vs 0.7757; last-frame
  +0.0033 ± 0.0021.
- **Early falsification** (`checkpoints/proxy/earlykill.py`): runs with the same data, seq len,
  steps and val cadence as the baseline are killed once the mean of their last 5 in-training
  val points is > 0.008 worse than the baseline's at the same steps (from step 1,300).
- **Runner:** `checkpoints/proxy/runner.sh` pops `checkpoints/proxy/queue.txt` one line at a
  time.

## Metrics

`train.py`'s val averages the loss over every window position, so position 0 (no history)
weighs as much as position 179. At inference the bot predicts only the newest frame, with full
context. `tools/eval_last_frame.py` scores exactly that: each model on the newest frame only,
with its real inference context (180 frames for full-causal models; the whole receptive field
`num_layers × (W−1) + 1` = 1,075 frames for window-180 models, equal to streaming), on one
fixed set of held-out targets, with paired standard errors. Cond-heads models are scored with
teacher-forced earlier-head classes (the training loss). Last-frame loss is lower than window
val (0.771 vs 0.775 for the baseline) because full-context frames are easier. Comparisons that
include a window-180 model use targets with ≥ 1,075 frames of history (later in the game, so
all losses are lower: baseline 0.748).

## Results

### Scratch proxies (C cell, 6.5k steps unless noted)

| Run | Change | Fixed val | Last-frame Δ vs base | Notes |
|---|---|---|---|---|
| p1-C-base-s1 / s2 | — | 0.7749 / 0.7757 | — / +0.0033 ± 0.0021 | baseline seeds |
| p1-C-muon02-s1 | Muon lr 0.02 | 0.7622 | −0.0127 ± 0.0025 | |
| p1-C-muon3k5-s1 | Muon lr 0.02, 3,500 steps | 0.7710 | −0.0037 ± 0.0022 | 54% of the steps |
| p1-C-muon04-s1 | Muon lr 0.04 | — | — | early-killed, step 2,145, +0.0087 vs AdamW |
| p1-C-soap-s1 | SOAP lr 3e-3 | — | — | early-killed, step 1,300, +0.116 |
| p1-C-flex180-s1 | FlexAttention (no attn dropout) | 0.7773 | +0.0027 ± 0.0021 | |
| p1-C-swa720-s1 | window 180, 720-frame chunks, flex | — | +0.0007 ± 0.0026 * | |
| p1-C-swa720w90-s1 | window 90 | — | +0.0078 ± 0.0026 * | 17% faster step |
| p1-C-muonswa720-s1 | Muon + window 180 | — | −0.0198 ± 0.0031 * | Muon alone −0.0185 * |
| p1-C-muonswa3k5-s1 | Muon + window 180, 3,500 steps | — | −0.0137 ± 0.0028 * | |
| p1-C-patch2-s1 | 2-frame temporal patching | — | — | early-killed, step 1,430, +0.0118; ~2% faster |
| p1-C-mtp1-s1 | multi-token prediction (t+1), weight 0.5 | 0.7743 | +0.0010 ± 0.0022 | 5% slower |
| p1-C-kd-s1 | distill from champion, α = 1 | 0.7593 | −0.0158 ± 0.0022 | ~20% slower/step |
| p1-C-kdself-s1 | distill from p1-C-base-s2 | 0.7702 | −0.0041 ± 0.0019 | ~30% slower/step; beats its teacher by 0.0074 |

\* late-frame targets (≥ 1,075 frames of history). Window-720 runs' own val is over 720-frame
chunks and isn't comparable, so it's omitted.

Equal wall-clock for distillation: kd at step 5,200 (≈ 80% of the steps, its schedule not
finished) had in-training val 0.7610 vs the baseline's 0.7711 at 6,500, so the champion teacher
still wins; kdself at ~step 4,500 (≈ equal time) was 0.7731, so it loses.

### Warm start from the champion (2k steps, lr 1e-4, fresh optimizer and schedule)

| Run | Change | Last-frame Δ vs warmctrl |
|---|---|---|
| p1-C-warmctrl-s1 | — | — |
| p1-C-warmcond-s1 | `--cond-heads` | −0.0227 ± 0.0015 |
| p1-C-warmflex180-s1 | FlexAttention | +0.0008 ± 0.0005 * |
| p1-C-warmswa720-s1 | window 180, 720-frame chunks | −0.0008 ± 0.0010 * |
| champion (`hf_eval/fox-master`), not fine-tuned | — | −0.0050 ± 0.0008 * |

Every fine-tune ended ~0.005 worse than the untouched champion: 2k steps at lr 1e-4 on 5% of
the data costs a little. It applies equally to all variants, so the comparisons stand, but a
real conversion of production models to the sliding window should fine-tune on the full data at
a lower lr.

### Step budget (native sim, bf16, 480 games each, `checkpoints/stepbudget_20261022/`)

| A (raw checkpoint) | vs B | Record | A win rate [95%] | p | Stock margin |
|---|---|---|---|---|---|
| control step 140k | control tail-SWA final (224k–280k) | 223–257 | 46.5% [42.0, 50.9] | 0.13 | −0.20 |
| control step 210k | same | 233–247 | 48.5% [44.1, 53.0] | 0.55 | −0.04 |

The raw checkpoints are mid-cosine (high LR, not averaged), so this understates what a shorter
run with a full decay would reach. The fixed harness's same-model control is 248–232 (p = 0.49).

### Training-step speed (box, 2×5090, production config, fixed batch; script at `checkpoints/box_20261022/relpos_proxies/mine/bench.py`)

| Relpos attention path | DDP steps/s | Matches fp32? |
|---|---|---|
| fp32 (current) | 9.26 | — |
| SDPA with the Shaw term as an additive mask | 10.55 (+14%) | loss to 3e-6, grads 8e-4 rel (fp32, no dropout); proxy val 0.7895 vs 0.7892 |
| FlexAttention | 10.73 (+16%) | no attention dropout |
| bf16 matmuls | +7% | diverged in training |

Also measured: data loading 48 batches/s vs ~10 needed; DDP overhead ~4%; the per-step
`.item()` sync costs nothing (GPU-bound); validation (raw + EMA, every 2,800 steps) is ~6% of
production wall-clock and the per-step EMA loop ~2%. In the short proxies validation is ~20% of
wall-clock (val every 1% of steps).

### Inference (idle GPU, per frame, `mimic/kv_cache.py` vs recomputing the 180-frame window)

| Batch | Precision | Eager: window → KV | CUDA graph: window → KV |
|---|---|---|---|
| 1 | fp32 | 1.18 → 1.06 ms | 0.87 → 0.35 ms (2.5×) |
| 1 | bf16 | 1.47 → 1.28 ms | 0.53 → 0.33 ms (1.6×) |
| 64 | fp32 | 22.6 → 1.14 ms | 22.4 → 0.82 ms (27×) |
| 64 | bf16 | 13.4 → 1.40 ms | 13.3 → 0.61 ms (22×) |

## Code (all off by default; the default path is bit-identical to the previous HEAD)

- `--relpos-impl {fp32,bf16,sdpa,flex}`: training-only kernel for the relpos attention; eval and
  inference always use fp32.
- `--attn-window N` (+ `--seq-len`): sliding-window relpos in every path. The fp32 windowed path
  scores by distance, so lengths beyond `block_size` (1,024) work.
- `mimic/kv_cache.py` — `RelPosKVCache`: ring buffer, one write per layer per frame, all
  per-step state on device (CUDA-graph capturable), optional bf16 cache.
- `tools/eval_last_frame.py`: inference-faithful paired last-frame eval.
- `--patch P`, `--mtp K` / `--mtp-weight`, `--distill-from` / `--distill-alpha`: the falsified
  or conditional variants above, kept for reproduction.
- `FramePredictor.features()`: encoder + backbone + final norm, the input to the heads.

## Not done / test status

- Single seed everywhere; proxy scale only. Nothing here is h2h-validated in play except the
  step budget.
- Multi-character pretraining (Falco/Marth/Sheik, 60 shards, 33M frames → Fox fine-tune at
  6.5k and 2k steps) was still running when this was written.
- Fox all-ranks pretraining wasn't testable: only metadata is local.
- SDPA relpos was only tested on the (now destroyed) box.

## Next

- Production candidate: Muon + window 180 on 720-frame chunks at a reduced step budget, then
  h2h vs the champion (fp32, Dolphin canaries).
- Convert existing production characters to the sliding window by fine-tuning (full data,
  lower lr) to get KV-cache inference without a retrain; wire `RelPosKVCache` into the sim
  runner and `tools/inference_utils.py`.
- Use 2k-step warm starts from the champion for shape-compatible research changes.
