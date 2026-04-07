# Agent Prompt: Investigate HAL vs MIMIC Performance Gap

## Goal

Find and fix the remaining differences between HAL (Eric Gu's Melee AI) and our MIMIC reproduction that cause MIMIC to overfit and play worse. HAL 4-stocks level 9 CPU. Our best MIMIC takes 1-2 stocks.

## What's been done

We matched HAL's architecture exactly (26.3M params, verified shape-for-shape against HAL's checkpoint). We matched the data pipeline: HAL normalization, 9-cluster c-stick targets, early_release button encoding, no warmup, cosine LR with eta_min=1e-6. For the first time we can compare val loss directly — HAL scores 1.03 on our val data, our best val is 1.15 (12% gap at peak, widening to 1.6+ by end of training).

## The core problem

**MIMIC overfits where HAL doesn't.** HAL's train-val gap stays at ~0.22 throughout training. Ours starts at ~0.15 and grows to 0.68+. HAL's val loss is stable (0.74-0.86 on its own val set). Ours starts good then degrades steadily.

## Key files

- `docs/research-notes-2026-04-06.md` — full documentation of what's been tried
- `docs/research-notes-2026-04-05.md` — architecture matching details  
- `docs/research-notes-2026-04-04-c.md` — original diff table
- `mimic/model.py` — MIMIC model (CausalSelfAttentionRelPos, HALTransformerBlock, HALPredictionHeads)
- `mimic/frame_encoder.py` — HALFlatEncoder (feature selection, normalization, projection)
- `mimic/dataset.py` — StreamingMeleeDataset (random window sampling, character filter)
- `train.py` — training loop with HAL-style loss computation
- `tools/validate_checkpoint.py` — evaluates HAL and MIMIC checkpoints on same val data
- `tools/run_hal_model.py` — HAL's architecture reimplemented, loads HAL checkpoint, plays via Dolphin
- `tools/run_mimic_via_hal_loop.py` — runs MIMIC checkpoint through HAL's inference loop

## HAL's source code

On Machine E (ssh -p 11335 root@66.222.138.178):
- `/root/hal/hal/training/simple_trainer.py` — training loop
- `/root/hal/hal/training/trainer.py` — base trainer with val_loop, closed_loop eval
- `/root/hal/hal/training/streaming_dataset.py` — `HALStreamingDataset.__getitem__` with `random.choice(["p1","p2"])` perspective selection
- `/root/hal/hal/preprocess/preprocessor.py` — preprocessing: remaps p1/p2 → ego/opponent per sample
- `/root/hal/hal/preprocess/input_configs.py` — feature transforms (normalize, standardize, etc.)
- `/root/hal/hal/preprocess/transformations.py` — `convert_multi_hot_to_one_hot_early_release`
- `/root/hal/hal/training/optim.py` — optimizer setup (AdamW, 2D+ decay)
- `/root/hal/hal/data/process_replays.py` — raw .slp → MDS extraction

## What's been verified as matching

- Architecture: every parameter shape matches HAL's checkpoint
- Normalization: same raw values produce same normalized output (verified numerically)
- Feature order: percent, stock, facing, invulnerable, jumps_left, on_ground, shield, pos_x, pos_y
- Weight init: normal(0,0.02), residual scaling, zeros bias — same as HAL
- Optimizer: AdamW, same param grouping, same betas/eps/wd
- LR schedule: CosineAnnealingLR, no warmup, eta_min=1e-6
- Loss: plain CE on all 4 heads (shoulder, c_stick, main_stick, buttons)
- Dropout: 0.2

## What's known to be different

1. **Perspective selection**: HAL randomly picks p1 or p2 per sample at training time. Our shards have both perspectives as separate fixed entries. Same total data, but HAL's randomization provides implicit augmentation — the model can't predict which perspective it'll see for a given game. This is the most likely cause of the overfitting difference.

2. **Data ordering**: HAL uses Mosaic StreamingDataset with `shuffle=True` (global shuffle across all games). Our IterableDataset shuffles within shards, iterates shards sequentially. Less global randomness.

3. **Val set**: HAL validates on 30 games (4096 samples). We validate on 644 games (6400 samples). HAL's smaller val set may appear more stable.

4. **Frame extraction**: HAL's `process_replays.py` extracts frame data. Our `slp_to_shards.py` extracts independently. Both use melee-py's Console but may differ in subtle frame alignment (e.g., how the first/last frames are handled, controller state timing).

## Training data

On Machine E:
- `/root/slp_fox_public/` — 3,229 Fox .slp files (HAL's exact training data)
- `/root/FRAME/data/fox_hal_norm/` — 168 train shards built with HAL normalization + 9-cluster c-stick + early_release buttons
- `/root/FRAME/data/fox_public_shards/` — older shards (both perspectives, standardized normalization)
- `/root/hal/data/fox_mds/` — HAL's MDS-format training data

## GPU machines

- Machine C: ssh -p 22874 root@194.14.47.19 (8× RTX 5090 32GB)
- Machine E: ssh -p 11335 root@66.222.138.178 (8× RTX 5090 32GB)
- Machine F: ssh -p 18619 root@74.2.96.10 (8× RTX 5090 32GB)

## Specific things to investigate

1. **Implement random perspective selection per sample** — store one entry per game with p1/p2 data, randomly pick perspective at training time. This is the most likely fix for the overfitting gap. HAL's `streaming_dataset.py:_get_item` shows exactly how.

2. **Compare frame extraction** — take the same .slp file, extract through HAL's `process_replays.py` and our `slp_to_shards.py`, compare the raw arrays frame-by-frame. Any difference in frame count, feature values, or alignment could explain the gap.

3. **Check if HAL's MDS data has any augmentation or preprocessing we missed** — look at `add_reward_to_episode`, `compute_returns`, `sample_from_episode` in detail. We assumed these don't affect BC training, but verify.

4. **Profile the val loss curve** — use `validate_checkpoint.py` on checkpoints at multiple steps (13K, 26K, 52K, 78K, 104K, 131K, etc.) to get a fine-grained val loss curve. Compare the shape to HAL's wandb curve (available via `wandb.Api()` on Machine E, project "hal", run "l8wtjjdk").

5. **Test if global data shuffling matters** — implement a DataLoader that shuffles all windows globally (not per-shard). If val loss improves, data ordering is a factor.

## Training command (latest)

```bash
torchrun --nproc_per_node=8 train.py \
  --model hal --encoder hal_flat \
  --hal-mode --hal-minimal-features --hal-controller-encoding \
  --stick-clusters hal37 --plain-ce --no-amp \
  --lr 3e-4 --batch-size 64 \
  --max-samples 16777216 \
  --data-dir data/fox_hal_norm \
  --controller-offset --self-inputs \
  --no-compile --run-name hal-pipeline-match \
  --nccl-timeout 7200 --no-warmup --cosine-min-lr 1e-6
```

## How to validate

```bash
# Compare HAL vs any MIMIC checkpoint on same val data
python tools/validate_checkpoint.py \
  --checkpoint /root/hal/runs/2026-03-30_22-16-34/arch@GPTv5Controller-512-6-8-dropout_local_batch_size@64_n_samples@16777216/000016777216.pt \
  --checkpoint-b checkpoints/my_model.pt \
  --data-dir data/fox_hal_norm \
  --n-batches 64

# HAL baseline on our val: loss_total=1.03
```

## Key insight

The 12% val loss gap at best (1.15 vs 1.03) is small and might not matter much for gameplay. The bigger issue is that our model overfits as training progresses — val rises from 1.15 to 1.6+ while train drops to 0.6. If we fix the overfitting, the model at step 117K (where it has learned strategies) would also have good val loss, and likely play well. The overfitting fix is more important than closing the initial 12% gap.
