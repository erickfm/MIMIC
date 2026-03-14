# MIMIC Experiment Results

All runs: 2M samples from data/full, best val = lowest val/total across checkpoints, train = final logged training loss.

## Phase 1-2: Encoder & Backbone

Default encoder, various presets and LRs.

### Scaling (model size)

| Run | Preset | Params | LR | Batch | train/total | val/total | val/cdir | val/btn |
|-----|--------|--------|----|-------|-------------|-----------|----------|---------|
| `scale-tiny` | tiny (256d/4L) | 6.4M | 1e-3 | 128 | 0.0999 | 0.0742 | 99.2% | 99.4% |
| `scale-small` | small (512d/4L) | 16.3M | 3e-4 | 128 | 0.0929 | 0.0929 | 98.6% | 99.2% |
| `scale-medium` | medium (768d/4L) | 32.4M | 8e-4 | 128 | 0.1000 | 0.0797 | 99.1% | 99.4% |
| `scale-base` | base (1024d/4L) | 54.9M | 1e-3 | 64 | 0.1120 | 0.0736 | 99.1% | 99.8% |
| `scale-deep` | deep (512d/8L) | 28.9M | 3e-4 | 64 | 0.1193 | 0.0876 | 98.8% | 99.1% |
| `scale-wide-shallow` | 1024d/2L | 62.0M | 3e-4 | 32 | 0.1138 | 0.0904 | 98.5% | 99.4% |
| `scale-xlarge` | xlarge | 105.3M | 1e-4 | 32 | 0.0261 | 0.1084 | 98.3% | 99.4% |
| `scale-xxlarge` | xxlarge | 232.0M | 1e-4 | 16 | 0.1076 | 0.0908 | 98.9% | 99.5% |

xlarge train loss (0.0261) way below val (0.1084) = overfitting at bs=32/62.5k steps. Medium and base hit the best val loss-per-param trade-off.

### Context length (small preset, 512d/4L)

| Run | Seq len | Batch | train/total | val/total | val/cdir | val/btn |
|-----|---------|-------|-------------|-----------|----------|---------|
| `ctx-60` | 60 | 128 | 0.0801 | 0.0801 | 99.1% | 99.3% |
| `ctx-240` | 240 | 32 | 0.1472 | 0.1278 | 99.2% | 98.8% |
| `ctx-360` | 360 | 16 | 0.0035 | 0.0966 | 100.0% | 99.2% |

ctx-360 massively overfit (train 0.0035 vs val 0.0966) -- bs=16 for 125k steps on a small model. The 100% cdir_acc on val is likely noise from small eval batches.

### Positional encoding (small preset)

| Run | Method | train/total | val/total | val/cdir | val/btn |
|-----|--------|-------------|-----------|----------|---------|
| `posenc-learned` | Learned | 0.0938 | 0.0876 | 99.0% | 99.3% |
| `posenc-rope` | RoPE | 0.1123 | 0.0848 | 99.6% | 99.5% |
| `posenc-sinusoidal` | Sinusoidal | 0.1249 | 0.0893 | 99.2% | 99.2% |

RoPE slightly better than learned at small scale. Sinusoidal worst.

### Attention variants (small preset)

| Run | Variant | train/total | val/total | val/cdir | val/btn | Notes |
|-----|---------|-------------|-----------|----------|---------|-------|
| `attn-alibi` (A) | ALiBi | 0.0971 | 0.0834 | 99.3% | 99.3% | |
| `attn-alibi` (C) | ALiBi | 0.0068 | 0.0067 | 100.0% | 100.0% | Overfit or bug -- ignore |
| `attn-gqa` | GQA | 0.0976 | 0.0957 | 98.9% | 99.1% | |
| `attn-sliding` | Sliding window | 0.1546 | 0.0405 | 99.6% | 99.8% | val < train is suspicious |

### Phase 1-2 takeaways

- **hybrid16** encoder won over flat, composite8, and default (55-token) encoders
- **medium** backbone (768d/4L) hit the best loss-per-param sweet spot at ~32M params
- **lr 8e-4** outperformed 3e-4 and 5e-4 at medium scale
- Larger models (xlarge, xxlarge) overfit at 2M samples with small batch sizes
- RoPE slightly beat learned pos-enc at small scale; ALiBi showed promise but inconsistent results

---

## Phase 3: Architecture & Loss Refinement

medium (768d/4L/3072ff) + hybrid16 encoder, lr=8e-4, 2M samples, data/full.

### Depth

| Run | Layers | Params | Batch | step/s | train/total | val/total | val/cdir | val/btn |
|-----|--------|--------|-------|--------|-------------|-----------|----------|---------|
| `depth-2L` | 2 | 18.3M | 512 | 2.3 | 0.1077 | 0.0874 | 99.2% | 99.4% |
| `baseline` | 4 | 32.4M | 384 | 3.0 | 0.0862 | **0.0680** | **99.3%** | **99.6%** |
| `depth-6L` | 6 | 46.6M | 256 | 4.1 | 0.0981 | 0.0854 | 99.2% | 99.5% |
| `depth-8L` | 8 | 60.8M | 192 | 4.8 | 0.1094 | 0.0885 | 98.9% | 99.5% |

4 layers wins. Deeper models have higher train loss too -- not enough steps to converge at 2M samples with smaller batches.

### Width

| Run | d_model | Params | Batch | step/s | train/total | val/total | val/cdir | val/btn |
|-----|---------|--------|-------|--------|-------------|-----------|----------|---------|
| `width-512` | 512 | 16.3M | 512 | 2.5 | 0.0901 | 0.0860 | 99.3% | 99.4% |
| `baseline` | 768 | 32.4M | 384 | 3.0 | 0.0862 | **0.0680** | **99.3%** | **99.6%** |
| `width-1024` | 1024 | 54.9M | 256 | 4.4 | 0.1136 | 0.0870 | 99.1% | 99.4% |

768 is the sweet spot. 1024 has higher train loss (underfitting -- fewer steps at bs=256).

### Context Length

| Run | Seq len | Time | Batch | step/s | train/total | val/total | val/cdir | val/btn |
|-----|---------|------|-------|--------|-------------|-----------|----------|---------|
| `ctx-30` | 30 | 0.5s | 768 | 0.8 | 0.0975 | 0.0887 | 99.0% | 99.4% |
| `baseline` | 60 | 1.0s | 384 | 3.0 | 0.0862 | 0.0680 | 99.3% | 99.6% |
| `ctx-90` | 90 | 1.5s | 256 | 1.9 | 0.0880 | 0.0802 | 99.0% | 99.5% |
| `ctx-120` | 120 | 2.0s | 192 | 2.3 | 0.0830 | 0.0830 | 99.1% | 99.4% |
| `ctx-180` | 180 | 3.0s | 128 | 2.9 | 0.1072 | **0.0551** | **99.6%** | **99.6%** |

ctx-180 wins decisively (val 0.0551). Train loss is higher (0.1072) because bs=128 means fewer gradient steps, but the model generalizes better with 3 seconds of context. No overfitting -- healthy train > val gap.

### Positional Encoding

| Run | Method | Batch | step/s | train/total | val/total | val/cdir | val/btn |
|-----|--------|-------|--------|-------------|-----------|----------|---------|
| `baseline` | Learned | 384 | 3.0 | 0.0862 | **0.0680** | **99.3%** | **99.6%** |
| `pos-rope` | RoPE | 384 | 1.4 | 0.0709 | 0.0709 | 99.2% | 99.5% |

Learned edges out RoPE (0.0680 vs 0.0709). RoPE has lower train loss but doesn't generalize as well at 60-frame context. Worth retesting at 180 frames where extrapolation matters more.

### Loss Functions

| Run | Stick loss | Btn loss | Batch | step/s | train/total | val/total | val/cdir | val/btn |
|-----|-----------|----------|-------|--------|-------------|-----------|----------|---------|
| `baseline` | MSE | BCE | 384 | 3.0 | 0.0862 | 0.0680 | 99.3% | 99.6% |
| `loss-huber` | Huber | BCE | 384 | 1.6 | 0.0752 | **0.0610** | **99.3%** | 99.5% |
| `loss-discrete` | Discrete 32x32 | BCE | 384 | 1.4 | 1.5768 | 1.4346 | 98.5% | 98.7% |
| `loss-focal-btn` | MSE | Focal BCE | 384 | 1.5 | 0.0631 | 0.0631 | 99.0% | 99.5% |

Huber beats MSE on both train and val -- more robust to outlier stick values. Focal BCE matches on val with lower train loss. Discrete loss not comparable on total (CE vs MSE scale).

### Observations

- **Context is king**: ctx-180 (3s) delivered the best val/total of any run (0.0551), beating even loss-function improvements. The model benefits from seeing longer gameplay sequences -- approach patterns, combo follow-ups, and recovery trajectories span multiple seconds.
- **Depth/width**: 4L/768d baseline is hard to beat at 2M samples. Deeper/wider models don't have enough steps to converge (train loss still high).
- **Positional encoding**: Learned beats RoPE at 60 frames. RoPE worth retesting at 180 frames.
- **Loss**: Huber is the best stick loss (0.0610 vs 0.0680 MSE). Focal BCE competitive for buttons.
- **No overfitting**: Train/val gaps are healthy across Phase 3. The Phase 1-2 runs with tiny batch sizes (xlarge bs=32, ctx-360 bs=16) overfit badly.

---

## Summary

Best configuration from all experiments:

| Setting | Winner | Runner-up |
|---------|--------|-----------|
| Depth | 4 layers | 6 layers (close) |
| Width | 768 (d_model) | 512 (close, half the params) |
| Context | 180 frames (3s) | 60 frames |
| Pos encoding | Learned | RoPE (close) |
| Stick loss | Huber | MSE |
| Button loss | BCE | Focal BCE (close) |

Top 5 runs by val/total:

| Rank | Run | val/total | train/total | val/cdir | val/btn | Config |
|------|-----|-----------|-------------|----------|---------|--------|
| 1 | `ctx-180` | **0.0551** | 0.1072 | 99.6% | 99.6% | medium, hybrid16, lr=8e-4, 180 frames |
| 2 | `loss-huber` | **0.0610** | 0.0752 | 99.3% | 99.5% | medium, hybrid16, lr=8e-4, Huber, 60 frames |
| 3 | `loss-focal-btn` | **0.0631** | 0.0631 | 99.0% | 99.5% | medium, hybrid16, lr=8e-4, Focal BCE, 60 frames |
| 4 | `baseline` | **0.0680** | 0.0862 | 99.3% | 99.6% | medium, hybrid16, lr=8e-4, 60 frames |
| 5 | `pos-rope` | **0.0709** | 0.0709 | 99.2% | 99.5% | medium, hybrid16, lr=8e-4, RoPE, 60 frames |

## Phase 4: No Opponent Inputs (in progress)

**Hypothesis**: Opponent controller inputs (analog stick, buttons, c-stick direction) are effectively dead when playing against a CPU at inference time (all neutral/zero), creating a train-test distribution mismatch. Removing them forces the model to rely on observable game state (positions, actions, flags) rather than opponent intent signals that won't exist during deployment.

**Changes**: `--no-opp-inputs` flag strips `opp_buttons`, `opp_analog`, `opp_c_dir` (and Nana equivalents) from feature groups. Encoder drops the OPP_INPUT composite token (16 → 15 tokens in hybrid16). Button/nana-button encoders take only self inputs (12-dim vs 24-dim).

**Sweep**: 21 runs across 21 GPUs, medium (768d/4L) + hybrid16 + learned pos-enc + MSE stick + BCE btn.

| Context | Seeds | Steps | Duration | Samples |
|---------|-------|-------|----------|---------|
| 60 (bs=384) | s1-s4 | 65,000 | ~6h | ~25M |
| 60 (bs=384) | s5-s8 | 80,000 | ~7.4h | ~31M |
| 60 (bs=384) | s9-s10 | 120,000 | ~11h | ~46M |
| 60 (bs=384) | s11 | 260,000 | ~24h | ~100M |
| 180 (bs=128) | s1-s4 | 65,000 | ~6.2h | ~8.3M |
| 180 (bs=128) | s5-s7 | 80,000 | ~7.7h | ~10.2M |
| 180 (bs=128) | s8-s9 | 120,000 | ~11.5h | ~15.4M |
| 180 (bs=128) | s10 | 250,000 | ~24h | ~32M |

Wandb group: `no-opp-inputs`. Results pending.
