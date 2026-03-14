# FRAME Experiment Results

## Phase 1-2: Encoder & Backbone (completed)

Ran on full dataset. Key findings that informed Phase 3 defaults:

- **hybrid16** encoder won over flat, composite8, and default (55-token) encoders
- **medium** backbone (768d/4L) hit the best loss-per-param sweet spot at ~32M params
- **lr 8e-4** outperformed 3e-4 and 5e-4 at this scale
- Longer context helped (ctx-180, ctx-240 beat ctx-60) but with diminishing returns
- ALiBi positional encoding showed promise but had a bug (fixed mid-run)

## Phase 3: Architecture & Loss Refinement

Baseline: medium (768d/4L/3072ff) + hybrid16 encoder, lr=8e-4, 2M samples, data/full

Best validation metrics reported (lowest val/total across checkpoints).

### Axis 1: Depth

| Run | Layers | Params | Batch | step/s | val/total | val/cdir_acc | val/btn_acc | Notes |
|-----|--------|--------|-------|--------|-----------|--------------|-------------|-------|
| `depth-2L` | 2 | 18.3M | 512 | 2.3 | 0.0874 | 99.2% | 99.4% | Done |
| `baseline` | 4 | 32.4M | 384 | 3.0 | **0.0680** | **99.3%** | **99.6%** | Done |
| `depth-6L` | 6 | 46.6M | 256 | 4.1 | 0.0854 | 99.2% | 99.5% | Done |
| `depth-8L` | 8 | 60.8M | 192 | 4.8 | 0.0885 | 98.9% | 99.5% | Done |

4 layers wins. More depth doesn't help at 2M samples -- the larger models are likely underfitting on compute, not data.

### Axis 2: Width

| Run | d_model | Params | Batch | step/s | val/total | val/cdir_acc | val/btn_acc | Notes |
|-----|---------|--------|-------|--------|-----------|--------------|-------------|-------|
| `width-512` | 512 | 16.3M | 512 | 2.5 | 0.0860 | 99.3% | 99.4% | Done |
| `baseline` | 768 | 32.4M | 384 | 3.0 | **0.0680** | **99.3%** | **99.6%** | Done |
| `width-1024` | 1024 | 54.9M | 256 | 4.4 | 0.0870 | 99.1% | 99.4% | Done |

768 is the sweet spot. 512 is close but slightly worse; 1024 is no better and 2x the params.

### Axis 3: Context Length

| Run | Seq len | Time | Batch | step/s | val/total | val/cdir_acc | val/btn_acc | Notes |
|-----|---------|------|-------|--------|-----------|--------------|-------------|-------|
| `ctx-30` | 30 | 0.5s | 768 | 0.8 | 0.0887 | 99.0% | 99.4% | Done |
| `baseline` | 60 | 1.0s | 384 | 3.0 | 0.0680 | 99.3% | 99.6% | Done |
| `ctx-90` | 90 | 1.5s | 256 | 1.9 | 0.0802 | 99.0% | 99.5% | Done |
| `ctx-120` | 120 | 2.0s | 192 | 2.3 | 0.0830 | 99.1% | 99.4% | Done |
| `ctx-180` | 180 | 3.0s | 128 | 2.9 | **0.0551** | **99.6%** | **99.6%** | Done |

ctx-180 (3 seconds) wins decisively once fully converged -- 0.0551 vs 0.0680 baseline. It looked worse early on due to fewer gradient updates per sample, but caught up and surpassed everything. More context genuinely helps the model.

### Axis 4: Positional Encoding

| Run | Method | Batch | step/s | val/total | val/cdir_acc | val/btn_acc | Notes |
|-----|--------|-------|--------|-----------|--------------|-------------|-------|
| `baseline` | Learned | 384 | 3.0 | **0.0680** | **99.3%** | **99.6%** | Done |
| `pos-rope` | RoPE | 384 | 1.4 | 0.0709 | 99.2% | 99.5% | Done |

Learned edges out RoPE on total loss (0.0680 vs 0.0709). Close, but learned is simpler and slightly better at 60-frame context. RoPE could be worth revisiting at longer context lengths (180+) where its extrapolation properties matter more.

### Axis 5: Loss Functions

| Run | Stick loss | Btn loss | Batch | step/s | val/total | val/cdir_acc | val/btn_acc | Notes |
|-----|-----------|----------|-------|--------|-----------|--------------|-------------|-------|
| `baseline` | MSE | BCE | 384 | 3.0 | 0.0680 | 99.3% | 99.6% | Done |
| `loss-huber` | Huber | BCE | 384 | 1.6 | **0.0610** | **99.3%** | 99.5% | Done |
| `loss-discrete` | Discrete 32x32 | BCE | 384 | 1.4 | 1.4346 | 98.5% | 98.7% | Done; total not comparable (CE vs MSE) |
| `loss-focal-btn` | MSE | Focal BCE | 384 | 1.5 | 0.0631 | 99.0% | 99.5% | Done |

Huber loss beats MSE on val/total (0.0610 vs 0.0680) with matching cdir_acc -- expected since Huber is more robust to outlier stick values. Focal BCE for buttons is competitive (0.0631) and may improve rare-button recall. Discrete loss is not directly comparable on total but cdir/btn accuracy is lower.

### Observations

- **Context is king**: ctx-180 (3s) delivered the best val/total of any run (0.0551), beating even the loss-function improvements. The model benefits from seeing longer sequences of gameplay -- approach patterns, combo follow-ups, and recovery trajectories all span multiple seconds.
- **Depth/width**: The 4L/768d baseline is hard to beat at 2M samples. Larger models need more data or longer training.
- **Positional encoding**: Learned beats RoPE slightly at 60 frames. Worth retesting RoPE at 180-frame context where extrapolation could matter.
- **Loss**: Huber is the best stick loss (0.0610 vs 0.0680 MSE) -- cleaner gradients on outlier stick positions. Focal BCE is competitive for buttons and may improve rare-button recall.
- **Discrete stick loss**: Accuracy metrics are worse across the board (98.5% cdir, 98.7% btn); the 32x32 binning may be too coarse.

---

## Phase 3 Summary

Best configuration from Phase 3 sweeps:

| Setting | Winner | Runner-up |
|---------|--------|-----------|
| Depth | 4 layers | 6 layers (close) |
| Width | 768 (d_model) | 512 (close, half the params) |
| Context | 180 frames (3s) | 60 frames |
| Pos encoding | Learned | RoPE (close) |
| Stick loss | Huber | MSE |
| Button loss | BCE | Focal BCE (close) |

## Phase 4: Champion (planned)

Combine best settings from Phase 3 and train longer on full data:
- 768d / 4L / Huber stick loss / BCE buttons / Learned pos-enc / 180-frame context
- Train on full 94k replay dataset for multiple epochs
- Evaluate on held-out validation set + qualitative inference testing