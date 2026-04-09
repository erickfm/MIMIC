# Research Notes — 2026-04-09: GPU Throughput Optimization & Position Encoding Experiments

## Goal

Maximize training throughput on RTX 4090 while maintaining or improving model quality.
Previous best: relpos at 2.8 step/s, val loss 0.977 (on 7.6K game dataset).

## Throughput Optimizations Applied

### Training Pipeline

| Change | File | Impact |
|--------|------|--------|
| Explicit TF32 flags | train.py:82-83 | Clarity (functionally redundant with `set_float32_matmul_precision("high")`) |
| `torch.compile(mode="reduce-overhead")` | train.py:551 | CUDA graph capture, kernel fusion |
| Graph break fix: `accum_loss += micro_loss.detach()` | train.py:975 | Eliminated `.item()` GPU-CPU sync inside compile |
| BF16 AMP enabled by default | train.py | ~1.5-2x on matmuls |

### DataLoader Optimization

| Change | File | Impact |
|--------|------|--------|
| `prefetch_factor=4` | train.py:496 | 32 batches buffered (8 workers x 4) vs 16 |
| `mmap=True` on `torch.load` | dataset.py:122,162 | 2ms shard "load" vs ~1s for 800MB shards |
| 800MB shards (reshard from 3.8GB) | tools/reshard.py | Reduced per-shard load stall; better page cache utilization |

**GPU utilization:** 93-96% SM during compute (up from ~70% effective with old 3.8GB shards).
Workers at 8 with mmap use ~1.2GB RSS each (vs ~4GB without mmap). No swap pressure.

**Note:** Increasing workers to 12 with mmap caused swap thrashing on 64GB system. 8 workers is the sweet spot.

### Net Throughput Gains (RoPE model, 6 layers)

| Config | step/s |
|--------|--------|
| FP32, no compile, 3.8GB shards, no mmap | 2.8 |
| BF16 + compile + 3.8GB shards | 9.1 |
| BF16 + compile + 800MB shards + mmap + prefetch=4 | **11.4** |

## Position Encoding Experiments

All experiments: 6-layer, d_model=512, 8 heads, 2048 FFN, batch 512, 16M samples,
data: fox_hal_800m (7,600 games), cosine LR 3e-4 -> 1e-6.

### Results

| Run | pos_enc | Variant | Dropout | Val Loss | step/s | Notes |
|-----|---------|---------|---------|----------|--------|-------|
| hal-rope-v1 | rope | Standard (theta=10000) | 0.2 | 1.12 | 8.8 | Plateaued at step ~8K |
| hal-rope-lt-v1 | rope | Lower theta (1000) | 0.2 | 1.13 | 9.2 | No improvement over standard |
| hal-rope-lf-v1 | rope | Learnable freqs | 0.2 | >1.12 | 11.4* | Worse than standard |
| hal-flex-v1 | flexbias | Learned bias table (FlexAttention) | 0.2 | 1.14 | 11.3 | Scalar bias not expressive enough |
| hal-ropeflex-v1 | rope+flexbias | RoPE + learned bias | 0.2 | 1.15 | 10.7 | Combined, still not enough |
| hal-rope-deep-v1 | rope | Standard, 12 layers | 0.1 | killed early | 6.9 | User wanted speed over depth |
| hal-rope-drop01-v1 | rope | Standard (theta=10000) | 0.1 | **1.083** | 10.1 | **Best RoPE result** |
| (reference) hal | relpos | Shaw et al. 2018 | 0.2 | 0.977 | 4.4 | Gold standard |

*11.4 step/s was with 12 workers (memory pressure); actual sustainable speed ~9-10 step/s.

### Key Finding: Dropout Matters More Than RoPE Variant

All RoPE variants with dropout=0.2 plateau at ~1.12 regardless of frequency tuning,
learned parameters, or additive bias tables. Dropping to 0.1 pushed val loss to 1.083 —
a bigger improvement than any architectural change to RoPE.

### Why RoPE Underperforms RelPos

Shaw relpos learns a content-dependent bias per distance: `Q_i . E[i-j]` where E is a
64-dim learned embedding per distance per head (131K position params total). This is a
free-form lookup table that can represent arbitrary nonlinear distance functions.

RoPE encodes distance via rotations producing smooth cosine similarity. The dot product
between rotated Q and K decomposes to a sum of 32 cosines at fixed frequencies. It
fundamentally cannot represent sharp, discontinuous distance patterns.

For Melee, combo timing windows create sharp frame-distance dependencies (e.g., "frame 8
is a true combo, frame 9 is not"). Shaw's lookup table captures this; RoPE's cosines can't.

### FlexAttention Analysis

FlexAttention (PyTorch 2.11) was tested as a way to add learned distance bias inside a
fused attention kernel. Performance: ~85-90% of Flash Attention speed (11.3 step/s vs
~11.4 for pure RoPE). However, a scalar-per-distance bias table (2K params) lacks the
content interaction that makes Shaw relpos work (131K params with Q-dependent dot product).

## RoPE Variants Researched But Not Yet Tested

### Tier 1: Most Promising

**Selective RoPE** (arXiv 2511.17388, Nov 2025)
- Rotation angles become content-dependent: `omega = conv1d(W @ q)`, cumulative sum
- Combo-relevant frames cluster in angular space; idle frames spread out
- Flash compatible (modifies Q/K before dot product)
- ~25 lines to implement

**CARoPE** (arXiv 2507.23083, Jul 2025)
- Per-head frequency modulated by token content: `f(x) = 1/(softplus(x@W)+1)`
- Phase accumulates via content-dependent modulation
- Flash compatible
- ~20 lines

**RoPE + NoPE Hybrid** (arXiv 2501.18795, Jan 2025)
- Skip RoPE on alternating layers; NoPE layers attend purely by content
- Flash compatible; trivial to implement (~5 lines)

### Tier 2: Quick Experiments

**xPos** (ACL 2023)
- Exponential decay on Q/K magnitudes based on position
- Adds monotonic decay on top of cosine; breaks pure rotation symmetry
- Flash compatible; ~15 lines
- Designed for length extrapolation; may not help at fixed 256 frames

**Randomized Positions** (ACL 2023)
- Non-uniform random position indices during training
- Acts as regularization; forces model to learn relative patterns
- Flash compatible; ~5 lines

### Tier 3: More Complex

**GRAPE** (ICLR 2026) — learned rotation planes via rank-2 skew-symmetric generators
**LieRE** (ICML 2025) — full rotation matrices via Lie algebra (O(d^3) cost)
**ComRoPE** (CVPR 2025) — larger block rotations with commutativity constraint
**WavPE** (Feb 2025) — wavelet-based multi-scale bias (works via FlexAttention)

### Not Useful for Our Case

**YaRN / LongRoPE** — context window extension; irrelevant at 256 frames
**DoPE** — inference-time fix; we train from scratch
**TAPA** — not Flash compatible

## Full Fox Dataset Built

Built shards from all Fox .slp files on HuggingFace (`erickfm/slippi-public-dataset-v3.7/FOX`).

| Dataset | Games | Train Shards | Val Shards | Size |
|---------|-------|-------------|------------|------|
| fox_hal_local (7.6K, old) | 7,600 | 28 x 3.8GB | 2 x 3.8GB | 110GB |
| fox_hal_800m (7.6K, resharded) | 7,600 | 135 x 830MB | 7 x 830MB | 110GB |
| fox_hal_full (all Fox) | ~10,000 | 189 x 810MB | 10 x 810MB | 150GB |

Quality filters applied (matching HAL): min 1500 frames, both players take damage,
one player loses all stocks. Character filter: Fox only (character=1).

## New Model Presets Added

| Preset | Description |
|--------|-------------|
| `hal-learned` | HAL arch with learned absolute position embeddings |
| `hal-rope` | HAL arch with RoPE (theta=10000) |
| `hal-rope-lt` | HAL arch with RoPE (theta=1000) |
| `hal-rope-lf` | HAL arch with RoPE (learnable frequencies) |
| `hal-flex` | HAL arch with FlexAttention learned distance bias |
| `hal-ropeflex` | HAL arch with RoPE + FlexAttention bias combined |
| `hal-rope-deep` | 12-layer RoPE (41M params) |

## Files Modified

- `train.py` — TF32 flags, compile mode, graph break fix, NUM_WORKERS, prefetch_factor
- `mimic/model.py` — New presets, RoPE refactor (configurable theta, learnable freqs), FlexAttention
- `mimic/dataset.py` — mmap=True on torch.load
- `tools/reshard.py` — New script: reshard .pt files into smaller chunks
- `CLAUDE.md` — Updated training command, throughput notes

## Next Steps

1. Train on full Fox dataset (fox_hal_full, ~10K games) with larger/wider model
2. Implement Selective RoPE or CARoPE (content-dependent rotation angles)
3. Try RoPE+NoPE hybrid (5-line experiment)
4. Budget: must finish by 2026-04-10
