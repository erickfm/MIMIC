# Research Notes — 2026-04-06

## Summary

Matched MIMIC's data pipeline to HAL's: normalization, c-stick targets, button encoding, training config. For the first time, we can compare val loss directly — HAL scores 1.03 on our val data, our best is 1.15 (12% gap). The gap is the closest yet but widens during training (overfitting). Three character models trained (Fox, Falco, Captain Falcon). Root cause of remaining overfitting gap is unresolved.

---

## Pipeline Fixes Applied

### 1. HAL normalization in shards
- Rebuilt shards with `--hal-norm data/fox_public_shards/hal_norm.json`
- Per-feature transforms: `normalize` (percent, stock, facing, invulnerable, jumps_left, on_ground), `invert_normalize` (shield_strength), `standardize` (position_x, position_y)
- Verified: percent=0→-1.0, stock=4→1.0, facing=1→1.0. Matches HAL's preprocessed gamestate exactly.

### 2. Boolean flag normalization
- Added `* 2.0 - 1.0` to flags (on_ground, facing, invulnerable) in `HALFlatEncoder.forward()`
- Converts 0/1 → -1/+1, matching HAL's normalize transform on boolean features

### 3. 9-cluster c-stick targets from raw x/y
- `slp_to_shards.py` now computes c-stick targets from raw `c_x`, `c_y` using `HAL_CSTICK_CLUSTERS_9`
- Diagonal clusters (indices 5-8) now appear as training targets
- Shard c_dir shape: `(N, 9)` instead of `(N, 5)`
- `train.py` auto-detects target width — only remaps 5→9 for old shards

### 4. Early-release button targets
- Implemented HAL's `convert_multi_hot_to_one_hot_early_release` logic in shard builder
- When multiple buttons pressed: keeps newest press, releases older ones
- Stored as `btns_single` (int64 single-label) alongside raw `btns` (multi-hot)
- `train.py` uses `btns_single` when present, falls back to priority-based conversion

### 5. Training config alignment
- `--no-warmup`: CosineAnnealingLR from step 1 (HAL has no warmup)
- `--cosine-min-lr 1e-6`: matches HAL's `eta_min`
- Removed `find_unused_parameters=True` from DDP (HAL doesn't use it)

---

## Val Loss Comparison (directly comparable for first time)

HAL's checkpoint evaluated on our HAL-normalized val data: **1.03** (baseline).

| Run | Best val | At step 117K | Trend |
|-----|---------|-------------|-------|
| Pipeline-match (this run) | **1.15** (step ~8K) | 1.41 | Rising (overfitting) |
| HAL on our val data | **1.03** | — | Stable (0.74-0.86 on HAL's own val) |

Gap at best val: **12%** (1.15 vs 1.03).

Previous runs were not directly comparable because they used different normalization. For reference:
- Both-perspective (old norm): best val 1.12, but on differently-normalized data
- Fox-only (old norm): best val 1.16, different normalization
- HAL's own wandb: final val 0.81, but on HAL's 30-game val set

---

## Three-Character Training (machines C, E, F)

Trained Fox, Falco, Captain Falcon models overnight on 8×RTX 5090 each. These used the OLD preprocessing (standardized normalization, 5-class c-stick, priority buttons). Results are baseline but NOT directly comparable to HAL.

| Character | Machine | Train btn_f1 | Val btn_f1 | Data source |
|-----------|---------|-------------|-----------|-------------|
| Fox | E | 90% | 75% | fox_public_shards (both persp, old norm) |
| Cpt Falcon | F | 92% | 79% | HF ranked (single persp) |
| Falco | C | 92% | 77% | HF ranked (single persp) |

Note: Falcon and Falco used HF ranked shards which may have different tensorization than our current pipeline. Fox used local .slp files resharded with current code.

---

## HAL Architecture & Data Pipeline Verification

### Confirmed matches:
- Architecture: 26,274,803 params, every shape matches HAL checkpoint
- Normalization: verified on same raw values — identical output
- Feature order: percent, stock, facing, invulnerable, jumps_left, on_ground, shield, pos_x, pos_y
- Input projection: Linear(164, 512)
- Attention: relative position with skew matrix, block_size=1024
- Heads: shoulder(3) → c_stick(9) → main_stick(37) → buttons(5), autoregressive with detach
- Weight init: normal(0, 0.02), residual scaling 0.02/sqrt(2*n_layer) for c_proj
- Optimizer: AdamW, same param grouping (2D+ decay, <2D no decay)

### Confirmed different (minor):
- Perspective: both perspectives as separate shard entries vs HAL's random p1/p2 per sample
- Val set: 644 games vs HAL's 30 games
- Data ordering: IterableDataset (per-shard shuffle) vs Mosaic StreamingDataset (global shuffle)
- `sample_from_episode`: HAL picks one random window per `__getitem__`; we sample ~100 per shard visit

### HAL findings:
- HAL uses `random.choice(["p1", "p2"])` per sample — BOTH in training AND validation
- HAL's `ego_character: null` — no character filtering
- HAL's MDS has 2,830 train games, 30 val games
- HAL's `n_val_samples = 4096` (64 batches at bs=64)
- HAL's button targets use `encode_buttons_one_hot_no_shoulder_early_release`
- HAL's closed-loop eval plays vs CPU level 9 every 4M samples ("fox_rainbow" matchups)

---

## Remaining Gap Analysis

The 12% val loss gap (1.15 vs 1.03) and the overfitting behavior (val rises from 1.15 to 1.6+ while HAL stays at 0.81) remain unexplained. Investigated causes:

### Ruled out:
- Architecture mismatch — verified shape-for-shape
- Normalization mismatch — verified same values for same raw input
- Feature selection/ordering — verified 9 features in HAL's exact order
- LR schedule — both use CosineAnnealingLR, no warmup, eta_min=1e-6
- Loss function — both use plain CE on all heads, same target encoding
- Weight init — both use same scheme
- Optimizer param groups — both decay same parameters

### Not yet ruled out:
1. **Perspective randomization** — HAL picks random p1/p2 per sample (mild regularization). We use fixed separate entries. Could reduce overfitting.
2. **Data ordering** — Mosaic StreamingDataset provides more global shuffling than our per-shard IterableDataset. Could affect optimization trajectory.
3. **Val set size** — HAL's 30-game val may appear more stable simply from being small. Our 644-game val may reveal overfitting that HAL's doesn't show.
4. **Subtle data differences** — even with HAL normalization, the shard builder may extract features slightly differently from HAL's `process_replays.py` (e.g., frame alignment, controller state timing).
5. **Unknown unknowns** — something in HAL's pipeline we haven't discovered yet.

---

## Gameplay Results Summary

| Model | Step | Stocks taken | Stocks lost |
|-------|------|-------------|------------|
| HAL original | final | 4 | 1 |
| Both-persp (old norm) | 117K | 2 | 4 |
| Fox-only (old norm) | 262K | 1 | 4 |
| Fox-only (old norm) | 13K (best val) | 0 | 4 |
| Pipeline-match | 8K (best val) | 0 | 4 |
| Pipeline-match | 117K | *testing* | — |

---

## Validation Tool

Created `tools/validate_checkpoint.py` — evaluates any checkpoint (HAL or MIMIC format, auto-detected) on val data and reports per-head CE losses. Supports side-by-side comparison.

```bash
python tools/validate_checkpoint.py \
  --checkpoint checkpoints/hal_original.pt \
  --checkpoint-b checkpoints/my_model.pt \
  --data-dir data/fox_hal_norm
```

---

## Files Modified

| File | Change |
|------|--------|
| `mimic/frame_encoder.py` | Flag normalization: 0/1 → -1/+1 |
| `mimic/dataset.py` | Random window sampling, character filter, non-distributed val |
| `tools/slp_to_shards.py` | `--hal-norm`, 9-cluster c-stick from raw x/y, early_release buttons, `--character` filter |
| `train.py` | Auto-detect c-dir width, early_release button support, remove find_unused_parameters, no-warmup/cosine-min-lr |
| `tools/validate_checkpoint.py` | New: HAL vs MIMIC val loss comparison tool |

---

## Next Steps

1. Investigate overfitting gap — implement random perspective selection per sample to match HAL's data augmentation
2. Consider storing one entry per game in shards with p1/p2 columns, selecting perspective at training time
3. Run validate_checkpoint.py on pipeline-match checkpoints at various steps to track val loss vs HAL baseline
4. Test pipeline-match step 117K gameplay (in progress)
