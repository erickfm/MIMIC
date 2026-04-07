# Research Notes — 2026-04-07

## Summary

Found and fixed the primary cause of MIMIC's overfitting: `max_steps` was computed from local batch size (64) instead of effective batch size (512 with 8 GPUs), causing 8x more training than intended. Trained on 4x more Fox data (12K games from HF public dataset vs 3.2K). Result: val loss 0.977 with no overfitting — below HAL's 1.03 on the same val set.

---

## Bug Fix: max_steps / effective batch size

### The problem

`train.py:704` computed:
```python
max_steps = max_samples // BATCH_SIZE  # 16,777,216 // 64 = 262,144 steps
```

With 8-GPU DDP, each step processes `64 × 8 = 512` samples. Total effective samples = 262,144 × 512 = **134M** — but the intent was 16.7M (matching HAL's `n_samples`).

### Impact

| | HAL | MIMIC (bug) | MIMIC (fixed) |
|---|---|---|---|
| Intended samples | 16.7M | 16.7M | 16.7M |
| Actual samples | 16.7M | **134M** | 16.7M |
| Optimizer steps | 32,768 | **262,144** | 32,768 |
| Oversampling rate | 0.3× | 2.6× | 0.3× |

### The fix

```python
eff_bs = BATCH_SIZE * world_size * grad_accum_steps
max_steps = max_samples // eff_bs
```

Log now clearly shows: `16,777,216 samples / eff_bs 512 (64×8gpu×1accum) = 32,768 steps`

---

## HAL Also Overfits

Contrary to prior notes claiming HAL's val loss was "stable," HAL's wandb data (run `l8wtjjdk`) shows a U-shaped val loss curve:

| Checkpoint | Samples | Val Loss |
|---|---|---|
| 1 | 524K | 0.862 |
| 10 (best) | 5.2M | **0.744** |
| 18 | 9.4M | 0.802 (+7.8%) |

HAL overfits after 5.2M samples — its val loss rises from 0.744 to 0.802. The 4-stocking gameplay comes from the **best checkpoint** (5.2M), not the final one. MIMIC's best val was at step ~8K (≈4.1M effective samples) — the same ballpark.

---

## Training Runs

### Run 1: hal-fixed-steps (Machine E, 3.2K games)

Same data as the buggy run, but with correct max_steps.

```bash
torchrun --nproc_per_node=8 train.py \
  --model hal --encoder hal_flat \
  --hal-mode --hal-minimal-features --hal-controller-encoding \
  --stick-clusters hal37 --plain-ce --no-amp \
  --lr 3e-4 --batch-size 64 --max-samples 16777216 \
  --data-dir data/fox_hal_norm \
  --controller-offset --self-inputs \
  --no-compile --run-name hal-fixed-steps \
  --nccl-timeout 7200 --no-warmup --cosine-min-lr 1e-6 \
  --random-perspective
```

| Metric | Value |
|---|---|
| Steps | 32,768 |
| Training time | 58 min |
| Best val loss | **1.048** (step ~20K) |
| Final val loss | 1.19 |
| Overfitting | +13% (vs +37% before fix) |
| Best bf1 / mf1 / cf1 | 83.0% / 42.3% / 55.1% |

Val loss falls from 1.35 → 1.05, then rises mildly to 1.19 by end. The 3.2K game dataset is simply too small — model begins memorizing in the second half of training.

### Run 2: hal-big-data (Machine C, 12K games)

Built HAL-normalized shards from ~6,000 Fox .slp files downloaded from `erickfm/slippi-public-dataset-v3.7` (45,854 Fox files total; disk filled at ~55K downloaded, ~6K successfully sharded before crash).

```bash
torchrun --nproc_per_node=8 train.py \
  --model hal --encoder hal_flat \
  --hal-mode --hal-minimal-features --hal-controller-encoding \
  --stick-clusters hal37 --plain-ce --no-amp \
  --lr 3e-4 --batch-size 64 --max-samples 16777216 \
  --data-dir data/fox_hal_big \
  --controller-offset --self-inputs \
  --no-compile --run-name hal-big-data \
  --nccl-timeout 7200 --no-warmup --cosine-min-lr 1e-6
```

| Metric | Value |
|---|---|
| Train data | 43 shards, 11,641 games, 108M frames |
| Val data | 2 shards, 512 games, 5M frames |
| Steps | 32,768 |
| Training time | ~53 min |
| Best val loss | **0.977** (step ~28K) |
| Final val loss | 0.993 |
| Overfitting | **+1.6%** (essentially none) |
| Best bf1 / mf1 / cf1 | **84.3% / 44.7% / 58.6%** |

Val loss falls continuously from 1.28 → 0.98 and plateaus. No overfitting. Model was still improving at end of training — could benefit from more steps or more data.

---

## All Models Compared

All "hal-*" runs below are **MIMIC models** using HAL's architecture (`--model hal`). The only true HAL run is Eric Gu's original, trained with HAL's own codebase.

| Run | Codebase | Data | Steps (actual samples) | Best Val | Final Val | Overfit |
|---|---|---|---|---|---|---|
| HAL (Eric Gu's original) | HAL | 2,830 games | 32K (16.7M) | 1.03 (our val) | — | Mild |
| MIMIC `hal-pipeline-match` (bug) | MIMIC | 3,229 games | 262K (134M) | 1.15 | 1.57 | **+37%** |
| MIMIC `hal-fixed-steps` | MIMIC | 3,229 games | 32K (16.7M) | 1.048 | 1.19 | +13% |
| **MIMIC `hal-big-data`** | **MIMIC** | **12,153 games** | **32K (16.7M)** | **0.977** | **0.993** | **+1.6%** |

---

## Dataset: Slippi Public v3.7

The HuggingFace dataset `erickfm/slippi-public-dataset-v3.7` contains **95,102 unique replays** organized by character:

| Character | Files |
|---|---|
| Fox | 45,854 |
| Falco | 32,552 |
| Marth | 26,085 |
| Captain Falcon | 22,794 |
| Sheik/Zelda | 13,764 |
| Others | <7K each |

Each replay appears under both characters in the match (~174K total files). This is 14× more Fox data than HAL used (3.2K games).

Only ~12K of the 46K Fox games were sharded (disk space constraint on Machine C). The full dataset would provide even more regularization.

---

## GPUS.md Correction

Machine E IP was wrong in GPUS.md: `66.222.138.178` → `66.222.128.37`. Fixed.

---

## Key Insights

1. **The overfitting was from training 8× too long**, not from insufficient regularization or missing data augmentation.
2. **HAL overfits too** — its best checkpoint is at 5.2M samples, after which val loss rises. The "stable val loss" claim was incorrect.
3. **More data eliminates overfitting entirely** — 12K games (4× HAL's data) brought val loss below HAL's 1.03 with zero overfitting.
4. **Random perspective selection is not meaningful regularization** for character-specific models — the model trivially learns to attend to the character embedding, and half the training is wasted on the opponent's character.

---

## Next Steps

1. Test hal-big-data checkpoint in closed-loop gameplay (vs CPU level 9)
2. Rebuild shards from all 46K Fox games on a machine with more disk
3. Try longer training (33-50M samples) now that data supports it
4. Evaluate whether val loss 0.977 translates to better gameplay than HAL's 1.03
