# FRAME Architecture Search Plan

Systematic search for the best architecture for next-frame controller prediction in Super Smash Bros. Melee.

**Compute budget:** 20x NVIDIA RTX 4090 (24 GB VRAM each)
**Dataset:** Full streaming dataset (HuggingFace shards)
**Tracking:** wandb project `FRAME`, grouped by experiment phase

---

## Overview

The model has three major architectural axes to explore:

1. **Frame encoder** -- how raw game state becomes a vector per timestep
2. **Temporal backbone** -- how the sequence of frame vectors is processed
3. **Output design** -- heads, loss functions, target representation

Each phase below isolates one axis. All runs within a phase share the same
config for the other axes so results are directly comparable.

### Experiment schedule

| Phase | Question | GPUs | Runs | Wall time (est.) |
|-------|----------|------|------|------------------|
| 0 | Baseline calibration | 4 | 4 | 4 h |
| 1 | Frame encoder variants | 16 | 16 | 8 h |
| 2 | Temporal backbone scaling | 20 | 20 | 10 h |
| 3 | Output / loss ablations | 8 | 8 | 6 h |
| 4 | Best-of combination | 4 | 4 | 12 h |

Total: ~40 hours of wall time, run sequentially across phases.
Each phase can be launched as a single `sweep.sh` invocation.

---

## Phase 0: Baseline calibration (4 GPUs, ~4 h)

Establish reproducible baselines with the current architecture at two scales.
These numbers anchor every comparison that follows.

| GPU | Run | Config |
|-----|-----|--------|
| 0 | `baseline-small` | `--model small --lr 5e-4 --epochs 3` |
| 1 | `baseline-base` | `--model base --lr 3e-4 --epochs 3` |
| 2 | `baseline-small-long` | `--model small --lr 5e-4 --epochs 10` |
| 3 | `baseline-base-long` | `--model base --lr 3e-4 --epochs 10` |

**Wandb group:** `phase0-baseline`
**Metrics to record:** final val loss (total + per-head), cdir_acc, btn_acc, step/s throughput

---

## Phase 1: Frame encoder variants (16 GPUs, ~8 h)

This is the highest-impact axis. The current encoder creates ~41 individual
tokens per frame and runs 2-layer intra-frame self-attention over them.
Alternatives to test:

### 1A. Flat concat (no intra-frame attention)

Replace `FrameEncoder` with a simple approach: embed categoricals, project
numeric groups, concatenate everything, project to `d_model` via a 2-layer MLP.
No `_GroupAttention` at all.

**Why:** Establishes whether intra-frame attention is actually helping or just
adding parameters and FLOPs. If flat concat matches current quality, the
intra-frame transformer is wasted compute.

**Implementation:** New `FlatFrameEncoder` class. ~60 lines. Reuses the same
embedding tables and `_mlp` builders, but replaces `_GroupAttention` + `[CLS]`
pooling with `cat → Linear → GELU → Linear`.

| GPU | Run | Config |
|-----|-----|--------|
| 0 | `flat-small` | small backbone, flat encoder |
| 1 | `flat-base` | base backbone, flat encoder |

### 1B. Composite tokens (8 semantic groups)

Revive the `tokenize-concepts` approach: group features into 8 semantic tokens
(`GAME_STATE`, `SELF_INPUT`, `OPP_INPUT`, `SELF_STATE`, `OPP_STATE`,
`NANA_SELF`, `NANA_OPP`, `PROJECTILES`). Each token mixes its categoricals +
numerics + bools internally before intra-frame attention sees them.

Key difference from the old branch: use the current shared embedding tables
and `norm_first=True` attention, not the old tiny scaled embeddings.

**Why:** 8 tokens means intra-frame attention is 8x8 instead of 41x41 --
roughly 25x cheaper. If quality holds, this is a strict win for throughput
and lets the temporal backbone be deeper for the same total FLOP budget.

**Implementation:** New `CompositeFrameEncoder`. Reuse `_GroupAttention`.
Each composite token: `cat(emb_1, ..., emb_k, mlp(numerics), mlp(bools)) → LayerNorm → Linear → GELU → d_intra`.

| GPU | Run | Config |
|-----|-----|--------|
| 2 | `composite8-small` | small backbone, 8-token encoder |
| 3 | `composite8-base` | base backbone, 8-token encoder |

### 1C. Hybrid grouping (16 tokens)

A middle ground: group by entity (self, opp, self_nana, opp_nana) but keep
inputs separate from state, and split projectiles into individual slots.
~16 tokens total.

| GPU | Run | Config |
|-----|-----|--------|
| 4 | `hybrid16-small` | small backbone, 16-token encoder |
| 5 | `hybrid16-base` | base backbone, 16-token encoder |

### 1D. Intra-frame attention depth

Test whether the current 2-layer intra-frame transformer is right-sized.

| GPU | Run | Config |
|-----|-----|--------|
| 6 | `intra1-base` | current 41-token encoder, 1 intra-frame layer |
| 7 | `intra0-base` | current 41-token encoder, 0 intra-frame layers (mean pool) |

### 1E. Multi-query pooling (k > 1)

Instead of a single `[CLS]` query, use k=4 learned queries. Output k vectors
per frame, project each to `d_model`, sum them. Tests whether a single pooling
query is a bottleneck.

| GPU | Run | Config |
|-----|-----|--------|
| 8 | `kquery4-base` | k=4 queries, base backbone |

### 1F. Embedding dimension scaling

Test whether full-width (256-d) embeddings for every categorical are needed,
or whether scaled embeddings (like `tokenize-concepts`) are sufficient.

| GPU | Run | Config |
|-----|-----|--------|
| 9 | `scaled-emb-base` | `emb_dim = max(16, int(card**0.25 * 16))` per categorical |

### 1G. D_INTRA width

The intra-frame token width is hardcoded at 256. Test 128 and 512.

| GPU | Run | Config |
|-----|-----|--------|
| 10 | `dintra128-base` | D_INTRA=128, base backbone |
| 11 | `dintra512-base` | D_INTRA=512, base backbone |

### 1H. Dropout in encoder

Re-enable dropout in the frame encoder at two levels.

| GPU | Run | Config |
|-----|-----|--------|
| 12 | `drop10-base` | DROPOUT_P=0.10 in encoder, base backbone |
| 13 | `drop20-base` | DROPOUT_P=0.20 in encoder, base backbone |

### 1I. Encoder + backbone joint scaling

If the encoder is cheap enough (flat or composite8), test a deeper backbone
at the same total param count.

| GPU | Run | Config |
|-----|-----|--------|
| 14 | `composite8-deep` | 8-token encoder + `deep` preset (d=512, L=8) |
| 15 | `flat-deep` | flat encoder + `deep` preset |

**Wandb group:** `phase1-encoder`

**Decision gate:** Pick the encoder variant that gives the best val loss. If
two are within noise, prefer the faster one (higher step/s).

---

## Phase 2: Temporal backbone scaling (20 GPUs, ~10 h)

Lock in the winning encoder from Phase 1. Sweep the temporal transformer
across model size, depth, width, and context length.

### 2A. Model size scaling (8 GPUs)

| GPU | Preset | LR | Note |
|-----|--------|----|------|
| 0 | tiny (256d/4L) | 1e-3 | ~5M params |
| 1 | small (512d/4L) | 5e-4 | ~15M params |
| 2 | medium (768d/4L) | 3e-4 | ~31M params |
| 3 | base (1024d/4L) | 3e-4 | ~54M params |
| 4 | deep (512d/8L) | 5e-4 | ~24M params |
| 5 | wide-shallow (1536d/2L) | 3e-4 | ~38M params |
| 6 | `xlarge` (1024d/8L) | 1e-4 | ~100M params (new preset) |
| 7 | `xxlarge` (1536d/8L) | 1e-4 | ~180M params (new preset) |

### 2B. Context length (6 GPUs)

Does the model benefit from seeing more history? Test at the best model size
from 2A.

| GPU | Seq len | Note |
|-----|---------|------|
| 8 | 30 | 0.5 sec |
| 9 | 60 | 1.0 sec (current default) |
| 10 | 120 | 2.0 sec |
| 11 | 180 | 3.0 sec |
| 12 | 240 | 4.0 sec |
| 13 | 360 | 6.0 sec |

Longer contexts will require reduced batch size to fit in 24 GB. Scale LR
proportionally (`lr_new = lr_base * sqrt(bs_new / bs_base)`).

### 2C. Positional encoding (3 GPUs)

| GPU | Method | Note |
|-----|--------|------|
| 14 | Learned (current) | Baseline |
| 15 | RoPE | Better length generalization |
| 16 | Sinusoidal | Classical, no learned params |

### 2D. Attention variants (3 GPUs)

| GPU | Variant | Note |
|-----|---------|------|
| 17 | Sliding window (W=30) + global | Longformer-style, helps with long contexts |
| 18 | ALiBi (no pos emb) | Linear attention bias, extrapolates well |
| 19 | Grouped-query attention (GQA, 2 KV heads) | Faster inference for deployment |

**Wandb group:** `phase2-backbone`

**Decision gate:** Plot loss vs params (scaling law). Pick the knee of the
curve -- the model size beyond which more params stop helping given the
dataset size. Select best context length and positional encoding.

---

## Phase 3: Output / loss ablations (8 GPUs, ~6 h)

Lock in encoder + backbone from Phases 1-2. Test output design.

### 3A. Loss function variants (4 GPUs)

| GPU | Change | Note |
|-----|--------|------|
| 0 | Huber loss for sticks | More robust to outlier frames (e.g., respawn) |
| 1 | Quantile regression for sticks | Predict 10th/50th/90th percentile |
| 2 | Cosine loss + MSE for sticks | Penalizes direction errors more than magnitude |
| 3 | Weighted BCE for buttons | Upweight rare buttons (Z, D-pad) by inverse frequency |

### 3B. Target representation (2 GPUs)

| GPU | Change | Note |
|-----|--------|------|
| 4 | Discretize sticks (32 bins per axis) | Turns regression into classification, like MuZero |
| 5 | Predict stick delta (next - current) | May be easier to learn than absolute position |

### 3C. Multi-step rollout (2 GPUs)

| GPU | Rollout | Note |
|-----|---------|------|
| 6 | R=1 (current) | Predict next frame |
| 7 | R=4 | Predict 4 frames ahead, average loss across [t+1..t+4] |

**Wandb group:** `phase3-output`

---

## Phase 4: Best-of combination (4 GPUs, ~12 h)

Take the winners from each phase and train longer on full data.

| GPU | Run | Config |
|-----|-----|--------|
| 0 | `champion-1` | Best encoder + best backbone + best loss, 10 epochs |
| 1 | `champion-2` | Same as above, second-best encoder (if close) |
| 2 | `champion-long` | Best overall, 30 epochs |
| 3 | `champion-large` | Best encoder/loss, largest backbone that fits, 10 epochs |

**Wandb group:** `phase4-champion`

---

## Implementation checklist

Code changes needed before launching (in order of priority):

### Required for Phase 1

- [ ] **`FlatFrameEncoder`** in `frame_encoder.py`: concat all group outputs, 2-layer MLP to `d_model`. No `_GroupAttention`. Flag: `--encoder flat`
- [ ] **`CompositeFrameEncoder`** in `frame_encoder.py`: 8 semantic group tokens using `CompositeToken`-style mixing, then `_GroupAttention`. Flag: `--encoder composite8`
- [ ] **`HybridFrameEncoder`**: ~16 tokens (entity-level groups). Flag: `--encoder hybrid16`
- [ ] **Configurable `D_INTRA`**: make it a `ModelConfig` field instead of a module-level constant
- [ ] **Configurable `DROPOUT_P`**: pass from `ModelConfig.dropout` to encoder
- [ ] **Configurable `k_query`**: expose as CLI arg
- [ ] **Configurable encoder intra-frame layers**: expose `nlayers` as CLI arg
- [ ] **`--encoder` CLI arg** in `train.py` to select encoder variant
- [ ] **Scaled embedding option**: `--scaled-emb` flag that sets `emb_dim = max(16, int(card**0.25 * 16))`

### Required for Phase 2

- [ ] **`xlarge` and `xxlarge` presets** in `MODEL_PRESETS`
- [ ] **RoPE positional encoding** option in `model.py`
- [ ] **Sinusoidal positional encoding** option in `model.py`
- [ ] **ALiBi** option
- [ ] **`--pos-enc` CLI arg** (`learned`, `rope`, `sinusoidal`, `alibi`)
- [ ] **Sliding window attention** option
- [ ] **GQA option** (n_kv_heads as config field)

### Required for Phase 3

- [ ] **Huber loss** option for stick regression
- [ ] **Quantile regression head** variant
- [ ] **Discretized stick head** (32x32 classification)
- [ ] **Delta-target mode** in dataset
- [ ] **Multi-step rollout** (R=4) in dataset windowing + loss averaging

### Infrastructure

- [ ] **`sweep_arch.sh`**: new sweep launcher that supports `--encoder` flag
- [ ] **Auto batch-size finder**: halve BS until it fits VRAM, log actual BS to wandb

---

## Evaluation criteria

Every run is compared on the same metrics:

| Metric | Weight | Why |
|--------|--------|-----|
| `val/total` | Primary | Overall prediction quality |
| `val/cdir_acc` | High | C-stick is hardest head; accuracy here indicates real game understanding |
| `val/btn_acc` | High | Button prediction is safety-critical (wrong buttons = SD) |
| `val/loss_main` | Medium | Stick precision matters for movement quality |
| `perf/step_per_sec` | Tiebreaker | Faster = more training in same budget, better for live inference |

For final model selection, also evaluate:
- **Inference latency** (must be < 1 frame = 16.67 ms on target hardware)
- **Qualitative playback** against CPU opponents using `inference.py`

---

## Running a phase

Each phase uses the existing `sweep.sh` infrastructure. Example for Phase 1:

```bash
# Phase 1: encoder variants (16 runs across 16 GPUs on a 20-GPU node)
bash sweep.sh --host root@<IP> --port <PORT> \
    --group phase1-encoder --wandb-key "$WANDB_KEY" \
    --data-dir data/full \
    --run "0  base 3e-4 --encoder flat --wandb-tags encoder,flat" \
    --run "1  small 5e-4 --encoder flat --wandb-tags encoder,flat" \
    --run "2  base 3e-4 --encoder composite8 --wandb-tags encoder,composite" \
    --run "3  small 5e-4 --encoder composite8 --wandb-tags encoder,composite" \
    --run "4  base 3e-4 --encoder hybrid16 --wandb-tags encoder,hybrid" \
    --run "5  small 5e-4 --encoder hybrid16 --wandb-tags encoder,hybrid" \
    --run "6  base 3e-4 --intra-layers 1 --wandb-tags encoder,depth" \
    --run "7  base 3e-4 --intra-layers 0 --wandb-tags encoder,depth" \
    --run "8  base 3e-4 --k-query 4 --wandb-tags encoder,kquery" \
    --run "9  base 3e-4 --scaled-emb --wandb-tags encoder,scaled-emb" \
    --run "10 base 3e-4 --d-intra 128 --wandb-tags encoder,dintra" \
    --run "11 base 3e-4 --d-intra 512 --wandb-tags encoder,dintra" \
    --run "12 base 3e-4 --dropout 0.10 --wandb-tags encoder,dropout" \
    --run "13 base 3e-4 --dropout 0.20 --wandb-tags encoder,dropout" \
    --run "14 deep 5e-4 --encoder composite8 --wandb-tags encoder,joint" \
    --run "15 deep 5e-4 --encoder flat --wandb-tags encoder,joint"
```

---

## GPU hour budget

| Phase | GPUs | Hours | GPU-hours |
|-------|------|-------|-----------|
| 0 | 4 | 4 | 16 |
| 1 | 16 | 8 | 128 |
| 2 | 20 | 10 | 200 |
| 3 | 8 | 6 | 48 |
| 4 | 4 | 12 | 48 |
| **Total** | | **~40 h wall** | **440 GPU-h** |

At ~$0.40/GPU-hr (4090 spot), total cost is approximately **$176**.
