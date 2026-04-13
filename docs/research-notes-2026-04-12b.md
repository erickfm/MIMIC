# Research Notes — 2026-04-12b: MIMIC Beats HAL 4-2 (Clean 300M Run)

## TL;DR

First clean MIMIC run with all three known bugs fixed simultaneously. Trained on
305M frames (32,577 Fox games) and **beat HAL 4 stocks to 2** head-to-head on
Final Destination. Previous "MIMIC beats HAL" runs all had at least one of the
three bugs and were not apples-to-apples comparisons.

**Checkpoint:** `checkpoints/hal-7class-v2-300M_best.pt`
**WandB:** https://wandb.ai/erickfm/MIMIC/runs/wl1pp6cb
**Val loss:** 0.671 | **Btn F1:** 88.5% | **Main F1:** 57.5%

---

## The Three Bugs (All Fixed)

Every prior 7-class run had at least one of these:

### 1. Gamestate leak (fixed 2026-04-11 via v2 shard alignment)

`console.step()` returns **post-frame** game state (`action`, `position`,
`percent`) alongside **pre-frame** controller inputs. So at frame `i`,
`self_action = KNEE_BEND (24)` appears on the same frame as
`target = JUMP` — the model could read the answer from the action embedding
instead of learning to initiate.

**Leak rate:** 58.2% of JUMP onsets had `self_action=KNEE_BEND` in old shards.
Fixed by shifting targets forward by 1 frame in `slp_to_shards.py`:
```python
shifted_targets = {k: v[1:] for k, v in targets.items()}
shifted_states  = {k: v[:-1] for k, v in states.items()}
```
New leak rate: 2.0%.

### 2. Missing `--self-inputs` (fixed 2026-04-12)

The v2 training command in CLAUDE.md dropped `--self-inputs` along with
`--controller-offset`. But these are different things:
- `--controller-offset`: shifts controller tensor in dataloader (not needed with v2 shards)
- `--self-inputs`: tells the encoder to USE `self_controller` as input

Without `--self-inputs`, `no_self_inputs=True` and the encoder completely
ignores the 56-dim controller one-hot. That's 33% of the non-embedding input
features gone. HAL's architecture uses `gamestate(18) + controller(54)` — the
controller is half the dynamic input.

Comparing prior runs (all on v2 shards):

| Run | self_inputs | Val total | Btn F1 | Main F1 |
|-----|-------------|-----------|--------|---------|
| hal-7class-v2 | OFF | 2.27 | 59.4% | 15.5% |
| hal-7class-v2-long | OFF | 2.33 | 58.8% | 15.4% |
| **hal-7class-v2-300M** | **ON** | **0.67** | **88.5%** | **57.5%** |

Dropping `--self-inputs` made main stick F1 crater from 57% → 15%. Sticks are
highly autocorrelated and the model needs the last frame's stick to predict
the current one; without it, the model is flying blind.

### 3. Insufficient data (overfitting at small scale)

Early v2-si experiments trained on 10M frames (20 shards) and overfit by step
2500, bottoming at val loss 0.82. Increasing to 51M frames (102 shards) pushed
the bottom to 0.73 at step 10K. At 305M frames (607 shards), val loss reached
0.671 with no overfitting through 32K steps.

| Data | Steps | Val Loss | Btn F1 | Main F1 | Overfitting |
|------|-------|----------|--------|---------|-------------|
| 10M frames | 2.4K | 0.82 | 82% | 55% | Yes, step 2500 |
| 51M frames | 10K | 0.73 | 87.5% | 55% | None |
| 305M frames | 32K | 0.671 | 88.5% | 57.5% | None |

---

## Head-to-Head vs HAL

Full game, ~200 seconds, Final Destination, Fox vs Fox, temperature=1.0.

Stock trajectory:
```
4-4 (start)
4-3 (MIMIC first kill, HAL at 143%)
3-3 (trading)
3-2 (MIMIC edge)
3-1 (MIMIC builds lead)
2-1 (HAL answers)
2-0 (MIMIC closes it out, HAL 39%)
```

**Result:** MIMIC (P1) WINS, 2 stocks to 0.

This is the first MIMIC model where the comparison is fair. Earlier "MIMIC
beats HAL" claims (2026-04-08, 2026-04-11) used models trained with the
gamestate leak or without self_inputs.

---

## Training Command

```bash
python3 train.py \
  --model hal --encoder hal_flat \
  --hal-mode --hal-minimal-features --hal-controller-encoding \
  --stick-clusters hal37 --plain-ce \
  --lr 3e-4 --batch-size 64 --grad-accum-steps 8 \
  --max-samples 16777216 \
  --data-dir data/fox_v2_32k \
  --self-inputs \
  --reaction-delay 0 \
  --run-name hal-7class-v2-300M \
  --no-warmup --cosine-min-lr 1e-6
```

Key flags:
- `--self-inputs` — **critical**, not dropped as the v2 docs suggested
- `--reaction-delay 0` — correct for v2 shards (shift is baked in)
- No `--controller-offset` — v2 alignment already correct
- Single GPU, batch 64 × grad_accum 8 = effective batch 512

Training time: ~3 hours on RTX 4090 for 32,768 steps at ~3 step/s.

---

## Data Pipeline

### 1. Download .slp from HuggingFace

`erickfm/slippi-public-dataset-v3.7` contains ~46K Fox replays across 6 batches.
Use `snapshot_download` (not per-file) for faster bulk download with auth token.

```python
from huggingface_hub import snapshot_download
snapshot_download(
    "erickfm/slippi-public-dataset-v3.7",
    repo_type="dataset",
    allow_patterns="FOX/**/*.slp",
    local_dir="data/fox_slp",
    max_workers=8,
)
```

Authenticated download is much faster than anonymous (rate limits are much
looser with a valid HF token). Snapshot of 32K files took ~1 hour with auth.

### 2. Shard with slp_to_shards.py

```bash
python3 tools/slp_to_shards.py \
  --slp-dir data/fox_slp_snapshot \
  --meta-dir data/fox_v2_fresh \
  --repo erickfm/mimic-melee --no-upload \
  --staging-dir data/fox_v2_32k --keep-staging \
  --shard-gb 0.8 \
  --hal-norm data/fox_v2_fresh/hal_norm.json \
  --character 1 \
  --workers 8 --val-frac 0.05
```

Metadata directory needs: `norm_stats.json`, `cat_maps.json`,
`stick_clusters.json`, `controller_combos.json` (7-class version with
`{"n_combos": 7}`), `hal_norm.json`.

Note: `slp_to_shards.py` copies `norm_stats`, `cat_maps`, `stick_clusters` to
the output dir but NOT `controller_combos.json` or `hal_norm.json` — copy
those manually after sharding.

Sharding 30,792 train files took ~72 min (8 workers). Output: 607 train
shards + 33 val shards, 305M train frames total.

---

## Verifying Shards Are v2 (Quick Sanity Check)

```python
import torch
shard = torch.load("data/fox_v2_32k/train_shard_000.pt", map_location="cpu", weights_only=False)
action = shard["states"]["self_action"]
btns = shard["targets"]["btns_single"]
total, leaked = 0, 0
for i in range(1, len(btns)):
    if btns[i].item() == 3 and btns[i-1].item() != 3:  # 7-class JUMP
        total += 1
        if action[i].item() == 24:  # KNEE_BEND
            leaked += 1
print(f"JUMP leak: {leaked}/{total} ({leaked/total*100:.1f}%)")
# <5% = v2 (correct), >50% = old (leaked)
```

For v2 shards, train with `--reaction-delay 0` and NO `--controller-offset`.
For old shards (fox_hal_full etc.), train with `--reaction-delay 1
--controller-offset --self-inputs`.

---

## head_to_head.py for Mixed-Class Models

HAL's original checkpoint uses 5-class buttons (54-dim controller) while
MIMIC's v2 models use 7-class (56-dim). `tools/inference_utils.py` was
extended to:

1. Handle HAL's bare state dict (no `"config"` key) via key remapping:
   - `stage_emb` → `encoder.stage_emb`
   - `transformer.h.{i}.attn` → `blocks.{i}.self_attn`
   - `button_head` → `heads.btn_head`
   - etc.

2. `head_to_head.py` now builds a per-player context with matching
   `n_combos` so each model sees the correct controller encoding in its
   input. P1 can be 7-class, P2 can be 5-class (or vice versa).

Run:
```bash
python3 tools/head_to_head.py \
  --p1-checkpoint checkpoints/hal-7class-v2-300M_best.pt \
  --p2-checkpoint /home/erick/projects/hal/checkpoints/000005242880.pt \
  --dolphin-path .../dolphin-emu \
  --iso-path "..." \
  --data-dir data/fox_v2_32k \
  --stage FINAL_DESTINATION
```

---

## What's Next

1. **Train HAL on the same data** — true apples-to-apples. HAL uses its own
   MDS pipeline (`hal/data/process_replays.py`) pointed at the same .slp
   source, then `hal/training/simple_trainer.py`. Would compare MIMIC's
   codebase vs HAL's codebase on identical training data.

2. **Scale further** — 305M frames is already 3x HAL's original training
   budget. Could train longer or try larger models.

3. **Validate gameplay more broadly** — single head-to-head is a small
   sample. Should run multiple matches across both port orderings to
   control for port-swap effects.
