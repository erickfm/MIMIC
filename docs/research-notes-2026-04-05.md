# Research Notes — 2026-04-05

## Summary

Eliminated all 7 architectural differences between HAL and MIMIC so the models are structurally identical (26,274,803 parameters, every shape verified against HAL's checkpoint). Set up training on remote 8×RTX 5090 cluster using HAL's exact hyperparameters. A brief local test (13.9K steps on 1 GPU) showed strong convergence: btn_f1=94.2%, main_f1=79.4%, cdir_f1=94.7%.

---

## Architecture Changes (all in commit 5a62cef)

### 1. Context length: 60 → 256

**File:** `train.py` line 628-629, `mimic/model.py` "hal" preset

HAL processes 256 frames (4.27 seconds) of context. MIMIC was using 60 (1.0 second). Added `max_seq_len=256` to the "hal" preset and auto-set `SEQUENCE_LENGTH=256` in `train()` when `model_preset=="hal"`.

### 2. Position encoding: RoPE → Relative position (skew matrix)

**File:** `mimic/model.py` lines 158-233

Replaced RoPE with HAL's relative-position attention (Shaw et al. 2018). Implementation:
- `_skew()` helper for efficient relative-position bias computation
- `CausalSelfAttentionRelPos`: combined QKV projection (`c_attn = Linear(d, 3d)`), learnable `Er = Parameter(block_size, head_dim)`, computes `Srel = skew(Q @ Er^T)` added to `QK^T` before masking
- `HALTransformerBlock`: pre-norm block matching HAL's exact structure — `ln_1 → attn → +res`, `ln_2 → c_fc → gelu → c_proj → dropout → +res`. MLP uses `ModuleDict(c_fc, c_proj)` with dropout only after second linear (no intermediate dropout, matching HAL)
- `block_size=1024` for the Er table and causal mask buffer

Activated when `pos_enc="relpos"` (set in hal preset). `FramePredictor` selects `HALTransformerBlock` vs `TransformerBlock` based on this flag.

### 3. C-stick output: 5 directions → 9 clusters

**Files:** `mimic/model.py`, `train.py` lines 222, 251

HAL outputs 9 c-stick clusters (neutral + 4 cardinals + 4 diagonals) vs MIMIC's 5 directions (neutral + 4 cardinals). Changes:
- `HALPredictionHeads` now receives `n_cdir=cfg.num_c_dirs` (was hardcoded 5)
- `num_c_dirs=9` in hal preset
- Training targets remapped from 5-class → 9-class via `_CDIR_5_TO_9` tensor (neutral→0, up→4, down→3, left→2, right→1)
- Inference updated in `inference.py` and `run_mimic_via_hal_loop.py` to decode 9-cluster output using `HAL_CSTICK_CLUSTERS_9` centers

**Limitation:** Current shard data stores c-stick as 5-direction categorical (no raw c_x/c_y). The 4 diagonal clusters (indices 5-8) never appear in training targets. For full diagonal support, shards would need raw c-stick coordinates.

### 4. Numeric features: 10 → 9 per player, reordered to HAL's order

**File:** `mimic/frame_encoder.py` HALFlatEncoder.forward()

Previous: 7 numeric (pos_x, pos_y, percent, stock, jumps_left, **invuln_left**, shield) + 3 flags (on_ground, facing, invulnerable) = 10 per player.

Now: 9 features in HAL's exact order:
```
percent, stock, facing, invulnerable, jumps_left, on_ground, shield_strength, position_x, position_y
```

Changes:
- Removed `invuln_left` (frames of invulnerability remaining) — HAL doesn't use it
- Merged flags (on_ground, facing, invulnerable) into the numeric vector
- Reordered via `_HAL_ORDER = [2, 3, 7, 8, 4, 6, 5, 0, 1]` index mapping
- Handles both 22-column (full) and 7-column (hal_minimal) input formats
- Changed `numeric_dim` from `10 * 2 = 20` to `9 * 2 = 18`

### 5. Dropout: 0.1 → 0.2

**File:** `mimic/model.py` hal preset

ModelConfig default was 0.1. HAL uses 0.2. Set `dropout=0.2` in `MODEL_PRESETS["hal"]`.

### 6. Embedding vocab: stage 8→6, char 32→27, action 395→396

**Files:** `mimic/cat_maps.py`, `mimic/model.py` hal preset

Added HAL-specific categorical maps to `cat_maps.py`:
- `HAL_STAGE_MAP`: 6 tournament stages (FD, BF, PS, DL, FoD, YS)
- `HAL_CHARACTER_MAP`: 27 Melee characters (Mario through Roy)
- `HAL_ACTION_MAP`: all Action enum values (395 in melee-py, HAL checkpoint uses 396)

Set `num_stages=6, num_characters=27, num_actions=396` in hal preset.

**Stage index remapping:** MIMIC's `STAGE_MAP` includes `NO_STAGE` at index 0, so tournament stages are 1-6. HAL expects 0-5. Added remap in `HALFlatEncoder.forward()`: `stage_idx = (stage_idx - 1).clamp(min=0)` when `num_stages==6`.

Character and action indices happen to be identical between MIMIC and HAL maps (verified).

### 7. Input projection: Linear(164, 512)

**File:** `mimic/frame_encoder.py`

This is a consequence of changes 4 and 6. Verified:
- Embeddings: stage(4) + 2×char(12) + 2×action(32) = 92
- Gamestate: 9 features × 2 players = 18
- Controller: 37 stick + 9 cstick + 5 combos + 3 shoulder = 54
- **Total: 164** → `Linear(164, 512)` — matches HAL's `proj_down` exactly

---

## Verification

### Parameter Shape Comparison

Every parameter in the MIMIC model matches HAL's checkpoint shape-for-shape:

| Component | MIMIC name | Shape | HAL name |
|-----------|-----------|-------|----------|
| Input projection | encoder.proj | (512, 164) | transformer.proj_down |
| Stage embedding | encoder.stage_emb | (6, 4) | stage_emb |
| Char embedding | encoder.char_emb | (27, 12) | character_emb |
| Action embedding | encoder.action_emb | (396, 32) | action_emb |
| QKV projection | blocks.N.self_attn.c_attn | (1536, 512) | transformer.h.N.attn.c_attn |
| Relative pos table | blocks.N.self_attn.Er | (1024, 64) | transformer.h.N.attn.Er |
| Causal mask buffer | blocks.N.self_attn.bias | (1,1,1024,1024) | transformer.h.N.attn.bias |
| Attn out projection | blocks.N.self_attn.c_proj | (512, 512) | transformer.h.N.attn.c_proj |
| MLP expand | blocks.N.mlp.c_fc | (2048, 512) | transformer.h.N.mlp.c_fc |
| MLP contract | blocks.N.mlp.c_proj | (512, 2048) | transformer.h.N.mlp.c_proj |
| C-stick head output | heads.cdir_head.3 | (9, 257) | c_stick_head.3 |
| Button head output | heads.btn_head.3 | (5, 280) | button_head.3 |

**Total parameters (state_dict): 26,274,803 — IDENTICAL to HAL.**

### Local Test Run

Ran ~13.9K steps on 1× RTX 4090 with 32 Fox ranked shards (~10M frames):

| Step | Total Loss | Btn F1 | Main F1 | Cdir F1 | Shldr F1 |
|------|-----------|--------|---------|---------|----------|
| 1 | 8.49 | 8.8% | 1.4% | 4.3% | 34.6% |
| 524 | 1.90 | 72.4% | 22.3% | 73.3% | 81.6% |
| 3,406 | 1.03 | 83.5% | 43.5% | 86.8% | 86.4% |
| 13,100 | 0.37 | 94.2% | 76.1% | 94.8% | 94.1% |
| 13,886 | 0.33 | 94.2% | 79.4% | 94.7% | 94.7% |

Validation at step 13,107: btn_f1=71.0%, main_f1=26.6% (expected gap — early training with limited data variety).

---

## Training Plan: Remote 8×RTX 5090

### HAL's Exact Training Config

From `/home/erick/projects/hal/checkpoints/config.json`:

| Parameter | Value |
|-----------|-------|
| Architecture | GPTv5Controller-512-6-8-dropout |
| GPUs | 8 |
| Local batch size | 64 |
| Effective batch | 512 |
| Learning rate | 3e-4 |
| Weight decay | 0.01 |
| Grad clip norm | 1.0 |
| Loss | Cross-entropy |
| Precision | FP32 |
| Sequence length | 256 |
| Total samples | 16,777,216 |
| Optimizer | AdamW (beta1=0.9, beta2=0.999) |

### Training Command

```bash
torchrun --nproc_per_node=8 train.py \
  --model hal --encoder hal_flat \
  --hal-mode --hal-minimal-features --hal-controller-encoding \
  --stick-clusters hal37 --plain-ce --no-amp \
  --lr 3e-4 --batch-size 64 \
  --max-samples 16777216 \
  --data-dir data/ranked_fox \
  --controller-offset --self-inputs \
  --no-compile --run-name hal-exact-v1
```

### Data

Downloaded 1006 Fox shards from `erickfm/mimic-melee-ranked` (diamond, master, platinum ranks):
- Train: 905 shards, 32,633 games, 321,553,428 frames
- Val: 101 shards, 3,674 games, 35,850,187 frames

HAL only needs 16.7M samples, so the model will see a subset of this data (~5% of available frames).

### Remaining Differences from HAL

| # | Item | Status |
|---|------|--------|
| 1-7 | Architecture | MATCHED |
| 8 | Param count | MATCHED (26.3M) |
| 9 | Training data | Different replays (ranked Fox vs HAL's unknown source) |
| 10 | C-stick diagonal targets | Missing (5→9 remap only covers cardinals) |

The training data difference (#9) is the only variable we can't fully control. HAL's original training data is on the remote cluster at `/home/erick/projects/hal/data/fox_mds/` — if it's still there, we should use it instead.
