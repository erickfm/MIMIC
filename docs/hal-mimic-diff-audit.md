# HAL vs MIMIC: Definitive Diff Audit

## Purpose

Exhaustive list of every difference between HAL's working pipeline and MIMIC's pipeline. Each item is marked as MATCHED, UNMATCHED, or VERIFIED IRRELEVANT based on actual tests.

## Status Summary

HAL 4-stocked level 9 CPU. MIMIC's model produces 25/256 non-NONE frames on training data (good) but ~3% non-NONE at inference (bad). The gap is in inference, not training.

---

## TRAINING

### Data
- [x] **MATCHED**: Same 3,230 Fox .slp files from `erickfm/slippi-public-dataset-v3.7`
- [ ] **UNMATCHED**: Data format differs. HAL uses MDS shards (mosaicml-streaming). MIMIC uses .pt shards (custom). Both derive from the same .slp files but preprocessing differs.

### Target alignment
- [x] **MATCHED**: `--reaction-delay 0 --controller-offset` gives HAL's exact behavior: predict frame i's controller from frame i's gamestate + frame i-1's controller. Verified via data inspection (Finding 15).

### Model architecture
- [x] **MATCHED**: 512-d, 6 layers, 8 heads, dropout=0.2 (`--model hal`)
- [x] **MATCHED**: Flat encoder (`--encoder flat`)
- [x] **MATCHED**: Single-label 5-class buttons, combined 3-class shoulder, LN heads (`--hal-mode`)
- [x] **MATCHED**: Autoregressive heads with detached gradients
- [ ] **UNMATCHED**: Position encoding. HAL uses relative (Music Transformer skew). MIMIC uses RoPE.
- [ ] **UNMATCHED**: Input projection. HAL concatenates all embeddings + gamestate + controller into one vector, projects to d_model via single Linear. MIMIC uses the flat encoder which processes each feature group through separate MLPs, then concatenates and projects.

### Loss
- [x] **MATCHED**: Plain cross-entropy, unweighted sum (`--plain-ce`)

### Optimizer / schedule  
- [x] **MATCHED**: AdamW, lr=3e-4, betas=(0.9, 0.999), wd=1e-2
- [x] **MATCHED**: No warmup (`--no-warmup`)
- [x] **MATCHED**: CosineAnnealingLR to 1e-6 (`--cosine-min-lr 1e-6`)
- [x] **MATCHED**: Grad clip 1.0

### Training hyperparameters
- [x] **MATCHED**: seq_len=256 (`--seq-len 256`)
- [x] **MATCHED**: FP32 (`--no-amp`)
- [ ] **UNMATCHED**: Batch size. HAL uses 512/GPU (effective 1024 on 2 GPUs). MIMIC uses 64/GPU (effective 512 on 8 GPUs).
- [ ] **UNMATCHED**: Total samples. HAL trained 16.8M samples. MIMIC trained 2M samples (default max_samples).

### Input features
- [x] **MATCHED**: Self-inputs enabled (`--self-inputs`)
- [x] **MATCHED**: Nana/projectiles dropped (`--lean-features`)
- [ ] **UNMATCHED**: Player numeric columns. HAL uses 9 per player. MIMIC uses 22 per player (includes ECB, speeds, hitlag/hitstun). `--hal-minimal-features` exists but crashes due to model/data shape mismatch.
- [ ] **UNMATCHED**: Controller feedback encoding. HAL encodes as 54-dim one-hot (cluster indices for all outputs concatenated). MIMIC encodes as raw floats (4 analog + 12 binary buttons + 1 categorical c_dir).
- [ ] **UNMATCHED**: Global features. HAL uses only stage embedding. MIMIC uses 20 global numeric values (distance, frame, 18 stage geometry columns).
- [ ] **UNMATCHED**: Flags. HAL uses facing, invulnerable, on_ground, jumps_left as numeric inputs. MIMIC has a separate 5-dim flags tensor (on_ground, off_stage, facing, invulnerable, moonwalkwarning).
- [ ] **UNMATCHED**: Action elapsed. HAL does not use action_frame. MIMIC includes it.
- [ ] **UNMATCHED**: Port/costume embeddings. HAL does not use port or costume. MIMIC embeds both.

### Stick clusters
- [x] **MATCHED**: 37 hand-designed clusters (`--stick-clusters hal37`), re-clustered at runtime from raw main_x/main_y

### C-stick
- [ ] **UNMATCHED**: HAL uses 9 clusters (cardinals + diagonals). MIMIC uses 5 classes (cardinals only).

---

## INFERENCE

### Context window
- [x] **MATCHED**: Pre-fill to seq_len=256 on first frame. Verified batch 1 has shape (1, 256, ...).

### Output decoding
- [x] **MATCHED**: Multinomial sampling for buttons, sticks, shoulder (not argmax).

### Controller feedback
- [x] **MATCHED** (partially): `_prev_sent` tracks actual sent values and writes them to next frame's row. Verified via FB_CHECK: buttons show up (btns=[0,0,1,0,0] when X pressed).
- [ ] **UNMATCHED**: Encoding format. MIMIC feeds back raw float values (main_x, main_y, binary buttons). HAL feeds back one-hot cluster indices (54-dim vector). The model was trained seeing raw floats, so this is internally consistent — but the representation differs from HAL.

### Console setup
- [x] **MATCHED** (as of latest fix): `is_dolphin=True`, `tmp_home_directory=True`, `setup_gecko_codes=True`, `blocking_input=False`, `online_delay=0`, matching HAL's `get_gui_console_kwargs`.

### Feature values at inference
- [ ] **UNMATCHED**: ECB columns (8 per player) have garbage values at inference from live game. Clamped to [-10, 10] but training data has clean values. Model sees different distributions.
- [ ] **UNMATCHED**: Speed columns (5 per player) may differ between live game and training data normalization.
- [ ] **UNMATCHED**: Global numeric (distance, frame, stage geometry) — inference values differ significantly from training (mean -1.12 vs 0.30).
- [ ] **UNMATCHED**: Action encoding. Training data action indices go up to 395. Inference actions differ in distribution.
- [ ] **UNMATCHED**: Opponent controller (opp_analog, opp_buttons, opp_c_dir). Present in training data. Was MISSING in inference until recently. Now present via game readback but values may differ from training encoding.

### Warmup / frame skipping
- [ ] **UNMATCHED**: HAL skips `eval_warmup_frames` (1 frame). MIMIC does not skip any frames.

---

## PROVEN IRRELEVANT (tested, no impact)

- Opponent controller inputs: zeroing opp_buttons/analog/c_dir didn't change prediction count (12/256 → 12/256). The model doesn't rely on opponent controller state.

---

## PROVEN IMPORTANT (tested, significant impact)

- **Self-controller feedback**: Zeroing self_buttons/analog/c_dir reduced non-NONE from 25/256 to 12/256 (halved). The model relies heavily on self-controller state to decide when to act.
- **Context window length**: T=1 input produces degenerate NONE=100%. T=256 (pre-filled) is required.
- **Multinomial vs argmax**: Argmax always picks NONE (97% prior). Multinomial allows rare actions to fire.

---

## MOST LIKELY REMAINING CAUSES

Based on the evidence:

1. **Feature value mismatch at inference** — the model sees different numeric distributions at inference vs training. This is confirmed by tensor comparison. The 13 extra features (ECB, speeds, stage geometry, action_elapsed, port, costume) have different value distributions at inference, and some (ECB) have garbage values even after clamping.

2. **Controller feedback encoding** — MIMIC feeds raw floats but HAL feeds one-hot cluster indices. Both models were trained with their own encoding so this is internally consistent. But the raw float encoding may be less informative for the model.

3. **Position encoding (RoPE vs relative)** — untested. Could affect how the model attends to temporal patterns.

4. **Batch size / total samples** — MIMIC trained with effective batch 512 for 2M default samples. HAL trained with effective batch 1024 for 16.8M samples. MIMIC may be undertrained.
