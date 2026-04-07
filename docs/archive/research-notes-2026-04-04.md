# Research Notes — 2026-04-04

## Summary

Extensive investigation into the MIMIC/HAL calibration gap. Multiple architectural and encoding changes were implemented and tested. The definitive finding: **HAL's own trained model performs poorly through our inference code**, proving the gap is in inference preprocessing — not training, architecture, or model calibration.

---

## Changes Implemented

### 1. HAL-style One-Hot Controller Feedback Encoding

**What:** Replaced raw float controller feedback (12 binary buttons + 4 analog floats + 5-class c_dir) with a single concatenated one-hot vector matching HAL's encoding.

**Components:**
- Main stick: nearest of 37 hand-designed clusters → 37-dim one-hot
- C-stick: nearest of 9 clusters → 9-dim one-hot  
- Buttons: data-driven combo classes → N-dim one-hot
- Shoulder: max(L,R) nearest of [0.0, 0.4, 1.0] → 3-dim one-hot
- Total: 37 + 9 + N + 3 dimensions

**Data-driven button combos:**
- `tools/build_controller_combos.py` scans all shards, finds unique button combinations
- Collapses X|Y → Jump, L|R → Shoulder before counting
- Fox data: 32 unique combos found across 59M frames
- Also tested with HAL's exact 5 classes (A, B, Jump, Z, NONE)

**Files:** `mimic/features.py`, `mimic/dataset.py`, `mimic/frame_encoder.py`, `mimic/model.py`, `train.py`, `inference.py`, `tools/build_controller_combos.py`

### 2. HAL-Flat Encoder

**What:** New encoder type `hal_flat` that matches HAL's exact input projection — single concat → Linear, no per-group MLPs.

**Architecture:**
```
concat([
    stage_emb(4), char_emb(12)*2, action_emb(32)*2,  # = 92 dims
    self_numeric + self_flags (10),                    # = 10 dims  
    opp_numeric + opp_flags (10),                      # = 10 dims
    self_controller (37+9+N+3),                        # = 49+N dims
]) → Linear(164, 512) → Dropout → Transformer
```

**Embedding dimensions match HAL exactly:** stage=4, character=12, action=32.

**File:** `mimic/frame_encoder.py` (HALFlatEncoder class)

### 3. HAL Normalization

**What:** Switched from z-score normalization to HAL's exact per-feature transforms.

**Transforms by feature:**

| Feature | HAL Transform | Formula |
|---------|--------------|---------|
| pos_x, pos_y | standardize | (x - mean) / std |
| percent, stock, jumps_left | normalize | 2*(x - min)/(max - min) - 1 |
| on_ground, facing, invulnerable | normalize | 2*(x - min)/(max - min) - 1 |
| shield_strength | invert_normalize | 2*(max - x)/(max - min) - 1 |
| invuln_left | normalize | (always 0, degenerate) |

**Critical difference found:** For a STANDING frame:

| Feature | MIMIC z-score | HAL normalize |
|---------|--------------|---------------|
| jumps_left=2 | +0.657 | -0.333 |
| shield_strength=60 | +0.339 | -1.000 |
| invulnerable=0 | 0.000 | -1.000 |

Three features had opposite signs or completely different values for the same game state.

**Implementation:** 
- `hal_norm.json` stores HAL's stats (min, max, mean, std, transform type) per feature
- `slp_to_shards.py --hal-norm` applies HAL transforms during tensorization
- `inference.py` loads hal_norm.json and applies at runtime

### 4. Baked Controller Encoding in Shards

**What:** Controller encoding computed from raw values during tensorization (before normalization), stored directly as `self_controller` in shards. No runtime encoding needed.

**Why:** Runtime encoding required denormalizing values back to raw, which was buggy when switching normalization schemes. Baking avoids this entirely.

**Verification:** `tools/verify_inference.py` confirmed the baked encoding matches what inference produces — except the baked version is always correct (uses raw values directly).

### 5. Codebase Cleanup

**Deleted 8 legacy files:**
- `tools/tensorize.py` — superseded by slp_to_shards.py
- `tools/upload_dataset.py` — superseded by slp_to_shards.py --upload
- `tools/_legacy_extract.py` — marked legacy
- `tools/test_closed_loop.py` — hardcoded paths, broken with HAL mode
- `tools/test_closed_loop_comprehensive.py` — same
- `tools/test_airdodge_methods.py` — one-off experiment
- `tools/generate_wavedash_replay.py` — depended on legacy extract
- `tools/sweep_single_gpu.sh` — outdated hyperparameter combos

**Rewrote eval.py:** Full HAL mode support (single-label buttons, combined shoulder, per-combo F1 breakdown).

**Fixed inference.py:** Simplified press_output(), fixed top-3 logging for combo classes, consolidated imports, added btn-temperature flag, added HAL normalization for flags.

**Vectorized train.py:** `_multi_hot_to_combo_label` replaced per-frame Python loop with integer encoding + tensor lookup (~100x faster).

### 6. Verification Tools

**`tools/verify_inference.py`:** Extracts frames from .slp, processes through both training pipeline and inference pipeline, compares every tensor value. Confirmed ALL MATCH across all 8 MIMIC checkpoints for all feature types.

**`tools/run_hal_model.py`:** Minimal reimplementation of HAL's GPTv5Controller architecture (relative position encoding, autoregressive heads). Loads HAL's checkpoint and runs through our Dolphin loop.

---

## Training Runs

All runs on Machine E (8× RTX 5090), Fox public replay data (~53M frames).

| Run | Encoder | Norm | Ctrl Enc | Classes | rd | Val bf1 | STANDING NONE |
|-----|---------|------|----------|---------|-----|---------|---------------|
| HAL (reference, HAL code) | GPTv5 | HAL | HAL native | 5 | 0-eq | ~88% | **94.8%** |
| hal-flat-32class | hal_flat | z-score | one-hot 32 | 32 | 0 | 80.3% | 99.0% |
| hal-flat-5class | hal_flat | z-score | one-hot 5 | 5 | 0 | 90.7% | 99.9% |
| hal-flat-5class-rd1 | hal_flat | z-score | one-hot 5 | 5 | 1 | 81.1% | 99.9% |
| hal-5c-halnorm (buggy) | hal_flat | HAL (dataset convert) | one-hot 5 | 5 | 0 | 90.5% | 98.4% |
| hal-halnorm-v2 (32-class) | hal_flat | HAL (baked) | baked 32 | 32 | 0 | 80.8% | 98.7% |
| hal-5c-halnorm-v3 | hal_flat | HAL (baked) | baked 5 | 5 | 0 | 90.8% | 99.0% |

**Observations:**
- 5-class consistently outperforms 32-class on val bf1 (90.7-90.8% vs 80.3-80.8%)
- HAL normalization improved STANDING calibration slightly (99.9% → 98.4-99.0%)
- No configuration achieved HAL's 94.8% STANDING calibration
- Calibration gap exists from step 1 of training — not overfitting
- rd=1 did not meaningfully improve calibration (99.9%)

---

## Bugs Found and Fixed

### 1. Flag Normalization Mismatch (Critical)
**Bug:** Inference applied z-score normalization via `norm_stats` dict. Flags (on_ground, facing, invulnerable) were NOT in norm_stats (they're boolean 0/1, stored raw in shards). With HAL normalization, shards stored flags as [-1, +1], but inference passed them as raw [0, 1].

**Impact:** Model trained on [-1, +1] flags received [0, 1] at inference. Complete feature mismatch.

**Fix:** Added second normalization pass in `_process_one_row` for HAL-norm columns not in norm_stats.

### 2. Controller Encoding Denormalization (Critical)
**Bug:** `encode_controller_onehot` in dataset.py denormalized analog values using z-score stats (`raw = val * std + mean`) before finding nearest stick clusters. When shards switched to HAL normalization, the z-score denormalization produced wrong raw values → wrong cluster assignments.

**Verified:** `verify_inference.py` showed 14/16200 mismatches in controller encoding, with massive errors (neutral stick assigned to full-left cluster, dist 0.49 vs 0.006).

**Fix:** Baked controller encoding into shards from raw values (pre-normalization). No runtime denormalization needed.

### 3. 32-Class Combos Accidentally Used Instead of 5
**Bug:** First HAL-norm training run loaded 32-combo controller_combos.json instead of the intended 5-class version. The run completed with 32 classes, and the second launch (with fixed 5-class file) crashed because GPUs were still occupied by the first run.

**Impact:** Wasted ~1 hour of training + confusion about which checkpoint was which.

**Fix:** Manual verification of checkpoint's `n_controller_combos` field.

### 4. Inference Logging Crash (High)
**Bug:** `log_frame()` in InferenceLogger tried to index `IDX_TO_BUTTON[i]` with combo class indices (up to 31), but list only had 12 elements.

**Fix:** Added HAL-aware logging path that uses combo names.

### 5. Default CPU Level (Medium)
**Bug:** Default CPU level was 7, HAL evaluates against level 9.

**Fix:** Changed default to 9.

---

## Definitive Finding: The Inference Preprocessing Gap

### The Test

Loaded HAL's own trained checkpoint (the one that 4-stocks level 9 CPU through HAL's `play.py`) into our reimplemented inference loop (`tools/run_hal_model.py`).

### The Result

**HAL's model through HAL's play.py:** 4-stocks level 9 CPU. Well-calibrated predictions, intentional gameplay matching training data.

**HAL's model through our run_hal_model.py:** Performs poorly. Does not 4-stock. Does not look like training data. Similar quality to MIMIC models through our inference.

### What This Proves

1. **The model is good.** HAL's checkpoint contains a working Melee AI.
2. **Our inference preprocessing is wrong.** The same model produces bad gameplay through our code.
3. **All previous calibration measurements on training data were misleading.** The 94.8% vs 99% STANDING NONE gap we measured may have been a symptom of different preprocessing between train-time eval and our offline measurement scripts.
4. **Training changes were likely irrelevant.** We matched architecture, encoding, normalization — but the model was fine all along. The issue is in how we convert live Dolphin gamestate → model input tensors at inference time.

### What This Does NOT Prove

We cannot yet say which specific part of the preprocessing is wrong:
- Feature extraction from gamestate (could be wrong values)
- Normalization (could be applying wrong transforms)  
- Controller encoding (could be encoding wrong values)
- Action/character mapping (could be mapping to wrong indices)
- Context window management (could be filling/rolling wrong)
- Output decoding (could be interpreting predictions wrong)
- Controller output (could be sending wrong values to Dolphin)

### Implications

1. **Stop investigating training differences.** The training produced a good model (proven by HAL's play.py). Our training may also produce good models — we just can't tell because our inference is broken.

2. **Focus on matching HAL's play.py exactly.** The reference implementation is `hal/eval/play.py` + `hal/preprocess/preprocessor.py`. Our `run_hal_model.py` needs to match this line-by-line.

3. **All previous claims about model quality need re-evaluation.** The STANDING calibration measurements, the val bf1 numbers — these were run on training data through the training pipeline, which is correct. But the inference behavior doesn't match because inference preprocessing differs.

4. **The verify_inference.py tool validated the wrong thing.** It confirmed training and inference pipelines produce the same tensors for the same raw data. But it didn't verify that the raw data extraction from Dolphin matches what HAL extracts. The Dolphin gamestate → raw values step could be different.

---

## Open Questions

1. **What exactly does HAL's `extract_eval_gamestate_as_tensordict()` extract?** This is the function that converts live Dolphin gamestate to the format HAL's preprocessor expects. We need to compare this to our gamestate extraction.

2. **How does HAL's `preprocessor.preprocess_inputs()` handle the controller offset at inference?** HAL applies `frame_offset=-1` during training but NOT during inference (the Dolphin latency provides natural offset). We apply `controller_offset` in the dataset but use `_prev_sent` at inference. Are these equivalent?

3. **How does HAL's context window filling work?** HAL fills from the left (position 0 → seq_len), predicting at the current position index. Our cache fills from the right (deque), always predicting at position -1. Both should be equivalent after warmup, but during the first 256 frames they differ significantly.

4. **Does HAL's output decoding match ours?** HAL uses `preprocessor.postprocess_preds()` to convert model outputs to controller inputs. We decode manually in `press_output()`. Any difference in how stick clusters are sampled, buttons are selected, or shoulder values are sent could cause different gameplay.

5. **Are the HAL action/character/stage mappings correct in our reimplementation?** HAL uses `IDX_BY_CHARACTER` (27 characters) while MIMIC uses `CHARACTER_MAP` (all characters). If HAL's Fox is at index 1 in HAL's mapping but a different index in MIMIC's, the character embedding is wrong.

---

## Remaining Unverified Claims

Every claim below needs to be independently verified against HAL's actual code:

- [ ] Our gamestate extraction produces the same raw values as HAL's `extract_eval_gamestate_as_tensordict()`
- [ ] Our HAL normalization produces the same normalized values as HAL's `preprocess_input_features()`
- [ ] Our controller encoding produces the same one-hot vector as HAL's `concat_controller_inputs()`
- [ ] Our action index mapping matches HAL's `IDX_BY_ACTION`
- [ ] Our character index mapping matches HAL's `IDX_BY_CHARACTER`
- [ ] Our stage index mapping matches HAL's `IDX_BY_STAGE`
- [ ] Our context window filling is equivalent to HAL's sliding window
- [ ] Our output decoding (stick, c-stick, buttons, shoulder) matches HAL's `postprocess_preds()`
- [ ] Our Dolphin controller commands produce the same game behavior as HAL's `send_controller_inputs()`

Each of these is a potential source of the inference gap. None have been verified.
