# HAL vs MIMIC: Definitive Diff Audit

## PRINCIPLE

**We should NOT have to do anything special for MIMIC to reproduce HAL. If HAL doesn't do it, we shouldn't have to. No tricks, no seeding, no workarounds. If the model needs help, there's still a mismatch between training and inference.**

---

## Current Status

The model produces 21.2% non-NONE on training data but ~0% at inference. This is a train/inference mismatch. The mismatch has NOT been fully identified yet despite multiple debugging sessions. Prior "fixes" (clamping, pre-fill, feedback) were based on speculation, not confirmed root causes.

### What we KNOW:
1. Zeroing `self_buttons` on training data drops non-NONE from 215/1024 to 39/1024 (82% drop)
2. Zeroing `self_analog` drops to 63/1024 (71% drop)
3. At inference, `self_buttons` is ALL ZEROS in every diagnostic batch (1, 3, 300)
4. `_prev_sent` correctly tracks sent values, but sent buttons are always [0,0,0,0,0] because the model never presses buttons
5. The model never presses buttons because it doesn't see button feedback
6. This is a chicken-and-egg loop — BUT HAL doesn't have this problem

### What this means:
HAL also starts with no button presses on frame 1. HAL's model sees neutral controller feedback on the first frames. Yet HAL immediately starts pressing buttons. MIMIC doesn't. Therefore:
- Either MIMIC's model is less confident on initial actions (training difference), OR
- MIMIC's inference feeds the model different values than training even on the very first frame (encoding difference)

### THE REAL QUESTION:
On the very first frame of a training window (position 0 after controller_offset), what does `self_buttons` look like? It should be all zeros (offset shifts by -1, position 0 gets zeros). If the model learned to press buttons when `self_buttons` is zero (position 0 of training windows), then it should also press buttons at inference when `self_buttons` is zero. If it doesn't, something else in the input differs.

---

## TRAINING: MATCHED vs UNMATCHED

### Data
- [x] Same 3,230 Fox .slp files from `erickfm/slippi-public-dataset-v3.7`
- [ ] Data format: HAL=MDS, MIMIC=.pt. Same source but different preprocessing.

### Target alignment
- [x] `--reaction-delay 0 --controller-offset` matches HAL's frame_offset=-1. Verified (Finding 15).

### Model architecture
- [x] 512-d, 6 layers, 8 heads, dropout=0.2
- [x] Flat encoder
- [x] Single-label 5-class buttons, combined 3-class shoulder
- [x] Autoregressive heads with detached gradients
- [x] Plain CE loss, unweighted sum
- [ ] Position encoding: RoPE (MIMIC) vs relative/skew (HAL)
- [ ] Input projection: MIMIC has per-group MLPs then concat. HAL has single concat → Linear.

### Optimizer/schedule
- [x] AdamW, lr=3e-4, no warmup, cosine to 1e-6, grad clip 1.0

### Training params
- [x] seq_len=256, FP32
- [ ] Batch size: HAL=1024 effective, MIMIC=512 effective
- [ ] Total samples: HAL=16.8M, MIMIC=2M default

### Input features (CRITICAL — most likely source of mismatch)
- [x] Self-inputs enabled, nana/projectiles dropped
- [ ] Player numeric: MIMIC=22 cols (pos, pct, stock, jumps, 5 speeds, hitlag, hitstun, invuln, shield, 8 ECB). HAL=9 cols (pos, pct, stock, jumps, invuln, shield, facing, on_ground). **13 extra features in MIMIC.**
- [ ] Controller feedback encoding: MIMIC=raw floats (4 analog + 12 binary buttons + 1 cat c_dir). HAL=54-dim one-hot (cluster indices).
- [ ] Global features: MIMIC=20 numeric (distance, frame, 18 stage geometry). HAL=stage embedding only.
- [ ] Flags: MIMIC=5-dim tensor. HAL=inline in numeric.
- [ ] Action elapsed: MIMIC includes. HAL does not.
- [ ] Port/costume: MIMIC embeds both. HAL does not use.
- [ ] C-stick: MIMIC=5 classes. HAL=9 clusters.

---

## INFERENCE: MATCHED vs UNMATCHED

### Console setup
- [x] Matched HAL's `get_gui_console_kwargs` (gecko codes, tmp_home, etc.)

### Output decoding  
- [x] Multinomial sampling for all outputs

### Controller feedback
- [x] `_prev_sent` tracks actual sent values
- [!] Buttons in `_prev_sent` are always zero because model never presses buttons (chicken-and-egg). HAL doesn't have this problem — its model presses buttons from frame 1.

### Context window
- [x] Pre-filled to 256 frames
- [!] Pre-filled with copies of first neutral frame. During training, position 0 also has zeros for controller (due to offset). So this should be equivalent — but needs verification.

### Feature values at inference
- [ ] NOT VERIFIED: Do the actual tensor values for non-controller features (self_numeric, opp_numeric, global numeric, flags, action, etc.) match what the model sees during training for equivalent gamestates?
- [ ] ECB values: garbage at inference, clamped to [-10, 10]. Training has clean values. Clamping proven not to change predictions.
- [ ] self_analog normalization: inference values in [-0.45, 0.17], training values in [-1.6, 2.3]. **DIFFERENT RANGES** — unclear if this matters since they're from different gamestates.

---

## PROVEN FACTS

| Test | Result |
|------|--------|
| Zero self_buttons on training data | 215→39 non-NONE (82% drop) |
| Zero self_analog on training data | 215→63 non-NONE (71% drop) |
| Zero self_c_dir on training data | No change |
| Zero opp controller on training data | No change |
| Zero self_numeric on training data | No change |
| Zero global numeric on training data | No change |
| Clamp [-10,10] on training data | No change |
| Model on training data (10 batches) | 21.2% non-NONE (544/2560) |
| Model at inference | ~0-3% non-NONE |
| `_prev_sent` buttons at inference | Always [0,0,0,0,0] |
| `_prev_sent` analog at inference | Sometimes non-neutral (main_x varies) |

---

## STILL TODO (in priority order)

1. **Verify training position 0 behavior**: Check what the model predicts at position 0 of training windows (where self_buttons/analog are zeros due to controller_offset). If it predicts NONE there too, then the chicken-and-egg theory is correct. If it predicts actions, then something else differs at inference.

2. **Exact tensor comparison**: Take position 0 of a training window and compare EVERY value to position 0 of an inference window. Not summary stats — exact values for every feature. The mismatch will be visible.

3. **Run MIMIC model through HAL's eval pipeline**: If possible, adapter the model to work with HAL's eval.py. This eliminates inference.py entirely and tests whether the model itself is the problem or the inference code.

4. **Check if `self_analog` feedback is actually working**: The analog values at inference show some variation but the diagnostic batches show frozen values. There may be a similar issue to buttons.

5. **Match remaining feature differences**: If all else fails, match the feature set exactly (drop extra 13 columns, match encoding format). This requires `--hal-minimal-features` runtime masking fix + retraining.

---

## DEFINITIVE FINDING (2026-04-02)

### The gap is in TRAINING, not inference

Compared button predictions during STANDING (action=14) between HAL's best checkpoint and MIMIC's best checkpoint, both on their own training/val data:

| Metric | HAL (best, 5.2M samples) | MIMIC (co_best, 26k steps) |
|--------|-------------------------|---------------------------|
| Mean NONE during STANDING | **94.8%** | **99.9%** |
| Mean X (jump) during STANDING | **4.6%** | **0.0%** |
| Frames with NONE > 99% | **20.2%** | **99.8%** |
| Frames with NONE > 99.9% | **4.8%** | **99.7%** |

HAL's model gives jump a 4.6% chance during STANDING — enough for multinomial sampling to fire jump ~3 times per second. MIMIC gives jump 0.0% — multinomial never fires.

### What this means

The inference pipeline is NOT the problem. The model genuinely predicts differently. Both models trained on the same .slp replay data but produce very different calibrations during STANDING. The remaining unmatched training differences (see list above) are the cause.

### Inference is confirmed working

- Feature tensors at inference match training tensors in format and encoding
- self_analog mismatch (zeros vs normalized neutral) is NOT the cause (tested: no effect)
- Missing opp controller features are NOT the cause (tested: no effect)  
- Clamping is NOT the cause (tested: no effect)
- The 80→0 non-NONE drop when setting action=STANDING on training tensors proves this is about what the model learned, not how inference feeds it

### Most likely causes of the calibration gap (priority order)

1. **Data preprocessing** — HAL processes .slp→MDS differently than MIMIC processes .slp→.pt. The same replay may produce different feature values. HAL's preprocessing normalizes differently (min-max vs z-score), encodes controller differently (one-hot clusters vs raw floats), uses different feature columns.

2. **Total training samples** — HAL trained 16.8M samples on 27M frames. MIMIC may have trained fewer effective samples (2M default in train.py).

3. **Position encoding** — RoPE vs relative. Different temporal attention patterns could lead to different calibration.

4. **Input projection** — HAL's single concat→Linear vs MIMIC's per-group MLP→concat. Different feature mixing before the transformer.

5. **Feature set** — 13 extra features in MIMIC may dilute the signal, making the model less confident on any single prediction.
