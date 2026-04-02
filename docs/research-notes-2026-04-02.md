# Research Notes — 2026-04-02

## Context

Continued from [2026-03-31](research-notes-2026-03-31.md). The `mimic-hal-fox-co` run (reaction_delay=0, controller_offset) finished training on Machine E. We now have three MIMIC training runs and one HAL run on the same 3.2k Fox replay data, plus closed-loop eval results.

---

## Finding 15: controller_offset confirmed correct via data inspection

Verified that `--controller-offset` properly hides the answer from the model:

```
Frame 3048 (target X goes 0→1):
  WITHOUT offset: model sees X=1, target X=1 → answer in input (trivial)
  WITH offset:    model sees X=0 (from frame 3047), target X=1 → must predict (correct)
```

At frame 3048, the player first presses X. Without offset, the model sees X=1 in `self_buttons` and just needs to copy it. With offset, the model sees frame 3047's X=0 and must predict the new press from gamestate context alone.

On held buttons (frame 3049+), state[i-1] already shows X=1 and target is still X=1 — the model sees "I was pressing X" and correctly predicts "keep pressing X." This matches HAL's `frame_offset=-1` behavior exactly.

---

## Finding 16: reaction_delay=0 without offset is a degenerate task

Three training configurations on the same Fox data:

| Config | Val Loss | bf1 | mf1 | NONE pred rate | Closed-loop |
|--------|----------|-----|-----|----------------|-------------|
| rd=1, no offset | 0.707 | 87.8% | 54.5% | ~92% at inference | Presses some buttons (148/3455 frames) |
| rd=0, no offset | 0.078 | 100% | 87% | 100% at inference | Completely frozen |
| **rd=0 + offset** | **0.089** | **91.4%** | **55.1%** | **80% on training data** | Presses buttons but character appears idle |

`rd=0` without offset gives trivially perfect metrics because `state[i] == target[i]` — the model copies its input. In closed-loop, neutral initial state → copies neutral → locked at NONE=100%.

`rd=0 + offset` is the correct HAL-equivalent setup. The model achieves better metrics than rd=1 (91.4% vs 87.8% bf1) because predicting "what to press now" given "what I pressed last frame" is an easier and more natural task than predicting "what to press next frame."

---

## Finding 17: MIMIC's model is well-calibrated on training data

Ran detailed probability analysis on the `mimic-hal-fox-co_best` checkpoint (step 26,832):

### On training data (val split, with controller_offset):

| Class | Target Rate | Pred Rate | F1 | Precision | Recall | Support |
|-------|-------------|-----------|-----|-----------|--------|---------|
| A | 6.2% | 6.4% | 91.0% | 90.3% | 91.7% | 1,597 |
| B | 5.4% | 5.6% | 93.8% | 92.9% | 94.6% | 1,391 |
| X (jump) | 7.7% | 7.9% | 91.3% | 90.2% | 92.4% | 1,972 |
| Z | 0.3% | 0.3% | 89.0% | 89.0% | 89.0% | 82 |
| NONE | 80.3% | 79.9% | 98.1% | 98.3% | 97.9% | 20,558 |

**Overall accuracy: 96.9%. Avg loss: 0.089.**

The model's NONE prediction rate (79.9%) matches the target rate (80.3%) almost exactly. Per-class F1 is 89-98% across all classes. The model is well-calibrated.

### Comparison with HAL's best checkpoint (5.2M samples):

| Metric | HAL best | MIMIC co_best |
|--------|----------|---------------|
| Mean NONE prob | 86.8% | 76.5% |
| Mean X (jump) prob | 6.7% | 16.6% |
| Frames NONE<90% | 19.0% | 25.9% |

**MIMIC is actually MORE confident on button presses than HAL** when evaluated on training data with the correct dataloader settings (controller_offset=True). The earlier comparison showing MIMIC at 92.4% NONE was wrong — it used the wrong dataloader config (no controller_offset), which corrupted the self-input features.

---

## Finding 18: The inference gap is confirmed as a feature encoding mismatch

### Evidence:

| Setting | NONE rate |
|---------|-----------|
| Training data (val split, controller_offset) | **80%** |
| Live inference (Dolphin) | **~92-97%** |

The same model produces dramatically different probability distributions depending on whether it sees training data or live game data. The model itself is fine — the inference pipeline constructs features that look different from training features.

### What this rules out:
- ❌ Model calibration — model is well-calibrated on training data
- ❌ Architecture differences from HAL — model has better metrics than HAL
- ❌ reaction_delay / controller_offset — correctly matching HAL
- ❌ Context window pre-fill — implemented
- ❌ Multinomial sampling — implemented
- ❌ Game readback for feedback — implemented (in HAL mode)

### What this points to:
- ✅ Feature normalization differs between training and inference
- ✅ Categorical encoding differs (cat_maps lookup paths)
- ✅ Extra features (ECB, speeds) have garbage values at inference despite clamping
- ✅ Self-controller encoding at inference doesn't match training format
- ✅ Frame encoder receives different tensor shapes/values

### Next step:
Save a training data batch and an inference batch, compare tensor values feature-by-feature. The discrepancy will be visible in the raw numbers. This is the definitive debugging approach — stop guessing, compare tensors directly.

---

## Closed-Loop Eval Summary

| Model | Config | Button presses | Non-neutral sticks | Beat CPU? |
|-------|--------|---------------|-------------------|-----------|
| HAL (Gu's code) | HAL eval.py | Many (coherent) | Many | **Yes** (4-stocked) |
| MIMIC rd=1 (old) + argmax | Old inference | 0 | 0 | No |
| MIMIC rd=1 + sampling + fixes | Fixed inference | 148 / 3,455 | 1,137 / 3,455 | No (Dolphin disconnected) |
| MIMIC rd=0 (no offset) | Fixed inference | 0 | 0 | No (NONE=100%) |
| MIMIC rd=0+offset | Fixed inference | 42 / 3,455 | 905 / 3,455 | No (appears idle) |

The rd=1 model with sampling actually produced more button presses (148) than the rd=0+offset model (42), despite having worse training metrics. This is because the rd=1 model's NONE confidence was lower at inference (the wrong dataloader in my earlier test made it look like 92%, but the actual inference behavior was more diverse due to different feature distribution).

---

## Machine Status

| Machine | Task | Status |
|---------|------|--------|
| E | `mimic-hal-fox-co` training | **Completed** (step 31,250) |
| C | Ranked dataset sharding | Unknown (need to check) |
| F | Ranked dataset sharding | Unknown (need to check) |
| G | Ranked dataset sharding + slippi public upload | Unknown (need to check) |

---

## Next Steps

1. **Debug inference feature mismatch** — save a training batch tensor and an inference batch tensor, diff them feature-by-feature to find exactly which features differ and by how much
2. Fix the identified mismatches
3. Re-run closed-loop eval
4. If it works, scale to more data (ranked dataset upload in progress)
