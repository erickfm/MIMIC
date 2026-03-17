# Diagnostic Evaluation Report — 2026-03-17

## Summary

Evaluated 5 top checkpoints from the full-sweep using new diagnostic metrics.
**Finding: transition-frame accuracy is the primary bottleneck, not overall accuracy.**

## Results Table

| Model | btn_f1 | main_f1 | main_top1 | crit_btn_f1 | btn_trans_acc | btn_steady_acc | stick_trans_acc | stick_steady_acc |
|-------|--------|---------|-----------|-------------|---------------|----------------|-----------------|------------------|
| g1-d100-r2 (91.8%) | 89.2% | 35.3% | 84.7% | 89.5% | **7.4%** | 99.6% | **28.5%** | 96.9% |
| g2-base (91.2%) | 88.6% | 35.4% | 84.7% | 88.3% | **8.2%** | 99.5% | **27.4%** | 96.7% |
| g2-drop05 (90.1%) | 89.2% | 38.3% | 85.1% | 86.7% | **7.1%** | 99.3% | **27.8%** | 97.2% |
| g1-d100-r1 (89.0%) | 88.4% | 34.2% | 84.9% | 87.2% | **9.6%** | 99.1% | **25.0%** | 96.2% |
| g1-d50-r1 (89.7%) | 89.4% | 34.5% | 84.6% | 89.8% | **6.4%** | 99.6% | **28.8%** | 96.4% |

Note: wandb btn_f1 (from training) was computed differently from eval btn_f1 -- different val batches, different random samples. The relative ordering holds but absolute numbers differ slightly.

## Key Findings

### 1. Transition accuracy is the bottleneck (90%+ gap)

- **Button transitions:** 6-10% accuracy vs 99%+ on steady-state frames. The model fails to predict when buttons should change.
- **Stick transitions:** 25-29% accuracy vs 96%+ on steady-state frames. The model fails to predict when stick position should change.
- Transitions are only ~7% of button frames and ~17% of stick frames, so the model gets high overall accuracy by always predicting "no change."
- All Melee tech skill requires frame-precise input CHANGES. This directly explains why inference looks bad.

### 2. main_f1 is low (34-38%) despite 85% top-1 accuracy

- 63 stick clusters are not learned equally. The common clusters (neutral, cardinals) dominate.
- Macro F1 exposes that the model ignores rare clusters (precise angles for wavedash, DI, etc.).
- This is less of a problem than transitions since top-1 accuracy is decent, but it limits precise control.

### 3. Stick confidence is only 75%

- The model hedges between clusters ~25% of the time.
- This means the argmax prediction is correct but not confident, which could cause instability in inference (small perturbations flip the prediction).

### 4. Neutral rate is NOT the problem

- Predicted neutral rate (35-39%) matches ground truth (31-36%) within 1.08x.
- The model isn't "lazy" -- it just doesn't know WHEN to change.

### 5. Per-button breakdown is reasonable

- Critical buttons (A, B, X, Y, L, R) range from 72-94% F1.
- L and R (shield/wavedash) are the strongest at 91-94%.
- X (jump) is weakest at 72-89% -- varies most across models.
- D-pad and START are correctly predicted as never-pressed.

## Implications for Next Experiments

The core problem is NOT:
- Model too small (g2-base at 54.8M performs identically)
- Not enough data (100% data = same as 50%)
- Wrong dropout (all models show same pattern)

The core problem IS:
- **Class imbalance in time**: 93% of frames are steady-state, so the model optimizes for "don't change"
- **No explicit incentive for transition accuracy**: the loss treats every frame equally

### Promising directions

1. **Transition-weighted loss**: Upweight frames where the ground truth changes. Detect transitions in the target and multiply loss by 5-10x on those frames.
2. **Focal loss on transitions**: Apply higher focal gamma specifically to transition frames.
3. **Action chunking / multi-frame prediction**: Predict the next N frames instead of just 1. This forces the model to predict upcoming changes.
4. **Auxiliary "change prediction" head**: Add a binary head that predicts "will input change next frame?" -- gives the model explicit signal about transitions.
5. **Data augmentation at transitions**: Oversample sequences that contain transitions during training.
6. **Sequence-level loss**: Instead of per-frame CE, use a loss that penalizes incorrect transition timing (e.g., CTC-style).
