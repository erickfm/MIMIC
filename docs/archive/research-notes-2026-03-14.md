# Research Notes — 2026-03-14

## Context

MIMIC (Melee Imitation Model for Input Cloning) — a behavior cloning bot for Super Smash Bros. Melee. Transformer-based model trained on Slippi replay data to predict controller inputs from game state.

## Problem Statement

The bot's inference output looks nothing like human play. It mostly stands still or walks off the edge. Training metrics (btn_f1, main_top1_acc) appeared saturated, masking the real failure mode.

---

## Pipeline Audit

### Feature Mismatches Found

Several discrepancies between training data feature extraction and live inference feature extraction in `inference.py` were identified and fixed. These were subtle (normalization differences, column ordering) but contributed to the model seeing out-of-distribution inputs at inference time.

### Stick Prediction is the Core Failure

Offline diagnostic (`tools/diagnose.py`, `tools/test_closed_loop_comprehensive.py`) showed:
- **Button prediction**: 94-98% F1 — appears strong
- **Stick prediction**: ~1% top-1 accuracy — catastrophically bad

The model predicts neutral stick on nearly every frame regardless of context.

---

## Closed-Loop Evaluation: Wavedash Test

### Motivation

Narrowed the problem to a minimal reproducible case: Falco wavedashing back and forth on FD against a stationary Falco. If the model can't learn this simple, repetitive pattern, there's a fundamental issue.

### Data Generation

Built `tools/generate_wavedash_replay.py` using libmelee to programmatically control P1 Falco. Key findings during development:

- **Airdodge requires digital L press**: `press_button(BUTTON_L)` is required. `press_shoulder(BUTTON_L, 1.0)` (analog only) does NOT trigger airdodge, only ground shield. Discovered via systematic testing in `test_airdodge_methods.py`.
- **Y must be released for 1 frame between presses** for the game engine to register a new jump input.
- **Airdodge input must be sent on the first airborne frame**, not during jumpsquat (KNEE_BEND). Pressing L during jumpsquat "consumes" the input.
- **Slippi replay files from libmelee are often corrupt**, so we bypassed `.slp` entirely and exported directly to parquet using functions from `slippi-frame-extractor`.

Final output: 7200 frames, 336 perfect wavedashes, exported as parquet with the canonical schema.

### Training

Preprocessed with `preprocess.py` and `tools/build_clusters.py`. Stick clusters correctly identified the 3 unique positions (neutral, left-down, right-down). Trained `--model small` for 3000 steps with `--stick-loss clusters --autoregressive-heads`.

### Results: Model Collapses to Majority Class

Despite converging loss, inference revealed:
- **Main stick**: Always neutral (0.50, 0.50). Never predicted wavedash angles.
- **BUTTON_Y**: Predicted ~5.9% probability (matching its ~4.7% base rate in training data). Random timing.
- **BUTTON_L**: Predicted 0.0% probability. Never fires.
- **Shoulders**: Always 0.

The model learned to minimize loss by predicting the marginal frequency of each output, ignoring sequential context entirely.

---

## Root Cause Analysis

### Why Multi-Label BCE Collapses on Rare Buttons

With 12 independent sigmoid outputs, each button is an independent binary classifier. For Y (pressed ~5% of frames), the BCE-minimizing output is sigmoid ≈ 0.05 — the base rate. Even in the correct context, the model only needs to nudge the logit slightly (e.g., to 0.15) to reduce loss. It never reaches the 0.5 inference threshold.

**The previous wavedash training used `--btn-loss bce` (the default), not focal.** The `focal_bce()` function exists in the code but was never enabled for this experiment.

### Why Plain Cross-Entropy Collapses on Stick Clusters

`focal_loss()` is implemented and used for c-dir, but stick clusters and shoulder bins use bare `F.cross_entropy`. With ~95% neutral stick frames, the loss-minimizing prediction is "always neutral."

### Inference Threshold is Too Aggressive

Even if the model learned to predict Y at 0.3 on the correct frames (a significant improvement over 0.05), the hardcoded 0.5 threshold in `PredictionHeads.threshold_buttons()` would never fire.

---

## HAL (Eric Gu) — Reference Analysis

Eric Gu's HAL bot: 20M param Transformer, 3 billion frames, 95% winrate vs CPU.

### What HAL Does

- **Standard cross-entropy everywhere** — no focal loss, no class weighting, no oversampling
- **Single-label buttons**: {NO_BUTTON, A, B, JUMP, Z} via "most recent press" rule. Cross-entropy forces the model to commit to exactly one class per frame, which structurally avoids the sigmoid threshold problem.
- **Coarse stick discretization**: 21-37 k-means clusters for main stick (we use 63)
- **Autoregressive MLP output heads** (we have this too)
- **Teacher forcing** with previous controller inputs
- **Uniform random sampling** over episodes — no special handling of rare actions

### What HAL Does NOT Do

- No focal loss
- No class weights
- No oversampling of rare actions
- No curriculum learning
- No data augmentation
- No Gaussian histogram loss (tried it, didn't help — "injected more noise than necessary")

### HAL's #1 Recommendation

> "By far the most important thing I did was to overfit on a single synthetic example, and debug until I was sure that my training and closed loop eval data distributions perfectly matched."

He generated a `multishine.py` script, trained until epsilon loss, verified perfect reproduction in emulator. This is exactly what our wavedash test should achieve.

### Why HAL's Single-Label Approach Works Without Focal Loss

With softmax CE over {NO_BUTTON, A, B, JUMP, Z}: on the frame where JUMP should be pressed, the gradient directly pushes all probability mass toward JUMP. The model doesn't need to cross any threshold — it just needs to make JUMP the argmax. This naturally handles class imbalance because CE always gives gradient toward the correct class regardless of frequency.

---

## Planned Fix (Generalizable, Not Distribution-Specific)

### Decision: Keep Multi-Label Buttons

Melee has legitimate simultaneous button presses (grab = A+R, shield+button combos, etc.). Rather than losing this expressivity with single-label, we compensate with focal loss + threshold tuning.

### Changes

1. **Focal loss everywhere**: Enable `focal_bce` for buttons (was `bce`), extend `focal_loss()` to stick clusters and shoulder bins
2. **Lower inference threshold**: From 0.5 to ~0.2 for button sigmoid, with CLI-tunable `--btn-threshold`
3. **Overfit wavedash to epsilon**: Retrain with focal losses, verify correct closed-loop behavior

### If This Doesn't Work

Fall back to single-label buttons (HAL-style) as the nuclear option.

---

## Research on Class Imbalance (Non-Melee)

### Focal Loss

- Downweights easy examples via (1-p_t)^gamma modulation
- gamma=2.0 is standard, already implemented in our codebase
- Rarely used in LLM training because it "breaks key assumptions" of language modeling, but our task is structured prediction (fixed action space), not open-ended generation — focal loss is appropriate here.

### Temperature Sampling

- At inference: scaling logits by 1/T before softmax sharpens (T<1) or flattens (T>1) the distribution
- Useful for controlling diversity vs. precision in predictions
- We already have a `--temperature` argument in inference

### Behavior Cloning from Imbalanced Data (2025)

- Recent work (Springer, Autonomous Robots 2025) proves formally that imbalanced data leads to imbalanced policies under equal weighting
- Meta-gradient rebalancing and criticality-aware data curation are proposed solutions, but require distribution knowledge
- Training longer alone does NOT resolve long-tail challenges without additional interventions

### Key Takeaway

The most generalizable solutions are: (1) reformulate the loss to be less sensitive to class frequency (focal loss), and (2) reformulate the output space to reduce imbalance (coarser discretization, single-label). Distribution-specific solutions (weighting, oversampling) are effective but require knowledge we won't have for all of Melee.

---

## Phase 1 Results: Focal Loss Fix (same day, later session)

### Changes Made (~15 lines total)

1. **`mimic/model.py`**: Changed `btn_loss` default from `"bce"` to `"focal"`. Changed `btn_threshold` default from 0.5 to 0.2.
2. **`train.py`**: Replaced `F.cross_entropy` for stick clusters and shoulder bins with existing `focal_loss()` (which includes gamma=2.0 + label smoothing 0.1). Updated argparse help text.
3. **`inference.py`**: Added `--btn-threshold` CLI argument (default 0.2). Updated button firing logic and feedback loop to use configurable threshold instead of hardcoded 0.5.

### Training Results (wavedash-focal-p1, 10k steps)

Training progression:
- Steps 1-2000: btn_f1 ~66% (same as before — base rate prediction). btn_loss ~0.004.
- Steps 2000-2100: Brief instability spike (loss 0.30, f1 drops to 0%). Recovered.
- Steps 7000-7500: btn_f1 starts climbing — 73%, then 76%.
- Steps 8000-8500: Rapid acceleration — btn_f1 reaches 92-97%. Recall jumps from ~50% to 95%+.
- Steps 9000-10000: Convergence — btn_f1 ~98%, btn_loss 0.0002 (near epsilon).

Final metrics:
- **btn_f1: 98.0%**, btn_precision: 98.2%, btn_recall: 97.8%
- **btn_loss: 0.00025** (was ~0.004 with plain BCE — never improved)
- **loss_main: 0.00001**, loss_l: 0.0, loss_r: 0.0

### Offline Evaluation (200 windows, threshold=0.2)

| Metric | Before (plain BCE) | After (focal BCE) |
|--------|--------------------|--------------------|
| Stick cluster top-1 | ~1% | **96.0%** |
| Button Y F1 | 0% (never fired) | **100%** (conf 0.79-0.93) |
| Button L F1 | 0% (never fired) | **100%** (conf 1.0) |
| False positives | N/A | **0** |
| False negatives | N/A | **0** |

### Why Focal Loss Worked

The key mechanism: focal BCE with gamma=2.0 reduces the gradient contribution from easy negatives (the 95% of frames where a button is NOT pressed) by a factor of ~(0.05)^2 = 0.0025. This leaves the hard positives (frames where a button SHOULD be pressed but the model doesn't predict it) contributing almost full gradient. Over many steps, the model learns contextual predictions rather than marginal frequencies.

Interestingly, the model needed ~7000 steps to "break through" the base-rate plateau. The cosine LR schedule reaching its tail (lower LR) coincided with the breakthrough — the model was slowly building up correct contextual representations during the plateau, and the breakthrough came when the representation was good enough for the focal weighting to dominate.

### Implication for Full Training

Phase 2 (button-combo vocabulary) was not needed. Focal loss alone resolved the class imbalance problem on this controlled test case. The next step is to apply these same loss settings to the full Melee dataset and verify the model learns diverse actions across real gameplay scenarios.

Checkpoint: `checkpoints/wavedash-focal-p1_step010000.pt`
