# Research Notes — 2026-04-02 (Part B)

## Context

Continued from [2026-04-02](research-notes-2026-04-02.md). After confirming the inference pipeline is working correctly (Finding 18) and that the gap is in training calibration (Finding 21), we investigated why MIMIC's model predicts NONE=99.9% during STANDING while HAL predicts NONE=94.8%.

---

## Finding 22: The model reads the answer from the action embedding

With `reaction_delay=0` and `controller_offset`, the model sees frame i's action state and predicts frame i's button. We discovered that the action state changes ON THE SAME FRAME as the button press — not the next frame as assumed.

Data evidence:
```
frame 112: action=14 (STANDING)  target_press=0
frame 113: action=341 (B_ATTACK) target_press=1  ← action already changed
frame 114: action=341 (B_ATTACK) target_press=1
```

The model at frame 113 sees `action=341` and predicts `B=0.999`. It's reading the consequence (action state) to predict the cause (button press).

**Verified HAL has the same pattern:**
```
HAL frame 7: action=42  target=NONE
HAL frame 8: action=18  target=A   ← same-frame change
HAL frame 9: action=56  target=A
```

HAL also sees the resulting action on the same frame. Both models "cheat" equally.

### Why this isn't actually cheating

The action at frame i reflects the game engine's processing of inputs up through frame i-1. It's current state, not future information. At inference, the model sees the exact same thing — the current action state after Dolphin processed the previous frame's input.

The action is a consequence of PAST inputs, not the current one. If the player pressed jump on frame i-1, by frame i the action is KNEEBEND. The model sees KNEEBEND and predicts "hold jump" — this is correct behavior, not cheating.

---

## Finding 23: reaction_delay=1 was the wrong approach

With `reaction_delay=1`, the model predicts frame i+1's button from frame i's gamestate. At inference, the prediction gets applied on frame i (immediately), not frame i+1. This creates a 1-frame timing mismatch.

Training results comparison (same data, same architecture):

| Config | Val Loss | bf1 | mf1 | Inference timing match? |
|--------|----------|-----|-----|------------------------|
| rd=0, no offset | 0.078 | 100% | 87% | Yes but trivial (answer in input) |
| rd=0, offset | 0.598 | 91.6% | 55.6% | **Yes** — correct setup |
| rd=1, offset | 1.047 | 82.6% | 41.3% | No — off by one frame |

`rd=0 + offset` is the correct HAL-equivalent: predict what to press NOW, with controller feedback from the previous frame. The rd=1 run made the task unnecessarily harder and didn't match inference timing.

---

## Finding 24: STANDING calibration gap was caused by extra features, not architecture

The rd=0+offset model (without hal_minimal) had NONE=99.9% during STANDING. HAL had NONE=94.8%. After systematically ablating features, we found the model was overly sensitive to the action embedding because it had many extra features reinforcing the "STANDING = idle" signal (ECB, speeds, hitlag/hitstun, action_elapsed, port, costume).

The `--hal-minimal-features` flag strips down to HAL's exact feature set:
- **Kept**: pos_x, pos_y, percent, stock, jumps_left, invuln_left, shield_strength (7 numeric) + on_ground, facing, invulnerable (3 flags) = 10 per player
- **Dropped**: 5 speeds, hitlag, hitstun, 8 ECB values, action_elapsed, port, costume, off_stage, moonwalkwarning, global numeric (distance, frame, stage geometry)
- **Result**: 21.3M params, 10 raw tokens (down from 22M/15 tokens)

---

## Finding 25: Systematic ablation identified self_buttons and self_analog as critical

Fed training data through the model and zeroed one feature group at a time:

| Feature zeroed | Non-NONE predictions | Delta from baseline |
|---|---|---|
| Baseline (training data) | 215/1024 | — |
| self_buttons | 39/1024 | **-176 (82% drop)** |
| self_analog | 63/1024 | **-152 (71% drop)** |
| self_c_dir | 215/1024 | 0 |
| self_numeric | 214/1024 | -1 |
| opp_numeric | 214/1024 | -1 |
| numeric (global) | 216/1024 | +1 |
| self_flags | 214/1024 | -1 |
| action_elapsed | 212/1024 | -3 |
| self_action | 0/1024 | **-215 (100% drop)** |
| opp_controller | 53/1024 (from 215) | -162 |
| stage | 216/1024 | +1 |

**self_buttons, self_analog, and self_action are the three critical features.** The model is nearly 100% dependent on the action state and self-controller feedback for deciding whether to act. All other features (numeric, flags, global, opponent) have essentially zero impact.

---

## Finding 26: Inference feedback was confirmed working but insufficient

The `_prev_sent` mechanism correctly tracks and feeds back actual sent values:
- Buttons: confirmed via FB_CHECK debug logs (buttons show 1 when pressed)
- Analog: confirmed (main_x varies between 0.0 and 1.0)
- Values reach the model tensors via `_process_one_row`

The feedback is correct but sparse — the model rarely presses buttons during STANDING, so the feedback rarely contains button presses, creating a stable "idle" equilibrium.

---

## Finding 27: HAL's model also can't predict button presses during STANDING

Examined HAL's predictions at STANDING→press transitions:
```
HAL frame 22: action=14(STANDING) target=NONE  pred=NONE (97.6%)
HAL frame 23: action=14(STANDING) target=X     pred=NONE (97.6%)  ← misses the press!
HAL frame 24: action=24(KNEEBEND) target=X     pred=X (89.2%)    ← correct after action changes
```

HAL also predicts NONE during STANDING and only switches to the correct button after the action changes. The 4.6% average jump probability during STANDING comes from a minority of frames where NONE drops to 80-90%, not from any frame predicting jump at 50%+.

Both models are reactive (predict based on current action state) rather than proactive (predict initiating actions). The difference is HAL has a wider spread in its NONE confidence, creating more sampling opportunities.

---

## Current Training Run

**`mimic-hal-fox-minimal-rd0`** on Machine E (8× RTX 5090):
```
torchrun --nproc_per_node=8 train.py \
  --model hal --hal-mode --self-inputs --encoder flat \
  --seq-len 256 --lr 3e-4 --no-warmup --cosine-min-lr 1e-6 \
  --plain-ce --dropout 0.2 --no-amp --batch-size 64 \
  --stick-clusters hal37 --lean-features --hal-minimal-features \
  --no-compile --reaction-delay 0 --controller-offset \
  --data-dir data/fox_public_shards --seed 42
```

Config: rd=0, controller_offset, hal_minimal (10 per player = 7 numeric + 3 flags), no port/costume/global_num, 3-bin shoulder, 37 stick clusters, 21.3M params.

Early metrics at step 806: bf1=92.0%, mf1=42.7%. About 55 minutes remaining.

**Data alignment verified:**
- Position i: model sees gamestate[i] (current action state) + controller[i-1] (offset)
- Target: controller[i] (what to press now)
- Position 0: controller is all zeros (offset, no previous frame)
- At inference: identical — sees current gamestate + what was sent last frame → predicts now

---

## All Training Runs (same Fox data, same base architecture)

| Run | rd | offset | minimal | Val Loss | bf1 | mf1 | Notes |
|-----|-----|--------|---------|----------|-----|-----|-------|
| HAL (Gu's code) | 0-equiv | built-in | HAL native | 0.744 | ~88% | ~50% | Working in closed-loop |
| mimic rd=1, no offset | 1 | No | No | 0.707 | 87.8% | 54.5% | Pressed some buttons at inference |
| mimic rd=0, no offset | 0 | No | No | 0.078 | 100% | 87% | Trivial (answer in input) |
| mimic rd=0, offset | 0 | Yes | No | 0.598 | 91.6% | 55.6% | NONE=99.9% during STANDING |
| mimic rd=1, offset | 1 | Yes | No | 1.047 | 82.6% | 41.3% | Off-by-one timing mismatch |
| mimic rd=0, offset, minimal | 0 | Yes | Yes | ~0.88* | 92.0%* | 42.7%* | **Current run (in progress)** |
| mimic rd=1, offset, minimal | 1 | Yes | Yes | 1.058 | 82.5% | 41.2% | Killed — wrong timing |

*Early metrics at step 806/31250.

---

## Remaining Unmatched Differences

1. **Position encoding**: RoPE (MIMIC) vs relative/skew (HAL)
2. **Input projection**: per-group MLP then concat (MIMIC) vs single concat→Linear (HAL)
3. **Controller feedback encoding**: raw floats 4+12+1 (MIMIC) vs 54-dim one-hot clusters (HAL)
4. **C-stick**: 5 classes (MIMIC) vs 9 clusters (HAL)

---

## Next Steps

1. Wait for `mimic-hal-fox-minimal-rd0` to finish
2. Check STANDING calibration — compare to HAL's 94.8%
3. Test closed-loop inference
4. If still underperforming, implement one-hot controller feedback encoding (item 3 above)

---

## Finding 28: hal_minimal_features did not improve STANDING calibration

Trained `mimic-hal-fox-minimal-rd0` with rd=0, controller_offset, and hal_minimal_features (7 numeric + 3 flags = 10 per player, no port/costume/global_num, 3-bin shoulder). 21.3M params, 10 raw tokens.

STANDING calibration comparison:

| Model | STANDING mean NONE | STANDING NONE<0.99 | STANDING X (jump) |
|-------|-------------------|-------------------|-------------------|
| HAL (Gu's code) | **94.8%** | **80%** | **4.6%** |
| MIMIC rd=0+offset (22 numeric) | 99.9% | 0.2% | 0.0% |
| MIMIC rd=1+offset (22 numeric) | 95.2% | 59.8% | 3.3% |
| MIMIC rd=0+offset+minimal (10 per player) | **98.6%** | **1.8%** | **0.7%** |

The minimal features model is slightly better than the non-minimal rd=0 model (98.6% vs 99.9% NONE) but still far from HAL's 94.8%. Reducing the feature set alone doesn't fix the calibration gap.

### Inference bug found and fixed

The `--hal-minimal-features` flag caused a CUDA assert at inference: the encoder tried to index `self_numeric[..., [0,1,2,3,4,12,13]]` but at inference with `hal_minimal=True` in `build_feature_groups`, the numeric tensor only had 7 columns (not 22). Fixed by checking tensor width before indexing (commit `c8ba1bd`).

---

## Finding 29: rd=0 vs rd=1 STANDING calibration

The rd=1+offset model (without minimal features) had the BEST STANDING calibration of any MIMIC run: 95.2% NONE, 59.8% of frames with NONE<0.99, 3.3% jump probability. This closely matched HAL (94.8%, 80%, 4.6%).

But rd=1 has a timing mismatch — it predicts frame i+1's button and applies it on frame i. Despite this, the STANDING calibration was better because the harder prediction task (can't cheat via action) forced the model to rely more on context and less on the action embedding.

rd=0 models (both with and without minimal features) have worse STANDING calibration because they can read the answer from the action state, making them more peaked/confident on the majority class.

### The dilemma

- rd=0: matches inference timing perfectly, but model learns to be reactive (reads action) → poor STANDING calibration
- rd=1: doesn't match inference timing (off by one), but model is more proactive → better STANDING calibration

HAL uses rd=0-equivalent and ALSO has good STANDING calibration. The remaining architectural differences must be why HAL avoids the over-confidence that MIMIC rd=0 exhibits.

---

## Remaining Unmatched Differences (updated)

| Difference | Status | Likely Impact |
|-----------|--------|---------------|
| Position encoding (RoPE vs relative/skew) | Unmatched | Unknown — affects temporal attention patterns |
| Input projection (per-group MLP vs concat→Linear) | Unmatched | May affect how features mix before transformer |
| Controller feedback encoding (raw floats vs one-hot clusters) | Unmatched | HAL's discrete encoding may produce cleaner gradients |
| C-stick (5 classes vs 9 clusters) | Unmatched | Low impact |
| Feature set (now minimal) | **Matched** | Was not the cause |
| Shoulder (now 3-bin) | **Matched** | Was not the cause |
| Port/costume (now dropped) | **Matched** | Was not the cause |

---

## All Training Runs (updated)

| Run | rd | offset | minimal | Val Loss | bf1 | mf1 | STANDING NONE | STANDING X |
|-----|-----|--------|---------|----------|-----|-----|--------------|------------|
| HAL (Gu's code) | 0-eq | built-in | native | 0.744 | ~88% | ~50% | **94.8%** | **4.6%** |
| mimic rd=1, no offset | 1 | No | No | 0.707 | 87.8% | 54.5% | — | — |
| mimic rd=0, no offset | 0 | No | No | 0.078 | 100% | 87% | — | — |
| mimic rd=0, offset | 0 | Yes | No | 0.598 | 91.6% | 55.6% | 99.9% | 0.0% |
| mimic rd=1, offset | 1 | Yes | No | 1.047 | 82.6% | 41.3% | **95.2%** | **3.3%** |
| mimic rd=0, offset, minimal | 0 | Yes | Yes | ~0.6* | ~91%* | ~45%* | 98.6% | 0.7% |
| mimic rd=1, offset, minimal | 1 | Yes | Yes | 1.058 | 82.5% | 41.2% | — | — |

*Training still in progress.

---

## Next Steps

1. The three remaining architectural differences (position encoding, input projection, controller encoding) need to be addressed
2. Controller feedback as one-hot clusters is the most concrete change to implement next
3. Alternatively, try a simple temperature scaling approach: train with rd=0 but apply a temperature <1.0 at inference to sharpen the distribution — though this is a workaround, not a fix
