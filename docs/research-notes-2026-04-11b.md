# Research Notes — 2026-04-11b: Replay Analysis, Bistable Inference, Relpos Retrain

## Overview

Deep investigation into why the 7-class model "doesn't work" in inference.
The documented explanation (NONE-bias controller feedback loop) was **wrong**.
The model is actually **bistable** — it plays well in some games and gets stuck
in others, depending on stochastic sampling in the first few frames.

Currently retraining with relpos (Shaw attention) instead of learned position
encoding to test whether that resolves the stuck-mode problem.

---

## 1. The Documented Explanation Was Wrong

The 04-11 research notes diagnosed the inference problem as a "NONE-bias
feedback loop" where the model's NONE predictions feed back as controller
input, reinforcing NONE predictions indefinitely. This was supported by a
"controller dependency" experiment showing 95.5% → 31.8% accuracy when
the controller input was zeroed out.

### Problems with that analysis

**The zeroing test was invalid.** Zeroing the 56-dim controller one-hot
(all dims = 0) creates an out-of-distribution input the model never saw in
training. A properly encoded NONE controller (valid one-hot with neutral
stick cluster, NONE button class, shoulder off) only drops accuracy from
91.4% to 88.3% — the model handles it fine.

**Autoregressive simulation works.** When we replace the controller input
with the model's own predictions (one frame feeding the next) but keep real
game states from training data, accuracy drops only 3% (91.4% → 88.3%) and
the prediction distribution is essentially unchanged. The controller feedback
loop is not the issue.

---

## 2. The Real Finding: Bistable Inference

Analyzed all 14 saved replay files from the 04-10/04-11 inference sessions
using py-slippi. The model exhibits two distinct modes:

### Active mode (model is fighting)

| Replay | Duration | Buttons | Idle% | Top Action |
|--------|----------|---------|-------|------------|
| Game_20260410T225527 | 96s | 19.7% | 2.9% | ESCAPE (15%) |
| Game_20260410T230933 | 291s | 29.4% | 3.8% | ATTACK_100_LOOP (14%) |
| Game_20260411T001935 | 482s | **40.2%** | 2.6% | ATTACK_100_LOOP (16%) |
| Game_20260411T003010 | 369s | **41.8%** | 7.9% | CATCH (13%) |

In active mode, Fox rapid-jabs, grabs, shields, dodges, down-tilts, and
does forward aerials. 20-40% button press rate, comparable to human training
data (~28% non-NONE). **The model CAN play.**

### Stuck mode (model is idle)

| Replay | Duration | Buttons | Idle% | Top Action |
|--------|----------|---------|-------|------------|
| Game_20260411T004604 | 22s | 0.0% | 92.3% | WAIT (92%) |
| Game_20260411T014416 | 66s | 5.5% | 64.4% | SQUAT_WAIT (39%) |
| Game_20260411T033046 | 238s | 1.0% | 84.2% | WAIT (69%) |

In stuck mode, Fox stands (WAIT) or crouches (SQUAT_WAIT with stick full
down at (0, -1)). The model IS producing non-trivial output (crouching
requires stick down), but it can't break out of these stable attractor states.

### Why bistable?

The model uses `temperature=1.0` with multinomial sampling. Whether it
bootstraps into active mode depends on whether the initial random samples
produce a non-NONE button press. In training data:
- 98.9% of STANDING frames have NONE target
- 99.8% of DASHING frames have NONE target
- Buttons only fire on transition frames (1-3 frames per action initiation)

If early samples happen to produce a button press, the game state diverges
from idle and the model enters active mode. If not, it gets trapped in a
WAIT/SQUAT_WAIT attractor.

### py-slippi parsing note

The `buttons.physical` attribute in py-slippi returns IntFlag values, not
booleans. `phys.A` returns the constant 256 (the flag's bit value) regardless
of whether A is actually pressed. The correct check is
`bool(phys & Buttons.Physical.A)` or checking the raw `int(phys)` bitmask.
Initial analysis incorrectly showed "all buttons pressed 100%" due to this.

---

## 3. Action-State → Button Correlation (rd=0)

With reaction_delay=0, the game state at frame t already reflects button[t]:

| Action State | NONE% | Notes |
|-------------|-------|-------|
| STANDING (14) | 98.9% | Only 1.1% have buttons |
| DASHING (20) | 99.8% | Essentially never has buttons |
| SQUAT_WAIT | ~99% | Holding down, no buttons |

At JUMP onset (first frame of jump press), 56.4% of frames show KNEE_BEND
(the result of pressing jump). The game state encodes the answer.

With rd=1, STANDING → button transitions are 4.5% (vs 1.1% for rd=0), giving
the model 4x more action-initiation signal from idle states.

---

## 4. Why Previous Model Worked

The successful "hal-filtered" model (MIMIC 3-0 vs HAL) used:
- `--model hal` (**relpos** position encoding)
- 5-class buttons (early-release)
- Data on a different machine with HAL's checkpoint norm stats

All models on the current machine used `--model hal-learned` (learned
position encoding). Relpos learns content-dependent distance patterns via
Shaw attention (Q·E[i-j] with 131K learned position params), which may
handle temporal transitions (idle → active) better than learned absolute
position embeddings.

---

## 5. Relpos Retrain (In Progress)

Training `hal-7class-relpos` — same 7-class config but with relpos:

```bash
python3 train.py \
  --model hal --encoder hal_flat \
  --hal-mode --hal-minimal-features --hal-controller-encoding \
  --stick-clusters hal37 --plain-ce \
  --lr 3e-4 --batch-size 512 \
  --max-samples 16777216 \
  --data-dir data/fox_hal_full \
  --controller-offset --self-inputs \
  --reaction-delay 0 \
  --run-name hal-7class-relpos \
  --no-warmup --cosine-min-lr 1e-6
```

- **Data:** fox_hal_full, 17,319 games, 7-class encoding
- **GPU:** RTX 5090, single GPU, batch_size=512
- **Steps:** 32,768 (16.7M samples)
- **wandb:** https://wandb.ai/erickfm/MIMIC/runs/vv745v68

### Early metrics (step ~9500 / 32768)

| Metric | hal-7class-rd0 (learned) | hal-7class-relpos (current) |
|--------|--------------------------|----------------------------|
| Val btn F1 | 90.2% | **91.1%** (and still improving) |
| Val main F1 | 55.4% | 57.1% |
| Speed | ~10 step/s | ~4.0 step/s |

Relpos is 2.5x slower than learned because the Shaw attention forces FP32
on Q@K^T + Srel (can't use Flash Attention). But metrics are already better.

---

## 6. Compile Mode Fix

`torch.compile(model, mode="reduce-overhead")` was added in commit ec3cc7b
(GPU optimization) but regressed to `torch.compile(model)` (default mode) in
commit ed87d7c (7-class refactor). Restored in this commit.

---

## 7. Potential Quick Fixes for Stuck Mode

Even without retraining:
1. **Higher temperature** (1.5-2.0) for the first ~60 frames to increase
   exploration probability
2. **Forced warmup:** press random buttons for 1-2 seconds to bootstrap
   into active mode
3. **Self-input dropout** (`--si-drop-start 0.0 --si-drop-end 0.5
   --si-drop-max 0.5`) — forces model to initiate actions without controller
   history during training
