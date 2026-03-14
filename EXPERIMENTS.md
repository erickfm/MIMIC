# FRAME Phase 3: Architecture & Loss Refinement

Systematic experiments to refine depth, width, context length, positional
encoding, and loss design for next-frame Melee controller prediction.

**Baseline:** medium backbone (768d/4L) + hybrid16 encoder ≈ 32M params
**Compute:** 3 machines × 6-8 RTX 4090 GPUs (15 runs in parallel)
**Budget:** ~1 hour per run via `--max-samples 2000000`
**Tracking:** wandb project `FRAME`, group `phase3`

---

## Design principles

Every experiment below is motivated by the same question: **what makes the
model better at imitating human Melee play?**

The model watches a sliding window of game frames and predicts the next
controller state (stick, triggers, c-stick, buttons). It must:

1. **Understand game state** -- positions, velocities, actions, hitbox
   timing, projectiles all interact within a single frame
2. **Recognise temporal patterns** -- approaches, combos, edge-guards,
   tech chases, and recovery sequences unfold over 10-180+ frames
3. **Output precise inputs** -- the difference between a tilt and a smash
   attack is ~0.15 of stick travel; a 1-frame button timing error can
   mean a missed L-cancel or an SD
4. **Run in real time** -- inference must complete in < 16.67 ms (one frame
   at 60 fps)

---

## Baseline config (shared across all axes)

```
python3 train.py \
  --data-dir data/full \
  --max-samples 2000000 \
  --model medium \
  --encoder hybrid16 \
  --lr 8e-4 \
  --batch-size 128 \
  --run-name baseline \
  --wandb-group phase3 \
  --wandb-tags phase3,baseline
```

| Setting | Value |
|---------|-------|
| d_model | 768 |
| Heads | 8 |
| Layers | 4 |
| FF dim | 3072 (4× expansion) |
| Encoder | hybrid16 (16 entity-level tokens + intra-frame attention) |
| d_intra | 256 |
| Seq len | 60 (1 second) |
| Batch size | 128 |
| LR | 8e-4 |
| Pos enc | learned |
| Stick loss | MSE |
| Button loss | BCE |
| Params | ~32.4M |
| Samples | 2M (~1 hr on 4090) |

---

## Axis 1: Depth (number of transformer layers)

**Question:** How much sequential reasoning does Melee imitation need?

Each transformer layer adds one more level of compositional temporal
reasoning. In Melee terms:

- **2 layers**: Can correlate pairs of events ("opponent jumped → I should
  anti-air") but struggles with multi-step chains
- **4 layers**: Can compose short sequences ("opponent whiffed → I dash in
  → grab") -- covers most neutral interactions
- **6 layers**: Multi-step combos and edge-guard reads ("up-throw → read DI
  → follow-up aerial → drift to cover options")
- **8 layers**: Deep sequential planning, but diminishing returns with only
  60 frames of context

**Tradeoff:** Deeper = slower throughput = fewer samples in the 1-hr budget.
A 2L model at 10 step/s sees more data than an 8L model at 4 step/s. The
question is whether the per-sample learning is worth the throughput cost.

**Held constant:** d_model=768, ff=3072, hybrid16, lr=8e-4, seq=60, bs=128

| Run | Layers | ~Params | Tag |
|-----|--------|---------|-----|
| `depth-2L` | 2 | 18.3M | `phase3,depth` |
| `depth-4L` | 4 | 32.4M | `phase3,depth,baseline` |
| `depth-6L` | 6 | 46.6M | `phase3,depth` |
| `depth-8L` | 8 | 60.8M | `phase3,depth` |

---

## Axis 2: Width (d_model)

**Question:** How much per-frame representational capacity does the model need?

Width determines the bottleneck through which each frame's information
flows into the temporal transformer. In Melee, a single frame carries:

- Both players' positions, velocities, ECB geometry (continuous)
- Both players' action states + frame counters (categorical + timing)
- Up to 8 projectiles with positions and types
- Analog inputs, digital buttons, shield health, hitlag, etc.

The hybrid16 encoder compresses all of this into a single `d_model`-dim
vector. A wider bottleneck preserves more information but adds quadratic
cost in attention and linear cost in FFN.

**Held constant:** 4 layers, hybrid16, lr=8e-4, seq=60, bs=128, ff=4×d_model

| Run | d_model | FF | ~Params | Tag |
|-----|---------|-----|---------|-----|
| `width-512` | 512 | 2048 | 16.3M | `phase3,width` |
| `width-768` | 768 | 3072 | 32.4M | `phase3,width,baseline` |
| `width-1024` | 1024 | 4096 | 54.9M | `phase3,width` |

---

## Axis 3: Context length (window size)

**Question:** How much game history does the model need to see?

This is the most Melee-specific axis. Different gameplay situations require
different amounts of temporal context:

| Situation | Frames needed | Time |
|-----------|--------------|------|
| Shield / spot-dodge reaction | 5-15 | 0.08-0.25s |
| Tech chase read | 20-40 | 0.33-0.67s |
| Neutral footsies | 30-60 | 0.5-1.0s |
| Combo strings | 30-90 | 0.5-1.5s |
| Edge-guard full sequence | 60-180 | 1-3s |
| Respawn invincibility | 120 | 2s |
| Ledge stalling / planking | 180-360 | 3-6s |

At 60 frames (1 second), the model sees most neutral interactions but
misses the full arc of edge-guards and knows nothing about respawn
context. Longer windows capture more but require smaller batch sizes
(higher VRAM per sample) and the model must learn to attend over longer
ranges.

**Key tradeoff:** Batch size shrinks with longer sequences, giving noisier
gradients and fewer effective samples. The model also has more positions
to attend over, increasing the signal-to-noise ratio in attention.

**Held constant:** medium (768d/4L/3072ff), hybrid16, lr=8e-4

| Run | Seq len | Secs | Batch size | Tag |
|-----|---------|------|------------|-----|
| `ctx-30` | 30 | 0.5s | 256 | `phase3,context` |
| `ctx-60` | 60 | 1.0s | 128 | `phase3,context,baseline` |
| `ctx-90` | 90 | 1.5s | 96 | `phase3,context` |
| `ctx-120` | 120 | 2.0s | 64 | `phase3,context` |
| `ctx-180` | 180 | 3.0s | 48 | `phase3,context` |

---

## Axis 4: Positional encoding (RoPE vs learned)

**Question:** Does the model benefit from explicit relative-position bias?

Learned positional embeddings encode absolute position in the window
("this is frame 42 of 60"). RoPE encodes relative distance between
frames, so the model naturally learns patterns like "opponent was in
hitstun N frames ago" without having to infer it from absolute positions.

For Melee, relative timing is what matters physically -- the game engine
doesn't care which frame number we're at in our observation window, it
cares about the elapsed time between events. A tech window is always
20 frames after hitlag ends, regardless of where that falls in our window.

RoPE also generalises to unseen sequence lengths, which could matter if
we later want to train at one window size and infer at another.

**Held constant:** medium+hybrid16, lr=8e-4, bs=128, seq=60

| Run | Method | Tag |
|-----|--------|-----|
| `pos-learned` | Learned | `phase3,posenc,baseline` |
| `pos-rope` | RoPE | `phase3,posenc` |

---

## Axis 5: Loss functions

**Question:** Are the current loss functions well-matched to Melee's
controller output distributions?

### Main stick — the multimodality problem

The main analog stick has a highly multimodal distribution. In neutral,
the correct input might be "dash right" OR "dash left" OR "stand still"
depending on the player's read. MSE regression averages between modes,
predicting lukewarm values (e.g., 0.55) that don't correspond to any
real Melee action.

Melee also has hard zone boundaries that MSE doesn't respect:
- **Deadzone** (< 0.28 from center): no input registered
- **Tilt zone** (0.3-0.65): walk, tilt attack
- **Smash zone** (> 0.8): run, smash attack

An MSE error of 0.10 that crosses the tilt/smash boundary changes the
executed move entirely. An MSE error of 0.10 within the smash zone
is irrelevant.

**Discretising** the stick into bins lets the model output a probability
distribution. It can express "70% dash right, 20% stand, 10% dash left"
rather than being forced into a point estimate.

**Huber** loss reduces the influence of outlier frames (death animations,
respawn, game-start countdown) where stick values are meaningless noise.
The model shouldn't be penalised as hard for getting these wrong.

### Buttons — the class imbalance problem

Button presses are extremely rare events. Most frames have all 12 buttons
unpressed. Standard BCE gives equal weight to every (button, frame) pair,
so the vast majority of gradient signal says "keep predicting 0."

But button timing is critical in Melee:
- A-button for L-cancel must land within a 7-frame window
- B-button for tech must be precise to ~20 frames
- Z-button for aerial interrupt has specific timing
- Wrong button at wrong time can SD

**Focal BCE** downweights easy-to-classify examples (the 95%+ of unpressed
buttons the model already gets right) and concentrates gradient on the
uncertain cases -- exactly the rare presses that matter for gameplay.

### L/R triggers — bimodal with dead gradient

Trigger values are almost always 0.0 (released) or near 1.0 (full press).
MSE on these produces tiny loss and tiny gradients for the dominant "0"
class. The model easily learns to output ~0 and rarely gets signal about
when to press. Huber loss provides more uniform gradients near zero.

**Held constant:** medium+hybrid16, lr=8e-4, bs=128, seq=60, learned pos-enc

| Run | Stick loss | Button loss | Note | Tag |
|-----|-----------|-------------|------|-----|
| `loss-baseline` | MSE | BCE | Current defaults | `phase3,loss,baseline` |
| `loss-huber` | Huber | BCE | Robust to outlier frames | `phase3,loss` |
| `loss-discrete` | Discrete 32×32 | BCE | Multimodal stick distributions | `phase3,loss` |
| `loss-focal-btn` | MSE | Focal BCE | Focus on rare button presses | `phase3,loss` |

---

## Run allocation (15 GPUs)

The baseline appears across multiple axes but is a single run.

| GPU | Run | Machine |
|-----|-----|---------|
| 0 | `baseline` (= depth-4L = width-768 = ctx-60 = pos-learned = loss-baseline) | A |
| 1 | `depth-2L` | A |
| 2 | `depth-6L` | A |
| 3 | `depth-8L` | A |
| 4 | `width-512` | A |
| 5 | `width-1024` | A |
| 6 | `ctx-30` | B |
| 7 | `ctx-90` | B |
| 8 | `ctx-120` | B |
| 9 | `ctx-180` | B |
| 10 | `pos-rope` | B |
| 11 | `loss-huber` | B |
| 12 | `loss-discrete` | C |
| 13 | `loss-focal-btn` | C |
| 14 | *(spare)* | C |

---

## Evaluation criteria

| Metric | Role | Why |
|--------|------|-----|
| `val/total` | **Primary** | Overall prediction quality |
| `val/cdir_acc` | High | C-stick accuracy = game understanding |
| `val/btn_acc` | High | Button accuracy = safety (wrong button = SD) |
| `val/loss_main` | Medium | Stick precision = movement quality |
| `perf/step_per_sec` | Tiebreaker | Faster = more data in same budget |

For final model selection, also consider:
- **Inference latency**: must be < 16.67 ms
- **Qualitative play**: test against CPU using `inference.py`

---

## Decision gates

After Phase 3 completes:

1. **Depth vs width:** Plot val/total vs params for the depth and width
   sweeps. If deeper models beat wider ones at similar param count,
   prefer depth (better sequential reasoning). If wider wins, the
   bottleneck is per-frame representation.

2. **Context length:** Plot val/total vs seq_len. Look for diminishing
   returns. Also check val/cdir_acc specifically -- c-stick decisions
   (smash attacks, aerials) depend on positional context that longer
   windows capture. If ctx-120 is noticeably better than ctx-60 for
   cdir but not for buttons, it confirms that longer context helps with
   spatial decision-making.

3. **RoPE vs learned:** If RoPE wins at seq=60, re-run the best context
   length with RoPE (it may help more at longer ranges).

4. **Losses:** If Huber or discrete beats MSE, the data has significant
   outlier/multimodal structure that the current loss ignores. If focal
   button loss helps btn_acc without hurting other metrics, adopt it.

5. **Combine winners** in a Phase 4 champion run trained longer.
