# Research Notes — 2026-03-21

## Context

Continued from [2026-03-19](research-notes-2026-03-19.md). Active runs at start of session:
- **Machine C**: `full-nsi-ctx256-seed42` — medium (32M), ctx=256, no self-inputs, lr=5e-5
- **Machine E**: `full-nsi-ctx256-seed43` — medium (32M), ctx=256, no self-inputs, lr=5e-5
- **Machine D**: `full-nsi-ctx256-huge-seed42` — huge (621M), ctx=256, no self-inputs, lr=5e-5
- **Machine F**: `full-si-ctx60-seed42` — medium (32M), ctx=60, self-inputs, lr=5e-5 (previously diverged)

F's original self-inputs run had diverged: gradient norms exploded from ~5 to 180,000 between steps 400k-878k, crashing btn_f1 from 88% to 35%.

---

## Finding 1: Self-inputs instability is not an LR problem

Ran three fresh self-inputs runs on Machine F (medium model, ctx=60, seed=42), varying only the peak learning rate:

| Run | Peak LR | Blowup step | gnorm trajectory |
|-----|---------|-------------|------------------|
| `full-si-ctx60-seed42` (original) | 5e-5 | ~400k | 5 → 180,000 |
| `full-si-ctx60-seed42-lr4e5` | 4e-5 | ~410k | 3 → 22 → 40 (killed) |
| `full-si-ctx60-seed42-lr3e5` | 3e-5 | ~490k | 4 → 30 → 138 → 754 → 7,404 |

All three diverged in the same training region (400-600k steps). Lower LR only delayed the explosion slightly. The instability is structural — self-inputs create a sharp loss landscape that the medium model can't handle.

Prior to divergence, all runs reached ~88-92% train btn_f1, confirming self-inputs enable much faster learning vs no-self-inputs (~55% at the same point).

## Finding 2: Medium model without self-inputs plateaus at ~40% val btn_f1

Machines C and E ran to ~1.1M steps (29% of training) with stable gnorms (~1.0) but val btn_f1 plateaued around 38-42%:

| Machine | Steps reached | Val btn_f1 (last 5) | Train btn_f1 |
|---------|--------------|---------------------|--------------|
| C (seed=42) | 1.15M (29.5%) | 42→41→41→39→40% | ~50% |
| E (seed=43) | 1.13M (29.0%) | 42→39→38→41→41% | ~55% |

Train > val gap suggests overfitting. The medium model (32M params) may be capacity-limited for ctx=256 without self-inputs. **Killed both runs.**

## Finding 3: Huge model (621M) is much more capable

Machine D's huge model at same config (ctx=256, no self-inputs, lr=5e-5) reached 77% train btn_f1 at only 11.5% through training with perfectly stable gnorms (0.48-0.60). Only 2 val checkpoints so far (~39% val btn_f1), too early to assess plateau. This run continues.

## Finding 4: Autoregressive prediction feedback already exists

Investigated whether to implement autoregressive feedback (feed model predictions back as self-inputs at inference, like HAL's approach). Found it's already implemented in `inference.py:644-662` — when `no_self_inputs=False` and `--no-pred-feedback` is not set, the model feeds its previous predictions back as self-inputs. No code changes needed.

This means the training/inference gap for self-inputs runs is standard exposure bias (teacher forcing vs autoregressive generation), same as any LLM.

---

## Action: Curriculum masking on Machine F

Since self-inputs diverge at all learning rates but enable much faster early learning, implemented scheduled self-input dropout (curriculum masking):

### Implementation (commit `7f33676`)

**`mimic/frame_encoder.py`:** Added `si_drop_prob` attribute to `_BaseFrameEncoder`. In `_build_raw_tokens()`, when `si_drop_prob > 0`, applies per-sample Bernoulli mask that zeros `self_analog`, `self_buttons`, `self_c_dir` (+ nana variants). Stochastic during training, deterministic zeros during eval.

**`train.py`:** Added `--si-drop-start`, `--si-drop-end`, `--si-drop-max` CLI args. Linear ramp schedule from 0 to `si_drop_max` over the configured training fraction. Dual validation: normal val at `si_drop_prob=0` (full self-inputs), plus `val_nsi/*` at `si_drop_prob=1.0` (no self-inputs). Logs `curriculum/si_drop_prob` to wandb.

Also added `--no-load-optim` flag (commit `6790199`) for resuming with fresh optimizer/scheduler state.

### Launch

```bash
BG=1 bash parallel.sh F -- \
  --model medium --seq-len 60 --batch-size 64 --max-samples 250000000 \
  --no-compile --self-inputs --seed 42 \
  --lr 3e-5 \
  --si-drop-start 0.05 --si-drop-end 0.5 \
  --run-name full-si-curriculum-ctx60-seed42
```

Schedule: `si_drop` ramps 0→1.0 over steps 195k (5%) to 1.95M (50%). Steps 0-195k train with full self-inputs (fast learning). By the 400k danger zone, `si_drop` ≈ 12%. Steps 1.95M+ train with fully masked self-inputs.

### Hypothesis

The instability is caused by the model over-relying on self-inputs, creating sharp loss landscape features that amplify gradient norms. Gradually masking self-inputs forces the model to diversify its feature reliance, potentially smoothing the landscape enough to prevent divergence.

### Result: Curriculum slowed but did not prevent divergence

The curriculum run on F diverged at ~586k steps (si_drop=0.22), later than the non-curriculum runs (~400-490k) but with the same exponential gnorm pattern:

| Step | si_drop | gnorm |
|------|---------|-------|
| 469k | 0.16 | 6.11 |
| 586k | 0.22 | 65.68 |
| 762k | 0.32 | 785 |

Killed at gnorm=785. The curriculum delayed divergence by ~100-180k steps compared to plain self-inputs, but the fundamental instability remains for the medium model.

---

## Finding 5: Medium no-SI plateaus at ~40% val btn_f1 regardless of character or context

Ran Fox (ctx=60) and Falco (ctx=60) single-character runs. Both hit the same ~36-43% val btn_f1 ceiling seen on the all-character ctx=256 runs (C/E). Fox ctx=180 also reached 43% val after only 2 val checkpoints.

The ~40% val btn_f1 ceiling appears to be a medium model limitation, not a data or context issue.

---

## Pivot: Character-specific runs with curriculum masking

Launched character-specific runs to test whether:
1. Single-character data (smoother loss landscape?) helps curriculum masking survive
2. Longer context (180 vs 60) improves learning rate per step

Data sizes:
- FOX: 443M frames (48k games), 331 shards
- FALCO: 292M frames (32k games), 240 shards

---

## HAL Reference Analysis (2026-03-23)

Deep-dived into Eric Gu's HAL architecture and training. Key differences from MIMIC:

| | **HAL** | **MIMIC (medium)** |
|---|---|---|
| Params | ~20M | ~32M |
| Sequence layers | 6 × 512-d | 4 × 768-d |
| Frame encoding | Flat concat → linear | Intra-frame transformer (2 layers) |
| Pos encoding | Shaw relative | RoPE |
| Buttons | Single-label CE (6 classes) | Multi-hot focal BCE (12 outputs) |
| Sticks | 36 hand-designed clusters, plain CE | 30 k-means clusters, focal CE |
| Dropout | 0.2 | 0.1 |
| LR | 3e-4 | 5e-5 (was), 3e-4 (new) |
| Grad clip | 1.0 | none (was), 1.0 (new) |
| Context | 256 | 60 / 180 |

Critical insight: **HAL uses gradient clipping at 1.0** — we weren't clipping at all, which directly caused our self-inputs gradient explosions.

---

## Finding 6: Gradient clipping solves self-inputs instability at ctx=60

Launched self-inputs runs with `--grad-clip-norm 1.0 --lr 3e-4` (matching HAL):

| Run | ctx | Step | gnorm | btn_f1 | Status |
|-----|-----|------|-------|--------|--------|
| `falco-med-ctx60-si-lr3e4-clip1-s42` (C) | 60 | 898k | 1.23 | 86.3% | **stable** — past all previous death zones |
| `falco-med-ctx180-si-lr3e4-clip1-s43` (E) | 180 | 410k | 126,100 | 43.0% | **diverging** despite clipping |

ctx=60 + clip + lr=3e-4 is fully stable through 898k steps (previous runs exploded at 400-500k). First successful self-inputs run past the danger zone.

ctx=180 + clip + lr=3e-4 explodes — gnorm reaches 126k. The clipping prevents NaN but the model degrades. The same lr=3e-4 that HAL uses at ctx=256 doesn't work for us at ctx=180, suggesting architectural differences (intra-frame attention, focal loss, multi-hot buttons) amplify instabilities at longer sequences.

---

## Note on historical sweep validity (2026-03-23)

The hyperparameter sweep results from 2026-03-16 and 2026-03-17 were obtained under a specific regime:
- Single GPU, batch=256, lr=5e-5, no grad clipping, focal loss, no self-inputs, 50M samples

Current training regime differs significantly:
- DDP 8 GPUs, eff batch=512, lr=3e-4, grad clipping=1.0, self-inputs, 250M samples, character-specific data

Findings like "dropout=0.05 is best" and "lr=5e-5 is the only stable lr" may not transfer. These results should be re-validated under current conditions before being relied upon.

---

## Active runs (2026-03-23)

| Machine | Run | Model | Config | Status |
|---------|-----|-------|--------|--------|
| **C** | `falco-med-ctx60-si-lr3e4-clip1-s42` | medium (32M) | Falco, ctx=60, SI, lr=3e-4, clip=1.0 | running (23%, 86.3% btn_f1, gnorm=1.23) |
| **D** | `all-huge-ctx256-nsi-lr5e5-s42` | huge (621M) | All chars, ctx=256, no SI, lr=5e-5 | running (26%, 80.2% btn_f1, gnorm=0.40) |
| **E** | `falco-med-ctx180-si-lr3e4-clip1-s43` | medium (32M) | Falco, ctx=180, SI, lr=3e-4, clip=1.0 | **diverging** (gnorm=126k) |
| **F** | `fox-med-ctx180-nsi-lr5e5-s42` | medium (32M) | Fox, ctx=180, no SI, lr=5e-5 | running (18.5%, 49.2% btn_f1, gnorm=1.11) |

---

## Run Naming Convention

Format: `{char}-{model}-{ctx}-{si_mode}-{lr}-s{seed}`

| Field | Values | Description |
|-------|--------|-------------|
| `char` | `fox`, `falco`, `marth`, `all`, ... | Character dataset |
| `model` | `med`, `huge`, `giant` | Model preset |
| `ctx` | `ctx60`, `ctx180`, `ctx256` | Sequence length |
| `si_mode` | `nsi` (no self-inputs), `si` (self-inputs), `sicur` (SI + curriculum) | Self-input config |
| `lr` | `lr5e5`, `lr3e5`, `lr2e5` | Peak learning rate |
| `s{seed}` | `s42`, `s43`, ... | Random seed |

Examples:
- `fox-med-ctx60-nsi-lr5e5-s42` — Fox only, medium model, ctx=60, no self-inputs
- `all-huge-ctx256-nsi-lr5e5-s42` — All characters, huge model, ctx=256, no self-inputs
- `falco-med-ctx180-sicur-lr5e5-s43` — Falco only, medium, ctx=180, self-inputs with curriculum
