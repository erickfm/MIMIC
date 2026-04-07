# Research Notes — 2026-03-16

## Context

Continued from [2026-03-15](research-notes-2026-03-15.md). Previous session achieved perfect closed-loop wavedash at 60fps. Today's goal: find hyperparameter configurations that overfit the wavedash dataset faster and more reliably, and update training defaults based on findings.

---

## Phase 1: Speed Sweep Infrastructure

### Problem

The canonical wavedash training (`wavedash-canonical`) converged to val_f1=98.5% in ~20,000 steps (2,978s wall time on Machine A) with the `medium` preset (32.4M params). Can we find faster or more reliable configurations?

### Infrastructure Added

- **`train.py`**: Added `--target-val-f1`, `--max-wall-time`, `--val-frac`, and `--warmup-steps` CLI args. Training exits early when val F1 hits the target or wall time is exceeded.
- **`sweep/launch.sh`**: SSH-based launcher for remote experiments with per-GPU assignment, common training args, and per-run overrides.
- **`sweep/status.sh`**, **`sweep/kill.sh`**, **`sweep/results.sh`**: Operational helpers for monitoring, termination, and leaderboard generation.
- **`sweep/experiments.jsonl`**, **`sweep/queue.txt`**: Structured experiment tracking and backfill queue.

### Hardware

22x RTX 4090 GPUs across 3 machines (Machine A: 6 GPUs, B: 7, C: 8). Machine A runs ~6.9 step/s, B ~2.5, C ~1.8 for the shallow model at batch=128.

---

## Phase 2: Architecture + LR Sweep (62 experiments)

### Key Finding: Learning Rate Sensitivity

Only lr=5e-5 reliably converges to 98.5% F1. Higher LRs (8e-4, 1e-4, 7.5e-5) cause most models to plateau at 66-90% F1 (base-rate collapse or early plateau). This held across all model sizes tested.

> **Caveat (added 2026-03-23):** This sweep was on the wavedash task (single-pattern overfit) with single GPU, batch=256, no grad clipping. On full Melee with grad clipping, lr=3e-4 (HAL's default) works at ctx=60 with self-inputs. The lr=5e-5 conclusion may be specific to the no-clip, focal-loss regime.

### Key Finding: RoPE > Learned Positional Encoding

RoPE consistently saved ~1,500-2,000 optimization steps vs learned positional encoding. RoPE encodes relative position directly into attention via rotation matrices — for sequential frame prediction where "2 frames ago" always means the same thing regardless of absolute position, this is a free structural prior that learned encoding must spend steps rediscovering.

### Key Finding: Depth vs Width

- `tiny` (1.3M), `small` (8.2M): Cannot reach 98.5% — capacity-limited, plateau at 90-97%.
- `medium` (32.4M, 4 layers): Reaches 98.5% but only at lr=5e-5 with ~18,000 steps.
- `shallow` (d_model=1024, 2 layers): Wide but shallow. Fastest architecture for wavedash specifically because the task requires pattern matching (wide) not compositional reasoning (deep).

### Best Architecture Result

`shallow + RoPE + lr=5e-5`: Hit 98.51% in 2,348.5s (21.1% faster than baseline). However, this result had favorable seed variance — a replication run only reached 97.4%.

---

## Phase 3: Batch Size + Warmup Sweep (21 experiments)

Base config: `shallow + RoPE + lr=5e-5`. Swept batch sizes {32, 64, 128, 192, 256, 512} and warmup steps {0, 50, 100, 200, 500, 750, 1000, 1500, 2000}.

### Warmup Findings

| Warmup | Outcome | Notes |
|--------|---------|-------|
| 0 steps | **Catastrophic** (88% peak) | Model never recovers from bad initial trajectory |
| 50 steps | Borderline (98.2%) | Barely missed target |
| 100 steps | Hit (98.52%) | Sufficient but marginal |
| 200 steps (old default) | **Unreliable** (1/3 hit rate) | Sometimes works, sometimes plateaus at 97% |
| 500 steps | Barely missed (98.49%) | Just under threshold |
| 1000 steps (5%) | **Reliable** (2/2 hits, avg 2,598s) | Sweet spot |
| 1500 steps | Missed (97.4%) | Too much time at low LR |
| 2000 steps | Hit (98.57%) | Works but wastes early steps |

The 5% warmup gives Adam time to build accurate moment estimates across all interacting loss terms (focal BCE for buttons, focal CE for stick clusters and shoulders, MSE/Huber for regression) before committing to the full learning rate.

### Batch Size Findings

| Batch | Outcome | Notes |
|-------|---------|-------|
| 32 | Failed (91% peak) | Too noisy per-step, insufficient data per gradient |
| 64 | Failed (96% peak in 20k steps) | Converges slowly per-step, runs out of step budget |
| 128 | Unreliable (varies with warmup) | Sensitive to other hyperparameters |
| 256 | **Always hit** (3/3, 98.6-98.7% peak) | Robust regardless of warmup choice |
| 512 | Slow (94% at step 8.7k) | Per-step cost too high |

**batch=256 never failed** regardless of warmup setting (200, 500, or 1000). Each gradient step sees 2x more data, reducing noise by ~41% (1/sqrt(2)). This makes optimization robust to other hyperparameter choices — the optimizer doesn't get unlucky on any given step.

### Combined Results

The most reliable configuration overall: `batch=256 + warmup=500`. Hit 98.58% at step 14,200 (71% of budget) with 5,800 steps of headroom. Every batch=256 variant hit the target, while batch=128 variants were sensitive to warmup length.

The fastest reliable configuration: `batch=128 + warmup=1000`. Hit 98.51-98.54% in 2,571-2,626s wall time. Faster than batch=256 variants (~3,130s) due to lower per-step cost, but only reliable with warmup=1000 specifically.

---

## Updated Training Defaults

Based on sweep findings, updated four defaults that generalize across datasets:

| Setting | Old Default | New Default | Rationale |
|---------|-------------|-------------|-----------|
| `LEARNING_RATE` | 8e-4 | **5e-5** | Only LR that reliably converges for this loss landscape |
| `BATCH_SIZE` | 128 | **256** | Robust to other hyperparameter choices, never failed |
| `WARMUP_FRAC` | 0.01 (1%) | **0.05 (5%)** | Gives Adam time to calibrate moment estimates for multi-task loss |
| `ModelConfig.pos_enc` | "learned" | **"rope"** | Free structural prior for relative position in sequential data |

Architecture (`--model`) is intentionally NOT changed — `shallow` won on wavedash but complex behaviors need depth. Choose model size based on task complexity.

---

## Files Changed

| File | Change |
|------|--------|
| `train.py` | New defaults: `BATCH_SIZE=256`, `LEARNING_RATE=5e-5`, `WARMUP_FRAC=0.05`. Added `--warmup-steps` CLI arg for explicit override. Added `--target-val-f1`, `--max-wall-time`, `--val-frac` for sweep control. |
| `mimic/model.py` | `ModelConfig.pos_enc` default changed from `"learned"` to `"rope"` |
| `sweep/launch.sh` | Updated COMMON args to use shallow+RoPE base config for sweep2 |

---

## Hardcoded: Cluster Stick Loss + Autoregressive Heads

**Decision**: `stick_loss="clusters"` and `autoregressive_heads=True` are now the only supported configuration. All MSE/huber stick regression code and non-autoregressive head paths have been removed from the codebase.

**Why MSE/huber stick loss was removed**:

MSE/huber regression on raw stick (x, y) values causes base-rate collapse: the model learns to predict neutral (0.5, 0.5) on every frame because that minimizes squared error against the highly peaked distribution of stick positions. The loss gradient provides no useful signal for distinguishing between the ~63 distinct stick positions that actually matter for gameplay. Discrete cluster classification with focal loss solves this by treating each meaningful stick region as a separate class, giving the model proper gradients on rare but critical inputs (smash attacks, wavedash angles, etc.).

**Why non-autoregressive heads were removed**:

Without autoregressive chaining, each output head (L shoulder, R shoulder, c-stick direction, main stick, buttons) predicts independently. This breaks the strong conditional dependencies between outputs -- e.g., a wavedash requires L-shoulder press *and* a specific diagonal stick angle *simultaneously*. With independent heads, the model must independently arrive at the correct joint distribution, which it fails to learn reliably. The autoregressive chain (L → R → cdir → main → buttons) gives each subsequent head access to the previous predictions via `.detach()` conditioning, breaking the joint prediction into tractable conditional steps.

**Code removed**:

- `mimic/model.py`: MSE/huber branch in `PredictionHeads.__init__`, non-AR branch in `__init__` and `forward`
- `train.py`: `_stick_regression_loss()` function, `--stick-loss` and `--autoregressive-heads` CLI args, MSE/huber branch in `compute_loss()`, `_mse`/`_huber` loss modules
- `inference.py`: All `if cfg.stick_loss == "clusters"` / `else` guards in cluster loading, logging, feedback, and controller output

**Updated defaults**:

- `ModelConfig.stick_loss` = `"clusters"` (was `"mse"`)
- `ModelConfig.autoregressive_heads` = `True` (was `False`)
- `ModelConfig.dropout` = `0.1` (was `0.0`, for generalization on full dataset)
- `--clusters-path` defaults to `data/full/stick_clusters.json`

---

## Status

Training defaults are now sweep-proven. The configuration `batch=256, lr=5e-5, warmup=5%, RoPE, clusters, AR-heads, dropout=0.1` is the recommended starting point for any new behavior dataset. Architecture depth should scale with task complexity.

Next steps: train on full Melee dataset with updated defaults, verify generalization beyond single-behavior overfitting.
