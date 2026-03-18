# Research Notes — 2026-03-17

## Context

Continued from [2026-03-16](research-notes-2026-03-16.md). Previous session launched a 21-run full-dataset training sweep on 21 GPUs. Today's goals: analyze final sweep results, run diagnostic evaluation on top checkpoints, test live inference, and identify the next architectural change to improve gameplay quality.

---

## Phase 1: Full-Sweep Final Results

### Setup

21 runs across 3 machines (21x RTX 4090), split into two groups:

- **Group 1 (10 runs)**: Data scaling — 10%, 25%, 50%, 75%, 100% of the 1.81B-frame corpus, two seeds each.
- **Group 2 (11 runs)**: Hyperparameter tweaks — dropout, label smoothing, weight decay, sequence length, model size. All on 50% data, 50M samples.

Baseline config: medium model (32.4M params), RoPE, batch=256, lr=5e-5, warmup=5%, dropout=0.1, label_smoothing=0.1, seq_len=60.

### Key Results

| Finding | Detail |
|---------|--------|
| **Best run** | g2-drop05 (dropout=0.05) at **90.0% val btn_f1** — the only run to break 90% |
| **Data scaling is flat** | 10%-100% data all land in 87-90% btn_f1, within seed variance (~1.7%) |
| **Dropout sweet spot** | 0.05 (90.0%) > 0.1 baseline (88.9%) > 0.15 (89.3%) > 0.2 (85.3%) > 0.0 (84.5%) |
| **Larger model doesn't help** | g2-base (51.8M params) scored 84.9% — worse than medium |
| **Label smoothing matters late** | g2-ls00 (no smoothing) led mid-training at 89.6%, collapsed to 87.7% by finish |
| **Lower weight decay helps** | g2-wd1e3 (1e-3 vs 1e-2) at 89.5%, the #2 tweak |
| **Shorter sequences hurt** | seq_len=30 scored 87.8%, seq_len=120 scored 86.3% (still training) |

### Rankings Reshuffled Dramatically

Mid-training and final rankings diverged significantly:

- g2-ls00 (no label smoothing): #1 at midpoint (89.6%) → dropped to 87.7% (overfit)
- g2-drop20 (heavy dropout): #2 at midpoint (88.9%) → dropped to 85.3% (underfit)
- g2-drop05 (light dropout): near bottom at midpoint (85.3%) → surged to 90.0% (slow and steady)

Lesson: early metrics are misleading. Low dropout models start slow but generalize best.

### Data Scaling Interpretation

The flatness of Group 1 results (all within ~2% of each other) was initially surprising. But the 50M-sample budget means even the 100% data runs only see each frame ~0.5 times on average. The model is nowhere near overfitting to the data — it's underfitting at all data scales. More data would likely help with significantly longer training.

---

## Phase 2: Diagnostic Evaluation

### Motivation

The best model (g2-drop05, 90% btn_f1) played "ok but weird" in live Melee inference. High offline metrics clearly did not translate to good gameplay. We needed to understand WHY.

### New Tooling: eval.py

Created a standalone diagnostic evaluation script with metrics beyond standard F1:

- **Per-button F1/precision/recall**: individual breakdown for all 12 buttons
- **Stick diagnostics**: neutral rate (predicted vs ground truth), logit entropy, top-1 confidence, non-neutral top-1 accuracy
- **Transition accuracy**: accuracy on frames where ground truth changes vs. frames where it stays the same

### Results (5 top checkpoints)

| Model | btn_f1 | main_f1 | btn_trans_acc | btn_steady_acc | stick_trans_acc | stick_steady_acc |
|-------|--------|---------|---------------|----------------|-----------------|------------------|
| g1-d100-r2 | 89.2% | 35.3% | **7.4%** | 99.6% | **28.5%** | 96.9% |
| g2-base | 88.6% | 35.4% | **8.2%** | 99.5% | **27.4%** | 96.7% |
| g2-drop05 | 89.2% | 38.3% | **7.1%** | 99.3% | **27.8%** | 97.2% |
| g1-d100-r1 | 88.4% | 34.2% | **9.6%** | 99.1% | **25.0%** | 96.2% |
| g1-d50-r1 | 89.4% | 34.5% | **6.4%** | 99.6% | **28.8%** | 96.4% |

### Key Finding: Transition Accuracy Is the Bottleneck

The 90%+ gap between steady-state and transition accuracy is the single most important finding:

- **Button transitions**: 6-10% accuracy vs 99%+ on steady-state frames
- **Stick transitions**: 25-29% accuracy vs 96%+ on steady-state frames
- Transitions are only ~7% of button frames and ~17% of stick frames
- The model achieves high overall accuracy by predicting "don't change" on every frame
- All Melee tech skill requires frame-precise input CHANGES — this directly explains poor gameplay

### Other Diagnostic Findings

- **main_f1 is low (34-38%)** despite 85% top-1 accuracy. The 63 stick clusters are not learned equally — neutral and cardinal directions dominate, rare clusters (wavedash angles, DI angles) are ignored. Macro F1 exposes this.
- **Neutral rate is NOT the problem.** Predicted neutral rate (35-39%) matches ground truth (31-36%). The model isn't "lazy" or stuck on neutral — it just doesn't know WHEN to change.
- **Per-button F1 is reasonable.** Critical buttons (A, B, X, Y, L, R) range 72-94%. L and R (shield/wavedash) are strongest at 91-94%. X (jump) is weakest and most variable (72-89%).
- **Stick confidence is only 75%.** The model hedges between clusters, which could cause instability in inference.

---

## Phase 3: Live Inference Testing

### Setup

Downloaded the g2-drop05 checkpoint (the sweep winner at 90% btn_f1). Ran inference against a CPU opponent on Dolphin with blocking_input=True.

### Observations

- The model played recognizable Melee — moving, attacking, sometimes shielding
- Behavior was "ok but kinda weird" — timing felt off, inputs seemed delayed or sticky
- BUTTON_START was being predicted and pressed, pausing the game → fixed by adding `BLOCKED_BUTTONS = {melee.enums.Button.BUTTON_START}` in inference.py
- After the fix, gameplay looked slightly better but still clearly not human-quality
- The model would get stuck in loops (repeating the same action) or fail to react to opponent actions

### Interpretation

The diagnostic findings explain the live behavior precisely:

1. The model correctly maintains steady state (~99% accuracy when nothing should change)
2. It fails to initiate actions at the right time (7% button transition accuracy)
3. It fails to adjust stick position at the right time (27% stick transition accuracy)
4. In gameplay, this manifests as a bot that "holds" inputs too long and reacts late

---

## Phase 4: Solution Analysis

### The Core Problem

The model predicts absolute controller state every frame. ~93% of frames are identical to the previous frame. The loss-minimizing strategy is "predict what was pressed last frame." This is not a model capacity problem, a data quantity problem, or a hyperparameter problem — it's a **problem formulation** problem.

### Solutions Considered and Rejected

**1. Transition-weighted loss** — Upweight frames where ground truth changes by 5-10x.

Rejected because: bespoke weighting introduces a new hyperparameter that trades off transition accuracy against steady-state accuracy. Tuning this weight becomes whack-a-mole — improve transitions, hurt steady-state, re-tune, repeat. Not a generalizable solution.

**2. Auxiliary change-prediction heads** — Add binary heads predicting "does this output change next frame?" to provide explicit transition gradient signal.

Rejected because: this is essentially a specialized module for transition detection. If transition patterns are context-dependent (different characters, situations), a generic change head may not add value over the backbone learning transitions organically. It's an indirect form of transition weighting with extra parameters.

**3. Self-controller input dropout** — Zero out self-controller features with some probability during training, forcing the model to infer from game state.

Rejected because: this is not meaningfully different from regular dropout. Regular dropout zeroes random hidden neurons but the self-controller signal survives through the remaining 90% of neurons. Self-controller input dropout removes the entire information channel for a frame. The distinction is real (capacity noise vs. channel removal) but marginal in practice.

**4. Predict deltas (XOR)** — Predict "what changes" instead of "what is pressed."

Rejected because: cascading error risk. One wrong prediction (a flip that shouldn't have happened, or a missed flip) permanently corrupts the state. With absolute prediction, the model can self-correct on the next frame. With deltas, errors accumulate via XOR and require a second compensating error to undo.

### Chosen Solution: Remove Self-Controller Inputs Entirely

**Rationale:**

1. **Game state already encodes input consequences.** `self_action` (400+ states like WAIT, DASH, JUMPSQUAT, LANDING) + `self_action_frame` (which frame within the animation) + position + velocity + flags = the game's internal state machine. If the character is in jumpsquat, jump was pressed. If airborne with diagonal velocity, a wavedash angle was input. The model doesn't need to be told what was pressed — it can see the result.

2. **Eliminates train/inference distribution shift completely.** Currently, training uses ground-truth controller state as input, but inference uses the model's own (imperfect) predictions fed back in. This mismatch causes cascading errors. With no self-controller inputs, there is no feedback loop — the model reads game state from Dolphin identically at training and inference time.

3. **Follows proven precedent.** `no_opp_inputs=True` already improved performance by removing opponent controller inputs that distracted from learning. Self-controller inputs are worse — they provide a shortcut that lets the model avoid learning from game state entirely ("just copy what I was doing").

4. **Matches human cognition.** Players don't attend to their own inputs when playing. They watch the screen (game state) and decide what to press based on what they see. The model should do the same.

5. **Simplifies inference.** The entire `_prev_pred` prediction feedback mechanism (decoding model outputs back to input features, handling feedback state, the `--no-pred-feedback` flag) becomes unnecessary. The inference loop just reads game state and predicts outputs.

### What Gets Removed from Input

| Removed | Dim | Description |
|---------|-----|-------------|
| `self_buttons` | 12 | Binary button press state (A, B, X, Y, Z, L, R, START, D-pad) |
| `self_analog` | 4 | Main stick X/Y, L shoulder, R shoulder (float) |
| `self_c_dir` | 1 | C-stick direction (categorical, 5 classes) |
| `self_nana_*` | same | Ice Climbers partner (AI-controlled, also a shortcut) |

### What Stays in Input (Game State)

| Kept | Description |
|------|-------------|
| `self_action` | 400+ action states (WAIT, WALK, DASH, JUMPSQUAT, LANDING, etc.) |
| `self_action_frame` | Which frame within the current animation |
| `self_pos_x/y` | Character position |
| 5 velocity components | air_x, ground_x, attack_x, attack_y, self_y |
| `self_percent`, `self_stock` | Damage, lives |
| `self_jumps_left` | Remaining jumps |
| `self_hitlag/hitstun/invuln_left` | Frame counters for combat states |
| `self_shield_strength` | Current shield health |
| 8 ECB values | Environmental collision box |
| 5 flags | on_ground, off_stage, facing, invulnerable, moonwalkwarning |
| All opponent/nana/projectile/stage features | Full game state |

The `self_action + self_action_frame` combination is the game's internal state machine — it deterministically encodes the consequence of all past inputs.

---

## Phase 5: Sequence Length Hypothesis

Longer input windows (T=120, T=240 instead of T=60) contain more transition events per sample. At 60fps, T=60 covers 1 second — most 1-second windows contain 0-2 transitions. T=240 covers 4 seconds and would typically contain 5-15 transitions, giving the model more signal per training example.

However, the sweep showed seq_len=120 scored 86.3% (below baseline 88.9%) with self-controller inputs present. This may be because longer sequences + self-controller inputs makes the "copy previous" shortcut even more effective (more context to copy from). With self-controller inputs removed, longer sequences may behave differently.

Worth testing as a side experiment alongside no-self-inputs.

---

## Implementation

Added `--no-self-inputs` flag following the same pattern as `--no-opp-inputs`:

| File | Change |
|------|--------|
| `features.py` | `build_feature_groups(no_self_inputs=...)` removes `buttons`, `analog`, and `c_dir` from self/self_nana groups |
| `frame_encoder.py` | All 3 encoder types skip self-controller tokens, update token counts |
| `model.py` | `ModelConfig.no_self_inputs` field, wired to encoder construction |
| `train.py` | `--no-self-inputs` CLI argument |
| `inference.py` | Skip `_prev_pred` feedback when `no_self_inputs=True` |
| `eval.py` | Pass `no_self_inputs` from checkpoint config |

---

## Phase 6: DDP Training & Speed Optimization

### DDP Implementation

Integrated PyTorch Distributed Data Parallel into `train.py` and `dataset.py` to enable multi-GPU training of a single model. Created `parallel.sh` as a launcher wrapping `torchrun` over SSH.

| File | Change |
|------|--------|
| `train.py` | Auto-detect `torchrun` via `RANK` env var, `DDP` wrapping, rank-0 logging/checkpointing, `dist.barrier()` sync, `--scale-lr` flag |
| `dataset.py` | `rank`/`world_size` params for file-level DDP sharding across ranks + worker-level sharding within each rank |
| `parallel.sh` | New bash launcher: machine registry (A/B/C/D), SSH-based `torchrun` dispatch, foreground/background modes, `DRY=1` |

### Pre-Tensorized Dataset Pipeline

Parquet-based streaming was bottlenecked by `pd.read_parquet()` + pandas preprocessing on every epoch. Created `tensorize.py` to pre-convert datasets into `.pt` shard files.

**tensorize.py**: Reads parquets, applies full preprocessing (categorical mapping, normalization, windowing), stacks windows into contiguous tensors, distributes across N shards. Output: `train_shard_000.pt` ... `train_shard_063.pt` + `val_shard_*.pt` + `tensor_meta.json`.

**dataset.py changes**: `StreamingMeleeDataset` auto-detects `tensor_meta.json` in the data dir. If present, uses fast `torch.load()` + tensor indexing path (`_iter_tensorized`). Otherwise falls back to parquet path (`_iter_parquet`). Removed the previous `_ensure_ddp_shard_files` auto-duplication code — tensorization with sufficient shards replaces it.

**Wavedash overfit dataset**: 28,740 windows from 1 parquet → 64 train shards (2.5 GB) + 8 val shards (2.5 GB). Each rank×worker gets a unique shard file, eliminating I/O contention.

### wandb.log Optimization

Removed per-step `wandb.log()` call (25+ metrics × every step = heavy HTTP overhead). Metrics are now only logged at `log_interval` via the existing averaged-metric path. Per-step accumulation into `_run_sums` is unchanged.

### Wavedash Overfit DDP Results

**Baseline** (parquet file duplication, `--no-compile`): **17.2 step/s** on 8×RTX 5090

**Optimized** (tensorized shards, `torch.compile`, gated wandb.log): **18.6 step/s** on 8×RTX 5090 (Machine C, PyTorch 2.8.0) — **+8.1% throughput**, still climbing at step 9,672.

Machine D (same 8×RTX 5090 but PyTorch 2.10.0) showed only 9.7 step/s — a suspected `torch.compile` regression in the newer PyTorch version. Run killed and relaunched.

| Config | step/s | Notes |
|--------|--------|-------|
| 1×5090 single-GPU (parquet) | 9.2 | `docs/research-notes-2026-03-15.md` |
| 8×5090 DDP parquet+file-dup | 17.2 | Previous best, `--no-compile` |
| 8×5090 DDP tensorized+compile | 18.6+ | Current best, still ramping |

---

## Status

DDP + tensorized pipeline running on two 8×5090 pods. Machine C (PyTorch 2.8) reaching 18.6 step/s and climbing. Machine D (PyTorch 2.10) relaunched after a performance regression.

Next: monitor convergence to target val_f1=98.5%, compare wall time to overfit against single-GPU baseline.
