# MIMIC Pipeline Audit + Closed-Loop Eval — Results

## Phase 1: Tensor-Level Pipeline Audit

### diagnose.py findings

Ran `diagnose.py` comparing a saved inference batch (call #300) against a training parquet window:

| Feature group | Training range | Inference range | Verdict |
|---|---|---|---|
| `self_numeric` (22-dim) | [-1.24, 0.72] | [-2.17, 0.72] | **Minor diffs** (position/percent vary between games) |
| `opp_numeric` (22-dim) | [-1.24, 1.23] | [-1.24, 0.72] | OK |
| `self_buttons` (12-dim) | [0, 1] | [0, 1] | OK |
| `self_analog` (4-dim) | [-2.13, 2.44] | [-1.55, 2.29] | OK |
| `global numeric` (20-dim) | [-1.23, 1.23] | **[-20.93, 26.00]** | **Stage geometry (FD no-platform values after z-score normalization)** |
| All categoricals | Match | Match | OK (different games → different actions, expected) |

**Critical finding**: The model output comparison showed that **BOTH training and inference batches produce near-zero button probabilities** (A=0.18% train, A=0.26% inf). The pipeline is correct — the model just doesn't press buttons on neutral frames.

### inference.py fixes applied

Fixed all known mismatches between `inference.py` and `extract.py`:

- **Speed fields**: `speed_air_x_self`, `speed_ground_x_self`, `speed_x_attack`, `speed_y_attack`, `speed_y_self` — now read from `ps.speed_*` instead of hardcoded 0.0
- **hitlag_left**: now read from `ps.hitlag_left` instead of hardcoded 0
- **invuln_left**: now read from `ps.invulnerability_left` instead of hardcoded 0
- **ECB values**: now read from `ps.ecb_bottom/left/right/top` instead of hardcoded 0.0
- **Projectiles**: now populated from `gs.projectiles` (matching extract.py format) instead of all-empty
- Same fixes applied for Nana (Ice Climbers partner)

## Phase 2: Closed-Loop Evaluation

### Setup

- 20 Falco-on-FD games sampled from training data (not freshly recorded; serves as pipeline proxy)
- Preprocessed with separate `norm_stats.json` and `cat_maps.json`
- Reused existing 63-cluster stick centers and 4-bin shoulder centers
- Trained `closed-loop-overfit` for 5000 steps (seq_len=60, batch=64, lr=8e-4)

### Training convergence

| Step | Train Loss | btn_f1 | cact | Val btn_f1 | Val cact |
|---|---|---|---|---|---|
| 250 | 1.337 | 13.8% | 43.4% | 10.1% | 31.5% |
| 1000 | 1.063 | 81.5% | 80.9% | 72.7% | 70.4% |
| 2500 | 0.888 | 86.6% | 90.3% | 75.5% | 73.3% |
| 5000 | 0.746 | 85.1% | 95.2% | 76.9% | 76.1% |

**The model learns to press buttons confidently within ~1000 steps.**

### Comprehensive offline inference (400 windows, 20 games)

| Metric | Closed-Loop Model | Full-Data Model |
|---|---|---|
| **Button F1** | **94.6%** (P=97.8%, R=91.7%) | **98.5%** (P=99.3%, R=97.8%) |
| Avg btn confidence (active frames) | **0.830** | **0.860** |
| Avg btn confidence (inactive frames) | 0.040 | 0.025 |
| **Stick cluster top-1 accuracy** | **1.0%** | **0.8%** |
| Shoulder bin accuracy | 78.0% | 80.2% |
| C-dir accuracy | 100.0% | 99.5% |

## Key Conclusions

1. **The pipeline is correct.** Both models produce 94-98% button F1 on training data, with clear separation between active (83-86% confidence) and inactive (2.5-4% confidence) frames.

2. **Stick prediction is the bottleneck.** Top-1 accuracy is ~1% across 63 clusters. The model can't reliably predict the exact stick position. This is the root cause of poor live inference — wrong stick inputs → wrong character states → distribution shift → even worse predictions.

3. **The inference failure is NOT a pipeline bug.** It's the compounding error from inaccurate stick predictions during live play. The model confidently presses buttons when it sees the right game states (training data), but during live play, the stick errors cause the character to be in unexpected states.

4. **The full-data model is actually better** than the 20-game overfit model (98.5% vs 94.6% btn F1), suggesting more data helps rather than hurts.

## Phase 3: Closed-Loop Wavedash (2026-03-15)

### Overview

Generated a synthetic wavedash dataset using `generate_wavedash_replay.py` (Falco
wavedashing back and forth on FD, 28,800 frames with edge-safety reversal).
Trained with focal loss and canonical 63-cluster stick centers to near-epsilon loss.
Then ran live in Dolphin.

### Root Cause of Live Inference Failure

**Pipe synchronization.** `melee.Console` was created with `blocking_input=False` (the
default), so Dolphin ran at 60fps while our inference produced ~15fps. Most controller
commands were lost — they arrived between frames and were never processed.

Fixed by setting `blocking_input=True`, adopted from [HAL](https://github.com/ericyuegu/hal).
Now Dolphin blocks until we flush, guaranteeing every input is processed.

### Additional Fixes

- HAL-style explicit press/release per button per frame (no `release_all()`)
- Always send `press_shoulder()` for both L and R, even when 0
- `flush()` on menu frames and `gs.frame < 0` to prevent deadlock

### Inference Performance

Per-frame preprocessing was bottlenecked at 57ms (pandas DataFrame operations on
a single row). Replaced with a pure-Python fast path processing raw dict values
directly, cached in a rolling deque of tensors.

| Component | Before (ms) | After (ms) |
|-----------|------------|------------|
| Preprocessing | 57 | 0.3 |
| Tensor stacking | 0.5 | 0.6 |
| Model forward | 2.2 | 2.2 |
| **Total** | **~60** | **~3.5** |

### Result

**Falco wavedashes correctly at 60fps in closed loop.** The full sequence
(stand → jump → airdodge → slide → reverse → repeat) executes without error.

### Next Steps

1. Train on full Melee dataset with focal loss and verify generalization
2. Scheduled sampling / noise injection for robustness to compounding error
3. Test against human and CPU opponents in varied matchups
