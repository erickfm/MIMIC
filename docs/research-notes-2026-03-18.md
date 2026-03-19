# Research Notes — 2026-03-18

## Context

Continued from [2026-03-17](research-notes-2026-03-17.md). Previous session implemented `--no-self-inputs` and DDP infrastructure. Today's goals: make no_self_inputs the default, validate it experimentally, run batch size sweep for DDP, and launch full-data 1B-sample training runs.

---

## Phase 1: no_self_inputs Default + Diverging Run Cleanup

### Diverging full-ddp-D-500M

The `full-ddp-D-500M` run on Machine D (8×RTX 5090, PyTorch 2.10) was found diverging at step 346,622/976,562 with exploding gradient norms:

| Step range | Grad norm | Val btn_f1 |
|------------|-----------|------------|
| 0–200k | 1.5–6 (healthy) | 88.7% → 89.3% (improving) |
| 230k | 9–21 | — |
| 280k | 44–110 | — |
| 346k | **436** | 84.6% (degrading) |

Best checkpoint at step 195,312 (val btn_f1=89.3%). Run killed. Root cause: likely LR too high for this training stage without gradient clipping configured (note: `GRAD_CLIP_NORM=1.0` IS hardcoded in train.py — so this was clipped at 1.0 and still diverging, meaning the underlying gradients were massive).

This run used self-controller inputs (the old default). The `--no-self-inputs` flag was implemented on 03-17 but never used in any full-data run.

### Making no_self_inputs Default

Flipped the default across the codebase:

| File | Change |
|------|--------|
| `mimic/model.py` | `ModelConfig.no_self_inputs = True` |
| `train.py` | `get_model(no_self_inputs=True)`, `get_datasets(no_self_inputs=True)` |
| `train.py` | CLI: `--no-self-inputs` (opt-in) → `--self-inputs` (opt-in to re-enable) |

---

## Phase 2: Self-Inputs Ablation (Wavedash)

### Setup

4 DDP runs across machines C/D/E/F (8×RTX 5090 each), wavedash dataset, seed-matched pairs:

| Machine | Seed | Self Inputs | Run Name |
|---------|------|-------------|----------|
| C | 42 | No (default) | wd-nsi-seed42 |
| D | 42 | Yes (`--self-inputs`) | wd-si-seed42 |
| E | 43 | No (default) | wd-nsi-seed43 |
| F | 43 | Yes (`--self-inputs`) | wd-si-seed43 |

Config: medium model, batch=256/GPU (eff 2048), lr=5e-5, max_samples=2M, target_val_f1=0.985, --no-compile, wandb group `self-inputs-ablation`.

### DDP Fixes Required

1. **NCCL timeout**: Increased from default 600s to 1800s in `dist.init_process_group` to prevent the transient epoch-boundary desync crash from 03-17.
2. **find_unused_parameters**: Added `find_unused_parameters=True` to DDP wrapper. With `no_self_inputs=True`, self-controller encoder parameters are unused in the forward pass, causing DDP's gradient reduction to fail.

### Results

**Self-inputs made zero difference on wavedash:**

| | No Self (seed 42) | Self (seed 42) | No Self (seed 43) | Self (seed 43) |
|---|---|---|---|---|
| Best Val F1 | 96.11% | 96.13% | 96.05% | 96.05% |
| Val Precision | 97.3% | 97.4% | 97.7% | 97.7% |
| Val Recall | 94.9% | 94.9% | 94.4% | 94.4% |

Convergence curves were **identical step-for-step** within seed pairs. Delta across all checkpoints: 0.0% ± 0.1%. Self-controller inputs are a no-op on wavedash — the model doesn't use them even when available.

None of the 4 runs reached the 98.5% target. All plateaued at ~96.1%. The bottleneck was the effective batch size (256×8=2048), not the self-inputs flag.

---

## Phase 3: DDP Batch Size Sweep (Wavedash)

### Setup

Swept per-GPU batch size {32, 64, 128, 256} to find what matches the canonical single-GPU batch=256 convergence. All on wavedash, seed=42, target 98.5%, --no-compile.

| Machine | Per-GPU Batch | Effective Batch | Max Steps |
|---------|---------------|-----------------|-----------|
| C | 32 | 256 | 62,500 |
| D | 64 | 512 | 31,250 |
| E | 128 | 1,024 | 15,625 |
| F | 256 | 2,048 | 7,812 |

### Results

| | bs=32 (eff 256) | bs=64 (eff 512) | bs=128 (eff 1024) | bs=256 (eff 2048) |
|---|---|---|---|---|
| **Target 98.5%** | **REACHED** | **REACHED** | Not reached | Not reached |
| Best Val F1 | 98.64% | 98.62% | 98.28% | 96.11% |
| Steps to target | 37,500 | 17,182 | — | — |
| Wall time | 1,726s | 2,322s | 1,829s | 3,415s |
| Throughput | 21.7 step/s | 7.4 step/s | 8.5 step/s | 2.3 step/s |

### Key Findings

- **bs=32 and bs=64 both hit 98.5%** — effective batch 256 and 512 work for DDP wavedash
- **bs=64 converged in fewer steps** (17k vs 37.5k) and fewer epochs (38 vs 42) — effective batch 512 may be the DDP sweet spot
- **bs=128 (eff 1024) just missed** at 98.28% — ran out of step budget, could likely hit with more training
- **bs=256 (eff 2048) plateaus at 96.1%** — too large, can't close the last 2.5%
- Machine D (PyTorch 2.10) ran at half speed (7.4 vs 21.7 step/s) confirming the torch.compile regression

---

## Phase 4: Full-Data 1B-Sample Runs

### Setup

Launched 2 runs on full Melee dataset (200-shard, 505M frames, 60k games) with no_self_inputs=True:

| Machine | Run | Batch (eff) | Seed | Max Steps |
|---------|-----|-------------|------|-----------|
| E | full-bs128-seed43 | 128 (1024) | 43 | 976,562 |
| F | full-bs128-seed42 | 128 (1024) | 42 | 976,562 |

Config: medium model, lr=5e-5, --no-compile, wandb group `full-1B`. ETA ~27 hours each.

Two additional bs=64 runs (seed 42 and 43) were planned but blocked by CUDA zombie issues on replacement machines (see Phase 5).

### Early Observation: no_self_inputs Dramatically Slower on Full Data

At ~70k steps, both runs show btn_f1 of only 35-50%. For comparison, the prior `full-ddp-D-500M` run (WITH self-inputs) hit 89% btn_f1 by step 48k.

| Metric | full-bs128-seed42 (step 78k) | full-ddp-D-500M (step 48k, self-inputs) |
|--------|-----|-----|
| btn_f1 | 50.8% | 89.7% |
| main_f1 | 11.3% | 35.0% |
| total_loss | 1.55 | 0.61 |

This is the first evidence that removing self-controller inputs **significantly hurts learning speed on full Melee data**, despite making zero difference on wavedash. The wavedash task is too simple (single repeating pattern) to reveal this effect.

This aligns with HAL's finding (Eric Gu): "at small scale, inductive biases are important, such as explicitly adding previous controller presses instead of inferring from changes in game state."

The runs are continuing to see where no_self_inputs plateaus at the 1B sample budget.

---

## Phase 5: Machine Fleet Changes

### GPU Fleet Issues

Machine C (194.14.47.19, old port 22824) had only 30 GB disk — couldn't fit the full dataset. Machine D (142.127.93.36:11559) ran PyTorch 2.10 at half speed.

Attempted replacements hit CUDA zombie issues: launching torchrun before all deps were installed (missing `melee` package) crashed after CUDA init, leaving GPU contexts locked. Container restarts on the same physical host didn't help — the GPU state was corrupted at the host level.

| Attempt | IP | Result |
|---------|-----|--------|
| 50.145.48.91 | CUDA zombied from failed launch |
| 74.2.96.11 | CUDA zombied from failed launch |
| 74.2.96.62 | CUDA broken on arrival (same physical host?) |
| 194.14.47.19:22874 | **Working** — new Machine C |

Machine D still needs replacement.

### Lesson Learned

**Always run full dep install + CUDA verification before first torchrun.** A crash mid-CUDA-init in a container permanently locks GPUs until the physical host is reset. Use `setup.sh` or verify all imports manually first.

### Dataset Distribution

The HuggingFace dataset was renamed from `frame-melee` to `mimic-melee`. The full repo (2.59 TB, 647 shards) doesn't fit on 1 TB machines. Uploaded the 200-shard version (505M frames, 761 GB) to `erickfm/mimic-melee-500M` from Machine F.

Data transfer between machines solved by pushing Machine F's SSH public key to other machines for direct rsync.

---

## Phase 6: HAL Comparison (Eric Gu)

### Architecture Comparison

Deep analysis of the [HAL codebase](https://github.com/ericyuegu/hal) revealed several key design differences:

| | HAL | MIMIC |
|---|---|---|
| Params | 20M (512-d, 6L, 8H) | 32M (768-d, 4L, 8H) |
| Seq len | 256 frames (4.3s) | 60 frames (1s) |
| LR | 3e-4 | 5e-5 (6x lower) |
| Training samples | 16.8M | 50M–1B |
| Training time | 5 hours, 2× 3090 | 27+ hours, 8× 5090 |
| Result | 95% win rate vs CPU | 90% offline F1, poor live play |
| Frame encoder | Flat concat → linear | 16-token intra-frame attention |
| Buttons | Single-label softmax (6 classes) | 12-dim multi-label BCE |
| Stick clusters | 37 hand-designed (Melee angles) | 30-63 k-means from data |
| Controller feedback | Yes (critical per author) | Removed (default) |
| Loss | Plain cross-entropy | Focal loss (gamma=2) |
| Pos encoding | Relative (Music Transformer) | RoPE |

### Efficiency Gap

HAL achieves a **playable bot with ~100x less compute**. Primary factors:

1. **Higher LR (3e-4 vs 5e-5)** — trains fast and hard
2. **Simpler frame encoder** — flat concat vs our expensive 16-token attention
3. **Simpler button formulation** — 1-of-6 softmax vs 12-dim multi-label
4. **Controller feedback** — explicit shortcut the model can leverage
5. **Longer context** — 256 frames captures more transitions per sample
6. **Hand-designed clusters** — hit exact Melee mechanics

### Implications

The HAL comparison suggests our architecture may be over-engineered (complex frame encoder, multi-label buttons) while missing key inductive biases (controller feedback, Melee-specific clusters, longer context). Simplifying the architecture while adding domain-specific structure could dramatically improve training efficiency.

---

## Status

- **E**: `full-bs128-seed43` training, ~27h remaining
- **F**: `full-bs128-seed42` training, ~24h remaining, also uploaded mimic-melee-500M to HuggingFace
- **C** (194.14.47.19:22874): set up with deps, needs data rsync from F
- **D**: needs replacement machine
- Two bs=64 runs pending machine availability
