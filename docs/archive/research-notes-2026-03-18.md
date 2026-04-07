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

## Phase 7: Self-Inputs Control Run + Batch Size Theory

### Setup

Launched `full-bs128-si-seed42` on Machine C with `--self-inputs` — same seed, same batch size as Machine F's no-self-inputs run. Direct comparison.

### Mid-Training Observation (step ~200k)

| | C: self-inputs seed42 | E: no-self seed43 | F: no-self seed42 |
|---|---|---|---|
| Step | 190k | 205k | 259k |
| Train btn_f1 | 43.9% | 38.6% | 56.3% |
| **Val btn_f1** | **39.9%** | **39.6%** | **39.1%** |
| Val main_f1 | 6.6% | 7.1% | 8.0% |
| Val loss | 1.96 | 1.82 | 1.83 |
| step/s | 14.2 | 10.0 | 11.0 |

### Corrected Understanding: Batch Size, Not Self-Inputs

**The earlier conclusion that no_self_inputs "dramatically hurts learning" was wrong.** Machine C with self-inputs has nearly identical val metrics to the no-self-inputs runs. All three runs are underperforming equally.

The confound is **effective batch size**. All three runs use batch=128 × 8 GPUs = eff 1024. Prior successful runs used:

| Run | Eff batch | Result |
|-----|-----------|--------|
| Sweep (single GPU, batch=256) | 256 | 88-90% val btn_f1 |
| full-ddp-D-500M (batch=64 × 8) | 512 | 89% btn_f1 at step 48k |
| Wavedash bs=32 (batch=32 × 8) | 256 | Hit 98.5% target |
| Wavedash bs=64 (batch=64 × 8) | 512 | Hit 98.5% target |
| Wavedash bs=128 (batch=128 × 8) | 1024 | Missed target (98.28%) |
| Wavedash bs=256 (batch=256 × 8) | 2048 | Plateaued at 96.1% |
| **Current full-data runs (batch=128 × 8)** | **1024** | **~39% btn_f1 at 200k+ steps** |

The pattern is consistent: eff batch ≤512 converges well, eff batch ≥1024 underperforms. This holds on both wavedash and full data. On wavedash the gap was small (98.5% vs 98.3%), but on full Melee data it's catastrophic (89% vs 39% at comparable step counts).

The self-inputs vs no-self-inputs question **cannot be evaluated at eff batch 1024** because the batch size itself prevents convergence. Any real self-inputs effect is masked by the batch size bottleneck.

### Why Large Batch Hurts More on Full Data Than Wavedash

Wavedash is a single repeating pattern — larger batches still capture the full distribution. Full Melee has thousands of characters, matchups, situations, and tech patterns. At eff batch 1024, each gradient step averages over too many diverse examples, washing out the signal from rare but important patterns (wavedash angles, combo starters, DI). Smaller batches see fewer examples per step but each step gets a sharper gradient signal, enabling the model to learn rare patterns without them being averaged away.

This is consistent with HAL's finding: "going to 1024 batch size made the model generalize much worse — smaller batch sizes seem to help regularize learning."

### Current Theory

1. **Effective batch size is the dominant variable** — more important than self-inputs, dropout, or most other hyperparameters on full Melee data
2. **Eff batch 256-512 is the sweet spot** for our model/LR/data combination, consistent across wavedash and full data
3. **Self-inputs effect is still unknown** — needs evaluation at eff batch ≤512
4. **LR may interact with batch size** — HAL uses 3e-4 (6x higher) with batch 1024. We might be able to use larger batches with higher LR, but our sweep found only 5e-5 worked. That sweep used self-inputs though.

### Final Results (Runs Killed 2026-03-19)

All three runs plateaued and were killed. LR had decayed to 1/6th–1/12th of peak with no improvement in val metrics.

| | C: self-inputs seed42 | E: no-self seed43 | F: no-self seed42 |
|---|---|---|---|
| **Final step** | 801k / 976k (82%) | 635k / 976k (65%) | 727k / 976k (74%) |
| **Val btn_f1** | **41.5%** | 38.3% | 38.5% |
| **Val main_f1** | 7.1% | 7.7% | 8.1% |
| **Val loss** | 1.84 | 1.97 | 1.69 |
| **LR at kill** | 4.3e-6 | — | 8.4e-6 |
| **Samples seen** | ~820M | ~650M | ~745M |

**Conclusion:** Eff batch 1024 plateaus at ~40% val btn_f1 on full Melee data regardless of self-inputs. Self-inputs gave a marginal +3% edge (41.5% vs 38.5%) but this is negligible compared to the 89% achieved at eff batch 512 by the old full-ddp-D-500M run.

### Why Large Batch Causes Such a Large Gap

The 40% vs 89% gap from a 2x batch size change is extreme. Two contributing factors beyond gradient averaging:

1. **Half the gradient updates for the same data.** At eff 1024 with 976k max_steps, the model gets half the optimization steps compared to eff 512 with 1.95M steps for the same 1B samples. The cosine LR schedule is tied to max_steps, so the model has half as many steps at high LR before decay kicks in.

2. **LR not scaled with batch size.** The standard fix for large-batch training is linear LR scaling. We used the same lr=5e-5 for both eff 512 and 1024. Doubling to 1e-4 for eff 1024 might close the gap. However, the prior sweep found only lr=5e-5 converged — though that sweep used self-inputs and single-GPU batch=256.

These are testable hypotheses for future runs.

### Next Steps

1. **Priority: bs=64 runs on full data** (eff batch 512) — the proven working config. Need Machine D replacement. Run with both seeds (42, 43) and both self-inputs conditions.

2. **LR scaling experiment** — test lr=1e-4 at eff 1024 (bs=128×8) to determine if the gap is purely an LR issue. If this works, 8-GPU DDP becomes viable at bs=128 which is more throughput-efficient.

3. **Self-inputs ablation at eff 512** — the only valid way to test self-inputs. The eff-1024 runs were confounded by batch size.

4. **HAL-inspired changes** — single-label buttons, longer context (256 frames), higher LR. Each could be tested independently at eff 512.

---

---

## Phase 8: Critical Bug — `--self-inputs` Flag Broken (Found 2026-03-19)

### The Bug

In `train.py` `get_model()`, the override logic for `no_self_inputs` only applied when the value was `True`:

```python
if no_self_inputs:
    overrides["no_self_inputs"] = True
```

When `--self-inputs` was passed (`no_self_inputs=False`), the override was **never set**. Since `ModelConfig.no_self_inputs` was changed to default `True` in commit `83fd59c`, the config always received `True` regardless of the CLI flag.

### Impact

**Every run since commit `83fd59c` (2026-03-18) trained with `no_self_inputs=True`**, including runs explicitly launched with `--self-inputs`. This affected:

- All 4 wavedash self-inputs ablation runs (both conditions were actually identical)
- All 3 eff-1024 full-data runs (the "self-inputs control" on Machine C was not actually using self-inputs)
- All 3 eff-512 full-data runs (Machine F's "self-inputs" run was identical to C and E)

### Consequences

1. **The batch size theory was wrong.** The ~40% btn_f1 plateau on full data was not caused by effective batch size. It was caused by `no_self_inputs=True` — the model cannot learn full Melee gameplay without self-controller feedback. The old `full-ddp-D-500M` run that hit 89% btn_f1 was the last run on the old code where self-inputs worked.

2. **The wavedash ablation was accidentally valid.** Both conditions truly ran with `no_self_inputs=True`, confirming that self-inputs don't matter for wavedash (a conclusion that happened to be correct, for the wrong reason).

3. **All batch size conclusions are confounded.** The eff-1024 vs eff-512 comparison was actually comparing two runs that both lacked self-inputs. The batch size effect may still be real, but we can't confirm it from these runs.

### Fix

```python
# Before (broken)
if no_opp_inputs:
    overrides["no_opp_inputs"] = True
if no_self_inputs:
    overrides["no_self_inputs"] = True

# After (fixed)
overrides["no_opp_inputs"] = no_opp_inputs
overrides["no_self_inputs"] = no_self_inputs
```

Note: the same pattern existed for `no_opp_inputs`, meaning `--opp-inputs` was also broken. Both are now fixed.

### Root Cause

The original override logic (`if value: overrides[key] = True`) was designed for flags that default to `False` — it only needs to override when turning something ON. When we flipped the default to `True`, the logic needed to also handle turning it OFF, but we didn't update it.

---

## Phase 9: Next Experiment — Context Window + Capacity (Planned 2026-03-19)

### Motivation

With the `--self-inputs` bug found, we know no_self_inputs=True plateaus at ~40% btn_f1 on full data. Before giving up on no-self-inputs entirely, we want to test whether **longer context** (256 frames, matching HAL) helps the model infer intention from game state trajectory instead of needing explicit controller feedback. HAL uses 256-frame context and self-inputs — we want to see if long context alone can compensate.

### Planned Runs

| Machine | Run | Context | Self-Inputs | Model | Samples |
|---------|-----|---------|-------------|-------|---------|
| C | `full-nsi-ctx256-seed42` | 256 frames | No | medium (~32M) | 250M |
| E | `full-nsi-ctx256-seed43` | 256 frames | No | medium (~32M) | 250M |
| F | `full-si-ctx60-seed42` | 60 frames | **Yes** (fixed flag) | medium (~32M) | 250M |
| New D | `full-nsi-ctx256-xxl-seed42` | 256 frames | No | xxlarge (~228M) | 250M |

**What we learn:**
- C vs E: seed variance at ctx=256 no-self-inputs
- C vs F: does ctx=256 no-self-inputs match ctx=60 self-inputs? (the key question)
- New D: does a 7x larger model help when inputs are removed? (capacity test)

All runs at eff batch 512 (bs=64 × 8 GPUs), lr=5e-5, --no-compile.

### Available Model Presets

| Preset | d_model | Layers | Heads | ~Params |
|--------|---------|--------|-------|---------|
| medium | 768 | 4 | 8 | ~32M |
| base | 1024 | 4 | 8 | ~51M |
| xlarge | 1024 | 8 | 8 | ~102M |
| xxlarge | 1536 | 8 | 12 | ~228M |

---

## Status (End of 2026-03-19)

- All runs killed, bug fix synced to C/E/F
- **C** (`194.14.47.19:22874`): 8× RTX 5090, ready, has data + deps + fixed code
- **D** (`50.145.48.107:12404`): **8× H200 144 GB**, 2 TB disk, PyTorch 2.4 — needs setup (deps, code, data)
- **E** (`66.222.138.178:11335`): 8× RTX 5090, ready, has data + deps + fixed code
- **F** (`74.2.96.10:18619`): 8× RTX 5090, ready, has data + deps + fixed code
- All 4 machines available, ready to launch context window ablation + capacity test
