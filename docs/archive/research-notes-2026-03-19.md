# Research Notes — 2026-03-19

## Context

Continued from [2026-03-18](research-notes-2026-03-18.md). Previous session found the `--self-inputs` flag was broken (all runs since 03-18 trained with `no_self_inputs=True`), killed the plateaued eff-1024 runs, and planned a new experiment: context window ablation (ctx=256 vs ctx=60) + capacity test (huge model on H200s). Today's goals:

1. Harden the training pipeline for long-context and large-model runs
2. Launch 3 ablation runs on 5090 machines (C/E/F)
3. Set up Machine D (8× H200 144 GB), deploy code + data, launch huge model
4. Monitor all 4 runs

---

## Phase 1: Pipeline Hardening

### Commit `f242c39` — "Scale training for ctx=256 ablation + capacity test on H200s"

Changes across 4 files (+105, -42 lines):

| File | Changes |
|------|---------|
| `parallel.sh` | Fixed SSH port for Machine C (→22874), updated Machine D host+port (50.145.48.107:12404) |
| `mimic/model.py` | Added `huge` (~621M) and `giant` (~1.4B) model presets with widened `d_intra=512` and `head_hidden=512/768` |
| `mimic/dataset.py` | Added dataset shrinkage warning when long context windows skip >5% of games in a shard |
| `train.py` | Gradient accumulation with DDP `no_sync`, `--nccl-timeout` (default 1800s), `--grad-clip-norm` CLI args, effective batch warning >512, DDP OOM diagnostics (rank, device, peak memory) |

### Architecture Analysis: Bottleneck Widening

The huge and giant presets override `d_intra` and `head_hidden` from their defaults (256) to avoid bottlenecks:

**Frame encoder bottleneck (`d_intra`):** Each game frame passes through intra-frame attention at dimension `d_intra` before being projected up to `d_model`. With the default d_intra=256 feeding into d_model=2048 (huge), the encoder compresses 8× — the backbone has 8× more representational capacity than the encoder can feed it. Widening to d_intra=512 reduces this to 4×.

**Prediction head bottleneck (`head_hidden`):** The d_model=2048 representation is compressed to `head_hidden` before final action predictions. At the default 256, that's another 8× compression at the output. Widening to head_hidden=512 (huge) / 768 (giant) keeps the ratio manageable.

Without these overrides, the huge model's extra transformer capacity would be bottlenecked at both ends — it couldn't absorb richer frame representations or express finer-grained predictions.

---

## Phase 2: 5090 Runs Launched

### Configuration

All runs use eff batch 512 (batch=64 × 8 GPUs), lr=5e-5, `--no-compile`, 250M max samples (3,906,250 steps). Log interval = 19,531 steps (~0.5% of training).

| Machine | Run Name | Context | Self-Inputs | Seed | Tests |
|---------|----------|---------|-------------|------|-------|
| C (`194.14.47.19:22874`) | `full-nsi-ctx256-seed42` | 256 frames | No | 42 | Baseline |
| E (`66.222.138.178:11335`) | `full-nsi-ctx256-seed43` | 256 frames | No | 43 | Seed variance vs C |
| F (`74.2.96.10:18619`) | `full-si-ctx60-seed42` | 60 frames | **Yes** (fixed flag) | 42 | Self-inputs control vs C |

### Launch Issues

Machine E required relaunch — stale GPU processes from prior runs caused worker OOM on first attempt. Resolved by killing zombie processes before relaunching.

### GPU Memory Usage

| Machine | Config | Memory (per GPU) | Utilization |
|---------|--------|-------------------|-------------|
| C | ctx=256, no-self-inputs | 18,278 MiB / 32,607 MiB (56%) | 94–100% |
| E | ctx=256, no-self-inputs | 13,920 MiB / 32,607 MiB (43%) | 80–87% |
| F | ctx=60, self-inputs | 4,473 MiB / 32,607 MiB (14%) | 67–83% |

Context length is the dominant memory factor: ctx=256 uses 3–4× more memory than ctx=60. The C vs E memory discrepancy (18.3 vs 13.9 GB for identical configs) may reflect different PyTorch memory allocation patterns across seeds or timing of the snapshot.

### Step 1 Metrics

| Metric | C (ctx=256, nsi, s42) | E (ctx=256, nsi, s43) | F (ctx=60, si, s42) |
|--------|---|---|---|
| total_loss | 6.862 | 6.686 | 6.746 |
| main_loss | 4.025 | 3.993 | 4.007 |
| btn_f1 | 7.0% | 4.3% | 4.2% |
| btn_precision | 3.7% | 2.3% | 2.2% |
| btn_recall | 81.5% | 32.4% | 35.7% |
| main_f1 | 0.3% | 0.5% | 0.4% |
| gnorm | 20.18 | 20.17 | 19.94 |
| lr | 5.00e-7 | 5.00e-7 | 5.00e-7 |

All runs showing expected random-init behavior: high btn_recall / low precision (predicting too many buttons), gnorm ~20 (will decrease during warmup), LR at start of warmup ramp. Machine F shows a `find_unused_parameters` warning (self-inputs encoder params used, but likely opponent input params unused since `no_opp_inputs=True` is the default).

---

## Phase 3: H200 Setup + Huge Model Launch

### Machine D Specs

- **GPUs**: 8× NVIDIA H200 (144 GB each)
- **Disk**: 2.0 TB
- **PyTorch**: Pre-installed with CUDA support
- **Location**: `50.145.48.107:12404`

### Setup Process

1. Deployed code via tarball (scp, no git/rsync on remote)
2. Ran `setup.sh` with `HF_REPO=erickfm/mimic-melee-500M` (200-shard, 505M frames, 761 GB)
3. Dataset is public — no HF token required

### Smoke Tests

**Single-GPU test** (1 GPU, 500 samples, huge model):
- Ran to completion (7 steps)
- Model confirmed at 621,072,746 parameters
- No OOM issues

**DDP test** (8 GPUs, 5000 samples, huge model):
- Ran to completion (78 steps)
- Effective batch size: 512 (64 × 8) confirmed
- Gradient norms decreased from ~90 to ~0.67 during warmup — model learning

### GPU Memory: Huge Model on H200

| | Predicted | Actual |
|---|---|---|
| Peak memory (per GPU) | ~21 GB | 52.8 GB |
| Memory utilization | 15% of 144 GB | 37% of 144 GB |

The 2.5× higher-than-predicted memory is likely due to DDP gradient buffers and optimizer states for 621M parameters. Still well within H200's 144 GB headroom.

### Full Run Launch

Launched via `BG=1 bash parallel.sh D` with config:
- Model: huge (621M params, d_model=2048, 12 layers, 16 heads)
- Context: 256 frames, no self-inputs, seed 42
- Batch: 64 per GPU (eff 512), lr=5e-5, --no-compile
- Run name: `full-nsi-ctx256-huge-seed42`
- wandb: [MIMIC/runs/899jcfps](https://wandb.ai/erickfm/MIMIC/runs/899jcfps)

Step 1 metrics: total_loss=6.898, gnorm=90.44 (4.5× higher than medium model's ~20, expected for 19× more parameters).

---

## Phase 4: Monitoring Status

### Initial Check (Step 1)

All 4 runs confirmed training with high GPU utilization. Step 1 completed on all machines. Log interval = 19,531 steps (max_steps × 0.005).

| Machine | GPUs | Run | GPU Util | Memory/GPU |
|---------|------|-----|----------|------------|
| C (5090) | 8× 32 GB | full-nsi-ctx256-seed42 | 100% | 18.3 GB (56%) |
| E (5090) | 8× 32 GB | full-nsi-ctx256-seed43 | 82% | 13.9 GB (43%) |
| F (5090) | 8× 32 GB | full-si-ctx60-seed42 | 72% | 4.5 GB (14%) |
| D (H200) | 8× 144 GB | full-nsi-ctx256-huge-seed42 | 100% | 52.8 GB (37%) |

### Mid-Session Check (~1% through)

| Machine | Step | btn_f1 | main_f1 | total_loss | gnorm | step/s |
|---------|------|--------|---------|------------|-------|--------|
| F (ctx=60, si) | 117,186 (3%) | **88.3%** | **34.7%** | 0.58 | 4.35 | 15.5 |
| D (ctx=256, huge) | 19,531 (0.5%) | 36.4% | 6.4% | 1.75 | 1.40 | 3.1 |
| C (ctx=256, nsi) | 39,062 (1%) | 29.4% | 5.6% | 1.62 | 1.19 | 5.1 |
| E (ctx=256, nsi) | 39,062 (1%) | 27.2% | 6.3% | 1.72 | 1.15 | 7.7 |

### Early Observations

1. **F (self-inputs) is dominant.** 88.3% btn_f1 at 3% of training, tracking the old successful `full-ddp-D-500M` run. Self-inputs + ctx=60 at eff-512 is the proven formula.

2. **D (huge model) leads C/E despite fewer steps.** 36.4% btn_f1 at step 19,531 vs C's 29.4% at step 39,062 — the 621M model is learning roughly 2× faster per step than the 32M model. Capacity scaling helps in the no-self-inputs regime.

3. **C vs E seed variance is tight.** 29.4% vs 27.2% btn_f1 at the same step — good reproducibility, ~2% noise.

4. **Throughput scales inversely with compute.** F (ctx=60, medium) at 15.5 step/s, C/E (ctx=256, medium) at 5–8 step/s, D (ctx=256, huge) at 3.1 step/s.

5. **ctx=256 no-self-inputs runs already past 27-29% btn_f1.** The eff-1024 no-self-inputs runs plateaued at ~40%. At only 1% through training, C/E are already at 27-29% — on track to surpass the plateau if the learning curve continues.

### wandb Links

- C: [n783gmpa](https://wandb.ai/erickfm/MIMIC/runs/n783gmpa)
- E: [5vpl2tec](https://wandb.ai/erickfm/MIMIC/runs/5vpl2tec)
- F: [rke803tw](https://wandb.ai/erickfm/MIMIC/runs/rke803tw)
- D: [899jcfps](https://wandb.ai/erickfm/MIMIC/runs/899jcfps)

---

## Planned Comparisons

| Comparison | Pair | Question |
|------------|------|----------|
| Seed variance | C vs E | How much do ctx=256 no-self-inputs runs vary by seed? |
| Context vs self-inputs | C vs F | Does ctx=256 no-self-inputs match ctx=60 self-inputs? (The key question — can long context compensate for removing controller feedback?) |
| Capacity scaling | C vs D | Does 19× more parameters (huge ~621M vs medium ~32M) help in the no-self-inputs ctx=256 regime? |

### What Success Looks Like

- **C/E above 40% btn_f1**: Long context helps no-self-inputs surpass the eff-1024 plateau (all prior no-self-inputs runs capped at ~40%)
- **C ≈ F**: Context window can substitute for self-controller feedback
- **D >> C**: The medium model is capacity-limited in this regime; scaling helps

### What Would Be Concerning

- **C/E ≈ 40%**: Long context doesn't help — no-self-inputs is fundamentally limited regardless of context
- **F >> C**: Self-inputs still dominant even with 4.3× more context; HAL's finding about controller feedback being critical holds
- **D ≈ C**: The bottleneck is in the task formulation, not model capacity
