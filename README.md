# MIMIC: Melee Imitation Model for Input Cloning

> **For agent/developer orientation, see [CLAUDE.md](CLAUDE.md).** It covers
> naming gotchas, stats file pitfalls, data directories, and common mistakes.

MIMIC is a behavior-cloning bot for Super Smash Bros. Melee. It watches
human replays and learns to predict controller inputs from game state. At
inference it drives a controller through Dolphin (the GameCube emulator) via
libmelee at 60 fps.

The reference implementation is [HAL](https://github.com/ericyuegu/hal) by
Eric Gu. MIMIC reproduces HAL's architecture and training pipeline, then
scales to more data. HAL's best checkpoint 4-stocks a level 9 CPU.

## Current Architecture (HAL-Matching)

Using `--model hal`, MIMIC matches HAL's GPTv5Controller (~19.95M params):

```
Slippi Frame ──► HALFlatEncoder (Linear 164→512) ──► 512-d per-frame vector
                                                          │
256-frame window ──► + Relative Position Encoding ────────┘
                         │
                    6× Pre-Norm Causal Transformer Blocks (512-d, 8 heads)
                         │
                    Autoregressive Output Heads (with detach)
                         │
              ┌──────────┼──────────┬───────────┐
           shoulder(3) c_stick(9) main_stick(37) buttons(5)
```

### Controller Mapping

| Physical Input | Head | Classes | Method |
|---|---|---|---|
| Main Stick (x, y) | `main_stick` | 37 | Hand-designed cluster centers |
| C-Stick | `c_stick` | 9 | Neutral + 4 cardinal + 4 diagonal |
| Shoulder (max L/R) | `shoulder` | 3 | Centers [0.0, 0.4, 1.0] |
| Buttons | `buttons` | 5 | Single-label: A, B, Jump(X\|Y), Z, None |

All heads use plain cross-entropy loss. Heads are chained autoregressively
(shoulder → c_stick → main_stick → buttons) with detached conditioning.

### Input Features

9 numeric features per player (ego + opponent = 18 total):
`percent, stock, facing, invulnerable, jumps_left, on_ground, shield_strength, position_x, position_y`

Plus categorical embeddings: stage(4d), 2x character(12d), 2x action(32d).
Plus controller state from previous frame (54d one-hot).
Total input: 164 dimensions → projected to 512.

## Training

```bash
# Multi-GPU (8x)
torchrun --nproc_per_node=8 train.py \
  --model hal --encoder hal_flat \
  --hal-mode --hal-minimal-features --hal-controller-encoding \
  --stick-clusters hal37 --plain-ce --no-amp \
  --lr 3e-4 --batch-size 64 \
  --max-samples 16777216 \
  --data-dir data/fox_hal_local \
  --controller-offset --self-inputs \
  --no-compile --run-name <name> \
  --nccl-timeout 7200 --no-warmup --cosine-min-lr 1e-6

# Single GPU (use grad accumulation to match effective batch 512)
python3 train.py \
  --model hal --encoder hal_flat \
  --hal-mode --hal-minimal-features --hal-controller-encoding \
  --stick-clusters hal37 --plain-ce --no-amp \
  --lr 3e-4 --batch-size 64 --grad-accum-steps 8 \
  --max-samples 16777216 \
  --data-dir data/fox_hal_local \
  --controller-offset --self-inputs \
  --no-compile --run-name <name> \
  --no-warmup --cosine-min-lr 1e-6
```

Key settings: AdamW, no warmup, CosineAnnealingLR with eta_min=1e-6,
dropout 0.2, weight decay 0.01, gradient clip 1.0, sequence length 256,
FP32 (no AMP).

## Inference

```bash
# Run HAL checkpoint
python3 tools/run_hal_model.py \
  --checkpoint /home/erick/projects/hal/checkpoints/000005242880.pt \
  --dolphin-path /path/to/dolphin-emu \
  --iso-path /path/to/melee.iso \
  --character FOX --cpu-character FOX --cpu-level 9

# Run MIMIC checkpoint
python3 tools/run_mimic_via_hal_loop.py \
  --checkpoint checkpoints/<name>.pt \
  --dolphin-path /path/to/dolphin-emu \
  --iso-path /path/to/melee.iso
```

**Do not run inference while training on the same GPU.** Frame drops from
GPU contention make gameplay look broken.

## Data

Training uses HAL-normalized tensor shards in `data/fox_hal_local/` (7,600
Fox games, 70M frames). Shards are built from Slippi .slp replays using
`tools/slp_to_shards.py` with `--hal-norm`.

Raw replays: [erickfm/slippi-public-dataset-v3.7](https://huggingface.co/datasets/erickfm/slippi-public-dataset-v3.7)
(95K replays, 45K Fox games, organized by character).

## Project Structure

```
.
├── CLAUDE.md               # Agent/developer orientation (read this first)
├── train.py                # Training loop (DDP, grad accum, HAL mode)
├── inference.py            # Real-time inference via libmelee
├── eval.py                 # Offline validation metrics
│
├── mimic/                  # Core library
│   ├── model.py            # FramePredictor, ModelConfig, HAL presets
│   ├── frame_encoder.py    # HALFlatEncoder + other encoder variants
│   ├── features.py         # Cluster centers, controller encoding, normalization
│   ├── dataset.py          # StreamingMeleeDataset (per-game tensor shards)
│   └── cat_maps.py         # Categorical mappings
│
├── tools/                  # Data pipeline and diagnostics
│   ├── slp_to_shards.py    # .slp replays → tensor shards
│   ├── run_hal_model.py    # HAL checkpoint inference (our reimplementation)
│   ├── run_mimic_via_hal_loop.py  # MIMIC checkpoint inference
│   ├── validate_checkpoint.py     # Compare checkpoints on val data
│   ├── verify_hal_pipeline.py     # Validate preprocessing vs HAL
│   ├── diagnose.py         # Pipeline debugging
│   └── gameplay_health.py  # Inference log analysis
│
├── checkpoints/            # Saved model checkpoints
├── data/fox_hal_local/     # Active training data (HAL-normalized)
├── docs/                   # Current research notes
└── docs/archive/           # Historical research notes
```

## Model Presets

| Preset | d_model | Heads | Layers | Seq Len | Pos Enc | ~Params |
|---|---|---|---|---|---|---|
| `hal` | 512 | 8 | 6 | 256 | RelPos | ~20M |
| `tiny` | 256 | 4 | 4 | 60 | RoPE | ~3.5M |
| `small` | 512 | 8 | 4 | 60 | RoPE | ~16M |
| `medium` | 768 | 8 | 4 | 60 | RoPE | ~32M |

## Dependencies

```bash
pip install torch numpy wandb huggingface_hub melee==0.45.1
```

## Research Notes

Historical research journal in `docs/archive/` (2026-03-14 through 2026-04-06)
and current notes in `docs/` (2026-04-07+). These document the project's
evolution but may contain claims that were later proven wrong. See CLAUDE.md
for what's currently verified as correct.
