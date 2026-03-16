# MIMIC: Melee Imitation Model for Input Cloning

MIMIC is an imitation-learning bot for Super Smash Bros. Melee. Given a
window of recent gameplay frames, it predicts the controller inputs a human
player would enter on the next frame. The model observes both players' full
game state (positions, actions, speeds, buttons, analog sticks, projectiles,
stage geometry) through Slippi replay data and learns to reproduce human
decision-making at 60 fps.

## Goal

Train a transformer to play Melee by imitating human inputs. The model
watches a sliding window of past frames and outputs a complete controller
state for the next frame: analog stick position, C-stick direction, trigger
pressures, and digital button presses. At inference time it connects to
Dolphin via libmelee and drives a controller in real time.

## Controller Mapping

The GameCube controller has five input groups. Each maps to a dedicated
prediction head in `model.py`:

| Physical Input | Model Head | Dims | Output | Loss | Notes |
|---|---|---|---|---|---|
| Main Stick (x, y) | `main_head` | 2 | Raw (clamp at inference) | MSE | Center 0.5; deadzone ~0.28, tilt < 0.65, smash > 0.8 |
| L Trigger (analog) | `L_head` | 1 | Raw (clamp at inference) | MSE | 0 = released, 1 = full press |
| R Trigger (analog) | `R_head` | 1 | Raw (clamp at inference) | MSE | 0 = released, 1 = full press |
| C-Stick | `cdir_head` | 5 | Logits | Focal CE | Neutral / Up / Down / Left / Right |
| Digital Buttons (12) | `btn_head` | 12 | Logits | BCE | A, B, X, Y, Z, L, R, Start, D-pad x4 |

Regression heads output unbounded values during training (no Sigmoid).
MSE naturally pushes outputs into [0, 1] since all targets live there.
At inference, outputs are clamped to [0, 1].

## Architecture

```
Slippi Frame ──► HybridFrameEncoder (16 entity tokens + attention) ──► 768-d per-frame vector
                                                                            │
60-frame window ──► + Learned Positional Embeddings ────────────────────────┘
                         │
                    4× Pre-Norm Causal Transformer Blocks
                         │
                    All T hidden states (autoregressive)
                         │
              ┌──────────┼──────────┬───────────┐
          main_xy    L / R val   c_dir_5way   btn_12way
```

### Phase 1 -- Frame Understanding (`frame_encoder.py`)

Each frame is decomposed into 16 entity-level tokens via the **hybrid16**
encoder: game state, self/opponent identity, action, state, and input
tokens, plus Nana and projectile groups. Each token mixes its constituent
categorical embeddings and numeric MLP encodings into a 256-d vector. A
2-layer self-attention block with a learned [CLS] query token pools these
16 tokens into a single 256-d frame summary, then projects up to 768-d.

Why 16 entity tokens: a middle ground between the original 55 fine-grained
tokens (expensive 55×55 attention) and a flat concatenation (no cross-group
interaction). 16 tokens let the model learn which entities and feature
types are most relevant to each other (e.g., opponent position + opponent
action = threat assessment) while keeping intra-frame attention cheap (16×16).

### Phase 2 -- Temporal Patterns (`model.py`)

A 4-layer pre-norm causal transformer operates over the 60-frame window
(1 second at 60 fps). Causal masking ensures position T can only attend to
positions 0..T. This is where the model learns sequential patterns:
approach sequences, attack commitments, recovery trajectories, combo
follow-ups. 60 frames captures most neutral interactions and short combos.

The backbone supports multiple positional encoding schemes (`--pos-enc`):
learned (default), RoPE, sinusoidal, and ALiBi. Flash Attention via
`F.scaled_dot_product_attention(is_causal=True)` is used for speed;
`torch.compile` fuses the full forward pass.

### Phase 3 -- Action Prediction (`model.py` PredictionHeads)

All T hidden states are fed into five independent prediction heads, each
with a 256-d hidden layer and GELU activation. Every position in the
sequence predicts the controller state R frames ahead, providing T×
more training signal than predicting from only the last position. This is
autoregressive training analogous to GPT-style next-token prediction
where the "token" is a full controller state.

### Weight Initialization

GPT-2-style initialization: all Linear weights are `N(0, 0.02)`, biases
are zeroed, and Embedding weights are `N(0, 0.02)`. Residual output
projections (attention out_proj and feedforward final linear) are scaled
by `1/sqrt(2 * num_layers)` for stable gradient flow through deep
residual chains.

### Reaction Delay

`REACTION_DELAY = 1` means "given game state through frame T, predict the
input at frame T+1." This is NOT modeling human reaction time. Frame T
already reflects the result of frame T's input, so a delay of 1 captures
the statistical relationship between observed states and the action that
follows.

## Design Decisions

### L/R Triggers -- Kept Separate

L and R are mechanically identical in Melee (both shield, tech, L-cancel,
wavedash). However:

- Players develop habits (e.g., L for shield, R for wavedash). The training
  data encodes these per-player patterns.
- The model naturally learns near-zero predictions for whichever trigger a
  player doesn't use.
- `BUTTON_L` / `BUTTON_R` digital clicks are also in the button head and
  correlate with analog values approaching 1.0. This redundancy is harmless
  and lets the model learn the full trigger behavior (analog pressure +
  digital click).
- At inference we could post-process to a single trigger if desired.

### X/Y Buttons -- Kept Separate

Both are jump, but they occupy 2 of 12 dimensions in the multi-label BCE
button head. The model learns per-player preferences naturally. No special
handling needed.

### C-Stick 5-Way Classification

Melee functionally treats the c-stick as 5 discrete states (neutral, up,
down, left, right). `features.encode_cstick_dir` uses a deadzone and
cardinal-dominance rule that matches the game engine's behavior. Diagonal
inputs are rare and resolve to one cardinal direction.

### Main Stick 2D Regression

Raw MSE over [0,1] targets is the simplest approach. Melee has
discontinuous stick zones (deadzone ~0.28, tilt ~0.3-0.65, smash > 0.8)
so MSE treats all positional errors equally regardless of zone boundaries.
This is adequate for initial R&D; a zone-aware loss or discretization is
a future improvement.

### No Sigmoid on Regression Heads

Trigger and stick targets are overwhelmingly bimodal (0.0 or 1.0) with
rare intermediate values. With Sigmoid, predicting 0.0 requires logits
at -inf and 1.0 requires +inf -- exactly the most common targets sit
where gradients vanish. Without Sigmoid, the model outputs 0.0 or 1.0
directly with clean, proportional MSE gradients. Under BF16, raw values
near 0 and 1 are perfectly representable.

### Dead Button Dimensions

BUTTON_START and D-pad (5 of 12 button outputs) are almost never pressed
during gameplay. The model quickly learns ~0 for these. Not harmful but
wastes a small amount of head capacity.

## Future Improvements

- **Combine Huber + no-opp-inputs + ctx-180**: The best stick loss (Huber)
  hasn't been tested with the best feature config (no-opp-inputs) at the
  best context length (180 frames). This combination should push metrics
  further.
- **More data**: All experiments used 2M samples. Scaling to the full
  dataset (~100M+ frames) should help with the cdir_active ceiling.
- **Better c-stick modeling**: cdir_active_acc at 71.7% is the weakest
  link. Consider focal loss for c-stick (rare active frames), or a
  hierarchical predict-then-direction approach.
- **Loss reweighting**: Task losses have different magnitudes (MSE on
  near-zero L/R is tiny vs Focal CE on c-stick). Normalizing gradient
  contributions across heads could improve joint learning.
- **Main stick zone-aware loss**: Discretize into Melee-relevant zones or
  use a mixture distribution that captures the multi-modal nature of stick
  positions (center cluster, cardinal extremes).
- **Multi-step prediction**: Predicting frames T+1 through T+K
  simultaneously could help the model learn action planning and committal
  sequences.

## Model Presets

Pass `--model <preset>` to `train.py` for scaling experiments:

| Preset | d_model | Heads | Layers | FF dim | ~Params |
|---|---|---|---|---|---|
| `tiny` | 256 | 4 | 4 | 1024 | ~3.5M |
| `small` | 512 | 8 | 4 | 2048 | ~16M |
| `medium` (default) | 768 | 8 | 4 | 3072 | ~32M |
| `base` | 1024 | 8 | 4 | 4096 | ~55M |
| `shallow` | 1024 | 8 | 2 | 4096 | ~27M |

## Default Configuration

| Setting | Value |
|---|---|
| Window length (W) | 60 frames |
| Reaction delay (R) | 1 frame |
| Model width (d_model) | 768 (medium preset) |
| Transformer layers | 4 |
| Attention heads | 8 |
| Feedforward dim | 3072 (4x expansion) |
| Frame encoder | hybrid16 (16 entity tokens) |
| Intra-frame width | 256 |
| Batch size | 128 |
| Learning rate | 8e-4 |
| Training samples | 2M (~1 hr on 4090) |
| AMP dtype | bfloat16 |
| Parameters | ~32M (medium + hybrid16) |
| GPU tested | RTX 4090 24 GB |

## Project Structure

```
.
├── train.py            # Training loop & checkpointing
├── preprocess.py       # Compute metadata (norm stats, categorical maps, file index)
├── inference.py        # Real-time single-window inference via libmelee
├── model.py            # FramePredictor + Transformer heads
├── frame_encoder.py    # Intra-frame cross-attention encoder
├── features.py         # Shared feature engineering (columns, preprocessing, tensors)
├── dataset.py          # Dataset classes (raw parquet debug + streaming)
├── cat_maps.py         # Melee enum → dense index maps
├── setup.sh                     # One-command setup for fresh machines
├── upload_dataset.py            # Package parquets into tar shards and upload to HF
├── generate_wavedash_replay.py  # Synthetic wavedash data generator (libmelee)
├── closedloop_debug.py          # Frame-by-frame tensor comparison (train vs inference)
├── diagnose.py                  # Offline inference diagnostic tool
├── checkpoints/                 # Saved *.pt files
├── docs/                        # Research notes and session logs
└── data/                        # Slippi parquet files + source replays
    └── source_replays/          # Original .slp files for reproduction
```

## Quick Start (New Machine)

Spin up any machine with an NVIDIA GPU, clone the repo, and run:

```bash
bash setup.sh --run
```

This installs deps, downloads the dataset from HuggingFace, runs
preprocessing if needed, and starts a 1-epoch training run.

For scaling experiments:

```bash
bash setup.sh --run --model small     # 14M param model
bash setup.sh --run --model tiny      # 3.5M param model
```

The full dataset is hosted at [erickfm/frame-melee](https://huggingface.co/datasets/erickfm/frame-melee) (~94k replays, 86 GB as tar shards).
A smaller 2k-replay subset is at [erickfm/frame-melee-subset](https://huggingface.co/datasets/erickfm/frame-melee-subset) for quick experiments.

## Manual Setup

Install dependencies:

```bash
pip install torch numpy pandas pyarrow wandb huggingface_hub melee==0.45.1 typing-extensions
```

## Data

Two datasets are hosted on HuggingFace:

| Dataset | Replays | Parquets | Size | Use |
|---|---|---|---|---|
| [erickfm/frame-melee](https://huggingface.co/datasets/erickfm/frame-melee) | ~94k | 188k | ~86 GB (tar shards) | Full training |
| [erickfm/frame-melee-subset](https://huggingface.co/datasets/erickfm/frame-melee-subset) | ~1k | 2k | ~780 MB | Quick experiments |

The full dataset is stored as tar shards (`shard_000.tar` ... `shard_NNN.tar`) for efficient
download. `setup.sh` automatically extracts them. Both datasets include precomputed metadata:

| File | Description |
|---|---|
| `cat_maps.json` | Dynamic categorical mappings (ports, costumes, projectile subtypes) |
| `norm_stats.json` | Per-column mean/std for feature normalization |
| `file_index.json` | Frame counts per file (train/val split + length estimation) |

Parquet files are generated from Slippi `.slp` replays using
[slippi-frame-extractor](https://github.com/erickfm/slippi-frame-extractor).

Download the full dataset:

```bash
bash setup.sh
```

Or manually:

```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('erickfm/frame-melee', repo_type='dataset', local_dir='data/full')
"
# Extract tar shards
for f in data/full/shard_*.tar; do tar xf "$f" -C data/full && rm "$f"; done
```

For the smaller subset:

```bash
bash setup.sh --repo erickfm/frame-melee-subset --data-dir data/subset
```

To regenerate metadata from raw parquets:

```bash
python3 preprocess.py --data-dir data/full
```

## Training

Point `--data-dir` at a parquet directory (auto-detects metadata for
streaming, falls back to slow in-memory loading):

```bash
# Default: medium model ~32M params, hybrid16 encoder, 2M samples (~1 hr)
python3 train.py

# Full epoch on the full dataset
python3 train.py --max-samples 0 --epochs 1

# Fixed number of steps
python3 train.py --max-steps 80000

# Train with a different model size
python3 train.py --model small
python3 train.py --model base

# Quick experiment on the 2k subset
python3 train.py --data-dir data/subset --model tiny --max-samples 0 --epochs 1

# Resume from checkpoint
python3 train.py --resume checkpoints/step_XXXXXX.pt

# Experiment with loss functions
python3 train.py --stick-loss huber            # Robust stick regression
python3 train.py --stick-loss discrete         # Discretised stick (32x32 bins)
python3 train.py --btn-loss focal              # Focal BCE for buttons

# Positional encoding variants
python3 train.py --pos-enc rope                # Rotary position embeddings

# Disable torch.compile for debugging
python3 train.py --no-compile --debug
```

Logging, validation, and checkpoint intervals auto-scale as a percentage
of total steps (~0.5% log, ~5% val/checkpoint, ~1% warmup).

## Inference

Run the bot against a CPU in Dolphin:

```bash
python3 inference.py \
  --checkpoint checkpoints/noi_ctx180_65k_machC.pt \
  --dolphin-path ~/.config/Slippi\ Launcher/netplay/Slippi_Online-x86_64.AppImage \
  --iso-path ~/Downloads/Super\ Smash\ Bros.\ Melee\ \(USA\)\ \(En,Ja\)\ \(Rev\ 2\).iso
```

If you installed [Slippi Launcher](https://slippi.gg/) normally, the Dolphin executable lives at:

```
~/.config/Slippi Launcher/netplay/Slippi_Online-x86_64.AppImage
```

You can also set environment variables to avoid passing paths every time:

```bash
export DOLPHIN_PATH="$HOME/.config/Slippi Launcher/netplay/Slippi_Online-x86_64.AppImage"
export ISO_PATH="$HOME/Downloads/Super Smash Bros. Melee (USA) (En,Ja) (Rev 2).iso"
python3 inference.py --checkpoint checkpoints/noi_ctx180_65k_machC.pt
```

**Stopping:** Press `Ctrl+C` to stop inference. You will need to **close
Dolphin manually** after the script exits -- the Slippi AppImage runs in its
own process group so the script cannot kill it automatically. You can close
the Dolphin window normally, or run:

```bash
pkill -f AppRun.wrapped
```

Options:

| Flag | Description |
|------|-------------|
| `--checkpoint` | Path to a `.pt` checkpoint (auto-discovers latest if omitted) |
| `--dolphin-path` | Path to Slippi Dolphin AppImage (or set `DOLPHIN_PATH`) |
| `--iso-path` | Path to Melee ISO (or set `ISO_PATH`) |
| `--data-dir` | Directory containing `cat_maps.json`, `norm_stats.json`, `stick_clusters.json` |
| `--cpu-level` | CPU opponent level, 1-9 (default: 7). Use 0 for human/no opponent |
| `--character` | Bot character (default: FALCO) |
| `--cpu-character` | Opponent character (default: FALCO) |
| `--stage` | Stage (default: FINAL_DESTINATION) |
| `--deterministic` | Use threshold-based button decisions instead of stochastic sampling |
| `--btn-threshold` | Sigmoid threshold for button presses (default: 0.2) |
| `--temperature` | Temperature for stick/shoulder cluster sampling (default: 1.0 = argmax) |
| `--no-pred-feedback` | Use game controller readback instead of model predictions for self-inputs |
| `--diag-log-all` | Save every raw row dict to pickle for offline debugging |
| `--debug` | Verbose frame-by-frame logging |

### Architecture Notes

**Pipe synchronization**: The console is created with `blocking_input=True`, which
makes Dolphin wait for a pipe FLUSH before advancing each frame. This ensures
every controller input is processed, at the cost of capping the game's framerate
to our inference speed. Without this, Dolphin runs at 60fps and most inputs are
missed because inference takes longer than one frame period.

**Controller protocol**: Following [HAL's pattern](https://github.com/ericyuegu/hal),
every button is explicitly pressed or released on every frame (no `release_all()`),
shoulder analogs are always sent, and `flush()` is called exactly once per frame.

**Performance**: Per-frame preprocessing uses a pure-Python fast path (~0.3ms)
instead of pandas (~57ms). Combined with `torch.compile` and a rolling tensor
cache, total per-frame inference is ~3.5ms (model: 2.2ms, preprocessing: 0.3ms,
tensor stacking: 0.6ms), well under the 16.7ms budget for 60fps.
