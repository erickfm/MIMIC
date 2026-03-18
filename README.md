# MIMIC: Melee Imitation Model for Input Cloning

MIMIC is an imitation-learning bot for Super Smash Bros. Melee. Given a
window of recent gameplay frames, it predicts the controller inputs a human
player would enter on the next frame. The model observes game state
(positions, actions, speeds, projectiles, stage geometry) through Slippi
replay data and learns to reproduce human decision-making at 60 fps.
Self-controller inputs are excluded by default — the model learns purely
from game state, eliminating train/inference distribution shift.

## Goal

Train a transformer to play Melee by imitating human inputs. The model
watches a sliding window of past frames and outputs a complete controller
state for the next frame: analog stick position, C-stick direction, trigger
pressures, and digital button presses. At inference time it connects to
Dolphin via libmelee and drives a controller in real time.

## Controller Mapping

The GameCube controller has five input groups. Each maps to a dedicated
prediction head in `mimic/model.py`. Heads are chained autoregressively
(L → R → cdir → main → buttons) so each head conditions on previous
predictions:

| Physical Input | Model Head | Dims | Output | Loss | Notes |
|---|---|---|---|---|---|
| Main Stick (x, y) | `main_head` | 30 clusters | Logits | Focal CE | K-means cluster classification over stick positions |
| L Trigger (analog) | `L_head` | 4 bins | Logits | Focal CE | Discretized into shoulder bins via k-means |
| R Trigger (analog) | `R_head` | 4 bins | Logits | Focal CE | Discretized into shoulder bins via k-means |
| C-Stick | `cdir_head` | 5 | Logits | Focal CE | Neutral / Up / Down / Left / Right |
| Digital Buttons (12) | `btn_head` | 12 | Logits | BCE | A, B, X, Y, Z, L, R, Start, D-pad x4 |

## Architecture

```
Slippi Frame ──► HybridFrameEncoder (16 entity tokens + attention) ──► 768-d per-frame vector
                                                                            │
60-frame window ──► + RoPE Positional Encoding ────────────────────────────┘
                         │
                    4× Pre-Norm Causal Transformer Blocks
                         │
                    All T hidden states (autoregressive chain)
                         │
              ┌──────────┼──────────┬───────────┐
          main_cluster  L/R bin   c_dir_5way   btn_12way
```

### Phase 1 -- Frame Understanding (`mimic/frame_encoder.py`)

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

### Phase 2 -- Temporal Patterns (`mimic/model.py`)

A 4-layer pre-norm causal transformer operates over the 60-frame window
(1 second at 60 fps). Causal masking ensures position T can only attend to
positions 0..T. This is where the model learns sequential patterns:
approach sequences, attack commitments, recovery trajectories, combo
follow-ups. 60 frames captures most neutral interactions and short combos.

The backbone uses RoPE (Rotary Position Embeddings) by default, encoding
relative position directly into attention via rotation matrices. Flash
Attention via `F.scaled_dot_product_attention(is_causal=True)` is used for
speed; `torch.compile` fuses the full forward pass.

### Phase 3 -- Action Prediction (`mimic/model.py` PredictionHeads)

All T hidden states are fed through an autoregressive chain of prediction
heads. The chain order is L → R → cdir → main → buttons, where each
subsequent head receives the previous head's predictions (detached) as
conditioning input. Every position in the sequence predicts the controller
state R frames ahead, providing T× more training signal than predicting
from only the last position.

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
down, left, right). `mimic.features.encode_cstick_dir` uses a deadzone and
cardinal-dominance rule that matches the game engine's behavior. Diagonal
inputs are rare and resolve to one cardinal direction.

### Main Stick Cluster Classification

Raw MSE regression on stick (x, y) causes base-rate collapse: the model
learns to predict neutral (0.5, 0.5) on every frame because that minimizes
squared error against the peaked distribution of stick positions. Discrete
cluster classification with focal loss treats each meaningful stick region
as a separate class, giving the model proper gradients on rare but critical
inputs (smash attacks, wavedash angles, etc.). 30 clusters are determined
via k-means++ on the training data.

### Dead Button Dimensions

BUTTON_START and D-pad (5 of 12 button outputs) are almost never pressed
during gameplay. The model quickly learns ~0 for these. Not harmful but
wastes a small amount of head capacity.

## Future Improvements

- **More data**: Scaling to the full dataset (~100M+ frames) should improve
  generalization beyond single-behavior tasks.
- **Better c-stick modeling**: cdir_active_acc has room to improve.
  Hierarchical predict-then-direction or sampling-based approaches may help.
- **Multi-step prediction**: Predicting frames T+1 through T+K
  simultaneously could help the model learn action planning and committal
  sequences.
- **Opponent modeling**: Currently opponent and self controller inputs are
  both excluded by default. Re-introducing them with a better conditioning
  mechanism could improve reactive play.

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
| Positional encoding | RoPE |
| Batch size | 256 |
| Learning rate | 5e-5 |
| Warmup | 5% of total steps |
| Dropout | 0.1 |
| AMP dtype | bfloat16 |
| Parameters | ~32M (medium + hybrid16) |
| GPU tested | RTX 4090 24 GB |

## Project Structure

```
.
├── train.py                # Training loop & checkpointing
├── inference.py            # Real-time single-window inference via libmelee
├── eval.py                 # Validation metrics on a checkpoint
├── parallel.sh             # Multi-GPU DDP launcher (SSH + torchrun)
├── setup.sh                # One-command setup for fresh machines
│
├── mimic/                  # Core library
│   ├── model.py            # FramePredictor + Transformer heads
│   ├── frame_encoder.py    # Intra-frame cross-attention encoder
│   ├── features.py         # Feature engineering (columns, preprocessing, tensors)
│   ├── dataset.py          # Streaming tensor shard dataset for training
│   └── cat_maps.py         # Melee enum → dense index maps
│
├── tools/                  # Standalone scripts & diagnostics
│   ├── upload_dataset.py   # Pretokenize parquets → tensor shards + upload to HF
│   ├── tensorize.py        # Pre-window shards for max local throughput
│   ├── build_clusters.py   # K-means stick/shoulder cluster centers
│   ├── generate_wavedash_replay.py  # Synthetic wavedash data generator
│   └── diagnose.py         # Pipeline debug (train vs inference tensor comparison)
│
├── checkpoints/            # Saved *.pt files
├── docs/                   # Research notes and session logs
└── data/                   # Pretokenized tensor shards + metadata
```

## Quick Start (New Machine)

Spin up any machine with an NVIDIA GPU, clone the repo, and run:

```bash
bash setup.sh --run
```

This installs deps, downloads pretokenized tensor shards from HuggingFace,
and starts a training run. Data is ready to train immediately -- no
preprocessing steps needed.

For scaling experiments:

```bash
bash setup.sh --run --model small     # 14M param model
bash setup.sh --run --model tiny      # 3.5M param model
```

The full dataset is hosted at [erickfm/mimic-melee](https://huggingface.co/datasets/erickfm/mimic-melee) (~94k replays, pretokenized tensor shards).
A smaller subset is at [erickfm/mimic-melee-subset](https://huggingface.co/datasets/erickfm/mimic-melee-subset) for quick experiments.

## Manual Setup

Install dependencies:

```bash
pip install torch numpy pandas pyarrow wandb huggingface_hub melee==0.45.1 typing-extensions
```

## Data

Three datasets are hosted on HuggingFace as pretokenized tensor shards:

| Dataset | Replays | Games | Size | Use |
|---|---|---|---|---|
| [erickfm/mimic-melee](https://huggingface.co/datasets/erickfm/mimic-melee) | ~94k | 188,416 | 2.59 TB | Full training |
| [erickfm/mimic-melee-subset](https://huggingface.co/datasets/erickfm/mimic-melee-subset) | ~1k | 2,000 | 26.7 GB | Quick experiments |
| [erickfm/mimic-melee-wavedash](https://huggingface.co/datasets/erickfm/mimic-melee-wavedash) | 1 (synthetic) | 57,480 windows | 4.93 GB | Wavedash fine-tuning |

Game counts are 2x replay counts because each replay is tensorized from both
players' perspectives.

Each dataset contains `.pt` shard files with pretokenized game data (all
preprocessing, normalization, and target building done ahead of time),
plus metadata:

| File | Description |
|---|---|
| `tensor_manifest.json` | Shard list, game counts, train/val split |
| `norm_stats.json` | Per-column mean/std for feature normalization |
| `cat_maps.json` | Dynamic categorical mappings (ports, costumes, projectile subtypes) |
| `stick_clusters.json` | Stick position and shoulder trigger cluster centers |

Download the full dataset:

```bash
bash setup.sh
```

Or manually:

```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('erickfm/mimic-melee', repo_type='dataset', local_dir='data/full')
"
```

For the smaller subset:

```bash
bash setup.sh --repo erickfm/mimic-melee-subset --data-dir data/subset
```

Source parquet files are generated from Slippi `.slp` replays using
[slippi-frame-extractor](https://github.com/erickfm/slippi-frame-extractor).

## Training

Point `--data-dir` at a directory containing tensor shards
(`tensor_manifest.json` or `tensor_meta.json`):

```bash
# Default: medium model ~32M params, hybrid16 encoder
python3 train.py

# Full epoch on the full dataset
python3 train.py --max-samples 0 --epochs 1

# Fixed number of steps
python3 train.py --max-steps 80000

# Train with a different model size
python3 train.py --model small
python3 train.py --model base

# Quick experiment on the subset
python3 train.py --data-dir data/subset --model tiny --max-samples 0 --epochs 1

# Resume from checkpoint
python3 train.py --resume checkpoints/step_XXXXXX.pt

# Positional encoding variants
python3 train.py --pos-enc learned        # Learned positional embeddings

# Disable torch.compile for debugging
python3 train.py --no-compile --debug
```

Logging, validation, and checkpoint intervals auto-scale as a percentage
of total steps (~0.5% log, ~5% val/checkpoint, ~5% warmup).

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
| `--self-inputs` | Include self controller inputs (excluded by default) |
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
