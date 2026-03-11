# FRAME: Fixed-step Realtime Action Matrix Estimator

FRAME is an imitation-learning bot for Super Smash Bros. Melee. Given a
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

| Physical Input | Model Head | Dims | Activation | Loss | Notes |
|---|---|---|---|---|---|
| Main Stick (x, y) | `main_head` | 2 | Sigmoid [0,1] | MSE | Center 0.5; deadzone ~0.28, tilt < 0.65, smash > 0.8 |
| L Trigger (analog) | `L_head` | 1 | Sigmoid [0,1] | MSE | 0 = released, 1 = full press |
| R Trigger (analog) | `R_head` | 1 | Sigmoid [0,1] | MSE | 0 = released, 1 = full press |
| C-Stick | `cdir_head` | 5 | Softmax | Focal CE | Neutral / Up / Down / Left / Right |
| Digital Buttons (12) | `btn_head` | 12 | Sigmoid | BCE | A, B, X, Y, Z, L, R, Start, D-pad x4 |

## Architecture

```
Slippi Frame ──► FrameEncoder (intra-frame attention) ──► 1024-d per-frame vector
                                                              │
60-frame window ──► + Learned Positional Embeddings ──────────┘
                         │
                    4× Pre-Norm Causal Transformer Blocks
                         │
                    Last hidden state h_T
                         │
              ┌──────────┼──────────┬───────────┐
          main_xy    L / R val   c_dir_5way   btn_12way
```

### Phase 1 -- Frame Understanding (`frame_encoder.py`)

Each frame is decomposed into 40+ heterogeneous feature-group tokens:
categorical embeddings (stage, characters, actions, costumes, c-stick
directions, projectile types) and numeric MLP encodings (positions, speeds,
ECBs, shield, buttons, flags, analog values). A 2-layer self-attention
block with a learned [CLS] query token pools these into a single 256-d
frame summary, then projects up to 1024-d.

Why attention over flat concat: the model can learn which feature groups
are most relevant to each other within a single frame (e.g., opponent
position + opponent action = threat assessment) rather than relying on a
fixed concatenation order.

### Phase 2 -- Temporal Patterns (`model.py`)

A 4-layer pre-norm causal transformer operates over the 60-frame window
(1 second at 60 fps). Causal masking ensures position T can only attend to
positions 0..T. This is where the model learns sequential patterns:
approach sequences, attack commitments, recovery trajectories, combo
follow-ups. 60 frames captures most neutral interactions and short combos.

Flash Attention via `F.scaled_dot_product_attention(is_causal=True)` is
used for speed; `torch.compile` fuses the full forward pass.

### Phase 3 -- Action Prediction (`model.py` PredictionHeads)

The last hidden state h_T (which has attended to all 60 prior frames) feeds
into five independent prediction heads, each with a 256-d hidden layer.
This is standard GPT-style next-token prediction where the "token" is a
full controller state.

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

Sigmoid + MSE over [0,1] is the simplest approach. Melee has discontinuous
stick zones (deadzone ~0.28, tilt ~0.3-0.65, smash > 0.8) so MSE treats
all positional errors equally regardless of zone boundaries. This is
adequate for initial R&D; a zone-aware loss or discretization is a future
improvement.

### Dead Button Dimensions

BUTTON_START and D-pad (5 of 12 button outputs) are almost never pressed
during gameplay. The model quickly learns ~0 for these. Not harmful but
wastes a small amount of head capacity.

## Future Improvements

- **Loss reweighting**: Task losses have different magnitudes (MSE on
  near-zero L/R is tiny vs Focal CE on c-stick). Normalizing gradient
  contributions across heads could improve joint learning.
- **Main stick zone-aware loss**: Discretize into Melee-relevant zones or
  use a mixture distribution that captures the multi-modal nature of stick
  positions (center cluster, cardinal extremes).
- **Dead button masking**: Zero out loss for BUTTON_START and D-pad to
  reduce gradient noise.
- **Longer context windows**: 60 frames covers immediate interactions;
  120-180 frames would capture more strategic context (edge-guarding
  sequences, respawn patterns).
- **Multi-step prediction**: Predicting frames T+1 through T+K
  simultaneously could help the model learn action planning and committal
  sequences.

## Default Configuration

| Setting | Value |
|---|---|
| Window length (W) | 60 frames |
| Reaction delay (R) | 1 frame |
| Model width (d_model) | 1024 |
| Transformer layers | 4 |
| Attention heads | 8 |
| Feedforward dim | 2048 |
| Intra-frame width | 256 |

## Project Structure

```
.
├── train.py            # Training loop & checkpointing
├── preprocess.py       # Compute metadata (norm stats, categorical maps, file index)
├── inference.py        # Real-time single-window inference
├── model.py            # FramePredictor + Transformer heads
├── frame_encoder.py    # Intra-frame cross-attention encoder
├── features.py         # Shared feature engineering (columns, preprocessing, tensors)
├── dataset.py          # Dataset classes (raw parquet debug + streaming)
├── cat_maps.py         # Melee enum → dense index maps
├── forward_check.py    # NaN check on cold forward pass
├── weight_check.py     # Audit checkpoints for NaN/Inf weights
├── dataset_check.py    # Inspect parquet columns and mappings
├── inspect_window.py   # Audit window features and targets
├── checkpoints/        # Saved *.pt files
└── data/               # Slippi parquet files + source replays
    └── source_replays/ # Original .slp files for reproduction
```

## Setup

Install dependencies using [Poetry](https://python-poetry.org/):

```bash
poetry install
```

## Data

Parquet files are generated from Slippi `.slp` replays using [slippi-frame-extractor](https://github.com/erickfm/slippi-frame-extractor).

## Preprocessing

Compute training metadata (one-time, streams files, ~20 MB peak RAM):

```bash
python3 preprocess.py --data-dir data/subset
python3 preprocess.py --data-dir data/full
```

This produces three small JSON files alongside the parquets:
- `norm_stats.json` -- per-column mean/std for normalization
- `cat_maps.json` -- dynamic categorical mappings (ports, costumes)
- `file_index.json` -- frame counts per file (for train/val split + length estimation)

## Training

Point `--data-dir` at a parquet directory (auto-detects metadata for streaming, falls back to slow in-memory loading):

```bash
python3 train.py [--max-steps 2000] [--data-dir ./data/subset] [--debug] [--resume checkpoints/step_XXXXXX.pt] [--no-compile]
```

## Inference

```bash
python3 inference.py --dolphin-path /path/to/dolphin --iso-path /path/to/melee.iso [--debug]
```

## Notes

The `overfit_log*.txt` files in this repo are from intentional overfit experiments on a small fsmash-only subset. Near-zero main/l/r/btn losses in those logs are expected -- the model learned the dominant neutral/no-press pattern on that narrow data.
