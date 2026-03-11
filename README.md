# FRAME: Fixed-step Realtime Action Matrix Estimator
FRAME is a decoder-only transformer model for predicting next-frame inputs in Super Smash Bros. Melee. Built around fixed-step causal attention, it estimates future controller states from sequences of gameplay data.


## Features
- Decoder-only architecture (GPT-like)

- Intra-frame cross-attention encoder over feature groups

- Supports real-time inference on live frame sequences

- Trained on Slippi-derived datasets

- Predicts analog stick, C-stick, trigger, and button inputs

- Multi-task learning with adaptive loss rebalancing


## Default configuration
| setting                | value                                       |
|------------------------|---------------------------------------------|
| Window length (W)      | 60 frames                                   |
| Reaction delay (R)     | 1 frame                                     |
| Model width (d_model)  | 1024                                        |
| Transformer layers     | 4                                           |
| Attention heads        | 8                                            |
| Feedforward dim        | 2048                                        |
| Intra-frame width      | 256                                         |


## Project Structure
```
.
├── train.py            # Training loop & checkpointing
├── eval.py             # Validation / quick metrics
├── inference.py        # Real-time single-window inference
├── model.py            # FramePredictor + Transformer heads
├── frame_encoder.py    # Intra-frame cross-attention encoder
├── dataset.py          # Parquet → tensor windows
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

Parquet files are generated from Slippi `.slp` replays using [slippi-frame-extractor](https://github.com/erickfm/slippi-frame-extractor):

```bash
python extract.py data/source_replays -o data
```

## Training

```bash
python train.py [--debug] [--resume checkpoints/epoch_XX.pt]
```

## Inference

```bash
python inference.py --dolphin-path /path/to/dolphin --iso-path /path/to/melee.iso [--debug]
```
