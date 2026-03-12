#!/usr/bin/env bash
set -euo pipefail

# FRAME - Quick-start setup for a fresh machine with an NVIDIA GPU.
# Usage:
#   export HF_TOKEN=hf_...
#   bash setup.sh              # downloads data + installs deps
#   bash setup.sh --run        # also starts a 1-epoch training run
#   bash setup.sh --run --model small   # train with a smaller model

REPO_ID="${HF_REPO:-erickfm/frame-melee-subset}"
DATA_DIR="${DATA_DIR:-data/subset}"
RUN_AFTER=false
EXTRA_TRAIN_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run)   RUN_AFTER=true; shift ;;
        --repo)  REPO_ID="$2"; shift 2 ;;
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        *)       EXTRA_TRAIN_ARGS+=("$1"); shift ;;
    esac
done

echo "=== FRAME setup ==="
echo "  HF repo:  $REPO_ID"
echo "  Data dir: $DATA_DIR"

# ── 1. System deps (idempotent) ──────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Install Python 3.10+."
    exit 1
fi

# ── 2. Python deps ───────────────────────────────────────────────────────────
echo ""
echo "── Installing Python dependencies ──"
pip install torch numpy pandas pyarrow wandb huggingface_hub melee==0.45.1 \
    typing-extensions --quiet 2>/dev/null \
  || pip install torch numpy pandas pyarrow wandb huggingface_hub melee==0.45.1 \
    typing-extensions --quiet --break-system-packages

# ── 3. Download dataset from HuggingFace ─────────────────────────────────────
echo ""
echo "── Downloading dataset ──"
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "WARNING: HF_TOKEN not set. Download may fail for private repos."
fi

python3 -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id='${REPO_ID}',
    repo_type='dataset',
    local_dir='${DATA_DIR}',
    token=os.environ.get('HF_TOKEN'),
)
print(f'Dataset downloaded to ${DATA_DIR}')
"

# ── 4. Run preprocessing if metadata missing ─────────────────────────────────
if [[ ! -f "${DATA_DIR}/norm_stats.json" ]] || [[ ! -f "${DATA_DIR}/cat_maps.json" ]] || [[ ! -f "${DATA_DIR}/file_index.json" ]]; then
    echo ""
    echo "── Running preprocessing (metadata only) ──"
    python3 preprocess.py --data-dir "$DATA_DIR"
fi

# ── 5. Verify GPU ────────────────────────────────────────────────────────────
echo ""
echo "── GPU check ──"
python3 -c "
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'  GPU: {name} ({mem:.0f} GB)')
else:
    print('  WARNING: No CUDA GPU detected. Training will be very slow.')
"

N_FILES=$(ls "${DATA_DIR}"/*.parquet 2>/dev/null | wc -l)
echo "  Data: ${N_FILES} parquet files in ${DATA_DIR}"
echo ""
echo "=== Setup complete ==="

# ── 6. Optionally start training ─────────────────────────────────────────────
if $RUN_AFTER; then
    echo ""
    echo "── Starting training ──"
    python3 train.py --data-dir "$DATA_DIR" "${EXTRA_TRAIN_ARGS[@]}"
fi
