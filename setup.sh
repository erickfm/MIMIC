#!/usr/bin/env bash
set -euo pipefail

# MIMIC - Setup for a fresh machine with an NVIDIA GPU.
#
# Usage:
#   bash setup.sh                       # install deps, emulator, ISO, download data
#   bash setup.sh --run                 # also starts training after setup
#   bash setup.sh --rsync               # pull pre-built shards from Machine A
#   bash setup.sh --run --model small   # extra args forwarded to train.py

DATA_DIR="${DATA_DIR:-data/fox_hal_full}"
EMULATOR_DIR="emulator"
ISO_PATH="melee.iso"
RUN_AFTER=false
RSYNC_DATA=false
EXTRA_TRAIN_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run)       RUN_AFTER=true; shift ;;
        --rsync)     RSYNC_DATA=true; shift ;;
        --data-dir)  DATA_DIR="$2"; shift 2 ;;
        *)           EXTRA_TRAIN_ARGS+=("$1"); shift ;;
    esac
done

echo "=== MIMIC setup ==="

# ── 1. Python deps ──────────────────────────────────────────────────────────
echo ""
echo "── Installing Python dependencies ──"
pip install torch numpy pandas pyarrow wandb tensordict \
    huggingface_hub melee==0.45.1 --quiet 2>/dev/null \
  || pip install torch numpy pandas pyarrow wandb tensordict \
    huggingface_hub melee==0.45.1 --quiet --break-system-packages

# ── 2. Dolphin emulator ────────────────────────────────────────────────────
echo ""
echo "── Setting up Dolphin emulator ──"
if [[ -x "$EMULATOR_DIR/squashfs-root/usr/bin/dolphin-emu" ]]; then
    echo "  Dolphin already extracted."
else
    if [[ ! -f emulator.tar.gz ]]; then
        echo "  ERROR: emulator.tar.gz not found. Run: git lfs pull"
        exit 1
    fi
    echo "  Extracting emulator.tar.gz ..."
    mkdir -p "$EMULATOR_DIR"
    tar xzf emulator.tar.gz -C "$EMULATOR_DIR" --strip-components=0
    echo "  Dolphin ready at $EMULATOR_DIR/squashfs-root/usr/bin/dolphin-emu"
fi

# ── 3. Melee ISO ───────────────────────────────────────────────────────────
echo ""
echo "── Setting up Melee ISO ──"
if [[ -f "$ISO_PATH" ]]; then
    echo "  ISO found at $ISO_PATH"
else
    echo "  Downloading ISO ..."
    curl -L -o melee.zip "https://melee.today/download/melee.zip"
    unzip -o melee.zip -d _iso_tmp
    # Find the .iso inside the zip and move it to repo root
    ISO_FILE=$(find _iso_tmp -name "*.iso" -o -name "*.ISO" | head -1)
    if [[ -z "$ISO_FILE" ]]; then
        echo "  ERROR: No .iso found in melee.zip"
        rm -rf _iso_tmp melee.zip
        exit 1
    fi
    mv "$ISO_FILE" "$ISO_PATH"
    rm -rf _iso_tmp melee.zip
    echo "  ISO saved to $ISO_PATH"
fi

# ── 4. Data ─────────────────────────────────────────────────────────────────
echo ""
echo "── Setting up training data ──"
if [[ -d "$DATA_DIR" ]] && ls "$DATA_DIR"/train_shard_*.pt &>/dev/null; then
    SHARD_COUNT=$(ls "$DATA_DIR"/train_shard_*.pt | wc -l)
    echo "  Found $SHARD_COUNT shards in $DATA_DIR"
elif $RSYNC_DATA; then
    echo "  Pulling pre-built shards from Machine A ..."
    mkdir -p "$DATA_DIR"
    rsync -avz --progress -e "ssh -p 22877" \
        root@194.14.47.19:/root/MIMIC/data/fox_hal_full/ "$DATA_DIR/"
    echo "  Data synced to $DATA_DIR"
else
    echo "  No shards found in $DATA_DIR."
    echo "  Options:"
    echo "    1. bash setup.sh --rsync          # pull shards from Machine A"
    echo "    2. python tools/slp_to_shards.py  # build from raw .slp replays"
    echo ""
    echo "  Skipping data setup."
fi

# ── 5. GPU check ────────────────────────────────────────────────────────────
echo ""
echo "── GPU check ──"
python3 -c "
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_mem / 1e9
        print(f'  GPU {i}: {name} ({mem:.0f} GB)')
else:
    print('  WARNING: No CUDA GPU detected. Training will be very slow.')
"

echo ""
echo "=== Setup complete ==="
echo "  Dolphin:  $EMULATOR_DIR/squashfs-root/usr/bin/dolphin-emu"
echo "  ISO:      $ISO_PATH"
echo "  Data:     $DATA_DIR"

# ── 6. Optionally start training ────────────────────────────────────────────
if $RUN_AFTER; then
    echo ""
    echo "── Starting training ──"
    python3 train.py --data-dir "$DATA_DIR" "${EXTRA_TRAIN_ARGS[@]}"
fi
