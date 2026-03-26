#!/usr/bin/env bash
# sweep_single_gpu.sh — Launch 8 single-GPU experiments on one machine.
#
# Usage:
#   bash tools/sweep_single_gpu.sh              # run locally
#   ssh -p PORT HOST "cd /root/FRAME && bash tools/sweep_single_gpu.sh"
#
# Each GPU runs an independent training process with different configs.
# Logs go to logs/sweep/<run-name>.log

set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs/sweep

DATA="${1:-data/fox/FOX}"  # pass data dir as first arg, default Fox
COMMON="--seq-len 180 --batch-size 64 --grad-accum-steps 4 --max-samples 250000000 \
  --no-compile --self-inputs --seed 42 --grad-clip-norm 1.0 --encoder flat --data-dir $DATA"

launch() {
    local gpu=$1 name=$2; shift 2
    echo "GPU $gpu: $name"
    CUDA_VISIBLE_DEVICES=$gpu nohup python3 train.py $COMMON "$@" --run-name "$name" \
        > "logs/sweep/${name}.log" 2>&1 &
}

# GPU 0: Baseline (medium, focal, ls=0.1, dropout=0.1, warmup=5%)
launch 0 sweep-baseline \
    --model medium --lr 3e-4

# GPU 1: Plain CE (no focal loss, no label smoothing)
launch 1 sweep-plain-ce \
    --model medium --lr 3e-4 --plain-ce --label-smoothing 0

# GPU 2: No label smoothing (keep focal)
launch 2 sweep-no-ls \
    --model medium --lr 3e-4 --label-smoothing 0

# GPU 3: No warmup
launch 3 sweep-no-warmup \
    --model medium --lr 3e-4 --warmup-steps 0

# GPU 4: Dropout 0.2 (HAL's value)
launch 4 sweep-drop20 \
    --model medium --lr 3e-4 --dropout 0.2

# GPU 5: 16 heads (more attention specialization)
launch 5 sweep-16heads \
    --model medium --lr 3e-4 --n-heads 16

# GPU 6: HAL shape (6 layers × 512-d)
launch 6 sweep-hal-shape \
    --model hal --lr 3e-4

# GPU 7: Full HAL-like combo
launch 7 sweep-hal-combo \
    --model hal --lr 3e-4 --plain-ce --label-smoothing 0 --dropout 0.2 --warmup-steps 0

echo ""
echo "All 8 launched. Check logs:"
echo "  tail -f logs/sweep/sweep-*.log"
echo "  nvidia-smi"
