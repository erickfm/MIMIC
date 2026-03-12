#!/usr/bin/env bash
set -euo pipefail

# sweep_scaling.sh -- Power law scaling experiments
#
# Usage:
#   bash sweep_scaling.sh --round p1r1   # Phase 1, Round 1 (7 runs)
#   bash sweep_scaling.sh --round p1r2   # Phase 1, Round 2 (7 runs)
#   bash sweep_scaling.sh --round p1r3   # Phase 1, Round 3 (1 run)
#   bash sweep_scaling.sh --round p2r1   # Phase 2, Round 1 (7 runs)
#   bash sweep_scaling.sh --round p2r2   # Phase 2, Round 2 (1 run)
#
# All runs use: --no-compile, wandb-group scaling-v1, data/subset

ROUND=""
DATA_DIR="data/subset"
GROUP="scaling-v1"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --round) ROUND="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$ROUND" ]]; then
    echo "ERROR: --round is required"
    echo "Options: p1r1, p1r2, p1r3, p2r1, p2r2"
    exit 1
fi

cd /root/FRAME
mkdir -p logs/scaling

launch() {
    local GPU=$1 MODEL=$2 LR=$3 TAGS=$4
    shift 4
    local EXTRA="$*"

    local LOG_NAME="${MODEL}-lr${LR}"
    if [[ -n "$EXTRA" ]]; then
        for arg in $EXTRA; do
            case "$arg" in
                --batch-size) ;;
                [0-9]*) LOG_NAME="${MODEL}-lr${LR}-bs${arg}" ;;
            esac
        done
    fi

    local CMD="CUDA_VISIBLE_DEVICES=$GPU python3 train.py"
    CMD+=" --model $MODEL --lr $LR"
    CMD+=" --data-dir $DATA_DIR"
    CMD+=" --no-compile"
    CMD+=" --wandb-tags $TAGS"
    CMD+=" --wandb-group $GROUP"
    if [[ -n "$EXTRA" ]]; then
        CMD+=" $EXTRA"
    fi

    echo "  GPU $GPU: $MODEL lr=$LR ${EXTRA:+($EXTRA)} -> logs/scaling/${LOG_NAME}.log"
    nohup env $CMD > "logs/scaling/${LOG_NAME}.log" 2>&1 &
}

echo "=== Scaling Sweep: $ROUND ==="
echo "  Group: $GROUP"
echo ""

case "$ROUND" in
    # ─── Phase 1: LR Sweep (BS=200, 1 epoch) ───
    p1r1)
        echo "Phase 1 Round 1: tiny/{1e-4,3e-4,2e-3}, small/{1e-4,3e-4,2e-3}, medium/1e-4"
        launch 0 tiny   1e-4 "scaling,lr-sweep,tiny"
        launch 1 tiny   3e-4 "scaling,lr-sweep,tiny"
        launch 2 tiny   2e-3 "scaling,lr-sweep,tiny"
        launch 3 small  1e-4 "scaling,lr-sweep,small"
        launch 4 small  3e-4 "scaling,lr-sweep,small"
        launch 5 small  2e-3 "scaling,lr-sweep,small"
        launch 6 medium 1e-4 "scaling,lr-sweep,medium"
        ;;
    p1r2)
        echo "Phase 1 Round 2: medium/{3e-4,5e-4,1e-3,2e-3}, base/{1e-4,5e-4,1e-3}"
        launch 0 medium 3e-4 "scaling,lr-sweep,medium"
        launch 1 medium 5e-4 "scaling,lr-sweep,medium"
        launch 2 medium 1e-3 "scaling,lr-sweep,medium"
        launch 3 medium 2e-3 "scaling,lr-sweep,medium"
        launch 4 base   1e-4 "scaling,lr-sweep,base"
        launch 5 base   5e-4 "scaling,lr-sweep,base"
        launch 6 base   1e-3 "scaling,lr-sweep,base"
        ;;
    p1r3)
        echo "Phase 1 Round 3: base/2e-3"
        launch 0 base 2e-3 "scaling,lr-sweep,base"
        ;;

    # ─── Phase 2: Batch Size Sweep (optimal LR, 1 epoch) ───
    # LR values here are placeholders -- replace with Phase 1 optimal LRs
    p2r1)
        echo "Phase 2 Round 1: {tiny,small,medium,base}/BS=100, {tiny,small,medium}/BS=400"
        echo "*** Verify LR values match Phase 1 optimal before launching! ***"
        launch 0 tiny   LR_TINY   "scaling,bs-sweep,tiny"   --batch-size 100
        launch 1 small  LR_SMALL  "scaling,bs-sweep,small"  --batch-size 100
        launch 2 medium LR_MEDIUM "scaling,bs-sweep,medium" --batch-size 100
        launch 3 base   LR_BASE   "scaling,bs-sweep,base"   --batch-size 100
        launch 4 tiny   LR_TINY   "scaling,bs-sweep,tiny"   --batch-size 400
        launch 5 small  LR_SMALL  "scaling,bs-sweep,small"  --batch-size 400
        launch 6 medium LR_MEDIUM "scaling,bs-sweep,medium" --batch-size 400
        ;;
    p2r2)
        echo "Phase 2 Round 2: base/BS=400"
        echo "*** Verify LR value matches Phase 1 optimal before launching! ***"
        launch 0 base LR_BASE "scaling,bs-sweep,base" --batch-size 400
        ;;
    *)
        echo "Unknown round: $ROUND"
        echo "Options: p1r1, p1r2, p1r3, p2r1, p2r2"
        exit 1
        ;;
esac

sleep 2
echo ""
echo "=== Launched. GPU status: ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo ""
echo "Monitor: tail -f logs/scaling/*.log"
