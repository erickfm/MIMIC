#!/usr/bin/env bash
set -euo pipefail

# sweep.sh -- Launch parallel training runs across GPUs on a remote machine.
#
# Usage:
#   bash sweep.sh --host root@38.65.239.56 --port 27159 \
#       --group sweep-v1 --wandb-key "your-key" \
#       --run "0 small 5e-4" \
#       --run "1 medium 3e-4" \
#       --run "2 deep 5e-4 --seq-len 120"
#
# Each --run is: "GPU_ID PRESET LR [extra train.py flags]"
# Runs are launched via nohup so they survive SSH disconnect.

HOST=""
PORT=22
GROUP=""
WANDB_KEY=""
DATA_DIR="data/subset"
RUNS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)      HOST="$2"; shift 2 ;;
        --port)      PORT="$2"; shift 2 ;;
        --group)     GROUP="$2"; shift 2 ;;
        --wandb-key) WANDB_KEY="$2"; shift 2 ;;
        --data-dir)  DATA_DIR="$2"; shift 2 ;;
        --run)       RUNS+=("$2"); shift 2 ;;
        *)           echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$HOST" ]]; then
    echo "ERROR: --host is required"
    exit 1
fi

if [[ ${#RUNS[@]} -eq 0 ]]; then
    echo "ERROR: at least one --run is required"
    exit 1
fi

echo "=== FRAME Sweep ==="
echo "  Host:  $HOST:$PORT"
echo "  Group: ${GROUP:-<none>}"
echo "  Runs:  ${#RUNS[@]}"
echo ""

REMOTE_CMD=""

if [[ -n "$WANDB_KEY" ]]; then
    REMOTE_CMD+="export WANDB_API_KEY='$WANDB_KEY'; "
fi

REMOTE_CMD+="cd /root/FRAME && "
REMOTE_CMD+="mkdir -p logs && "

for run_spec in "${RUNS[@]}"; do
    read -r GPU PRESET LR EXTRA <<< "$run_spec" || true

    GROUP_FLAG=""
    if [[ -n "$GROUP" ]]; then
        GROUP_FLAG="--wandb-group $GROUP"
    fi

    TAGS="sweep,$PRESET"
    if [[ "$EXTRA" == *"--seq-len"* ]]; then
        TAGS+=",seq-len"
    fi
    if [[ "$EXTRA" == *"--num-layers"* ]]; then
        TAGS+=",arch"
    fi

    CMD="CUDA_VISIBLE_DEVICES=$GPU python3 train.py"
    CMD+=" --model $PRESET --lr $LR"
    CMD+=" --data-dir $DATA_DIR"
    CMD+=" --wandb-tags $TAGS"
    CMD+=" $GROUP_FLAG"
    if [[ -n "$EXTRA" ]]; then
        CMD+=" $EXTRA"
    fi

    LOG_NAME="${PRESET}-lr${LR}-gpu${GPU}"
    REMOTE_CMD+="nohup $CMD > logs/${LOG_NAME}.log 2>&1 & "
    echo "  GPU $GPU: $PRESET lr=$LR ${EXTRA:+($EXTRA)}"
done

REMOTE_CMD+="echo 'All runs launched'; jobs -l"

echo ""
echo "Launching on $HOST..."
ssh -p "$PORT" "$HOST" "$REMOTE_CMD"
echo ""
echo "=== Sweep launched. Check wandb or SSH in to monitor. ==="
echo "  ssh -p $PORT $HOST 'tail -f /root/FRAME/logs/*.log'"
