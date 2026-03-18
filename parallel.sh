#!/usr/bin/env bash
# parallel.sh -- Launch DDP training across GPUs on a remote machine.
#
# Usage:
#   bash parallel.sh MACHINE [NGPUS] -- [TRAIN_ARGS...]
#
# MACHINE: A, B, C, or "local"
# NGPUS:   Number of GPUs (default: all GPUs on the machine)
#
# Everything after "--" is forwarded to train.py.
#
# Examples:
#   # All 8 GPUs on Machine C:
#   bash parallel.sh C -- --run-name ddp-test --max-samples 50000000
#
#   # 4 GPUs on Machine C:
#   bash parallel.sh C 4 -- --run-name ddp-test --batch-size 32
#
#   # All GPUs on local machine:
#   bash parallel.sh local -- --run-name ddp-local
#
#   # Dry run (show command without executing):
#   DRY=1 bash parallel.sh C -- --run-name ddp-test

set -euo pipefail

# --- Machine registry ---
declare -A HOSTS PORTS GPUS DIRS
HOSTS[A]="root@203.57.40.63";  PORTS[A]=10015;  GPUS[A]=6;  DIRS[A]="/root/FRAME"  # OFFLINE
HOSTS[B]="root@38.65.239.14";  PORTS[B]=28750;  GPUS[B]=7;  DIRS[B]="/root/FRAME"  # OFFLINE
HOSTS[C]="root@194.14.47.19";  PORTS[C]=22824;  GPUS[C]=8;  DIRS[C]="/root/FRAME"
HOSTS[D]="root@142.127.93.36"; PORTS[D]=11559;  GPUS[D]=8;  DIRS[D]="/root/FRAME"
HOSTS[E]="root@66.222.138.178"; PORTS[E]=11335; GPUS[E]=8;  DIRS[E]="/root/FRAME"
HOSTS[F]="root@74.2.96.10";     PORTS[F]=18619; GPUS[F]=8;  DIRS[F]="/root/FRAME"

# --- Parse args ---
MACHINE="${1:?Usage: parallel.sh MACHINE [NGPUS] -- [TRAIN_ARGS...]}"
MACHINE="${MACHINE^^}"  # uppercase
shift

# Optional NGPUS before "--"
NGPUS=""
if [[ "${1:-}" != "--" && "${1:-}" != "" ]]; then
    NGPUS="$1"
    shift
fi

# Skip the "--" separator
if [[ "${1:-}" == "--" ]]; then
    shift
fi

TRAIN_ARGS="$*"

# --- Resolve machine ---
if [[ "$MACHINE" == "LOCAL" ]]; then
    # Local launch
    if [[ -z "$NGPUS" ]]; then
        NGPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
    fi

    CMD="torchrun --nproc_per_node=$NGPUS train.py $TRAIN_ARGS"

    echo "═══════════════════════════════════════════════════"
    echo "  DDP local: $NGPUS GPUs"
    echo "  $CMD"
    echo "═══════════════════════════════════════════════════"

    if [[ "${DRY:-}" == "1" ]]; then
        echo "[DRY RUN] Would execute locally."
        exit 0
    fi

    exec $CMD
else
    # Remote launch
    if [[ -z "${HOSTS[$MACHINE]+x}" ]]; then
        echo "Error: Unknown machine '$MACHINE'. Use A, B, C, D, or local."
        exit 1
    fi

    HOST="${HOSTS[$MACHINE]}"
    PORT="${PORTS[$MACHINE]}"
    REMOTE_DIR="${DIRS[$MACHINE]}"
    MAX_GPUS="${GPUS[$MACHINE]}"

    if [[ -z "$NGPUS" ]]; then
        NGPUS="$MAX_GPUS"
    fi
    if (( NGPUS > MAX_GPUS )); then
        echo "Error: Machine $MACHINE only has $MAX_GPUS GPUs (requested $NGPUS)"
        exit 1
    fi

    # Extract --run-name for log file naming
    RUN_NAME="ddp-run"
    for ((i=1; i<=$#; i++)); do
        arg="${!i}"
        if [[ "$arg" == "--run-name" ]]; then
            next=$((i+1))
            RUN_NAME="${!next}"
            break
        fi
    done

    REMOTE_CMD="cd $REMOTE_DIR && mkdir -p logs/ddp && \
torchrun --nproc_per_node=$NGPUS train.py $TRAIN_ARGS"

    REMOTE_CMD_BG="cd $REMOTE_DIR && mkdir -p logs/ddp && \
nohup torchrun --nproc_per_node=$NGPUS train.py $TRAIN_ARGS \
> logs/ddp/${RUN_NAME}.log 2>&1 & echo pid=\$!"

    echo "═══════════════════════════════════════════════════"
    echo "  DDP on Machine $MACHINE: $NGPUS / $MAX_GPUS GPUs"
    echo "  Host: $HOST:$PORT"
    echo "  torchrun --nproc_per_node=$NGPUS train.py $TRAIN_ARGS"
    echo "═══════════════════════════════════════════════════"

    if [[ "${DRY:-}" == "1" ]]; then
        echo "[DRY RUN] Would SSH and execute."
        exit 0
    fi

    if [[ "${BG:-}" == "1" ]]; then
        echo "Launching in background ..."
        ssh -o ConnectTimeout=10 -p "$PORT" "$HOST" "$REMOTE_CMD_BG"
        echo "Log: ssh -p $PORT $HOST 'tail -f $REMOTE_DIR/logs/ddp/${RUN_NAME}.log'"
    else
        # Run in foreground (Ctrl+C kills it)
        ssh -o ConnectTimeout=10 -t -p "$PORT" "$HOST" "$REMOTE_CMD"
    fi
fi
