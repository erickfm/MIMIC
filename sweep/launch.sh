#!/usr/bin/env bash
# sweep/launch.sh -- Launch a single training run on a remote machine/GPU.
#
# Usage:  bash sweep/launch.sh MACHINE GPU RUN_NAME [EXTRA_ARGS]
#
#   MACHINE: A, B, or C
#   GPU:     GPU index (0-7)
#   RUN_NAME: wandb run name (also used for log file)
#   EXTRA_ARGS: additional train.py arguments (quoted string)
#
# Example:
#   bash sweep/launch.sh B 2 wd-tiny "--model tiny"
#   bash sweep/launch.sh A 0 wd-baseline "--lr 5e-5"

set -euo pipefail

MACHINE_A="root@203.57.40.63"
PORT_A=10015
MACHINE_B="root@38.65.239.14"
PORT_B=28750
MACHINE_C="root@38.65.239.56"
PORT_C=45107

COMMON="--data-dir data/wavedash_v2 --model medium --stick-loss clusters \
  --clusters-path data/full/stick_clusters.json --label-smoothing 0.0 \
  --autoregressive-heads --seq-len 30 --max-steps 20000 \
  --target-val-f1 0.985 --val-frac 0.01 \
  --wandb-group wavedash-speed"

MACHINE="${1:?Usage: launch.sh MACHINE GPU RUN_NAME [EXTRA_ARGS]}"
GPU="${2:?Usage: launch.sh MACHINE GPU RUN_NAME [EXTRA_ARGS]}"
RUN_NAME="${3:?Usage: launch.sh MACHINE GPU RUN_NAME [EXTRA_ARGS]}"
EXTRA="${4:-}"

case "$MACHINE" in
    A|a) HOST=$MACHINE_A; PORT=$PORT_A ;;
    B|b) HOST=$MACHINE_B; PORT=$PORT_B ;;
    C|c) HOST=$MACHINE_C; PORT=$PORT_C ;;
    *) echo "Error: MACHINE must be A, B, or C"; exit 1 ;;
esac

echo "[launch] $RUN_NAME → $HOST:$PORT GPU $GPU"
echo "  extra: $EXTRA"

ssh -p "$PORT" "$HOST" "cd /root/FRAME && mkdir -p logs/sweep && \
    CUDA_VISIBLE_DEVICES=$GPU nohup python3 train.py \
    $COMMON --run-name $RUN_NAME $EXTRA \
    > logs/sweep/$RUN_NAME.log 2>&1 &"

echo "[launch] $RUN_NAME started (log: logs/sweep/$RUN_NAME.log)"
