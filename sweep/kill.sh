#!/usr/bin/env bash
# sweep/kill.sh -- Kill a training process on a specific machine/GPU.
#
# Usage:  bash sweep/kill.sh MACHINE GPU
#
# Example:
#   bash sweep/kill.sh B 2

set -euo pipefail

MACHINE_A="root@203.57.40.63"
PORT_A=10015
MACHINE_B="root@38.65.239.14"
PORT_B=28750
MACHINE_C="root@38.65.239.56"
PORT_C=45107

MACHINE="${1:?Usage: kill.sh MACHINE GPU}"
GPU="${2:?Usage: kill.sh MACHINE GPU}"

case "$MACHINE" in
    A|a) HOST=$MACHINE_A; PORT=$PORT_A ;;
    B|b) HOST=$MACHINE_B; PORT=$PORT_B ;;
    C|c) HOST=$MACHINE_C; PORT=$PORT_C ;;
    *) echo "Error: MACHINE must be A, B, or C"; exit 1 ;;
esac

echo "[kill] Finding train.py on $HOST:$PORT GPU $GPU ..."

PID=$(ssh -p "$PORT" "$HOST" \
    "ps aux | grep 'CUDA_VISIBLE_DEVICES=$GPU.*python3 train.py' | grep -v grep | awk '{print \$2}'" 2>/dev/null)

if [ -z "$PID" ]; then
    echo "[kill] No train.py found on GPU $GPU"
    exit 0
fi

echo "[kill] Killing PID $PID on GPU $GPU ..."
ssh -p "$PORT" "$HOST" "kill $PID" 2>/dev/null
sleep 1

STILL=$(ssh -p "$PORT" "$HOST" "ps -p $PID -o pid= 2>/dev/null" 2>/dev/null || true)
if [ -n "$STILL" ]; then
    echo "[kill] Process still alive, sending SIGKILL ..."
    ssh -p "$PORT" "$HOST" "kill -9 $PID" 2>/dev/null
fi
echo "[kill] Done."
