#!/usr/bin/env bash
# sweep_phase3.sh -- Launch Phase 3 experiments across 3 machines
#
# Usage:
#   bash sweep_phase3.sh [machines]
#
#   machines: "all" (default), "A", "B", "C"
#
# Prerequisites:
#   - data/full with metadata on each machine (run setup.sh first)
#   - Latest code pushed: git push && ssh <machine> "cd /root/FRAME && git pull"

set -euo pipefail

MACHINE_A="root@203.57.40.63"
PORT_A=10015
MACHINE_B="root@38.65.239.14"
PORT_B=28750
MACHINE_C="root@38.65.239.56"
PORT_C=45107

GROUP="phase3"
DATA="data/full"
SAMPLES=2000000

COMMON="--data-dir $DATA --max-samples $SAMPLES --encoder hybrid16 --wandb-group $GROUP"

launch() {
    local host=$1 port=$2 gpu=$3 name=$4
    shift 4
    local extra="$*"
    echo "  GPU $gpu: $name"
    ssh -p "$port" "$host" "cd /root/FRAME && \
        CUDA_VISIBLE_DEVICES=$gpu nohup python3 train.py \
        $COMMON --run-name $name $extra \
        > logs/$name.log 2>&1 &"
}

launch_machine_a() {
    echo "=== Machine A ($MACHINE_A:$PORT_A) ==="
    ssh -p $PORT_A $MACHINE_A "cd /root/FRAME && mkdir -p logs"

    # GPU 0: baseline (medium/4L/768d = 32M)
    launch $MACHINE_A $PORT_A 0 baseline \
        --model medium --lr 8e-4 --batch-size 384 \
        --wandb-tags "phase3,baseline,depth,width,context,posenc,loss"

    # GPU 1: depth-2L (18M, more headroom)
    launch $MACHINE_A $PORT_A 1 depth-2L \
        --model medium --lr 8e-4 --batch-size 512 --num-layers 2 \
        --wandb-tags "phase3,depth"

    # GPU 2: depth-6L (47M)
    launch $MACHINE_A $PORT_A 2 depth-6L \
        --model medium --lr 8e-4 --batch-size 256 --num-layers 6 \
        --wandb-tags "phase3,depth"

    # GPU 3: depth-8L (61M)
    launch $MACHINE_A $PORT_A 3 depth-8L \
        --model medium --lr 8e-4 --batch-size 192 --num-layers 8 \
        --wandb-tags "phase3,depth"

    # GPU 4: width-512 (16M, lots of headroom)
    launch $MACHINE_A $PORT_A 4 width-512 \
        --model small --lr 8e-4 --batch-size 512 \
        --wandb-tags "phase3,width"

    # GPU 5: width-1024 (55M)
    launch $MACHINE_A $PORT_A 5 width-1024 \
        --model base --lr 8e-4 --batch-size 256 \
        --wandb-tags "phase3,width"

    echo "  Machine A: 6 runs launched"
}

launch_machine_b() {
    echo "=== Machine B ($MACHINE_B:$PORT_B) ==="
    ssh -p $PORT_B $MACHINE_B "cd /root/FRAME && mkdir -p logs"

    # GPU 0: ctx-30 (half seq = lots of headroom)
    launch $MACHINE_B $PORT_B 0 ctx-30 \
        --model medium --lr 8e-4 --seq-len 30 --batch-size 768 \
        --wandb-tags "phase3,context"

    # GPU 1: ctx-90
    launch $MACHINE_B $PORT_B 1 ctx-90 \
        --model medium --lr 8e-4 --seq-len 90 --batch-size 256 \
        --wandb-tags "phase3,context"

    # GPU 2: ctx-120
    launch $MACHINE_B $PORT_B 2 ctx-120 \
        --model medium --lr 8e-4 --seq-len 120 --batch-size 192 \
        --wandb-tags "phase3,context"

    # GPU 3: ctx-180
    launch $MACHINE_B $PORT_B 3 ctx-180 \
        --model medium --lr 8e-4 --seq-len 180 --batch-size 128 \
        --wandb-tags "phase3,context"

    # GPU 4: pos-rope
    launch $MACHINE_B $PORT_B 4 pos-rope \
        --model medium --lr 8e-4 --batch-size 384 --pos-enc rope \
        --wandb-tags "phase3,posenc"

    # GPU 5: loss-huber
    launch $MACHINE_B $PORT_B 5 loss-huber \
        --model medium --lr 8e-4 --batch-size 384 --stick-loss huber \
        --wandb-tags "phase3,loss"

    echo "  Machine B: 6 runs launched"
}

launch_machine_c() {
    echo "=== Machine C ($MACHINE_C:$PORT_C) ==="
    ssh -p $PORT_C $MACHINE_C "cd /root/FRAME && mkdir -p logs"

    # GPU 0: loss-discrete (slightly more head params from 32x32 output)
    launch $MACHINE_C $PORT_C 0 loss-discrete \
        --model medium --lr 8e-4 --batch-size 384 --stick-loss discrete \
        --wandb-tags "phase3,loss"

    # GPU 1: loss-focal-btn
    launch $MACHINE_C $PORT_C 1 loss-focal-btn \
        --model medium --lr 8e-4 --batch-size 384 --btn-loss focal \
        --wandb-tags "phase3,loss"

    echo "  Machine C: 2 runs launched"
}

TARGET="${1:-all}"
case "$TARGET" in
    all) launch_machine_a; launch_machine_b; launch_machine_c ;;
    A|a) launch_machine_a ;;
    B|b) launch_machine_b ;;
    C|c) launch_machine_c ;;
    *) echo "Usage: $0 [all|A|B|C]"; exit 1 ;;
esac

echo ""
echo "=== Phase 3: 14 runs launched (1 baseline + 13 variants) ==="
echo "Monitor: wandb.ai/erickfm/FRAME (group: phase3)"
echo "Logs: ssh <machine> 'tail -f /root/FRAME/logs/<run>.log'"
