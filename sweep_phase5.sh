#!/usr/bin/env bash
# sweep_phase5.sh -- Phase 5: Discrete clusters + autoregressive heads
#
# Usage:
#   bash sweep_phase5.sh [machines]
#
#   machines: "all" (default), "A", "B", "C"
#
# Prerequisites:
#   - data/full with metadata + stick_clusters.json on each machine
#   - Latest code pushed: git push && ssh <machine> "cd /root/FRAME && git pull"

set -euo pipefail

MACHINE_A="root@203.57.40.63"
PORT_A=10015
MACHINE_B="root@38.65.239.14"
PORT_B=28750
MACHINE_C="root@38.65.239.56"
PORT_C=45107

GROUP="phase5-clusters"
DATA="data/full"

# All runs: clusters + no-opp-inputs (default) + medium model + hybrid16
COMMON="--data-dir $DATA --model medium --encoder hybrid16 --stick-loss clusters --wandb-group $GROUP"
AR="--autoregressive-heads"

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
    echo "=== Machine A ($MACHINE_A:$PORT_A) — 6 GPUs ==="
    ssh -p $PORT_A $MACHINE_A "cd /root/FRAME && mkdir -p logs"

    # Phase 5a: Quick sanity (5000 steps, ~30 min)
    launch $MACHINE_A $PORT_A 0 cls-ar-ctx60-q \
        $AR --seq-len 60 --batch-size 256 --max-steps 5000 --lr 8e-4 --seed 1 \
        --wandb-tags "phase5,quick,autoreg,ctx60"

    launch $MACHINE_A $PORT_A 1 cls-ctx60-q \
        --seq-len 60 --batch-size 256 --max-steps 5000 --lr 8e-4 --seed 1 \
        --wandb-tags "phase5,quick,independent,ctx60"

    launch $MACHINE_A $PORT_A 2 cls-ar-ctx180-q \
        $AR --seq-len 180 --batch-size 96 --max-steps 5000 --lr 8e-4 --seed 1 \
        --wandb-tags "phase5,quick,autoreg,ctx180"

    launch $MACHINE_A $PORT_A 3 cls-ctx180-q \
        --seq-len 180 --batch-size 96 --max-steps 5000 --lr 8e-4 --seed 1 \
        --wandb-tags "phase5,quick,independent,ctx180"

    # Phase 5b: 2-hour runs (20000 steps)
    launch $MACHINE_A $PORT_A 4 cls-ar-ctx60-20k \
        $AR --seq-len 60 --batch-size 256 --max-steps 20000 --lr 8e-4 --seed 1 \
        --wandb-tags "phase5,2h,autoreg,ctx60"

    launch $MACHINE_A $PORT_A 5 cls-ar-ctx60-20k-s2 \
        $AR --seq-len 60 --batch-size 256 --max-steps 20000 --lr 8e-4 --seed 2 \
        --wandb-tags "phase5,2h,autoreg,ctx60"

    echo "  Machine A: 6 runs launched"
}

launch_machine_b() {
    echo "=== Machine B ($MACHINE_B:$PORT_B) — 7 GPUs ==="
    ssh -p $PORT_B $MACHINE_B "cd /root/FRAME && mkdir -p logs"

    # Phase 5b: 2-hour runs (20000 steps)
    launch $MACHINE_B $PORT_B 0 cls-ar-ctx180-20k \
        $AR --seq-len 180 --batch-size 96 --max-steps 20000 --lr 8e-4 --seed 1 \
        --wandb-tags "phase5,2h,autoreg,ctx180"

    launch $MACHINE_B $PORT_B 1 cls-ar-ctx180-20k-s2 \
        $AR --seq-len 180 --batch-size 96 --max-steps 20000 --lr 8e-4 --seed 2 \
        --wandb-tags "phase5,2h,autoreg,ctx180"

    launch $MACHINE_B $PORT_B 2 cls-ar-ctx60-lr5e4 \
        $AR --seq-len 60 --batch-size 256 --max-steps 20000 --lr 5e-4 --seed 1 \
        --wandb-tags "phase5,2h,autoreg,ctx60,lr-sweep"

    launch $MACHINE_B $PORT_B 3 cls-ar-ctx60-lr1e3 \
        $AR --seq-len 60 --batch-size 256 --max-steps 20000 --lr 1e-3 --seed 1 \
        --wandb-tags "phase5,2h,autoreg,ctx60,lr-sweep"

    launch $MACHINE_B $PORT_B 4 cls-ar-ctx180-lr5e4 \
        $AR --seq-len 180 --batch-size 96 --max-steps 20000 --lr 5e-4 --seed 1 \
        --wandb-tags "phase5,2h,autoreg,ctx180,lr-sweep"

    launch $MACHINE_B $PORT_B 5 cls-ar-ctx180-lr1e3 \
        $AR --seq-len 180 --batch-size 96 --max-steps 20000 --lr 1e-3 --seed 1 \
        --wandb-tags "phase5,2h,autoreg,ctx180,lr-sweep"

    launch $MACHINE_B $PORT_B 6 cls-ctx180-20k \
        --seq-len 180 --batch-size 96 --max-steps 20000 --lr 8e-4 --seed 1 \
        --wandb-tags "phase5,2h,independent,ctx180"

    echo "  Machine B: 7 runs launched"
}

launch_machine_c() {
    echo "=== Machine C ($MACHINE_C:$PORT_C) — 8 GPUs ==="
    ssh -p $PORT_C $MACHINE_C "cd /root/FRAME && mkdir -p logs"

    # Phase 5c: Long runs (40K-65K steps, 4-6h)
    launch $MACHINE_C $PORT_C 0 cls-ar-ctx180-40k \
        $AR --seq-len 180 --batch-size 96 --max-steps 40000 --lr 8e-4 --seed 1 \
        --wandb-tags "phase5,long,autoreg,ctx180"

    launch $MACHINE_C $PORT_C 1 cls-ar-ctx180-40k-s2 \
        $AR --seq-len 180 --batch-size 96 --max-steps 40000 --lr 8e-4 --seed 2 \
        --wandb-tags "phase5,long,autoreg,ctx180"

    launch $MACHINE_C $PORT_C 2 cls-ar-ctx180-40k-s3 \
        $AR --seq-len 180 --batch-size 96 --max-steps 40000 --lr 8e-4 --seed 3 \
        --wandb-tags "phase5,long,autoreg,ctx180"

    launch $MACHINE_C $PORT_C 3 cls-ar-ctx60-40k \
        $AR --seq-len 60 --batch-size 256 --max-steps 40000 --lr 8e-4 --seed 1 \
        --wandb-tags "phase5,long,autoreg,ctx60"

    launch $MACHINE_C $PORT_C 4 cls-ar-ctx180-65k \
        $AR --seq-len 180 --batch-size 96 --max-steps 65000 --lr 8e-4 --seed 1 \
        --wandb-tags "phase5,long,autoreg,ctx180"

    launch $MACHINE_C $PORT_C 5 cls-ar-ctx60-65k \
        $AR --seq-len 60 --batch-size 256 --max-steps 65000 --lr 8e-4 --seed 1 \
        --wandb-tags "phase5,long,autoreg,ctx60"

    launch $MACHINE_C $PORT_C 6 cls-ar-ctx180-lr5e4-65k \
        $AR --seq-len 180 --batch-size 96 --max-steps 65000 --lr 5e-4 --seed 1 \
        --wandb-tags "phase5,long,autoreg,ctx180,lr-sweep"

    launch $MACHINE_C $PORT_C 7 cls-ar-ctx180-lr3e4-65k \
        $AR --seq-len 180 --batch-size 96 --max-steps 65000 --lr 3e-4 --seed 1 \
        --wandb-tags "phase5,long,autoreg,ctx180,lr-sweep"

    echo "  Machine C: 8 runs launched"
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
echo "=== Phase 5: 21 runs launched (4 quick + 9 x 2h + 8 long) ==="
echo "Monitor: wandb.ai/erickfm/MIMIC (group: phase5-clusters)"
echo "Logs: ssh <machine> 'tail -f /root/FRAME/logs/<run>.log'"
