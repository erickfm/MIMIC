#!/usr/bin/env bash
# sweep_no_opp_inputs.sh -- Launch 21 no-opp-inputs training runs across 3 machines
#
# Usage:
#   bash sweep_no_opp_inputs.sh [machines]
#
#   machines: "all" (default), "A", "B", "C"
#
# Prerequisites:
#   - data/full with metadata on each machine
#   - Latest code pushed: git push && ssh <machine> "cd /root/MIMIC && git pull"

set -euo pipefail

MACHINE_A="root@203.57.40.63"
PORT_A=10015
MACHINE_B="root@38.65.239.14"
PORT_B=28750
MACHINE_C="root@38.65.239.56"
PORT_C=45107

GROUP="no-opp-inputs"
DATA="data/full"
TAGS="no-opp-inputs,context"

COMMON="--data-dir $DATA --model medium --lr 8e-4 --encoder hybrid16 --pos-enc learned --no-opp-inputs --wandb-group $GROUP --wandb-tags $TAGS"

launch() {
    local host=$1 port=$2 gpu=$3 name=$4
    shift 4
    local extra="$*"
    echo "  GPU $gpu: $name"
    ssh -p "$port" "$host" "cd /root/MIMIC && \
        CUDA_VISIBLE_DEVICES=$gpu nohup python3 train.py \
        $COMMON --run-name $name $extra \
        > logs/$name.log 2>&1 &"
}

launch_machine_a() {
    echo "=== Machine A ($MACHINE_A:$PORT_A) — 6 GPUs: ctx-60 s1-s6 ==="
    ssh -p $PORT_A $MACHINE_A "cd /root/MIMIC && mkdir -p logs"

    # ctx-60 s1-s4: 65,000 steps (~6h)
    launch $MACHINE_A $PORT_A 0 noi-ctx60-s1 \
        --seq-len 60 --batch-size 384 --max-steps 65000 --seed 1

    launch $MACHINE_A $PORT_A 1 noi-ctx60-s2 \
        --seq-len 60 --batch-size 384 --max-steps 65000 --seed 2

    launch $MACHINE_A $PORT_A 2 noi-ctx60-s3 \
        --seq-len 60 --batch-size 384 --max-steps 65000 --seed 3

    launch $MACHINE_A $PORT_A 3 noi-ctx60-s4 \
        --seq-len 60 --batch-size 384 --max-steps 65000 --seed 4

    # ctx-60 s5-s6: 80,000 steps (~7.4h)
    launch $MACHINE_A $PORT_A 4 noi-ctx60-s5 \
        --seq-len 60 --batch-size 384 --max-steps 80000 --seed 5

    launch $MACHINE_A $PORT_A 5 noi-ctx60-s6 \
        --seq-len 60 --batch-size 384 --max-steps 80000 --seed 6

    echo "  Machine A: 6 runs launched"
}

launch_machine_b() {
    echo "=== Machine B ($MACHINE_B:$PORT_B) — 7 GPUs: ctx-60 s7-s11 + ctx-180 s1-s2 ==="
    ssh -p $PORT_B $MACHINE_B "cd /root/MIMIC && mkdir -p logs"

    # ctx-60 s7-s8: 80,000 steps (~7.4h)
    launch $MACHINE_B $PORT_B 0 noi-ctx60-s7 \
        --seq-len 60 --batch-size 384 --max-steps 80000 --seed 7

    launch $MACHINE_B $PORT_B 1 noi-ctx60-s8 \
        --seq-len 60 --batch-size 384 --max-steps 80000 --seed 8

    # ctx-60 s9-s10: 120,000 steps (~11.1h)
    launch $MACHINE_B $PORT_B 2 noi-ctx60-s9 \
        --seq-len 60 --batch-size 384 --max-steps 120000 --seed 9

    launch $MACHINE_B $PORT_B 3 noi-ctx60-s10 \
        --seq-len 60 --batch-size 384 --max-steps 120000 --seed 10

    # ctx-60 s11: 260,000 steps (~24h)
    launch $MACHINE_B $PORT_B 4 noi-ctx60-s11 \
        --seq-len 60 --batch-size 384 --max-steps 260000 --seed 11

    # ctx-180 s1-s2: 65,000 steps (~6.2h)
    launch $MACHINE_B $PORT_B 5 noi-ctx180-s1 \
        --seq-len 180 --batch-size 128 --max-steps 65000 --seed 1

    launch $MACHINE_B $PORT_B 6 noi-ctx180-s2 \
        --seq-len 180 --batch-size 128 --max-steps 65000 --seed 2

    echo "  Machine B: 7 runs launched"
}

launch_machine_c() {
    echo "=== Machine C ($MACHINE_C:$PORT_C) — 8 GPUs: ctx-180 s3-s10 ==="
    ssh -p $PORT_C $MACHINE_C "cd /root/MIMIC && mkdir -p logs"

    # ctx-180 s3-s4: 65,000 steps (~6.2h)
    launch $MACHINE_C $PORT_C 0 noi-ctx180-s3 \
        --seq-len 180 --batch-size 128 --max-steps 65000 --seed 3

    launch $MACHINE_C $PORT_C 1 noi-ctx180-s4 \
        --seq-len 180 --batch-size 128 --max-steps 65000 --seed 4

    # ctx-180 s5-s7: 80,000 steps (~7.7h)
    launch $MACHINE_C $PORT_C 2 noi-ctx180-s5 \
        --seq-len 180 --batch-size 128 --max-steps 80000 --seed 5

    launch $MACHINE_C $PORT_C 3 noi-ctx180-s6 \
        --seq-len 180 --batch-size 128 --max-steps 80000 --seed 6

    launch $MACHINE_C $PORT_C 4 noi-ctx180-s7 \
        --seq-len 180 --batch-size 128 --max-steps 80000 --seed 7

    # ctx-180 s8-s9: 120,000 steps (~11.5h)
    launch $MACHINE_C $PORT_C 5 noi-ctx180-s8 \
        --seq-len 180 --batch-size 128 --max-steps 120000 --seed 8

    launch $MACHINE_C $PORT_C 6 noi-ctx180-s9 \
        --seq-len 180 --batch-size 128 --max-steps 120000 --seed 9

    # ctx-180 s10: 250,000 steps (~24h)
    launch $MACHINE_C $PORT_C 7 noi-ctx180-s10 \
        --seq-len 180 --batch-size 128 --max-steps 250000 --seed 10

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
echo "=== No-Opp-Inputs Sweep: 21 runs launched (11 ctx-60 + 10 ctx-180) ==="
echo "Monitor: wandb.ai/erickfm/MIMIC (group: no-opp-inputs)"
echo "Logs: ssh <machine> 'tail -f /root/MIMIC/logs/<run>.log'"
