#!/usr/bin/env bash
# sweep_arch.sh – Launch architecture search experiments across remote GPU machines
# ─────────────────────────────────────────────────────────────────────────────
# Usage:
#   bash sweep_arch.sh phase01      # Phase 0 (baselines) + Phase 1 (encoders)
#   bash sweep_arch.sh phase2       # Phase 2 (backbone scaling)
#   bash sweep_arch.sh phase3       # Phase 3 (output/loss ablations)
#   bash sweep_arch.sh phase4       # Phase 4 (champions)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Machine definitions ──────────────────────────────────────────────────────
MACHINE_A="root@203.57.40.63"
PORT_A=10015
GPUS_A=6

MACHINE_B="root@38.65.239.14"
PORT_B=28750
GPUS_B=7

MACHINE_C="root@38.65.239.56"
PORT_C=45107
GPUS_C=8

DATA_DIR="data/full"
WORK_DIR="/root/FRAME"

# ── Helper: launch a run on a remote machine ─────────────────────────────────
launch_run() {
    local HOST=$1 PORT=$2 GPU=$3 RUN_NAME=$4
    shift 4
    local EXTRA_ARGS="$*"

    echo "[launch] ${HOST}:${PORT} GPU=${GPU} name=${RUN_NAME} ${EXTRA_ARGS}"
    ssh -o StrictHostKeyChecking=no -p "${PORT}" "${HOST}" \
        "cd ${WORK_DIR} && nohup bash -c '
            export CUDA_VISIBLE_DEVICES=${GPU}
            python3 train.py --data-dir ${DATA_DIR} \
                --run-name ${RUN_NAME} \
                ${EXTRA_ARGS} \
                > logs/${RUN_NAME}.log 2>&1
        ' > /dev/null 2>&1 &"
}

# ── Helper: ensure log dir exists on a machine ───────────────────────────────
ensure_logdir() {
    local HOST=$1 PORT=$2
    ssh -o StrictHostKeyChecking=no -p "${PORT}" "${HOST}" "mkdir -p ${WORK_DIR}/logs"
}

# ── Phase 0 + Phase 1 (20 GPUs, simultaneous) ───────────────────────────────
run_phase01() {
    echo "=== Phase 0 + Phase 1: Baselines + Encoder Variants ==="
    echo "  Machine A: 6 runs (4 baselines + 2 encoder)"
    echo "  Machine B: 7 runs (encoder variants)"
    echo "  Machine C: 7 runs (encoder variants)"
    echo ""

    ensure_logdir "$MACHINE_A" "$PORT_A"
    ensure_logdir "$MACHINE_B" "$PORT_B"
    ensure_logdir "$MACHINE_C" "$PORT_C"

    # ── Machine A (6 GPUs): Phase 0 baselines + 2 encoder runs ──
    launch_run "$MACHINE_A" "$PORT_A" 0 baseline-small \
        --model small --lr 5e-4 --epochs 3 \
        --wandb-group phase0-baseline --wandb-tags "phase0,baseline,small"

    launch_run "$MACHINE_A" "$PORT_A" 1 baseline-base \
        --model base --lr 3e-4 --epochs 3 \
        --wandb-group phase0-baseline --wandb-tags "phase0,baseline,base"

    launch_run "$MACHINE_A" "$PORT_A" 2 baseline-small-long \
        --model small --lr 5e-4 --epochs 10 \
        --wandb-group phase0-baseline --wandb-tags "phase0,baseline,small,long"

    launch_run "$MACHINE_A" "$PORT_A" 3 baseline-base-long \
        --model base --lr 3e-4 --epochs 10 \
        --wandb-group phase0-baseline --wandb-tags "phase0,baseline,base,long"

    launch_run "$MACHINE_A" "$PORT_A" 4 flat-small \
        --model small --lr 5e-4 --epochs 1 --encoder flat \
        --wandb-group phase1-encoder --wandb-tags "phase1,encoder,flat,small"

    launch_run "$MACHINE_A" "$PORT_A" 5 flat-base \
        --model base --lr 3e-4 --epochs 1 --encoder flat \
        --wandb-group phase1-encoder --wandb-tags "phase1,encoder,flat,base"

    # ── Machine B (7 GPUs): Composite, hybrid, intra-depth, k-query ──
    launch_run "$MACHINE_B" "$PORT_B" 0 composite8-small \
        --model small --lr 5e-4 --epochs 1 --encoder composite8 \
        --wandb-group phase1-encoder --wandb-tags "phase1,encoder,composite8,small"

    launch_run "$MACHINE_B" "$PORT_B" 1 composite8-base \
        --model base --lr 3e-4 --epochs 1 --encoder composite8 \
        --wandb-group phase1-encoder --wandb-tags "phase1,encoder,composite8,base"

    launch_run "$MACHINE_B" "$PORT_B" 2 hybrid16-small \
        --model small --lr 5e-4 --epochs 1 --encoder hybrid16 \
        --wandb-group phase1-encoder --wandb-tags "phase1,encoder,hybrid16,small"

    launch_run "$MACHINE_B" "$PORT_B" 3 hybrid16-base \
        --model base --lr 3e-4 --epochs 1 --encoder hybrid16 \
        --wandb-group phase1-encoder --wandb-tags "phase1,encoder,hybrid16,base"

    launch_run "$MACHINE_B" "$PORT_B" 4 intra1-base \
        --model base --lr 3e-4 --epochs 1 --intra-layers 1 \
        --wandb-group phase1-encoder --wandb-tags "phase1,encoder,intra-depth"

    launch_run "$MACHINE_B" "$PORT_B" 5 intra0-base \
        --model base --lr 3e-4 --epochs 1 --intra-layers 0 \
        --wandb-group phase1-encoder --wandb-tags "phase1,encoder,intra-depth"

    launch_run "$MACHINE_B" "$PORT_B" 6 kquery4-base \
        --model base --lr 3e-4 --epochs 1 --k-query 4 \
        --wandb-group phase1-encoder --wandb-tags "phase1,encoder,kquery"

    # ── Machine C (7/8 GPUs): Scaled-emb, d-intra, dropout, joint ──
    launch_run "$MACHINE_C" "$PORT_C" 0 scaled-emb-base \
        --model base --lr 3e-4 --epochs 1 --scaled-emb \
        --wandb-group phase1-encoder --wandb-tags "phase1,encoder,scaled-emb"

    launch_run "$MACHINE_C" "$PORT_C" 1 dintra128-base \
        --model base --lr 3e-4 --epochs 1 --d-intra 128 \
        --wandb-group phase1-encoder --wandb-tags "phase1,encoder,dintra"

    launch_run "$MACHINE_C" "$PORT_C" 2 dintra512-base \
        --model base --lr 3e-4 --epochs 1 --d-intra 512 \
        --wandb-group phase1-encoder --wandb-tags "phase1,encoder,dintra"

    launch_run "$MACHINE_C" "$PORT_C" 3 drop10-base \
        --model base --lr 3e-4 --epochs 1 --dropout 0.10 \
        --wandb-group phase1-encoder --wandb-tags "phase1,encoder,dropout"

    launch_run "$MACHINE_C" "$PORT_C" 4 drop20-base \
        --model base --lr 3e-4 --epochs 1 --dropout 0.20 \
        --wandb-group phase1-encoder --wandb-tags "phase1,encoder,dropout"

    launch_run "$MACHINE_C" "$PORT_C" 5 composite8-deep \
        --model deep --lr 5e-4 --epochs 1 --encoder composite8 \
        --wandb-group phase1-encoder --wandb-tags "phase1,encoder,joint,composite8"

    launch_run "$MACHINE_C" "$PORT_C" 6 flat-deep \
        --model deep --lr 5e-4 --epochs 1 --encoder flat \
        --wandb-group phase1-encoder --wandb-tags "phase1,encoder,joint,flat"

    echo ""
    echo "=== Phase 0+1: 20 runs launched across 3 machines ==="
    echo "Monitor: https://wandb.ai/erickfm/FRAME"
}

# ── Phase 2: backbone scaling (placeholder, depends on Phase 1 results) ─────
run_phase2() {
    local BEST_ENC="${1:-default}"
    echo "=== Phase 2: Backbone Scaling (encoder=${BEST_ENC}) ==="

    ensure_logdir "$MACHINE_A" "$PORT_A"
    ensure_logdir "$MACHINE_B" "$PORT_B"
    ensure_logdir "$MACHINE_C" "$PORT_C"

    ENC_FLAG=""
    [ "$BEST_ENC" != "default" ] && ENC_FLAG="--encoder ${BEST_ENC}"

    # 2A: Model size scaling (Machine A, 6 GPUs + Machine B GPU 0-1)
    launch_run "$MACHINE_A" "$PORT_A" 0 scale-tiny \
        --model tiny --lr 1e-3 --epochs 1 $ENC_FLAG \
        --wandb-group phase2-backbone --wandb-tags "phase2,scale,tiny"

    launch_run "$MACHINE_A" "$PORT_A" 1 scale-small \
        --model small --lr 5e-4 --epochs 1 $ENC_FLAG \
        --wandb-group phase2-backbone --wandb-tags "phase2,scale,small"

    launch_run "$MACHINE_A" "$PORT_A" 2 scale-medium \
        --model medium --lr 3e-4 --epochs 1 $ENC_FLAG \
        --wandb-group phase2-backbone --wandb-tags "phase2,scale,medium"

    launch_run "$MACHINE_A" "$PORT_A" 3 scale-base \
        --model base --lr 3e-4 --epochs 1 $ENC_FLAG \
        --wandb-group phase2-backbone --wandb-tags "phase2,scale,base"

    launch_run "$MACHINE_A" "$PORT_A" 4 scale-deep \
        --model deep --lr 5e-4 --epochs 1 $ENC_FLAG \
        --wandb-group phase2-backbone --wandb-tags "phase2,scale,deep"

    launch_run "$MACHINE_A" "$PORT_A" 5 scale-wide-shallow \
        --model wide-shallow --lr 3e-4 --epochs 1 $ENC_FLAG \
        --wandb-group phase2-backbone --wandb-tags "phase2,scale,wide-shallow"

    launch_run "$MACHINE_B" "$PORT_B" 0 scale-xlarge \
        --model xlarge --lr 1e-4 --epochs 1 $ENC_FLAG \
        --wandb-group phase2-backbone --wandb-tags "phase2,scale,xlarge"

    launch_run "$MACHINE_B" "$PORT_B" 1 scale-xxlarge \
        --model xxlarge --lr 1e-4 --epochs 1 $ENC_FLAG \
        --wandb-group phase2-backbone --wandb-tags "phase2,scale,xxlarge"

    # 2B: Context length (Machine B GPU 2-7)
    launch_run "$MACHINE_B" "$PORT_B" 2 ctx-30 \
        --model base --lr 3e-4 --epochs 1 --seq-len 30 $ENC_FLAG \
        --wandb-group phase2-backbone --wandb-tags "phase2,context,30"

    launch_run "$MACHINE_B" "$PORT_B" 3 ctx-60 \
        --model base --lr 3e-4 --epochs 1 --seq-len 60 $ENC_FLAG \
        --wandb-group phase2-backbone --wandb-tags "phase2,context,60"

    launch_run "$MACHINE_B" "$PORT_B" 4 ctx-120 \
        --model base --lr 3e-4 --epochs 1 --seq-len 120 --batch-size 100 $ENC_FLAG \
        --wandb-group phase2-backbone --wandb-tags "phase2,context,120"

    launch_run "$MACHINE_B" "$PORT_B" 5 ctx-180 \
        --model base --lr 3e-4 --epochs 1 --seq-len 180 --batch-size 64 $ENC_FLAG \
        --wandb-group phase2-backbone --wandb-tags "phase2,context,180"

    launch_run "$MACHINE_B" "$PORT_B" 6 ctx-240 \
        --model base --lr 2e-4 --epochs 1 --seq-len 240 --batch-size 48 $ENC_FLAG \
        --wandb-group phase2-backbone --wandb-tags "phase2,context,240"

    # 2C: Positional encoding (Machine C GPUs 0-2)
    launch_run "$MACHINE_C" "$PORT_C" 0 posenc-learned \
        --model base --lr 3e-4 --epochs 1 --pos-enc learned $ENC_FLAG \
        --wandb-group phase2-backbone --wandb-tags "phase2,posenc,learned"

    launch_run "$MACHINE_C" "$PORT_C" 1 posenc-rope \
        --model base --lr 3e-4 --epochs 1 --pos-enc rope $ENC_FLAG \
        --wandb-group phase2-backbone --wandb-tags "phase2,posenc,rope"

    launch_run "$MACHINE_C" "$PORT_C" 2 posenc-sinusoidal \
        --model base --lr 3e-4 --epochs 1 --pos-enc sinusoidal $ENC_FLAG \
        --wandb-group phase2-backbone --wandb-tags "phase2,posenc,sinusoidal"

    # 2D: Attention variants (Machine C GPUs 3-5)
    launch_run "$MACHINE_C" "$PORT_C" 3 attn-sliding \
        --model base --lr 3e-4 --epochs 1 --attn-variant sliding $ENC_FLAG \
        --wandb-group phase2-backbone --wandb-tags "phase2,attn,sliding"

    launch_run "$MACHINE_C" "$PORT_C" 4 attn-alibi \
        --model base --lr 3e-4 --epochs 1 --pos-enc alibi $ENC_FLAG \
        --wandb-group phase2-backbone --wandb-tags "phase2,attn,alibi"

    launch_run "$MACHINE_C" "$PORT_C" 5 attn-gqa \
        --model base --lr 3e-4 --epochs 1 --n-kv-heads 2 $ENC_FLAG \
        --wandb-group phase2-backbone --wandb-tags "phase2,attn,gqa"

    echo ""
    echo "=== Phase 2: 20 runs launched ==="
}

# ── Phase 3: output/loss ablations (8 GPUs) ─────────────────────────────────
run_phase3() {
    local BEST_ENC="${1:-default}"
    local BEST_MODEL="${2:-base}"
    local BEST_LR="${3:-3e-4}"
    echo "=== Phase 3: Output/Loss Ablations (encoder=${BEST_ENC}, model=${BEST_MODEL}) ==="

    ensure_logdir "$MACHINE_A" "$PORT_A"

    ENC_FLAG=""
    [ "$BEST_ENC" != "default" ] && ENC_FLAG="--encoder ${BEST_ENC}"

    # 3A: Loss function variants
    launch_run "$MACHINE_A" "$PORT_A" 0 loss-huber \
        --model "$BEST_MODEL" --lr "$BEST_LR" --epochs 1 --stick-loss huber $ENC_FLAG \
        --wandb-group phase3-output --wandb-tags "phase3,loss,huber"

    launch_run "$MACHINE_A" "$PORT_A" 1 loss-quantile \
        --model "$BEST_MODEL" --lr "$BEST_LR" --epochs 1 --stick-loss quantile $ENC_FLAG \
        --wandb-group phase3-output --wandb-tags "phase3,loss,quantile"

    launch_run "$MACHINE_A" "$PORT_A" 2 loss-discrete \
        --model "$BEST_MODEL" --lr "$BEST_LR" --epochs 1 --stick-loss discrete $ENC_FLAG \
        --wandb-group phase3-output --wandb-tags "phase3,loss,discrete"

    launch_run "$MACHINE_A" "$PORT_A" 3 loss-mse-baseline \
        --model "$BEST_MODEL" --lr "$BEST_LR" --epochs 1 --stick-loss mse $ENC_FLAG \
        --wandb-group phase3-output --wandb-tags "phase3,loss,mse"

    # 3B: Target representation
    launch_run "$MACHINE_A" "$PORT_A" 4 target-delta \
        --model "$BEST_MODEL" --lr "$BEST_LR" --epochs 1 --delta-targets $ENC_FLAG \
        --wandb-group phase3-output --wandb-tags "phase3,target,delta"

    launch_run "$MACHINE_A" "$PORT_A" 5 target-absolute \
        --model "$BEST_MODEL" --lr "$BEST_LR" --epochs 1 $ENC_FLAG \
        --wandb-group phase3-output --wandb-tags "phase3,target,absolute"

    echo ""
    echo "=== Phase 3: 6 runs launched ==="
}

# ── Phase 4: champion runs (4 GPUs) ─────────────────────────────────────────
run_phase4() {
    echo "=== Phase 4: Champion Runs ==="
    echo "  Configure manually based on Phase 1-3 results."
    echo "  Template:"
    echo "    launch_run MACHINE PORT GPU champion-1 --model <best> --lr <best> --epochs 10 --encoder <best> ..."
}

# ── Main dispatch ────────────────────────────────────────────────────────────
case "${1:-help}" in
    phase01)
        run_phase01
        ;;
    phase2)
        run_phase2 "${2:-default}"
        ;;
    phase3)
        run_phase3 "${2:-default}" "${3:-base}" "${4:-3e-4}"
        ;;
    phase4)
        run_phase4
        ;;
    *)
        echo "Usage: bash sweep_arch.sh {phase01|phase2|phase3|phase4}"
        echo ""
        echo "  phase01           Run Phase 0 (baselines) + Phase 1 (encoder variants)"
        echo "  phase2 [encoder]  Run Phase 2 (backbone scaling) with winning encoder"
        echo "  phase3 [enc] [model] [lr]  Run Phase 3 (loss ablations)"
        echo "  phase4            Print Phase 4 template"
        exit 1
        ;;
esac
