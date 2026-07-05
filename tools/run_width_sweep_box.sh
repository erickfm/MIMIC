#!/usr/bin/env bash
# Width sweep on the 2x5090 box: does extra width help? 512 (baseline) / 768 / 1024,
# all from random init, 32k steps, seed 42, identical recipe (only --model differs).
# eff-batch = 256 * 2 gpus * 1 = 512 (matches the 4090 knee-cap baseline). Sequential
# (each run uses both GPUs via torchrun). Run detached: nohup bash tools/run_width_sweep_box.sh
set -uo pipefail
cd /workspace/MIMIC
DATA_DIR=data/foxrank_master_v2
mkdir -p logs
COMMON=(--encoder mimic_flat --mimic-mode --mimic-controller-encoding
        --stick-clusters hal37 --plain-ce
        --lr 3e-4 --batch-size 256 --grad-accum-steps 1
        --max-samples 16777216 --data-dir "$DATA_DIR"
        --self-inputs --reaction-delay 0
        --seed 42 --no-warmup --cosine-min-lr 1e-6 --ema-decay 0.999)

for spec in "mimic:512" "mimic-w768:768" "mimic-w1024:1024"; do
  preset="${spec%%:*}"; w="${spec##*:}"
  RUN="fox-width${w}-20260622"
  echo "=== $(date +%H:%M) START ${RUN} (preset=${preset}) ==="
  torchrun --nproc_per_node=2 train.py --model "${preset}" "${COMMON[@]}" \
    --run-name "${RUN}" > "logs/${RUN}.log" 2>&1
  echo "=== $(date +%H:%M) DONE ${RUN} :: $(grep -oE 'Best val_loss=[0-9.]+' logs/${RUN}.log | tail -1) ==="
done
echo "WIDTH_SWEEP_DONE $(date +%H:%M)"
