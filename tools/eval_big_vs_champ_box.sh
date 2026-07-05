#!/usr/bin/env bash
# Eval the in-flight big w1024 run vs the production champ on the SAME held-out
# val set (foxrank_master_full, the w1024 run's own val split) at matched seeds.
# Single GPU (id 0) + small batch + no-compile so it doesn't disturb the running
# DDP training on both cards. Run detached on the box.
set -uo pipefail
cd /workspace/MIMIC
export CUDA_VISIBLE_DEVICES=0
COMMON="--encoder mimic_flat --mimic-mode --mimic-controller-encoding --stick-clusters hal37 --plain-ce --data-dir data/foxrank_master_full --self-inputs --reaction-delay 0 --batch-size 128 --no-compile"
OUT=logs/big_vs_champ_eval.txt; : > "$OUT"

for spec in "w1024:mimic-w1024:checkpoints/fox-w1024-long-20260623_bestloss.pt" \
            "champ:mimic:checkpoints/fox-master-20260616-long_bestloss.pt"; do
  tag="${spec%%:*}"; rest="${spec#*:}"; preset="${rest%%:*}"; ckpt="${rest##*:}"
  for seed in 42 123 7; do
    line=$(python3 train.py --eval-only "$ckpt" --model "$preset" --seed "$seed" $COMMON 2>/dev/null \
           | grep -iE "val[ /]total=" | tail -1)
    echo "${tag} seed=${seed} | ${line}" | tee -a "$OUT"
  done
done
echo "BIG_VS_CHAMP_DONE" | tee -a "$OUT"
