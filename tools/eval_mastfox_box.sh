#!/usr/bin/env bash
# Build tail-SWA of the master-Fox run, then seed-matched eval of SWA, the run's
# bestloss, and the production champ on the master-Fox held-out val set.
set -uo pipefail
cd /workspace/MIMIC
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=disabled
COMMON="--encoder mimic_flat --mimic-mode --mimic-controller-encoding --stick-clusters hal37 --plain-ce --data-dir data/foxrank_mastfox --self-inputs --reaction-delay 0 --batch-size 128 --no-compile"
OUT=logs/mastfox_eval.txt; : > "$OUT"

python3 tools/average_checkpoints.py --run checkpoints/fox-mastfox-20260625 --last 5 \
  --out checkpoints/AVG_mastfox.pt 2>&1 | tail -2 | tee -a "$OUT"

for spec in "swa:checkpoints/AVG_mastfox.pt" \
            "bestloss:checkpoints/fox-mastfox-20260625_bestloss.pt" \
            "champ:checkpoints/fox-master-20260616-long_bestloss.pt"; do
  tag="${spec%%:*}"; ckpt="${spec#*:}"
  for seed in 42 123 7; do
    v=$(python3 train.py --eval-only "$ckpt" --model mimic --seed "$seed" $COMMON 2>/dev/null | grep -oE "val/total=[0-9.]+" | head -1)
    echo "${tag} seed=${seed} ${v}" | tee -a "$OUT"
  done
done
echo "MASTFOX_EVAL_DONE" | tee -a "$OUT"
