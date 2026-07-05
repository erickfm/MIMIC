#!/usr/bin/env bash
# Score the width sweep seed-matched: build tail-SWA (last 5 step-ckpts) per width,
# then --eval-only at seeds 42/123/7 with the matching preset. Run on the box.
set -uo pipefail
cd /workspace/MIMIC
COMMON="--encoder mimic_flat --mimic-mode --mimic-controller-encoding --stick-clusters hal37 --plain-ce --data-dir data/foxrank_master_v2 --self-inputs --reaction-delay 0 --batch-size 256 --no-compile"
OUT=logs/width_eval.txt; : > "$OUT"

for spec in "512:mimic" "768:mimic-w768" "1024:mimic-w1024"; do
  w="${spec%%:*}"; preset="${spec##*:}"
  python3 tools/average_checkpoints.py --run "checkpoints/fox-width${w}-20260622" --last 5 \
    --out "checkpoints/AVG_width${w}.pt" 2>&1 | tail -1
done

for spec in "512:mimic" "768:mimic-w768" "1024:mimic-w1024"; do
  w="${spec%%:*}"; preset="${spec##*:}"
  for seed in 42 123 7; do
    for tag in "swa:checkpoints/AVG_width${w}.pt" "raw:checkpoints/fox-width${w}-20260622_bestloss.pt"; do
      kind="${tag%%:*}"; ckpt="${tag##*:}"
      v=$(python3 train.py --eval-only "$ckpt" --model "$preset" --seed "$seed" $COMMON 2>/dev/null | grep -oE "val/total=[0-9.]+" | head -1)
      echo "width${w} ${kind} seed=${seed} ${v}" | tee -a "$OUT"
    done
  done
done
echo "WIDTH_EVAL_DONE" | tee -a "$OUT"
