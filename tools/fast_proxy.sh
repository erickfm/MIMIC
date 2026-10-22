#!/usr/bin/env bash
# Fast local research proxy (~5 min on an RTX 4090): a short run of the current recipe, tail-SWA of
# the last 5 checkpoints, fixed-subset val, one result line with wall time.
#
#   tools/fast_proxy.sh NAME PRESET STEPS SEED [extra train.py flags...]
#
# Muon (the default optimizer), no cond-heads (pass --cond-heads to test it), eff batch 768,
# validation only at the end. Results: checkpoints/fastproxy/results.tsv
set -euo pipefail
cd "$(dirname "$0")/.."
name=$1 model=$2 steps=$3 seed=$4; shift 4
data=${FAST_PROXY_DATA:-data/mastfox_c9_half}
bs=768 ga=1
[ "$model" = mimic ] && bs=384 && ga=2
out=checkpoints/fastproxy; mkdir -p $out/logs
t0=$(date +%s)
WANDB_MODE=${WANDB_MODE:-disabled} python3 train.py --model $model --encoder mimic_flat --mimic-mode \
    --mimic-controller-encoding --stick-clusters hal37 --plain-ce --lr 3e-4 --batch-size $bs \
    --grad-accum-steps $ga --max-samples $((bs * ga * steps)) --data-dir $data --self-inputs \
    --reaction-delay 0 --run-name $name --no-warmup --cosine-min-lr 1e-6 --val-frac 1.0 \
    --no-cond-heads --seed $seed "$@" > $out/logs/$name.log 2>&1
python3 tools/average_checkpoints.py --out $out/$name.avg.pt --run checkpoints/$name --last 5 \
    >> $out/logs/$name.log 2>&1
WANDB_MODE=disabled python3 train.py --model $model --encoder mimic_flat --mimic-mode \
    --mimic-controller-encoding --stick-clusters hal37 --plain-ce --batch-size 768 --data-dir $data \
    --self-inputs --reaction-delay 0 --seed 0 --no-cond-heads --eval-only $out/$name.avg.pt "$@" \
    > $out/logs/$name.eval.log 2>&1
line=$(grep "^EVAL_ONLY" $out/logs/$name.eval.log | tail -1)
printf "%s\t%s\t%s\t%s\t%s\t%ss\t%s\n" "$(date -u +%FT%TZ)" "$name" "$model" "$steps" "$seed" \
    "$(( $(date +%s) - t0 ))" "$line" >> $out/results.tsv
rm -f checkpoints/${name}_step*.pt checkpoints/${name}_best.pt checkpoints/${name}_bestloss.pt
