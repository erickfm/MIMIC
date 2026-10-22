#!/usr/bin/env bash
# One research-proxy run: train the champion recipe at reduced scale, tail-SWA the last 5
# checkpoints (as production does), score it on a fixed val subset, append one result line.
#
#   tools/proxy_run.sh NAME MODEL DATA_DIR STEPS SEED [extra train.py flags...]
#
# Single GPU, eff batch 768 (768 x 1, or 384 x 2 for the 512-wide model), lr 3e-4 cosine to 1e-6,
# no warmup, as the production runs. Pinned to the pre-2026-10-22 proxy settings (val every 1%,
# no cond-heads, AdamW) so results stay comparable with older proxies; extra flags override them.
# Results: checkpoints/proxy/results.tsv
set -euo pipefail
cd "$(dirname "$0")/.."
name=$1 model=$2 data=$3 steps=$4 seed=$5; shift 5
bs=768 ga=1
[ "$model" = mimic ] && bs=384 && ga=2
bs=${PROXY_BS:-$bs} ga=${PROXY_GA:-$ga}   # e.g. long-chunk runs keep tokens/step with fewer, longer windows
out=checkpoints/proxy; mkdir -p $out/logs
WANDB_MODE=${WANDB_MODE:-disabled} python3 train.py --model $model --encoder mimic_flat --mimic-mode \
    --mimic-controller-encoding --stick-clusters hal37 --plain-ce --lr 3e-4 --batch-size $bs \
    --grad-accum-steps $ga --max-samples $((bs * ga * steps)) --data-dir $data --self-inputs \
    --reaction-delay 0 --run-name $name --no-warmup --cosine-min-lr 1e-6 --lean-metrics \
    --val-frac 0.01 --no-cond-heads --optimizer adamw --seed $seed "$@" > $out/logs/$name.log 2>&1
python3 tools/average_checkpoints.py --out $out/$name.avg.pt --run checkpoints/$name --last 5 \
    >> $out/logs/$name.log 2>&1
# fixed val subset: same seed and batch size for every run
WANDB_MODE=disabled python3 train.py --model $model --encoder mimic_flat --mimic-mode \
    --mimic-controller-encoding --stick-clusters hal37 --plain-ce --batch-size ${PROXY_EVAL_BS:-768} --data-dir $data \
    --self-inputs --reaction-delay 0 --seed 0 --no-cond-heads --eval-only $out/$name.avg.pt "$@" \
    > $out/logs/$name.eval.log 2>&1
line=$(grep "^EVAL_ONLY" $out/logs/$name.eval.log | tail -1)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -u +%FT%TZ)" "$name" "$model" "$(basename $data)" "$steps" \
    "$seed" "$line" >> $out/results.tsv
rm -f checkpoints/${name}_step*.pt checkpoints/${name}_best.pt checkpoints/${name}_bestloss.pt   # keep the average + logs
