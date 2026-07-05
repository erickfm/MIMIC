#!/usr/bin/env bash
# Serial reshard (proven single-consumer path, --workers 96) of the already-
# staged full master-Fox set, then train. Used after the parallel path failed
# to scale. Staging (mastfox_full_slp) already done.
set -uo pipefail
cd /workspace/MIMIC
export WANDB_MODE=offline
STAGE=data/mastfox_full_slp
DATA=data/foxrank_mastfox_full
RUN=fox-mastfox-full-20260626
mkdir -p logs
[ -f "$DATA/mimic_norm.json" ] || { echo "ABORT: no mimic_norm.json"; exit 3; }
nslp=$(find "$STAGE" -maxdepth 1 -name '*.slp' | wc -l)
echo "[RS $(date +%H:%M)] SERIAL reshard, staged=$nslp"
[ "$nslp" -lt 90000 ] && { echo "ABORT: staged too few ($nslp)"; exit 5; }
rm -f "$DATA"/*.pt "$DATA/tensor_manifest.json" 2>/dev/null || true

python3 tools/slp_to_shards.py --slp-dir "$STAGE" --meta-dir "$DATA" \
  --mimic-norm "$DATA/mimic_norm.json" --character 1 --staging-dir "$DATA" \
  --repo erickfm/mimic-fox-v2 --no-upload --keep-staging \
  --shard-gb 4.0 --val-frac 0.1 --seed 42 --workers 96
[ -f "$DATA/tensor_manifest.json" ] || { echo "ABORT: reshard produced no manifest"; exit 4; }
echo "[RS] done: shards=$(ls "$DATA"/*.pt 2>/dev/null | wc -l) size=$(du -sh "$DATA" | cut -f1)"
python3 -c "import json;m=json.load(open('$DATA/tensor_manifest.json'));print('[RS] train_games=%d val_games=%d frames=%d'%(m['n_train_games'],m['n_val_games'],m['n_train_frames']))"

echo "[FREE $(date +%H:%M)] reshard done — removing staged .slp to free disk for training"
rm -rf "$STAGE"
echo "[FREE] $(df -h /workspace | tail -1)"

echo "[TR $(date +%H:%M)] train 20M on full master-Fox (3 GPU)"
ARGS=(--model mimic --encoder mimic_flat --mimic-mode --mimic-controller-encoding
      --stick-clusters hal37 --plain-ce --lr 3e-4 --batch-size 256 --grad-accum-steps 1
      --max-samples 215040000 --data-dir "$DATA" --self-inputs --reaction-delay 0
      --seed 42 --no-warmup --cosine-min-lr 1e-6 --ema-decay 0.999 --mirror-aug 0.5 --run-name "$RUN")
attempt=1
while [ "$attempt" -le 40 ]; do
  latest=$(ls -t checkpoints/${RUN}_step*.pt 2>/dev/null | head -1)
  if [ "$attempt" -eq 1 ] || [ -z "$latest" ]; then
    echo "[TR $(date +%H:%M)] attempt $attempt: fresh"
    torchrun --nproc_per_node=3 train.py "${ARGS[@]}" > "logs/${RUN}.log" 2>&1
  else
    echo "[TR $(date +%H:%M)] attempt $attempt: resume $latest"
    torchrun --nproc_per_node=3 train.py "${ARGS[@]}" --resume "$latest" > "logs/${RUN}.a${attempt}.log" 2>&1
  fi
  rc=$?
  if grep -qhE "^Done\. " logs/${RUN}*.log 2>/dev/null; then echo "[TR] COMPLETE rc=$rc"; break; fi
  echo "[TR $(date +%H:%M)] exited rc=$rc — resume in 30s"; attempt=$((attempt+1)); sleep 30
done
echo "FULL_MASTERFOX_SERIAL_FINISHED $(date +%H:%M)"
