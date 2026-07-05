#!/usr/bin/env bash
# Train the 20M mimic model on the FULL clean master-Fox set (all ~102K games
# where the Fox itself is master, uncapped), to test the data-bound hypothesis
# vs the disk-capped 80K production model. Same recipe (3 GPU, eff-batch 768,
# mirror, 280k steps) so it's a data-only comparison. Reclaims the old 80K
# shards + non-master .slp for disk. wandb offline, resume-on-crash.
set -uo pipefail
cd /workspace/MIMIC
export WANDB_MODE=offline
STAGE=data/mastfox_full_slp
DATA=data/foxrank_mastfox_full
OLD=data/foxrank_mastfox          # old 80K shards (delete to free space)
RUN=fox-mastfox-full-20260626
mkdir -p "$STAGE" "$DATA" logs

echo "[KILL $(date +%H:%M)] stop leftovers"
pkill -f torchrun 2>/dev/null || true; pkill -9 -f "train.py" 2>/dev/null || true; sleep 2

echo "[META $(date +%H:%M)] copy norm from $OLD"
cp -n "$OLD"/*.json "$DATA/" 2>/dev/null || true
rm -f "$DATA/tensor_manifest.json" 2>/dev/null || true   # stale manifest from cp
[ -f "$DATA/mimic_norm.json" ] || { echo "ABORT: no mimic_norm.json"; exit 3; }

echo "[SCAN $(date +%H:%M)] re-scan master-Fox across all current slp dirs"
python3 tools/filter_masterfox.py data/mastfox_slp data/_held_slp data/fox_ranked_slp
nkeep=$(wc -l < data/masterfox_keep.txt)
echo "[SCAN] master-Fox total=$nkeep"
[ "$nkeep" -lt 90000 ] && { echo "ABORT: expected ~102K master-Fox, got $nkeep"; exit 5; }

echo "[STAGE $(date +%H:%M)] consolidate $nkeep master-Fox .slp into $STAGE (mv = free)"
xargs -a data/masterfox_keep.txt -P8 -I{} mv {} "$STAGE/" 2>/dev/null || true
staged=$(find "$STAGE" -maxdepth 1 -name '*.slp' | wc -l)
echo "[STAGE] staged=$staged of $nkeep"
# Safety: only delete the leftover (non-master) .slp if ~all master-Fox moved.
thresh=$(( nkeep * 98 / 100 ))
if [ "$staged" -lt "$thresh" ]; then echo "ABORT: only $staged/$nkeep staged — not deleting anything"; exit 6; fi

echo "[FREE $(date +%H:%M)] delete old 80K shards + leftover non-master .slp"
rm -rf "$OLD"
rm -f data/_held_slp/*.slp data/fox_ranked_slp/*.slp 2>/dev/null || true
echo "[FREE] $(df -h /workspace | tail -1)"

echo "[RS $(date +%H:%M)] reshard full master-Fox -> $DATA"
python3 tools/slp_to_shards.py --slp-dir "$STAGE" --meta-dir "$DATA" \
  --mimic-norm "$DATA/mimic_norm.json" --character 1 --staging-dir "$DATA" \
  --repo erickfm/mimic-fox-v2 --no-upload --keep-staging \
  --shard-gb 4.0 --val-frac 0.1 --seed 42 --workers 96
[ -f "$DATA/tensor_manifest.json" ] || { echo "ABORT: reshard produced no manifest"; exit 4; }
echo "[RS] done: shards=$(ls "$DATA"/*.pt 2>/dev/null | wc -l) size=$(du -sh "$DATA" | cut -f1)"
python3 -c "import json;m=json.load(open('$DATA/tensor_manifest.json'));print('[RS] train_games=%d val_games=%d frames=%d'%(m['n_train_games'],m['n_val_games'],m['n_train_frames']))"

echo "[TR $(date +%H:%M)] train 20M on full master-Fox (3 GPU, eff-batch 768)"
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
echo "FULL_MASTERFOX_FINISHED $(date +%H:%M)"
