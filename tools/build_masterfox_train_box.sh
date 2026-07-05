#!/usr/bin/env bash
# Rebuild the dataset as MASTER-FOX-ONLY (Fox itself is master rank, opponents
# any rank), then train the 20M mimic model on it. Discards the mixed-rank
# shards. Disk-aware cap, 3-GPU training, wandb offline, resume-on-crash.
set -uo pipefail
cd /workspace/MIMIC
export WANDB_MODE=offline
export HF_HUB_DISABLE_XET=1
DATA=data/foxrank_mastfox      # new clean shard dir
STAGE=data/mastfox_slp         # only the kept master-Fox .slp (reshard source)
RUN=fox-mastfox-20260625
mkdir -p "$DATA" "$STAGE" logs

echo "[KILL $(date +%H:%M)] stop any leftover training"
pkill -f train_allmaster_box.sh 2>/dev/null || true
pkill -f torchrun 2>/dev/null || true
pkill -9 -f "train.py" 2>/dev/null || true
sleep 3

echo "[META $(date +%H:%M)] preserve metadata, then reclaim mixed shards"
cp -n data/foxrank_allmaster/*.json "$DATA/" 2>/dev/null || true
[ -f "$DATA/mimic_norm.json" ] || { echo "ABORT: no mimic_norm.json to seed $DATA"; exit 3; }
rm -rf data/foxrank_allmaster
echo "[META] $(df -h /workspace | tail -1)"

echo "[FILTER $(date +%H:%M)] scan pool for master-Fox games"
python3 tools/filter_masterfox.py
nkeep=$(wc -l < data/masterfox_keep.txt)
echo "[FILTER] master_fox files=$nkeep"
[ "$nkeep" -lt 1000 ] && { echo "ABORT: implausibly few master-Fox games ($nkeep)"; exit 5; }

# --- disk-aware cap ---
free_kb=$(df --output=avail /workspace | tail -1)
budget_kb=$((free_kb - 140*1024*1024))
game_kb=13900; rn=1148; rd=1000
max_games=$(( budget_kb / game_kb ))
max_slp=$(( max_games * rd / rn ))
echo "[CAP] free=${free_kb}KB -> max_slp=${max_slp} (have ${nkeep})"
if [ "$nkeep" -gt "$max_slp" ]; then keepn="$max_slp"; else keepn="$nkeep"; fi
shuf --random-source=<(yes 42) data/masterfox_keep.txt | head -n "$keepn" > data/mastfox_select.txt
echo "[CAP] selecting $(wc -l < data/mastfox_select.txt) files into $STAGE"
xargs -a data/mastfox_select.txt -P8 -I{} mv {} "$STAGE/" 2>/dev/null || true
echo "[STAGE] staged=$(find "$STAGE" -maxdepth 1 -name '*.slp' | wc -l)"

echo "[RS $(date +%H:%M)] reshard master-Fox -> $DATA"
python3 tools/slp_to_shards.py --slp-dir "$STAGE" --meta-dir "$DATA" \
  --mimic-norm "$DATA/mimic_norm.json" --character 1 --staging-dir "$DATA" \
  --repo erickfm/mimic-fox-v2 --no-upload --keep-staging \
  --shard-gb 4.0 --val-frac 0.1 --seed 42 --workers 96
[ -f "$DATA/tensor_manifest.json" ] || { echo "ABORT: reshard produced no manifest"; exit 4; }
echo "[RS] done: shards=$(ls "$DATA"/*.pt 2>/dev/null | wc -l) size=$(du -sh "$DATA" | cut -f1)"
python3 -c "import json;m=json.load(open('$DATA/tensor_manifest.json'));print('[RS] train_games=%d val_games=%d frames=%d'%(m['n_train_games'],m['n_val_games'],m['n_train_frames']))"

echo "[TR $(date +%H:%M)] launch 20M training on master-Fox (3 GPUs, wandb offline)"
ARGS=(--model mimic --encoder mimic_flat --mimic-mode --mimic-controller-encoding
      --stick-clusters hal37 --plain-ce
      --lr 3e-4 --batch-size 256 --grad-accum-steps 1
      --max-samples 215040000
      --data-dir "$DATA" --self-inputs --reaction-delay 0
      --seed 42 --no-warmup --cosine-min-lr 1e-6 --ema-decay 0.999
      --mirror-aug 0.5 --run-name "$RUN")
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
echo "MASTERFOX_RUN_FINISHED $(date +%H:%M)"
