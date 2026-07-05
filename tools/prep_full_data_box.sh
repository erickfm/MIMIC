#!/usr/bin/env bash
# Restore the full 48K master-master games and reshard into a NEW dir
# (foxrank_master_full) so the 18K width-eval set is left intact. On success,
# chain-launch the big 1024 long run. Detached: nohup bash tools/prep_full_data_box.sh
set -uo pipefail
cd /workspace/MIMIC
SLP_DIR=data/fox_ranked_slp
HELD=data/_held_slp
SRC=data/foxrank_master_v2          # has the metadata jsons
DATA_DIR=data/foxrank_master_full

mkdir -p "$DATA_DIR"
for j in mimic_norm cat_maps controller_combos stick_clusters norm_stats norm_minmax; do
  cp -n "$SRC/$j.json" "$DATA_DIR/" 2>/dev/null || true
done

echo "[$(date +%H:%M)] restoring held games"
if [ -d "$HELD" ]; then
  find "$HELD" -maxdepth 1 -name '*.slp' -print0 | xargs -0 -P16 -I{} mv {} "$SLP_DIR/" 2>/dev/null || true
fi
echo "[$(date +%H:%M)] total slp=$(find "$SLP_DIR" -maxdepth 1 -name '*.slp' | wc -l)"

echo "[$(date +%H:%M)] reshard full set -> $DATA_DIR"
python3 tools/slp_to_shards.py \
  --slp-dir "$SLP_DIR" --meta-dir "$DATA_DIR" \
  --mimic-norm "$DATA_DIR/mimic_norm.json" --character 1 \
  --staging-dir "$DATA_DIR" --repo erickfm/mimic-fox-v2 \
  --no-upload --keep-staging --shard-gb 4.0 --val-frac 0.1 --seed 42 --workers 96

if [ -f "$DATA_DIR/tensor_manifest.json" ]; then
  echo "[$(date +%H:%M)] reshard done (shards=$(ls "$DATA_DIR"/*.pt|wc -l)); launching big training"
  nohup bash tools/train_big_long_box.sh > logs/train_big_launcher.log 2>&1 &
  echo "training launcher pid $!"
else
  echo "[$(date +%H:%M)] RESHARD FAILED — no manifest; NOT launching training"
fi
