#!/usr/bin/env bash
# Robust (detached) subset + reshard: the full 43,737-game reshard is ~3.5h
# (slp_to_shards has a single-consumer bottleneck — load avg ~6 on 96 cores).
# Cut to ~18k games (~production scale, avoids overfit confound) for a ~75-min
# reshard that produces train+val+manifest. Run: nohup bash tools/subset_reshard_box.sh
set -uo pipefail
cd /workspace/MIMIC
SLP_DIR=data/fox_ranked_slp
HELD=data/_held_slp          # sibling dir, OUTSIDE slp-dir so reshard can't see it

echo "[$(date +%H:%M)] killing any running reshard"
pkill -9 -f slp_to_shards 2>/dev/null || true
sleep 4

mkdir -p "$HELD"
echo "[$(date +%H:%M)] subsetting: keep 18000, hold the rest"
find "$SLP_DIR" -maxdepth 1 -name '*.slp' -printf '%f\n' | shuf | tail -n +18001 \
  | xargs -P16 -I{} mv "$SLP_DIR/{}" "$HELD/"
echo "[$(date +%H:%M)] kept=$(find "$SLP_DIR" -maxdepth 1 -name '*.slp'|wc -l) held=$(ls "$HELD"/*.slp 2>/dev/null|wc -l)"

echo "[$(date +%H:%M)] clearing partial shards"
rm -f data/foxrank_master_v2/train_shard_*.pt data/foxrank_master_v2/val_shard_*.pt data/foxrank_master_v2/tensor_manifest.json

echo "[$(date +%H:%M)] reshard"
python3 tools/slp_to_shards.py \
  --slp-dir "$SLP_DIR" --meta-dir data/foxrank_master_v2 \
  --mimic-norm data/foxrank_master_v2/mimic_norm.json --character 1 \
  --staging-dir data/foxrank_master_v2 --repo erickfm/mimic-fox-v2 \
  --no-upload --keep-staging --shard-gb 4.0 --val-frac 0.1 --seed 42 --workers 96

echo "SUBSET_RESHARD_DONE $(date +%H:%M) shards=$(ls data/foxrank_master_v2/*.pt 2>/dev/null|wc -l)"
