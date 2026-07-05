#!/usr/bin/env bash
# One-off: rebuild data/foxrank_master_v2 on the 2x5090 box from HF ranked
# replays, reusing the already-present metadata (mimic_norm/cat_maps/
# controller_combos/stick_clusters/norm_stats) so normalization matches the
# production baseline exactly. Run detached: nohup bash tools/build_fox_master_box.sh
set -uo pipefail
cd /workspace/MIMIC
DATA_DIR=data/foxrank_master_v2
SLP_DIR=data/fox_ranked_slp
TAR_DIR=$SLP_DIR/_tars
mkdir -p "$SLP_DIR" "$TAR_DIR" logs

echo "[A $(date +%H:%M)] downloading master-* Fox tarballs from HF"
hf download erickfm/melee-ranked-replays --repo-type dataset \
  --include "FOX/FOX_master-master_a*.tar.gz" "FOX/FOX_master-diamond_a*.tar.gz" "FOX/FOX_master-platinum_a*.tar.gz" \
  --local-dir "$TAR_DIR"
n_tar=$(ls "$TAR_DIR"/FOX/FOX_master-*_a*.tar.gz 2>/dev/null | wc -l)
echo "[A] fetched $n_tar tarballs ($(du -sh "$TAR_DIR" | cut -f1))"
[ "$n_tar" -eq 0 ] && { echo "ABORT: no tarballs"; exit 2; }

echo "[B $(date +%H:%M)] extracting"
for t in "$TAR_DIR"/FOX/FOX_master-*_a*.tar.gz; do tar -xzf "$t" -C "$SLP_DIR/"; done
n_slp=$(find "$SLP_DIR" -maxdepth 1 -name '*.slp' | wc -l)
echo "[B] $n_slp .slp files"

echo "[E $(date +%H:%M)] resharding -> $DATA_DIR (reusing existing norm/meta)"
python3 tools/slp_to_shards.py \
  --slp-dir "$SLP_DIR" --meta-dir "$DATA_DIR" \
  --mimic-norm "$DATA_DIR/mimic_norm.json" --character 1 \
  --staging-dir "$DATA_DIR" --no-upload --keep-staging \
  --shard-gb 4.0 --val-frac 0.1 --seed 42

n_shards=$(ls "$DATA_DIR"/*.pt 2>/dev/null | wc -l)
echo "BUILD_DONE $(date +%H:%M) n_shards=$n_shards size=$(du -sh "$DATA_DIR" | cut -f1)"
