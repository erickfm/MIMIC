#!/usr/bin/env bash
# Faster retry: master-master tier only, xet DISABLED (plain HTTP) — the xet
# backend throttled to ~2 MB/s. master-master is the highest-quality tier and
# plenty for a 32k directional width test (all widths share it).
set -uo pipefail
cd /workspace/MIMIC
export HF_HUB_DISABLE_XET=1
DATA_DIR=data/foxrank_master_v2
SLP_DIR=data/fox_ranked_slp
TAR_DIR=$SLP_DIR/_tars
mkdir -p "$SLP_DIR" "$TAR_DIR" logs

echo "[A2 $(date +%H:%M)] download master-master only, xet OFF (HTTP)"
hf download erickfm/melee-ranked-replays --repo-type dataset \
  --include "FOX/FOX_master-master_a*.tar.gz" --local-dir "$TAR_DIR"
n_tar=$(ls "$TAR_DIR"/FOX/FOX_master-master_a*.tar.gz 2>/dev/null | wc -l)
echo "[A2] $n_tar tarballs ($(du -sh "$TAR_DIR" | cut -f1))"
[ "$n_tar" -eq 0 ] && { echo "ABORT no tarballs"; exit 2; }

echo "[B2 $(date +%H:%M)] extract"
for t in "$TAR_DIR"/FOX/FOX_master-master_a*.tar.gz; do tar -xzf "$t" -C "$SLP_DIR/"; done
echo "n_slp=$(find "$SLP_DIR" -maxdepth 1 -name '*.slp' | wc -l)"

echo "[E2 $(date +%H:%M)] reshard"
python3 tools/slp_to_shards.py --slp-dir "$SLP_DIR" --meta-dir "$DATA_DIR" \
  --mimic-norm "$DATA_DIR/mimic_norm.json" --character 1 --staging-dir "$DATA_DIR" \
  --no-upload --keep-staging --shard-gb 4.0 --val-frac 0.1 --seed 42
echo "BUILD2_DONE $(date +%H:%M) shards=$(ls "$DATA_DIR"/*.pt 2>/dev/null | wc -l) size=$(du -sh "$DATA_DIR" | cut -f1)"
