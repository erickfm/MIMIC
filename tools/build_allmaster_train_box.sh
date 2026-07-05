#!/usr/bin/env bash
# End-to-end: build the all-master FOX set (master-master + master-diamond +
# master-platinum) and train the canonical 20M `mimic` model on it for ~10-11h
# of steps. Disk-aware: the full all-master shard set (~2.4TB) does NOT fit on
# this 2TB box, so we reclaim the redundant master-master shard dirs and cap the
# resharded game count to free disk minus a safety buffer (uses ALL games if
# they happen to fit). Resilient: training resumes from the latest step ckpt.
set -uo pipefail
cd /workspace/MIMIC
export HF_HUB_DISABLE_XET=1            # xet backend throttles to ~2 MB/s here

SLP=data/fox_ranked_slp               # already holds 48,596 master-master .slp
HELD=data/_held_slp                   # excess .slp parked here (same FS, free mv)
TAR=$SLP/_tars
DATA=data/foxrank_allmaster           # new all-master shard dir
META_SRC=data/foxrank_master_full     # has mimic_norm/cat_maps/combos/etc
RUN=fox-allmaster-20260625
mkdir -p "$SLP" "$TAR" "$HELD" logs "$DATA"

echo "[DL $(date +%H:%M)] download master-diamond + master-platinum (xet OFF)"
# The `hf` CLI treats a 2nd glob as a positional filename (404s). One call per
# tier, each with a single --include (the proven-working pattern). hf resumes.
for tier in master-diamond master-platinum; do
  echo "[DL $(date +%H:%M)] tier=$tier"
  hf download erickfm/melee-ranked-replays --repo-type dataset \
    --include "FOX/FOX_${tier}_a*.tar.gz" --local-dir "$TAR"
done
ntar=$(ls "$TAR"/FOX/FOX_master-{diamond,platinum}_a*.tar.gz 2>/dev/null | wc -l)
echo "[DL] tarballs=$ntar size=$(du -sh "$TAR" 2>/dev/null | cut -f1)"
[ "$ntar" -eq 0 ] && { echo "ABORT: no tarballs downloaded"; exit 2; }

echo "[EX $(date +%H:%M)] extract into $SLP"
for t in "$TAR"/FOX/FOX_master-diamond_a*.tar.gz "$TAR"/FOX/FOX_master-platinum_a*.tar.gz; do
  [ -f "$t" ] && tar -xzf "$t" -C "$SLP/" 2>/dev/null
done
rm -rf "$TAR"                          # reclaim tarball space
total_slp=$(find "$SLP" -maxdepth 1 -name '*.slp' | wc -l)
echo "[EX] total all-master .slp=$total_slp ($(du -sh "$SLP" | cut -f1))"

echo "[META $(date +%H:%M)] copy metadata from $META_SRC"
for j in mimic_norm cat_maps controller_combos stick_clusters norm_stats norm_minmax; do
  cp -n "$META_SRC/$j.json" "$DATA/" 2>/dev/null || true
done
[ -f "$DATA/mimic_norm.json" ] || { echo "ABORT: missing mimic_norm.json"; exit 3; }

echo "[FREE $(date +%H:%M)] reclaim redundant master-master shard dirs"
# Big-run checkpoints persist on /workspace; master_full shards are regenerable.
ls -lah checkpoints/fox-w1024-long-20260623_best*.pt 2>/dev/null
rm -rf data/foxrank_master_full data/foxrank_master_v2
echo "[FREE] $(df -h /workspace | tail -1)"

# --- disk-aware subset ---
free_kb=$(df --output=avail /workspace | tail -1)
buffer_kb=$((140*1024*1024))           # leave 140G headroom
budget_kb=$((free_kb - buffer_kb))
game_kb=13900                          # measured ~13.5 MB/game on master_full
ratio_num=1148; ratio_den=1000         # ~1.148 games per .slp
max_games=$(( budget_kb / game_kb ))
max_slp=$(( max_games * ratio_den / ratio_num ))
echo "[SUB] free=${free_kb}KB budget=${budget_kb}KB -> max_games=${max_games} max_slp=${max_slp} (have ${total_slp})"
if [ "$total_slp" -gt "$max_slp" ]; then
  echo "[SUB] capping: moving $((total_slp - max_slp)) excess .slp to $HELD (seeded shuf)"
  find "$SLP" -maxdepth 1 -name '*.slp' -printf '%f\n' \
    | shuf --random-source=<(yes 42) | tail -n +"$((max_slp + 1))" \
    | sed "s#^#$SLP/#" | xargs -P8 -I{} mv {} "$HELD/"
else
  echo "[SUB] full all-master fits — keeping all $total_slp .slp"
fi
echo "[SUB] resharding .slp=$(find "$SLP" -maxdepth 1 -name '*.slp' | wc -l)"

echo "[RS $(date +%H:%M)] reshard -> $DATA"
python3 tools/slp_to_shards.py --slp-dir "$SLP" --meta-dir "$DATA" \
  --mimic-norm "$DATA/mimic_norm.json" --character 1 --staging-dir "$DATA" \
  --repo erickfm/mimic-fox-v2 --no-upload --keep-staging \
  --shard-gb 4.0 --val-frac 0.1 --seed 42 --workers 96
[ -f "$DATA/tensor_manifest.json" ] || { echo "ABORT: reshard produced no manifest"; exit 4; }
echo "[RS] done: shards=$(ls "$DATA"/*.pt 2>/dev/null | wc -l) size=$(du -sh "$DATA" | cut -f1)"
python3 -c "import json;m=json.load(open('$DATA/tensor_manifest.json'));print('[RS] train_games=%d val_games=%d frames=%d'%(m['n_train_games'],m['n_val_games'],m['n_train_frames']))"

echo "[TR $(date +%H:%M)] launch 20M training on all-master, ~10-11h steps"
ARGS=(--model mimic --encoder mimic_flat --mimic-mode --mimic-controller-encoding
      --stick-clusters hal37 --plain-ce
      --lr 3e-4 --batch-size 256 --grad-accum-steps 1
      --max-samples 215040000              # ~280k steps @ eff-batch 768 (3 GPUs)
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
echo "ALLMASTER_RUN_FINISHED $(date +%H:%M)"
