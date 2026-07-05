#!/usr/bin/env bash
# Train the 20M mimic model on the already-built all-master shards. Standalone
# (the build is done; data lives in data/foxrank_allmaster). WANDB offline since
# the reprovisioned box has no wandb key — metrics save locally, syncable later.
set -uo pipefail
cd /workspace/MIMIC
export WANDB_MODE=offline
DATA=data/foxrank_allmaster
RUN=fox-allmaster-20260625
mkdir -p logs
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
echo "ALLMASTER_TRAIN_FINISHED $(date +%H:%M)"
