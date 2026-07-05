#!/usr/bin/env bash
# Big-and-long run: mimic-w1024 (82M), 960k steps from init, eff-batch 512,
# mirror aug, cosine to 1e-6, on the full 48K master-master set. Resilient:
# resumes from the latest step-checkpoint if the run crashes (flaky box).
set -uo pipefail
cd /workspace/MIMIC
RUN=fox-w1024-long-20260623
ARGS=(--model mimic-w1024 --encoder mimic_flat --mimic-mode --mimic-controller-encoding
      --stick-clusters hal37 --plain-ce
      --lr 3e-4 --batch-size 256 --grad-accum-steps 1
      --max-samples 491520000
      --data-dir data/foxrank_master_full
      --self-inputs --reaction-delay 0
      --seed 42 --no-warmup --cosine-min-lr 1e-6 --ema-decay 0.999
      --mirror-aug 0.5 --run-name "$RUN")

attempt=1
while [ "$attempt" -le 40 ]; do
  latest=$(ls -t checkpoints/${RUN}_step*.pt 2>/dev/null | head -1)
  if [ "$attempt" -eq 1 ] || [ -z "$latest" ]; then
    echo "[$(date +%H:%M)] attempt $attempt: fresh from init"
    torchrun --nproc_per_node=2 train.py "${ARGS[@]}" > "logs/${RUN}.log" 2>&1
  else
    echo "[$(date +%H:%M)] attempt $attempt: resume from $latest"
    torchrun --nproc_per_node=2 train.py "${ARGS[@]}" --resume "$latest" \
      > "logs/${RUN}.a${attempt}.log" 2>&1
  fi
  rc=$?
  if grep -qhE "^Done\. " logs/${RUN}*.log 2>/dev/null; then
    echo "[$(date +%H:%M)] COMPLETE (rc=$rc)"; break
  fi
  echo "[$(date +%H:%M)] exited rc=$rc — will resume in 30s"
  attempt=$((attempt+1)); sleep 30
done
echo "BIG_RUN_FINISHED $(date +%H:%M)"
