#!/usr/bin/env bash
# Train the world-model baseline on a prebuilt _v2 shard dir.
#
# Recipe (mirrors the BC baseline from retrain_all_baseline.sh, adapted for
# WM): torchrun --nproc_per_node=2, mimic-wm preset (20.5M params with the
# action_elapsed head), Huber numeric loss, bs=256 × 1 accum, 16.7M samples,
# bf16 AMP, relpos Shaw attention.
#
# Per memory: the WM learns physics, which is rank-invariant, so we prefer
# the _all_v2 shard dirs (all ranks) over _v2 master-only. Pass in whatever
# shard dir you want with $DATA_DIR; any dir with train_shard_*.pt works
# (tensor_manifest.json is optional — the dataset auto-splits if missing).
#
# Usage:
#   bash tools/train_wm_baseline.sh                         # fox default
#   DATA_DIR=data/falco_v2 CHAR=falco bash tools/train_wm_baseline.sh
#   NPROC=1 bash tools/train_wm_baseline.sh                 # single-GPU

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CHAR="${CHAR:-fox}"
# Prefer _all_v2 (all ranks, physics-invariant) if it exists, else _v2.
if [[ -z "${DATA_DIR:-}" ]]; then
  if [[ -d "${REPO_ROOT}/data/${CHAR}_all_v2" ]]; then
    DATA_DIR="${REPO_ROOT}/data/${CHAR}_all_v2"
  else
    DATA_DIR="${REPO_ROOT}/data/${CHAR}_v2"
  fi
fi

DATE_TAG="${DATE_TAG:-$(date -u +%Y%m%d)}"
RUN_NAME="${RUN_NAME:-${CHAR}-wm-${DATE_TAG}-baseline}"
NPROC="${NPROC:-2}"
MAX_SAMPLES="${MAX_SAMPLES:-16777216}"
BATCH_SIZE="${BATCH_SIZE:-256}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LR="${LR:-3e-4}"
SEQ_LEN="${SEQ_LEN:-180}"
NUM_WORKERS="${NUM_WORKERS:-4}"

echo "=== WM baseline train ==="
echo "  char:        ${CHAR}"
echo "  data_dir:    ${DATA_DIR}"
echo "  run_name:    ${RUN_NAME}"
echo "  nproc:       ${NPROC}  (torchrun --nproc_per_node=${NPROC})"
echo "  batch:       ${BATCH_SIZE} × ${GRAD_ACCUM} accum × ${NPROC} gpus "
echo "               = $((BATCH_SIZE * GRAD_ACCUM * NPROC)) effective"
echo "  max_samples: ${MAX_SAMPLES}"
echo "  seq_len:     ${SEQ_LEN}"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "ERROR: shard dir ${DATA_DIR} does not exist" >&2
  exit 1
fi

cd "${REPO_ROOT}"

torchrun --nproc_per_node="${NPROC}" --standalone tools/train_wm.py \
  --model mimic-wm \
  --data-dir "${DATA_DIR}" \
  --run-name "${RUN_NAME}" \
  --batch-size "${BATCH_SIZE}" \
  --grad-accum-steps "${GRAD_ACCUM}" \
  --max-samples "${MAX_SAMPLES}" \
  --seq-len "${SEQ_LEN}" \
  --lr "${LR}" \
  --num-workers "${NUM_WORKERS}" \
  --no-warmup --cosine-min-lr 1e-6 \
  --numeric-loss huber \
  --val-every 500 \
  --save-every 4000 \
  --wandb --wandb-project mimic-wm --wandb-tags baseline "${CHAR}"
