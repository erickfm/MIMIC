#!/usr/bin/env bash
# Hand-rolled finish for ice_climbers after the bash orchestrator was
# killed mid-way to disable the mis-calibrated watchdog. Preserves the
# running norm_stats python subprocess; waits for it to write
# norm_stats.json, then runs shard + train + upload + cleanup with
# NO watchdog so the run gets all 32,768 cosine-decay steps.

set -uo pipefail
REPO_ROOT="/root/MIMIC"
C="ice_climbers"
HF_BUCKET="ICE_CLIMBERS"
IDX=10
DATE_TAG="20260419"
SLP_DIR="${REPO_ROOT}/data/${C}_ranked_slp"
DATA_DIR="${REPO_ROOT}/data/${C}_v2"
RUN_NAME="${C}-${DATE_TAG}-xxl"
LOG_FILE="${REPO_ROOT}/checkpoints/${RUN_NAME}.log"
QLOG="${REPO_ROOT}/checkpoints/xxl_queue.log"
STATE_FILE="${REPO_ROOT}/checkpoints/xxl_queue_state"

log() { printf "[%s] %s\n" "$(date -u +%Y-%m-%d_%H:%M:%S)" "$*" | tee -a "${QLOG}" ; }

log "=== finish_ice_climbers: waiting for norm_stats.json to appear ==="
until [[ -f "${DATA_DIR}/norm_stats.json" ]]; do sleep 30; done
log "norm_stats.json present"

# Build mimic_norm.json if not already done
if [[ ! -f "${DATA_DIR}/mimic_norm.json" ]]; then
  log "building mimic_norm.json"
  python3 "${REPO_ROOT}/tools/build_mimic_norm.py" \
    --norm-stats "${DATA_DIR}/norm_stats.json" \
    --minmax    "${DATA_DIR}/norm_minmax.json" \
    --out       "${DATA_DIR}/mimic_norm.json" 2>&1 | tee -a "${QLOG}" | tail -10
fi

# stick_clusters + controller_combos
if [[ ! -f "${DATA_DIR}/stick_clusters.json" ]]; then
  cp "${REPO_ROOT}/hf_checkpoints/fox/stick_clusters.json" "${DATA_DIR}/stick_clusters.json"
  log "copied stick_clusters.json"
fi
if [[ ! -f "${DATA_DIR}/controller_combos.json" ]]; then
  cat > "${DATA_DIR}/controller_combos.json" <<'JSON'
{
    "button_names": ["A", "B", "Z", "JUMP", "TRIG", "A_TRIG", "NONE"],
    "n_combos": 7,
    "class_scheme": "melee_7class"
}
JSON
  log "wrote controller_combos.json"
fi

# Shard
if [[ ! -f "${DATA_DIR}/tensor_manifest.json" ]]; then
  log "[E/${C}] sharding (idx=${IDX})"
  python3 "${REPO_ROOT}/tools/slp_to_shards.py" \
    --slp-dir "${SLP_DIR}" \
    --meta-dir "${DATA_DIR}" \
    --mimic-norm "${DATA_DIR}/mimic_norm.json" \
    --character "${IDX}" \
    --staging-dir "${DATA_DIR}" \
    --repo "erickfm/mimic-${C}-v2" \
    --no-upload --keep-staging \
    --shard-gb 4.0 --val-frac 0.1 --seed 42 \
    2>&1 | tee -a "${QLOG}" | tail -40
  if [[ ! -f "${DATA_DIR}/tensor_manifest.json" ]]; then
    log "[E/${C}] shard FAILED"
    exit 1
  fi
else
  log "[E/${C}] shards already present"
fi

# Train — NO watchdog, full 32,768 steps
log "[F/${C}] training ${RUN_NAME} WITHOUT watchdog (full cosine decay)"
cd "${REPO_ROOT}"
unset WANDB_MODE
_wk="$(awk -F= '/^WANDB_API_KEY=/{print $2}' "${REPO_ROOT}/.env" 2>/dev/null | tr -d '[:space:]')"
if [[ -z "${_wk}" ]]; then export WANDB_MODE=disabled ; fi

: > "${LOG_FILE}"
torchrun --nproc_per_node=2 train.py \
  --model mimic-xxl --encoder mimic_flat \
  --mimic-mode --mimic-controller-encoding \
  --stick-clusters hal37 --plain-ce \
  --lr 3e-4 --batch-size 64 --grad-accum-steps 4 \
  --max-samples 16777216 \
  --data-dir "data/${C}_v2" \
  --self-inputs --reaction-delay 0 \
  --run-name "${RUN_NAME}" \
  --no-warmup --cosine-min-lr 1e-6 --nccl-timeout 3600 \
  > "${LOG_FILE}" 2>&1
train_rc=$?
log "[F/${C}] training done (rc=${train_rc})"

# Upload + cleanup (treat any rc as success if bestloss.pt exists)
BEST="${REPO_ROOT}/checkpoints/${RUN_NAME}_bestloss.pt"
if [[ -f "${BEST}" ]]; then
  best_step="$(grep -oP 'Best val_loss=[0-9.]+ @ step \K\d+' "${LOG_FILE}" 2>/dev/null | tail -1)"
  [[ -z "${best_step}" ]] && best_step="32000"
  tag="$(( (best_step + 500) / 1000 ))k"
  cp -f "${BEST}" "${REPO_ROOT}/checkpoints/${RUN_NAME}-${tag}.pt"
  log "[G/${C}] renamed → ${RUN_NAME}-${tag}.pt"
  log "[G/${C}] uploading → erickfm/MIMIC/${C}/"
  python3 "${REPO_ROOT}/tools/upload_char.py" \
    --char "${C}" --checkpoint "${BEST}" \
    --data-dir "${DATA_DIR}" --log "${LOG_FILE}" 2>&1 | tee -a "${QLOG}" | tail -10
  log "[H/${C}] cleaning up"
  rm -f "${DATA_DIR}/"*.pt 2>/dev/null || true
  rm -rf "${SLP_DIR}" 2>/dev/null || true
  echo "train-done-${C}" >> "${STATE_FILE}"
  log "====== DONE ${C} ======"
else
  log "[G/${C}] no ${BEST} — not uploading"
  exit 1
fi
