#!/usr/bin/env bash
# After peach sharding finishes (CPU subprocess is running in background):
#   1. Train peach BASELINE (mimic preset, ~20M, full features, full 32k steps, NO watchdog)
#   2. Upload peach baseline to HF (overwrites prior xxl peach)
#   3. Clean peach raw + shards
#   4. Run falco continue (xxl, resume from step 9828, no watchdog)
#
# Why baseline for peach: xxl gives only ~2-3% val-loss improvement for 4× compute
# and 8× checkpoint storage — not worth it for production deployment.
# User decision 2026-04-20.

set -uo pipefail
REPO_ROOT="/root/MIMIC"
DATE_TAG="20260420"  # new run tag to distinguish from today's xxl attempts
QLOG="${REPO_ROOT}/checkpoints/xxl_queue.log"
log() { printf "[%s] %s\n" "$(date -u +%Y-%m-%d_%H:%M:%S)" "$*" | tee -a "${QLOG}" ; }

C="peach"; HF_BUCKET="PEACH"; IDX=9
SLP_DIR="${REPO_ROOT}/data/${C}_ranked_slp"
DATA_DIR="${REPO_ROOT}/data/${C}_v2"
RUN_NAME="${C}-${DATE_TAG}-baseline"
LOG_FILE="${REPO_ROOT}/checkpoints/${RUN_NAME}.log"

log "=== peach baseline + falco continue sequence ==="

# 1. Wait for peach shards to finish writing
log "[wait] polling for peach tensor_manifest.json"
until [[ -f "${DATA_DIR}/tensor_manifest.json" ]]; do sleep 30; done
log "[wait] peach shards ready"

# 2. Train peach BASELINE (mimic preset, full features, 32k steps, no watchdog)
log "[F/${C}] training ${RUN_NAME} — mimic preset (20M), full features, full 32k steps"
cd "${REPO_ROOT}"
unset WANDB_MODE
_wk="$(awk -F= '/^WANDB_API_KEY=/{print $2}' "${REPO_ROOT}/.env" 2>/dev/null | tr -d '[:space:]')"
if [[ -z "${_wk}" ]]; then export WANDB_MODE=disabled ; fi

: > "${LOG_FILE}"
torchrun --nproc_per_node=2 train.py \
  --model mimic --encoder mimic_flat \
  --mimic-mode --mimic-controller-encoding \
  --stick-clusters hal37 --plain-ce \
  --lr 3e-4 --batch-size 256 --grad-accum-steps 1 \
  --max-samples 16777216 \
  --data-dir "data/${C}_v2" \
  --self-inputs --reaction-delay 0 \
  --run-name "${RUN_NAME}" \
  --no-warmup --cosine-min-lr 1e-6 --nccl-timeout 3600 \
  > "${LOG_FILE}" 2>&1
train_rc=$?
log "[F/${C}] training done (rc=${train_rc})"

# 3. Upload + rename
BEST="${REPO_ROOT}/checkpoints/${RUN_NAME}_bestloss.pt"
if [[ -f "${BEST}" ]]; then
  best_step="$(grep -oP 'Best val_loss=[0-9.]+ @ step \K\d+' "${LOG_FILE}" 2>/dev/null | tail -1)"
  [[ -z "${best_step}" ]] && best_step="32000"
  tag="$(( (best_step + 500) / 1000 ))k"
  cp -f "${BEST}" "${REPO_ROOT}/checkpoints/${RUN_NAME}-${tag}.pt"
  log "[G/${C}] renamed → ${RUN_NAME}-${tag}.pt"
  log "[G/${C}] uploading → erickfm/MIMIC/${C}/ (replacing prior xxl peach)"
  python3 "${REPO_ROOT}/tools/upload_char.py" \
    --char "${C}" --checkpoint "${BEST}" \
    --data-dir "${DATA_DIR}" --log "${LOG_FILE}" 2>&1 | tee -a "${QLOG}" | tail -8
else
  log "[G/${C}] no ${BEST} — skipping upload"
fi

# 4. Clean up peach data (frees disk before falco shard rebuild)
log "[H/${C}] cleaning peach raw + shards"
rm -f "${DATA_DIR}/"*.pt 2>/dev/null || true
rm -rf "${SLP_DIR}" 2>/dev/null || true
df -h /root | tail -1 | tee -a "${QLOG}"
log "====== DONE peach baseline ======"

# 5. Falco xxl continue (keeps the earlier user request to continue falco)
log "=== handing off to continue_xxl_char.sh falco ==="
exec "${REPO_ROOT}/tools/continue_xxl_char.sh" falco FALCO 22
