#!/usr/bin/env bash
# Continues an xxl training run that was killed mid-cosine. Re-does data
# prep (since we cleaned shards between runs) then launches train.py
# with --resume pointing at the latest step checkpoint, which restores
# optimizer/scheduler/step counter so training picks up exactly where it
# stopped — right through the full cosine tail.
#
# Usage: tools/continue_xxl_char.sh <char> <HF_BUCKET> <character_idx>
# Example: tools/continue_xxl_char.sh peach PEACH 9

set -uo pipefail
if (( $# != 3 )); then
  echo "usage: $0 <char> <HF_BUCKET> <character_idx>" >&2
  exit 2
fi
C="$1"; HF_BUCKET="$2"; IDX="$3"
REPO_ROOT="/root/MIMIC"
DATE_TAG="20260419"
RUN_NAME="${C}-${DATE_TAG}-xxl"
SLP_DIR="${REPO_ROOT}/data/${C}_ranked_slp"
TAR_DIR="${SLP_DIR}/_tars"
DATA_DIR="${REPO_ROOT}/data/${C}_v2"
LOG_FILE="${REPO_ROOT}/checkpoints/${RUN_NAME}.log"
QLOG="${REPO_ROOT}/checkpoints/xxl_queue.log"

log() { printf "[%s] %s\n" "$(date -u +%Y-%m-%d_%H:%M:%S)" "$*" | tee -a "${QLOG}" ; }

# ---- find latest step checkpoint ----
LATEST_CKPT="$(ls -v "${REPO_ROOT}/checkpoints/${RUN_NAME}_step"*.pt 2>/dev/null | tail -1)"
if [[ -z "${LATEST_CKPT}" ]]; then
  log "ERROR: no step checkpoint found for ${RUN_NAME}; cannot continue"
  exit 1
fi
log "=== continue_xxl ${C}: resuming from $(basename "${LATEST_CKPT}") ==="

# ---- wait for any in-flight tar extract from the killed queue ----
log "waiting for any in-flight tar subprocess to finish"
while pgrep -f "tar -xzf.*${C}_ranked_slp" > /dev/null 2>&1; do sleep 10; done
log "tar subprocesses clear"

# ---- download (idempotent via HF cache) + extract (idempotent via sentinel) ----
mkdir -p "${SLP_DIR}" "${TAR_DIR}" "${DATA_DIR}"
log "[A/${C}] ensuring master-* tarballs present"
hf download erickfm/melee-ranked-replays \
  --repo-type dataset \
  --include "shards/${HF_BUCKET}_master-master_a*.tar.gz" \
  --include "shards/${HF_BUCKET}_master-diamond_a*.tar.gz" \
  --include "shards/${HF_BUCKET}_master-platinum_a*.tar.gz" \
  --local-dir "${TAR_DIR}" 2>&1 | tee -a "${QLOG}" | tail -20

shopt -s nullglob
TARS=( "${TAR_DIR}/shards/${HF_BUCKET}_master-"*"_a"*".tar.gz" )
shopt -u nullglob
MARK="${SLP_DIR}/.extracted_tarballs"
touch "${MARK}"
log "[B/${C}] extracting missing tarballs"
for tar in "${TARS[@]}"; do
  tname="$(basename "${tar}")"
  if grep -qxF "${tname}" "${MARK}"; then continue ; fi
  tar -xzf "${tar}" -C "${SLP_DIR}/" && echo "${tname}" >> "${MARK}"
done
N_SLP="$(find "${SLP_DIR}" -maxdepth 1 -name '*.slp' | wc -l)"
log "[B/${C}] ${N_SLP} .slp files"

# ---- metadata should still be present (we only cleaned shards + raw slp) ----
for f in norm_stats.json mimic_norm.json stick_clusters.json controller_combos.json; do
  if [[ ! -f "${DATA_DIR}/${f}" ]]; then
    log "[C/${C}] missing ${f} — cannot continue (need to rebuild metadata from scratch)"
    exit 1
  fi
done
log "[C/${C}] all metadata present"

# ---- shard ----
if [[ ! -f "${DATA_DIR}/tensor_manifest.json" ]]; then
  log "[E/${C}] rebuilding v2 shards"
  python3 "${REPO_ROOT}/tools/slp_to_shards.py" \
    --slp-dir "${SLP_DIR}" --meta-dir "${DATA_DIR}" \
    --mimic-norm "${DATA_DIR}/mimic_norm.json" \
    --character "${IDX}" --staging-dir "${DATA_DIR}" \
    --repo "erickfm/mimic-${C}-v2" --no-upload --keep-staging \
    --shard-gb 4.0 --val-frac 0.1 --seed 42 \
    2>&1 | tee -a "${QLOG}" | tail -40
  if [[ ! -f "${DATA_DIR}/tensor_manifest.json" ]]; then
    log "[E/${C}] shard FAILED"; exit 1
  fi
else
  log "[E/${C}] shards already present"
fi

# ---- train with --resume ----
log "[F/${C}] resuming training from ${LATEST_CKPT}"
cd "${REPO_ROOT}"
unset WANDB_MODE
_wk="$(awk -F= '/^WANDB_API_KEY=/{print $2}' "${REPO_ROOT}/.env" 2>/dev/null | tr -d '[:space:]')"
if [[ -z "${_wk}" ]]; then export WANDB_MODE=disabled ; fi

# NOTE: keep the same run-name + log file so wandb / bestloss.pt threading stays consistent.
# The existing log file will be appended to; training will resume at the step stored in the ckpt.
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
  --resume "${LATEST_CKPT}" \
  >> "${LOG_FILE}" 2>&1
train_rc=$?
log "[F/${C}] training done (rc=${train_rc})"

# ---- upload + cleanup ----
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
    --data-dir "${DATA_DIR}" --log "${LOG_FILE}" 2>&1 | tee -a "${QLOG}" | tail -8
  log "[H/${C}] cleaning up raw + shards"
  rm -f "${DATA_DIR}/"*.pt
  rm -rf "${SLP_DIR}"
  log "====== DONE ${C} (continued) ======"
else
  log "[G/${C}] no ${BEST}"; exit 1
fi
