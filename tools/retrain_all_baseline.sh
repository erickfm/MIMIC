#!/usr/bin/env bash
# Retrain every character with the baseline `mimic` preset (20M params,
# full features under the post-2026-04-20 13-col schema, NO watchdog
# through patience=12 early-stop). Skips peach (already trained
# 2026-04-20 as peach-20260420-baseline, val 0.6322, uploaded).
#
# Decision (user, 2026-04-20): xxl's ~2-3% val-loss gain isn't worth 4x
# compute and 8x storage. All production chars go baseline.
#
# Pipeline per char:
#   A. Download master-* tars from erickfm/melee-ranked-replays
#   B. Extract (idempotent)
#   C. Build metadata if missing (norm_stats + mimic_norm)
#   D. Write 7-class controller_combos if missing
#   E. Shard — produces 13-col numeric shards (new schema, invuln+ECB dropped)
#   F. Train baseline (mimic preset, --lr 3e-4 bs 256 grad_accum 1, 32k steps,
#      patience=12 watchdog on val-loss)
#   G. Upload to erickfm/MIMIC/<char>/
#   H. Clean raw + shards
#
# Resume via checkpoints/retrain_baseline_state (one line per completed char).

set -uo pipefail
REPO_ROOT="/root/MIMIC"
DATE_TAG="20260420"
QLOG="${REPO_ROOT}/checkpoints/retrain_baseline.log"
STATE_FILE="${REPO_ROOT}/checkpoints/retrain_baseline_state"

CHARS=(
  "fox|FOX|1"
  "falco|FALCO|22"
  "marth|MARTH|18"
  "sheik|ZELDA_SHEIK|7"
  "cptfalcon|CPTFALCON|2"
  "luigi|LUIGI|17"
  "puff|JIGGLYPUFF|15"
  "ice_climbers|ICE_CLIMBERS|10"
)

log() { printf "[%s] %s\n" "$(date -u +%Y-%m-%d_%H:%M:%S)" "$*" | tee -a "${QLOG}" ; }
already_done() { [[ -f "${STATE_FILE}" ]] && grep -qxF "$1" "${STATE_FILE}" ; }
mark_done() { echo "$1" >> "${STATE_FILE}" ; }

process_char() {
  local C="$1" HF_BUCKET="$2" IDX="$3"
  local tag="train-done-${C}"
  if already_done "${tag}"; then
    log "${C} already done per state; skipping"
    return 0
  fi

  log "====== START ${C} (bucket=${HF_BUCKET} idx=${IDX}) ======"
  local slp_dir="${REPO_ROOT}/data/${C}_ranked_slp"
  local tar_dir="${slp_dir}/_tars"
  local data_dir="${REPO_ROOT}/data/${C}_v2"
  local run_name="${C}-${DATE_TAG}-baseline"
  local log_file="${REPO_ROOT}/checkpoints/${run_name}.log"
  mkdir -p "${slp_dir}" "${tar_dir}" "${data_dir}" "${REPO_ROOT}/checkpoints"

  # ---- A. Download ----
  log "[A/${C}] downloading master-* tarballs"
  hf download erickfm/melee-ranked-replays --repo-type dataset \
    --include "shards/${HF_BUCKET}_master-master_a*.tar.gz" \
    --include "shards/${HF_BUCKET}_master-diamond_a*.tar.gz" \
    --include "shards/${HF_BUCKET}_master-platinum_a*.tar.gz" \
    --local-dir "${tar_dir}" 2>&1 | tee -a "${QLOG}" | tail -20
  shopt -s nullglob
  local tars=( "${tar_dir}/shards/${HF_BUCKET}_master-"*"_a"*".tar.gz" )
  shopt -u nullglob
  if (( ${#tars[@]} == 0 )); then
    log "[A/${C}] ERROR: no master-* tars for ${HF_BUCKET}; skipping"
    return 1
  fi
  log "[A/${C}] fetched ${#tars[@]} tarballs"

  # ---- B. Extract ----
  local mark="${slp_dir}/.extracted_tarballs"
  touch "${mark}"
  for tar in "${tars[@]}"; do
    local t="$(basename "${tar}")"
    grep -qxF "${t}" "${mark}" || { tar -xzf "${tar}" -C "${slp_dir}/" && echo "${t}" >> "${mark}"; }
  done
  log "[B/${C}] $(find "${slp_dir}" -maxdepth 1 -name '*.slp' | wc -l) .slp files"

  # ---- C. Metadata ----
  if [[ ! -f "${data_dir}/norm_stats.json" ]]; then
    log "[C/${C}] building norm_stats (5000 files sample)"
    python3 "${REPO_ROOT}/tools/build_norm_stats.py" \
      --slp-dir "${slp_dir}" --out-dir "${data_dir}" --n-files 5000 \
      2>&1 | tee -a "${QLOG}" | tail -10
    python3 "${REPO_ROOT}/tools/build_mimic_norm.py" \
      --norm-stats "${data_dir}/norm_stats.json" \
      --minmax    "${data_dir}/norm_minmax.json" \
      --out       "${data_dir}/mimic_norm.json" \
      2>&1 | tee -a "${QLOG}" | tail -5
  fi
  [[ ! -f "${data_dir}/stick_clusters.json" ]] && \
    cp "${REPO_ROOT}/hf_checkpoints/fox/stick_clusters.json" "${data_dir}/stick_clusters.json"

  # ---- D. controller_combos ----
  if [[ ! -f "${data_dir}/controller_combos.json" ]]; then
    cat > "${data_dir}/controller_combos.json" <<'JSON'
{
    "button_names": ["A", "B", "Z", "JUMP", "TRIG", "A_TRIG", "NONE"],
    "n_combos": 7,
    "class_scheme": "melee_7class"
}
JSON
  fi

  # ---- E. Shard — always rebuild (schema may have changed; stale manifest would mismatch) ----
  rm -f "${data_dir}/tensor_manifest.json" "${data_dir}/tensor_meta.json" \
        "${data_dir}/"train_shard_*.pt "${data_dir}/"val_shard_*.pt 2>/dev/null
  log "[E/${C}] sharding (13-col schema, idx=${IDX})"
  python3 "${REPO_ROOT}/tools/slp_to_shards.py" \
    --slp-dir "${slp_dir}" \
    --meta-dir "${data_dir}" \
    --mimic-norm "${data_dir}/mimic_norm.json" \
    --character "${IDX}" \
    --staging-dir "${data_dir}" \
    --repo "erickfm/mimic-${C}-v2" \
    --no-upload --keep-staging \
    --shard-gb 4.0 --val-frac 0.1 --seed 42 \
    2>&1 | tee -a "${QLOG}" | tail -30
  if [[ ! -f "${data_dir}/tensor_manifest.json" ]]; then
    log "[E/${C}] shard FAILED; skipping ${C}"
    return 1
  fi

  # ---- F. Train baseline with patience=12 watchdog ----
  log "[F/${C}] training ${run_name} — mimic baseline, 32k steps, patience=12"
  cd "${REPO_ROOT}"
  unset WANDB_MODE
  local _wk
  _wk="$(awk -F= '/^WANDB_API_KEY=/{print $2}' "${REPO_ROOT}/.env" 2>/dev/null | tr -d '[:space:]')"
  [[ -z "${_wk}" ]] && export WANDB_MODE=disabled

  : > "${log_file}"
  torchrun --nproc_per_node=2 train.py \
    --model mimic --encoder mimic_flat \
    --mimic-mode --mimic-controller-encoding \
    --stick-clusters hal37 --plain-ce \
    --lr 3e-4 --batch-size 256 --grad-accum-steps 1 \
    --max-samples 16777216 \
    --data-dir "data/${C}_v2" \
    --self-inputs --reaction-delay 0 \
    --run-name "${run_name}" \
    --no-warmup --cosine-min-lr 1e-6 --nccl-timeout 3600 \
    > "${log_file}" 2>&1 &
  local train_pid=$!
  log "[F/${C}] torchrun PID=${train_pid}"

  (
    local patience_limit=12
    local min_val=""
    local patience=0
    while kill -0 "${train_pid}" 2>/dev/null; do
      sleep 60
      local cur
      cur="$(grep -oP 'val total=\K[0-9.]+' "${log_file}" 2>/dev/null | tail -1)"
      [[ -z "${cur}" ]] && continue
      if [[ -z "${min_val}" ]] || awk "BEGIN{exit !(${cur} + 0 < ${min_val} + 0)}"; then
        min_val="${cur}"; patience=0
      else
        patience=$((patience + 1))
      fi
      if (( patience >= patience_limit )); then
        printf "[%s] [watchdog/%s] val=%s didn't beat min=%s for %d evals — killing\n" \
          "$(date -u +%H:%M:%S)" "${C}" "${cur}" "${min_val}" "${patience}" | tee -a "${QLOG}"
        pkill -TERM -P "${train_pid}" 2>/dev/null; kill -TERM "${train_pid}" 2>/dev/null
        sleep 5
        pkill -KILL -P "${train_pid}" 2>/dev/null; kill -KILL "${train_pid}" 2>/dev/null
        break
      fi
    done
  ) &
  local watchdog_pid=$!
  wait "${train_pid}" 2>/dev/null
  local train_rc=$?
  kill -TERM "${watchdog_pid}" 2>/dev/null || true
  wait "${watchdog_pid}" 2>/dev/null || true

  local best="${REPO_ROOT}/checkpoints/${run_name}_bestloss.pt"
  if [[ ! -f "${best}" ]]; then
    log "[F/${C}] no ${best} (rc=${train_rc}); skipping upload"
    return 1
  fi
  log "[F/${C}] training done (rc=${train_rc}, bestloss present)"

  # ---- G. Upload ----
  local best_step
  best_step="$(grep -oP 'Best val_loss=[0-9.]+ @ step \K\d+' "${log_file}" 2>/dev/null | tail -1)"
  [[ -z "${best_step}" ]] && best_step="32000"
  local step_tag="$(( (best_step + 500) / 1000 ))k"
  cp -f "${best}" "${REPO_ROOT}/checkpoints/${run_name}-${step_tag}.pt"
  log "[G/${C}] renamed → ${run_name}-${step_tag}.pt"
  log "[G/${C}] uploading → erickfm/MIMIC/${C}/"
  python3 "${REPO_ROOT}/tools/upload_char.py" \
    --char "${C}" --checkpoint "${best}" \
    --data-dir "${data_dir}" --log "${log_file}" 2>&1 | tee -a "${QLOG}" | tail -6

  # ---- H. Cleanup ----
  rm -f "${data_dir}/"*.pt 2>/dev/null
  rm -rf "${slp_dir}" 2>/dev/null
  log "[H/${C}] df -h /root:"
  df -h /root | tail -1 | tee -a "${QLOG}"

  mark_done "${tag}"
  log "====== DONE ${C} ======"
  return 0
}

log "=============== RETRAIN BASELINE QUEUE START ==============="
log "chars: ${CHARS[*]} (peach already done)"
for entry in "${CHARS[@]}"; do
  IFS='|' read -r C HF_BUCKET IDX <<< "${entry}"
  process_char "${C}" "${HF_BUCKET}" "${IDX}" || log "${C} errored; moving on"
done
log "=============== RETRAIN BASELINE QUEUE END ==============="
