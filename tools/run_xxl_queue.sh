#!/usr/bin/env bash
# Autonomous xxl queue: trains mimic-xxl (d=1024/L=12/h=16/d_ff=4096 GELU,
# ~154M params, full features — no --mimic-minimal-features) for each
# character in sequence, uploading to erickfm/MIMIC/<char>/ after each
# training completes.
#
# Sequence:
#   0. Wait for the current puff xxl run (if still training) to exit.
#      Then upload puff xxl to HF and clean up puff raw + shards.
#   1-3. For each of peach, falco, ice_climbers:
#        A. Download master-* tars from HF into data/<char>_ranked_slp/_tars
#        B. Extract (idempotent via sentinel file)
#        C. Build norm_stats + mimic_norm if missing (5000 files sample).
#           Copy stick_clusters.json from hf_checkpoints/fox/ (char-agnostic).
#        D. Write 7-class controller_combos.json if missing.
#        E. Shard via slp_to_shards.py.
#        F. Train mimic-xxl with 20-eval early-stop watchdog.
#        G. Upload best-loss checkpoint + metadata to HF.
#        H. Clean up char data (raw .slp + shards) to free disk.
#
# Resume via checkpoints/xxl_queue_state (holds last-completed step tag).

set -uo pipefail  # NOTE: no -e — we want to continue past per-char failures.

REPO_ROOT="/root/MIMIC"
DATE_TAG="20260419"
MODEL_PRESET="mimic-xxl"
STATE_FILE="${REPO_ROOT}/checkpoints/xxl_queue_state"
QUEUE_LOG="${REPO_ROOT}/checkpoints/xxl_queue.log"

# Characters to process (2026-04-20 redo run: peach + falco only, IC
# excluded because its 2507-game dataset overfits xxl no matter how
# long we train).
CHARS=(
  "peach|PEACH|9"
  "falco|FALCO|22"
)

# mimic-xxl VRAM-friendly batch sizing. Same eff_bs=512 as mimic-fullfeat.
XXL_BATCH_SIZE=64
XXL_GRAD_ACCUM=4

log() { printf "[%s] %s\n" "$(date -u +%Y-%m-%d_%H:%M:%S)" "$*" | tee -a "${QUEUE_LOG}" ; }

mark_done() { echo "$1" > "${STATE_FILE}" ; }
already_done() {
  [[ -f "${STATE_FILE}" ]] && grep -qxF "$1" "${STATE_FILE}"
}

# ---------- 0. Wait for puff xxl to finish ----------
puff_run_name="puff-${DATE_TAG}-mimic-xxl"
puff_log="${REPO_ROOT}/checkpoints/${puff_run_name}.log"

wait_for_puff_xxl() {
  log "=== step 0: waiting for puff xxl run to exit ==="
  while true; do
    # Any torchrun/train.py with mimic-xxl run_name still alive?
    if ! pgrep -f "train.py.*${puff_run_name}" > /dev/null 2>&1; then
      log "puff xxl not running (or already exited)"
      break
    fi
    sleep 60
  done
  # One more sanity check on the log for clean completion.
  if grep -qE "^Done\." "${puff_log}" 2>/dev/null; then
    log "puff xxl completed cleanly (Done. line found)"
    return 0
  else
    log "WARNING: puff xxl log does not contain 'Done.' — process may have crashed."
    log "Continuing anyway; best/bestloss checkpoints may still be usable."
    return 0
  fi
}

# ---------- puff xxl finalization ----------
finalize_puff_xxl() {
  if already_done "puff-xxl-uploaded"; then
    log "puff xxl already finalized (per state file); skipping"
    return 0
  fi
  log "=== finalizing puff xxl ==="
  local best="${REPO_ROOT}/checkpoints/${puff_run_name}_bestloss.pt"
  if [[ ! -f "${best}" ]]; then
    log "no ${best} — skipping puff xxl upload"
    return 1
  fi
  # Parse best step for renamed checkpoint
  local step
  step="$(grep -oP 'Best val_loss=[0-9.]+ @ step \K\d+' "${puff_log}" 2>/dev/null | tail -1)"
  [[ -z "${step}" ]] && step="32000"
  local tag="$(( (step + 500) / 1000 ))k"
  cp -f "${best}" "${REPO_ROOT}/checkpoints/${puff_run_name}-${tag}.pt"
  log "renamed puff xxl → ${puff_run_name}-${tag}.pt"

  log "uploading puff xxl → erickfm/MIMIC/puff/"
  python3 "${REPO_ROOT}/tools/upload_char.py" \
    --char puff \
    --checkpoint "${best}" \
    --data-dir "${REPO_ROOT}/data/puff_v2" \
    --log "${puff_log}" 2>&1 | tee -a "${QUEUE_LOG}"
  local up_rc=${PIPESTATUS[0]}
  if (( up_rc != 0 )); then
    log "puff xxl upload FAILED rc=${up_rc} (checkpoint still local)"
    return 1
  fi
  log "puff xxl upload done"
  mark_done "puff-xxl-uploaded"
  return 0
}

cleanup_puff_data() {
  if already_done "puff-cleanup"; then
    log "puff cleanup already done; skipping"
    return 0
  fi
  log "=== cleaning puff raw + shards to free disk ==="
  # Keep metadata JSONs under data/puff_v2/ in case we want to retrain puff
  # later (they're tiny and HF has a copy anyway). Remove .pt shards and raw .slp.
  rm -f "${REPO_ROOT}/data/puff_v2/"*.pt 2>/dev/null || true
  rm -rf "${REPO_ROOT}/data/puff_ranked_slp" 2>/dev/null || true
  log "freed disk after puff; df -h /root:"
  df -h /root | tail -1 | tee -a "${QUEUE_LOG}"
  mark_done "puff-cleanup"
}

# ---------- per-char training block ----------
process_char() {
  local C="$1" HF_BUCKET="$2" IDX="$3"
  local run_tag="train-done-${C}"
  if already_done "${run_tag}"; then
    log "${C} already trained per state file; skipping"
    return 0
  fi

  log "====== START ${C} (bucket=${HF_BUCKET} idx=${IDX}) ======"
  local slp_dir="${REPO_ROOT}/data/${C}_ranked_slp"
  local tar_dir="${slp_dir}/_tars"
  local data_dir="${REPO_ROOT}/data/${C}_v2"
  local run_name="${C}-${DATE_TAG}-xxl"
  local log_file="${REPO_ROOT}/checkpoints/${run_name}.log"
  mkdir -p "${slp_dir}" "${tar_dir}" "${data_dir}" "${REPO_ROOT}/checkpoints"

  # ---- A. Download master-* tars ----
  log "[A/${C}] downloading master-* tarballs from HF"
  hf download erickfm/melee-ranked-replays \
    --repo-type dataset \
    --include "shards/${HF_BUCKET}_master-master_a*.tar.gz" \
    --include "shards/${HF_BUCKET}_master-diamond_a*.tar.gz" \
    --include "shards/${HF_BUCKET}_master-platinum_a*.tar.gz" \
    --local-dir "${tar_dir}" 2>&1 | tee -a "${QUEUE_LOG}" | tail -40
  shopt -s nullglob
  local tars=( "${tar_dir}/shards/${HF_BUCKET}_master-"*"_a"*".tar.gz" )
  shopt -u nullglob
  if (( ${#tars[@]} == 0 )); then
    log "[A/${C}] ERROR no master-* tars for ${HF_BUCKET}; skipping char"
    return 1
  fi
  log "[A/${C}] fetched ${#tars[@]} tarballs"

  # ---- B. Extract ----
  local mark="${slp_dir}/.extracted_tarballs"
  touch "${mark}"
  log "[B/${C}] extracting tarballs"
  for tar in "${tars[@]}"; do
    local tname; tname="$(basename "${tar}")"
    if grep -qxF "${tname}" "${mark}"; then continue ; fi
    tar -xzf "${tar}" -C "${slp_dir}/"
    echo "${tname}" >> "${mark}"
  done
  local n_slp; n_slp="$(find "${slp_dir}" -maxdepth 1 -name '*.slp' | wc -l)"
  log "[B/${C}] ${n_slp} .slp files total"

  # ---- C. Build metadata if missing ----
  if [[ ! -f "${data_dir}/norm_stats.json" ]]; then
    log "[C/${C}] building norm_stats (5000 files sample)"
    python3 "${REPO_ROOT}/tools/build_norm_stats.py" \
      --slp-dir "${slp_dir}" --out-dir "${data_dir}" --n-files 5000 \
      2>&1 | tee -a "${QUEUE_LOG}" | tail -20
    python3 "${REPO_ROOT}/tools/build_mimic_norm.py" \
      --norm-stats "${data_dir}/norm_stats.json" \
      --minmax    "${data_dir}/norm_minmax.json" \
      --out       "${data_dir}/mimic_norm.json" \
      2>&1 | tee -a "${QUEUE_LOG}" | tail -10
  else
    log "[C/${C}] norm_stats already present, skipping build"
  fi
  if [[ ! -f "${data_dir}/stick_clusters.json" ]]; then
    cp "${REPO_ROOT}/hf_checkpoints/fox/stick_clusters.json" "${data_dir}/stick_clusters.json"
    log "[C/${C}] copied stick_clusters.json from hf_checkpoints/fox/"
  fi

  # ---- D. Controller combos (7-class) ----
  if [[ ! -f "${data_dir}/controller_combos.json" ]]; then
    log "[D/${C}] writing 7-class controller_combos.json"
    cat > "${data_dir}/controller_combos.json" <<'JSON'
{
    "button_names": ["A", "B", "Z", "JUMP", "TRIG", "A_TRIG", "NONE"],
    "n_combos": 7,
    "class_scheme": "melee_7class"
}
JSON
  else
    log "[D/${C}] controller_combos already present"
  fi

  # ---- E. Shard ----
  if [[ ! -f "${data_dir}/tensor_manifest.json" ]]; then
    log "[E/${C}] building v2 shards (character idx=${IDX})"
    python3 "${REPO_ROOT}/tools/slp_to_shards.py" \
      --slp-dir "${slp_dir}" \
      --meta-dir "${data_dir}" \
      --mimic-norm "${data_dir}/mimic_norm.json" \
      --character "${IDX}" \
      --staging-dir "${data_dir}" \
      --repo "erickfm/mimic-${C}-v2" \
      --no-upload --keep-staging \
      --shard-gb 4.0 --val-frac 0.1 --seed 42 \
      2>&1 | tee -a "${QUEUE_LOG}" | tail -40
    local shard_rc=$?
    if [[ ! -f "${data_dir}/tensor_manifest.json" ]]; then
      log "[E/${C}] shard FAILED (no tensor_manifest.json produced); skipping char"
      return 1
    fi
  else
    log "[E/${C}] shards already present, skipping"
  fi

  # ---- F. Train mimic-xxl (full features, eff_bs 512, 32768 steps) ----
  log "[F/${C}] training ${MODEL_PRESET} (2-GPU DDP, bs ${XXL_BATCH_SIZE}, grad_accum ${XXL_GRAD_ACCUM}, eff_bs $((XXL_BATCH_SIZE * 2 * XXL_GRAD_ACCUM)), 32768 steps, early-stop patience=20)"
  cd "${REPO_ROOT}"
  unset WANDB_MODE
  local _wk; _wk="$(awk -F= '/^WANDB_API_KEY=/{print $2}' "${REPO_ROOT}/.env" 2>/dev/null | tr -d '[:space:]')"
  if [[ -z "${_wk}" ]]; then export WANDB_MODE=disabled ; fi

  : > "${log_file}"
  torchrun --nproc_per_node=2 train.py \
    --model "${MODEL_PRESET}" --encoder mimic_flat \
    --mimic-mode --mimic-controller-encoding \
    --stick-clusters hal37 --plain-ce \
    --lr 3e-4 --batch-size "${XXL_BATCH_SIZE}" --grad-accum-steps "${XXL_GRAD_ACCUM}" \
    --max-samples 16777216 \
    --data-dir "data/${C}_v2" \
    --self-inputs --reaction-delay 0 \
    --run-name "${run_name}" \
    --no-warmup --cosine-min-lr 1e-6 --nccl-timeout 3600 \
    > "${log_file}" 2>&1 &
  local train_pid=$!
  log "[F/${C}] torchrun PID=${train_pid}"

  # Watchdog DISABLED for xxl (2026-04-20). patience=20 is calibrated for
  # 20M-param smoother val curves; xxl val is noisier and often reaches
  # best-val late in cosine decay (puff xxl best was step 30,738/32,768),
  # so the watchdog was killing runs prematurely. Let training run full
  # 32k steps. If overfit is a concern for small datasets, handle it
  # manually or reduce max_samples.
  wait "${train_pid}" 2>/dev/null
  local train_rc=$?

  # Watchdog kills torchrun's subprocesses which makes the parent exit
  # with a variety of non-zero codes (1, 130, 137, 143). As long as a
  # bestloss checkpoint exists, treat the run as a legitimate early-stop
  # and proceed to upload + cleanup. Only a pre-bestloss crash skips upload.
  local bestloss_ck="${REPO_ROOT}/checkpoints/${run_name}_bestloss.pt"
  if [[ -f "${bestloss_ck}" ]]; then
    log "[F/${C}] training done (rc=${train_rc}, bestloss.pt present)"
  else
    log "[F/${C}] training FAIL rc=${train_rc} (no bestloss.pt) — skipping upload/cleanup"
    return 1
  fi

  # ---- G. Upload ----
  local best="${REPO_ROOT}/checkpoints/${run_name}_bestloss.pt"
  if [[ -f "${best}" ]]; then
    local step; step="$(grep -oP 'Best val_loss=[0-9.]+ @ step \K\d+' "${log_file}" 2>/dev/null | tail -1)"
    [[ -z "${step}" ]] && step="32000"
    local tag="$(( (step + 500) / 1000 ))k"
    cp -f "${best}" "${REPO_ROOT}/checkpoints/${run_name}-${tag}.pt"
    log "[G/${C}] renamed best → ${run_name}-${tag}.pt"
    log "[G/${C}] uploading → erickfm/MIMIC/${C}/"
    python3 "${REPO_ROOT}/tools/upload_char.py" \
      --char "${C}" --checkpoint "${best}" \
      --data-dir "${data_dir}" --log "${log_file}" 2>&1 | tee -a "${QUEUE_LOG}" | tail -15
    local up_rc=${PIPESTATUS[0]}
    if (( up_rc != 0 )); then
      log "[G/${C}] upload FAILED rc=${up_rc} (checkpoint is local in ${best})"
    else
      log "[G/${C}] upload done"
    fi
  else
    log "[G/${C}] no ${best} — skipping upload"
  fi

  # ---- H. Cleanup raw + shards ----
  log "[H/${C}] cleaning up raw + shards"
  rm -f "${data_dir}/"*.pt 2>/dev/null || true
  rm -rf "${slp_dir}" 2>/dev/null || true
  log "[H/${C}] df -h /root:"
  df -h /root | tail -1 | tee -a "${QUEUE_LOG}"

  mark_done "${run_tag}"
  log "====== DONE ${C} ======"
  return 0
}

# =========================================================================
# MAIN
# =========================================================================
log "=============== XXL QUEUE START ==============="
log "preset=${MODEL_PRESET}  date_tag=${DATE_TAG}  chars=${CHARS[*]}"

wait_for_puff_xxl
finalize_puff_xxl || log "puff-xxl finalization returned nonzero; continuing"
cleanup_puff_data

for entry in "${CHARS[@]}"; do
  IFS='|' read -r C HF_BUCKET IDX <<< "${entry}"
  process_char "${C}" "${HF_BUCKET}" "${IDX}" || log "process_char ${C} returned nonzero; moving on"
done

log "=============== XXL QUEUE END ==============="
