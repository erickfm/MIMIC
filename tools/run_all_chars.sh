#!/usr/bin/env bash
# Sequential per-character pipeline: for each char, pull all master-* .slp
# tarballs (master-master, master-diamond, master-platinum) from
# erickfm/melee-ranked-replays, shard via slp_to_shards.py, train a relpos
# v2 model with early-stop watchdog, mirror small artifacts to /workspace,
# upload char to HF, then advance to the next character. One character at a
# time — no concurrency across characters.
#
# Resume: reads /workspace/MIMIC/.pipeline_state for the last completed char.
# Artifact layout and command rationale: /root/.claude/plans/ok-so-the-past-partitioned-goblet.md

set -euo pipefail

REPO_ROOT="/root/MIMIC"
WS_ROOT="/workspace/MIMIC"
DATE_TAG="20260417"
STATE_FILE="${WS_ROOT}/.pipeline_state"

mkdir -p "${WS_ROOT}/data" "${WS_ROOT}/checkpoints"

# Order: char | HF bucket prefix | --character index
# marth/fox/falco first — user wants those retrained and re-uploaded first.
CHARS=(
  "marth|MARTH|18"
  "fox|FOX|1"
  "falco|FALCO|22"
  "sheik|ZELDA_SHEIK|7"
  "cptfalcon|CPTFALCON|2"
  "puff|JIGGLYPUFF|15"
  "luigi|LUIGI|17"
)

log() { printf "[%s] %s\n" "$(date -u +%H:%M:%S)" "$*" ; }

last_done=""
if [[ -f "${STATE_FILE}" ]]; then
  last_done="$(tr -d '[:space:]' < "${STATE_FILE}")"
  log "resume: last completed char = '${last_done}'"
fi

skip=true
if [[ -z "${last_done}" ]]; then skip=false ; fi

for entry in "${CHARS[@]}"; do
  IFS='|' read -r C HF_BUCKET IDX <<< "${entry}"

  if ${skip}; then
    if [[ "${C}" == "${last_done}" ]]; then
      log "resume: found boundary at '${C}', next char will run"
      skip=false
    fi
    continue
  fi

  log "====== START ${C} (bucket=${HF_BUCKET} idx=${IDX}) ======"
  SLP_DIR="${REPO_ROOT}/data/${C}_ranked_slp"
  TAR_DIR="${SLP_DIR}/_tars"
  DATA_DIR="${REPO_ROOT}/data/${C}_v2"
  RUN_NAME="${C}-${DATE_TAG}-relpos"
  LOG_FILE="${REPO_ROOT}/checkpoints/${RUN_NAME}.log"
  mkdir -p "${SLP_DIR}" "${TAR_DIR}" "${DATA_DIR}" "${REPO_ROOT}/checkpoints"

  # -------- A. Download all master-* tarballs (master-master + master-diamond + master-platinum) --------
  log "[A/${C}] downloading master-* tarballs"
  hf download erickfm/melee-ranked-replays \
    --repo-type dataset \
    --include "shards/${HF_BUCKET}_master-master_a*.tar.gz" \
    --include "shards/${HF_BUCKET}_master-diamond_a*.tar.gz" \
    --include "shards/${HF_BUCKET}_master-platinum_a*.tar.gz" \
    --local-dir "${TAR_DIR}"

  shopt -s nullglob
  TARS=( "${TAR_DIR}/shards/${HF_BUCKET}_master-"*"_a"*".tar.gz" )
  shopt -u nullglob
  if (( ${#TARS[@]} == 0 )); then
    log "[A/${C}] ERROR no master-* tarballs for ${HF_BUCKET}; aborting"
    exit 2
  fi
  log "[A/${C}] fetched ${#TARS[@]} tarballs"

  # -------- B. Extract tarballs (idempotent — file names are unique hash, overwrite-safe) --------
  # Track which tarballs have been extracted via a sentinel in SLP_DIR. Skip re-extracting.
  EXTRACT_MARK="${SLP_DIR}/.extracted_tarballs"
  touch "${EXTRACT_MARK}"
  log "[B/${C}] extracting tarballs (skipping already-extracted)"
  for tar in "${TARS[@]}"; do
    tar_name="$(basename "${tar}")"
    if grep -qxF "${tar_name}" "${EXTRACT_MARK}"; then
      continue
    fi
    tar -xzf "${tar}" -C "${SLP_DIR}/"
    echo "${tar_name}" >> "${EXTRACT_MARK}"
  done
  N_SLP="$(find "${SLP_DIR}" -maxdepth 1 -name '*.slp' | wc -l)"
  log "[B/${C}] ${N_SLP} .slp files total"

  # -------- C. Build metadata (only if not already present) --------
  if [[ ! -f "${DATA_DIR}/norm_stats.json" ]]; then
    log "[C/${C}] building norm_stats + mimic_norm"
    python3 "${REPO_ROOT}/tools/build_norm_stats.py" \
      --slp-dir "${SLP_DIR}" \
      --out-dir "${DATA_DIR}" --n-files 5000
    python3 "${REPO_ROOT}/tools/build_mimic_norm.py" \
      --norm-stats "${DATA_DIR}/norm_stats.json" \
      --minmax    "${DATA_DIR}/norm_minmax.json" \
      --out       "${DATA_DIR}/mimic_norm.json"
  else
    log "[C/${C}] norm_stats already present, skipping build"
  fi
  # stick_clusters is character-agnostic (k-means on normalized stick positions);
  # copy from hf_checkpoints/fox/ which ships with setup.sh --models.
  if [[ ! -f "${DATA_DIR}/stick_clusters.json" ]]; then
    cp "${REPO_ROOT}/hf_checkpoints/fox/stick_clusters.json" "${DATA_DIR}/stick_clusters.json"
    log "[C/${C}] copied stick_clusters.json from hf_checkpoints/fox/"
  fi

  # -------- D. Write controller_combos.json (must exist BEFORE sharding so slp_to_shards
  #            produces the composite self_controller tensor in each shard). 7-class
  #            rule-based scheme, identical across chars.
  if [[ ! -f "${DATA_DIR}/controller_combos.json" ]]; then
    log "[D/${C}] writing 7-class controller_combos.json"
    cat > "${DATA_DIR}/controller_combos.json" <<'JSON'
{
    "button_names": ["A", "B", "Z", "JUMP", "TRIG", "A_TRIG", "NONE"],
    "n_combos": 7,
    "class_scheme": "melee_7class"
}
JSON
  else
    log "[D/${C}] controller_combos already present"
  fi

  # -------- E. Build v2 shards --------
  if [[ ! -f "${DATA_DIR}/tensor_manifest.json" ]]; then
    log "[E/${C}] building v2 shards"
    python3 "${REPO_ROOT}/tools/slp_to_shards.py" \
      --slp-dir "${SLP_DIR}" \
      --meta-dir "${DATA_DIR}" \
      --mimic-norm "${DATA_DIR}/mimic_norm.json" \
      --character "${IDX}" \
      --staging-dir "${DATA_DIR}" \
      --repo "erickfm/mimic-${C}-v2" \
      --no-upload --keep-staging \
      --shard-gb 4.0 --val-frac 0.1 --seed 42
  else
    log "[E/${C}] shards already present, skipping"
  fi

  # -------- F. Train (with early-stop watchdog) --------
  log "[F/${C}] training (2-GPU DDP, bs 256, eff_bs 512, 32,768 steps, early-stop patience=20)"
  cd "${REPO_ROOT}"
  # wandb: if WANDB_API_KEY missing/empty, disable; otherwise let train.py handle init via .env.
  unset WANDB_MODE
  _wk="$(awk -F= '/^WANDB_API_KEY=/{print $2}' "${REPO_ROOT}/.env" 2>/dev/null | tr -d '[:space:]')"
  if [[ -z "${_wk}" ]]; then
    log "[F/${C}] WANDB_API_KEY empty in .env → WANDB_MODE=disabled"
    export WANDB_MODE=disabled
  else
    log "[F/${C}] wandb enabled (key len=${#_wk})"
  fi

  : > "${LOG_FILE}"
  # Launch training in background so we can attach an early-stop watchdog.
  torchrun --nproc_per_node=2 train.py \
    --model mimic --encoder mimic_flat \
    --mimic-mode --mimic-minimal-features --mimic-controller-encoding \
    --stick-clusters hal37 --plain-ce \
    --lr 3e-4 --batch-size 256 --grad-accum-steps 1 \
    --max-samples 16777216 \
    --data-dir "data/${C}_v2" \
    --self-inputs --reaction-delay 0 \
    --run-name "${RUN_NAME}" \
    --no-warmup --cosine-min-lr 1e-6 --nccl-timeout 3600 \
    > "${LOG_FILE}" 2>&1 &
  TRAIN_PID=$!
  log "[F/${C}] torchrun PID=${TRAIN_PID}"

  # Watchdog: polls the log, tracks the minimum val_loss, and kills training if
  # val has not improved for PATIENCE_LIMIT consecutive evaluations.
  # bestloss.pt is saved at the min, so an early stop preserves the best checkpoint.
  (
    PATIENCE_LIMIT=20
    min_val=""
    patience=0
    while kill -0 "${TRAIN_PID}" 2>/dev/null; do
      sleep 60
      cur="$(grep -oP 'val total=\K[0-9.]+' "${LOG_FILE}" 2>/dev/null | tail -1)"
      [[ -z "${cur}" ]] && continue
      if [[ -z "${min_val}" ]] || awk "BEGIN{exit !(${cur} + 0 < ${min_val} + 0)}"; then
        min_val="${cur}"
        patience=0
      else
        patience=$((patience + 1))
      fi
      if (( patience >= PATIENCE_LIMIT )); then
        printf "[%s] [watchdog/%s] val=%s did not beat min=%s for %d evals — killing training\n" \
          "$(date -u +%H:%M:%S)" "${C}" "${cur}" "${min_val}" "${patience}" >> /workspace/MIMIC/pipeline.log
        pkill -TERM -P "${TRAIN_PID}" 2>/dev/null
        kill -TERM "${TRAIN_PID}" 2>/dev/null
        sleep 5
        pkill -KILL -P "${TRAIN_PID}" 2>/dev/null
        kill -KILL "${TRAIN_PID}" 2>/dev/null
        break
      fi
    done
  ) &
  WATCHDOG_PID=$!

  set +e
  wait "${TRAIN_PID}"
  TRAIN_RC=$?
  set -e
  kill -TERM "${WATCHDOG_PID}" 2>/dev/null || true
  wait "${WATCHDOG_PID}" 2>/dev/null || true

  # rc 143 (SIGTERM) / 137 (SIGKILL) are watchdog-initiated kills — treat as success
  # since _bestloss.pt was saved when val hit its minimum.
  if (( TRAIN_RC == 0 || TRAIN_RC == 143 || TRAIN_RC == 137 || TRAIN_RC == 130 )); then
    log "[F/${C}] training done (rc=${TRAIN_RC})"
  else
    log "[F/${C}] FAIL rc=${TRAIN_RC}"
    exit "${TRAIN_RC}"
  fi

  # -------- G. Mirror + verify + advance --------
  log "[G/${C}] mirroring small artifacts to /workspace"
  mkdir -p "${WS_ROOT}/data/${C}_v2"
  for f in norm_stats.json norm_minmax.json cat_maps.json stick_clusters.json \
           mimic_norm.json controller_combos.json tensor_manifest.json tensor_meta.json; do
    [[ -f "${DATA_DIR}/${f}" ]] && cp -f "${DATA_DIR}/${f}" "${WS_ROOT}/data/${C}_v2/"
  done
  [[ -f "${REPO_ROOT}/checkpoints/${RUN_NAME}_bestloss.pt" ]] && \
    cp -f "${REPO_ROOT}/checkpoints/${RUN_NAME}_bestloss.pt" "${WS_ROOT}/checkpoints/" || true
  LATEST="$(ls -t "${REPO_ROOT}/checkpoints/${RUN_NAME}"_step*.pt 2>/dev/null | head -1 || true)"
  [[ -n "${LATEST}" ]] && cp -f "${LATEST}" "${WS_ROOT}/checkpoints/"

  log "[G/${C}] val loss lines:"
  grep -E 'Best val_loss=' "${LOG_FILE}" || log "[G/${C}] (no 'Best val_loss=' line found in log)"

  log "[G/${C}] /workspace df:"
  df -h /workspace | tail -1

  # -------- H. Upload bestloss checkpoint + metadata to erickfm/MIMIC/<char>/ --------
  BEST_CKPT="${REPO_ROOT}/checkpoints/${RUN_NAME}_bestloss.pt"
  if [[ -f "${BEST_CKPT}" ]]; then
    log "[H/${C}] uploading to huggingface.co/erickfm/MIMIC/${C}/"
    set +e
    python3 "${REPO_ROOT}/tools/upload_char.py" \
      --char "${C}" \
      --checkpoint "${BEST_CKPT}" \
      --data-dir "${DATA_DIR}" \
      --log "${LOG_FILE}" 2>&1 | tee -a "/workspace/MIMIC/pipeline.log"
    UP_RC=${PIPESTATUS[0]}
    set -e
    if (( UP_RC != 0 )); then
      log "[H/${C}] upload FAILED rc=${UP_RC} (continuing, checkpoint is still local)"
    else
      log "[H/${C}] upload done"
    fi
  else
    log "[H/${C}] no bestloss checkpoint found, skipping upload"
  fi

  echo "${C}" > "${STATE_FILE}"
  log "====== DONE ${C} ======"
done

log "all characters complete"
