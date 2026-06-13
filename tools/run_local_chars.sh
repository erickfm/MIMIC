#!/usr/bin/env bash
# Sequential per-character BC pipeline for THIS box (single RTX 4090, 32 cores,
# repo at /home/erick/projects/MIMIC). Adapted from tools/run_all_chars.sh
# (remote 2x5090 version) with three changes:
#   - single-GPU training (python3, bs 256 x grad-accum 2 = eff 512; ~10GB VRAM,
#     ~4.5 step/s measured -> ~2h per char for 32,768 steps)
#   - FULL features (current default; the remote script predates the fullfeat
#     default and passed --mimic-minimal-features)
#   - prefetch: char N+1 downloads/extracts/shards on CPU while char N trains
#     on GPU. Raw .slp deleted after sharding; shards deleted after upload.
#
# Per-char failure tolerance: a failed prepare or train logs and advances to
# the next character instead of killing the run.

set -uo pipefail

REPO_ROOT="/home/erick/projects/MIMIC"
DATE_TAG="20260612"

# hf downloads have been observed hanging on dead CLOSE-WAIT sockets with no
# read timeout. Cap socket reads, and wrap each download attempt in timeout(1)
# below so a hang becomes a retry instead of an all-night stall.
export HF_HUB_DOWNLOAD_TIMEOUT=30
LOG_ROOT="${REPO_ROOT}/logs/allchars_local"
PIPE_LOG="${LOG_ROOT}/pipeline.log"
mkdir -p "${LOG_ROOT}" "${REPO_ROOT}/checkpoints"

# char | HF bucket | melee.Character enum value
CHARS=(
  "samus|SAMUS|13"
  "ganondorf|GANONDORF|25"
  "doc|DOC|21"
  "pikachu|PIKACHU|12"
  "yoshi|YOSHI|14"
)

log() { printf "[%s] %s\n" "$(date -u +%F\ %H:%M:%S)" "$*" >> "${PIPE_LOG}" ; }

prepare_char() {
  local C=$1 HF_BUCKET=$2 IDX=$3
  local SLP_DIR="${REPO_ROOT}/data/${C}_ranked_slp"
  local TAR_DIR="${SLP_DIR}/_tars"
  local DATA_DIR="${REPO_ROOT}/data/${C}_v2"
  mkdir -p "${SLP_DIR}" "${TAR_DIR}" "${DATA_DIR}"

  if [[ -f "${DATA_DIR}/tensor_manifest.json" ]]; then
    log "[prep/${C}] shards already present, skipping"
    return 0
  fi

  local avail_gb
  avail_gb=$(df --output=avail -BG "${REPO_ROOT}" | tail -1 | tr -dc '0-9')
  if (( avail_gb < 250 )); then
    log "[prep/${C}] only ${avail_gb}G free — refusing to prepare"
    return 3
  fi

  log "[prep/${C}] downloading master-* tarballs"
  local ok=1 attempt
  for attempt in 1 2 3 4 5; do
    # 90-min hard cap per attempt; downloads resume, so a slow-but-alive
    # attempt that gets cut just continues on the next attempt.
    # NOTE: all glob patterns must follow a SINGLE --include flag — repeating
    # the flag overwrites earlier patterns (argparse nargs='*'), which is how
    # the first run of this pipeline silently fetched only master-platinum.
    timeout -k 60 5400 hf download erickfm/melee-ranked-replays --repo-type dataset \
      --include "${HF_BUCKET}/${HF_BUCKET}_master-master_a*.tar.gz" \
                "${HF_BUCKET}/${HF_BUCKET}_master-diamond_a*.tar.gz" \
                "${HF_BUCKET}/${HF_BUCKET}_master-platinum_a*.tar.gz" \
      --local-dir "${TAR_DIR}" && { ok=0; break; }
    log "[prep/${C}] download attempt ${attempt} failed/timed out; retrying in 60s"
    sleep 60
  done
  if (( ok != 0 )); then log "[prep/${C}] download FAILED after 5 attempts"; return 1; fi

  shopt -s nullglob
  local TARS=( "${TAR_DIR}/${HF_BUCKET}/${HF_BUCKET}_master-"*"_a"*".tar.gz" )
  shopt -u nullglob
  if (( ${#TARS[@]} == 0 )); then log "[prep/${C}] no tarballs found"; return 1; fi
  log "[prep/${C}] ${#TARS[@]} tarballs fetched"

  local EXTRACT_MARK="${SLP_DIR}/.extracted_tarballs" tar tn
  touch "${EXTRACT_MARK}"
  for tar in "${TARS[@]}"; do
    tn="$(basename "${tar}")"
    grep -qxF "${tn}" "${EXTRACT_MARK}" && continue
    tar -xzf "${tar}" -C "${SLP_DIR}/" || { log "[prep/${C}] extract FAILED: ${tn}"; return 1; }
    echo "${tn}" >> "${EXTRACT_MARK}"
  done
  log "[prep/${C}] $(find "${SLP_DIR}" -maxdepth 1 -name '*.slp' | wc -l) .slp files"

  if [[ ! -f "${DATA_DIR}/norm_stats.json" ]]; then
    log "[prep/${C}] building norm_stats + mimic_norm"
    nice -n 10 python3 "${REPO_ROOT}/tools/build_norm_stats.py" \
      --slp-dir "${SLP_DIR}" --out-dir "${DATA_DIR}" --n-files 5000 \
      || { log "[prep/${C}] build_norm_stats FAILED"; return 1; }
    python3 "${REPO_ROOT}/tools/build_mimic_norm.py" \
      --norm-stats "${DATA_DIR}/norm_stats.json" \
      --minmax "${DATA_DIR}/norm_minmax.json" \
      --out "${DATA_DIR}/mimic_norm.json" \
      || { log "[prep/${C}] build_mimic_norm FAILED"; return 1; }
  fi

  if [[ ! -f "${DATA_DIR}/stick_clusters.json" ]]; then
    cp "${REPO_ROOT}/hf_checkpoints/fox/stick_clusters.json" "${DATA_DIR}/stick_clusters.json"
  fi

  if [[ ! -f "${DATA_DIR}/controller_combos.json" ]]; then
    cat > "${DATA_DIR}/controller_combos.json" <<'JSON'
{
    "button_names": ["A", "B", "Z", "JUMP", "TRIG", "A_TRIG", "NONE"],
    "n_combos": 7,
    "class_scheme": "melee_7class"
}
JSON
  fi

  log "[prep/${C}] building v2 shards"
  nice -n 10 python3 "${REPO_ROOT}/tools/slp_to_shards.py" \
    --slp-dir "${SLP_DIR}" \
    --meta-dir "${DATA_DIR}" \
    --mimic-norm "${DATA_DIR}/mimic_norm.json" \
    --character "${IDX}" \
    --staging-dir "${DATA_DIR}" \
    --repo "erickfm/mimic-${C}-v2" \
    --no-upload --keep-staging \
    --shard-gb 0.8 --val-frac 0.1 --seed 42 --workers 24 \
    || { log "[prep/${C}] slp_to_shards FAILED"; return 1; }

  if [[ ! -f "${DATA_DIR}/tensor_manifest.json" ]]; then
    log "[prep/${C}] sharding produced no tensor_manifest.json"
    return 1
  fi

  log "[prep/${C}] sharding done — deleting raw .slp + tarballs"
  rm -rf "${SLP_DIR}"
  return 0
}

train_char() {
  local C=$1
  local RUN_NAME="${C}-${DATE_TAG}-fullfeat"
  local LOG_FILE="${LOG_ROOT}/${RUN_NAME}.log"
  local DATA_DIR="${REPO_ROOT}/data/${C}_v2"

  cd "${REPO_ROOT}"
  log "[train/${C}] starting (bs 256, ga 2, eff 512, 32,768 steps, fullfeat)"
  : > "${LOG_FILE}"
  python3 train.py \
    --model mimic --encoder mimic_flat \
    --mimic-mode --mimic-controller-encoding \
    --stick-clusters hal37 --plain-ce \
    --lr 3e-4 --batch-size 256 --grad-accum-steps 2 \
    --max-samples 16777216 \
    --data-dir "data/${C}_v2" \
    --self-inputs --reaction-delay 0 \
    --run-name "${RUN_NAME}" \
    --no-warmup --cosine-min-lr 1e-6 \
    > "${LOG_FILE}" 2>&1 &
  local TRAIN_PID=$!
  log "[train/${C}] PID=${TRAIN_PID}"

  # Early-stop watchdog (same semantics as run_all_chars.sh): poll the log
  # every 60s; if the latest val loss hasn't beaten the minimum for 20
  # consecutive polls (~20 min), kill training. _bestloss.pt is already saved.
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
        log "[watchdog/${C}] val=${cur} no improvement vs min=${min_val} for ${patience} polls — stopping training"
        pkill -TERM -P "${TRAIN_PID}" 2>/dev/null
        kill -TERM "${TRAIN_PID}" 2>/dev/null
        sleep 5
        pkill -KILL -P "${TRAIN_PID}" 2>/dev/null
        kill -KILL "${TRAIN_PID}" 2>/dev/null
        break
      fi
    done
  ) &
  local WATCHDOG_PID=$!

  wait "${TRAIN_PID}"
  local TRAIN_RC=$?
  kill -TERM "${WATCHDOG_PID}" 2>/dev/null || true
  wait "${WATCHDOG_PID}" 2>/dev/null || true

  if (( TRAIN_RC == 0 || TRAIN_RC == 143 || TRAIN_RC == 137 || TRAIN_RC == 130 )); then
    log "[train/${C}] done (rc=${TRAIN_RC})"
  else
    log "[train/${C}] FAILED rc=${TRAIN_RC} — see ${LOG_FILE}"
    return 1
  fi

  grep -E 'Best val_loss=' "${LOG_FILE}" >> "${PIPE_LOG}" || true

  local BEST_CKPT="${REPO_ROOT}/checkpoints/${RUN_NAME}_bestloss.pt"
  if [[ -f "${BEST_CKPT}" ]]; then
    # Upload in the background — the home uplink runs as slow as ~5 kB/s and
    # an inline upload leaves the GPU idle for hours. Shards are deleted only
    # after a successful upload.
    log "[upload/${C}] uploading to erickfm/MIMIC/${C}/ (background)"
    (
      if python3 "${REPO_ROOT}/tools/upload_char.py" \
          --char "${C}" --checkpoint "${BEST_CKPT}" \
          --data-dir "${DATA_DIR}" --log "${LOG_FILE}" \
          > "${LOG_ROOT}/upload_${C}.log" 2>&1; then
        log "[upload/${C}] done — deleting shards"
        rm -f "${DATA_DIR}"/*.pt
      else
        log "[upload/${C}] FAILED (checkpoint still local; keeping shards)"
      fi
    ) &
    UPLOAD_PIDS+=( $! )
  else
    log "[upload/${C}] no bestloss checkpoint found"
  fi
  return 0
}

# ---- main ----
UPLOAD_PIDS=()
log "====== pipeline start: ${CHARS[*]} ======"

# Prepare the first character in the foreground (GPU is idle anyway).
IFS='|' read -r C0 B0 I0 <<< "${CHARS[0]}"
prepare_char "${C0}" "${B0}" "${I0}" >> "${LOG_ROOT}/prep_${C0}.log" 2>&1 \
  || log "[prep/${C0}] FAILED — will be skipped"

for i in "${!CHARS[@]}"; do
  IFS='|' read -r C B IDX <<< "${CHARS[$i]}"

  # Kick off prefetch of the next character while this one trains.
  PREP_PID=""
  if (( i + 1 < ${#CHARS[@]} )); then
    IFS='|' read -r NC NB NI <<< "${CHARS[$((i+1))]}"
    ( prepare_char "${NC}" "${NB}" "${NI}" ) >> "${LOG_ROOT}/prep_${NC}.log" 2>&1 &
    PREP_PID=$!
  fi

  if [[ -f "${REPO_ROOT}/data/${C}_v2/tensor_manifest.json" ]]; then
    log "====== TRAIN ${C} ======"
    train_char "${C}" || log "[train/${C}] char failed — moving on"
  else
    log "====== SKIP ${C} (no shards) ======"
  fi

  if [[ -n "${PREP_PID}" ]]; then
    wait "${PREP_PID}" || true
  fi
done

if (( ${#UPLOAD_PIDS[@]} > 0 )); then
  log "====== all training done; waiting on ${#UPLOAD_PIDS[@]} background uploads ======"
  for p in "${UPLOAD_PIDS[@]}"; do
    wait "${p}" || true
  done
fi

log "====== pipeline complete ======"
