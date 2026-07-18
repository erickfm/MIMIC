#!/usr/bin/env bash
# All-character BC pipeline for a rented pod (everything under /workspace —
# the rest of the pod filesystem is ephemeral). Adapted from
# tools/run_local_chars.sh with these deltas:
#   - MASTER-RANK FILTER: after extract, tools/filter_masterchar.py keeps only
#     games where every target-char port is master (the mixed-rank pools are
#     only ~60% master otherwise — see CLAUDE.md "Ranked dataset").
#   - c-stick-fixed shards (nearest-of-9 history; needs repo >= fb7bb1a).
#   - current recipe: fullfeat, no mirror, seed 42, EMA 0.999.
#   - MIMIC_NUM_WORKERS for the dataloader; GPU-count toggle: single GPU runs
#     python3 (bs 256 x ga 2), multi-GPU runs torchrun with per-GPU batch
#     512/NGPU (integer-truncated: non-power-of-2 GPU counts drift the
#     effective batch by <=2 — max_steps recomputes from the actual product,
#     so the sample budget stays exact).
#   - watchdog counts EVALS, not wall-clock polls (run_local's poll-counting
#     could kill a run for a 20-min stall; here patience = 12 evals with no
#     new best val).
#   - stage-only "upload": upload_char.py --no-push stages into
#     _hf_staging_single/<char>/ (its hardcoded path). Nothing is pushed to
#     HF without explicit user approval (promotion policy). The EMA bestloss
#     checkpoint is copied alongside as model_ema.pt.
#   - PREP_ONLY=1 runs data prep for every char and skips training (useful
#     while the GPU is unavailable; reruns skip chars whose shards exist).
#
# Per-char failure tolerance: a failed prepare or train logs and advances.
set -uo pipefail

REPO_ROOT="/workspace/MIMIC"
DATE_TAG="20260718"
export HF_HUB_DOWNLOAD_TIMEOUT=30
export HF_HOME="/workspace/.hf_home"          # cache + token survive restarts
export WANDB_MODE=offline
export MIMIC_NUM_WORKERS="${MIMIC_NUM_WORKERS:-48}"
LOG_ROOT="${REPO_ROOT}/logs/allchars_pod"
PIPE_LOG="${LOG_ROOT}/pipeline.log"
STAGE_ROOT="${REPO_ROOT}/_hf_staging_single"  # upload_char.py's hardcoded stage dir
META_SRC="${REPO_ROOT}/tools/meta"
mkdir -p "${LOG_ROOT}" "${REPO_ROOT}/checkpoints"

# char | HF bucket | libmelee Character enum name (value resolved at runtime)
# Production-relevant characters first. Fox is skipped (current champion).
CHARS=(
  "falco|FALCO|FALCO"
  "cptfalcon|CPTFALCON|CPTFALCON"
  "puff|JIGGLYPUFF|JIGGLYPUFF"
  "luigi|LUIGI|LUIGI"
  "marth|MARTH|MARTH"
  "sheik|SHEIK|SHEIK"
  "peach|PEACH|PEACH"
  "pikachu|PIKACHU|PIKACHU"
  "samus|SAMUS|SAMUS"
  "yoshi|YOSHI|YOSHI"
  "ganondorf|GANONDORF|GANONDORF"
  "doc|DOC|DOC"
  "mario|MARIO|MARIO"
  "dk|DK|DK"
  "link|LINK|LINK"
  "ness|NESS|NESS"
  "ylink|YLINK|YLINK"
  "bowser|BOWSER|BOWSER"
  "gameandwatch|GAMEANDWATCH|GAMEANDWATCH"
  "mewtwo|MEWTWO|MEWTWO"
  "roy|ROY|ROY"
  "pichu|PICHU|PICHU"
  "kirby|KIRBY|KIRBY"
  "ice_climbers|ICE_CLIMBERS|POPO"
  "zelda|ZELDA|ZELDA"
)

log() { printf "[%s] %s\n" "$(date -u +%F\ %H:%M:%S)" "$*" | tee -a "${PIPE_LOG}" >/dev/null ; }

# ── Preflight: fail the whole run immediately on missing prerequisites ──────
preflight() {
  local ok=0
  for f in "${META_SRC}/stick_clusters_hal37.json" \
           "${META_SRC}/controller_combos_7class.json"; do
    [[ -f "$f" ]] || { echo "PREFLIGHT: missing $f (repo too old? need >= the tools/meta commit)"; ok=1; }
  done
  python3 -c "import melee, peppi_py" 2>/dev/null \
    || { echo "PREFLIGHT: python deps missing (melee/peppi_py)"; ok=1; }
  python3 - <<'PYEOF' || ok=1
import numpy as np, sys
from pathlib import Path
sys.path.insert(0, "/workspace/MIMIC")
from mimic.features import HAL_STICK_CLUSTERS_37, load_cluster_centers
sc, sh = load_cluster_centers(
    clusters_path=Path("/workspace/MIMIC/tools/meta/stick_clusters_hal37.json"))
assert sc is not None and sc.shape == (37, 2), f"bad stick centers: {None if sc is None else sc.shape}"
assert np.allclose(np.sort(np.asarray(sc).ravel()),
                   np.sort(HAL_STICK_CLUSTERS_37.ravel()), atol=1e-6), \
    "stick_clusters file != built-in HAL_STICK_CLUSTERS_37"
assert sh is not None, "shoulder centers missing from stick_clusters file"
PYEOF
  return $ok
}

prepare_char() {
  local C=$1 HF_BUCKET=$2 ENUM=$3
  local SLP_DIR="${REPO_ROOT}/data/${C}_ranked_slp"
  local MASTER_DIR="${REPO_ROOT}/data/${C}_master_slp"
  local TAR_DIR="${SLP_DIR}/_tars"
  local DATA_DIR="${REPO_ROOT}/data/${C}_v2"
  mkdir -p "${SLP_DIR}" "${TAR_DIR}" "${DATA_DIR}" "${MASTER_DIR}"

  local IDX
  IDX=$(python3 -c "import melee; print(melee.Character.${ENUM}.value)") \
    || { log "[prep/${C}] bad enum ${ENUM}"; return 1; }

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
    # NOTE: all glob patterns after a SINGLE --include (repeated flags
    # overwrite earlier patterns — the documented platinum-only trap).
    timeout -k 60 5400 hf download erickfm/melee-ranked-replays --repo-type dataset \
      --include "${HF_BUCKET}/${HF_BUCKET}_master-master_a*.tar.gz" \
                "${HF_BUCKET}/${HF_BUCKET}_master-diamond_a*.tar.gz" \
                "${HF_BUCKET}/${HF_BUCKET}_master-platinum_a*.tar.gz" \
      --local-dir "${TAR_DIR}" && { ok=0; break; }
    log "[prep/${C}] download attempt ${attempt} failed/timed out; retrying in 60s"
    sleep 60
  done
  (( ok != 0 )) && { log "[prep/${C}] download FAILED after 5 attempts"; return 1; }

  shopt -s nullglob
  local TARS=( "${TAR_DIR}/${HF_BUCKET}/${HF_BUCKET}_master-"*"_a"*".tar.gz" )
  shopt -u nullglob
  (( ${#TARS[@]} == 0 )) && { log "[prep/${C}] no tarballs found"; return 1; }
  log "[prep/${C}] ${#TARS[@]} tarballs fetched"

  local EXTRACT_MARK="${SLP_DIR}/.extracted_tarballs" tar tn
  touch "${EXTRACT_MARK}"
  for tar in "${TARS[@]}"; do
    tn="$(basename "${tar}")"
    grep -qxF "${tn}" "${EXTRACT_MARK}" && continue
    tar -xzf "${tar}" -C "${SLP_DIR}/" || { log "[prep/${C}] extract FAILED: ${tn}"; return 1; }
    echo "${tn}" >> "${EXTRACT_MARK}"
  done
  local n_pool
  n_pool=$(find "${SLP_DIR}" -maxdepth 1 -name '*.slp' | wc -l)
  log "[prep/${C}] extracted pool=${n_pool} .slp"

  log "[prep/${C}] master-rank filter"
  python3 "${REPO_ROOT}/tools/filter_masterchar.py" --char "${HF_BUCKET}" \
    "${SLP_DIR}" --out "${SLP_DIR}/master_keep.txt" --workers 96 \
    >> "${PIPE_LOG}" 2>&1 \
    || { log "[prep/${C}] filter FAILED"; return 1; }
  local n_keep
  n_keep=$(wc -l < "${SLP_DIR}/master_keep.txt")
  if (( n_keep < 300 )); then
    log "[prep/${C}] only ${n_keep} master games — skipping char (too thin)"
    rm -rf "${SLP_DIR}" "${MASTER_DIR}"
    return 4
  fi
  xargs -a "${SLP_DIR}/master_keep.txt" -P8 -I{} mv {} "${MASTER_DIR}/" 2>/dev/null || true
  local n_staged
  n_staged=$(find "${MASTER_DIR}" -maxdepth 1 -name '*.slp' | wc -l)
  log "[prep/${C}] master games staged=${n_staged}/${n_keep} — deleting non-master .slp"
  (( n_staged * 100 < n_keep * 98 )) && { log "[prep/${C}] staging came up short — abort"; return 1; }
  # Keep _tars + extract marker until sharding succeeds so a shard-stage
  # failure resumes from extraction, not a full re-download.
  rm -f "${SLP_DIR}"/*.slp

  if [[ ! -f "${DATA_DIR}/norm_stats.json" ]]; then
    log "[prep/${C}] building norm_stats + mimic_norm (master pool)"
    nice -n 10 python3 "${REPO_ROOT}/tools/build_norm_stats.py" \
      --slp-dir "${MASTER_DIR}" --out-dir "${DATA_DIR}" --n-files 5000 \
      || { log "[prep/${C}] build_norm_stats FAILED"; return 1; }
    python3 "${REPO_ROOT}/tools/build_mimic_norm.py" \
      --norm-stats "${DATA_DIR}/norm_stats.json" \
      --minmax "${DATA_DIR}/norm_minmax.json" \
      --out "${DATA_DIR}/mimic_norm.json" \
      || { log "[prep/${C}] build_mimic_norm FAILED"; return 1; }
  fi
  cp -n "${META_SRC}/stick_clusters_hal37.json" "${DATA_DIR}/stick_clusters.json" \
    || true
  cp -n "${META_SRC}/controller_combos_7class.json" "${DATA_DIR}/controller_combos.json" \
    || true
  for f in stick_clusters.json controller_combos.json; do
    [[ -f "${DATA_DIR}/${f}" ]] || { log "[prep/${C}] missing ${f} — abort"; return 1; }
  done

  log "[prep/${C}] building v2 shards (c-stick-fixed, idx=${IDX})"
  nice -n 10 python3 "${REPO_ROOT}/tools/slp_to_shards.py" \
    --slp-dir "${MASTER_DIR}" \
    --meta-dir "${DATA_DIR}" \
    --mimic-norm "${DATA_DIR}/mimic_norm.json" \
    --character "${IDX}" \
    --staging-dir "${DATA_DIR}" \
    --repo "erickfm/mimic-${C}-v2" \
    --no-upload --keep-staging \
    --shard-gb 0.8 --val-frac 0.1 --seed 42 --workers 96 \
    || { log "[prep/${C}] slp_to_shards FAILED"; return 1; }
  [[ -f "${DATA_DIR}/tensor_manifest.json" ]] \
    || { log "[prep/${C}] sharding produced no tensor_manifest.json"; return 1; }

  log "[prep/${C}] sharding done — deleting raw .slp + tarballs"
  rm -rf "${MASTER_DIR}" "${SLP_DIR}"
  return 0
}

train_char() {
  local C=$1
  local RUN_NAME="${C}-${DATE_TAG}-master-c9"
  local LOG_FILE="${LOG_ROOT}/${RUN_NAME}.log"
  local DATA_DIR="${REPO_ROOT}/data/${C}_v2"

  cd "${REPO_ROOT}"
  local NGPU
  NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
  (( NGPU < 1 )) && NGPU=1
  : > "${LOG_FILE}"
  local COMMON=(
    --model mimic --encoder mimic_flat
    --mimic-mode --mimic-controller-encoding
    --stick-clusters hal37 --plain-ce
    --lr 3e-4 --max-samples 16777216
    --data-dir "data/${C}_v2"
    --self-inputs --reaction-delay 0
    --seed 42 --ema-decay 0.999
    --run-name "${RUN_NAME}"
    --no-warmup --cosine-min-lr 1e-6
  )
  if (( NGPU == 1 )); then
    log "[train/${C}] single GPU (bs 256 x ga 2, eff 512)"
    python3 train.py "${COMMON[@]}" --batch-size 256 --grad-accum-steps 2 \
      > "${LOG_FILE}" 2>&1 &
  else
    local BS=$(( 512 / NGPU ))
    log "[train/${C}] ${NGPU} GPUs (bs ${BS} x ga 1, eff $(( BS * NGPU )))"
    torchrun --nproc_per_node="${NGPU}" train.py "${COMMON[@]}" \
      --batch-size "${BS}" --grad-accum-steps 1 --nccl-timeout 3600 \
      > "${LOG_FILE}" 2>&1 &
  fi
  local TRAIN_PID=$!
  log "[train/${C}] PID=${TRAIN_PID}"

  # Eval-counting watchdog: patience = 12 *new evals* without a new best.
  (
    PATIENCE_LIMIT=12
    min_val=""
    patience=0
    n_seen=0
    while kill -0 "${TRAIN_PID}" 2>/dev/null; do
      sleep 60
      n_now=$(grep -c 'val total=' "${LOG_FILE}" 2>/dev/null || true)
      [[ -z "${n_now}" ]] && continue
      (( n_now <= n_seen )) && continue    # no new eval since last look
      n_seen=${n_now}
      cur="$(grep -oP 'val total=\K[0-9.]+' "${LOG_FILE}" | tail -1)"
      [[ -z "${cur}" ]] && continue
      if [[ -z "${min_val}" ]] || awk "BEGIN{exit !(${cur} + 0 < ${min_val} + 0)}"; then
        min_val="${cur}"
        patience=0
      else
        patience=$((patience + 1))
      fi
      if (( patience >= PATIENCE_LIMIT )); then
        log "[watchdog/${C}] no new best for ${patience} evals (min=${min_val}) — stopping"
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
  local EMA_CKPT="${REPO_ROOT}/checkpoints/${RUN_NAME}_ema_bestloss.pt"
  if [[ -f "${BEST_CKPT}" ]]; then
    log "[stage/${C}] staging to ${STAGE_ROOT}/${C} (NO push — user approves)"
    python3 "${REPO_ROOT}/tools/upload_char.py" \
        --char "${C}" --checkpoint "${BEST_CKPT}" \
        --data-dir "${DATA_DIR}" --log "${LOG_FILE}" --no-push \
        > "${LOG_ROOT}/stage_${C}.log" 2>&1 \
      && log "[stage/${C}] staged" \
      || { log "[stage/${C}] staging FAILED (checkpoint kept in checkpoints/)"; return 0; }
    [[ -f "${EMA_CKPT}" && -d "${STAGE_ROOT}/${C}" ]] \
      && cp "${EMA_CKPT}" "${STAGE_ROOT}/${C}/model_ema.pt"
    # Keep norm/meta + val shards for later eval; drop train shards for disk.
    rm -f "${DATA_DIR}"/train_shard_*.pt
  else
    log "[stage/${C}] no bestloss checkpoint found"
  fi
  return 0
}

preflight || { log "[main] PREFLIGHT FAILED — aborting before any download"; exit 2; }

# ── PREP_ONLY: build all data, no training (e.g. GPU not yet available) ─────
if [[ "${PREP_ONLY:-0}" == "1" ]]; then
  log "[main] PREP_ONLY=1 — preparing all characters, no training"
  for entry in "${CHARS[@]}"; do
    IFS='|' read -r C BUCKET ENUM <<< "${entry}"
    prepare_char "${C}" "${BUCKET}" "${ENUM}" \
      > "${LOG_ROOT}/prep_${C}.log" 2>&1 \
      || log "[main] prepare(${C}) rc=$? — continuing"
  done
  log "[main] PREP_ONLY_DONE"
  exit 0
fi

# Main loop: prepare char N+1 in the background while char N trains.
PREP_PID=""
PREP_CHAR=""
for i in "${!CHARS[@]}"; do
  IFS='|' read -r C BUCKET ENUM <<< "${CHARS[$i]}"

  if [[ -n "${PREP_PID}" && "${PREP_CHAR}" == "${C}" ]]; then
    wait "${PREP_PID}"; PREP_RC=$?
  else
    prepare_char "${C}" "${BUCKET}" "${ENUM}" \
      > "${LOG_ROOT}/prep_${C}.log" 2>&1; PREP_RC=$?
  fi
  PREP_PID=""; PREP_CHAR=""

  if (( PREP_RC != 0 )); then
    log "[main] prepare(${C}) rc=${PREP_RC} — skipping char"
    continue
  fi

  # kick off next char's prepare on the CPUs while this one trains
  if (( i + 1 < ${#CHARS[@]} )); then
    IFS='|' read -r NC NBUCKET NENUM <<< "${CHARS[$((i+1))]}"
    prepare_char "${NC}" "${NBUCKET}" "${NENUM}" \
      > "${LOG_ROOT}/prep_${NC}.log" 2>&1 &
    PREP_PID=$!
    PREP_CHAR="${NC}"
    log "[main] prefetching ${NC} (pid ${PREP_PID}) while ${C} trains"
  fi

  train_char "${C}" || log "[main] train(${C}) failed — continuing"
done
[[ -n "${PREP_PID}" ]] && wait "${PREP_PID}" 2>/dev/null
log "[main] ALL_CHARS_DONE"
