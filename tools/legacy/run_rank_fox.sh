#!/usr/bin/env bash
# Train rank-specific Fox models (master/diamond/platinum) on the local 4090.
#
# Uses mixed-rank pairs but trains only the target-rank Fox's perspective:
# partition_fox_by_rank.py routes each Fox .slp into data/fox_<rank>_slp/ by
# the Fox player's netplay.name rank (mixed-rank Fox dittos excluded). Then the
# existing slp_to_shards.py --character 1 keeps that Fox perspective.
#
# Flow: stream all 6 Fox rank-pair tarballs (download -> extract -> partition
# (move) -> delete raw) -> build SHARED metadata once -> shard each rank ->
# train fox-<rank>-20260615. Shipping (upload + Discord) is intentionally NOT
# here — done separately after the benchmark + user confirm.
set -uo pipefail

REPO_ROOT="/home/erick/projects/MIMIC"
DATE_TAG="20260615"
LOG_ROOT="${REPO_ROOT}/logs/rank_fox"
PIPE_LOG="${LOG_ROOT}/pipeline.log"
STAGE="${REPO_ROOT}/data/_fox_stage"   # same fs as data/ so --move is instant
mkdir -p "${LOG_ROOT}" "${REPO_ROOT}/checkpoints"

PAIRS=( master-master diamond-diamond platinum-platinum
        master-diamond master-platinum diamond-platinum )
RANKS=( master diamond platinum )
export HF_HUB_DOWNLOAD_TIMEOUT=30

log() { printf "[%s] %s\n" "$(date -u +%F\ %H:%M:%S)" "$*" | tee -a "${PIPE_LOG}" ; }

# -------- prepare: download + partition every Fox rank pair --------
prepare_pair() {
  local pair=$1
  local mark="${LOG_ROOT}/.done_${pair}"
  [[ -f "${mark}" ]] && { log "[prep/${pair}] already partitioned, skip"; return 0; }
  local tdir="${STAGE}/${pair}"
  rm -rf "${tdir}"; mkdir -p "${tdir}"

  log "[prep/${pair}] downloading FOX ${pair} (6 archives)"
  local ok=1 attempt
  for attempt in 1 2 3 4 5; do
    timeout -k 60 5400 hf download erickfm/melee-ranked-replays --repo-type dataset \
      --include "FOX/FOX_${pair}_a*.tar.gz" \
      --local-dir "${tdir}/dl" && { ok=0; break; }
    log "[prep/${pair}] download attempt ${attempt} failed; retry in 60s"; sleep 60
  done
  (( ok != 0 )) && { log "[prep/${pair}] download FAILED"; return 1; }

  log "[prep/${pair}] extracting"
  shopt -s nullglob
  for tb in "${tdir}/dl/FOX/"FOX_${pair}_a*.tar.gz; do
    tar -xzf "${tb}" -C "${tdir}" || { log "[prep/${pair}] extract FAILED ${tb}"; return 1; }
  done
  shopt -u nullglob

  log "[prep/${pair}] partitioning by Fox-player rank (move)"
  python3 "${REPO_ROOT}/tools/partition_fox_by_rank.py" \
    --slp-dir "${tdir}" --out-root "${REPO_ROOT}/data" --move --workers 24 \
    2>&1 | tee -a "${PIPE_LOG}" || { log "[prep/${pair}] partition FAILED"; return 1; }

  rm -rf "${tdir}"          # raw tarballs + leftover (no-fox / excluded) .slp
  touch "${mark}"
  log "[prep/${pair}] done"
  return 0
}

# -------- shared metadata (built once from master Fox, reused for all ranks) --------
# Fixed path so callers never capture function stdout into the path string.
SHARED_META="${REPO_ROOT}/data/_fox_shared_meta"
build_shared_meta() {
  [[ -f "${SHARED_META}/mimic_norm.json" ]] && { log "[meta] reuse existing ${SHARED_META}"; return 0; }
  mkdir -p "${SHARED_META}"
  log "[meta] building shared norm from fox_master_slp"
  nice -n 10 python3 "${REPO_ROOT}/tools/build_norm_stats.py" \
    --slp-dir "${REPO_ROOT}/data/fox_master_slp" --out-dir "${SHARED_META}" --n-files 5000 || return 1
  python3 "${REPO_ROOT}/tools/build_mimic_norm.py" \
    --norm-stats "${SHARED_META}/norm_stats.json" --minmax "${SHARED_META}/norm_minmax.json" \
    --out "${SHARED_META}/mimic_norm.json" || return 1
  cp "${REPO_ROOT}/hf_checkpoints/fox/stick_clusters.json" "${SHARED_META}/stick_clusters.json"
  cat > "${SHARED_META}/controller_combos.json" <<'JSON'
{
    "button_names": ["A", "B", "Z", "JUMP", "TRIG", "A_TRIG", "NONE"],
    "n_combos": 7,
    "class_scheme": "melee_7class"
}
JSON
}

shard_rank() {
  local rank=$1 meta=$2
  # distinct dir name — data/fox_<rank>_v2 collides with pre-existing stale dirs
  local ddir="${REPO_ROOT}/data/foxrank_${rank}_v2"
  [[ -f "${ddir}/tensor_manifest.json" ]] && { log "[shard/${rank}] shards exist, skip"; return 0; }
  mkdir -p "${ddir}"
  cp -f "${meta}"/*.json "${ddir}/"   # all prereqs incl cat_maps.json
  log "[shard/${rank}] sharding fox_${rank}_slp (--character 1)"
  nice -n 10 python3 "${REPO_ROOT}/tools/slp_to_shards.py" \
    --slp-dir "${REPO_ROOT}/data/fox_${rank}_slp" \
    --meta-dir "${ddir}" --mimic-norm "${ddir}/mimic_norm.json" \
    --character 1 --staging-dir "${ddir}" \
    --repo "erickfm/mimic-fox-${rank}-v2" --no-upload --keep-staging \
    --shard-gb 0.8 --val-frac 0.1 --seed 42 --workers 24 \
    || { log "[shard/${rank}] FAILED"; return 1; }
  [[ -f "${ddir}/tensor_manifest.json" ]] || { log "[shard/${rank}] no manifest"; return 1; }
  log "[shard/${rank}] done; freeing raw .slp"
  rm -rf "${REPO_ROOT}/data/fox_${rank}_slp"
}

train_rank() {
  local rank=$1
  local run="fox-${rank}-${DATE_TAG}"
  local lf="${LOG_ROOT}/${run}.log"
  cd "${REPO_ROOT}"
  log "[train/${rank}] start ${run}"
  python3 train.py \
    --model mimic --encoder mimic_flat \
    --mimic-mode --mimic-controller-encoding \
    --stick-clusters hal37 --plain-ce \
    --lr 3e-4 --batch-size 256 --grad-accum-steps 2 \
    --max-samples 16777216 \
    --data-dir "data/foxrank_${rank}_v2" \
    --self-inputs --reaction-delay 0 \
    --run-name "${run}" \
    --no-warmup --cosine-min-lr 1e-6 > "${lf}" 2>&1 &
  local pid=$!
  ( PAT=20; mv=""; p=0
    while kill -0 "${pid}" 2>/dev/null; do
      sleep 60
      cur="$(grep -oP 'val total=\K[0-9.]+' "${lf}" 2>/dev/null | tail -1)"; [[ -z "${cur}" ]] && continue
      if [[ -z "${mv}" ]] || awk "BEGIN{exit !(${cur}+0<${mv}+0)}"; then mv="${cur}"; p=0; else p=$((p+1)); fi
      if (( p >= PAT )); then log "[watchdog/${rank}] no improve ${p} polls (min ${mv}) — stop"
        pkill -TERM -P "${pid}" 2>/dev/null; kill -TERM "${pid}" 2>/dev/null; sleep 5
        pkill -KILL -P "${pid}" 2>/dev/null; kill -KILL "${pid}" 2>/dev/null; break; fi
    done ) & local wd=$!
  wait "${pid}"; local rc=$?
  kill -TERM "${wd}" 2>/dev/null || true; wait "${wd}" 2>/dev/null || true
  if (( rc==0 || rc==143 || rc==137 || rc==130 )); then
    log "[train/${rank}] done (rc=${rc})"; grep -E 'Best val_loss=' "${lf}" | tee -a "${PIPE_LOG}" || true
  else log "[train/${rank}] FAILED rc=${rc}"; return 1; fi
}

# ---- main ----
log "====== rank-fox pipeline start ======"
for pair in "${PAIRS[@]}"; do prepare_pair "${pair}" || log "[prep/${pair}] FAILED — continuing"; done
for r in "${RANKS[@]}"; do
  n=$(find "${REPO_ROOT}/data/fox_${r}_slp" -name '*.slp' 2>/dev/null | wc -l)
  log "fox_${r}_slp: ${n} files"
done
build_shared_meta || { log "[meta] FAILED"; exit 1; }
META="${SHARED_META}"
log "[meta] shared metadata at ${META}"
for r in "${RANKS[@]}"; do shard_rank "${r}" "${META}" || log "[shard/${r}] FAILED — skip"; done
for r in "${RANKS[@]}"; do
  [[ -f "${REPO_ROOT}/data/foxrank_${r}_v2/tensor_manifest.json" ]] && train_rank "${r}" || log "[train/${r}] no shards — skip"
done
log "====== rank-fox pipeline complete ======"
