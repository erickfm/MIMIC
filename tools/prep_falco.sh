#!/usr/bin/env bash
# One-shot falco data prep — runs in parallel with fox sharding. When the
# main driver reaches falco, every step from A through E will hit a skip
# guard and jump straight to training.
set -euo pipefail
REPO_ROOT="/root/MIMIC"
C="falco"
HF_BUCKET="FALCO"
IDX=22
SLP_DIR="${REPO_ROOT}/data/${C}_ranked_slp"
TAR_DIR="${SLP_DIR}/_tars"
DATA_DIR="${REPO_ROOT}/data/${C}_v2"

log() { printf "[%s] [prep_falco] %s\n" "$(date -u +%H:%M:%S)" "$*" ; }

mkdir -p "${SLP_DIR}" "${TAR_DIR}" "${DATA_DIR}"

log "A download tarballs"
hf download erickfm/melee-ranked-replays --repo-type dataset \
  --include "shards/${HF_BUCKET}_master-master_a*.tar.gz" \
  --include "shards/${HF_BUCKET}_master-diamond_a*.tar.gz" \
  --include "shards/${HF_BUCKET}_master-platinum_a*.tar.gz" \
  --local-dir "${TAR_DIR}" > /dev/null

shopt -s nullglob
TARS=( "${TAR_DIR}/shards/${HF_BUCKET}_master-"*"_a"*".tar.gz" )
shopt -u nullglob
log "A fetched ${#TARS[@]} tarballs"

log "B extract"
EXTRACT_MARK="${SLP_DIR}/.extracted_tarballs"
touch "${EXTRACT_MARK}"
for tar in "${TARS[@]}"; do
  n="$(basename "${tar}")"
  grep -qxF "${n}" "${EXTRACT_MARK}" && continue
  tar -xzf "${tar}" -C "${SLP_DIR}/"
  echo "${n}" >> "${EXTRACT_MARK}"
done
N_SLP="$(find "${SLP_DIR}" -maxdepth 1 -name '*.slp' | wc -l)"
log "B ${N_SLP} .slp files"

log "C norm_stats"
if [[ ! -f "${DATA_DIR}/norm_stats.json" ]]; then
  python3 "${REPO_ROOT}/tools/build_norm_stats.py" \
    --slp-dir "${SLP_DIR}" --out-dir "${DATA_DIR}" --n-files 5000
  python3 "${REPO_ROOT}/tools/build_mimic_norm.py" \
    --norm-stats "${DATA_DIR}/norm_stats.json" \
    --minmax    "${DATA_DIR}/norm_minmax.json" \
    --out       "${DATA_DIR}/mimic_norm.json"
fi
[[ -f "${DATA_DIR}/stick_clusters.json" ]] || \
  cp "${REPO_ROOT}/hf_checkpoints/fox/stick_clusters.json" "${DATA_DIR}/stick_clusters.json"

log "D controller_combos"
if [[ ! -f "${DATA_DIR}/controller_combos.json" ]]; then
  cat > "${DATA_DIR}/controller_combos.json" <<'JSON'
{
    "button_names": ["A", "B", "Z", "JUMP", "TRIG", "A_TRIG", "NONE"],
    "n_combos": 7,
    "class_scheme": "melee_7class"
}
JSON
fi

log "E shards (32 workers — fox is using 64)"
if [[ ! -f "${DATA_DIR}/tensor_manifest.json" ]]; then
  python3 "${REPO_ROOT}/tools/slp_to_shards.py" \
    --slp-dir "${SLP_DIR}" \
    --meta-dir "${DATA_DIR}" \
    --mimic-norm "${DATA_DIR}/mimic_norm.json" \
    --character "${IDX}" \
    --staging-dir "${DATA_DIR}" \
    --repo "erickfm/mimic-${C}-v2" \
    --no-upload --keep-staging \
    --shard-gb 4.0 --val-frac 0.1 --seed 42 \
    --workers 32
fi

log "DONE — falco data ready. When the driver reaches falco, it will skip A-E and jump to training."
