#!/usr/bin/env bash
# Experiment: does a Fox model trained on ALL ranks pooled beat the master-only
# model head-to-head? Trains fox-allranks on data/fox_allranks_v2 (70k games,
# master+diamond+platinum pooled), then benchmarks vs fox-master (and platinum).
# Reuses existing shards — no download. Does NOT upload (experiment only).
set -uo pipefail
cd /home/erick/projects/MIMIC
export DISPLAY="${DISPLAY:-:0}"
DATE=20260616
LOG=logs/rank_fox; mkdir -p "${LOG}" reports
RUN=fox-allranks-${DATE}
TLOG="${LOG}/${RUN}.log"
log(){ printf "[%s] %s\n" "$(date -u +%H:%M:%S)" "$*" | tee -a "${LOG}/allranks.log"; }

log "training ${RUN} (70k games, all ranks pooled)"
python3 train.py --model mimic --encoder mimic_flat \
  --mimic-mode --mimic-controller-encoding --stick-clusters hal37 --plain-ce \
  --lr 3e-4 --batch-size 256 --grad-accum-steps 2 --max-samples 16777216 \
  --data-dir data/fox_allranks_v2 --self-inputs --reaction-delay 0 \
  --run-name "${RUN}" --no-warmup --cosine-min-lr 1e-6 > "${TLOG}" 2>&1 &
pid=$!
( PAT=20; mv=""; p=0
  while kill -0 "${pid}" 2>/dev/null; do
    sleep 60; cur="$(grep -oP 'val total=\K[0-9.]+' "${TLOG}" 2>/dev/null | tail -1)"; [[ -z "${cur}" ]] && continue
    if [[ -z "${mv}" ]] || awk "BEGIN{exit !(${cur}+0<${mv}+0)}"; then mv="${cur}"; p=0; else p=$((p+1)); fi
    if (( p>=PAT )); then log "watchdog: no improve ${p} (min ${mv}) stop"; pkill -TERM -P "${pid}" 2>/dev/null; kill -TERM "${pid}" 2>/dev/null; sleep 5; pkill -KILL -P "${pid}" 2>/dev/null; kill -KILL "${pid}" 2>/dev/null; break; fi
  done ) & wd=$!
wait "${pid}"; rc=$?; kill -TERM "${wd}" 2>/dev/null || true; wait "${wd}" 2>/dev/null || true
log "training done (rc=${rc})"; grep -E 'Best val_loss=' "${TLOG}" | tee -a "${LOG}/allranks.log" || true

sleep 60   # let GPU settle before Dolphin
DP=emulator/squashfs-root/usr/bin/dolphin-emu; ISO=melee.iso; META=data/foxrank_master_v2
ck="checkpoints/${RUN}_bestloss.pt"
bench(){ B=$1
  log "bench allranks vs ${B}"
  python3 tools/play.py --ckpt "${ck}" --opponent "checkpoints/fox-${B}-20260615_bestloss.pt" \
    --data-dir "${META}" --character FOX --opponent-character FOX --n-matches 15 --alternate-ports \
    --disable-audio --dolphin-path "${DP}" --iso-path "${ISO}" \
    --out "reports/allranks_vs_${B}.json" > "${LOG}/bench_allranks_vs_${B}.log" 2>&1 || log "bench vs ${B} errored"
  pkill -9 -x dolphin-emu 2>/dev/null; sleep 3
}
bench master
bench platinum
log "===== ALLRANKS EXPERIMENT DONE ====="
