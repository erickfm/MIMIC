#!/usr/bin/env bash
# Runs alongside run_rank_fox.sh. As each rank finishes training, uploads it to
# erickfm/MIMIC/fox-<rank>/ (ADDITIVE — new folders, never touches existing
# models). After all training completes (GPU free), runs the rank-ladder
# head-to-head benchmark. Upload is wrapped in timeout+retry (home uplink hangs).
set -uo pipefail
ROOT="/home/erick/projects/MIMIC"; cd "${ROOT}"
DATE=20260615
RANKS=( master diamond platinum )
PLOG="logs/rank_fox/pipeline.log"
FLOG="logs/rank_fox/finish.log"
mkdir -p reports
log() { printf "[%s] %s\n" "$(date -u +%H:%M:%S)" "$*" | tee -a "${FLOG}"; }

upload_rank() {
  local r=$1 ckpt="checkpoints/fox-${r}-${DATE}_bestloss.pt"
  [[ -f "${ckpt}" ]] || { log "upload/${r}: no checkpoint, skip"; return 1; }
  local a
  for a in 1 2 3 4 5 6; do
    if timeout 1200 python3 tools/upload_char.py --char "fox-${r}" \
         --checkpoint "${ckpt}" --data-dir "data/foxrank_${r}_v2" \
         --log "logs/rank_fox/fox-${r}-${DATE}.log" \
         >> "logs/rank_fox/upload_${r}.log" 2>&1; then
      log "upload/${r}: DONE -> erickfm/MIMIC/fox-${r}/"; return 0
    fi
    log "upload/${r}: attempt ${a} failed (uplink hang?), retry in 30s"; sleep 30
  done
  log "upload/${r}: FAILED after retries (checkpoint still local)"; return 1
}

# ---- per-rank upload as each finishes ----
for r in "${RANKS[@]}"; do
  log "waiting for fox-${r} training to finish..."
  until grep -q "\[train/${r}\] done" "${PLOG}" 2>/dev/null \
        || grep -q "pipeline complete" "${PLOG}" 2>/dev/null \
        || ! ps -eo cmd | grep -q "[t]ools/run_rank_fox.sh"; do sleep 60; done
  upload_rank "${r}" || true
done

# ---- benchmark (only once all training done; needs the GPU free) ----
log "waiting for full pipeline complete before benchmark..."
until grep -q "pipeline complete" "${PLOG}" 2>/dev/null \
      || ! ps -eo cmd | grep -q "[t]ools/run_rank_fox.sh"; do sleep 60; done
sleep 60   # let the GPU settle

export DISPLAY="${DISPLAY:-:0}"   # detached proc needs a display for Dolphin GUI
DP="${ROOT}/emulator/squashfs-root/usr/bin/dolphin-emu"
ISO="${ROOT}/melee.iso"
META="${ROOT}/data/foxrank_master_v2"   # shared metadata works for all 3
bench() {  # A B
  local A=$1 B=$2
  local ca="checkpoints/fox-${A}-${DATE}_bestloss.pt" cb="checkpoints/fox-${B}-${DATE}_bestloss.pt"
  [[ -f "${ca}" && -f "${cb}" ]] || { log "bench ${A} vs ${B}: missing ckpt, skip"; return; }
  log "bench ${A} vs ${B} (19 matches, realtime, alternate ports)"
  python3 tools/play.py --ckpt "${ca}" --opponent "${cb}" \
    --data-dir "${META}" --character FOX --opponent-character FOX \
    --n-matches 19 --alternate-ports --disable-audio \
    --dolphin-path "${DP}" --iso-path "${ISO}" \
    --out "reports/rankfox_${A}_vs_${B}.json" \
    > "logs/rank_fox/bench_${A}_vs_${B}.log" 2>&1 || log "bench ${A} vs ${B} errored (see log)"
}
bench master platinum
bench master diamond
bench diamond platinum
log "===== finish_rank_fox COMPLETE ====="
