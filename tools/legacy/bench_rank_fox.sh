#!/usr/bin/env bash
# Rank-ladder head-to-head for the fox-{master,diamond,platinum}-20260615 models.
# Realtime (FFW is unfaithful), alternate ports, 15 matches/matchup. Writes
# reports/rankfox_<A>_vs_<B>.json. play.py --character takes the enum NAME.
set -uo pipefail
cd /home/erick/projects/MIMIC
export DISPLAY="${DISPLAY:-:0}"
DATE=20260615
DP=emulator/squashfs-root/usr/bin/dolphin-emu
ISO=melee.iso
META=data/foxrank_master_v2   # shared metadata works for all 3 models
mkdir -p reports
BLOG=logs/rank_fox/bench.log

bench() {
  local A=$1 B=$2
  local ca="checkpoints/fox-${A}-${DATE}_bestloss.pt" cb="checkpoints/fox-${B}-${DATE}_bestloss.pt"
  echo "[$(date -u +%H:%M)] bench ${A} vs ${B} start" >> "${BLOG}"
  python3 tools/play.py --ckpt "${ca}" --opponent "${cb}" \
    --data-dir "${META}" --character FOX --opponent-character FOX \
    --n-matches 15 --alternate-ports --disable-audio \
    --dolphin-path "${DP}" --iso-path "${ISO}" \
    --out "reports/rankfox_${A}_vs_${B}.json" \
    > "logs/rank_fox/bench_${A}_vs_${B}.log" 2>&1 \
    && echo "[$(date -u +%H:%M)] ${A} vs ${B} DONE" >> "${BLOG}" \
    || echo "[$(date -u +%H:%M)] ${A} vs ${B} ERRORED" >> "${BLOG}"
  pkill -9 -x dolphin-emu 2>/dev/null; sleep 3
}

echo "[$(date -u +%H:%M)] ===== rank-fox benchmark start =====" > "${BLOG}"
bench master platinum
bench master diamond
bench diamond platinum
echo "[$(date -u +%H:%M)] ===== ALL BENCH DONE =====" >> "${BLOG}"
