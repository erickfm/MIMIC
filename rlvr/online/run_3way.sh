#!/usr/bin/env bash
# 3-way PPO comparison: l_cancel / shield_escape / combo_extend on Fox.
# Sequential because we share one GPU + one Dolphin actor at a time.
# All three start from the same BC base; same hyperparameters.
#
# Run from repo root:
#   DISPLAY=:99 bash rlvr/online/run_3way.sh 2>&1 | tee logs/3way.log
set -e

BASE_CKPT="hf_checkpoints/fox/model.pt"
DATA_DIR="hf_checkpoints/fox"
DOLPHIN="emulator_ffw/squashfs-root/usr/bin/dolphin-emu"
ISO="melee.iso"

EPISODES_PER_UPDATE=32
MAX_UPDATES=50
DATE=$(date -u +%Y%m%d)
CKPT_ROOT="checkpoints/3way_${DATE}"

mkdir -p "${CKPT_ROOT}" replays_3way logs

run_one () {
  local TASK=$1
  local TAG=$2
  echo ""
  echo "======================================================================"
  echo "[$(date -Iseconds)]  TRAIN ${TASK}  →  ${CKPT_ROOT}/${TAG}"
  echo "======================================================================"
  python3 -m rlvr.online.loop \
      --base-ckpt "${BASE_CKPT}" \
      --data-dir "${DATA_DIR}" \
      --dolphin-path "${DOLPHIN}" \
      --iso-path "${ISO}" \
      --task "${TASK}" \
      --run-name "${TAG}" \
      --episodes-per-update ${EPISODES_PER_UPDATE} \
      --max-updates ${MAX_UPDATES} \
      --checkpoint-every ${MAX_UPDATES} \
      --checkpoint-dir "${CKPT_ROOT}" \
      --use-exi-inputs --enable-ffw \
      --gfx-backend Null \
      --replay-dir "replays_3way/${TAG}"
}

run_one l_cancel_online      "3way-${DATE}-lcancel"
run_one shield_escape_online "3way-${DATE}-shieldesc"
run_one combo_extend_online  "3way-${DATE}-comboext"

echo ""
echo "[$(date -Iseconds)]  ALL 3 RUNS COMPLETE  →  ${CKPT_ROOT}/"
ls -lh "${CKPT_ROOT}/"
