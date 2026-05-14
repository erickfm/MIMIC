#!/usr/bin/env bash
# 4-way win-rate eval: BC baseline + the 3 PPO-trained checkpoints
# produced by run_3way.sh. N matches per ckpt vs CPU-level-9 Fox on FD.
# Runs after training completes.
#
# Run from repo root:
#   DISPLAY=:99 bash rlvr/online/run_3way_eval.sh 2>&1 | tee logs/3way_eval.log
set -e

DATE=${EVAL_DATE:-$(date -u +%Y%m%d)}
N_MATCHES=${N_MATCHES:-30}
CKPT_ROOT="checkpoints/3way_${DATE}"
REPORT_DIR="reports/3way_${DATE}"
DATA_DIR="hf_checkpoints/fox"
DOLPHIN="emulator_ffw/squashfs-root/usr/bin/dolphin-emu"
ISO="melee.iso"

mkdir -p "${REPORT_DIR}" replays_winrate logs

eval_one () {
  local LABEL=$1
  local CKPT=$2
  echo ""
  echo "======================================================================"
  echo "[$(date -Iseconds)]  EVAL ${LABEL}  (${N_MATCHES} matches)"
  echo "  ckpt: ${CKPT}"
  echo "======================================================================"
  if [[ ! -f "${CKPT}" ]]; then
    echo "  MISSING — skipping"
    return
  fi
  python3 -m rlvr.eval.winrate_vs_cpu \
      --ckpt "${CKPT}" \
      --data-dir "${DATA_DIR}" \
      --dolphin-path "${DOLPHIN}" \
      --iso-path "${ISO}" \
      --n-matches ${N_MATCHES} \
      --use-exi-inputs --enable-ffw \
      --gfx-backend Null \
      --disable-audio \
      --replay-dir "replays_winrate/${LABEL}" \
      --out "${REPORT_DIR}/${LABEL}.json"
}

# Baseline: starting BC ckpt (no further training)
eval_one bc-baseline  "hf_checkpoints/fox/model.pt"
# 3 PPO-trained ckpts (use _final.pt — the saved end-of-training snapshot)
eval_one lcancel      "${CKPT_ROOT}/3way-${DATE}-lcancel_final.pt"
eval_one shieldesc    "${CKPT_ROOT}/3way-${DATE}-shieldesc_final.pt"
eval_one comboext     "${CKPT_ROOT}/3way-${DATE}-comboext_final.pt"

echo ""
echo "[$(date -Iseconds)]  ALL EVALS DONE  →  ${REPORT_DIR}/"
echo ""
echo "=== summary ==="
for f in "${REPORT_DIR}"/*.json; do
  python3 -c "
import json, sys
r = json.load(open('$f'))
print(f\"{r['ckpt'].split('/')[-1]:<55}  N={r['n_matches']:>3}  W={r['win']}  L={r['loss']}  D={r['draw']}  win-rate={100*r['win_rate']:.1f}%\")
"
done
