#!/usr/bin/env bash
# Waits for the fox-master continuation to finish, then launches the
# warm-restart from the BEST-VAL checkpoint with the LR schedule reset
# (keep Adam momentum, fresh cosine over the full horizon).
set -uo pipefail
cd /home/erick/projects/MIMIC
CLOG=logs/rank_fox/warmrestart_chain.log
log(){ printf "[%s] %s\n" "$(date -u +%H:%M:%S)" "$*" >> "${CLOG}"; }

log "waiting for fox-master-20260616-cont to finish..."
while ps -eo cmd | grep -q "[f]ox-master-20260616-cont"; do sleep 30; done
sleep 20   # let GPU settle

RUN=fox-master-20260616-warmrestart
# resume from the CONTINUATION's best (val ~0.728, better than the original
# 0.7334) — finalized once the continuation process has exited above.
RESUME=checkpoints/fox-master-20260616-cont_bestloss.pt
log "launching warm-restart ${RUN} from ${RESUME}, LR schedule reset"
python3 train.py --model mimic --encoder mimic_flat \
  --mimic-mode --mimic-controller-encoding --stick-clusters hal37 --plain-ce \
  --lr 3e-4 --batch-size 256 --grad-accum-steps 2 --max-samples 16777216 \
  --data-dir data/foxrank_master_v2 --self-inputs --reaction-delay 0 \
  --resume "${RESUME}" --warm-restart \
  --run-name "${RUN}" --no-warmup --cosine-min-lr 1e-6 \
  > "logs/rank_fox/${RUN}.log" 2>&1
rc=$?
log "warm-restart finished (rc=${rc})"
grep -aE "Best val_loss=" "logs/rank_fox/${RUN}.log" >> "${CLOG}" 2>/dev/null || true
log "===== WARM-RESTART CHAIN DONE ====="
