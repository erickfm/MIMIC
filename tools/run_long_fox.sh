#!/usr/bin/env bash
# Long master-only Fox run: fresh, ~480k steps (~1 epoch), single cosine to
# 1e-6, no early-stop. Crash-resilient: if train.py dies before printing the
# final "Best val_loss=", resume from the latest periodic checkpoint (keeps
# optimizer + scheduler, continues the same schedule) and retry.
set -uo pipefail
cd /home/erick/projects/MIMIC
RUN=fox-master-20260616-long
LOG=logs/rank_fox/${RUN}.log
CLOG=logs/rank_fox/${RUN}_runner.log
mkdir -p logs/rank_fox checkpoints
log(){ printf "[%s] %s\n" "$(date -u +%F\ %H:%M:%S)" "$*" >> "${CLOG}"; }

ARGS="--model mimic --encoder mimic_flat --mimic-mode --mimic-controller-encoding \
--stick-clusters hal37 --plain-ce --lr 3e-4 --batch-size 256 --grad-accum-steps 2 \
--max-samples 245760000 --data-dir data/foxrank_master_v2 --self-inputs --reaction-delay 0 \
--run-name ${RUN} --no-warmup --cosine-min-lr 1e-6"

: > "${LOG}"   # fresh log so the "Best val_loss=" completion check is clean
log "===== long run start: ${RUN}, target 480k steps ====="
for attempt in $(seq 1 10); do
  # resume from latest periodic checkpoint if one exists (crash recovery)
  latest=$(ls -t checkpoints/${RUN}_step*.pt 2>/dev/null | head -1)
  resume=""
  if [[ -n "${latest}" ]]; then
    resume="--resume ${latest}"
    log "attempt ${attempt}: resuming from ${latest}"
  else
    log "attempt ${attempt}: fresh start"
  fi

  python3 train.py ${ARGS} ${resume} >> "${LOG}" 2>&1
  rc=$?

  if grep -aq "Best val_loss=" "${LOG}"; then
    log "attempt ${attempt}: COMPLETED (rc=${rc})"
    grep -aE "Best val_loss=" "${LOG}" | tail -1 >> "${CLOG}"
    break
  fi
  log "attempt ${attempt}: died early (rc=${rc}); will resume in 30s"
  pkill -9 -f "[t]rain.py.*${RUN}" 2>/dev/null
  sleep 30
done
log "===== LONG RUN DONE ====="
