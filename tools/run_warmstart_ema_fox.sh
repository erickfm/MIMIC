#!/usr/bin/env bash
# Experiment (2026-06-18): reach the long-run's val (0.713) in far fewer steps,
# batch held constant. Recipe routed by the wandb curves:
#   warm-start from the 0.7334 base  +  short MATCHED cosine (decay fully by the
#   end, not a 480k bloat)  +  EMA (eval/promote on averaged weights).
# Target: ~0.715 by ~80k, approach 0.713 by ~120k  (vs 442k for the long run).
# eff batch = 256 * 2 = 512 (SAME as the long run — batch is held constant).
# Crash-resilient: first attempt warm-starts from BASE; retries resume the
# latest periodic checkpoint (continue, EMA state restored from the ckpt).
set -uo pipefail
cd /home/erick/projects/MIMIC
RUN=fox-master-20260618-ws-ema
BASE=checkpoints/fox-master-20260615_bestloss.pt   # the 0.7334 original
LOG=logs/rank_fox/${RUN}.log
CLOG=logs/rank_fox/${RUN}_runner.log
mkdir -p logs/rank_fox checkpoints
log(){ printf "[%s] %s\n" "$(date -u +%F\ %H:%M:%S)" "$*" >> "${CLOG}"; }

COMMON="--model mimic --encoder mimic_flat --mimic-mode --mimic-controller-encoding \
--stick-clusters hal37 --plain-ce --lr 3e-4 --batch-size 256 --grad-accum-steps 2 \
--max-samples 61440000 --data-dir data/foxrank_master_v2 --self-inputs --reaction-delay 0 \
--run-name ${RUN} --no-warmup --cosine-min-lr 1e-6 --ema-decay 0.999"

: > "${LOG}"
log "===== warm-start+EMA run: ${RUN}, target ~120k steps, eff batch 512 ====="
for attempt in $(seq 1 10); do
  latest=$(ls -t checkpoints/${RUN}_step*.pt 2>/dev/null | head -1)
  if [[ -n "${latest}" ]]; then
    resume="--resume ${latest}"            # crash recovery: continue, keep EMA
    log "attempt ${attempt}: resuming from ${latest}"
  else
    resume="--resume ${BASE} --warm-restart"  # first launch: warm-start + LR reset
    log "attempt ${attempt}: warm-start from ${BASE}"
  fi
  python3 train.py ${COMMON} ${resume} >> "${LOG}" 2>&1
  rc=$?
  if grep -aq "Best val_loss=" "${LOG}"; then
    log "attempt ${attempt}: COMPLETED (rc=${rc})"
    grep -aE "Best val_loss=" "${LOG}" | tail -1 >> "${CLOG}"
    break
  fi
  log "attempt ${attempt}: died early (rc=${rc}); resume in 30s"
  pkill -9 -f "[t]rain.py.*${RUN}" 2>/dev/null
  sleep 30
done
log "===== WARM-START+EMA RUN DONE ====="
