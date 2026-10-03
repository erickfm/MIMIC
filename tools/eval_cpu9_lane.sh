#!/usr/bin/env bash
# One eval lane: for each char, pull from HF (if absent) and play N FFW
# matches vs Fox CPU-9, writing eval_results/<char>.json. Args: LANE PORT N CHARS...
set -uo pipefail
cd /home/erick/projects/MIMIC
export HF_TOKEN=$(grep '^HF_TOKEN' /home/erick/Documents/.env | cut -d= -f2)
LANE=$1; PORT=$2; N=$3; shift 3
CHARS=("$@")
SUFFIX="${SUFFIX:-}"   # output file suffix, e.g. _r2 (keeps round-1 json)
SEED="${SEED:-42}"
RES=eval_results
LOG=${RES}/lane_${LANE}.log
mkdir -p ${RES} replays_${LANE}
: > "${LOG}"

declare -A ENUM=(
  [mewtwo]=MEWTWO [ylink]=YLINK [ness]=NESS [roy]=ROY [mario]=MARIO
  [gameandwatch]=GAMEANDWATCH [bowser]=BOWSER [link]=LINK [doc]=DOC [dk]=DK
  [ganondorf]=GANONDORF [yoshi]=YOSHI [pikachu]=PIKACHU [luigi]=LUIGI
  [ice_climbers]=POPO [samus]=SAMUS [peach]=PEACH [cptfalcon]=CPTFALCON
  [puff]=JIGGLYPUFF [sheik]=SHEIK [marth]=MARTH [falco]=FALCO
)
STAGES=FINAL_DESTINATION,BATTLEFIELD,DREAMLAND,YOSHIS_STORY,FOUNTAIN_OF_DREAMS,POKEMON_STADIUM

log() { printf "[%s] %s\n" "$(date -u +%H:%M:%S)" "$*" >> "${LOG}"; }

for C in "${CHARS[@]}"; do
  E=${ENUM[$C]}
  if [[ ! -f hf_eval/${C}/model.pt ]]; then
    log "[$C] downloading from HF"
    hf download erickfm/MIMIC --include "${C}/*" --local-dir hf_eval >> "${LOG}" 2>&1
  fi
  [[ -f hf_eval/${C}/model.pt ]] || { log "[$C] NO model.pt — skip"; continue; }
  log "[$C] ${N} FFW matches: ${E} vs Fox CPU-9"
  timeout -k 30 5400 python3 tools/play.py \
    --ckpt hf_eval/${C}/model.pt --data-dir hf_eval/${C} \
    --opponent cpu:9 --character "${E}" --opponent-character FOX \
    --n-matches "${N}" --stages "${STAGES}" \
    --use-exi-inputs --enable-ffw --gfx-backend Null --disable-audio \
    `# vs-CPU is single-injected-bot FFW (faithful even on stock emulator_ffw),`\
    `# but use emulator_ss everywhere so no tool can accidentally run the`\
    `# broken dual-pad build. See docs/research-notes-2026-07-22.md.`\
    --dolphin-path emulator_ss/Binaries/dolphin-emu \
    --iso-path melee.iso --slippi-port "${PORT}" --replay-dir "replays_${LANE}" \
    --seed "${SEED}" --out "${RES}/${C}${SUFFIX}.json" >> "${LOG}" 2>&1
  if [[ -f "${RES}/${C}${SUFFIX}.json" ]]; then
    WR=$(python3 -c "import json;d=json.load(open('${RES}/${C}${SUFFIX}.json'));print(d.get('a_win_rate',d.get('win_rate','?')))" 2>/dev/null)
    log "[$C] DONE win_rate=${WR}"
  else
    log "[$C] FAILED (no json)"
  fi
  rm -f replays_${LANE}/*.slp 2>/dev/null
done
log "[lane ${LANE}] ALL DONE"
