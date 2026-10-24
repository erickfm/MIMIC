#!/usr/bin/env bash
# Confirm a sim RL run against frozen BC: 480 games at seed 51; if that win is significant, a
# second 480 at seed 61 (the acceptance bar in docs/research-notes-2026-10-23.md).
#   vr_rl_confirm.sh RUN_DIR
cd /home/erick/projects/MIMIC
P=${VR_POLICY:-hf_eval/fox-master}   # learner start = frozen BC reference = h2h opponent
d=$1 R=${VR_RL_DIR:-checkpoints/vrsim}; name=$(basename $d)
h() { python3 tools/sim_h2h.py --char FOX --a rl=$d/eval_final:bf16 --b bc=$P:bf16 \
        --templates replays_doc_rt --games 480 --envs 48 --workers 16 --seed $1 --out $d/h2h480_s$1.json \
        > $d/h2h480_s$1.log 2>&1
      line=$(grep -E ' vs ' $d/h2h480_s$1.log | grep -oE '[0-9]+-[0-9]+ .*p=[0-9.]+')
      echo "$(date -u +%FT%TZ) $name h2h480 seed$1 | $line" >> $R/results.txt; echo "$line"; }
l=$(h 51)
wr=$(echo "$l" | grep -oE 'win rate [0-9.]+' | awk '{print $3}'); p=$(echo "$l" | grep -oE 'p=[0-9.]+' | cut -d= -f2)
python3 -c "import sys; sys.exit(0 if float('$wr') > 0.5 and float('$p') < 0.05 else 1)" && h 61 > /dev/null
