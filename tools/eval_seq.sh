#!/usr/bin/env bash
# Sequential h2h evals (ONE Dolphin at a time — no concurrency contention).
# Discards the suspect concurrent-run results. Each play.py writes its --out
# incrementally so partial results survive a stall.
cd /home/erick/projects/MIMIC
export DISPLAY=:0
STAGES="FINAL_DESTINATION,BATTLEFIELD,DREAMLAND,YOSHIS_STORY,FOUNTAIN_OF_DREAMS,POKEMON_STADIUM"
DP=emulator/squashfs-root/usr/bin/dolphin-emu
DATA=data/foxrank_master_v2
ISO=melee.iso
mkdir -p reports logs
# discard suspect concurrent results
rm -f reports/mirror_vs_nomirror_stages.json reports/decode_T*.json
rm -f reports/seq_*.json

echo "=== [1/5] MIRROR: AVG_mirror(A) vs AVG_init-knee(B), 36 matches, 6 stages ($(date +%H:%M)) ==="
python3 tools/play.py --ckpt checkpoints/AVG_mirror_last5.pt --opponent checkpoints/AVG_init-knee_last5.pt \
  --data-dir "$DATA" --dolphin-path "$DP" --iso-path "$ISO" \
  --n-matches 36 --alternate-ports --stages "$STAGES" --out reports/seq_mirror.json > logs/seq_mirror.log 2>&1
echo "  mirror: $(python3 -c "import json;d=json.load(open('reports/seq_mirror.json'));print('%d-%d (%.3f)'%(d['a_wins'],d['b_wins'],d['a_win_rate']))" 2>/dev/null)"

# Decode: production fox-master bot @ T vs SAME model @ T=1.0 (self-play).
# T=1.0 first is the fairness control (should be ~0.50).
i=2
for T in 1.0 0.0 0.5 0.7; do
  echo "=== [$i/5] DECODE bot T=$T vs opp T=1.0, 16 matches, 6 stages ($(date +%H:%M)) ==="
  python3 tools/play.py --ckpt checkpoints/AVG_mastfox.pt --opponent checkpoints/AVG_mastfox.pt \
    --data-dir "$DATA" --dolphin-path "$DP" --iso-path "$ISO" \
    --n-matches 16 --alternate-ports --stages "$STAGES" \
    --temperature "$T" --opp-temperature 1.0 --out "reports/seq_decode_T${T}.json" > "logs/seq_decode_T${T}.log" 2>&1
  echo "  T=$T: $(python3 -c "import json;d=json.load(open('reports/seq_decode_T${T}.json'));print('%.3f (%d-%d)'%(d['a_win_rate'],d['a_wins'],d['b_wins']))" 2>/dev/null)"
  i=$((i+1))
done
echo "SEQ_EVAL_DONE $(date +%H:%M)"
