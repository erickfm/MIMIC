#!/usr/bin/env bash
# One sim RL run with verifiable rewards + a 192-game h2h screen vs frozen BC (docs/research-notes-2026-10-23.md).
#   vr_rl_run.sh NAME SPEC UPDATES [extra sim_rl flags]
# The recipe is sim_rl's dense defaults (since 2026-10-27: 16,384 decisions per update, 4 PPO epochs, lr 1e-5,
# critic baseline, per-decision KL 0.03, no global anchor; docs/research-notes-2026-10-27.md), so pass UPDATES=100:
#   VR_POLICY=checkpoints/smallfox/base_full56k vr_rl_run.sh NAME "stock_delta:1,damage_delta:1" 100
cd /home/erick/projects/MIMIC
P=${VR_POLICY:-hf_eval/fox-master}   # learner start = frozen BC reference = h2h opponent
R=${VR_RL_DIR:-checkpoints/vrsim} name=$1 spec=$2 upd=$3; shift 3
out=$R/$name; mkdir -p $out
python3 tools/sim_rl.py --policy-dir $P --templates replays_doc_rt --out-dir $out \
    --envs 64 --workers 16 --updates $upd --vr "$spec" "$@" > $out/train.log 2>&1 \
    || { echo "$(date -u +%FT%TZ) $name TRAIN FAILED (rc=$?)" >> $R/results.txt; exit 1; }
ck=$(ls $out/rl_u*.pt | sort | tail -1); ev=$out/eval_final; mkdir -p $ev
cp $P/*.json $ev/; ln -sf "$(realpath $ck)" $ev/model.pt
python3 tools/sim_h2h.py --char FOX --a rl=$ev:bf16 --b bc=$P:bf16 --templates replays_doc_rt \
    --games 192 --envs 48 --workers 16 --seed 41 --out $out/h2h192.json > $out/h2h192.log 2>&1
roll=$(python3 - "$out/metrics.jsonl" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.startswith("{")]
w = sum(r["games"]["win"] for r in rows[-10:]); l = sum(r["games"]["loss"] for r in rows[-10:])
first, last = rows[0]["vr"], rows[-1]["vr"]
vr = " ".join(f"{k} {first[k]['learn']:+.3f}->{last[k]['learn']:+.3f}" for k in last)
print(f"rollout last10 {w}-{l} | /min {vr} | kl {rows[-1]['kl']:.4f}")
PY
)
echo "$(date -u +%FT%TZ) $name [$spec] u$upd | h2h192: $(grep -E ' vs ' $out/h2h192.log | grep -oE '[0-9]+-[0-9]+ .*p=[0-9.]+') | $roll" >> $R/results.txt
