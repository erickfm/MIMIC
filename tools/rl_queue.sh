#!/usr/bin/env bash
# Run one RL recipe on a list of BC parents, one after another on one GPU (tools/run.py launch, seed 1;
# each run ends with its 192-game h2h vs its parent). Parents whose run is already done are skipped,
# so it's safe to restart. Waits first until WAIT_CMD exits 0 (use it to keep the GPU to one job,
# e.g. a test that the job currently on it has finished; "true" to start at once).
# To chain after another queue, wait on its PID ('! kill -0 PID'), not on a grep of the queue log:
# this script writes WAIT_CMD into that log, so the grep matches its own line and starts at once.
#
#   tools/rl_queue.sh RECIPE GPU 'WAIT_CMD' PARENT...
#   tools/rl_queue.sh recipes/rl-v4.yaml 0 true falco-S-bc2-hf28k-s1 marth-S-bc2-hf21k-s1
#
# Log: runs/rl_queue.gpu<GPU>.log (one line per start / result / failure).
set -u
cd "$(dirname "$0")/.."
recipe=$1 gpu=$2 wait=$3; shift 3
tag=$(python3 -c "import yaml;print(yaml.safe_load(open('$recipe'))['tag'])")
LOG=runs/rl_queue.gpu$gpu.log
say() { echo "[$(TZ=America/Los_Angeles date '+%a %-I:%M %p')] $*" >> "$LOG"; }
status() { python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('status',''))" "runs/$1/run.json" 2>/dev/null; }

say "queue: $recipe on GPU $gpu, ${#@} parents; waiting for: $wait"
until bash -c "$wait"; do sleep 60; done
for p in "$@"; do
  rl=$p.$tag-s1
  [ "$(status $rl)" = done ] && { say "$rl already done"; continue; }
  if [ -e runs/$rl ]; then mv runs/$rl runs/$rl.failed-$(date +%s); fi
  say "start $rl"
  CUDA_VISIBLE_DEVICES=$gpu python3 tools/run.py launch "$recipe" --parent "$p" --seed 1 >> runs/rl_queue.gpu$gpu.out 2>&1
  if [ "$(status $rl)" = done ]; then
    r=$(ls runs/$rl/eval/h2h_vs_${p}_g192_s41.result.json 2>/dev/null)
    say "done $rl: $( [ -n "$r" ] && python3 -c "import json;r=json.load(open('$r'));print(f\"{r['a_wins']}-{r['b_wins']} vs BC\")")"
  else
    say "FAILED $rl (runs/$rl/logs/train.log)"
  fi
done
say "queue finished"
