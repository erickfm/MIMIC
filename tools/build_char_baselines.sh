#!/usr/bin/env bash
# Build per-character canary baselines from human master-rank replays.
# For each char: download its master-master archive(s) from HF, extract, run
# validate_run --emit-baseline, merge into data/canary_baselines.json, clean up.
# Usage: tools/build_char_baselines.sh doc mewtwo peach ness
set -u
cd "$(dirname "$0")/.."
ROOT="$PWD"
BASE="data/canary_baselines.json"
TMP="/tmp/canary_corpus"
mkdir -p "$TMP"
declare -A BUCKET=( [puff]=JIGGLYPUFF [ice_climbers]=ICE_CLIMBERS )

merge() {  # merge a per-char json into the master baseline
  python3 - "$BASE" "$1" <<'PY'
import json, sys, os
base_p, add_p = sys.argv[1], sys.argv[2]
base = json.load(open(base_p)) if os.path.exists(base_p) else {}
base.update(json.load(open(add_p)))
json.dump(base, open(base_p, "w"), indent=1)
print(f"  merged {list(json.load(open(add_p)))} -> {base_p} ({len(base)} chars total)")
PY
}

for CH in "$@"; do
  BK="${BUCKET[$CH]:-$(echo "$CH" | tr '[:lower:]' '[:upper:]')}"
  echo "=== $CH (bucket $BK)"
  D="$TMP/$CH"; rm -rf "$D"; mkdir -p "$D"
  # a1 usually holds plenty; add a2 for the tiny/rare chars
  hf download erickfm/melee-ranked-replays --repo-type dataset \
    --include "${BK}/${BK}_master-master_a1.tar.gz" --local-dir "$D" >/dev/null 2>&1
  TAR=$(find "$D" -name '*.tar.gz' | head -1)
  [ -z "$TAR" ] && { echo "  no archive for $BK, skip"; continue; }
  tar xzf "$TAR" -C "$D" 2>/dev/null
  N=$(find "$D" -name '*.slp' | wc -l)
  echo "  $N replays extracted"
  [ "$N" -eq 0 ] && { echo "  no slp, skip"; continue; }
  PYTHONPATH="$ROOT" python3 tools/validate_run.py "$D" --chars "$CH" \
    --n-files 500 --emit-baseline "$TMP/${CH}_bl.json" >/dev/null 2>&1
  [ -f "$TMP/${CH}_bl.json" ] && merge "$TMP/${CH}_bl.json"
  rm -rf "$D"   # reclaim disk
done
echo "=== done. baseline table:"
python3 -c "import json;print(json.dumps(json.load(open('$BASE')),indent=1))"
