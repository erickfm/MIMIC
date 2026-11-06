#!/usr/bin/env bash
# Rebuild tools/patches/melee-unlocked-mimic.patch from the working tree in sim/melee-unlocked:
# the patch's header comment, then `git diff` of tracked files and the new files' full text.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SRC="$REPO/sim/melee-unlocked"
PATCH="$REPO/tools/patches/melee-unlocked-mimic.patch"
OUT="${1:-$PATCH}"
NEW="port/linux/replay_host.cpp port/linux/replay_host.h"
tmp=$(mktemp)
sed -n '1,/^diff --git/p' "$PATCH" | sed '$d' > "$tmp"
{
  cat "$tmp"
  for f in $( (git -C "$SRC" diff --name-only -- . ':!sourceport/extern/melee'; echo $NEW | tr " " "\n") | LC_ALL=C sort); do
    if [[ " $NEW " == *" $f "* ]]; then
      (cd "$SRC" && git diff --no-index -- /dev/null "$f" || true) | sed "s#^diff --git a/$f b/$f#diff --git a/$f b/$f#"
    else
      git -C "$SRC" diff -- "$f"
    fi
  done
} > "$OUT.new"
mv "$OUT.new" "$OUT"
rm -f "$tmp"
