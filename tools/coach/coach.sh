#!/usr/bin/env bash
# Start the coach (if it isn't running) and open it in the browser.
cd "$(dirname "$0")/../.."
URL=http://localhost:8765
if ! curl -s -o /dev/null "$URL/api/status"; then
  mkdir -p "$HOME/.cache/mimic-coach"
  nohup python3 tools/coach/serve.py > "$HOME/.cache/mimic-coach/server.log" 2>&1 &
  for _ in $(seq 60); do curl -s -o /dev/null "$URL/api/status" && break; sleep 0.5; done
fi
xdg-open "$URL" >/dev/null 2>&1 || echo "Open $URL"
