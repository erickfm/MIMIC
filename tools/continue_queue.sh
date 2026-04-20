#!/usr/bin/env bash
# Serial continue-queue: runs continue_xxl_char.sh for peach then falco.
set -uo pipefail
REPO_ROOT="/root/MIMIC"
"${REPO_ROOT}/tools/continue_xxl_char.sh" peach PEACH 9
"${REPO_ROOT}/tools/continue_xxl_char.sh" falco FALCO 22
echo "[$(date -u +%H:%M:%S)] continue_queue done"
