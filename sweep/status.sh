#!/usr/bin/env bash
# sweep/status.sh -- Show GPU utilization and running training jobs on all machines.
#
# Usage:  bash sweep/status.sh [MACHINE]
#   No args: show all machines.  A/B/C: show one machine.

set -euo pipefail

MACHINE_A="root@203.57.40.63"
PORT_A=10015
MACHINE_B="root@38.65.239.14"
PORT_B=28750
MACHINE_C="root@38.65.239.56"
PORT_C=45107

check_machine() {
    local label=$1 host=$2 port=$3
    echo "═══════════════════════════════════════════════════"
    echo "  Machine $label  ($host:$port)"
    echo "═══════════════════════════════════════════════════"

    echo "--- GPU Utilization ---"
    ssh -p "$port" "$host" "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader" 2>/dev/null || echo "  (ssh failed)"

    echo ""
    echo "--- Running train.py processes ---"
    ssh -p "$port" "$host" "ps aux | grep '[p]ython3 train.py' | awk '{print \"  GPU\" \$NF, \"pid=\" \$2, substr(\$0, index(\$0,\"--run-name\")+11, 40)}'" 2>/dev/null || echo "  (none)"

    echo ""
    echo "--- Latest log lines ---"
    ssh -p "$port" "$host" "for f in /root/FRAME/logs/sweep/wd-*.log; do [ -f \"\$f\" ] && echo \"  \$(basename \$f .log): \$(tail -1 \$f)\"; done 2>/dev/null" 2>/dev/null || echo "  (no logs)"
    echo ""
}

TARGET="${1:-all}"
case "$TARGET" in
    all)
        check_machine A "$MACHINE_A" "$PORT_A"
        check_machine B "$MACHINE_B" "$PORT_B"
        check_machine C "$MACHINE_C" "$PORT_C"
        ;;
    A|a) check_machine A "$MACHINE_A" "$PORT_A" ;;
    B|b) check_machine B "$MACHINE_B" "$PORT_B" ;;
    C|c) check_machine C "$MACHINE_C" "$PORT_C" ;;
    *) echo "Usage: status.sh [all|A|B|C]"; exit 1 ;;
esac
