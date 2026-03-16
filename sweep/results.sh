#!/usr/bin/env bash
# sweep/results.sh -- Parse logs across all machines for completed runs.
#
# Usage:  bash sweep/results.sh
#
# Outputs a leaderboard sorted by wall time (fastest first).

set -euo pipefail

MACHINE_A="root@203.57.40.63"
PORT_A=10015
MACHINE_B="root@38.65.239.14"
PORT_B=28750
MACHINE_C="root@38.65.239.56"
PORT_C=45107

TMPFILE=$(mktemp)
trap "rm -f $TMPFILE" EXIT

parse_machine() {
    local label=$1 host=$2 port=$3
    ssh -p "$port" "$host" "
        for f in /root/FRAME/logs/sweep/wd-*.log; do
            [ -f \"\$f\" ] || continue
            name=\$(basename \$f .log)

            target_line=\$(grep 'TARGET REACHED' \$f 2>/dev/null | tail -1)
            if [ -n \"\$target_line\" ]; then
                wall=\$(echo \"\$target_line\" | grep -oP 'in \\K[0-9.]+')
                step=\$(echo \"\$target_line\" | grep -oP 'step \\K[0-9]+')
                f1=\$(echo \"\$target_line\" | grep -oP 'val_f1=\\K[0-9.]+')
                echo \"HIT|$label|\$name|\${wall}s|step \$step|f1=\$f1\"
                continue
            fi

            wall_line=\$(grep 'WALL TIME LIMIT' \$f 2>/dev/null | tail -1)
            if [ -n \"\$wall_line\" ]; then
                step=\$(echo \"\$wall_line\" | grep -oP 'step \\K[0-9]+')
                echo \"TIMEOUT|$label|\$name|-|step \$step|-\"
                continue
            fi

            notreached=\$(grep 'TARGET NOT REACHED' \$f 2>/dev/null | tail -1)
            if [ -n \"\$notreached\" ]; then
                f1=\$(echo \"\$notreached\" | grep -oP 'best val_f1=\\K[0-9.]+')
                echo \"MISS|$label|\$name|-|-|f1=\$f1\"
                continue
            fi

            done_line=\$(grep '^Done\\.' \$f 2>/dev/null | tail -1)
            if [ -n \"\$done_line\" ]; then
                echo \"DONE|$label|\$name|-|-|-\"
                continue
            fi

            echo \"RUNNING|$label|\$name|-|-|-\"
        done
    " 2>/dev/null >> "$TMPFILE"
}

parse_machine A "$MACHINE_A" "$PORT_A"
parse_machine B "$MACHINE_B" "$PORT_B"
parse_machine C "$MACHINE_C" "$PORT_C"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              WAVEDASH SPEED SWEEP LEADERBOARD               ║"
echo "╠══════════════════════════════════════════════════════════════╣"

if [ -s "$TMPFILE" ]; then
    echo "║  Status  │ Mach │ Run Name              │ Wall   │ Step    │ F1"
    echo "║──────────┼──────┼───────────────────────┼────────┼─────────┼──────"
    sort -t'|' -k1,1 -k4,4n "$TMPFILE" | while IFS='|' read status mach name wall step f1; do
        printf "║  %-7s │  %s   │ %-21s │ %6s │ %-7s │ %s\n" "$status" "$mach" "$name" "$wall" "$step" "$f1"
    done
else
    echo "║  No results yet.                                            ║"
fi
echo "╚══════════════════════════════════════════════════════════════╝"
