#!/usr/bin/env python3
"""Analyze inference log output and report gameplay health metrics.

Parses the MAIN/C/L/BTN/top3 lines from run_hal_model.py or inference.py logs
and computes 8 diagnostic metrics that distinguish intentional play from
bugged/idle behavior.

Usage:
    python tools/run_hal_model.py ... 2>&1 | tee game.log
    python tools/gameplay_health.py game.log

    # Or pipe directly:
    python tools/run_hal_model.py ... 2>&1 | python tools/gameplay_health.py -
"""

import argparse
import re
import sys
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class HealthCheck:
    name: str
    key: str
    value: float
    threshold: float
    op: str  # ">=" or "<="
    unit: str

    @property
    def passed(self) -> bool:
        if self.op == ">=":
            return self.value >= self.threshold
        return self.value <= self.threshold

    def fmt_value(self) -> str:
        if self.unit == "%":
            return f"{self.value:.1f}%"
        elif self.unit == "/sec":
            return f"{self.value:.1f}/sec"
        elif self.unit == "count":
            return str(int(self.value))
        return f"{self.value:.4f}"

    def fmt_threshold(self) -> str:
        if self.unit == "%":
            return f"{self.op} {self.threshold:.0f}%"
        elif self.unit == "/sec":
            return f"{self.op} {self.threshold:.1f}/sec"
        elif self.unit == "count":
            return f"{self.op} {int(self.threshold)}"
        return f"{self.op} {self.threshold}"


def parse_log(lines: List[str]) -> List[dict]:
    """Parse MAIN/C/L/BTN/top3 lines into frame dicts."""
    pattern = re.compile(
        r"MAIN=\(([0-9.]+),([0-9.]+)\) "
        r"C=\(([0-9.]+),([0-9.]+)\) "
        r"L=([0-9.]+) "
        r"BTN=\[([^\]]*)\] +"
        r"top3=\[(.+)\]"
    )
    frames = []
    for line in lines:
        m = pattern.search(line)
        if not m:
            continue
        btn_str = m.group(6).strip()
        btns = [b.strip().strip("'") for b in btn_str.split(",") if b.strip()] if btn_str else []

        probs = {}
        for part in m.group(7).split():
            if "=" in part:
                name, val = part.split("=")
                probs[name] = float(val)

        frames.append({
            "mx": float(m.group(1)),
            "my": float(m.group(2)),
            "cx": float(m.group(3)),
            "cy": float(m.group(4)),
            "l": float(m.group(5)),
            "btns": btns,
            "probs": probs,
            "none_prob": probs.get("NONE", 0.0),
        })
    return frames


def compute_health(frames: List[dict]) -> List[HealthCheck]:
    """Compute 8 gameplay health metrics."""
    n = len(frames)
    if n == 0:
        return []

    duration_sec = n / 60.0

    # 1. Button press rate
    btn_initiations = sum(
        1 for i in range(1, n)
        if frames[i]["btns"] and not frames[i - 1]["btns"]
    )

    # 2. Button variety
    all_btns = set()
    for f in frames:
        all_btns.update(f["btns"])
    btn_variety = len(all_btns)

    # 3. Stick at neutral %
    mx = np.array([f["mx"] for f in frames])
    my = np.array([f["my"] for f in frames])
    neutral = ((np.abs(mx - 0.5) < 0.05) & (np.abs(my - 0.5) < 0.05)).sum()
    stick_neutral_pct = 100.0 * neutral / n

    # 4. Unique stick positions
    stick_positions = set()
    for f in frames:
        stick_positions.add((round(f["mx"], 2), round(f["my"], 2)))

    # 5. Mean NONE probability
    none_probs = np.array([f["none_prob"] for f in frames])
    none_mean = float(none_probs.mean())

    # 6. NONE < 0.99 %
    none_below_99 = 100.0 * (none_probs < 0.99).sum() / n

    # 7. Action initiation % (non-NONE is argmax)
    action_frames = sum(
        1 for f in frames
        if max(f["probs"], key=f["probs"].get) != "NONE"
    )
    action_pct = 100.0 * action_frames / n

    # 8. Shoulder active %
    l_vals = np.array([f["l"] for f in frames])
    shoulder_pct = 100.0 * (l_vals > 0.01).sum() / n

    return [
        HealthCheck("Button press rate", "btn_press_rate",
                    btn_initiations / duration_sec, 0.5, ">=", "/sec"),
        HealthCheck("Unique buttons", "btn_variety",
                    btn_variety, 3, ">=", "count"),
        HealthCheck("Stick at neutral", "stick_neutral_pct",
                    stick_neutral_pct, 80.0, "<=", "%"),
        HealthCheck("Unique stick positions", "stick_unique",
                    len(stick_positions), 10, ">=", "count"),
        HealthCheck("Mean NONE prob", "none_mean",
                    none_mean, 0.95, "<=", ""),
        HealthCheck("NONE < 0.99", "none_below_99_pct",
                    none_below_99, 10.0, ">=", "%"),
        HealthCheck("Non-NONE top pred", "action_initiation_pct",
                    action_pct, 5.0, ">=", "%"),
        HealthCheck("Shoulder active", "shoulder_active_pct",
                    shoulder_pct, 1.0, ">=", "%"),
    ]


def print_report(frames: List[dict], checks: List[HealthCheck]) -> bool:
    """Print health report. Returns True if all checks pass."""
    n = len(frames)
    duration = n / 60.0
    passed = sum(1 for c in checks if c.passed)
    all_pass = passed == len(checks)

    print(f"Frames: {n}  Duration: {duration:.1f}s")
    print()
    print(f"{'Metric':<26s} {'Value':<14s} {'Threshold':<14s} {'Status'}")
    print("-" * 64)
    for c in checks:
        status = "PASS" if c.passed else "FAIL"
        print(f"  {c.name:<24s} {c.fmt_value():<14s} {c.fmt_threshold():<14s} {status}")

    print()
    print(f"Result: {passed}/{len(checks)} checks passed", end="")
    if all_pass:
        print(" — HEALTHY")
    else:
        print(" — UNHEALTHY")

    return all_pass


def main():
    parser = argparse.ArgumentParser(description="Analyze inference gameplay health")
    parser.add_argument("logfile", help="Log file path, or '-' for stdin")
    args = parser.parse_args()

    if args.logfile == "-":
        lines = sys.stdin.readlines()
    else:
        with open(args.logfile) as f:
            lines = f.readlines()

    frames = parse_log(lines)
    if not frames:
        print("No gameplay frames found in log.")
        sys.exit(1)

    checks = compute_health(frames)
    healthy = print_report(frames, checks)
    sys.exit(0 if healthy else 1)


if __name__ == "__main__":
    main()
