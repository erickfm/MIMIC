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
    """Parse MAIN/C/L/BTN/top3 lines into frame dicts.
    Supports both log formats:
      run_hal_model.py: MAIN=(x,y) C=(x,y) L=v BTN=[...] top3=[...] S=n(p%) O=n(p%)
      inference.py:     MAIN=(x,y) C=d L=v R=v BTN=[...] top3=[...] S=n(p%) O=n(p%)
    """
    pattern = re.compile(
        r"MAIN=\(([0-9.]+),([0-9.]+)\) "
        r"C=(?:\(([0-9.]+),([0-9.]+)\)|(\d+)) "
        r"L=([0-9.]+)"
        r"(?: R=[0-9.]+)? "
        r"BTN=\[([^\]]*)\] +"
        r"top3=\[(.+?)\]"
    )
    gs_pattern = re.compile(r"S=(\d+)\((\d+)%\) O=(\d+)\((\d+)%\)")
    frames = []
    for line in lines:
        m = pattern.search(line)
        if not m:
            continue
        # C-stick: either (x,y) format or integer direction
        if m.group(3) is not None:
            cx, cy = float(m.group(3)), float(m.group(4))
        else:
            cx, cy = 0.5, 0.5  # direction index — treat as neutral for health check
        l_val = float(m.group(6))
        btn_str = m.group(7).strip()
        btns = [b.strip().strip("'") for b in btn_str.split(",") if b.strip()] if btn_str else []

        probs = {}
        for part in m.group(8).split():
            if "=" in part:
                name, val = part.split("=")
                probs[name] = float(val)

        # Parse gamestate (stocks/percent) if present
        gs_m = gs_pattern.search(line)
        self_stock = int(gs_m.group(1)) if gs_m else None
        self_pct = int(gs_m.group(2)) if gs_m else None
        opp_stock = int(gs_m.group(3)) if gs_m else None
        opp_pct = int(gs_m.group(4)) if gs_m else None

        frames.append({
            "mx": float(m.group(1)),
            "my": float(m.group(2)),
            "cx": cx,
            "cy": cy,
            "l": l_val,
            "btns": btns,
            "probs": probs,
            "none_prob": probs.get("NONE", 0.0),
            "self_stock": self_stock,
            "self_pct": self_pct,
            "opp_stock": opp_stock,
            "opp_pct": opp_pct,
        })
    return frames


def compute_health(frames: List[dict]) -> List[HealthCheck]:
    """Compute gameplay health metrics focused on distinguishing intentional play from mashing."""
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

    # ── Damage / stock metrics (requires gamestate in log) ──
    has_gs = frames[0]["self_stock"] is not None
    dmg_dealt = 0.0
    dmg_taken = 0.0
    stocks_taken = 0
    stocks_lost = 0
    if has_gs:
        for i in range(1, n):
            prev, cur = frames[i - 1], frames[i]
            # Damage dealt: opponent percent increased (same stock)
            if cur["opp_stock"] == prev["opp_stock"] and cur["opp_pct"] > prev["opp_pct"]:
                dmg_dealt += cur["opp_pct"] - prev["opp_pct"]
            # Damage taken: self percent increased (same stock)
            if cur["self_stock"] == prev["self_stock"] and cur["self_pct"] > prev["self_pct"]:
                dmg_taken += cur["self_pct"] - prev["self_pct"]
            # Stocks taken: opponent stock decreased
            if cur["opp_stock"] < prev["opp_stock"]:
                stocks_taken += prev["opp_stock"] - cur["opp_stock"]
            # Stocks lost: self stock decreased
            if cur["self_stock"] < prev["self_stock"]:
                stocks_lost += prev["self_stock"] - cur["self_stock"]

    checks = [
        # Intentionality metrics (distinguishes purposeful play from mashing)
        HealthCheck("Non-NONE top pred", "action_initiation_pct",
                    action_pct, 5.0, ">=", "%"),
        HealthCheck("Stick at neutral", "stick_neutral_pct",
                    stick_neutral_pct, 15.0, ">=", "%"),
        HealthCheck("Button press rate", "btn_press_rate",
                    btn_initiations / duration_sec, 0.5, ">=", "/sec"),
        HealthCheck("Mean NONE prob", "none_mean",
                    none_mean, 0.95, "<=", ""),
    ]

    if has_gs:
        checks.append(
            HealthCheck("Damage dealt", "dmg_dealt",
                        dmg_dealt, 50.0, ">=", "raw"))

    return checks


def print_report(frames: List[dict], checks: List[HealthCheck]) -> bool:
    """Print health report. Returns True if all checks pass."""
    n = len(frames)
    duration = n / 60.0
    passed = sum(1 for c in checks if c.passed)
    all_pass = passed == len(checks)
    has_gs = frames[0]["self_stock"] is not None

    print(f"Frames: {n}  Duration: {duration:.1f}s")
    print()

    # Checks table
    print(f"{'Metric':<26s} {'Value':<14s} {'Threshold':<14s} {'Status'}")
    print("-" * 64)
    for c in checks:
        status = "PASS" if c.passed else "FAIL"
        print(f"  {c.name:<24s} {c.fmt_value():<14s} {c.fmt_threshold():<14s} {status}")

    # Gamestate summary (always shown if available, not pass/fail)
    if has_gs:
        # Recompute here for display
        dmg_dealt = dmg_taken = 0.0
        stocks_taken = stocks_lost = 0
        for i in range(1, n):
            prev, cur = frames[i - 1], frames[i]
            if cur["opp_stock"] == prev["opp_stock"] and cur["opp_pct"] > prev["opp_pct"]:
                dmg_dealt += cur["opp_pct"] - prev["opp_pct"]
            if cur["self_stock"] == prev["self_stock"] and cur["self_pct"] > prev["self_pct"]:
                dmg_taken += cur["self_pct"] - prev["self_pct"]
            if cur["opp_stock"] < prev["opp_stock"]:
                stocks_taken += prev["opp_stock"] - cur["opp_stock"]
            if cur["self_stock"] < prev["self_stock"]:
                stocks_lost += prev["self_stock"] - cur["self_stock"]

        final_self_stock = frames[-1]["self_stock"]
        final_opp_stock = frames[-1]["opp_stock"]
        print()
        print("Game result:")
        print(f"  Stocks:  self={final_self_stock}  opp={final_opp_stock}  (took {stocks_taken}, lost {stocks_lost})")
        print(f"  Damage:  dealt={dmg_dealt:.0f}%  taken={dmg_taken:.0f}%  ratio={dmg_dealt/max(dmg_taken,1):.2f}")

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
