#!/usr/bin/env python3
"""
analyze_scaling.py -- Power law analysis for LR and batch size scaling experiments.

Usage:
    python analyze_scaling.py --phase 1 --log-dir logs/scaling   # LR sweep analysis
    python analyze_scaling.py --phase 2 --log-dir logs/scaling   # BS sweep analysis
    python analyze_scaling.py --phase 1 --log-dir logs/scaling --remote root@host:port  # fetch from remote

Reads final val losses from log files, combines with prior results,
fits power laws, and saves plots.
"""

import argparse
import re
import os
import subprocess
import json
import numpy as np
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not available, skipping plots")

MODEL_PARAMS = {
    "tiny":   5_357_703,
    "small":  15_217_543,
    "medium": 31_368_839,
    "base":   53_811_591,
}

PRIOR_RESULTS = {
    ("tiny",  5e-4): 0.114,
    ("tiny",  1e-3): 0.125,
    ("small", 5e-4): 0.104,
    ("small", 1e-3): 0.137,
    ("base",  3e-4): 0.113,
}


def parse_val_loss(log_path: str) -> float | None:
    """Extract final val total loss from a training log file."""
    val_pattern = re.compile(r"val total=([\d.]+)")
    last_val = None
    with open(log_path, "r") as f:
        for line in f:
            m = val_pattern.search(line)
            if m:
                last_val = float(m.group(1))
    return last_val


def parse_train_loss(log_path: str) -> float | None:
    """Extract final training total loss from a training log file."""
    train_pattern = re.compile(r"total=([\d.]+)")
    last_train = None
    with open(log_path, "r") as f:
        for line in f:
            m = train_pattern.search(line)
            if m:
                last_train = float(m.group(1))
    return last_train


def parse_log_filename(fname: str) -> dict | None:
    """Parse model, lr, bs from log filename like tiny-lr1e-4.log or tiny-lr1e-4-bs100.log"""
    m = re.match(r"(\w+)-lr([\de.\-]+?)(?:-bs(\d+))?\.log$", fname)
    if not m:
        return None
    model = m.group(1)
    lr = float(m.group(2))
    bs = int(m.group(3)) if m.group(3) else 200
    return {"model": model, "lr": lr, "bs": bs}


def fetch_remote_logs(remote: str, remote_dir: str, local_dir: str):
    """SCP log files from remote machine."""
    host_port = remote.split(":")
    host = host_port[0]
    port = host_port[1] if len(host_port) > 1 else "22"
    os.makedirs(local_dir, exist_ok=True)
    cmd = f"scp -P {port} {host}:{remote_dir}/*.log {local_dir}/"
    print(f"Fetching logs: {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def fit_power_law(x, y):
    """Fit y = a * x^b in log space. Returns (a, b, r_squared)."""
    log_x = np.log(x)
    log_y = np.log(y)
    coeffs = np.polyfit(log_x, log_y, 1)
    b, log_a = coeffs
    a = np.exp(log_a)
    y_pred = np.polyval(coeffs, log_x)
    ss_res = np.sum((log_y - y_pred) ** 2)
    ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
    r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return a, b, r_sq


def analyze_lr_sweep(log_dir: str, output_dir: str):
    """Phase 1: LR sweep analysis."""
    results = dict(PRIOR_RESULTS)

    for fname in sorted(os.listdir(log_dir)):
        if not fname.endswith(".log") or fname.startswith("auto"):
            continue
        info = parse_log_filename(fname)
        if info is None or info["bs"] != 200:
            continue
        val_loss = parse_val_loss(os.path.join(log_dir, fname))
        if val_loss is not None:
            results[(info["model"], info["lr"])] = val_loss

    print("\n=== LR Sweep Results ===")
    print(f"{'Model':>8s}  {'LR':>8s}  {'Val Loss':>10s}")
    print("-" * 30)
    for (model, lr), val in sorted(results.items(), key=lambda x: (x[0][0], x[0][1])):
        print(f"{model:>8s}  {lr:>8.0e}  {val:>10.4f}")

    models = sorted(MODEL_PARAMS.keys(), key=lambda m: MODEL_PARAMS[m])
    lr_opt = {}
    loss_opt = {}
    for model in models:
        model_results = {lr: val for (m, lr), val in results.items() if m == model}
        if not model_results:
            continue
        best_lr = min(model_results, key=model_results.get)
        lr_opt[model] = best_lr
        loss_opt[model] = model_results[best_lr]

    print("\n=== Optimal LR per Model ===")
    print(f"{'Model':>8s}  {'N':>12s}  {'LR_opt':>8s}  {'Val Loss':>10s}")
    print("-" * 44)
    for model in models:
        if model in lr_opt:
            print(f"{model:>8s}  {MODEL_PARAMS[model]:>12,}  {lr_opt[model]:>8.0e}  {loss_opt[model]:>10.4f}")

    if len(lr_opt) >= 3:
        N = np.array([MODEL_PARAMS[m] for m in models if m in lr_opt])
        LR = np.array([lr_opt[m] for m in models if m in lr_opt])
        L = np.array([loss_opt[m] for m in models if m in lr_opt])

        a_lr, b_lr, r2_lr = fit_power_law(N, LR)
        a_l, b_l, r2_l = fit_power_law(N, L)

        print(f"\n=== Power Law Fits ===")
        print(f"LR_opt = {a_lr:.4e} * N^({b_lr:.4f})   R²={r2_lr:.4f}")
        print(f"L_opt  = {a_l:.4e} * N^({b_l:.4f})   R²={r2_l:.4f}")

        summary = {
            "lr_sweep": {str(k): v for k, v in results.items()},
            "lr_opt": {m: float(lr_opt[m]) for m in lr_opt},
            "loss_opt": {m: float(loss_opt[m]) for m in loss_opt},
            "power_law_lr": {"a": float(a_lr), "b": float(b_lr), "r2": float(r2_lr)},
            "power_law_loss": {"a": float(a_l), "b": float(b_l), "r2": float(r2_l)},
        }
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "phase1_results.json"), "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nSaved results to {output_dir}/phase1_results.json")

        if HAS_MPL:
            _plot_lr_sweep(results, models, output_dir)
            _plot_power_laws(N, LR, L, models, lr_opt, a_lr, b_lr, a_l, b_l, output_dir)


def _plot_lr_sweep(results, models, output_dir):
    """Plot val_loss vs LR U-curves per model."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"tiny": "#1f77b4", "small": "#ff7f0e", "medium": "#2ca02c", "base": "#d62728"}
    for model in models:
        model_results = {lr: val for (m, lr), val in results.items() if m == model}
        if not model_results:
            continue
        lrs = sorted(model_results.keys())
        vals = [model_results[lr] for lr in lrs]
        n = MODEL_PARAMS[model]
        ax.plot(lrs, vals, "o-", color=colors.get(model, "gray"),
                label=f"{model} ({n/1e6:.1f}M)", markersize=8, linewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel("Learning Rate", fontsize=13)
    ax.set_ylabel("Validation Loss", fontsize=13)
    ax.set_title("LR Sweep: Val Loss vs Learning Rate", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "lr_ucurves.png"), dpi=150)
    print(f"Saved {output_dir}/lr_ucurves.png")
    plt.close(fig)


def _plot_power_laws(N, LR, L, models, lr_opt, a_lr, b_lr, a_l, b_l, output_dir):
    """Plot power law fits for LR_opt and L_opt vs N."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    N_fit = np.logspace(np.log10(N.min() * 0.5), np.log10(N.max() * 2), 100)

    ax1.scatter(N, LR, s=100, zorder=5, color="#1f77b4")
    ax1.plot(N_fit, a_lr * N_fit ** b_lr, "--", color="#1f77b4", alpha=0.7,
             label=f"LR = {a_lr:.2e} * N^({b_lr:.3f})")
    for i, m in enumerate([m for m in models if m in lr_opt]):
        ax1.annotate(m, (MODEL_PARAMS[m], lr_opt[m]),
                     textcoords="offset points", xytext=(10, 5), fontsize=10)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Model Parameters (N)", fontsize=12)
    ax1.set_ylabel("Optimal LR", fontsize=12)
    ax1.set_title("LR_opt vs Model Size", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.scatter(N, L, s=100, zorder=5, color="#d62728")
    ax2.plot(N_fit, a_l * N_fit ** b_l, "--", color="#d62728", alpha=0.7,
             label=f"L = {a_l:.2e} * N^({b_l:.3f})")
    loss_opt_dict = dict(zip([m for m in models if m in lr_opt],
                             [lr_opt[m] for m in models if m in lr_opt]))
    for i, m in enumerate([m for m in models if m in lr_opt]):
        ax2.annotate(m, (MODEL_PARAMS[m], L[i]),
                     textcoords="offset points", xytext=(10, 5), fontsize=10)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Model Parameters (N)", fontsize=12)
    ax2.set_ylabel("Optimal Val Loss", fontsize=12)
    ax2.set_title("L_opt vs Model Size", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "power_laws.png"), dpi=150)
    print(f"Saved {output_dir}/power_laws.png")
    plt.close(fig)


def analyze_bs_sweep(log_dir: str, output_dir: str):
    """Phase 2: Batch size sweep analysis."""
    phase1_path = os.path.join(output_dir, "phase1_results.json")
    if os.path.exists(phase1_path):
        with open(phase1_path) as f:
            phase1 = json.load(f)
        lr_opt = phase1.get("lr_opt", {})
        loss_opt_at_bs200 = phase1.get("loss_opt", {})
    else:
        print("WARNING: phase1_results.json not found, using defaults")
        lr_opt = {}
        loss_opt_at_bs200 = {}

    results = {}
    for model, loss in loss_opt_at_bs200.items():
        results[(model, 200)] = loss

    for fname in sorted(os.listdir(log_dir)):
        if not fname.endswith(".log") or fname.startswith("auto"):
            continue
        info = parse_log_filename(fname)
        if info is None or info["bs"] == 200:
            continue
        val_loss = parse_val_loss(os.path.join(log_dir, fname))
        if val_loss is not None:
            results[(info["model"], info["bs"])] = val_loss

    print("\n=== BS Sweep Results ===")
    print(f"{'Model':>8s}  {'BS':>6s}  {'Val Loss':>10s}")
    print("-" * 28)
    for (model, bs), val in sorted(results.items()):
        print(f"{model:>8s}  {bs:>6d}  {val:>10.4f}")

    models = sorted(MODEL_PARAMS.keys(), key=lambda m: MODEL_PARAMS[m])
    for model in models:
        model_results = {bs: val for (m, bs), val in results.items() if m == model}
        if len(model_results) >= 2:
            best_bs = min(model_results, key=model_results.get)
            print(f"\n  {model}: best BS={best_bs} (val={model_results[best_bs]:.4f})")

    summary = {"bs_sweep": {str(k): v for k, v in results.items()}}
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "phase2_results.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved results to {output_dir}/phase2_results.json")

    if HAS_MPL:
        _plot_bs_sweep(results, models, output_dir)


def _plot_bs_sweep(results, models, output_dir):
    """Plot val_loss vs batch size per model."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"tiny": "#1f77b4", "small": "#ff7f0e", "medium": "#2ca02c", "base": "#d62728"}
    for model in models:
        model_results = {bs: val for (m, bs), val in results.items() if m == model}
        if not model_results:
            continue
        bss = sorted(model_results.keys())
        vals = [model_results[bs] for bs in bss]
        n = MODEL_PARAMS[model]
        ax.plot(bss, vals, "o-", color=colors.get(model, "gray"),
                label=f"{model} ({n/1e6:.1f}M)", markersize=8, linewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel("Batch Size", fontsize=13)
    ax.set_ylabel("Validation Loss", fontsize=13)
    ax.set_title("BS Sweep: Val Loss vs Batch Size", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "bs_curves.png"), dpi=150)
    print(f"Saved {output_dir}/bs_curves.png")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2])
    parser.add_argument("--log-dir", type=str, default="logs/scaling")
    parser.add_argument("--output-dir", type=str, default="results/scaling")
    parser.add_argument("--remote", type=str, default=None,
                        help="Fetch logs from remote, e.g. root@host:port")
    args = parser.parse_args()

    if args.remote:
        fetch_remote_logs(args.remote, f"/root/MIMIC/{args.log_dir}", args.log_dir)

    if args.phase == 1:
        analyze_lr_sweep(args.log_dir, args.output_dir)
    elif args.phase == 2:
        analyze_bs_sweep(args.log_dir, args.output_dir)
