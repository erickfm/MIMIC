#!/usr/bin/env python3
"""Plot validation loss (x) vs head-to-head win rate (y) for the master-Fox
continued-training thread. Each point is a checkpoint's win rate against the
*same* common opponent (the original rank-master Fox, val 0.7334), so the
y-axis is apples-to-apples. The self-play control (original vs an identical
copy of itself) anchors the 0.7334 point at ~50%.

Usage: python3 tools/plot_val_vs_winrate.py
Writes reports/val_vs_winrate.png
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (label, val_loss, report.json, note)
POINTS = [
    ("original (self-play control)", 0.7334, "reports/selfplay_control.json", "baseline"),
    ("warm-restart",                 0.7176, "reports/warmrestart_vs_original.json", "keep Adam, reset LR"),
    ("long run (~442k steps)",       0.7130, "reports/long_vs_original.json", "shipped"),
]

xs, ys, labels, notes = [], [], [], []
for label, val, path, note in POINTS:
    d = json.load(open(path))
    wr = 100.0 * d["a_win_rate"]
    xs.append(val); ys.append(wr); labels.append(label); notes.append(note)
    print(f"{label:32s} val={val:.4f}  WR={wr:5.1f}%  ({d['a_wins']}-{d['b_wins']})")

fig, ax = plt.subplots(figsize=(7.5, 5.2))
ax.axhline(50, color="0.7", lw=1, ls="--", zorder=1)
ax.text(0.7332, 50.8, "50% (even)", color="0.5", fontsize=8, va="bottom")

# connect in val-loss order to show the monotone trend
order = sorted(range(len(xs)), key=lambda i: xs[i])
ax.plot([xs[i] for i in order], [ys[i] for i in order],
        color="0.75", lw=1.2, zorder=2)

ax.scatter(xs, ys, s=90, color="#2a6fdb", zorder=3, edgecolor="white", linewidth=1.2)
for x, y, lab, note in zip(xs, ys, labels, notes):
    ax.annotate(f"{lab}\nval {x:.4f} · {y:.0f}%",
                (x, y), textcoords="offset points", xytext=(10, -6),
                fontsize=8.5, ha="left", va="top")

ax.invert_xaxis()  # lower (better) loss to the right → trend reads bottom-left→top-right
ax.set_xlabel("validation loss  (lower = better →)")
ax.set_ylabel("head-to-head win rate vs original master (%)")
ax.set_title("Master-Fox: val loss vs in-game win rate\n(same data distribution, 21 matches each, common opponent)")
ax.set_ylim(40, 85)
ax.grid(True, alpha=0.25)
fig.tight_layout()
out = "reports/val_vs_winrate.png"
fig.savefig(out, dpi=150)
print(f"wrote {out}")
