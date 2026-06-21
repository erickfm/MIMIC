#!/usr/bin/env python3
"""Plot validation loss (x) vs head-to-head win rate (y) for the master-Fox
thread. Each point is a checkpoint's win rate against the *same* common
opponent (the original rank-master Fox), so the y-axis is apples-to-apples.
The self-play control (original vs an identical copy of itself) anchors the
original point at ~50%.

X-axis is **fixed seed-42 `train.py --eval-only` val** for every point (NOT the
in-training "best val", which the 2026-06-19 study found to be selection-biased
— see docs/research-notes-2026-06-19.md). Re-scoring on a common fixed subset is
what makes the x-axis comparable across runs.

The orange star is the less-reg + tail-SWA *recipe* (32k-step budget): it reaches
essentially the same val / win rate as the 442k-step long run at ~14× fewer steps.

Usage: python3 tools/plot_val_vs_winrate.py
Writes reports/val_vs_winrate.png
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (label, seed42_val, report.json, steps_note, is_recipe)
POINTS = [
    ("original (self-play control)", 0.7438, "reports/selfplay_control.json",      "32k steps",            False),
    ("warm-restart",                 0.7270, "reports/warmrestart_vs_original.json","~32k steps",           False),
    ("long run",                     0.7174, "reports/long_vs_original.json",       "442k steps · shipped", False),
    ("less-reg + SWA (recipe)",      0.7183, "reports/lessreg_swa_vs_original.json","32k steps · free",     True),
]

rows = []
for label, val, path, steps, is_recipe in POINTS:
    d = json.load(open(path))
    wr = 100.0 * d["a_win_rate"]
    rows.append((label, val, wr, steps, is_recipe))
    print(f"{label:30s} val={val:.4f}  WR={wr:5.1f}%  ({d['a_wins']}-{d['b_wins']})  [{steps}]")

fig, ax = plt.subplots(figsize=(7.8, 5.4))
ax.axhline(50, color="0.7", lw=1, ls="--", zorder=1)
ax.text(0.7436, 50.8, "50% (even)", color="0.5", fontsize=8, va="bottom")

# trend line over the continued-training thread only (the 3 blue points)
trend = sorted([r for r in rows if not r[4]], key=lambda r: r[1])
ax.plot([r[1] for r in trend], [r[2] for r in trend],
        color="0.75", lw=1.2, zorder=2)

for label, val, wr, steps, is_recipe in rows:
    if is_recipe:
        ax.scatter([val], [wr], s=240, color="#e8820c", marker="*",
                   zorder=4, edgecolor="white", linewidth=1.0)
        ax.annotate(f"{label}\nval {val:.4f} · {wr:.0f}% · {steps}",
                    (val, wr), textcoords="offset points", xytext=(-10, 12),
                    fontsize=8.5, ha="right", va="bottom", color="#b3650a")
    else:
        ax.scatter([val], [wr], s=90, color="#2a6fdb",
                   zorder=3, edgecolor="white", linewidth=1.2)
        ax.annotate(f"{label}\nval {val:.4f} · {wr:.0f}% · {steps}",
                    (val, wr), textcoords="offset points", xytext=(10, -6),
                    fontsize=8.5, ha="left", va="top")

ax.invert_xaxis()  # lower (better) loss to the right → trend reads bottom-left→top-right
ax.set_xlabel("validation loss  —  fixed seed-42 --eval-only  (lower = better →)")
ax.set_ylabel("head-to-head win rate vs original master (%)")
ax.set_title("Master-Fox: val loss vs in-game win rate\n"
             "(same data distribution · common opponent · seed-42 val · N=17–21 matches)")
ax.set_ylim(40, 85)
ax.grid(True, alpha=0.25)
fig.tight_layout()
out = "reports/val_vs_winrate.png"
fig.savefig(out, dpi=150)
print(f"wrote {out}")
