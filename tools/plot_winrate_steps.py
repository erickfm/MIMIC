#!/usr/bin/env python3
"""Plot CPU-9 win rate vs training steps, one point per character."""
import json, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import numpy as np

D = json.load(open("/tmp/claude-1000/-home-erick-projects-MIMIC/23e007ee-2554-47fc-aafa-fe2556349158/scratchpad/plotdata.json"))

steps = np.array([d["step"] for d in D], float)
win   = np.array([d["win"]*100 for d in D])
lo    = np.array([d["lo"]*100 for d in D])
hi    = np.array([d["hi"]*100 for d in D])
games = np.array([d["games"] for d in D], float)
names = [d["char"] for d in D]

fig, ax = plt.subplots(figsize=(12, 7.5), dpi=130)
fig.patch.set_facecolor("white"); ax.set_facecolor("white")

# color = training-set size (sequential, magnitude)
norm = colors.LogNorm(vmin=games.min(), vmax=games.max())
cmap = cm.get_cmap("viridis")
c = cmap(norm(games))

# 95% CI bars (recessive)
ax.errorbar(steps, win, yerr=[win-lo, hi-win], fmt="none",
            ecolor="#b8b8b8", elinewidth=1.2, capsize=3, zorder=1)
ax.scatter(steps, win, c=c, s=130, edgecolor="#333", linewidth=0.8, zorder=3)

# direct labels, nudged (dx,dy) to reduce collisions
off = {"mario":(7,-10),"roy":(7,8),"ness":(9,-4),"bowser":(7,9),
       "mewtwo":(9,-3),"ylink":(-6,-13),"gameandwatch":(9,-11),"link":(8,7),
       "yoshi":(9,-3),"luigi":(8,-11),"doc":(7,8),"dk":(8,-11),
       "ice_climbers":(9,7),"pikachu":(9,-11),"ganondorf":(8,7),
       "cptfalcon":(-4,-14),"puff":(6,9),"sheik":(8,7),"marth":(7,7),
       "falco":(7,7),"peach":(8,7),"samus":(8,7),"fox":(-6,-14)}
for x,y,n in zip(steps, win, names):
    dx,dyy = off.get(n,(7,7))
    ha = "right" if dx < 0 else "left"
    ax.annotate(n, (x,y), textcoords="offset points",
                xytext=(dx, dyy), fontsize=8, color="#222", ha=ha)

ax.set_xscale("log")
ax.set_xlim(2300, 300000)
ax.set_ylim(-6, 106)
ax.set_xticks([2500,4000,6000,10000,20000,40000,100000,224000])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(
    lambda v,_: f"{v/1000:g}k"))
ax.set_xlabel("Training steps (log scale)", fontsize=12)
ax.set_ylabel("CPU-9 win rate  (%, n=20, 95% CI)", fontsize=12)
ax.set_title("MIMIC per-character strength vs training length\n"
             "color = master training games (the real driver); x = steps trained",
             fontsize=13)
ax.grid(True, which="both", color="#eee", linewidth=0.8, zorder=0)
for s in ("top","right"): ax.spines[s].set_visible(False)

sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
cb = fig.colorbar(sm, ax=ax, pad=0.01)
cb.set_label("training games", fontsize=11)
cb.set_ticks([500,1000,2000,5000,10000,28000,97000])
cb.ax.set_yticklabels(["500","1k","2k","5k","10k","28k","97k"])

# Spearman of steps vs win, and games vs win, as a caption
def spearman(a,b):
    ra=np.argsort(np.argsort(a)); rb=np.argsort(np.argsort(b))
    return np.corrcoef(ra,rb)[0,1]
rs_steps = spearman(steps, win); rs_games = spearman(games, win)
fig.text(0.5, 0.005,
    f"Spearman ρ(steps, win) = {rs_steps:.2f}   ρ(games, win) = {rs_games:.2f}"
    "   — steps and games are near-collinear (budget scaled with data), so this is one confounded axis, not two",
    ha="center", fontsize=9, color="#555")

fig.tight_layout(rect=(0,0.03,1,1))
out = "eval_results/winrate_vs_steps.png"
fig.savefig(out, facecolor="white")
print("saved", out, "| rho steps", round(rs_steps,2), "games", round(rs_games,2))
