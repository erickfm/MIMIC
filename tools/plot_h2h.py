#!/usr/bin/env python3
"""Scatter: h2h Bradley-Terry Elo vs CPU-9 win rate, one point per character."""
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D=json.load(open("/tmp/claude-1000/-home-erick-projects-MIMIC/23e007ee-2554-47fc-aafa-fe2556349158/scratchpad/h2h_plotdata.json"))
D=[d for d in D if d["cpu9"] is not None]
cpu=np.array([d["cpu9"] for d in D]); elo=np.array([d["elo"] for d in D])
g=np.array([d["games"] for d in D]); names=[d["char"] for d in D]

fig,ax=plt.subplots(figsize=(12,7.5),dpi=130); fig.patch.set_facecolor("white"); ax.set_facecolor("white")
# size by games (confidence)
s=60+g*3
ax.scatter(cpu,elo,s=s,c="#3b6fb0",edgecolor="#222",linewidth=0.7,alpha=0.85,zorder=3)
off={"bowser":(6,6),"fox":(6,-13),"cptfalcon":(6,6),"falco":(6,-12),"marth":(-6,-13),
     "sheik":(6,6),"samus":(6,6),"ganondorf":(6,-12),"puff":(6,6),"dk":(6,-12),
     "ness":(6,6),"ylink":(-6,7),"mewtwo":(6,6),"link":(6,7),"mario":(6,-12),
     "gameandwatch":(6,-12),"roy":(6,-12),"yoshi":(6,6),"peach":(6,7),"doc":(6,-12),
     "ice_climbers":(6,6),"pikachu":(6,-12),"luigi":(6,6)}
for x,y,n in zip(cpu,elo,names):
    dx,dy=off.get(n,(6,6)); ax.annotate(n,(x,y),textcoords="offset points",xytext=(dx,dy),
        fontsize=8,color="#222",ha="right" if dx<0 else "left")
# trend line + rho
b,a=np.polyfit(cpu,elo,1); xs=np.array([cpu.min()-3,cpu.max()+3])
ax.plot(xs,a+b*xs,"--",color="#c0392b",linewidth=1.3,zorder=2,alpha=0.8)
def spearman(u,v):
    ru=np.argsort(np.argsort(u)); rv=np.argsort(np.argsort(v)); return np.corrcoef(ru,rv)[0,1]
rho=spearman(cpu,elo)
ax.set_xlabel("CPU-9 win rate (%)",fontsize=12)
ax.set_ylabel("Head-to-head Bradley-Terry Elo",fontsize=12)
ax.set_title(f"Head-to-head Elo vs CPU-9 win rate   "
             f"(Spearman ρ = {rho:+.2f}, marker size = h2h games)",fontsize=12)
ax.grid(True,color="#eee",linewidth=0.8,zorder=0)
for sp in ("top","right"): ax.spines[sp].set_visible(False)
fig.tight_layout()
out="eval_results/h2h_vs_cpu9.png"; fig.savefig(out,facecolor="white"); print("saved",out,"rho",round(rho,2))
