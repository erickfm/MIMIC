# Bradley-Terry ratings for the 2026-11-11 RL round-robin (docs/research-notes-2026-11-09.md). Usage: python3 tools/rl_ratings.py rr|all
import json, glob, math, sys
import numpy as np
P="fox-S-bc2-hf51k-s1"
names={"fox-M-bc2-mlr01-s1.rl2-s1":"fox-master (M 1e-5 u100)",
 f"{P}.rl2-fast-s1":"S 1e-5 s1 u100", f"{P}.rl2-fast-s1-u200":"S 1e-5 s1 u200", f"{P}.rl2-fast-s1-u400":"S 1e-5 s1 u400",
 f"{P}.rl3-s2-u200":"S 1e-5 s2 u200", f"{P}.rl3-s2-u400":"S 1e-5 s2 u400",
 f"{P}.rl3-lr2e5-s1":"S 2e-5 s1 u100", f"{P}.rl3-lr2e5-s1-u200":"S 2e-5 s1 u200", f"{P}.rl3-lr2e5-s1-u400":"S 2e-5 s1 u400",
 f"{P}.rl3-lr2e5-s2-u200":"S 2e-5 s2 u200", f"{P}.rl3-lr2e5-s2-u400":"S 2e-5 s2 u400",
 f"{P}.rl3-lr4e5-u200-s1":"S 4e-5 s1 u200",
 "fox-M-bc2-mlr01-s1.rl3-lr2e5-u200-s1":"M 2e-5 u200", "fox-M-bc2-mlr01-s1.rl3-lr2e5-u200-slow-s1":"M 2e-5 u200"}
mode=sys.argv[1]  # rr or all
res=[]
for f in glob.glob("runs/*/eval/*.result.json"):
    r=json.load(open(f))
    if r["a"] in names and r["b"] in names:
        if mode=="rr" and not (r["seed"]==43 and r["games"]==96): continue
        res.append((r["a"],r["b"],r["a_wins"],r["b_wins"]))
ids=sorted({x for r in res for x in r[:2]}); k={n:i for i,n in enumerate(ids)}; n=len(ids)
W=np.zeros((n,n))
for a,b,wa,wb in res: W[k[a],k[b]]+=wa; W[k[b],k[a]]+=wb
# BT via MM, with tiny prior
p=np.ones(n)
for _ in range(5000):
    N=W+W.T
    p=(W.sum(1)+0.5)/np.array([sum(N[i,j]/(p[i]+p[j]) for j in range(n) if j!=i)+1/(p[i]+1) for i in range(n)])
    p/=np.exp(np.mean(np.log(p)))
s=np.log(p); fm=k["fox-M-bc2-mlr01-s1.rl2-s1"]; s-=s[fm]
# SE from Fisher information
N=W+W.T; I=np.zeros((n,n))
for i in range(n):
    for j in range(n):
        if i!=j:
            q=p[i]*p[j]/(p[i]+p[j])**2*N[i,j]; I[i,j]-=q; I[i,i]+=q
idx=[i for i in range(n) if i!=fm]; C=np.linalg.inv(I[np.ix_(idx,idx)]); se=np.zeros(n); se[idx]=np.sqrt(np.diag(C))
print(f"{mode}: {len(res)} matches, {int(W.sum())} games")
for i in np.argsort(-s):
    print(f"  {names[ids[i]]:28s} {s[i]:+.2f} ± {se[i]:.2f}   (win% vs fox-master {100/(1+math.exp(-s[i])):.0f})")
chi=0; worst=[]
for a,b,wa,wb in res:
    m=wa+wb; e=m*p[k[a]]/(p[k[a]]+p[k[b]]); z=(wa-e)/math.sqrt(e*(1-e/m)); chi+=z*z; worst.append((abs(z),names[a],names[b],wa,wb,round(e)))
print(f"fit: chi2 {chi:.1f} on {len(res)-(n-1)} df"); [print("  ",w) for w in sorted(worst)[-3:]]
