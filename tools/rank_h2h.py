"""Fit a Bradley-Terry strength model from h2h results and compare to CPU-9.

Reads eval_results/h2h/<A>__vs__<B>.json (A=--character, a_wins for A) and the
CPU-9 pooled results (eval_results/<char>.json[+_r2]). Fits BT by the standard
MM iteration, converts to an Elo-like scale, and reports the ranking, each
char's aggregate h2h win rate + games, the CPU-9 win rate, and the rank shift.
"""
import json, glob, math, collections
from pathlib import Path

H2H = "eval_results/h2h"
E2C = {"MARTH":"marth","FALCO":"falco","CPTFALCON":"cptfalcon","SAMUS":"samus",
 "DOC":"doc","JIGGLYPUFF":"puff","DK":"dk","PEACH":"peach","GANONDORF":"ganondorf",
 "POPO":"ice_climbers","YOSHI":"yoshi","BOWSER":"bowser","ROY":"roy","PIKACHU":"pikachu",
 "MARIO":"mario","LUIGI":"luigi","GAMEANDWATCH":"gameandwatch","LINK":"link",
 "MEWTWO":"mewtwo","YLINK":"ylink","NESS":"ness","SHEIK":"sheik","FOX":"fox"}

def load_cpu9():
    wr={}
    for f in glob.glob("eval_results/*.json"):
        d=json.load(open(f)); c=E2C.get(d.get("config",{}).get("character"))
        if not c: continue
        wr.setdefault(c,[0,0]); wr[c][0]+=d["a_wins"]; wr[c][1]+=d["n_matches"]
    # fox-master eval saved as fox-master.json with character FOX -> already 'fox'
    return {c:(w/n if n else None, n) for c,(w,n) in wr.items()}

def load_h2h():
    wins=collections.defaultdict(float); games=collections.defaultdict(float)
    pair_games=collections.defaultdict(lambda:collections.defaultdict(float))
    pair_wins=collections.defaultdict(lambda:collections.defaultdict(float))
    raw={}
    for f in glob.glob(f"{H2H}/**/*__vs__*.json", recursive=True):
        d=json.load(open(f)); a,b=Path(f).stem.split("__vs__")
        aw,bw,dr=d["a_wins"],d["b_wins"],d.get("draws",0)
        aw+=0.5*dr; bw+=0.5*dr; n=aw+bw
        if n==0: continue
        wins[a]+=aw; wins[b]+=bw; games[a]+=n; games[b]+=n
        pair_games[a][b]+=n; pair_games[b][a]+=n
        pair_wins[a][b]+=aw; pair_wins[b][a]+=bw
        raw[(a,b)]=(aw,bw)
    return wins,games,pair_games,raw

def bradley_terry(chars, wins, pair_games, iters=2000, prior=1.0):
    """Regularized BT (MM iteration). A phantom opponent of strength 1 with
    `prior` virtual games (split 0.5/0.5) keeps undefeated/winless chars finite
    and the graph connected — needed for partial data and extreme results."""
    p={c:1.0 for c in chars}
    for _ in range(iters):
        np_={}
        for i in chars:
            denom=sum(pair_games[i][j]/(p[i]+p[j]) for j in chars if pair_games[i][j]>0)
            denom+=prior/(p[i]+1.0)                       # vs phantom (strength 1)
            num=wins[i]+prior*0.5
            np_[i]=max(num/denom, 1e-9)
        g=math.exp(sum(math.log(v) for v in np_.values())/len(np_))
        p={c:np_[c]/g for c in chars}
    return p

def main():
    cpu9=load_cpu9()
    wins,games,pair_games,raw=load_h2h()
    chars=sorted(games.keys())
    if not chars:
        print("no h2h results yet"); return
    p=bradley_terry(chars, wins, pair_games)
    elo={c:400*math.log10(p[c]) for c in chars}
    off=1500-(sum(elo.values())/len(elo)); elo={c:elo[c]+off for c in chars}
    order=sorted(chars, key=lambda c:-elo[c])
    cpu_order=sorted([c for c in chars if cpu9.get(c,(None,))[0] is not None],
                     key=lambda c:-cpu9[c][0])
    cpu_rank={c:i for i,c in enumerate(cpu_order)}
    print(f"{'rank':>4s} {'char':13s}{'Elo':>6s}{'h2h_win%':>9s}{'games':>6s}"
          f"{'cpu9%':>7s}{'cpu_rk':>7s}{'Δrank':>7s}")
    for i,c in enumerate(order):
        w=100*wins[c]/games[c]; c9=cpu9.get(c,(None,0))[0]
        c9s=f"{100*c9:.0f}" if c9 is not None else "n/a"
        cr=cpu_rank.get(c); dr=(cr-i) if cr is not None else None
        drs=f"{dr:+d}" if dr is not None else "n/a"
        print(f"{i+1:>4d} {c:13s}{elo[c]:6.0f}{w:8.0f}%{int(games[c]):6d}{c9s:>7s}"
              f"{(cr+1) if cr is not None else 0:>7d}{drs:>7s}")
    # rank correlation
    common=[c for c in order if cpu_rank.get(c) is not None]
    if len(common)>3:
        h=[order.index(c) for c in common]; cc=[cpu_rank[c] for c in common]
        import statistics
        def spearman(a,b):
            ra=[sorted(a).index(x) for x in a]; rb=[sorted(b).index(x) for x in b]
            n=len(a); m=statistics.mean;
            num=sum((ra[i]-m(ra))*(rb[i]-m(rb)) for i in range(n))
            den=(sum((x-m(ra))**2 for x in ra)*sum((x-m(rb))**2 for x in rb))**.5
            return num/den if den else 0
        print(f"\nSpearman rank corr (h2h vs CPU-9): {spearman(h,cc):.2f}  (n={len(common)})")
    print(f"\ntotal pairings: {len(raw)}, total games: {int(sum(games.values())/2)}")

if __name__=="__main__":
    main()

def dump(path):
    """Emit per-char {elo, h2h_win, games, cpu9} for plotting."""
    import json as _j
    cpu9=load_cpu9(); wins,games,pair_games,raw=load_h2h()
    chars=sorted(games.keys())
    p=bradley_terry(chars,wins,pair_games)
    elo={c:400*math.log10(p[c]) for c in chars}
    off=1500-sum(elo.values())/len(elo); 
    out=[{"char":c,"elo":round(elo[c]+off,1),"h2h_win":round(100*wins[c]/games[c],1),
          "games":int(games[c]),"cpu9":round(100*cpu9[c][0],1) if cpu9.get(c,(None,))[0] is not None else None}
         for c in chars]
    _j.dump(out,open(path,"w"),indent=1)
