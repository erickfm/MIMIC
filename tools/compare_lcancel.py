"""Compare the MODEL's L-cancel rate distribution to the TRAINING corpus it
learned from, per move, with Wilson 95% CIs. Uses the human-type filter so
model-vs-CPU runs isolate the bot; the all-human corpus is unaffected."""
import sys, glob, math, collections
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tools.lcancel_analysis import scan  # reuse the validated scanner (human-Fox only)

MOVES = ["NAIR", "FAIR", "BAIR", "UAIR", "DAIR"]


def rates(srcs, n=100000):
    files = []
    for s in srcs:
        files += glob.glob(s.rstrip("/") + "/*.slp")
    files = files[:n]
    recs = []
    with ProcessPoolExecutor(max_workers=16) as ex:
        for r in ex.map(scan, files, chunksize=8):
            recs.extend(r)
    out = {}
    for mv in MOVES:
        m = [x for x in recs if x[0] == mv]
        succ = sum(1 for (_, f, _, _) in m if f == 1)
        out[mv] = (succ, len(m))
    out["ALL"] = (sum(o[0] for o in out.values()), sum(o[1] for o in out.values()))
    return out, len(files)


def wilson(s, n):
    if n == 0:
        return (0, 0, 0)
    p = s / n
    z = 1.96
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return p*100, (c-h)*100, (c+h)*100


CORPUS = ["data/raw_slp/fox_master_master"]
MODEL = ["reports/health_foxvfox", "reports/baseline_replays"]

print("computing corpus (training humans) ...")
corp, nc = rates(CORPUS, n=int(sys.argv[1]) if len(sys.argv) > 1 else 1500)
print("computing model (live) ...")
mod, nm = rates(MODEL)

print(f"\n{'move':5s} | {'TRAINING humans':>26s} | {'MODEL (live)':>26s} | match?")
print("-" * 72)
for mv in MOVES + ["ALL"]:
    cs, cn = corp[mv]; ms, mn = mod[mv]
    cp, clo, chi = wilson(cs, cn)
    mp, mlo, mhi = wilson(ms, mn)
    overlap = not (mhi < clo or mlo > chi)  # CIs overlap?
    print(f"{mv:5s} | {cp:5.1f}% [{clo:4.1f},{chi:4.1f}] n={cn:5d} | "
          f"{mp:5.1f}% [{mlo:4.1f},{mhi:4.1f}] n={mn:4d} | {'yes' if overlap else 'NO'}")
