"""Aggregate L-cancel / realized-landing-lag analysis over many Fox replays.

For every Fox aerial landing (entry into NAIR..DAIR_LANDING = 70..74) record the
move, the engine l_cancel flag (1=success, 2=miss), the realized lag (frames in
the landing state), and what the landing EXITED into, categorized as:
  grounded  — clean landing (standing/crouch/shield/walk/turn/attack...)
  airborne  — slid off a ledge (jump/fall/tumble/aerial)
  damage    — knocked out of the landing (got hit)
  dead      — died

The question: of the l_cancel==2 "misses", how many actually cost full lag
(grounded + long) vs cost nothing (hit / slid-off / already-low lag)?
"""
import sys, glob, collections
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from melee import Action
from rlvr.state.peppi_adapter import Replay

LANDING = {70: "NAIR", 71: "FAIR", 72: "BAIR", 73: "UAIR", 74: "DAIR"}
AIRBORNE = set(range(25, 35)) | {36, 37, 38} | set(range(65, 70))
DEAD = set(range(0, 11))
DAMAGE = set(range(75, 92))  # DAMAGE_* family


def exit_cat(s):
    if s in DEAD: return "dead"
    if s in DAMAGE: return "damage"
    if s in AIRBORNE: return "airborne"
    return "grounded"


def _is_human(p):
    t = p.type
    return (t.value if hasattr(t, "value") else int(t)) == 0


def scan(path):
    try:
        r = Replay(path)
    except Exception:
        return []
    types_human = [_is_human(p) for p in r._game.start.players]
    out = []
    for i, c in enumerate(r.player_characters):
        if c != 1:
            continue
        if not types_human[i]:   # human-controlled Fox only (the model bot; excludes CPU Fox)
            continue
        st = np.asarray(r._post[i]["state"]).astype(int)
        lc = np.asarray(r._post[i]["l_cancel"]).astype(int)
        n = len(st)
        t = 1
        while t < n:
            if st[t] in LANDING and st[t-1] != st[t]:
                j = t
                while j < n and st[j] == st[t]:
                    j += 1
                lag = j - t
                ex = exit_cat(int(st[j])) if j < n else "grounded"
                out.append((LANDING[st[t]], int(lc[t]), lag, ex))
                t = j
            else:
                t += 1
    return out


def main(src, n_games):
    pat = src if src.endswith(".slp") else src.rstrip("/") + "/*.slp"
    files = sorted(glob.glob(pat))[:n_games]
    print(f"scanning {len(files)} Fox replays from {src} ...")
    recs = []
    with ProcessPoolExecutor(max_workers=16) as ex:
        for r in ex.map(scan, files, chunksize=8):
            recs.extend(r)
    print(f"total Fox aerial landings: {len(recs)}\n")
    moves = ["NAIR", "FAIR", "BAIR", "UAIR", "DAIR"]
    for mv in moves:
        m = [x for x in recs if x[0] == mv]
        if not m:
            continue
        # cancelled_min = median lag of CLEAN (grounded) L-cancelled landings
        cl_grounded = [lag for (_, f, lag, ex) in m if f == 1 and ex == "grounded"]
        cmin = int(np.median(cl_grounded)) if cl_grounded else -1
        ms = [(lag, ex) for (_, f, lag, ex) in m if f == 2]               # misses
        # UNIFORM rule: avoidable = max(0, realized_lag - cancelled_min), all exits
        avoid = [(max(0, lag - cmin), ex) for lag, ex in ms]
        real = [a for a, _ in avoid if a > 0]
        print(f"=== {mv}  (n={len(m)})  cancelled_min={cmin}f ===")
        print(f"  L-cancelled (flag1): n={sum(1 for _,f,_,_ in m if f==1)}")
        print(f"  MISSED (flag2): n={len(ms)}")
        if ms:
            print(f"    real misses (avoidable_lag>0): {len(real)} = {100*len(real)/len(ms):.0f}%"
                  f"   |  truly free (avoidable=0): {len(ms)-len(real)}")
            print(f"    mean avoidable_lag over all misses: {np.mean([a for a,_ in avoid]):.1f}f"
                  f"   (over real misses: {np.mean(real):.1f}f)")
        # per-exit: does the user's rule reclassify 'got hit' as real?
        for cat in ("grounded", "damage", "airborne", "dead"):
            sub = [(a) for a, e in avoid if e == cat]
            if sub:
                rr = sum(1 for a in sub if a > 0)
                print(f"    {cat:9s}: n={len(sub):4d}  real(avoidable>0)={100*rr/len(sub):3.0f}%"
                      f"  mean_avoidable={np.mean(sub):.1f}f")
        print()


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "data/raw_slp/fox_master_master"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
    main(src, n)
