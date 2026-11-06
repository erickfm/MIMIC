#!/usr/bin/env python3
"""Check V against sim rollouts from held-out games before the app relies on it.

At N random frames of held-out games (value.is_val), run K model rollouts of 300 frames (both
sides played by the BC models). Measures:
  1. V at the branch frame vs the rollouts' mean 5-second score (does V know who is winning?);
  2. per rollout, the early estimate Q = score over the first S frames + V at frame S, vs that
     rollout's own 5-second score (can V stand in for the rest of a rollout?), and how much it
     shrinks the spread between tries from the same spot (what the review needs).

  python3 tools/coach/value_check.py --char FALCO --value data/coach_rank/value_falco.pt --n 80
"""
import argparse, json, os, random, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "tools", ROOT / "tools" / "coach"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--char", required=True)
    ap.add_argument("--value", required=True)
    ap.add_argument("--replays", default="~/Slippi")
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--s", type=int, default=60, help="frames before V takes over")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda", help="cuda or cpu (smoke tests)")
    ap.add_argument("--all-held-out", action="store_true",
                    help="every game in --replays is held out (V was trained on other replays, e.g. ranked "
                         "archives the sim can't replay exactly): don't filter by value.is_val, and take "
                         "several spots per game when there are fewer games than --n")
    a = ap.parse_args()
    from branch import Brancher
    from serve import POLICY
    from value import Value, is_val, score, STOCK
    from sim_play import template_players
    rng = random.Random(1)
    # current-Slippi 1v1 games (the sim is exact on these), held out from V's training
    games = []
    from peppi_py import read_slippi
    import glob
    for p in sorted(glob.glob(os.path.join(os.path.expanduser(a.replays), "**", "*.slp"), recursive=True)):
        if not a.all_held_out and not is_val(p):
            continue
        try:
            g = read_slippi(p, skip_frames=True)
        except Exception:
            continue
        if g.start.slippi.version[:2] < (3, 18):
            continue
        try:
            pl = template_players(p)
        except Exception:
            continue
        if len(pl) != 2 or not all(c in POLICY for _, c in pl):
            continue
        if a.char not in (c for _, c in pl):
            continue
        n = (g.metadata or {}).get("lastFrame")
        if n is None:                                  # no metadata: the length from the frames
            n = int(read_slippi(p).frames.id[-1])
        for port, c in pl:
            if c == a.char and n > 1500:
                games.append((p, port, n, pl))
    rng.shuffle(games)
    print(f"{len(games)} held-out {a.char} perspectives; using {min(a.n, len(games))}", flush=True)
    if a.all_held_out and games and len(games) < a.n:  # several spots per game (a new random frame each)
        games = (games * (a.n // len(games) + 1))[:a.n]
    br = Brancher(a.device)
    V = Value(a.value, a.device)
    rows = []
    for p, port, n, pl in games[:a.n]:
        F = rng.randint(600, n - 400)
        pol = {q: str(ROOT / POLICY[c]) for q, c in pl}
        opp = [q for q, _ in pl if q != port][0]
        try:
            r = br.run(p, F, a.k, 300, pol, value_at=(port, [0, a.s]))
        except Exception as e:
            print("  skip:", str(e)[:120], flush=True)
            continue
        if not r["exact"] or any(len(ro["frames"]) < 301 for ro in r["rollouts"]):
            continue
        v0 = V(r["value_feats"][0][:1])[0]
        vs = V(r["value_feats"][a.s])
        tries = []
        for i, ro in enumerate(r["rollouts"]):
            fr = ro["frames"]
            def sc(lo, hi):
                pct = lambda q: [f[q][5] for f in fr[lo:hi + 1]]
                stk = lambda q: [f[q][6] for f in fr[lo:hi + 1]]
                dd = lambda v: float(np.clip(np.diff(v), 0, None).sum())
                ds = lambda v: float(np.clip(-np.diff(v), 0, None).sum())
                return np.array([dd(pct(opp)), dd(pct(port)), ds(stk(opp)), ds(stk(port))])
            full, early = sc(0, len(fr) - 1), sc(0, min(a.s, len(fr) - 1))
            tries.append(dict(full=float(score(full)), q=float(score(early) + score(vs[i]))))
        rows.append(dict(slp=p, frame=F, v0=float(score(v0)), tries=tries))
        print(f"  {len(rows)}: V0 {rows[-1]['v0']:+.1f}  rollouts mean {np.mean([t['full'] for t in tries]):+.1f}", flush=True)
    if a.out:
        json.dump(rows, open(a.out, "w"))
    v0 = np.array([r["v0"] for r in rows])
    mean_full = np.array([np.mean([t["full"] for t in r["tries"]]) for r in rows])
    full = np.array([[t["full"] for t in r["tries"]] for r in rows])
    q = np.array([[t["q"] for t in r["tries"]] for r in rows])
    corr = lambda x, y: float(np.corrcoef(x, y)[0, 1])
    print(f"\n{len(rows)} spots, {a.k} tries each")
    print(f"1. V at the spot vs mean of {a.k} tries' 5-s score: r = {corr(v0, mean_full):.2f} "
          f"(mean V {v0.mean():+.1f}, mean tries {mean_full.mean():+.1f})")
    print(f"2. per try, Q (first {a.s} frames + V) vs its own 5-s score: r = {corr(q.ravel(), full.ravel()):.2f}")
    sd_full = float(np.mean(full.std(1, ddof=1)))
    sd_q = float(np.mean(q.std(1, ddof=1)))
    print(f"   spread between tries from the same spot: 5-s score sd {sd_full:.1f}, Q sd {sd_q:.1f}")
    # does Q rank tries like their real outcomes do? within-spot correlation of deviations
    dq, df = (q - q.mean(1, keepdims=True)).ravel(), (full - full.mean(1, keepdims=True)).ravel()
    print(f"   within a spot, Q vs real outcome (which try went better): r = {corr(dq, df):.2f}")


if __name__ == "__main__":
    main()
