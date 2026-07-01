"""Controller-output canary — more drop-sensitive than action states, because a
dropped frame IS a forced-neutral input (center stick, no buttons).

Two readouts per set, bot isolated by human-type Fox:
  1. stick-distribution JS vs the training corpus (general).
  2. center-stick fraction: % of frames the main stick is near-neutral
     (|x|<0.2 and |y|<0.2). Drops force the stick to center, so this should
     climb ~linearly with the drop rate — a clean, interpretable drop signal.
"""
import sys, glob, collections
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from rlvr.state.peppi_adapter import Replay

# 5 bins per axis -> 25 stick cells; cell index of (x,y)
EDGES = np.array([-1.01, -0.5, -0.1, 0.1, 0.5, 1.01])


def _is_human(p):
    t = p.type
    return (t.value if hasattr(t, "value") else int(t)) == 0


def stick_cells(path):
    try:
        r = Replay(path)
    except Exception:
        return collections.Counter(), 0, 0
    th = [_is_human(p) for p in r._game.start.players]
    cnt = collections.Counter(); center = 0; total = 0
    for i, c in enumerate(r.player_characters):
        if c == 1 and th[i]:
            x = np.asarray(r._pre[i]["joystick_x"], dtype=float)
            y = np.asarray(r._pre[i]["joystick_y"], dtype=float)
            bx = np.clip(np.digitize(x, EDGES) - 1, 0, 4)
            by = np.clip(np.digitize(y, EDGES) - 1, 0, 4)
            for a, b in zip(bx, by):
                cnt[(int(a), int(b))] += 1
            center += int(np.sum((np.abs(x) < 0.2) & (np.abs(y) < 0.2)))
            total += len(x)
    return cnt, center, total


def agg(srcs, n=100000):
    fs = []
    for s in srcs:
        fs += glob.glob(s.rstrip("/") + "/*.slp")
    fs = fs[:n]
    cnt = collections.Counter(); ctr = 0; tot = 0
    with ProcessPoolExecutor(max_workers=16) as ex:
        for c, cc, tt in ex.map(stick_cells, fs, chunksize=8):
            cnt.update(c); ctr += cc; tot += tt
    return cnt, ctr, tot, len(fs)


def to_vec(cnt, keys, eps=1e-6):
    v = np.array([cnt.get(k, 0) for k in keys], dtype=float) + eps
    return v / v.sum()


def js(p, q):
    m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(a * np.log2(a / b)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


REF = ["data/raw_slp/fox_master_master"]
TESTS = {
    "clean (foxvfox)":    ["reports/health_foxvfox"],
    "clean (selfplay)":   ["reports/baseline_replays"],
    "drop 5%":            ["reports/drop_0.05"],
    "drop 20%":           ["reports/drop_0.20"],
    "drop 100% (broken)": ["reports/drop_1.0"],
}

rc, rctr, rtot, nref = agg(REF, n=int(sys.argv[1]) if len(sys.argv) > 1 else 800)
keys = [(a, b) for a in range(5) for b in range(5)]
refv = to_vec(rc, keys)
print(f"reference = corpus: {rtot:,} frames, {nref} replays, "
      f"center-stick {100*rctr/rtot:.1f}%\n")
print(f"{'set':22s} {'stick-JS':>9s} {'center-stick%':>14s}")
print("-" * 48)
for k, v in TESTS.items():
    c, ctr, tot, nf = agg(v)
    if tot == 0:
        print(f"{k:22s}   (no data)"); continue
    print(f"{k:22s} {js(refv, to_vec(c, keys)):9.3f} {100*ctr/tot:13.1f}%")
