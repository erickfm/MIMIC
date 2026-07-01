"""Distributional canary: compare the bot's action-state distribution to the
training corpus (what it learned from). A faithful pipeline -> low divergence;
corrupted inputs/decode -> the whole behavior distribution shifts -> high
divergence. Validated against the 100%-drop replays (known-broken control).

Bot isolated by human-type Fox (so model-vs-CPU runs measure only the bot)."""
import sys, glob, collections
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from rlvr.state.peppi_adapter import Replay


def _is_human(p):
    t = p.type
    return (t.value if hasattr(t, "value") else int(t)) == 0


def states_of(path):
    try:
        r = Replay(path)
    except Exception:
        return []
    th = [_is_human(p) for p in r._game.start.players]
    out = []
    for i, c in enumerate(r.player_characters):
        if c == 1 and th[i]:
            out.extend(int(s) for s in r._post[i]["state"])
    return out


def dist(srcs, n=100000):
    fs = []
    for s in srcs:
        fs += glob.glob(s.rstrip("/") + "/*.slp")
    fs = fs[:n]
    cnt = collections.Counter()
    with ProcessPoolExecutor(max_workers=16) as ex:
        for st in ex.map(states_of, fs, chunksize=8):
            cnt.update(st)
    return cnt, len(fs)


def to_vec(cnt, keys, eps=1e-6):
    v = np.array([cnt.get(k, 0) for k in keys], dtype=float) + eps
    return v / v.sum()


def js(p, q):
    m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(a * np.log2(a / b)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def tv(p, q):
    return 0.5 * float(np.sum(np.abs(p - q)))


REF = ["data/raw_slp/fox_master_master"]
TESTS = {
    "clean (foxvfox)":   ["reports/health_foxvfox"],
    "clean (selfplay)":  ["reports/baseline_replays"],
    "drop 5%":           ["reports/drop_0.05"],
    "drop 20%":          ["reports/drop_0.20"],
    "drop 100% (broken)":["reports/drop_1.0"],
}

ref_cnt, nref = dist(REF, n=int(sys.argv[1]) if len(sys.argv) > 1 else 800)
test = {k: dist(v) for k, v in TESTS.items()}
keys = sorted(set(ref_cnt).union(*[set(c) for c, _ in test.values()]))
refv = to_vec(ref_cnt, keys)
print(f"reference = master-Fox corpus: {sum(ref_cnt.values()):,} frames, "
      f"{len(ref_cnt)} distinct states, {nref} replays\n")
print(f"{'set':22s} {'replays':>7s} {'JS-div':>8s} {'TV-dist':>8s}")
print("-" * 50)
for k, (c, nf) in test.items():
    if sum(c.values()) == 0:
        print(f"{k:22s} {nf:7d}   (no data)"); continue
    pv = to_vec(c, keys)
    print(f"{k:22s} {nf:7d} {js(refv, pv):8.3f} {tv(refv, pv):8.3f}")
