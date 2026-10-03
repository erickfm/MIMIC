"""Per-character inference-faithfulness canary for h2h replays.

Generalizes tools/lcancel_analysis.py + tools/ctrl_canary.py to ANY character
(both are Fox-hardcoded). Same logic, parameterized by character id:

  - center-stick %  : fraction of frames the main stick is near-neutral
                      (|x|<0.2 & |y|<0.2). A broken/dropped-input pipeline
                      forces neutral -> climbs toward 100%. Char-specific
                      baseline, but ~100% = broken for ANY char.
  - L-cancel rate   : of aerial landings (states 70-74), fraction that hit
                      minimal lag (flag==1 OR realized avoidable lag == 0,
                      where cancelled_min self-calibrates per move from the
                      char's own clean L-cancelled landings). ~0% = mistimed.

Both players in an h2h are type=0 bots; the character id disambiguates, so
each character is validated by its own frames across every replay it appears in.

Usage:  python3 tools/h2h_canary.py <replay_dir> <char1,char2,...> [n_files]
"""
import sys, glob, collections
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import melee
from rlvr.state.peppi_adapter import Replay

LANDING = {70: "NAIR", 71: "FAIR", 72: "BAIR", 73: "UAIR", 74: "DAIR"}
AIRBORNE = set(range(25, 35)) | {36, 37, 38} | set(range(65, 70))
DEAD = set(range(0, 11))
DAMAGE = set(range(75, 92))


def exit_cat(s):
    if s in DEAD: return "dead"
    if s in DAMAGE: return "damage"
    if s in AIRBORNE: return "airborne"
    return "grounded"


def _is_human(p):
    t = p.type
    return (t.value if hasattr(t, "value") else int(t)) == 0


def scan(args):
    path, char_id = args
    try:
        r = Replay(path)
    except Exception:
        return None
    th = [_is_human(p) for p in r._game.start.players]
    center = total = 0
    landings = []  # (move, flag, lag, exit)
    for i, c in enumerate(r.player_characters):
        if c != char_id or not th[i]:
            continue
        x = np.asarray(r._pre[i]["joystick_x"], dtype=float)
        y = np.asarray(r._pre[i]["joystick_y"], dtype=float)
        center += int(np.sum((np.abs(x) < 0.2) & (np.abs(y) < 0.2)))
        total += len(x)
        st = np.asarray(r._post[i]["state"]).astype(int)
        lc = np.asarray(r._post[i]["l_cancel"]).astype(int)
        n = len(st); t = 1
        while t < n:
            if st[t] in LANDING and st[t-1] != st[t]:
                j = t
                while j < n and st[j] == st[t]:
                    j += 1
                ex = exit_cat(int(st[j])) if j < n else "grounded"
                landings.append((LANDING[st[t]], int(lc[t]), j - t, ex))
                t = j
            else:
                t += 1
    return (center, total, landings)


def lcancel_rate(landings):
    """Fraction of aerial landings that achieved minimal lag, per the
    realized-avoidable-lag rule (matches lcancel_analysis.py)."""
    if not landings:
        return None, 0
    good = tot = 0
    for mv in LANDING.values():
        m = [x for x in landings if x[0] == mv]
        if not m:
            continue
        clean = [lag for (_, f, lag, ex) in m if f == 1 and ex == "grounded"]
        cmin = int(np.median(clean)) if clean else 0
        for (_, f, lag, ex) in m:
            tot += 1
            if f == 1 or max(0, lag - cmin) == 0:
                good += 1
    return (good / tot if tot else None), tot


def main(replay_dir, chars, n_files):
    files = sorted(set(glob.glob(replay_dir.rstrip("/") + "/*.slp") +
                       glob.glob(replay_dir.rstrip("/") + "/**/*.slp", recursive=True)))[:n_files]
    name2id = {c: melee.Character[
        {"ice_climbers": "POPO", "puff": "JIGGLYPUFF"}.get(c, c.upper())].value
        for c in chars}
    print(f"canary over {len(files)} replays in {replay_dir}\n")
    print(f"{'char':13s}{'frames':>10s}{'center%':>9s}{'landings':>9s}{'Lcancel%':>9s}  verdict")
    rows = {}
    for c in chars:
        cid = name2id[c]
        cen = tot = 0; lands = []
        with ProcessPoolExecutor(max_workers=12) as ex:
            for res in ex.map(scan, [(f, cid) for f in files], chunksize=8):
                if res is None: continue
                cen += res[0]; tot += res[1]; lands += res[2]
        if tot == 0:
            print(f"{c:13s}{'(not present)':>28s}"); continue
        cpct = 100 * cen / tot
        lrate, nland = lcancel_rate(lands)
        # universal failure signatures: center near 100% = dropped inputs;
        # L-cancel collapsed = mistimed. Char baselines differ, so flag only
        # gross anomalies.
        bad = []
        if cpct > 80: bad.append("CENTER-HIGH(drops?)")
        if lrate is not None and nland >= 20 and lrate < 0.30:
            bad.append("LCANCEL-LOW(mistimed?)")
        verdict = "OK" if not bad else " ".join(bad)
        lr = f"{100*lrate:.0f}" if lrate is not None else "n/a"
        print(f"{c:13s}{tot:>10d}{cpct:>8.1f}%{nland:>9d}{lr:>8s}%  {verdict}")
        rows[c] = {"frames": tot, "center_pct": round(cpct,1),
                   "landings": nland, "lcancel_pct": round(100*lrate,1) if lrate else None,
                   "verdict": verdict}
    return rows


if __name__ == "__main__":
    rd = sys.argv[1]
    chars = sys.argv[2].split(",")
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 100000
    main(rd, chars, n)
