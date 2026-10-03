"""One-command faithfulness validator for a directory of .slp replays.

Runs the full canary suite so you don't have to eyeball each check by hand, and
prints a GREEN / YELLOW / RED flag per check with an overall verdict. Exit code
is 0 (green), 1 (yellow), or 2 (red) so it's usable in scripts / CI.

What it checks (each is a distinct failure mode — they are NOT interchangeable):

  integrity      >2-player replays (always wrong) and same-char-both-ports
                 (unscheduled mirror = the --alternate-ports char-select bug).
  center-stick%  dropped/neutralized inputs -> climbs toward 100%. BLIND to
                 mistiming (a stale pad is still off-center).
  L-cancel%      input TIMING. Collapses toward 0 when frames are mistimed
                 (stale-pad FFW, double-flush, wrong emulator). center-stick
                 CANNOT see this; only L-cancel can.
  landings/kf    aerial landings per 1000 frames. Collapses in the "mush"
                 signature (both bots mistimed -> knocked out of aerials, fast
                 deaths). Low L-cancel + low landings/kf together = real
                 mistiming; low L-cancel alone can just be a hard opponent mix.
  stick-JS       main-stick output distribution vs a reference (--ref). Catches
  state-JS       action-state distribution vs a reference. "wrong-but-active"
                 corruption that the scalar checks miss. Needs --ref.

--ref is the faithful reference to compare distributions against: a realtime or
emulator_ss single-instance run of the SAME matchup is the strictest choice
(isolates a harness/emulator change). Without --ref, JS checks are skipped and
the scalar checks fall back to absolute healthy bands.

Usage:
  python3 tools/validate_run.py <replay_dir> [--ref <ref_dir>] [--chars fox,bowser]
  # PYTHONPATH must include the repo root (needs rlvr.state.peppi_adapter)
"""
import sys, glob, argparse, collections
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import melee
from rlvr.state.peppi_adapter import Replay

LANDING = {70: "NAIR", 71: "FAIR", 72: "BAIR", 73: "UAIR", 74: "DAIR"}
AIRBORNE = set(range(25, 35)) | {36, 37, 38} | set(range(65, 70))
DEAD = set(range(0, 11))
DAMAGE = set(range(75, 92))
EDGES = np.array([-1.01, -0.5, -0.1, 0.1, 0.5, 1.01])
SKEYS = [(a, b) for a in range(5) for b in range(5)]

G, Y, R, GRAY = "GREEN", "YELLOW", "RED", "GRAY"
_ANSI = {G: "\033[92m", Y: "\033[93m", R: "\033[91m", GRAY: "\033[90m"}
_SYM = {G: "●", Y: "●", R: "●", GRAY: "○"}
_RANK = {GRAY: -1, G: 0, Y: 1, R: 2}


def paint(flag, text=None):
    return f"{_ANSI[flag]}{_SYM[flag]} {text if text is not None else flag}\033[0m"


def _human(p):
    t = p.type
    return (t.value if hasattr(t, "value") else int(t)) == 0


def exit_cat(s):
    if s in DEAD: return "dead"
    if s in DAMAGE: return "damage"
    if s in AIRBORNE: return "airborne"
    return "grounded"


def scan(args):
    """Per-replay, per-character-of-interest aggregates + replay-level integrity."""
    path, cids = args
    try:
        r = Replay(path)
    except Exception:
        return None
    th = [_human(p) for p in r._game.start.players]
    pcs = list(r.player_characters)
    integ = {"n_players": len(pcs), "same_char": len(set(pcs)) == 1 and len(pcs) == 2}
    out = {}
    for i, c in enumerate(pcs):
        if c not in cids or not th[i]:
            continue
        x = np.asarray(r._pre[i]["joystick_x"], float)
        y = np.asarray(r._pre[i]["joystick_y"], float)
        st = np.asarray(r._post[i]["state"]).astype(int)
        lc = np.asarray(r._post[i]["l_cancel"]).astype(int)
        stick = collections.Counter()
        bx = np.clip(np.digitize(x, EDGES) - 1, 0, 4)
        by = np.clip(np.digitize(y, EDGES) - 1, 0, 4)
        for a, b in zip(bx, by):
            stick[(int(a), int(b))] += 1
        state = collections.Counter(int(s) for s in st)
        landings = []
        n = len(st); t = 1
        while t < n:
            if st[t] in LANDING and st[t - 1] != st[t]:
                j = t
                while j < n and st[j] == st[t]:
                    j += 1
                ex = exit_cat(int(st[j])) if j < n else "grounded"
                landings.append((LANDING[st[t]], int(lc[t]), j - t, ex))
                t = j
            else:
                t += 1
        d = out.setdefault(c, {"center": 0, "total": 0, "stick": collections.Counter(),
                               "state": collections.Counter(), "landings": [], "frames": []})
        d["center"] += int(np.sum((np.abs(x) < 0.2) & (np.abs(y) < 0.2)))
        d["total"] += len(x)
        d["stick"].update(stick)
        d["state"].update(state)
        d["landings"] += landings
        d["frames"].append(len(x))
    return integ, out


def aggregate(replay_dir, cids, n_files):
    files = sorted(set(glob.glob(replay_dir.rstrip("/") + "/*.slp") +
                       glob.glob(replay_dir.rstrip("/") + "/**/*.slp", recursive=True)))[:n_files]
    per = {c: {"center": 0, "total": 0, "stick": collections.Counter(),
               "state": collections.Counter(), "landings": [], "frames": []} for c in cids}
    integ = {"n_files": len(files), "multiplayer": 0, "same_char": 0}
    with ProcessPoolExecutor(max_workers=12) as ex:
        for res in ex.map(scan, [(f, set(cids)) for f in files], chunksize=8):
            if res is None:
                continue
            ig, out = res
            if ig["n_players"] != 2:
                integ["multiplayer"] += 1
            if ig["same_char"]:
                integ["same_char"] += 1
            for c, d in out.items():
                p = per[c]
                p["center"] += d["center"]; p["total"] += d["total"]
                p["stick"].update(d["stick"]); p["state"].update(d["state"])
                p["landings"] += d["landings"]; p["frames"] += d["frames"]
    return per, integ, files


def lcancel_rate(landings):
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


def vec(cnt, keys, eps=1e-6):
    v = np.array([cnt.get(k, 0) for k in keys], float) + eps
    return v / v.sum()


def js(p, q):
    m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(a * np.log2(a / b)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def band(v, lo, hi, invert=False):
    """GREEN below lo, YELLOW in [lo,hi), RED at/above hi (invert=True flips)."""
    if invert:
        if v >= lo: return G
        if v >= hi: return Y
        return R
    if v < lo: return G
    if v < hi: return Y
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("replay_dir")
    ap.add_argument("--ref", default=None, help="faithful reference replay dir for JS comparison")
    ap.add_argument("--chars", default=None, help="comma names (default: auto-detect all present)")
    ap.add_argument("--n-files", type=int, default=100000)
    ap.add_argument("--min-landings", type=int, default=20, help="below this, L-cancel is GRAY")
    ap.add_argument("--baselines", default=None,
                    help="per-char baseline JSON (from --emit-baseline on a human "
                         "corpus). When a char has an entry, flags are RELATIVE to "
                         "that char's own normal, not a universal band. Chars without "
                         "an entry fall back to absolute bands.")
    ap.add_argument("--emit-baseline", default=None,
                    help="write the computed per-char metrics to this JSON as a "
                         "baseline table (run this on a faithful human corpus).")
    args = ap.parse_args()

    import json
    baselines = {}
    if args.baselines:
        try:
            baselines = json.load(open(args.baselines))
        except Exception as e:
            print(f"warning: could not load baselines {args.baselines}: {e}")

    id2name = {c.value: c.name for c in melee.Character}
    if args.chars:
        cids = [melee.Character[{"ice_climbers": "POPO", "puff": "JIGGLYPUFF"}.get(
            c, c.upper())].value for c in args.chars.split(",")]
    else:
        # auto-detect chars present (sample up to 60 replays)
        fs = sorted(set(glob.glob(args.replay_dir.rstrip("/") + "/*.slp") +
                        glob.glob(args.replay_dir.rstrip("/") + "/**/*.slp", recursive=True)))[:60]
        seen = collections.Counter()
        with ProcessPoolExecutor(max_workers=12) as ex:
            for res in ex.map(scan, [(f, set(id2name)) for f in fs], chunksize=8):
                if res:
                    for c in res[1]:
                        seen[c] += 1
        cids = [c for c, _ in seen.most_common()]

    per, integ, files = aggregate(args.replay_dir, cids, args.n_files)
    refper = None
    if args.ref:
        refper, _, _ = aggregate(args.ref, cids, args.n_files)

    print(f"\nvalidate_run: {integ['n_files']} replays in {args.replay_dir}"
          + (f"  vs ref {args.ref}" if args.ref else "  (no --ref: absolute bands, JS skipped)"))
    worst = _RANK[G]

    # ---- integrity (replay-level) ----
    mp, sc = integ["multiplayer"], integ["same_char"]
    fi = R if mp else G
    print(f"\n  integrity")
    print(f"    {paint(fi):32s} >2-player replays: {mp}"
          + ("  (always wrong)" if mp else ""))
    fsame = Y if sc else G
    print(f"    {paint(fsame):32s} same-char-both-ports: {sc}"
          + ("  (unscheduled mirror? verify vs schedule)" if sc else ""))
    worst = max(worst, _RANK[fi], _RANK[fsame])

    # ---- per-character canaries ----
    bl_note = " (baseline-relative)" if baselines else " (absolute bands)"
    hdr = f"\n  {'char':13s}{'frames':>9s}{'center%':>9s}{'Lcancel%':>9s}{'land/kf':>8s}"
    if baselines:
        hdr += "   [char baseline]"
    if refper:
        hdr += f"{'stickJS':>9s}{'stateJS':>9s}"
    print(f"  per-character canaries{bl_note}:")
    print(hdr)
    akeys = None
    emit = {}
    for c in cids:
        d = per[c]
        if d["total"] == 0:
            continue
        cpct = 100 * d["center"] / d["total"]
        lrate, nland = lcancel_rate(d["landings"])
        lpk = 1000 * nland / d["total"]
        name = id2name.get(c, str(c)).lower()
        emit[name] = {"lcancel": round(100 * lrate, 1) if lrate else None,
                      "center": round(cpct, 1), "land_kf": round(lpk, 2),
                      "n_landings": nland, "frames": d["total"]}
        bl = baselines.get(name)
        if bl and bl.get("lcancel") is not None:
            # flag RELATIVE to this character's own human-corpus normal
            f_center = band(cpct, bl["center"] + 12, bl["center"] + 25)      # higher=worse
            f_land = band(lpk, bl["land_kf"] * 0.7, bl["land_kf"] * 0.4, invert=True)  # lower=worse
            if lrate is None or nland < args.min_landings:
                f_lc = GRAY
            else:
                f_lc = band(lrate * 100, bl["lcancel"] - 10, bl["lcancel"] - 20, invert=True)
        else:
            # no baseline for this char -> universal absolute bands
            f_center = band(cpct, 45, 70)
            f_land = band(lpk, 0.5, 0.3, invert=True)
            if lrate is None or nland < args.min_landings:
                f_lc = GRAY
            else:
                f_lc = band(lrate * 100, 70, 45, invert=True)
        flags = [f_center, f_land, f_lc]
        cells = [
            paint(f_center, f"{cpct:.0f}%"),
            paint(f_lc, (f"{100*lrate:.0f}%" if lrate is not None else "n/a")),
            paint(f_land, f"{lpk:.2f}"),
        ]
        js_cells = []
        if refper and refper[c]["total"] > 0:
            rk = refper[c]
            akeys = sorted(set(d["state"]) | set(rk["state"]))
            sjs = js(vec(rk["stick"], SKEYS), vec(d["stick"], SKEYS))
            ajs = js(vec(rk["state"], akeys), vec(d["state"], akeys))
            f_sjs = band(sjs, 0.05, 0.15); f_ajs = band(ajs, 0.08, 0.20)
            flags += [f_sjs, f_ajs]
            js_cells = [paint(f_sjs, f"{sjs:.3f}"), paint(f_ajs, f"{ajs:.3f}")]
        worst = max(worst, *[_RANK[f] for f in flags])
        # ANSI codes inflate width; pad the plain columns, print colored cells raw
        line = (f"  {name:13s}{d['total']:>9d}"
                f"  {cells[0]}  {cells[1]}  {cells[2]}")
        bl = baselines.get(name)
        if baselines:
            if bl and bl.get("lcancel") is not None:
                line += f"   [Lc {bl['lcancel']:.0f}% ce {bl['center']:.0f}% kf {bl['land_kf']:.1f}]"
            else:
                line += "   [no baseline -> absolute]"
        if js_cells:
            line += f"  {js_cells[0]}  {js_cells[1]}"
        print(line)
        nl_note = "" if nland >= args.min_landings else f"  (only {nland} landings — L-cancel GRAY)"
        if nl_note:
            print(f"  {'':13s}{nl_note}")

    if args.emit_baseline:
        json.dump(emit, open(args.emit_baseline, "w"), indent=1)
        print(f"\n  wrote baseline for {len(emit)} chars -> {args.emit_baseline}")

    verdict = {0: G, 1: Y, 2: R}[worst]
    print("\n  " + paint(verdict, f"OVERALL: {verdict}"))
    if baselines:
        print("  bands are RELATIVE to each char's baseline: Lcancel red if >20pt below,")
        print("  yellow >10pt below; center red if >25pt above; land/kf red if <40% of baseline.")
    else:
        print("  guide: center%<45 / Lcancel>70 / land-per-kf>0.5 / stickJS<0.05 / stateJS<0.08 = green")
    print("  note: low L-cancel with HEALTHY land/kf is usually a hard opponent mix, not mistiming;")
    print("        low L-cancel AND low land/kf together is the real mistiming/mush signature.")
    sys.exit(worst)


if __name__ == "__main__":
    main()
