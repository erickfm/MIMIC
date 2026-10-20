#!/usr/bin/env python3
"""Fidelity check of the native sim's LIVE input path (rlvr/sim/melee_sim.py, --env mode).

Plays a recorded 1v1 .slp through MeleeSim using only each player's raw controller state from the
replay (pre-frame physical buttons, raw sticks, physical triggers), fed through pad_read exactly
like a live controller, and compares every post-frame against the recording with the same physics
criterion as tools/sim_replay_check.py. Unlike that check (replay playback injects the game's
processed inputs and restores the RNG every frame), nothing here is restored: the game processes
the pads itself and its RNG runs free after being set once, at frame -123, to the recording's
(offline games), or follows Slippi online's per-frame seed rule (online games).

Usage:
  python3 tools/sim_env_check.py <replay.slp> [--offset K]
  python3 tools/sim_env_check.py --batch <dir> [--n N] [--jobs J]
"""
import argparse, json, os, struct, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rlvr.sim.melee_sim import MeleeSim, Pad, DEFAULT_EXE  # noqa: E402
from tools.sim_replay_check import slp_events, physics_diff  # noqa: E402


def recorded(slp):
    pre, post = {}, {}
    for c, p in slp_events(slp):
        if c in (0x37, 0x38) and len(p) >= 6:
            k = (struct.unpack(">i", p[:4])[0], p[4], p[5])
            (pre if c == 0x37 else post)[k] = p
    return pre, post


def pad_from_pre(p):
    """The raw controller state a pre-frame records (what PADRead returned that frame)."""
    s8 = lambda o: struct.unpack_from("b", p, o)[0]
    trig = lambda o: min(255, round(struct.unpack_from(">f", p, o)[0] * 140))
    return Pad(button=struct.unpack_from(">H", p, 48)[0], stick_x=s8(58), stick_y=s8(63),
               sub_x=s8(64), sub_y=s8(65), trigger_l=trig(50), trigger_r=trig(54))


def check(slp, offset=0, exe=DEFAULT_EXE):
    pre, post = recorded(slp)
    last = max(k[0] for k in post)
    def pads_for(frame):
        out = []
        for port in range(4):
            p = pre.get((frame, port, 0))
            out.append(pad_from_pre(p) if p is not None else Pad())
        return out
    # The recording's RNG state at its first frame: matched once, then the RNG runs free (offline)
    # or follows Slippi online's per-frame seed (an online game: seeds step by 1 << 16).
    rec_seed = {struct.unpack(">i", p[:4])[0]: struct.unpack(">I", p[4:8])[0]
                for c, p in slp_events(slp) if c == 0x3A}
    s = [rec_seed[f] for f in sorted(rec_seed)[:3]]
    rng = "online" if len(s) == 3 and s[1] - s[0] == s[2] - s[1] == 0x10000 else "offline"
    # Slippi online's seed counter is its online clock: a pause in the recording (the env never
    # pauses) moves it ahead of the frame index, and from there the two RNG streams differ. Such a
    # recording is compared up to that frame.
    cut = None
    if rng == "online":
        base = s[0] & 0xFFFF
        cut = next((f for f in sorted(rec_seed) if rec_seed[f] != (((f + 123) << 16) | base) & 0xFFFFFFFF), None)
    if cut is not None:
        last = min(last, cut - 1)
    sim = MeleeSim(slp, exe=exe, seed=s[0], rng=rng)
    f = sim.reset(initial_pads=pads_for(-123))
    got = {}
    while True:
        for fr in sim.last_frames:
            for k, p in fr.post.items():
                got[(fr.index,) + k] = p
        if sim.done or f.index >= last:
            break
        f = sim.step(pads_for(f.index + 1 + offset))
        if f is None:
            break
    extra = sim.extra_frames
    sim.close()
    keys = sorted(k for k in post if 0 <= k[0] <= last)
    div = [k for k in keys if k not in got or physics_diff(post[k], got[k])]
    first = None
    if div:
        k = div[0]
        first = {"frame": k[0], "port": k[1], "follower": k[2],
                 "fields": physics_diff(post[k], got[k]) if k in got else ["missing"]}
    return {"slp": str(slp), "offset": offset, "rng": rng, "compared_until": last, "online_clock_jump": cut,
            "extra_frames": extra, "match_player_frames": len(keys),
            "divergent": len(div), "first_divergence": first, "exact": not div}


def check_safe(slp, offset=0, exe=DEFAULT_EXE):
    try:
        return check(slp, offset, exe)
    except Exception as e:  # noqa: BLE001 - reported per replay
        return {"slp": str(slp), "error": f"{type(e).__name__}: {e}", "exact": False}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("slp", nargs="?")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--exe", default=str(DEFAULT_EXE))
    ap.add_argument("--batch")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.slp:
        print(json.dumps(check(a.slp, a.offset, a.exe), indent=1))
        return
    files = sorted(str(p.absolute()) for p in Path(a.batch).glob("*.slp"))[:a.n]
    with ProcessPoolExecutor(a.jobs) as ex:
        res = list(ex.map(check_safe, files, [a.offset] * len(files), [a.exe] * len(files)))
    for r in res:
        if "error" in r:
            print(Path(r["slp"]).name, "ERROR", r["error"])
            continue
        d = r["first_divergence"]
        cut = f"  [online clock jumps at {r['online_clock_jump']}: compared to there]" if r["online_clock_jump"] else ""
        print(Path(r["slp"]).name, r["rng"], "EXACT" if r["exact"] else f"DIVERGES@{d['frame']} {d['fields']} "
              f"({r['divergent']}/{r['match_player_frames']})", cut)
    n = sum(r["exact"] for r in res)
    print(f"\n{n}/{len(res)} replays physics-exact through the live pad path")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
