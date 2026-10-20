#!/usr/bin/env python3
"""Byte-exact fidelity check of a native (decomp) Melee build against Slippi Dolphin replays.

The native Linux host (built by tools/build_native_sim.sh; `melee_linux --replay X.slp --replay-dump X.bin`)
plays a recorded .slp from its controller inputs only (resync off: positions, actions, percent
are simulated, not restored; the RNG seed is restored per frame exactly as Dolphin's playback
does) and dumps every Slippi event the game emits. This compares each native post-frame payload
against the original replay's post-frame payload byte for byte, per (frame, port, follower),
and names the first differing field.

Usage:
  python3 tools/sim_replay_check.py <orig.slp> <native.bin>            # one pair, verbose
  python3 tools/sim_replay_check.py --batch <slp_dir> [--exe sim/.../melee_linux] --iso melee.iso \
      [--n 200] [--jobs 24] [--out report.json]                         # run + check many
"""
import argparse, json, os, struct, subprocess, sys, tempfile, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

POST = 0x38
# Post-frame fields, offsets into the payload (command byte excluded = spec offset - 1).
POST_FIELDS = [
    (0x00, 4, "frame"), (0x04, 1, "port"), (0x05, 1, "follower"), (0x06, 1, "internal_char"),
    (0x07, 2, "action_state"), (0x09, 4, "pos_x"), (0x0D, 4, "pos_y"), (0x11, 4, "facing"),
    (0x15, 4, "percent"), (0x19, 4, "shield"), (0x1D, 1, "last_attack"), (0x1E, 1, "combo_count"),
    (0x1F, 1, "last_hit_by"), (0x20, 1, "stocks"), (0x21, 4, "action_frame"),
    (0x25, 5, "state_flags"), (0x2A, 4, "misc_as_hitstun"), (0x2E, 1, "airborne"),
    (0x2F, 2, "ground_id"), (0x31, 1, "jumps_left"), (0x32, 1, "lcancel_status"),
    (0x33, 1, "hurtbox_state"), (0x34, 4, "self_vel_air_x"), (0x38, 4, "self_vel_y"),
    (0x3C, 4, "kb_vel_x"), (0x40, 4, "kb_vel_y"), (0x44, 4, "self_vel_ground_x"),
    (0x48, 4, "hitlag"), (0x4C, 4, "animation_index"), (0x50, 2, "instance_hit_by"),
    (0x52, 2, "instance_id"),
]


def field_at(off):
    for o, n, name in POST_FIELDS:
        if o <= off < o + n:
            return name
    return f"byte_{off:#x}"


def slp_events(path):
    """Yield (command, payload) from a .slp's raw event stream."""
    d = Path(path).read_bytes()
    i = d.find(b"raw[$U#l")
    if i < 0:
        raise ValueError("no raw element")
    raw_len = struct.unpack(">I", d[i + 8:i + 12])[0]
    i += 12
    end = i + raw_len if raw_len else len(d)
    if d[i] != 0x35:
        raise ValueError("raw does not start with Event Payloads")
    n = d[i + 1]
    sizes = {d[i + 2 + 3 * k]: struct.unpack(">H", d[i + 3 + 3 * k:i + 5 + 3 * k])[0] for k in range((n - 1) // 3)}
    i += 1 + n
    while i < end:
        c = d[i]
        if c not in sizes:
            raise ValueError(f"unknown command {c:#x} at {i}")
        yield c, d[i + 1:i + 1 + sizes[c]]
        i += 1 + sizes[c]


def dump_events(path):
    d = Path(path).read_bytes()
    i = 0
    while i + 5 <= len(d):
        c = d[i]; n = struct.unpack("<I", d[i + 1:i + 5])[0]
        yield c, d[i + 5:i + 5 + n]
        i += 5 + n


def post_frames(events):
    out = {}
    for c, p in events:
        if c == POST and len(p) >= 6:
            out[(struct.unpack(">i", p[:4])[0], p[4], p[5])] = p
    return out


# The fields that are the game's physical state; a difference here (other than +0 vs -0) is a
# divergence. Everything else (counters, flags, signed zeros) is reported but not a divergence.
PHYSICS = [(0x07, 2, "action_state"), (0x09, 4, "pos_x"), (0x0D, 4, "pos_y"), (0x11, 4, "facing"),
           (0x15, 4, "percent"), (0x19, 4, "shield"), (0x20, 1, "stocks")]


def physics_diff(pa, pb):
    out = []
    for o, n, name in PHYSICS:
        x, y = pa[o:o + n], pb[o:o + n]
        if x != y and not (n == 4 and {x, y} <= {b"\x00\x00\x00\x00", b"\x80\x00\x00\x00"}):
            out.append(name)
    return out


def compare(orig_slp, native_bin):
    a = post_frames(slp_events(orig_slp))
    b = post_frames(dump_events(native_bin))
    keys = sorted(a)
    first_frame = keys[0][0] if keys else None
    last_frame = keys[-1][0] if keys else None
    missing = [k for k in keys if k not in b]
    mism = []
    field_counts = {}
    compared_bytes = None
    for k in keys:
        if k not in b:
            continue
        pa, pb = a[k], b[k]
        n = min(len(pa), len(pb))
        compared_bytes = n
        if pa[:n] != pb[:n]:
            offs = [o for o in range(n) if pa[o] != pb[o]]
            names = sorted({field_at(o) for o in offs})
            for nm in names:
                field_counts[nm] = field_counts.get(nm, 0) + 1
            mism.append((k, names))
    first_bad = mism[0] if mism else None
    # Divergence = any physics field differs (+0/-0 equal) or the native run lacks the frame.
    # Pre-match frames (< 0) are reported separately: the spawn animation carries a known 1-ulp
    # difference (melee-unlocked docs/replay-validation.md) that does not propagate.
    div_frames = [k for k in keys if k not in b or physics_diff(a[k], b[k])]
    match_div = [k for k in div_frames if k[0] >= 0]
    def describe(k):
        return {"frame": k[0], "port": k[1], "follower": k[2],
                "fields": physics_diff(a[k], b[k]) if k in b else ["missing"]}
    div = describe(match_div[0]) if match_div else None
    pre_div = len(div_frames) - len(match_div)
    return {
        "orig": str(orig_slp), "frames": [first_frame, last_frame], "player_frames": len(keys),
        "native_player_frames": len(b), "missing": len(missing),
        "first_missing": list(missing[0]) if missing else None,
        "mismatched_player_frames": len(mism),
        "first_mismatch": {"frame": first_bad[0][0], "port": first_bad[0][1], "follower": first_bad[0][2],
                           "fields": first_bad[1]} if first_bad else None,
        "first_divergence": div, "match_divergent_player_frames": len(match_div),
        "match_player_frames": sum(1 for k in keys if k[0] >= 0), "prematch_divergent_player_frames": pre_div,
        "field_counts": field_counts, "compared_payload_bytes": compared_bytes,
        "exact": not mism and not missing,
    }


def run_one(args):
    slp, exe, iso, tmp, idx = args
    tag = f"{idx:05d}_{Path(slp).stem}"   # unique even when replay basenames repeat
    dump = os.path.join(tmp, tag + ".bin")
    log = os.path.join(tmp, tag + ".log")
    t0 = time.time()
    p = subprocess.run([exe, "--iso", iso, "--headless", "--fast", "--frames", "40000",
                        "--card-dir", os.path.join(tmp, "card_" + tag),
                        "--replay", slp, "--replay-dump", dump, "--log-file", log],
                       cwd=Path(exe).resolve().parents[3], capture_output=True, text=True, timeout=600)
    wall = time.time() - t0
    stderr = p.stderr
    not_native = [l for l in stderr.splitlines() if "not implemented natively" in l]
    try:
        r = compare(slp, dump)
    except Exception as e:  # noqa: BLE001 - reported per replay
        r = {"orig": slp, "error": f"{type(e).__name__}: {e}", "exact": False}
    r.update(rc=p.returncode, wall_s=round(wall, 2), not_native_codes=not_native)
    for f in (dump, log):
        try:
            os.remove(f)
        except OSError:
            pass
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pair", nargs="*")
    ap.add_argument("--batch")
    ap.add_argument("--exe", default="sim/melee-unlocked/build-linux/port/linux/melee_linux")
    ap.add_argument("--iso", default="melee.iso")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.pair:
        print(json.dumps(compare(a.pair[0], a.pair[1]), indent=1))
        return
    import random
    files = sorted(str(p.absolute()) for p in Path(a.batch).glob("*.slp"))
    random.Random(a.seed).shuffle(files)
    files = files[:a.n]
    iso = str(Path(a.iso).resolve())
    exe = str(Path(a.exe).resolve())
    with tempfile.TemporaryDirectory() as tmp, ProcessPoolExecutor(a.jobs) as ex:
        results = []
        for i, r in enumerate(ex.map(run_one, [(f, exe, iso, tmp, i) for i, f in enumerate(files)])):
            results.append(r)
            tag = "EXACT" if r.get("exact") else ("ERROR " + r["error"] if "error" in r else
                  ("PHYSICS-EXACT" if not r["first_divergence"] else
                   f"DIVERGES@{r['first_divergence']['frame']} {r['first_divergence']['fields']} "
                   f"({r['match_divergent_player_frames']}/{r['match_player_frames']} match player-frames)") +
                  f"  (bytes differ on {r['mismatched_player_frames']}/{r['player_frames']})")
            print(f"[{i + 1}/{len(files)}] {Path(r['orig']).name}  {tag}  ({r.get('wall_s')}s)", flush=True)
    n_exact = sum(r.get("exact", False) for r in results)
    pf = sum(r.get("player_frames", 0) for r in results)
    bad = sum(r.get("mismatched_player_frames", 0) + r.get("missing", 0) for r in results)
    n_phys = sum(1 for r in results if "error" not in r and not r["first_divergence"])
    mpf = sum(r.get("match_player_frames", 0) for r in results)
    mdv = sum(r.get("match_divergent_player_frames", 0) for r in results)
    print(f"\n{n_exact}/{len(results)} replays byte-exact, {n_phys}/{len(results)} physics-exact; "
          f"{pf - bad}/{pf} player-frames byte-exact; match-phase physics divergence on {mdv}/{mpf} player-frames")
    if a.out:
        Path(a.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
