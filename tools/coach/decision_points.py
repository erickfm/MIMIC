#!/usr/bin/env python3
"""Decision points of one player in a .slp: the frames where they can act, read from the game's
own state as Slippi records it.

"Can act" follows the Training Lab's rule (sim melee-unlocked shim/mu_lab.c, lab_can_act) on the
post-frame fields:
  - not in hitlag (post.hitlag > 0) and not in hitstun (state flags 4, bit 0x02 = fp+0x221C:6);
  - and either a free state (standing, walking, dashing, running, crouching, jumping, falling,
    tumble, shield; plus first-jump rise, shield start, powershield, teeter, ledge hang, lying
    down, holding a grab and respawn platform, which the Lab omits), or normal landing once its
    animation frame reaches the landing lag,
  - or the move's script has allowed interrupts (IASA): state flags 1, bit 0x80 = fp+0x2218:0,
    `allow_interrupt`. The game sets that bit from a move's script but clears it only when some
    attacks start (decomp: ftaction.c ftAction_80071950 sets it; ftCo_AttackAir.c etc. clear
    it), so it often carries over from the previous move. It counts only after it turns on
    during the current move.

A window is a run of can-act frames. Its first frame is the decision point; the player then
moves through free states and commits when they enter a move that ends the window. Each window
records how it started (the move before it), the free-state path, the delay to commit and the
committed move.

Not covered: choices a move allows through its own code rather than the IASA bit. Measured on
50 diamond Fox games (2026-11-04): 23.5% of move changes that coincide with a press start from a
frame this rule calls can't-act. The large groups: jump-squat (short/full hop and direction,
wavedash, jump-cancel grab / up-smash / shine), shine jump-cancel (ground and air), run-brake
into crouch, platform-drop aerials, shield-drop, and techs (input during hitstun). Many of the
rest are presses that happen to coincide with a natural end (an L-cancel press on landing).
`--misses` lists them for one game.

  python3 tools/coach/decision_points.py GAME.slp --port 0 [--from F --to F] [--misses]
"""
import argparse, collections, json, sys

import numpy as np
from peppi_py import read_slippi

FREE = set(range(14, 22)) | set(range(25, 35)) | {38, 39, 40, 179} \
    | {13, 178, 182, 184, 192, 216, 246, 253}
# Lab list: 14-21 standing..running, 27-34 aerial jumps + falls, 38 tumble, 39-40 crouch,
# 179 shield. Added, each with an unconditional interrupt function in the decomp: 13 respawn
# platform, 178 shield start and 182 powershield (ftCo_GuardOn_IASA / ftCo_GuardReflect_IASA:
# out-of-shield options), 184/192 lying down (getup options), 216 holding a grab (pummel or
# throw), 246 teeter, 253 ledge hang, and
# 25-26 first-jump rise (aerials from the first frame).
LANDING, LANDING_LAG = 42, 3
# Normal landing interrupts once its animation frame reaches the character's normal landing lag
# (ftCo_Landing_IASA). The lag is a character attribute from the ISO, not in the .slp; 3 is
# Fox's, measured from replays (state_age at which players leave LANDING).
OPP_ENDS = ("DAMAGE_", "GRABBED", "THROWN_", "CAPTURE", "DEAD_", "SHIELD_STUN", "GRAB_PULL")
IASA_BIT, HITSTUN_BIT = 0x80, 0x02


def action_name(s):
    try:
        import melee
        return melee.Action(int(s)).name
    except Exception:
        return str(int(s))


def load(path, port):
    g = read_slippi(str(path))
    po = g.frames.ports[port].leader.post
    st = np.asarray(po.state, dtype=np.int64)
    f1 = np.asarray(po.state_flags[0], dtype=np.uint8)
    f4 = np.asarray(po.state_flags[3], dtype=np.uint8)
    hl = np.asarray(po.hitlag, dtype=np.float32) if po.hitlag is not None else np.zeros(len(st), np.float32)
    frames = np.asarray(g.frames.id, dtype=np.int64)
    age = np.asarray(po.state_age, dtype=np.float32)
    # Rollback re-records a frame; keep each frame id's last copy, in order.
    last = {f: i for i, f in enumerate(frames)}
    keep = np.array(sorted(last.values()))
    return g, frames[keep], st[keep], f1[keep], f4[keep], hl[keep], age[keep], keep


def can_act(st, f1, f4, hl, age):
    n = len(st)
    iasa = np.zeros(n, bool)
    seen_clear = False
    for t in range(n):
        if t == 0 or st[t] != st[t - 1]:
            seen_clear = False
        bit = bool(f1[t] & IASA_BIT)
        if not bit:
            seen_clear = True
        iasa[t] = bit and seen_clear
    free = np.isin(st, list(FREE)) | ((st == LANDING) & (age >= LANDING_LAG))
    return (hl <= 0) & ((f4 & HITSTUN_BIT) == 0) & (free | iasa), iasa


def windows(frames, st, ok):
    out, t, n = [], 0, len(st)
    while t < n:
        if not ok[t]:
            t += 1
            continue
        a = t
        while t < n and ok[t]:
            t += 1
        path = [action_name(st[a])]
        for k in range(a + 1, t):
            if st[k] != st[k - 1]:
                path.append(action_name(st[k]))
        out.append(dict(frame=int(frames[a]), idx=a, length=t - a,
                        came_from=action_name(st[a - 1]) if a else None,
                        path=path,
                        commit=action_name(st[t]) if t < n else None,
                        ended_by=("opponent" if t < n and action_name(st[t]).startswith(OPP_ENDS)
                                  else "player")))
    return out


def pressed(g, port, keep):
    """Per frame: a new physical button press or a stick moved >= 0.5 since the last frame
    (pre-frame inputs; the post-frame state of frame t already reflects frame t's input)."""
    pre = g.frames.ports[port].leader.pre
    b = np.asarray(pre.buttons_physical, dtype=np.int64)
    x, y = np.asarray(pre.joystick.x, np.float32), np.asarray(pre.joystick.y, np.float32)
    cx, cy = np.asarray(pre.cstick.x, np.float32), np.asarray(pre.cstick.y, np.float32)
    b, x, y, cx, cy = (v[keep] for v in (b, x, y, cx, cy))
    out = np.zeros(len(b), bool)
    out[1:] = (((b[1:] & ~b[:-1]) & 0x0F70) != 0) | (np.hypot(np.diff(x), np.diff(y)) >= 0.5) \
        | (np.hypot(np.diff(cx), np.diff(cy)) >= 0.5)
    return out


def misses(g, port, keep, st, ok):
    """Move changes the player caused (a new press or stick move on that frame or the one
    before) from a frame the rule calls can't-act."""
    pr = pressed(g, port, keep)
    c = collections.Counter()
    for t in range(1, len(st)):
        if st[t] != st[t - 1] and not ok[t - 1] and (pr[t] or pr[t - 1]):
            c[(action_name(st[t - 1]), action_name(st[t]))] += 1
    return c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("slp")
    ap.add_argument("--port", type=int, required=True, help="0-based port")
    ap.add_argument("--from", dest="lo", type=int, default=None)
    ap.add_argument("--to", dest="hi", type=int, default=None)
    ap.add_argument("--misses", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    g, frames, st, f1, f4, hl, age, keep = load(a.slp, a.port)
    ok, _ = can_act(st, f1, f4, hl, age)
    ws = [w for w in windows(frames, st, ok)
          if (a.lo is None or w["frame"] >= a.lo) and (a.hi is None or w["frame"] <= a.hi)]
    if a.json:
        json.dump(ws, sys.stdout)
        return
    for w in ws:
        print(f"{w['frame']:6d}  {w['length']:4d}f  from {str(w['came_from']):24s} "
              f"{' > '.join(w['path'])[:60]:60s} -> {w['commit']}"
              + ("  (opponent)" if w["ended_by"] == "opponent" else ""))
    print(f"{len(ws)} windows; can act on {ok.mean():.1%} of frames")
    if a.misses:
        print("\nmove changes the player made while the rule said they couldn't act (top 25):")
        for (x, y), n in misses(g, a.port, keep, st, ok).most_common(25):
            print(f"  {n:4d}  {x} -> {y}")


if __name__ == "__main__":
    main()
