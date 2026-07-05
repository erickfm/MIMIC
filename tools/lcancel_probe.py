"""Validate the L-cancel / landing-lag analysis on a few replays before scaling.
For each Fox aerial-landing event (entry into NAIR..DAIR_LANDING = 70..74),
print: move, l_cancel flag, measured lag (frames in the landing state), the
state it exits to (airborne = slid off?), and position."""
import sys
import numpy as np
from melee import Action
from rlvr.state.peppi_adapter import Replay

LANDING_STATES = {70: "NAIR", 71: "FAIR", 72: "BAIR", 73: "UAIR", 74: "DAIR"}
# airborne states a landing could exit into if it slid off a ledge:
# jumps (25-28), falls (29-34), special-fall (36-37), tumble (38), aerials (65-69)
AIRBORNE = set(range(25, 35)) | {36, 37, 38} | set(range(65, 70))

def analyze(path, max_events=12):
    r = Replay(path)
    fox_idxs = [i for i, c in enumerate(r.player_characters) if c == 1]
    print(f"\nFILE {path.split('/')[-1]}  fox_ports={[r.player_ports[i] for i in fox_idxs]}")
    for i in fox_idxs:
        st = np.asarray(r._post[i]["state"]).astype(int)
        lc = np.asarray(r._post[i]["l_cancel"]).astype(int)
        px = np.asarray(r._post[i]["position_x"]).astype(float)
        py = np.asarray(r._post[i]["position_y"]).astype(float)
        n = len(st)
        ev = 0
        t = 1
        while t < n and ev < max_events:
            if st[t] in LANDING_STATES and st[t-1] != st[t]:
                move = LANDING_STATES[st[t]]
                # run length in this landing state
                j = t
                while j < n and st[j] == st[t]:
                    j += 1
                lag = j - t
                exit_state = int(st[j]) if j < n else -1
                exit_name = Action(exit_state).name if exit_state >= 0 else "END"
                airborne = exit_state in AIRBORNE
                flag = int(lc[t])  # l_cancel flag at landing entry
                print(f"  {move:4s} lcancel={flag} lag={lag:2d}f exit={exit_name:18s}"
                      f" slid_off={airborne}  pos=({px[t]:6.1f},{py[t]:6.1f})")
                ev += 1
                t = j
            else:
                t += 1

if __name__ == "__main__":
    for p in sys.argv[1:]:
        analyze(p)
