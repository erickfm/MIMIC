#!/usr/bin/env python3
"""How does Melee's offline character select assign and cycle costumes? (diagnostic for tools/play.py)

libmelee reports costumes on the offline CSS as 0 and its menu helper only applies `costume` on the
Slippi online CSS, so play.py can't see or set colours before the game starts. This probe drives
both ports to the same character, presses Y a given number of times on one port once both are
locked in, starts the game, and prints the costume each port got (read from the game-start event).

  python3 tools/costume_probe.py --trials "none,1:1,1:2,1:3,2:1,2:2"     # PORT:N_PRESSES
"""
import argparse, json, sys, time
from pathlib import Path

import melee

ROOT = Path(__file__).resolve().parents[1]


def trial(a, port, presses):
    console = melee.Console(path=a.dolphin_path, is_dolphin=True, tmp_home_directory=True,
                            copy_home_directory=False, blocking_input=False, online_delay=0,
                            setup_gecko_codes=True, fullscreen=False, gfx_backend=a.gfx_backend,
                            disable_audio=True, save_replays=False)
    ctrl = {p: melee.Controller(console=console, port=p, type=melee.ControllerType.STANDARD) for p in (1, 2)}
    console.run(iso_path=a.iso)
    console.connect()
    for c in ctrl.values():
        c.connect()
    menu = {p: melee.MenuHelper() for p in (1, 2)}
    char = melee.Character[a.character]
    locked_at, presses_done, press_frame, t0 = {}, 0, None, time.time()
    out = None
    try:
        while time.time() - t0 < a.timeout:
            gs = console.step()
            if gs is None:
                continue
            if gs.menu_state in (melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH):
                out = {p: int(gs.players[p].costume) for p in (1, 2) if p in gs.players}
                break
            css = gs.menu_state == melee.Menu.CHARACTER_SELECT
            if css:
                for p in (1, 2):
                    ps = gs.players.get(p)
                    if ps is not None and ps.character is char and ps.coin_down and p not in locked_at:
                        locked_at[p] = gs.frame
            both = css and len(locked_at) == 2
            settled = both and gs.frame - max(locked_at.values()) > 30
            pressing = settled and port is not None and presses_done < presses
            start_ok = settled and not pressing and (port is None or gs.frame - (press_frame or 0) > 30)
            menu[1].menu_helper_simple(gs, ctrl[1], char, melee.Stage.FINAL_DESTINATION, autostart=False)
            menu[2].menu_helper_simple(gs, ctrl[2], char, melee.Stage.FINAL_DESTINATION,
                                       autostart=start_ok if css else True)
            if pressing:
                c = ctrl[port]
                c.release_all()
                if press_frame is None or gs.frame - press_frame >= 6:
                    c.press_button(melee.Button.BUTTON_Y)
                    press_frame = gs.frame
                    presses_done += 1
            for c in ctrl.values():
                c.flush()
    finally:
        console.stop()
    first = min(locked_at, key=locked_at.get) if len(locked_at) == 2 else None
    return {"port": port, "presses": presses, "costumes": out, "locked_first": first,
            "locked_at": {p: int(f) for p, f in locked_at.items()}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", default="none,1:1,1:2,1:3,2:1,2:2")
    ap.add_argument("--character", default="FOX")
    ap.add_argument("--dolphin-path", default=str(ROOT / "emulator/squashfs-root/usr/bin/dolphin-emu"))
    ap.add_argument("--iso", default=str(ROOT / "melee.iso"))
    ap.add_argument("--gfx-backend", default="")
    ap.add_argument("--timeout", type=float, default=120)
    a = ap.parse_args()
    for t in a.trials.split(","):
        port, n = (None, 0) if t == "none" else tuple(int(x) for x in t.split(":"))
        r = trial(a, port, n)
        print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
