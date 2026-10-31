"""MIMIC vs Phillip (vladfi1/slippi-ai) in one Dolphin: N games, alternating ports, rotating stages.

Runs in slippi-ai's venv (its libmelee fork, onnxruntime) with torch + pandas added, and imports
MIMIC's inference path (tools/inference_utils.py) unchanged. Phillip runs from a released .onnx
(https://drive.google.com/drive/folders/1U860HfE5FB2VKTN56TTkDffhZnu3B45J) through slippi-ai's own
eval_lib.Agent, with console_delay 0 (so its trained 18-frame delay is the whole delay). MIMIC runs
fp32 with no delay, as in production. Both pads are flushed once per frame by console.step
(decode_and_press's own flush is suppressed; a second flush would queue a stale input).

  ~/projects/slippi-ai/.venv/bin/python tools/vs_phillip.py --mimic runs/NAME/policy \
      --phillip ~/projects/slippi-ai-models/onnx/fox_d18_ditto_v4.onnx --games 20 --out DIR

--headless uses the ExiAI build (emulator_ss, fast-forward); otherwise realtime with a window.
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import melee
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
from inference_utils import (  # noqa: E402
    load_inference_context, load_mimic_model, build_frame, build_frame_p2,
    PlayerState, decode_and_press,
)
from play import _make_ctx  # noqa: E402

from slippi_ai import dolphin as dolphin_lib, eval_lib  # noqa: E402

ISO = "/home/erick/Documents/melee/Super Smash Bros. Melee (USA) (En,Ja) (Rev 2).iso"
STAGES = ["FINAL_DESTINATION", "BATTLEFIELD", "DREAMLAND"]


class NoFlush:
    """A Controller whose flush() does nothing: console.step flushes every pad once per frame."""

    def __init__(self, ctrl):
        self._ctrl = ctrl

    def flush(self):
        pass

    def __getattr__(self, name):
        return getattr(self._ctrl, name)


class CostumeMenu:
    """Wraps libmelee's MenuHelper to give each agent a fixed costume whichever port it plays on.

    libmelee can't read costumes on the offline character screen, so this works from the costume each
    game actually started with (Game Start event): before the next game it presses Y `pending[port]`
    times with the hand over the Fox portrait (one press = previous costume), holding autostart until
    done, one port at a time. Presses start SETTLE frames after both coins are down: Melee ignores Y
    just after a coin is placed (seen on the first character screen, where the coins go down right
    before the presses). Checked in Dolphin 2026-10-29: 4 of 4 games started in the wanted costumes.
    """

    SETTLE = 60
    PORT_GAP = 10   # Y on both ports in the same frame lost all four presses of a 2+2 plan; keep ports apart

    def __init__(self, helper, n_costumes=4):
        self._helper = helper
        self.n = n_costumes
        self.pending = {}

    def _press(self, cur, port):
        """One Y press: previous costume, skipping any the other Fox wears (Melee won't allow duplicates).
        The direction was fit to 19 of 19 game-to-game costume changes in the gm run (run4.log)."""
        other = {c for p, c in cur.items() if p != port}
        c = (cur[port] - 1) % self.n
        while c in other:
            c = (c - 1) % self.n
        return c

    def plan(self, current, target):
        """current/target: {port: costume}. Queue (port, presses) steps, one port at a time, that turn
        current into target; a swap goes through a free costume."""
        cur, self.steps = dict(current), []
        while cur != target:
            free = [p for p in target if cur[p] != target[p] and target[p] not in
                    {c for q, c in cur.items() if q != p}]
            p = free[0] if free else min(target)
            goal = target[p] if free else next(c for c in range(self.n) if c not in cur.values())
            n = 0
            while cur[p] != goal:
                cur[p] = self._press(cur, p)
                n += 1
            self.steps.append((p, n))
        self.pending = dict(self.steps[:1])
        self.css_frames = 0
        self.coins_down_at = None
        self.next_press = 0

    def _advance(self):
        self.steps = self.steps[1:]
        self.pending = dict(self.steps[:1])

    def menu_helper_simple(self, gs, controller, **kw):
        port = controller.port
        if gs.menu_state == melee.Menu.CHARACTER_SELECT:
            self.css_frames += 1
            if self.css_frames > 2400 and any(self.pending.values()):   # ~20 s per port: give up, don't hang
                logging.warning(f"costume presses not done after {self.css_frames // 2} frames: {self.steps}")
                self.steps, self.pending = [], {}
        if any(self.pending.values()):
            kw["autostart"] = False
        st = gs.players.get(port)
        foxes = [gs.players.get(p) for p in (1, 2)]
        if (gs.menu_state == melee.Menu.CHARACTER_SELECT and all(
                s is not None and s.character == kw["character_selected"] and s.coin_down for s in foxes)):
            if self.coins_down_at is None:
                self.coins_down_at = gs.frame
        else:
            self.coins_down_at = None
        if (self.pending.get(port) and self.coins_down_at is not None
                and gs.frame - self.coins_down_at >= self.SETTLE):
            idx = melee.enums.from_internal(st.character)
            row, col = 2 - idx // 9, idx % 9 + (idx // 9 == 2)
            tx, ty = -32.5 + 3.5 + col * 7.0, 1 + 3.5 + row * 7.0
            dx, dy = tx - st.cursor.x, ty - st.cursor.y
            if abs(dx) > 1.5 or abs(dy) > 1.5:
                controller.release_all()
                controller.tilt_analog(melee.Button.BUTTON_MAIN, 0.5 + 0.3 * (dx > 1.5) - 0.3 * (dx < -1.5),
                                       0.5 + 0.3 * (dy > 1.5) - 0.3 * (dy < -1.5))
            elif gs.frame >= self.next_press and not controller.prev.button[melee.Button.BUTTON_Y]:
                controller.release_all()
                controller.press_button(melee.Button.BUTTON_Y)
                self.pending[port] -= 1
                self.next_press = gs.frame + 2
                if not self.pending[port]:
                    self._advance()
                    self.next_press = gs.frame + self.PORT_GAP
            else:
                controller.release_all()
            return
        self._helper.menu_helper_simple(gs, controller, **kw)


def result_of(gs, mimic_port):
    """(mimic stocks, phillip stocks, mimic %, phillip %, winner) at a game's last frame."""
    opp = 2 if mimic_port == 1 else 1
    m, p = gs.players[mimic_port], gs.players[opp]
    if m.stock != p.stock:
        w = "mimic" if m.stock > p.stock else "phillip"
    elif m.percent != p.percent:
        w = "mimic" if m.percent < p.percent else "phillip"
    else:
        w = "draw"
    return dict(mimic_stocks=int(m.stock), phillip_stocks=int(p.stock), mimic_pct=round(float(m.percent), 1),
                phillip_pct=round(float(p.percent), 1), winner=w)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mimic", required=True, type=Path, help="policy dir (model.pt + norm JSONs)")
    ap.add_argument("--phillip", required=True, type=Path, help="Phillip .onnx")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--iso", default=ISO)
    ap.add_argument("--mimic-costume", type=int, default=3, help="Fox: 0 white, 1 red, 2 blue, 3 green")
    ap.add_argument("--phillip-costume", type=int, default=2)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    stages = [melee.Stage[s] for s in args.stages.split(",")]

    device = torch.device("cuda")
    model, cfg = load_mimic_model(str(args.mimic / "model.pt"), device)
    ctx = _make_ctx(load_inference_context(args.mimic), cfg)

    phillip = {port: eval_lib.build_agent(port=port, opponent_port=3 - port, path=str(args.phillip),
                                          console_delay=0, async_inference=False)
               for port in (1, 2)}
    for a in phillip.values():
        a.start()

    path = REPO / ("emulator_ss/Binaries/dolphin-emu" if args.headless else "emulator/squashfs-root/usr/bin/dolphin-emu")
    dolphin = dolphin_lib.Dolphin(
        path=str(path), iso=args.iso,
        players={1: dolphin_lib.AI(character=melee.Character.FOX),
                 2: dolphin_lib.AI(character=melee.Character.FOX)},
        stage=stages[0], online_delay=0, blocking_input=True, headless=args.headless,
        save_replays=True, replay_dir=str(args.out / "replays"), infinite_time=False,
    )
    for port, a in phillip.items():
        a.set_controller(dolphin.controllers[port])
    menu = dolphin.menu_helper = CostumeMenu(dolphin.menu_helper)

    def want(mimic_port):
        return {mimic_port: args.mimic_costume, 3 - mimic_port: args.phillip_costume}

    # A fresh Dolphin starts P1 red, P2 white (seen in every earlier run); corrected from game 1 on.
    menu.plan({1: 1, 2: 0}, want(1))

    results, game, last, mimic_port, state, t0 = [], 0, None, 1, None, None
    log = open(args.out / "games.jsonl", "a")
    try:
        for gs in dolphin.iter_gamestates(skip_menu_frames=False):
            if dolphin_lib.is_menu_state(gs):
                if last is not None:   # a game just ended
                    r = dict(game=game, stage=last.stage.name, mimic_port=mimic_port,
                             frames=int(last.frame), wall_s=round(time.time() - t0, 1),
                             **result_of(last, mimic_port))
                    results.append(r)
                    log.write(json.dumps(r) + "\n"); log.flush()
                    w = sum(x["winner"] == "mimic" for x in results)
                    logging.info(f"game {game}: {r['winner']} ({r['mimic_stocks']}-{r['phillip_stocks']} stocks, "
                                 f"{r['stage']}, MIMIC port {mimic_port}); MIMIC {w}-{len(results) - w}")
                    last = None
                    if game == args.games:
                        break
                    dolphin.stage = stages[(game // 2) % len(stages)]
                    menu.plan(costumes, want(1 if (game + 1) % 2 == 1 else 2))
                continue
            if gs.frame == dolphin_lib.INITIAL_FRAME:
                game += 1
                mimic_port = 1 if game % 2 == 1 else 2
                costumes = {p: int(gs.players[p].costume) for p in (1, 2)}
                ok = costumes == want(mimic_port)
                logging.info(f"game {game} costumes {costumes} (MIMIC port {mimic_port}, "
                             f"want {want(mimic_port)}){'' if ok else ' MISMATCH'}")
                state = PlayerState(model, cfg.max_seq_len, device, ctx=ctx)
                t0 = time.time()
            if len(gs.players) < 2 or state is None:
                continue
            phillip[3 - mimic_port].step(gs)
            frame = (build_frame if mimic_port == 1 else build_frame_p2)(gs, state.prev_sent, ctx)
            if frame is not None:
                state.push_frame(frame)
                state.prev_sent, _, _ = decode_and_press(
                    NoFlush(dolphin.controllers[mimic_port]), state.predict(), state.prev_sent)
            last = gs
    finally:
        w = sum(x["winner"] == "mimic" for x in results)
        d = sum(x["winner"] == "draw" for x in results)
        summary = dict(mimic=str(args.mimic), phillip=str(args.phillip), games=len(results),
                       mimic_wins=w, phillip_wins=len(results) - w - d, draws=d)
        (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
        logging.info(f"SUMMARY {summary}")
        for a in phillip.values():
            a.stop()
        dolphin.stop()


if __name__ == "__main__":
    main()
