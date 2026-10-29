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
                continue
            if gs.frame == dolphin_lib.INITIAL_FRAME:
                game += 1
                mimic_port = 1 if game % 2 == 1 else 2
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
