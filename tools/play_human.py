#!/usr/bin/env python3
"""Play a MIMIC checkpoint yourself: your GameCube controller (Wii U / Mayflash adapter in
Wii U mode) on one port, the bot on the other, realtime in Dolphin. Same inference path as
tools/play.py (PlayerState, build_frame / build_frame_p2, decode_and_press).

You pick your character and the stage and press START; the bot locks in --character on its
port and skips the results screen. Matches repeat until you close Dolphin or press Ctrl-C.

  python3 tools/play_human.py --ckpt hf_eval/fox-master/model.pt --data-dir hf_eval/fox-master \
      --dolphin-path emulator/squashfs-root/usr/bin/dolphin-emu --iso-path melee.iso
"""
import argparse, logging, signal, sys
from pathlib import Path

import melee

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from inference_utils import (PlayerState, build_frame, build_frame_p2, decode_and_press,  # noqa: E402
                             load_inference_context, load_mimic_model, set_cond_temperature)
from play import _make_ctx, _safe_stop  # noqa: E402

log = logging.getLogger("play_human")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", required=True, help="dir with the checkpoint's norm / map JSONs")
    ap.add_argument("--dolphin-path", required=True)
    ap.add_argument("--iso-path", required=True)
    ap.add_argument("--character", default="FOX", help="the bot's character")
    ap.add_argument("--costume", type=int, default=0)
    ap.add_argument("--human-port", type=int, default=1)
    ap.add_argument("--bot-port", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--gfx-backend", default="OGL")
    ap.add_argument("--replay-dir", default=str(ROOT / "replays_human"))
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    if a.human_port == a.bot_port:
        raise SystemExit("--human-port and --bot-port must differ")

    model, cfg = load_mimic_model(a.ckpt, a.device)
    set_cond_temperature(model, a.temperature)
    ctx = _make_ctx(load_inference_context(a.data_dir), cfg)
    # build_frame is the lower port's perspective, build_frame_p2 the higher port's
    build = build_frame if a.bot_port < a.human_port else build_frame_p2

    Path(a.replay_dir).mkdir(parents=True, exist_ok=True)
    console = melee.Console(path=a.dolphin_path, is_dolphin=True, tmp_home_directory=True,
                            copy_home_directory=False, blocking_input=True, online_delay=0,
                            setup_gecko_codes=True, fullscreen=False, gfx_backend=a.gfx_backend,
                            save_replays=True, replay_dir=a.replay_dir)
    bot = melee.Controller(console=console, port=a.bot_port, type=melee.ControllerType.STANDARD)
    melee.Controller(console=console, port=a.human_port, type=melee.ControllerType.GCN_ADAPTER)
    console.run(iso_path=a.iso_path)
    console.connect()
    bot.connect()
    log.info("connected: you on port %d (adapter), bot %s on port %d", a.human_port, a.character, a.bot_port)

    def stop(*_):
        _safe_stop(console)
        sys.exit(0)
    signal.signal(signal.SIGINT, stop)

    menu, char = melee.MenuHelper(), melee.Character[a.character]
    state, in_game = None, False
    while True:
        gs = console.step()
        if gs is None:
            continue
        if gs.menu_state not in (melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH):
            if in_game:
                ps = {p: int(s.stock) for p, s in gs.players.items()} if gs.players else {}
                log.info("match over %s", ps)
                in_game = False
            if gs.menu_state == melee.Menu.STAGE_SELECT:
                bot.release_all()        # you pick the stage
            else:                        # bot locks in its character; you press START
                menu.menu_helper_simple(gs, bot, char, melee.Stage.BATTLEFIELD, cpu_level=0,
                                        autostart=False, costume=a.costume)
            bot.flush()
            continue
        if not in_game:
            in_game = True
            state = PlayerState(model, cfg.max_seq_len, a.device, ctx=ctx)
            log.info("match start: stage %s", gs.stage.name)
        if len(gs.players) < 2:
            continue
        frame = build(gs, state.prev_sent, ctx)
        if frame is None:
            continue
        state.push_frame(frame)
        sent, _, _ = decode_and_press(bot, state.predict(), state.prev_sent, temperature=a.temperature)
        state.prev_sent = sent


if __name__ == "__main__":
    main()
