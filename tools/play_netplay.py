#!/usr/bin/env python3
"""Run a MIMIC checkpoint against a human via Slippi Online Direct Connect.

Usage:
    python tools/play_netplay.py \
      --checkpoint checkpoints/falco-7class-v2-full_best.pt \
      --dolphin-path /path/to/dolphin-emu \
      --iso-path /path/to/melee.iso \
      --data-dir data/falco_v2 \
      --character FALCO \
      --connect-code ERIK#456

The script launches Dolphin, auto-navigates to Slippi Online → Direct Connect,
enters the opponent's connect code, joins the match, and plays the bot.

When used by the Discord bot, the script prints machine-readable final lines:
    RESULT: win|loss|disconnect|no-opponent|timeout
    REPLAY: <absolute path to saved .slp>

Requirements (one-time setup):
    - Slippi account with valid user.json (created via Slippi Launcher)
    - Gecko codes enabled (libmelee sets this via setup_gecko_codes=True)
    - Melee 1.02 NTSC ISO
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import logging
import os
import signal
import time

import melee
import torch

from tools.inference_utils import (
    load_mimic_model, load_inference_context,
    build_frame, build_frame_p2,
    PlayerState, decode_and_press,
)

log = logging.getLogger("play_netplay")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    stream=sys.stderr,  # stdout is reserved for RESULT/REPLAY machine lines
)

parser = argparse.ArgumentParser(description="MIMIC vs human via Slippi Online Direct Connect")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--dolphin-path", required=True)
parser.add_argument("--iso-path", required=True)
parser.add_argument("--data-dir", required=True)
parser.add_argument("--character", default="FALCO",
                    help="Character the bot plays (FOX, FALCO, CPTFALCON, LUIGI)")
parser.add_argument("--stage", default="FINAL_DESTINATION")
parser.add_argument("--connect-code", required=True,
                    help="Opponent's Slippi direct connect code (e.g. ERIK#456)")
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--costume", type=int, default=0)
parser.add_argument("--no-opponent-timeout", type=float, default=120.0,
                    help="Seconds to wait for the opponent to connect before giving up")
parser.add_argument("--match-timeout", type=float, default=900.0,
                    help="Maximum seconds for a single match (15 min default)")
args = parser.parse_args()


def emit_result(result: str, replay_path: str = ""):
    """Print machine-readable final status for the Discord bot parser."""
    print(f"RESULT: {result}", flush=True)
    if replay_path:
        print(f"REPLAY: {replay_path}", flush=True)


# ---- Load model + inference context ----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    model, cfg = load_mimic_model(args.checkpoint, DEVICE)
    log.info("Loaded %s: %d params, encoder=%s",
             args.checkpoint,
             sum(p.numel() for p in model.parameters()),
             cfg.encoder_type)
    ctx = load_inference_context(args.data_dir)
except Exception as e:
    log.exception("Failed to load model/context")
    emit_result("failed")
    sys.exit(1)

BOT_CHAR = melee.Character[args.character]
STAGE = melee.Stage[args.stage]

# ---- Dolphin + controller setup ----
replay_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "replays")
os.makedirs(replay_dir, exist_ok=True)

console = melee.Console(
    path=args.dolphin_path,
    is_dolphin=True,
    tmp_home_directory=True,
    copy_home_directory=True,  # copy user's Slippi home so user.json is available
    blocking_input=True,
    online_delay=2,  # matches standard Slippi direct connect default
    setup_gecko_codes=True,
    fullscreen=False,
    gfx_backend="",
    disable_audio=False,
    use_exi_inputs=False,
    enable_ffw=False,
    save_replays=True,
    replay_dir=replay_dir,
)

# Single bot controller — opponent will be assigned to the other port via netplay
bot_ctrl = melee.Controller(
    console=console,
    port=1,
    type=melee.ControllerType.STANDARD,
)

console.run(iso_path=args.iso_path)
console.connect()
bot_ctrl.connect()
log.info("Connected to Dolphin. Target opponent code: %s", args.connect_code)

menu_helper = melee.MenuHelper()

def shutdown_handler(*_):
    try:
        console.stop()
    except Exception:
        pass
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)

# ---- Main loop: menu navigation → match → exit ----
player_state: PlayerState = None
detected_port: int = 0
match_started = False
match_start_time = None
last_known_state = "UNKNOWN"
start_time = time.time()
last_in_game_time = None

# Track stocks to detect game end robustly
initial_stocks = None
final_result = "timeout"
replay_path_out = ""

# Figure out which build_frame to call based on detected port
def _build_frame(gs, prev_sent, ctx_):
    if detected_port <= 1:
        return build_frame(gs, prev_sent, ctx_)
    else:
        return build_frame_p2(gs, prev_sent, ctx_)

try:
    while True:
        gs = console.step()
        if gs is None:
            continue

        now = time.time()

        # Menu / lobby navigation — let MenuHelper drive until we reach IN_GAME
        if gs.menu_state not in (melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH):
            if match_started:
                # We were in a game and now we're not — match ended
                log.info("Game ended (menu=%s)", gs.menu_state)
                break

            # Connect-code timeout: if we've been in menus for too long without
            # the opponent showing up, give up.
            if (now - start_time) > args.no_opponent_timeout:
                log.warning("No opponent connected within %ds, giving up", args.no_opponent_timeout)
                final_result = "no-opponent"
                break

            menu_helper.menu_helper_simple(
                gs,
                bot_ctrl,
                BOT_CHAR,
                STAGE,
                connect_code=args.connect_code,
                costume=args.costume,
                autostart=True,
                swag=False,
            )
            continue

        # --- IN_GAME ---
        if not match_started:
            match_started = True
            match_start_time = now
            last_known_state = "IN_GAME"
            # Detect which port the bot ended up on
            detected_port = melee.gamestate.port_detector(gs, BOT_CHAR, args.costume)
            if detected_port == 0:
                # Couldn't find our character yet; try again next frame
                match_started = False
                continue
            log.info("Match started. Bot is on port %d", detected_port)
            player_state = PlayerState(model, cfg.max_seq_len, DEVICE, ctx=ctx)
            initial_stocks = None

        last_in_game_time = now

        # Match timeout
        if (now - match_start_time) > args.match_timeout:
            log.warning("Match exceeded %ds timeout", args.match_timeout)
            final_result = "timeout"
            break

        # Build frame from bot's perspective
        frame = _build_frame(gs, player_state.prev_sent, ctx)
        if frame is None:
            continue

        player_state.push_frame(frame)
        preds = player_state.predict()

        new_sent, pressed, btn_names = decode_and_press(
            bot_ctrl, preds, player_state.prev_sent, temperature=args.temperature,
        )
        player_state.prev_sent = new_sent

        # Snapshot current stocks for win/loss detection
        try:
            players = sorted(gs.players.items())
            if len(players) >= 2:
                me = gs.players[detected_port]
                opp_ports = [p for p in gs.players if p != detected_port]
                opp = gs.players[opp_ports[0]] if opp_ports else None
                if initial_stocks is None:
                    initial_stocks = (me.stock, opp.stock if opp else 0)
                # Emit a periodic log
                if int(now) % 2 == 0 and int(now * 10) % 20 < 2:
                    log.info("BOT(p%d) %dstk %d%% | OPP %dstk %d%%",
                             detected_port, me.stock, int(me.percent),
                             opp.stock if opp else 0, int(opp.percent) if opp else 0)
        except Exception:
            pass

    # ---- Determine result ----
    try:
        me_final = gs.players.get(detected_port) if detected_port else None
        opp_ports = [p for p in gs.players if p != detected_port] if detected_port else []
        opp_final = gs.players.get(opp_ports[0]) if opp_ports else None
        if match_started and me_final is not None and opp_final is not None:
            if me_final.stock > 0 and opp_final.stock == 0:
                final_result = "win"
            elif me_final.stock == 0 and opp_final.stock > 0:
                final_result = "loss"
            else:
                # Match ended without a clear stock-out — likely timeout or disconnect
                if final_result == "timeout":
                    pass  # keep
                else:
                    final_result = "disconnect"
        elif not match_started:
            final_result = "no-opponent"
    except Exception:
        pass

    # Find the most recent .slp in replay_dir
    try:
        replays = sorted(
            (p for p in Path(replay_dir).glob("*.slp")),
            key=lambda p: p.stat().st_mtime,
        )
        if replays:
            replay_path_out = str(replays[-1].resolve())
    except Exception:
        pass

finally:
    try:
        console.stop()
    except Exception:
        pass

emit_result(final_result, replay_path_out)
