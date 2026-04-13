#!/usr/bin/env python3
"""Run a MIMIC checkpoint against a human via Slippi Online Direct Connect.

Usage:
    python tools/play_netplay.py \
      --checkpoint checkpoints/falco-20260412-relpos-28k.pt \
      --dolphin-path /path/to/dolphin-emu \
      --iso-path /path/to/melee.iso \
      --data-dir data/falco_v2 \
      --character FALCO \
      --connect-code WAVE#666

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
import json
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
                    help="Opponent's Slippi direct connect code (e.g. WAVE#666)")
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--costume", type=int, default=0)
parser.add_argument("--no-opponent-timeout", type=float, default=120.0,
                    help="Seconds to wait for the opponent to connect before giving up")
parser.add_argument("--opponent-lost-timeout", type=float, default=10.0,
                    help="Seconds to wait after the opponent disappears (disconnect / "
                         "crash) before giving up and exiting the direct-connect lobby")
parser.add_argument("--match-timeout", type=float, default=900.0,
                    help="Maximum seconds for a single match (15 min default)")
parser.add_argument("--bot-slippi-code", default=None,
                    help="Bot's own Slippi connect code (e.g. MIMIC#01). "
                         "Used to identify the bot's player in gs.players during "
                         "netplay matches. If omitted, read from user.json.")
parser.add_argument("--slippi-home", default=None,
                    help="Path to a directory containing Slippi/user.json "
                         "(libmelee's dolphin_home_path). Defaults to the "
                         "slippi_home/ dir at the repo root, then falls back "
                         "to ~/.config/SlippiOnline.")
args = parser.parse_args()


_REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve_slippi_home() -> str:
    """Find the Slippi home dir (one that contains Slippi/user.json)."""
    candidates = []
    if args.slippi_home:
        candidates.append(Path(args.slippi_home).expanduser())
    # Bundled in the repo (gitignored) — preferred for portability
    candidates.append(_REPO_ROOT / "slippi_home")
    # User-level default (libmelee's Linux default)
    candidates.append(Path.home() / ".config" / "SlippiOnline")
    for c in candidates:
        if (c / "Slippi" / "user.json").exists():
            return str(c.resolve())
    return ""


def _load_bot_code_from_user_json(slippi_home: str) -> str:
    """Read the bot's connect code from <slippi_home>/Slippi/user.json."""
    if not slippi_home:
        return ""
    path = Path(slippi_home) / "Slippi" / "user.json"
    try:
        return json.loads(path.read_text()).get("connectCode", "")
    except Exception:
        return ""


SLIPPI_HOME = _resolve_slippi_home()
BOT_CODE = (args.bot_slippi_code
            or _load_bot_code_from_user_json(SLIPPI_HOME)
            or "").strip()

if not SLIPPI_HOME:
    log.warning("No slippi_home found with Slippi/user.json — Dolphin will not "
                "be able to log in to Slippi Online. Place user.json at "
                "%s/slippi_home/Slippi/user.json or ~/.config/SlippiOnline/Slippi/user.json",
                _REPO_ROOT)
else:
    log.info("Slippi home: %s", SLIPPI_HOME)

if not BOT_CODE:
    print("WARNING: bot's own Slippi code not found; falling back to character-only "
          "port detection", file=sys.stderr)


def emit_result(result: str, replay_path: str = "",
                bot_stocks: int = -1, opp_stocks: int = -1,
                bot_percent: float = 0.0, opp_percent: float = 0.0):
    """Print machine-readable final status for the Discord bot parser."""
    print(f"RESULT: {result}", flush=True)
    if bot_stocks >= 0 and opp_stocks >= 0:
        print(f"SCORE: bot={bot_stocks}stk/{bot_percent:.0f}% "
              f"opp={opp_stocks}stk/{opp_percent:.0f}%", flush=True)
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

console_kwargs = dict(
    path=args.dolphin_path,
    is_dolphin=True,
    tmp_home_directory=True,
    copy_home_directory=True,  # copy the slippi home dir (for user.json)
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
# If we found a slippi_home with user.json, point libmelee at it explicitly.
# Otherwise libmelee uses its Linux default (~/.config/SlippiOnline).
if SLIPPI_HOME:
    console_kwargs["dolphin_home_path"] = SLIPPI_HOME

console = melee.Console(**console_kwargs)

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
# Track when we last saw an opponent (>=2 players in gs.players). Used to
# detect mid-match DCs: if the opponent's Dolphin crashes, the bot gets
# dumped back to the menu but the lobby stays open, and without this check
# it would sit there forever waiting for someone to press Start.
opponent_last_seen = None
opponent_ever_seen = False

# Track stocks to detect game end robustly. We remember the last-seen
# stock counts because by the time the script exits the in-game loop the
# gamestate may have already reset for post-game.
initial_stocks = None
last_seen_me_stock = None
last_seen_opp_stock = None
last_seen_me_percent = 0.0
last_seen_opp_percent = 0.0
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

        # Track opponent presence for DC detection. At least 2 players in
        # gs.players means we're either in a lobby/CSS with the opponent,
        # or in-match.
        if gs.players and len(gs.players) >= 2:
            opponent_last_seen = now
            opponent_ever_seen = True

        # Menu / lobby navigation — let MenuHelper drive until we reach IN_GAME
        if gs.menu_state not in (melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH):
            if match_started:
                # We were in a game and now we're not — match ended
                log.info("Game ended (menu=%s)", gs.menu_state)
                break

            # Opponent-DC detection: if we saw the opponent at some point
            # (in CSS / in-match) but they've been missing from gs.players
            # for longer than the DC timeout, bail out. This covers the
            # "user's Dolphin crashed mid-match and the bot is now stuck
            # on the post-game menu with an empty lobby" case.
            if (opponent_ever_seen
                    and opponent_last_seen is not None
                    and (now - opponent_last_seen) > args.opponent_lost_timeout):
                log.warning("Opponent missing for %.1fs, treating as disconnect",
                            now - opponent_last_seen)
                final_result = "disconnect"
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
            # Detect which port the bot ended up on. Prefer the bot's own
            # Slippi connect code (unambiguous, handles dittos and palette
            # swaps), then fall back to character+costume, then character only.
            detected_port = 0
            if BOT_CODE:
                for pid, p in gs.players.items():
                    if getattr(p, "connectCode", "") == BOT_CODE:
                        detected_port = pid
                        break
            if detected_port == 0:
                detected_port = melee.gamestate.port_detector(gs, BOT_CHAR, args.costume)
            if detected_port == 0:
                matches = [pid for pid, p in gs.players.items()
                           if p.character == BOT_CHAR]
                if len(matches) == 1:
                    detected_port = matches[0]
                    log.info("port_detector fallback: char-only match on port %d", detected_port)
                elif len(matches) > 1:
                    detected_port = matches[0]
                    log.warning("port_detector fallback: ditto detected and no "
                                "connectCode match, defaulting to port %d",
                                detected_port)
            if detected_port == 0:
                if gs.players:
                    log.debug("port_detector failed, players=%s",
                              {pid: (p.character.name, getattr(p, "connectCode", ""))
                               for pid, p in gs.players.items()})
                continue
            match_started = True
            match_start_time = now
            last_known_state = "IN_GAME"
            me = gs.players[detected_port]
            log.info("Match started. Bot is on port %d (char=%s costume=%d code=%s)",
                     detected_port, BOT_CHAR.name, me.costume,
                     getattr(me, "connectCode", "?"))
            player_state = PlayerState(model, cfg.max_seq_len, DEVICE, ctx=ctx)
            initial_stocks = None

        last_in_game_time = now

        # Match timeout
        if (now - match_start_time) > args.match_timeout:
            log.warning("Match exceeded %ds timeout", args.match_timeout)
            final_result = "timeout"
            break

        # Mid-match DC: opponent vanished from gs.players while IN_GAME.
        # (opponent_last_seen is refreshed above whenever len(players) >= 2.)
        if (opponent_last_seen is not None
                and (now - opponent_last_seen) > args.opponent_lost_timeout):
            log.warning("Opponent missing mid-match for %.1fs, treating as disconnect",
                        now - opponent_last_seen)
            final_result = "disconnect"
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

        # Snapshot current stocks for win/loss detection. Record the
        # last-seen non-zero stocks for each side so we can report scores
        # even after the game state resets.
        game_over = False
        try:
            players = sorted(gs.players.items())
            if len(players) >= 2:
                me = gs.players[detected_port]
                opp_ports = [p for p in gs.players if p != detected_port]
                opp = gs.players[opp_ports[0]] if opp_ports else None
                if initial_stocks is None:
                    initial_stocks = (me.stock, opp.stock if opp else 0)
                last_seen_me_stock = me.stock
                last_seen_me_percent = me.percent
                if opp is not None:
                    last_seen_opp_stock = opp.stock
                    last_seen_opp_percent = opp.percent
                # Game over: either side reached 0 stocks
                if me.stock == 0 or (opp is not None and opp.stock == 0):
                    game_over = True
                # Emit a periodic log
                if int(now) % 2 == 0 and int(now * 10) % 20 < 2:
                    log.info("BOT(p%d) %dstk %d%% | OPP %dstk %d%%",
                             detected_port, me.stock, int(me.percent),
                             opp.stock if opp else 0, int(opp.percent) if opp else 0)
        except Exception:
            pass

        if game_over:
            # Wait ~1 second for the dying animation to complete, then bail
            # out of the match. console.stop() in the finally block will kill
            # Dolphin, ending the netplay session cleanly so the opponent
            # doesn't linger in a rematch lobby.
            log.info("Stock-out detected. Ending match.")
            time.sleep(1.0)
            break

    # ---- Determine result ----
    # Use the last-seen stock counts during the in-game loop — by the time we
    # exit, gs.players may have already been reset for the post-game menu.
    try:
        if not match_started:
            final_result = "no-opponent"
        elif last_seen_me_stock is not None and last_seen_opp_stock is not None:
            if last_seen_me_stock > 0 and last_seen_opp_stock == 0:
                final_result = "win"
            elif last_seen_me_stock == 0 and last_seen_opp_stock > 0:
                final_result = "loss"
            elif last_seen_me_stock > last_seen_opp_stock:
                final_result = "win"  # time-out with stock lead
            elif last_seen_me_stock < last_seen_opp_stock:
                final_result = "loss"
            else:
                # Equal stocks — fall back to percent (lower wins)
                if last_seen_me_percent < last_seen_opp_percent:
                    final_result = "win"
                elif last_seen_me_percent > last_seen_opp_percent:
                    final_result = "loss"
                else:
                    final_result = "draw"
        # else: leave as "timeout" (set at top)
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

emit_result(
    final_result, replay_path_out,
    bot_stocks=last_seen_me_stock if last_seen_me_stock is not None else -1,
    opp_stocks=last_seen_opp_stock if last_seen_opp_stock is not None else -1,
    bot_percent=last_seen_me_percent,
    opp_percent=last_seen_opp_percent,
)
