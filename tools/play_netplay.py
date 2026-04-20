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
import select
import signal
import subprocess
import threading
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
parser.add_argument("--max-matches", type=int, default=1,
                    help="Play up to N matches back-to-back in one Dolphin session. "
                         "Default 1 (one-shot, matches pre-refactor CLI behavior). "
                         "Pass -1 for unlimited (Discord bot uses this).")
parser.add_argument("--rematch-timeout", type=float, default=30.0,
                    help="Seconds to wait in CSS after a match for the opponent to "
                         "ready up for the next one. Only applies to matches 2+; "
                         "the initial connect uses --no-opponent-timeout.")
parser.add_argument("--check-stdin", action="store_true",
                    help="Poll stdin non-blocking for 'STOP' — the Discord bot writes "
                         "this to the subprocess to end the chain cleanly after the "
                         "current match. Off by default for CLI use.")
parser.add_argument("--stall-timeout", type=float, default=30.0,
                    help="Seconds with no Dolphin frames before the watchdog kills "
                         "our child Dolphin process to unblock console.step(). "
                         "Covers mid-match netplay desyncs / opponent hard-DCs.")
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


def emit_match_start(idx: int):
    print(f"MATCH_START: {idx}", flush=True)


def emit_match_result(result: str, replay_path: str = "",
                      bot_stocks: int = -1, opp_stocks: int = -1,
                      bot_percent: float = 0.0, opp_percent: float = 0.0,
                      stage: str = ""):
    """Per-match status line(s) for the Discord bot parser."""
    print(f"RESULT: {result}", flush=True)
    if bot_stocks >= 0 and opp_stocks >= 0:
        print(f"SCORE: bot={bot_stocks}stk/{bot_percent:.0f}% "
              f"opp={opp_stocks}stk/{opp_percent:.0f}%", flush=True)
    if stage:
        print(f"STAGE: {stage}", flush=True)
    if replay_path:
        print(f"REPLAY: {replay_path}", flush=True)


def emit_session_end(reason: str):
    print(f"SESSION_END: {reason}", flush=True)


class _SessionExit(Exception):
    """Sentinel to unwind the outer match loop from deep inside the inner one."""


def _snapshot_replay_mtimes(replay_dir_: str) -> dict:
    try:
        return {p.name: p.stat().st_mtime for p in Path(replay_dir_).glob("*.slp")}
    except Exception:
        return {}


def _find_new_replay(replay_dir_: str, before_snapshot: dict) -> str:
    try:
        after = list(Path(replay_dir_).glob("*.slp"))
        new = [p for p in after
               if p.name not in before_snapshot
               or p.stat().st_mtime > before_snapshot[p.name]]
        new.sort(key=lambda p: p.stat().st_mtime)
        return str(new[-1].resolve()) if new else ""
    except Exception:
        return ""


# Shared state between the watchdog thread and the main loop.
# _last_frame_time updates whenever console.step() returns a non-None gs.
# _stop_flag gets set by the watchdog when STOP arrives on stdin; the main
# loop checks it every iteration. The watchdog also detects mid-match
# stalls (Dolphin blocked waiting for remote netplay frames that never
# come) and hard-kills our child Dolphin to unblock console.step() so the
# main loop can exit cleanly.
_last_frame_time: float = time.time()
_stop_flag: bool = False


def _watchdog():
    while True:
        time.sleep(1.0)
        # stdin STOP poll — runs here so STOP gets noticed even when the
        # main loop is blocked in console.step().
        if args.check_stdin:
            try:
                r, _, _ = select.select([sys.stdin], [], [], 0)
                if r:
                    line = sys.stdin.readline()
                    if line and line.strip().upper() == "STOP":
                        globals()["_stop_flag"] = True
                        log.info("Watchdog: STOP received")
            except (ValueError, OSError):
                pass
            except Exception:
                log.exception("Watchdog stdin poll failed")
        # Stall detection — if no frames from Dolphin for stall_timeout
        # seconds, assume Dolphin is wedged on a remote netplay wait and
        # hard-kill our child processes so the main loop unblocks.
        stall = time.time() - _last_frame_time
        if stall > args.stall_timeout:
            log.error("No Dolphin frames for %.1fs — killing child processes", stall)
            try:
                subprocess.run(
                    ["pkill", "-KILL", "-P", str(os.getpid())],
                    timeout=5, check=False,
                )
            except Exception:
                log.exception("pkill failed")
            # Mark that we're tearing down so the main loop's session_reason
            # reflects this instead of a generic exit.
            globals()["_stop_flag"] = True
            # Reset the timer so we don't pkill in a tight loop.
            globals()["_last_frame_time"] = time.time()


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
    emit_match_start(1)
    emit_match_result("failed")
    emit_session_end("error")
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
    gfx_backend="Vulkan",
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
        emit_session_end("signal")
    except Exception:
        pass
    try:
        console.stop()
    except Exception:
        pass
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)

# Watchdog: polls stdin for STOP and kills a wedged Dolphin.
threading.Thread(target=_watchdog, daemon=True, name="play_netplay_watchdog").start()

# ---- Main loop: outer match loop + inner per-match loop ----
#
# Session-persistent state (lives across matches):
#   opponent_last_seen / opponent_ever_seen — set once the opponent appears
#       in gs.players; used for DC detection. Do NOT reset per match.
#   match_idx — 1-based counter of matches started this session.
#   session_reason — why the session ends (max-matches / opponent-gone /
#       opponent-timeout / stopped / error / signal).
#   detected_port / player_state — per-match but kept at module scope so
#       _build_frame() can close over detected_port without shenanigans.
player_state: PlayerState = None
detected_port: int = 0
opponent_last_seen = None
opponent_ever_seen = False


def _build_frame(gs, prev_sent, ctx_):
    if detected_port <= 1:
        return build_frame(gs, prev_sent, ctx_)
    else:
        return build_frame_p2(gs, prev_sent, ctx_)


session_reason = "max-matches"
stop_requested = False
match_idx = 0

try:
    while True:
        match_idx += 1
        if args.max_matches > 0 and match_idx > args.max_matches:
            session_reason = "max-matches"
            break

        # Per-match reset. MenuHelper latches stage_selected /
        # frozen_stadium_selected to True once a stage is picked; in a
        # persistent-lobby rematch the stage picker isn't re-shown, but
        # reset defensively in case Slippi ever does surface it again.
        menu_helper.stage_selected = False
        menu_helper.frozen_stadium_selected = False
        player_state = None
        detected_port = 0
        match_started = False
        match_start_time = None
        initial_stocks = None
        last_seen_me_stock = None
        last_seen_opp_stock = None
        last_seen_me_percent = 0.0
        last_seen_opp_percent = 0.0
        match_stage_name = ""
        match_final_result = "timeout"
        dc_detected = False

        replay_snapshot = _snapshot_replay_mtimes(replay_dir)
        css_wait_start = time.time()

        # ---- Inner: one match (menu → CSS → in-game → game end) ----
        while True:
            gs = console.step()
            if gs is None:
                # Still check the STOP flag here — the watchdog sets it
                # even when frames have stalled, so we can react without
                # needing a fresh frame.
                if _stop_flag:
                    stop_requested = True
                    if not match_started:
                        log.info("STOP received pre-match; exiting session")
                        session_reason = "stopped"
                        raise _SessionExit()
                continue

            # Frame arrived — refresh the watchdog's stall clock.
            globals()["_last_frame_time"] = time.time()

            now = time.time()

            # STOP flag is set by the watchdog thread when stdin receives
            # "STOP\n" or when the watchdog had to kill a wedged Dolphin.
            # Pre-match: exit the session immediately. Mid-match: finish
            # the current match then exit (opponents should always see
            # their match complete).
            if _stop_flag:
                stop_requested = True
                if not match_started:
                    log.info("STOP received pre-match; exiting session")
                    session_reason = "stopped"
                    raise _SessionExit()

            # Track opponent presence for DC detection.
            if gs.players and len(gs.players) >= 2:
                opponent_last_seen = now
                opponent_ever_seen = True

            # --- Menu / CSS / post-game path ---
            if gs.menu_state not in (melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH):
                if match_started:
                    log.info("Game ended (menu=%s)", gs.menu_state)
                    break

                # Opponent DC in menus — either the first connect never
                # landed, or the opponent left the lobby after an earlier
                # match in this session.
                if (opponent_ever_seen
                        and opponent_last_seen is not None
                        and (now - opponent_last_seen) > args.opponent_lost_timeout):
                    log.warning("Opponent missing for %.1fs, treating as disconnect",
                                now - opponent_last_seen)
                    match_final_result = "disconnect"
                    dc_detected = True
                    break

                # CSS-wait timeout. Match 1 uses the longer no-opponent
                # timeout (covers initial Slippi handshake latency); match
                # 2+ uses the snappier rematch timeout for "opponent didn't
                # ready up on the CSS." The rematch case ends the session
                # without emitting a RESULT — this iteration's match never
                # started, so there's nothing to announce.
                if match_idx >= 2:
                    if (now - css_wait_start) > args.rematch_timeout:
                        log.warning("Opponent didn't ready up within %.1fs, "
                                    "ending session", args.rematch_timeout)
                        session_reason = "opponent-timeout"
                        raise _SessionExit()
                else:
                    if (now - css_wait_start) > args.no_opponent_timeout:
                        log.warning("No opponent connected within %ds, giving up",
                                    args.no_opponent_timeout)
                        match_final_result = "no-opponent"
                        dc_detected = True
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
                # Detect which port the bot ended up on. Prefer the bot's
                # own Slippi connect code (unambiguous — handles dittos and
                # palette swaps), then fall back to character+costume,
                # then character only.
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
                me = gs.players[detected_port]
                # Capture the actual stage played (opponent picks in Direct
                # Connect, not our --stage arg). Enum name → human-readable
                # string on the bot side.
                try:
                    match_stage_name = gs.stage.name if gs.stage else ""
                except Exception:
                    match_stage_name = ""
                log.info("Match %d started. Bot is on port %d (char=%s costume=%d code=%s stage=%s)",
                         match_idx, detected_port, BOT_CHAR.name, me.costume,
                         getattr(me, "connectCode", "?"), match_stage_name)
                # Fresh PlayerState drops any rolling context from a prior
                # match — position/stage/stocks all reset on a new match,
                # so carrying frames across would push the model OOD.
                player_state = PlayerState(model, cfg.max_seq_len, DEVICE, ctx=ctx)
                initial_stocks = None
                emit_match_start(match_idx)

            if (now - match_start_time) > args.match_timeout:
                log.warning("Match %d exceeded %ds timeout", match_idx, args.match_timeout)
                match_final_result = "timeout"
                break

            if (opponent_last_seen is not None
                    and (now - opponent_last_seen) > args.opponent_lost_timeout):
                log.warning("Opponent missing mid-match for %.1fs, treating as disconnect",
                            now - opponent_last_seen)
                match_final_result = "disconnect"
                dc_detected = True
                break

            frame = _build_frame(gs, player_state.prev_sent, ctx)
            if frame is None:
                continue

            player_state.push_frame(frame)
            preds = player_state.predict()

            new_sent, pressed, btn_names = decode_and_press(
                bot_ctrl, preds, player_state.prev_sent, temperature=args.temperature,
            )
            player_state.prev_sent = new_sent

            # Snapshot stocks each frame so we can resolve the result from
            # the last-seen values (post-game gamestate may reset before
            # we exit the inner loop).
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
                    if me.stock == 0 or (opp is not None and opp.stock == 0):
                        game_over = True
                    if int(now) % 2 == 0 and int(now * 10) % 20 < 2:
                        log.info("BOT(p%d) %dstk %d%% | OPP %dstk %d%%",
                                 detected_port, me.stock, int(me.percent),
                                 opp.stock if opp else 0, int(opp.percent) if opp else 0)
            except Exception:
                pass

            if game_over:
                log.info("Stock-out detected. Match %d ending.", match_idx)
                time.sleep(1.0)
                break

        # ---- Inner loop exited: resolve this match's result ----
        try:
            if not match_started:
                # Match never started. If we'd seen the opponent earlier in
                # the session, treat as disconnect; otherwise (match 1)
                # it's no-opponent.
                if match_final_result not in ("disconnect", "no-opponent"):
                    match_final_result = "disconnect" if opponent_ever_seen else "no-opponent"
            elif last_seen_me_stock is not None and last_seen_opp_stock is not None:
                if last_seen_me_stock > 0 and last_seen_opp_stock == 0:
                    match_final_result = "win"
                elif last_seen_me_stock == 0 and last_seen_opp_stock > 0:
                    match_final_result = "loss"
                elif last_seen_me_stock > last_seen_opp_stock:
                    match_final_result = "win"
                elif last_seen_me_stock < last_seen_opp_stock:
                    match_final_result = "loss"
                else:
                    if last_seen_me_percent < last_seen_opp_percent:
                        match_final_result = "win"
                    elif last_seen_me_percent > last_seen_opp_percent:
                        match_final_result = "loss"
                    else:
                        match_final_result = "draw"
            # else: match started but no stocks observed — leave as "timeout"
        except Exception:
            pass

        replay_path_out = _find_new_replay(replay_dir, replay_snapshot)

        # Emit this match's result. If MATCH_START was never printed
        # (match didn't reach IN_GAME — e.g. match 1 no-opponent), print
        # one now so the Discord bot can pair the lines cleanly.
        if not match_started:
            emit_match_start(match_idx)
        emit_match_result(
            match_final_result, replay_path_out,
            bot_stocks=last_seen_me_stock if last_seen_me_stock is not None else -1,
            opp_stocks=last_seen_opp_stock if last_seen_opp_stock is not None else -1,
            bot_percent=last_seen_me_percent,
            opp_percent=last_seen_opp_percent,
            stage=match_stage_name,
        )

        # Session-continue decisions.
        if stop_requested:
            session_reason = "stopped"
            break
        if dc_detected or match_final_result in ("no-opponent", "disconnect"):
            session_reason = "opponent-gone"
            break

except _SessionExit:
    pass
except Exception:
    log.exception("Unexpected error in session loop")
    session_reason = "error"
finally:
    try:
        console.stop()
    except Exception:
        pass

emit_session_end(session_reason)
