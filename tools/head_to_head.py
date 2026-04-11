#!/usr/bin/env python3
"""Run two MIMIC checkpoints head-to-head in Dolphin.

P1 (port 1) and P2 (port 2) are each driven by a model checkpoint.

Usage:
    python tools/head_to_head.py \
      --p1-checkpoint checkpoints/model_a.pt \
      --p2-checkpoint checkpoints/model_b.pt \
      --dolphin-path /path/to/dolphin-emu \
      --iso-path /path/to/melee.iso \
      --data-dir data/fox_hal_full
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse, atexit, logging, os, signal

import melee
import torch

from tools.inference_utils import (
    load_mimic_model, load_inference_context, build_frame, build_frame_p2,
    PlayerState, decode_and_press,
)

log = logging.getLogger("h2h")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  [%(levelname)s]  %(message)s")

parser = argparse.ArgumentParser(description="Head-to-head: two models fight")
parser.add_argument("--p1-checkpoint", required=True)
parser.add_argument("--p2-checkpoint", required=True)
parser.add_argument("--p1-character", default="FOX")
parser.add_argument("--p2-character", default="FOX")
parser.add_argument("--stage", default="FINAL_DESTINATION")
parser.add_argument("--dolphin-path", required=True)
parser.add_argument("--iso-path", required=True)
parser.add_argument("--data-dir", required=True)
parser.add_argument("--temperature", type=float, default=1.0)
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
log.info("Loading P1: %s", args.p1_checkpoint)
model_p1, cfg_p1 = load_mimic_model(args.p1_checkpoint, DEVICE)
desc_p1 = f"MIMIC({cfg_p1.encoder_type})"
log.info("  P1 = %s (%d params)", desc_p1, sum(p.numel() for p in model_p1.parameters()))

log.info("Loading P2: %s", args.p2_checkpoint)
model_p2, cfg_p2 = load_mimic_model(args.p2_checkpoint, DEVICE)
desc_p2 = f"MIMIC({cfg_p2.encoder_type})"
log.info("  P2 = %s (%d params)", desc_p2, sum(p.numel() for p in model_p2.parameters()))

ctx = load_inference_context(args.data_dir)
player1 = PlayerState(model_p1, cfg_p1.max_seq_len, DEVICE, ctx=ctx)
player2 = PlayerState(model_p2, cfg_p2.max_seq_len, DEVICE, ctx=ctx)

# Dolphin
replay_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "replays")
os.makedirs(replay_dir, exist_ok=True)
console = melee.Console(
    path=args.dolphin_path, is_dolphin=True, tmp_home_directory=True,
    copy_home_directory=False, blocking_input=True, online_delay=0,
    setup_gecko_codes=True, fullscreen=False, gfx_backend="",
    disable_audio=False, use_exi_inputs=False, enable_ffw=False,
    save_replays=True, replay_dir=replay_dir,
)
ctrl_p1 = melee.Controller(console=console, port=1, type=melee.ControllerType.STANDARD)
ctrl_p2 = melee.Controller(console=console, port=2, type=melee.ControllerType.STANDARD)
console.run(iso_path=args.iso_path)
console.connect(); ctrl_p1.connect(); ctrl_p2.connect()
log.info("Connected to Dolphin")

P1_CHAR = melee.Character[args.p1_character]
P2_CHAR = melee.Character[args.p2_character]
STAGE = melee.Stage[args.stage]

menu_p1 = melee.MenuHelper()
menu_p2 = melee.MenuHelper()

log.info("P1 (port 1): %s — %s", desc_p1, args.p1_checkpoint)
log.info("P2 (port 2): %s — %s", desc_p2, args.p2_checkpoint)

# Game loop
game_frame = 0
_was_in_game = False
_last_stocks = None

def _print_summary():
    if _last_stocks:
        s1, s2 = _last_stocks
        log.info("=" * 60)
        log.info("GAME OVER — %d frames (%.1fs)", game_frame, game_frame / 60.0)
        log.info("  P1 (%s): %d stocks", desc_p1, s1)
        log.info("  P2 (%s): %d stocks", desc_p2, s2)
        if s1 > s2: log.info("  Result: P1 (%s) WINS", desc_p1)
        elif s2 > s1: log.info("  Result: P2 (%s) WINS", desc_p2)
        else: log.info("  Result: DRAW")
        log.info("=" * 60)

def _shutdown(*a):
    _print_summary()
    ctrl_p1.disconnect(); ctrl_p2.disconnect()
    console.stop(); sys.exit(0)

atexit.register(_print_summary)
signal.signal(signal.SIGINT, _shutdown)

while True:
    gs = console.step()
    if gs is None:
        continue

    if gs.menu_state not in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
        if _was_in_game:
            _shutdown()
        menu_p1.menu_helper_simple(gs, ctrl_p1, P1_CHAR, STAGE, cpu_level=0, autostart=False)
        menu_p2.menu_helper_simple(gs, ctrl_p2, P2_CHAR, STAGE, cpu_level=0, autostart=True)
        ctrl_p1.flush(); ctrl_p2.flush()
        continue

    _was_in_game = True
    players = sorted(gs.players.items())
    if len(players) < 2:
        continue
    _, ps1 = players[0]
    _, ps2 = players[1]

    # Build frames from each player's perspective using shared build_frame
    frame_p1 = build_frame(gs, player1.prev_sent, ctx)       # ego=ps1, opp=ps2
    frame_p2 = build_frame_p2(gs, player2.prev_sent, ctx)    # ego=ps2, opp=ps1

    if frame_p1 is None or frame_p2 is None:
        continue

    player1.push_frame(frame_p1)
    player2.push_frame(frame_p2)

    preds_p1 = player1.predict()
    preds_p2 = player2.predict()

    new_sent_p1, _, _ = decode_and_press(ctrl_p1, preds_p1, player1.prev_sent, args.temperature)
    new_sent_p2, _, _ = decode_and_press(ctrl_p2, preds_p2, player2.prev_sent, args.temperature)
    player1.prev_sent = new_sent_p1
    player2.prev_sent = new_sent_p2

    _last_stocks = (ps1.stock, ps2.stock)

    if game_frame % 60 == 0:
        log.info("[f%d]  P1(%s) %dstk %.0f%%  |  P2(%s) %dstk %.0f%%",
                 game_frame, desc_p1, ps1.stock, ps1.percent,
                 desc_p2, ps2.stock, ps2.percent)

    game_frame += 1
