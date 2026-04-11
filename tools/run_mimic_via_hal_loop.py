#!/usr/bin/env python3
"""Run a MIMIC checkpoint against a CPU opponent in Dolphin.

Usage:
    python tools/run_mimic_via_hal_loop.py \
      --checkpoint checkpoints/hal-7class_best.pt \
      --dolphin-path /path/to/dolphin-emu \
      --iso-path /path/to/melee.iso \
      --data-dir data/fox_hal_full \
      --cpu-level 9
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse, logging, os, signal

import melee
import torch

from tools.inference_utils import (
    load_mimic_model, load_inference_context, build_frame,
    PlayerState, decode_and_press,
)

log = logging.getLogger("mimic_vs_cpu")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  [%(levelname)s]  %(message)s")

parser = argparse.ArgumentParser(description="MIMIC vs CPU")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--dolphin-path", required=True)
parser.add_argument("--iso-path", required=True)
parser.add_argument("--data-dir", required=True)
parser.add_argument("--character", default="FOX")
parser.add_argument("--cpu-character", default="FOX")
parser.add_argument("--cpu-level", type=int, default=9)
parser.add_argument("--stage", default="BATTLEFIELD")
parser.add_argument("--temperature", type=float, default=1.0)
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, cfg = load_mimic_model(args.checkpoint, DEVICE)
log.info("Loaded model: %d params, encoder=%s",
         sum(p.numel() for p in model.parameters()), cfg.encoder_type)

ctx = load_inference_context(args.data_dir)
player = PlayerState(model, cfg.max_seq_len, DEVICE, ctx=ctx)

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
ego_ctrl = melee.Controller(console=console, port=1, type=melee.ControllerType.STANDARD)
cpu_ctrl = melee.Controller(console=console, port=2, type=melee.ControllerType.STANDARD)
console.run(iso_path=args.iso_path)
console.connect(); ego_ctrl.connect(); cpu_ctrl.connect()
log.info("Connected")

menu_bot = melee.MenuHelper()
menu_cpu = melee.MenuHelper()
BOT_CHAR = melee.Character[args.character]
CPU_CHAR = melee.Character[args.cpu_character]
STAGE = melee.Stage[args.stage]

def shutdown(*a):
    console.stop(); sys.exit(0)
signal.signal(signal.SIGINT, shutdown)

_was_in_game = False
while True:
    gs = console.step()
    if gs is None:
        continue
    if gs.menu_state not in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
        if _was_in_game:
            log.info("Game ended.")
            shutdown()
        menu_bot.menu_helper_simple(gs, ego_ctrl, BOT_CHAR, STAGE, cpu_level=0, autostart=False)
        menu_cpu.menu_helper_simple(gs, cpu_ctrl, CPU_CHAR, STAGE, cpu_level=args.cpu_level, autostart=True)
        ego_ctrl.flush(); cpu_ctrl.flush()
        continue

    _was_in_game = True
    frame = build_frame(gs, player.prev_sent, ctx)
    if frame is None:
        continue

    player.push_frame(frame)
    preds = player.predict()

    new_sent, pressed, btn_names = decode_and_press(
        ego_ctrl, preds, player.prev_sent, temperature=args.temperature)
    player.prev_sent = new_sent

    # Log
    top3 = torch.nn.functional.softmax(
        preds["btn_logits"][0, -1].float() / args.temperature, dim=-1
    ).topk(min(3, len(btn_names)))
    top3_str = " ".join(f"{btn_names[i]}={v:.3f}"
                        for v, i in zip(top3.values.tolist(), top3.indices.tolist()))
    gs_str = ""
    players = sorted(gs.players.items())
    if len(players) >= 2:
        ps1, ps2 = players[0][1], players[1][1]
        gs_str = f"  S={ps1.stock}({ps1.percent:.0f}%) O={ps2.stock}({ps2.percent:.0f}%)"
    log.info("MAIN=(%.2f,%.2f) L=%.2f BTN=%s  top3=[%s]%s",
             new_sent["main_x"], new_sent["main_y"], new_sent["l_shldr"],
             pressed, top3_str, gs_str)
