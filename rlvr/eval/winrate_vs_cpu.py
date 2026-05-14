"""Play N full matches vs a CPU opponent, report win-rate.

A thin standalone driver — boots one Dolphin (Exi-AI + FFW by default
for speed), plays N matches back-to-back, tracks bot vs opp stocks at
each match end, prints a JSON report.

Win = bot_stocks > 0 AND opp_stocks == 0 at the end-of-match transition.
Loss = the reverse. Draw = both > 0 (timeout) or both == 0.

Usage:
    DISPLAY=:99 python -m rlvr.eval.winrate_vs_cpu \\
        --ckpt checkpoints/3way_20260513/3way-20260513-comboext_update0050.pt \\
        --data-dir hf_checkpoints/fox \\
        --dolphin-path emulator_ffw/squashfs-root/usr/bin/dolphin-emu \\
        --iso-path melee.iso \\
        --n-matches 50 \\
        --use-exi-inputs --enable-ffw \\
        --out reports/winrate-comboext.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import melee  # noqa: E402

from tools.inference_utils import (  # noqa: E402
    load_inference_context, load_mimic_model,
    build_frame, PlayerState, decode_and_press,
)


log = logging.getLogger("rlvr.eval.winrate")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  [%(levelname)s]  %(message)s")


def _result_from_stocks(bot_stocks: int, opp_stocks: int) -> str:
    if bot_stocks > 0 and opp_stocks == 0:
        return "win"
    if opp_stocks > 0 and bot_stocks == 0:
        return "loss"
    return "draw"


def run_winrate(
    ckpt: Path,
    data_dir: Path,
    dolphin_path: Path,
    iso_path: Path,
    n_matches: int,
    bot_port: int = 1,
    bot_character: str = "FOX",
    cpu_character: str = "FOX",
    cpu_level: int = 9,
    stage: str = "FINAL_DESTINATION",
    temperature: float = 1.0,
    use_exi_inputs: bool = False,
    enable_ffw: bool = False,
    gfx_backend: str = "",
    disable_audio: bool = False,
    replay_dir: Optional[Path] = None,
    device: str = "cuda",
    out: Optional[Path] = None,
) -> dict:
    if replay_dir is None:
        replay_dir = Path("replays_winrate")
    replay_dir.mkdir(parents=True, exist_ok=True)

    model, cfg = load_mimic_model(str(ckpt), device)
    ctx = load_inference_context(data_dir)

    console = melee.Console(
        path=str(dolphin_path), is_dolphin=True,
        tmp_home_directory=True, copy_home_directory=False,
        blocking_input=True, online_delay=0,
        setup_gecko_codes=True, fullscreen=False,
        gfx_backend=gfx_backend,
        disable_audio=disable_audio,
        use_exi_inputs=use_exi_inputs, enable_ffw=enable_ffw,
        save_replays=True, replay_dir=str(replay_dir),
    )
    ego_ctrl = melee.Controller(console=console, port=bot_port,
                                type=melee.ControllerType.STANDARD)
    cpu_port = 2 if bot_port == 1 else 1
    cpu_ctrl = melee.Controller(console=console, port=cpu_port,
                                type=melee.ControllerType.STANDARD)
    console.run(iso_path=str(iso_path))
    console.connect()
    ego_ctrl.connect()
    cpu_ctrl.connect()
    log.info("Dolphin connected, exi=%s ffw=%s gfx=%s",
             use_exi_inputs, enable_ffw, gfx_backend or "<default>")

    menu_bot = melee.MenuHelper()
    menu_cpu = melee.MenuHelper()
    BOT_CHAR = melee.Character[bot_character]
    CPU_CHAR = melee.Character[cpu_character]
    STAGE = melee.Stage[stage]

    player = PlayerState(model, cfg.max_seq_len, device, ctx=ctx)
    results = []
    matches_started = 0
    in_game = False
    last_bot_stocks = 4
    last_opp_stocks = 4
    last_frame = 0
    t_start = time.time()
    t_match_start = time.time()

    def shutdown(*_):
        try:
            console.stop()
        finally:
            sys.exit(1)
    signal.signal(signal.SIGINT, shutdown)

    while len(results) < n_matches:
        gs = console.step()
        if gs is None:
            continue

        if gs.menu_state not in (melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH):
            if in_game:
                t_match_end = time.time()
                result = _result_from_stocks(last_bot_stocks, last_opp_stocks)
                match_meta = {
                    "match_idx": matches_started,
                    "result": result,
                    "bot_stocks": last_bot_stocks,
                    "opp_stocks": last_opp_stocks,
                    "frames": last_frame,
                    "wall_seconds": round(t_match_end - t_match_start, 1),
                }
                results.append(match_meta)
                log.info("match %d/%d done: %s (bot=%d opp=%d  %ds)",
                         len(results), n_matches, result,
                         last_bot_stocks, last_opp_stocks,
                         match_meta["wall_seconds"])
                in_game = False
                player = PlayerState(model, cfg.max_seq_len, device, ctx=ctx)
            menu_bot.menu_helper_simple(
                gs, ego_ctrl, BOT_CHAR, STAGE,
                cpu_level=0, autostart=False)
            menu_cpu.menu_helper_simple(
                gs, cpu_ctrl, CPU_CHAR, STAGE,
                cpu_level=cpu_level, autostart=True)
            ego_ctrl.flush()
            cpu_ctrl.flush()
            continue

        if not in_game:
            in_game = True
            matches_started += 1
            t_match_start = time.time()
            log.info("match %d/%d starting...", matches_started, n_matches)

        # Track stocks every frame so the menu-transition reads the most
        # recent in-game value.
        ports = sorted(gs.players.items())
        bot_ps = gs.players.get(bot_port)
        opp_ps = gs.players.get(cpu_port)
        if bot_ps is not None:
            last_bot_stocks = int(bot_ps.stock)
        if opp_ps is not None:
            last_opp_stocks = int(opp_ps.stock)
        last_frame = int(gs.frame)

        frame = build_frame(gs, player.prev_sent, ctx)
        if frame is None:
            continue
        player.push_frame(frame)
        preds = player.predict()
        new_sent, pressed, btn_names = decode_and_press(
            ego_ctrl, preds, player.prev_sent, temperature=temperature)
        player.prev_sent = new_sent

    console.stop()

    t_total = time.time() - t_start
    n_win = sum(1 for r in results if r["result"] == "win")
    n_loss = sum(1 for r in results if r["result"] == "loss")
    n_draw = sum(1 for r in results if r["result"] == "draw")
    report = {
        "ckpt": str(ckpt),
        "data_dir": str(data_dir),
        "n_matches": len(results),
        "win_rate": n_win / max(1, len(results)),
        "win": n_win, "loss": n_loss, "draw": n_draw,
        "avg_frames": sum(r["frames"] for r in results) / max(1, len(results)),
        "wall_seconds_total": round(t_total, 1),
        "config": {
            "bot_character": bot_character,
            "cpu_character": cpu_character,
            "cpu_level": cpu_level,
            "stage": stage,
            "temperature": temperature,
            "use_exi_inputs": use_exi_inputs,
            "enable_ffw": enable_ffw,
        },
        "matches": results,
    }
    log.info("DONE. %d matches | W%d L%d D%d | win-rate=%.1f%% | total=%.0fs",
             len(results), n_win, n_loss, n_draw,
             100.0 * report["win_rate"], t_total)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--dolphin-path", required=True, type=Path)
    ap.add_argument("--iso-path", required=True, type=Path)
    ap.add_argument("--n-matches", type=int, default=50)
    ap.add_argument("--bot-port", type=int, default=1)
    ap.add_argument("--bot-character", default="FOX")
    ap.add_argument("--cpu-character", default="FOX")
    ap.add_argument("--cpu-level", type=int, default=9)
    ap.add_argument("--stage", default="FINAL_DESTINATION")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--use-exi-inputs", action="store_true")
    ap.add_argument("--enable-ffw", action="store_true")
    ap.add_argument("--gfx-backend", default="")
    ap.add_argument("--disable-audio", action="store_true")
    ap.add_argument("--replay-dir", type=Path, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    run_winrate(
        ckpt=args.ckpt, data_dir=args.data_dir,
        dolphin_path=args.dolphin_path, iso_path=args.iso_path,
        n_matches=args.n_matches, bot_port=args.bot_port,
        bot_character=args.bot_character, cpu_character=args.cpu_character,
        cpu_level=args.cpu_level, stage=args.stage,
        temperature=args.temperature,
        use_exi_inputs=args.use_exi_inputs, enable_ffw=args.enable_ffw,
        gfx_backend=args.gfx_backend, disable_audio=args.disable_audio,
        replay_dir=args.replay_dir, device=args.device,
        out=args.out,
    )


if __name__ == "__main__":
    main()
