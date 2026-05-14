"""Head-to-head win-rate: two MIMIC checkpoints in one Dolphin, N matches.

Reuses the two-policy pattern from `tools/head_to_head.py` but plays
N back-to-back matches and tallies wins from policy-A's perspective.

Optional --alternate-ports flips which policy plays which port every
other match to eliminate port-handedness bias.

Watchable mode (default): DISPLAY=:0 + default OpenGL + regular
emulator/.  Vulkan looks great but produces visual artifacts on
this machine — leave the gfx_backend blank to let Dolphin pick
OpenGL.
Headless FFW mode: DISPLAY=:99 + Null gfx + emulator_ffw/ +
--use-exi-inputs + --enable-ffw. Realtime is ~80s/match; FFW
~35s/match.

Run:
    DISPLAY=:0 python -m rlvr.eval.head_to_head_winrate \\
        --ckpt-a checkpoints/3way_20260514/3way-20260514-comboext_final.pt \\
        --ckpt-b hf_checkpoints/fox/model.pt \\
        --data-dir hf_checkpoints/fox \\
        --dolphin-path emulator/squashfs-root/usr/bin/dolphin-emu \\
        --iso-path melee.iso \\
        --n-matches 3 \\
        --alternate-ports \\
        --out reports/h2h-smoke.json
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
    build_frame, build_frame_p2,
    PlayerState, decode_and_press,
)


log = logging.getLogger("rlvr.eval.h2h")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  [%(levelname)s]  %(message)s")


def _make_ctx(ctx_base, cfg):
    """Per-player context with n_combos matching the model."""
    from mimic.features import BTN7_N_CLASSES
    ctx = dict(ctx_base)
    n = cfg.n_controller_combos
    if n == BTN7_N_CLASSES:
        ctx["combo_map"] = {}
        ctx["n_combos"] = n
    elif n == 5:
        ctx["combo_map"] = {
            (1, 0, 0, 0, 0): 0, (0, 1, 0, 0, 0): 1, (0, 0, 1, 0, 0): 2,
            (0, 0, 0, 1, 0): 3, (0, 0, 0, 0, 0): 4, (0, 0, 0, 0, 1): 4,
            (1, 0, 0, 0, 1): 0, (0, 1, 0, 0, 1): 1, (0, 0, 1, 0, 1): 2,
            (0, 0, 0, 1, 1): 3,
        }
        ctx["n_combos"] = 5
    return ctx


def run_h2h(
    ckpt_a: Path,
    ckpt_b: Path,
    data_dir: Path,
    dolphin_path: Path,
    iso_path: Path,
    n_matches: int,
    bot_character: str = "FOX",
    stage: str = "FINAL_DESTINATION",
    temperature: float = 1.0,
    use_exi_inputs: bool = False,
    enable_ffw: bool = False,
    gfx_backend: str = "",
    disable_audio: bool = False,
    alternate_ports: bool = False,
    a_costume: int = 3,  # green Fox by default — visually distinct from B
    b_costume: int = 0,
    replay_dir: Optional[Path] = None,
    device: str = "cuda",
    out: Optional[Path] = None,
) -> dict:
    if replay_dir is None:
        replay_dir = Path("replays_h2h")
    replay_dir.mkdir(parents=True, exist_ok=True)

    log.info("loading A: %s", ckpt_a)
    model_a, cfg_a = load_mimic_model(str(ckpt_a), device)
    log.info("loading B: %s", ckpt_b)
    model_b, cfg_b = load_mimic_model(str(ckpt_b), device)
    ctx_base = load_inference_context(data_dir)
    ctx_a = _make_ctx(ctx_base, cfg_a)
    ctx_b = _make_ctx(ctx_base, cfg_b)

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
    ctrl_1 = melee.Controller(console=console, port=1,
                              type=melee.ControllerType.STANDARD)
    ctrl_2 = melee.Controller(console=console, port=2,
                              type=melee.ControllerType.STANDARD)
    console.run(iso_path=str(iso_path))
    console.connect()
    ctrl_1.connect()
    ctrl_2.connect()
    log.info("Dolphin connected, exi=%s ffw=%s gfx=%s alternate_ports=%s",
             use_exi_inputs, enable_ffw, gfx_backend, alternate_ports)

    menu_1 = melee.MenuHelper()
    menu_2 = melee.MenuHelper()
    CHAR = melee.Character[bot_character]
    STAGE = melee.Stage[stage]

    results = []
    matches_started = 0
    in_game = False
    last_p1_stocks = 4
    last_p2_stocks = 4
    last_frame = 0

    # Per-match assignment of A/B to ports. Starts with A on port 1.
    a_on_port_1 = True
    p1_state: Optional[PlayerState] = None
    p2_state: Optional[PlayerState] = None

    def _new_match_states():
        """Allocate fresh PlayerStates for the current port assignment."""
        nonlocal p1_state, p2_state
        if a_on_port_1:
            p1_state = PlayerState(model_a, cfg_a.max_seq_len, device, ctx=ctx_a)
            p2_state = PlayerState(model_b, cfg_b.max_seq_len, device, ctx=ctx_b)
        else:
            p1_state = PlayerState(model_b, cfg_b.max_seq_len, device, ctx=ctx_b)
            p2_state = PlayerState(model_a, cfg_a.max_seq_len, device, ctx=ctx_a)

    _new_match_states()

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
                # Determine A's stocks based on the port it was on.
                a_stocks = last_p1_stocks if a_on_port_1 else last_p2_stocks
                b_stocks = last_p2_stocks if a_on_port_1 else last_p1_stocks
                if a_stocks > 0 and b_stocks == 0:
                    result = "a_wins"
                elif b_stocks > 0 and a_stocks == 0:
                    result = "b_wins"
                else:
                    result = "draw"
                t_match_end = time.time()
                meta = {
                    "match_idx": matches_started,
                    "a_on_port": 1 if a_on_port_1 else 2,
                    "result": result,
                    "a_stocks": a_stocks, "b_stocks": b_stocks,
                    "frames": last_frame,
                    "wall_seconds": round(t_match_end - t_match_start, 1),
                }
                results.append(meta)
                log.info("match %d/%d done: A=%d B=%d  %s  (A on port %d, %ds)",
                         len(results), n_matches,
                         a_stocks, b_stocks, result.upper(),
                         meta["a_on_port"], meta["wall_seconds"])
                in_game = False
                if alternate_ports:
                    a_on_port_1 = not a_on_port_1
                _new_match_states()
            # Costume per port follows the A/B assignment so the
            # visual identity tracks the policy across port flips.
            p1_costume = a_costume if a_on_port_1 else b_costume
            p2_costume = b_costume if a_on_port_1 else a_costume
            menu_1.menu_helper_simple(gs, ctrl_1, CHAR, STAGE,
                                      cpu_level=0, autostart=False,
                                      costume=p1_costume)
            menu_2.menu_helper_simple(gs, ctrl_2, CHAR, STAGE,
                                      cpu_level=0, autostart=True,
                                      costume=p2_costume)
            ctrl_1.flush()
            ctrl_2.flush()
            continue

        if not in_game:
            in_game = True
            matches_started += 1
            t_match_start = time.time()
            log.info("match %d/%d starting  (A on port %d)",
                     matches_started, n_matches, 1 if a_on_port_1 else 2)

        # Track live stocks every frame so the menu-transition reads the
        # last in-game value.
        ps1 = gs.players.get(1)
        ps2 = gs.players.get(2)
        if ps1 is not None:
            last_p1_stocks = int(ps1.stock)
        if ps2 is not None:
            last_p2_stocks = int(ps2.stock)
        last_frame = int(gs.frame)

        players = sorted(gs.players.items())
        if len(players) < 2:
            continue

        # Build frames from each port's perspective.
        if a_on_port_1:
            frame_1 = build_frame(gs, p1_state.prev_sent, ctx_a)     # A on port 1
            frame_2 = build_frame_p2(gs, p2_state.prev_sent, ctx_b)  # B on port 2
        else:
            frame_1 = build_frame(gs, p1_state.prev_sent, ctx_b)     # B on port 1
            frame_2 = build_frame_p2(gs, p2_state.prev_sent, ctx_a)  # A on port 2

        if frame_1 is None or frame_2 is None:
            continue

        p1_state.push_frame(frame_1)
        p2_state.push_frame(frame_2)
        preds_1 = p1_state.predict()
        preds_2 = p2_state.predict()
        new_sent_1, _, _ = decode_and_press(
            ctrl_1, preds_1, p1_state.prev_sent, temperature=temperature)
        new_sent_2, _, _ = decode_and_press(
            ctrl_2, preds_2, p2_state.prev_sent, temperature=temperature)
        p1_state.prev_sent = new_sent_1
        p2_state.prev_sent = new_sent_2

    console.stop()

    t_total = time.time() - t_start
    a_wins = sum(1 for r in results if r["result"] == "a_wins")
    b_wins = sum(1 for r in results if r["result"] == "b_wins")
    draws = sum(1 for r in results if r["result"] == "draw")
    n = len(results)

    report = {
        "ckpt_a": str(ckpt_a),
        "ckpt_b": str(ckpt_b),
        "data_dir": str(data_dir),
        "n_matches": n,
        "a_wins": a_wins, "b_wins": b_wins, "draws": draws,
        "a_win_rate": a_wins / max(1, n),
        "avg_a_stocks": (sum(r["a_stocks"] for r in results) / max(1, n)),
        "avg_b_stocks": (sum(r["b_stocks"] for r in results) / max(1, n)),
        "avg_frames": sum(r["frames"] for r in results) / max(1, n),
        "wall_seconds_total": round(t_total, 1),
        "config": {
            "bot_character": bot_character,
            "stage": stage,
            "temperature": temperature,
            "use_exi_inputs": use_exi_inputs,
            "enable_ffw": enable_ffw,
            "gfx_backend": gfx_backend,
            "alternate_ports": alternate_ports,
        },
        "matches": results,
    }
    log.info("DONE. N=%d | A=%d B=%d D=%d | A-win-rate=%.1f%% | "
             "avg_stocks A=%.2f B=%.2f | total=%.0fs",
             n, a_wins, b_wins, draws, 100.0 * report["a_win_rate"],
             report["avg_a_stocks"], report["avg_b_stocks"], t_total)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt-a", required=True, type=Path,
                    help="Policy A (the one whose win-rate we report).")
    ap.add_argument("--ckpt-b", required=True, type=Path,
                    help="Policy B (the opponent — typically BC baseline).")
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--dolphin-path", required=True, type=Path)
    ap.add_argument("--iso-path", required=True, type=Path)
    ap.add_argument("--n-matches", type=int, default=200)
    ap.add_argument("--bot-character", default="FOX")
    ap.add_argument("--stage", default="FINAL_DESTINATION")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--use-exi-inputs", action="store_true",
                    help="Required for --enable-ffw. Use emulator_ffw/ build.")
    ap.add_argument("--enable-ffw", action="store_true",
                    help="2.28x speedup but unwatchable (Null gfx required).")
    ap.add_argument("--gfx-backend", default="",
                    help="Empty -> Dolphin default (OpenGL on Linux), "
                         "which is the watchable mode here. Use Null "
                         "with --enable-ffw for headless training-style "
                         "runs. Vulkan tends to artifact on this box, "
                         "avoid for live-viewing runs.")
    ap.add_argument("--disable-audio", action="store_true")
    ap.add_argument("--a-costume", type=int, default=3,
                    help="Costume index for policy A (Fox: 0=default, "
                         "1=red, 2=black/blue, 3=green). Default 3 (green).")
    ap.add_argument("--b-costume", type=int, default=0,
                    help="Costume index for policy B. Default 0 (default Fox).")
    ap.add_argument("--alternate-ports", action="store_true",
                    help="Flip A/B port assignment every match to remove "
                         "port-handedness bias.")
    ap.add_argument("--replay-dir", type=Path, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    run_h2h(
        ckpt_a=args.ckpt_a, ckpt_b=args.ckpt_b,
        data_dir=args.data_dir,
        dolphin_path=args.dolphin_path, iso_path=args.iso_path,
        n_matches=args.n_matches,
        bot_character=args.bot_character, stage=args.stage,
        temperature=args.temperature,
        use_exi_inputs=args.use_exi_inputs, enable_ffw=args.enable_ffw,
        gfx_backend=args.gfx_backend, disable_audio=args.disable_audio,
        alternate_ports=args.alternate_ports,
        a_costume=args.a_costume, b_costume=args.b_costume,
        replay_dir=args.replay_dir, device=args.device,
        out=args.out,
    )


if __name__ == "__main__":
    main()
