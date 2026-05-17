"""Play Melee matches: a MIMIC policy vs a CPU or a second MIMIC policy.

One Dolphin instance, N back-to-back matches, optional win-rate tally.
This consolidates the four former match-runner scripts — the 2x2 of
{opponent: CPU | checkpoint} x {mode: watch one | tally N}:

    OLD SCRIPT              NEW INVOCATION
    play_vs_cpu.py          --opponent cpu:9   --n-matches 1
    head_to_head.py         --opponent <ckpt>  --n-matches 1
    winrate_vs_cpu.py       --opponent cpu:9   --n-matches 50  --out r.json
    head_to_head_winrate.py --opponent <ckpt>  --n-matches 200 --out r.json

`--n-matches 1` is the watchable single-game mode (add `--verbose` for
per-frame controller logging); `--n-matches N` plays N back-to-back and
tallies. A JSON report is written whenever `--out` is given (emitted
incrementally after every match, so a mid-run crash leaves a partial
report). The reported win-rate is always from `--ckpt`'s perspective
("A").

`--opponent` accepts `cpu`, `cpu:<level>` (1-9), or a checkpoint path.

Examples:
    # Watch the bot fight a level-9 CPU on Battlefield, verbose:
    python tools/play.py --ckpt checkpoints/falco-20260412-relpos-28k.pt \\
        --opponent cpu:9 --data-dir data/falco_v2 \\
        --dolphin-path emulator/squashfs-root/usr/bin/dolphin-emu \\
        --iso-path melee.iso --stage BATTLEFIELD --verbose

    # 200-match head-to-head win-rate vs a frozen BC checkpoint, headless FFW:
    DISPLAY=:99 python tools/play.py \\
        --ckpt checkpoints/3way_20260514/3way-20260514-comboext_final.pt \\
        --opponent hf_checkpoints/fox/model.pt --data-dir hf_checkpoints/fox \\
        --dolphin-path emulator_ffw/squashfs-root/usr/bin/dolphin-emu \\
        --iso-path melee.iso --n-matches 200 --alternate-ports \\
        --use-exi-inputs --enable-ffw --gfx-backend Null \\
        --out reports/h2h.json

Headless FFW mode: DISPLAY=:99 + Null gfx + emulator_ffw/ +
--use-exi-inputs + --enable-ffw. Realtime is ~80s/match; FFW ~35s/match.
Watchable mode: DISPLAY=:0, leave --gfx-backend blank (Dolphin picks
OpenGL on Linux); Vulkan tends to artifact on this box.
"""
from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import melee  # noqa: E402

from tools.inference_utils import (  # noqa: E402
    load_inference_context, load_mimic_model,
    build_frame, build_frame_p2,
    PlayerState, decode_and_press,
)


log = logging.getLogger("mimic.play")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  [%(levelname)s]  %(message)s")


def _parse_opponent(spec: str) -> Tuple[bool, int, Optional[Path]]:
    """Parse --opponent into (is_cpu, cpu_level, ckpt_path).

    'cpu' -> level 9; 'cpu:N' -> level N; anything else -> checkpoint path.
    """
    s = spec.strip()
    if s.lower() == "cpu":
        return True, 9, None
    if s.lower().startswith("cpu:"):
        lvl = int(s.split(":", 1)[1])
        if not 1 <= lvl <= 9:
            raise ValueError(f"CPU level must be 1-9, got {lvl}")
        return True, lvl, None
    p = Path(s)
    if not p.exists():
        raise FileNotFoundError(
            f"--opponent '{spec}' is neither 'cpu[:level]' nor an existing "
            f"checkpoint path")
    return False, 0, p


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


def _safe_stop(console):
    """Stop the console, working around a libmelee slippstream deadlock.

    libmelee's SlippstreamClient runs a worker subprocess that pushes
    Dolphin events through an mp.Pipe. Once the match loop ends we stop
    draining that pipe; under FFW the worker fills the ~64KB pipe buffer
    almost instantly and blocks in send_bytes(), past the point in its
    loop where it checks the shutdown flag. console.stop() then calls
    _slippstream.shutdown() -> _worker.join(), which waits forever for
    that wedged worker. SIGKILLing the worker first (SIGKILL can't be
    blocked) lets join() reap it immediately.
    """
    try:
        ss = getattr(console, "_slippstream", None)
        worker = getattr(ss, "_worker", None) if ss is not None else None
        if worker is not None and worker.is_alive():
            worker.kill()
    except Exception as e:
        log.warning("slippstream worker pre-kill failed: %s", e)
    console.stop()


def _ctrl_str(s):
    """Full controller output for a trace line: sticks + shoulders + 7 button bits."""
    bb = "".join(str(s.get(f"btn_{b}", 0)) for b in
                 ("BUTTON_A", "BUTTON_B", "BUTTON_X", "BUTTON_Y",
                  "BUTTON_Z", "BUTTON_L", "BUTTON_R"))
    return (f"{s['main_x']:.3f} {s['main_y']:.3f} {s['c_x']:.3f} {s['c_y']:.3f} "
            f"{s['l_shldr']:.3f} {s['r_shldr']:.3f} {bb}")


def run(
    ckpt: Path,
    opponent: str,
    data_dir: Path,
    dolphin_path: Path,
    iso_path: Path,
    n_matches: int = 1,
    character: str = "FOX",
    opponent_character: str = "FOX",
    stage: str = "FINAL_DESTINATION",
    temperature: float = 1.0,
    use_exi_inputs: bool = False,
    enable_ffw: bool = False,
    gfx_backend: str = "",
    disable_audio: bool = False,
    alternate_ports: bool = False,
    a_costume: int = 3,  # green Fox by default — visually distinct from B
    b_costume: int = 0,
    port_a: int = 1,
    port_b: int = 2,
    slippi_port: int = 51441,
    replay_dir: Optional[Path] = None,
    device: str = "cuda",
    out: Optional[Path] = None,
    seed: Optional[int] = None,
    trace: Optional[Path] = None,
    verbose: bool = False,
) -> dict:
    """Play `n_matches` and return a report dict.

    A = the `--ckpt` policy. B = the opponent (a CPU or a second policy).
    """
    opp_is_cpu, cpu_level, ckpt_b = _parse_opponent(opponent)

    if replay_dir is None:
        replay_dir = Path("replays")
    replay_dir.mkdir(parents=True, exist_ok=True)

    # Deterministic decode + per-frame trace, so a single-instance run and
    # a concurrent run can be diffed frame-by-frame.
    if seed is not None:
        torch.manual_seed(seed)
    trace_fh = open(trace, "w") if trace is not None else None

    log.info("loading A (policy): %s", ckpt)
    model_a, cfg_a = load_mimic_model(str(ckpt), device)
    ctx_base = load_inference_context(data_dir)
    ctx_a = _make_ctx(ctx_base, cfg_a)

    if opp_is_cpu:
        log.info("opponent B: CPU level %d", cpu_level)
        model_b = cfg_b = ctx_b = None
    else:
        log.info("loading B (policy): %s", ckpt_b)
        model_b, cfg_b = load_mimic_model(str(ckpt_b), device)
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
        slippi_port=slippi_port,
    )
    ctrl_pa = melee.Controller(console=console, port=port_a,
                               type=melee.ControllerType.STANDARD)
    ctrl_pb = melee.Controller(console=console, port=port_b,
                               type=melee.ControllerType.STANDARD)
    console.run(iso_path=str(iso_path))
    console.connect()
    ctrl_pa.connect()
    ctrl_pb.connect()
    log.info("Dolphin connected, exi=%s ffw=%s gfx=%s alternate_ports=%s",
             use_exi_inputs, enable_ffw, gfx_backend or "<default>",
             alternate_ports)

    menu_pa = melee.MenuHelper()
    menu_pb = melee.MenuHelper()
    A_CHAR = melee.Character[character]
    B_CHAR = melee.Character[opponent_character]
    STAGE = melee.Stage[stage]

    results = []
    matches_started = 0
    in_game = False
    last_pa_stocks = 4
    last_pb_stocks = 4
    last_frame = 0

    # Per-match assignment of A to a port. Starts with A on port_a.
    a_on_port_a = True
    a_state: Optional[PlayerState] = None
    b_state: Optional[PlayerState] = None

    def _new_match_states():
        """Allocate fresh PlayerStates for the current port assignment."""
        nonlocal a_state, b_state
        a_state = PlayerState(model_a, cfg_a.max_seq_len, device, ctx=ctx_a)
        b_state = (PlayerState(model_b, cfg_b.max_seq_len, device, ctx=ctx_b)
                   if model_b is not None else None)

    _new_match_states()

    t_start = time.time()
    t_match_start = time.time()

    def shutdown(*_):
        try:
            _safe_stop(console)
        finally:
            sys.exit(1)
    signal.signal(signal.SIGINT, shutdown)

    def _emit_report():
        """Build the report from results-so-far and write it (if --out).
        Called after every match so a mid-run Dolphin crash/freeze leaves
        a usable partial report instead of losing all N matches."""
        n_ = len(results)
        a_w = sum(1 for r in results if r["result"] == "a_wins")
        b_w = sum(1 for r in results if r["result"] == "b_wins")
        d_ = sum(1 for r in results if r["result"] == "draw")
        rep = {
            "ckpt_a": str(ckpt),
            "ckpt_b": (f"cpu:{cpu_level}" if opp_is_cpu else str(ckpt_b)),
            "data_dir": str(data_dir),
            "n_matches": n_,
            "n_matches_requested": n_matches,
            "a_wins": a_w, "b_wins": b_w, "draws": d_,
            "a_win_rate": a_w / max(1, n_),
            "avg_a_stocks": sum(r["a_stocks"] for r in results) / max(1, n_),
            "avg_b_stocks": sum(r["b_stocks"] for r in results) / max(1, n_),
            "avg_frames": sum(r["frames"] for r in results) / max(1, n_),
            "wall_seconds_total": round(time.time() - t_start, 1),
            "config": {
                "opponent": (f"cpu:{cpu_level}" if opp_is_cpu else str(ckpt_b)),
                "character": character,
                "opponent_character": opponent_character,
                "stage": stage, "temperature": temperature,
                "use_exi_inputs": use_exi_inputs, "enable_ffw": enable_ffw,
                "gfx_backend": gfx_backend, "alternate_ports": alternate_ports,
                "port_a": port_a, "port_b": port_b,
            },
            "matches": results,
        }
        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(rep, indent=2))
        return rep

    # --- FFW input-sync diagnostic ---
    _fd_hist = Counter()
    _prev_gf = None
    _fd_warns = 0

    while len(results) < n_matches:
        gs = console.step()
        if gs is None:
            continue

        if gs.menu_state not in (melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH):
            if in_game:
                # A's stocks depend on the port it was on this match.
                a_stocks = last_pa_stocks if a_on_port_a else last_pb_stocks
                b_stocks = last_pb_stocks if a_on_port_a else last_pa_stocks
                if a_stocks > 0 and b_stocks == 0:
                    result = "a_wins"
                elif b_stocks > 0 and a_stocks == 0:
                    result = "b_wins"
                else:
                    result = "draw"
                t_match_end = time.time()
                meta = {
                    "match_idx": matches_started,
                    "a_on_port": port_a if a_on_port_a else port_b,
                    "result": result,
                    "a_stocks": a_stocks, "b_stocks": b_stocks,
                    "frames": last_frame,
                    "wall_seconds": round(t_match_end - t_match_start, 1),
                }
                results.append(meta)
                _emit_report()  # incremental: survive a later crash/freeze
                log.info("match %d/%d done: A=%d B=%d  %s  (A on port %d, %ds)",
                         len(results), n_matches,
                         a_stocks, b_stocks, result.upper(),
                         meta["a_on_port"], meta["wall_seconds"])
                if enable_ffw:
                    log.info("  frame-delta histogram: %s",
                             dict(sorted(_fd_hist.items())))
                _fd_hist.clear()
                _prev_gf = None
                in_game = False
                if alternate_ports:
                    a_on_port_a = not a_on_port_a
                _new_match_states()

            # Costume / character / cpu-level per port follow the A/B
            # assignment so visual identity tracks the policy across flips.
            # autostart fires on port_b (the last menu handled).
            if a_on_port_a:
                pa = (A_CHAR, a_costume, 0)
                pb = (B_CHAR, b_costume, cpu_level if opp_is_cpu else 0)
            else:
                pa = (B_CHAR, b_costume, cpu_level if opp_is_cpu else 0)
                pb = (A_CHAR, a_costume, 0)
            menu_pa.menu_helper_simple(gs, ctrl_pa, pa[0], STAGE,
                                       cpu_level=pa[2], autostart=False,
                                       costume=pa[1])
            menu_pb.menu_helper_simple(gs, ctrl_pb, pb[0], STAGE,
                                       cpu_level=pb[2], autostart=True,
                                       costume=pb[1])
            ctrl_pa.flush()
            ctrl_pb.flush()
            continue

        if not in_game:
            in_game = True
            matches_started += 1
            t_match_start = time.time()
            log.info("match %d/%d starting  (A on port %d)",
                     matches_started, n_matches,
                     port_a if a_on_port_a else port_b)

        # Track live stocks every frame so the menu-transition reads the
        # last in-game value.
        ps_pa = gs.players.get(port_a)
        ps_pb = gs.players.get(port_b)
        if ps_pa is not None:
            last_pa_stocks = int(ps_pa.stock)
        if ps_pb is not None:
            last_pb_stocks = int(ps_pb.stock)
        last_frame = int(gs.frame)

        # FFW sync probe: a properly-awaited loop advances gs.frame by
        # exactly 1 per step. delta > 1 == Dolphin advanced frames
        # without waiting for our input.
        if _prev_gf is not None:
            _d = last_frame - _prev_gf
            if _d > 0:
                _fd_hist[_d] += 1
                if _d != 1 and _fd_warns < 40:
                    _fd_warns += 1
                    log.warning("frame delta=%d (%d->%d) processingtime=%.4fs",
                                _d, _prev_gf, last_frame,
                                console.processingtime)
        _prev_gf = last_frame

        if len(gs.players) < 2:
            continue

        # The player on port_a builds from the "p1" perspective, port_b
        # from the "p2" perspective. A is on port_a iff a_on_port_a.
        ctrl_a = ctrl_pa if a_on_port_a else ctrl_pb
        ctrl_b = ctrl_pb if a_on_port_a else ctrl_pa
        a_build = build_frame if a_on_port_a else build_frame_p2
        b_build = build_frame_p2 if a_on_port_a else build_frame

        frame_a = a_build(gs, a_state.prev_sent, ctx_a)
        if frame_a is None:
            continue
        frame_b = None
        if b_state is not None:
            frame_b = b_build(gs, b_state.prev_sent, ctx_b)
            if frame_b is None:
                continue

        a_state.push_frame(frame_a)
        preds_a = a_state.predict()
        new_sent_a, pressed_a, btn_a = decode_and_press(
            ctrl_a, preds_a, a_state.prev_sent, temperature=temperature)
        a_state.prev_sent = new_sent_a

        new_sent_b = None
        if b_state is not None:
            b_state.push_frame(frame_b)
            preds_b = b_state.predict()
            new_sent_b, _, _ = decode_and_press(
                ctrl_b, preds_b, b_state.prev_sent, temperature=temperature)
            b_state.prev_sent = new_sent_b
        # When B is a CPU it is driven by the game; we never press ctrl_b.

        ps_a = ps_pa if a_on_port_a else ps_pb
        ps_b = ps_pb if a_on_port_a else ps_pa

        if verbose:
            tk = min(3, len(btn_a))
            top = F.softmax(preds_a["btn_logits"][0, -1].float()
                            / max(temperature, 1e-6), dim=-1).topk(tk)
            top_str = " ".join(f"{btn_a[i]}={v:.3f}" for v, i in
                                zip(top.values.tolist(), top.indices.tolist()))
            extra = ""
            if ps_a is not None and ps_b is not None:
                extra = (f"  A={ps_a.stock}stk({ps_a.percent:.0f}%) "
                         f"B={ps_b.stock}stk({ps_b.percent:.0f}%)")
            log.info("[f%d] A MAIN=(%.2f,%.2f) L=%.2f BTN=%s top3=[%s]%s",
                     last_frame, new_sent_a["main_x"], new_sent_a["main_y"],
                     new_sent_a["l_shldr"], pressed_a, top_str, extra)
        elif n_matches == 1 and last_frame % 60 == 0 and \
                ps_a is not None and ps_b is not None:
            log.info("[f%d]  A %dstk %.0f%%  |  B %dstk %.0f%%",
                     last_frame, ps_a.stock, ps_a.percent,
                     ps_b.stock, ps_b.percent)

        if trace_fh is not None and ps_a is not None and ps_b is not None:
            b_ctrl = _ctrl_str(new_sent_b) if new_sent_b is not None else "cpu"
            trace_fh.write(
                f"{int(gs.frame)} "
                f"{ps_a.position.x:.3f} {ps_a.position.y:.3f} "
                f"{ps_a.percent:.1f} {ps_a.stock} {ps_a.action.value} "
                f"{ps_b.position.x:.3f} {ps_b.position.y:.3f} "
                f"{ps_b.percent:.1f} {ps_b.stock} {ps_b.action.value} "
                f"A {_ctrl_str(new_sent_a)} B {b_ctrl}\n")

    _safe_stop(console)
    if trace_fh is not None:
        trace_fh.close()

    report = _emit_report()
    log.info("DONE. N=%d | A=%d B=%d D=%d | A-win-rate=%.1f%% | "
             "avg_stocks A=%.2f B=%.2f | total=%.0fs",
             report["n_matches"], report["a_wins"], report["b_wins"],
             report["draws"], 100.0 * report["a_win_rate"],
             report["avg_a_stocks"], report["avg_b_stocks"],
             report["wall_seconds_total"])
    return report


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, type=Path,
                    help="Policy A — the checkpoint whose win-rate is reported.")
    ap.add_argument("--opponent", default="cpu:9",
                    help="Player B: 'cpu', 'cpu:<level>' (1-9), or a "
                         "checkpoint path. Default: cpu:9.")
    ap.add_argument("--data-dir", required=True, type=Path,
                    help="Inference-context dir (norm/combos); used for "
                         "both policies.")
    ap.add_argument("--dolphin-path", required=True, type=Path)
    ap.add_argument("--iso-path", required=True, type=Path)
    ap.add_argument("--n-matches", type=int, default=1,
                    help="1 = single watchable game; N = play N and tally.")
    ap.add_argument("--character", default="FOX", help="Character for A.")
    ap.add_argument("--opponent-character", default="FOX",
                    help="Character for B (CPU or policy).")
    ap.add_argument("--stage", default="FINAL_DESTINATION")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--use-exi-inputs", action="store_true",
                    help="Required for --enable-ffw. Use emulator_ffw/ build.")
    ap.add_argument("--enable-ffw", action="store_true",
                    help="~2.3x speedup but unwatchable (Null gfx required).")
    ap.add_argument("--gfx-backend", default="",
                    help="Empty -> Dolphin default (OpenGL on Linux, the "
                         "watchable mode). Use Null with --enable-ffw for "
                         "headless runs. Vulkan tends to artifact on this box.")
    ap.add_argument("--disable-audio", action="store_true")
    ap.add_argument("--alternate-ports", action="store_true",
                    help="Flip A/B port assignment every match to remove "
                         "port-handedness bias.")
    ap.add_argument("--a-costume", type=int, default=3,
                    help="Costume index for A (Fox: 0=default, 1=red, "
                         "2=black/blue, 3=green). Default 3 (green).")
    ap.add_argument("--b-costume", type=int, default=0,
                    help="Costume index for B. Default 0 (default Fox).")
    ap.add_argument("--port-a", type=int, default=1,
                    help="Slippi port for policy A (1-4). Default 1.")
    ap.add_argument("--port-b", type=int, default=2,
                    help="Slippi port for player B (1-4). Default 2.")
    ap.add_argument("--slippi-port", type=int, default=51441,
                    help="UDP port for this Dolphin's Slippi server. Give "
                         "concurrent instances distinct ports.")
    ap.add_argument("--replay-dir", type=Path, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=None,
                    help="Write a JSON win-rate report here (emitted "
                         "incrementally after each match).")
    ap.add_argument("--seed", type=int, default=None,
                    help="torch.manual_seed for deterministic decoding (debug).")
    ap.add_argument("--trace", type=Path, default=None,
                    help="Write a per-frame gamestate+decision trace here (debug).")
    ap.add_argument("--verbose", action="store_true",
                    help="Per-frame controller logging (the old play_vs_cpu "
                         "behavior). Default: a per-second status line when "
                         "--n-matches 1, silent otherwise.")
    args = ap.parse_args()

    run(
        ckpt=args.ckpt, opponent=args.opponent, data_dir=args.data_dir,
        dolphin_path=args.dolphin_path, iso_path=args.iso_path,
        n_matches=args.n_matches,
        character=args.character, opponent_character=args.opponent_character,
        stage=args.stage, temperature=args.temperature,
        use_exi_inputs=args.use_exi_inputs, enable_ffw=args.enable_ffw,
        gfx_backend=args.gfx_backend, disable_audio=args.disable_audio,
        alternate_ports=args.alternate_ports,
        a_costume=args.a_costume, b_costume=args.b_costume,
        port_a=args.port_a, port_b=args.port_b, slippi_port=args.slippi_port,
        replay_dir=args.replay_dir, device=args.device,
        out=args.out, seed=args.seed, trace=args.trace, verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
