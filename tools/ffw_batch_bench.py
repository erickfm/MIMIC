"""Batched multi-env FFW throughput bench.

Measures the real parallel-Dolphin scaling ceiling with the architecture that
actually matters for RL: ONE process, N Dolphins, and a SINGLE batched model
forward across all N envs each frame (vs the process-fan-out in play.py, which
runs N independent batch-1 forwards and is GPU-bound — see
docs/research-notes-2026-06-30.md).

Design: N worker threads each own one Console+bot-controller (bot on port 1,
CPU on port 2, FFW headless). libmelee's console.step() blocks in a C socket
read that releases the GIL, so the N steps overlap. Two barriers per frame
synchronize the round:
  1. all workers have stepped + built their frame,
  2. the coordinator has run one batched forward and scattered per-env preds,
then each worker decodes + presses. In-game frames are counted per env;
aggregate fps = total in-game frames / timed wall.

Usage:  python3 tools/ffw_batch_bench.py --n-envs 8 --rounds 1500
        python3 tools/ffw_batch_bench.py --sweep 1,4,8
"""
import argparse, threading, time, sys
from pathlib import Path
import torch
import melee
from tools.inference_utils import (
    load_mimic_model, load_inference_context, build_frame,
    PlayerState, decode_and_press,
)

DOLPHIN = "emulator_ffw/squashfs-root/usr/bin/dolphin-emu"
ISO = "melee.iso"
DATA_DIR = "data/foxrank_master_v2"
CKPT = "checkpoints/AVG_mastfox.pt"


def _make_ctx(ctx_base, cfg):
    from mimic.features import BTN7_N_CLASSES
    ctx = dict(ctx_base)
    if cfg.n_controller_combos == BTN7_N_CLASSES:
        ctx["combo_map"] = {}; ctx["n_combos"] = cfg.n_controller_combos
    return ctx


def env_batch(state):
    """Build the (1, seq_len, F) per-key dict from a PlayerState cache (the
    forward-less half of PlayerState.predict)."""
    frames = list(state._frame_cache)
    return {k: torch.cat([f[k] for f in frames], dim=0).unsqueeze(0)
            for k in frames[0]}


def run(n_envs, rounds, model, cfg, ctx, device, warmup_frames=180):
    seq_len = cfg.max_seq_len
    consoles, ctrls_bot, states = [], [], []
    menus_p1, menus_p2, ctrls_cpu = [], [], []

    # --- bring up N Dolphins ---
    for i in range(n_envs):
        con = melee.Console(
            path=DOLPHIN, is_dolphin=True, tmp_home_directory=True,
            copy_home_directory=False, blocking_input=True, online_delay=0,
            setup_gecko_codes=True, fullscreen=False, gfx_backend="Null",
            disable_audio=True, use_exi_inputs=True, enable_ffw=True,
            save_replays=False, slippi_port=51441 + i,
        )
        cb = melee.Controller(console=con, port=1, type=melee.ControllerType.STANDARD)
        cc = melee.Controller(console=con, port=2, type=melee.ControllerType.STANDARD)
        con.run(iso_path=ISO); con.connect(); cb.connect(); cc.connect()
        consoles.append(con); ctrls_bot.append(cb); ctrls_cpu.append(cc)
        menus_p1.append(melee.MenuHelper()); menus_p2.append(melee.MenuHelper())
        states.append(PlayerState(model, seq_len, device, ctx=ctx))
    print(f"[N={n_envs}] {n_envs} Dolphins connected", flush=True)

    # --- menu -> all envs IN_GAME and past the countdown ---
    FOX = melee.Character.FOX; FD = melee.Stage.FINAL_DESTINATION
    ready = [False] * n_envs
    t_menu = time.time()
    while not all(ready):
        for i, con in enumerate(consoles):
            if ready[i]:
                continue
            gs = con.step()
            if gs is None:
                continue
            if gs.menu_state in (melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH):
                if gs.frame > warmup_frames and len(gs.players) >= 2:
                    ready[i] = True
                continue
            menus_p1[i].menu_helper_simple(gs, ctrls_bot[i], FOX, FD, cpu_level=0,
                                           autostart=False, costume=0)
            menus_p2[i].menu_helper_simple(gs, ctrls_cpu[i], FOX, FD, cpu_level=7,
                                           autostart=True, costume=1)
            ctrls_bot[i].flush(); ctrls_cpu[i].flush()
        if time.time() - t_menu > 180:
            print(f"[N={n_envs}] menu timeout, ready={ready}", flush=True); break
    print(f"[N={n_envs}] all in-game ({time.time()-t_menu:.0f}s menu), timing {rounds} rounds", flush=True)

    # --- timed phase: N workers + coordinator, batched forward per round ---
    slot = [None] * n_envs        # per-env batch dict (or None if not in-game)
    pred = [None] * n_envs        # per-env sliced preds
    counts = [0] * n_envs
    b1 = threading.Barrier(n_envs + 1)
    b2 = threading.Barrier(n_envs + 1)

    def worker(i):
        con = consoles[i]; cb = ctrls_bot[i]; st = states[i]
        try:
            for _ in range(rounds):
                gs = con.step()
                ok = False
                if gs is not None:
                    if gs.menu_state in (melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH) \
                            and len(gs.players) >= 2:
                        fr = build_frame(gs, st.prev_sent, ctx)
                        if fr is not None:
                            st.push_frame(fr); slot[i] = env_batch(st); ok = True
                    else:  # match ended — keep menu alive so barriers stay balanced
                        menus_p1[i].menu_helper_simple(gs, cb, FOX, FD, cpu_level=0,
                                                       autostart=False, costume=0)
                        menus_p2[i].menu_helper_simple(gs, ctrls_cpu[i], FOX, FD,
                                                       cpu_level=7, autostart=True, costume=1)
                        cb.flush(); ctrls_cpu[i].flush()
                if not ok:
                    slot[i] = None
                b1.wait()          # coordinator runs the batched forward here
                b2.wait()
                if pred[i] is not None:
                    st.prev_sent, _, _ = decode_and_press(cb, pred[i], st.prev_sent)
                    counts[i] += 1
        except threading.BrokenBarrierError:
            return
        except Exception as e:
            print(f"[N={n_envs}] worker {i} died: {e!r}", flush=True)
            b1.abort(); b2.abort()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_envs)]
    gpu_util = []
    t0 = time.time()
    for t in threads:
        t.start()
    try:
        for r in range(rounds):
            b1.wait()
            active = [i for i in range(n_envs) if slot[i] is not None]
            for i in range(n_envs):
                pred[i] = None
            if active:
                keys = slot[active[0]].keys()
                mega = {k: torch.cat([slot[i][k] for i in active], dim=0).to(device)
                        for k in keys}
                with torch.no_grad():
                    out = model(mega)
                for j, i in enumerate(active):
                    pred[i] = {k: v[j:j + 1] for k, v in out.items()}
            b2.wait()
    except threading.BrokenBarrierError:
        print(f"[N={n_envs}] barrier broken — a worker failed; aborting run", flush=True)
    dt = time.time() - t0
    for t in threads:
        t.join()
    for con in consoles:
        try:
            con.stop()
        except Exception:
            pass

    total = sum(counts)
    agg = total / dt
    print(f"[N={n_envs}] wall={dt:.1f}s  in-game-frames={total}  "
          f"AGG={agg:.0f} fps ({agg/60:.1f}x realtime)  "
          f"per-env={agg/n_envs:.0f} fps ({agg/n_envs/60:.2f}x)", flush=True)
    return {"n": n_envs, "wall": dt, "frames": total, "agg_fps": agg,
            "per_env_fps": agg / n_envs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=1500)
    ap.add_argument("--sweep", type=str, default="", help="comma list, e.g. 1,4,8")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_mimic_model(CKPT, device)
    model.eval()
    ctx = _make_ctx(load_inference_context(DATA_DIR), cfg)
    print(f"model loaded, seq_len={cfg.max_seq_len}, device={device}", flush=True)

    Ns = [int(x) for x in args.sweep.split(",")] if args.sweep else [args.n_envs]
    results = []
    for n in Ns:
        results.append(run(n, args.rounds, model, cfg, ctx, device))
    print("\n=== SCALING (batched multi-env, one forward/round) ===", flush=True)
    for r in results:
        print(f"  N={r['n']:2d}: {r['agg_fps']:5.0f} fps agg ({r['agg_fps']/60:.1f}x)  "
              f"{r['per_env_fps']:4.0f} fps/env ({r['per_env_fps']/60:.2f}x)", flush=True)


if __name__ == "__main__":
    main()
