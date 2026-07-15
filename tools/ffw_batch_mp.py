"""Multiprocess async batched FFW throughput bench — the real ceiling.

Removes the GIL wall of tools/ffw_batch_bench.py: each env runs in its OWN
process (no shared interpreter lock), so per-env build_frame + decode_and_press
run truly in parallel. A central process owns the ONLY model copy (on GPU),
maintains each env's context window, and does one batched forward across whatever
envs are ready each cycle (async — it does not block on the slowest env).

Wire protocol (per env, per frame), tiny payloads:
  env -> central : ('F', {key: np.ndarray})     # build_frame output as numpy
  central -> env : {head: np.ndarray(1,1,C)}     # 4 head logits, last timestep
                   or 'STOP'
Envs are model-free; central is Dolphin-free. Uses spawn so central can init
CUDA before env processes start (fork-after-CUDA is unsafe; spawn gives fresh
env procs that never touch CUDA).

Usage:  python3 tools/ffw_batch_mp.py --sweep 4,8,16,24 --seconds 15
"""
import argparse, time, sys
import multiprocessing as mp
from multiprocessing.connection import wait
import numpy as np

DOLPHIN = "emulator_ffw/squashfs-root/usr/bin/dolphin-emu"
ISO = "melee.iso"
DATA_DIR = "data/foxrank_master_v2"
CKPT = "checkpoints/AVG_mastfox.pt"
HEADS = ["main_xy", "shoulder_val", "c_dir_logits", "btn_logits"]


# ----------------------------- env process -----------------------------
def env_proc(env_id, conn, n_combos, slippi_port, stop_ev, counts, warmup=180,
             replay_dir=None):
    import melee
    from tools.inference_utils import (
        load_inference_context, build_frame, decode_and_press)
    import torch

    ctx = dict(load_inference_context(DATA_DIR))
    ctx["combo_map"] = {} if n_combos == 7 else ctx.get("combo_map", {})
    ctx["n_combos"] = n_combos

    con = melee.Console(
        path=DOLPHIN, is_dolphin=True, tmp_home_directory=True,
        copy_home_directory=False, blocking_input=True, online_delay=0,
        setup_gecko_codes=True, fullscreen=False, gfx_backend="Null",
        disable_audio=True, use_exi_inputs=True, enable_ffw=True,
        save_replays=(replay_dir is not None),
        replay_dir=(str(replay_dir) if replay_dir else None),
        slippi_port=slippi_port)
    cb = melee.Controller(console=con, port=1, type=melee.ControllerType.STANDARD)
    cc = melee.Controller(console=con, port=2, type=melee.ControllerType.STANDARD)
    con.run(iso_path=ISO); con.connect(); cb.connect(); cc.connect()
    m1, m2 = melee.MenuHelper(), melee.MenuHelper()
    FOX, FD = melee.Character.FOX, melee.Stage.FINAL_DESTINATION

    def menu(gs):
        m1.menu_helper_simple(gs, cb, FOX, FD, cpu_level=0, autostart=False, costume=0)
        m2.menu_helper_simple(gs, cc, FOX, FD, cpu_level=7, autostart=True, costume=1)
        cb.flush(); cc.flush()

    # reach in-game past countdown
    while not stop_ev.is_set():
        gs = con.step()
        if gs is None:
            continue
        if gs.menu_state in (melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH):
            if gs.frame > warmup and len(gs.players) >= 2:
                break
        else:
            menu(gs)
    conn.send(('READY', env_id))

    prev_sent = None
    c = 0
    while not stop_ev.is_set():
        gs = con.step()
        if gs is None:
            continue
        if gs.menu_state in (melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH) \
                and len(gs.players) >= 2:
            fr = build_frame(gs, prev_sent, ctx)
            if fr is None:
                continue
            conn.send(('F', {k: v.numpy() for k, v in fr.items()}))
            msg = conn.recv()
            if msg == 'STOP':
                break
            preds = {k: torch.from_numpy(a) for k, a in msg.items()}
            prev_sent, _, _ = decode_and_press(cb, preds, prev_sent)
            c += 1
            counts[env_id] = c
        else:
            menu(gs)
    try:
        con.stop()
    except Exception:
        pass


# ----------------------------- central --------------------------------
def run(n_envs, seconds, model, cfg, ctx, device, replay_dir=None):
    import torch
    from tools.inference_utils import build_mock_frame
    seq_len = cfg.max_seq_len
    n_combos = cfg.n_controller_combos

    ctxmp = mp.get_context("spawn")
    stop_ev = ctxmp.Event()
    counts = ctxmp.Array('i', n_envs)
    parent_conns, procs = [], []
    for i in range(n_envs):
        pc, cc = ctxmp.Pipe()
        p = ctxmp.Process(target=env_proc,
                          args=(i, cc, n_combos, 51441 + i, stop_ev, counts, 180,
                                replay_dir))
        p.start(); parent_conns.append(pc); procs.append(p)

    # per-env rolling window buffers: {key: (seq_len, *F)} tensor, shifted in
    # place (one vectorized copy per key per frame) instead of cat-ing 180
    # tiny tensors every frame.
    bufs = [None] * n_envs
    conn_env = {pc: i for i, pc in enumerate(parent_conns)}

    def push(i, frame):
        if bufs[i] is None:
            mock = build_mock_frame(ctx)
            bufs[i] = {k: v.expand(seq_len, *v.shape[1:]).clone()
                       for k, v in mock.items()}
        b = bufs[i]
        for k, v in frame.items():
            b[k][:-1] = b[k][1:].clone()
            b[k][-1] = v[0]

    # wait for all envs in-game
    pending = set(parent_conns); t_up = time.time()
    while pending:
        for c in wait(list(pending), timeout=1.0):
            try:
                msg = c.recv()
            except EOFError:
                pending.discard(c); continue
            if msg[0] == 'READY':
                pending.discard(c)
        if time.time() - t_up > 240:
            print(f"[N={n_envs}] bringup timeout, {len(pending)} not ready", flush=True)
            break
    print(f"[N={n_envs}] {n_envs} envs in-game ({time.time()-t_up:.0f}s), timing {seconds}s", flush=True)

    total = 0
    t0 = time.time()
    while time.time() - t0 < seconds:
        ready = wait(parent_conns, timeout=0.5)
        active, frames = [], {}
        for c in ready:
            try:
                tag, payload = c.recv()
            except EOFError:
                continue
            if tag == 'F':
                i = conn_env[c]
                push(i, {k: torch.from_numpy(a) for k, a in payload.items()})
                active.append(i)
        if not active:
            continue
        keys = bufs[active[0]].keys()
        mega = {k: torch.stack([bufs[i][k] for i in active], dim=0).to(device)
                for k in keys}
        with torch.no_grad():
            out = model(mega)
        for j, i in enumerate(active):
            resp = {k: out[k][j:j + 1, -1:, :].contiguous().cpu().numpy() for k in HEADS}
            parent_conns[i].send(resp)
    dt = time.time() - t0

    # shutdown: unblock any env waiting on recv, then stop
    stop_ev.set()
    t_drain = time.time()
    while time.time() - t_drain < 4:
        ready = wait(parent_conns, timeout=0.2)
        for c in ready:
            try:
                c.recv()
            except (EOFError, OSError):
                continue
            try:
                c.send('STOP')
            except (BrokenPipeError, OSError):
                pass
    for p in procs:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()

    total = sum(counts[i] for i in range(n_envs))
    agg = total / dt
    print(f"[N={n_envs}] wall={dt:.1f}s  in-game-frames={total}  "
          f"AGG={agg:.0f} fps ({agg/60:.1f}x realtime)  "
          f"per-env={agg/n_envs:.0f} fps ({agg/n_envs/60:.2f}x)", flush=True)
    return {"n": n_envs, "agg_fps": agg, "per_env_fps": agg / n_envs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", type=str, default="8")
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--replay-dir", type=str, default=None,
                    help="If set, all envs save .slp here (for L-cancel analysis).")
    ap.add_argument("--ckpt", type=str, default=CKPT,
                    help="Checkpoint to roll out (default: production AVG_mastfox).")
    args = ap.parse_args()
    if args.replay_dir:
        import os
        os.makedirs(args.replay_dir, exist_ok=True)

    import torch
    from tools.inference_utils import load_mimic_model, load_inference_context
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_mimic_model(args.ckpt, device); model.eval()
    ctx = dict(load_inference_context(DATA_DIR))
    ctx["combo_map"] = {}; ctx["n_combos"] = cfg.n_controller_combos
    print(f"model loaded seq_len={cfg.max_seq_len} combos={cfg.n_controller_combos} device={device}", flush=True)

    Ns = [int(x) for x in args.sweep.split(",")]
    results = [run(n, args.seconds, model, cfg, ctx, device, args.replay_dir) for n in Ns]
    print("\n=== SCALING (multiprocess async batched) ===", flush=True)
    for r in results:
        print(f"  N={r['n']:2d}: {r['agg_fps']:5.0f} fps agg ({r['agg_fps']/60:.1f}x)  "
              f"{r['per_env_fps']:4.0f} fps/env ({r['per_env_fps']/60:.2f}x)", flush=True)


if __name__ == "__main__":
    main()
