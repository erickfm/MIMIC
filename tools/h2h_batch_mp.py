"""Batched multi-env FFW head-to-head — the throughput fix for the h2h campaign.

The single-instance / N-independent-process approach is GPU-bound (pitfall #18):
each Dolphin runs a batch-1 forward per frame, the GPU saturates at ~1-2 lanes,
and adding lanes just makes every match slower (measured: 10 lanes -> ~490s/match
vs ~60s single, ~same aggregate fps). This harness removes that wall the way the
RLVR rollout bench (tools/ffw_batch_mp.py) does: ONE central process owns every
model on the GPU and does batched forwards; N env processes each own a Dolphin
and are model-free.

The h2h twist over the self-play bench: envs play DIFFERENT character pairings,
so there is no single shared model to batch. Instead central loads ALL needed
checkpoints as {char: model} and, each cycle, groups the ready controller-slots
by character and runs one batched forward PER model over its slots. Two slots per
env (A-side, B-side). Batching efficiency comes from (a) the n_matches replicas of
a pairing and (b) a character-clustered job queue, so many concurrent slots share
a model. Correctness is identical to tools/play.py: same build_frame /
decode_and_press, same per-character normalization, same menu char-select gate
(both ports locked before autostart), and it runs on emulator_ss (faithful FFW,
see docs/research-notes-2026-07-22.md).

Wire protocol (per env):
  env -> central : ('START', env_id, a_char, b_char)       # (re)prefill both slots
                   ('F', env_id, {'a': npframe, 'b': npframe})
                   ('RESULT', env_id, a_char, b_char, aw, bw, dr)
                   ('DONE', env_id)
  central -> env : {'a': {head: np(1,1,C)}, 'b': {...}}     # both slots' logits
                   or 'STOP'

Usage:
  python3 tools/h2h_batch_mp.py --pairings sched.json --n-envs 12 --n-matches 3 \
      --out-dir eval_results/h2h/rerun_batched
"""
import argparse, json, time, os, queue, sys
import multiprocessing as mp
from multiprocessing.connection import wait
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `tools.*` imports resolve in spawned children
DOLPHIN = "emulator_ss/Binaries/dolphin-emu"
ISO = "melee.iso"
HEADS = ["main_xy", "shoulder_val", "c_dir_logits", "btn_logits"]
STAGES = ["FINAL_DESTINATION", "BATTLEFIELD", "DREAMLAND",
          "YOSHIS_STORY", "FOUNTAIN_OF_DREAMS", "POKEMON_STADIUM"]


def char_dir(c):
    return "hf_eval/fox-master" if c == "fox" else f"hf_eval/{c}"


def char_enum(c):
    return {"ice_climbers": "POPO", "puff": "JIGGLYPUFF"}.get(c, c.upper())


# ----------------------------- env process -----------------------------
def env_proc(env_id, conn, port, jobs_q, stop_ev, n_matches, seed, out_dir,
             replay_dir=None):
    import melee
    import torch
    from tools.inference_utils import (
        load_inference_context, build_frame, build_frame_p2, decode_and_press)

    # lazily-cached per-char inference context (norm/combos); envs are model-free
    _ctx_cache = {}
    def get_ctx(c):
        if c not in _ctx_cache:
            _ctx_cache[c] = load_inference_context(char_dir(c))
        return _ctx_cache[c]

    rep = None
    if replay_dir is not None:
        rep = str(Path(replay_dir) / str(env_id))
        os.makedirs(rep, exist_ok=True)
    con = melee.Console(
        path=DOLPHIN, is_dolphin=True, tmp_home_directory=True,
        copy_home_directory=False, blocking_input=True, online_delay=0,
        setup_gecko_codes=True, fullscreen=False, gfx_backend="Null",
        disable_audio=True, use_exi_inputs=True, enable_ffw=True,
        save_replays=(rep is not None), replay_dir=rep, slippi_port=port)
    cb = melee.Controller(console=con, port=1, type=melee.ControllerType.STANDARD)
    cc = melee.Controller(console=con, port=2, type=melee.ControllerType.STANDARD)
    con.run(iso_path=str(ROOT / ISO)); con.connect(); cb.connect(); cc.connect()
    STAGE_ENUMS = [melee.Stage[s] for s in STAGES]

    def send_recv(fa, fb):
        conn.send(('F', env_id, {'a': {k: v.numpy() for k, v in fa.items()},
                                 'b': {k: v.numpy() for k, v in fb.items()}}))
        msg = conn.recv()
        if msg == 'STOP':
            return None
        pa = {k: torch.from_numpy(a) for k, a in msg['a'].items()}
        pb = {k: torch.from_numpy(a) for k, a in msg['b'].items()}
        return pa, pb

    while not stop_ev.is_set():
        try:
            a_char, b_char = jobs_q.get_nowait()
        except queue.Empty:
            break
        A, B = melee.Character[char_enum(a_char)], melee.Character[char_enum(b_char)]
        ctx_a, ctx_b = get_ctx(a_char), get_ctx(b_char)
        m1, m2 = melee.MenuHelper(), melee.MenuHelper()
        aw = bw = dr = 0
        results = []
        a_on_p1 = True
        prev_a = prev_b = None
        last_p1 = last_p2 = 0
        in_game = False
        started = False
        cur_stage = STAGE_ENUMS[0]

        while not stop_ev.is_set() and len(results) < n_matches:
            gs = con.step()
            if gs is None:
                continue
            if gs.menu_state not in (melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH):
                if in_game:
                    # match ended: tally from last in-game stocks
                    a_stk = last_p1 if a_on_p1 else last_p2
                    b_stk = last_p2 if a_on_p1 else last_p1
                    if a_stk > 0 and b_stk == 0:
                        res = "a_wins"; aw += 1
                    elif b_stk > 0 and a_stk == 0:
                        res = "b_wins"; bw += 1
                    else:
                        res = "draw"; dr += 1
                    results.append({"result": res, "a_stocks": a_stk,
                                    "b_stocks": b_stk, "stage": cur_stage.name})
                    in_game = False
                    a_on_p1 = not a_on_p1           # alternate ports
                    prev_a = prev_b = None
                # port assignment: A on p1 iff a_on_p1
                cur_stage = STAGE_ENUMS[len(results) % len(STAGE_ENUMS)]
                p1 = (A, 0) if a_on_p1 else (B, 1)
                p2 = (B, 1) if a_on_p1 else (A, 0)
                # gate autostart until BOTH ports are locked on the intended
                # char with coin down (the play.py char-select fix, ported).
                css = gs.menu_state in (melee.Menu.CHARACTER_SELECT,
                                        melee.Menu.SLIPPI_ONLINE_CSS)
                def locked(pt, ch):
                    ps = gs.players.get(pt)
                    if ps is None:
                        return False
                    tgt = (melee.Character.ZELDA
                           if ch is melee.Character.SHEIK else ch)
                    return ps.character is tgt and ps.coin_down
                allow = (locked(1, p1[0]) and locked(2, p2[0])) if css else True
                m1.menu_helper_simple(gs, cb, p1[0], cur_stage, cpu_level=0,
                                      autostart=False, costume=p1[1])
                m2.menu_helper_simple(gs, cc, p2[0], cur_stage, cpu_level=0,
                                      autostart=allow, costume=p2[1])
                cb.flush(); cc.flush()
                started = False
                continue

            if not started:
                started = True
                in_game = True
                prev_a = prev_b = None
                conn.send(('START', env_id, a_char, b_char))
            if len(gs.players) < 2:
                continue
            ps1, ps2 = gs.players.get(1), gs.players.get(2)
            if ps1 is not None:
                last_p1 = int(ps1.stock)
            if ps2 is not None:
                last_p2 = int(ps2.stock)

            # A on p1 -> A uses p1 perspective (build_frame); else p2.
            a_build = build_frame if a_on_p1 else build_frame_p2
            b_build = build_frame_p2 if a_on_p1 else build_frame
            ctrl_a = cb if a_on_p1 else cc
            ctrl_b = cc if a_on_p1 else cb
            fa = a_build(gs, prev_a, ctx_a)
            fb = b_build(gs, prev_b, ctx_b)
            if fa is None or fb is None:
                continue
            got = send_recv(fa, fb)
            if got is None:
                break
            pa, pb = got
            # decode_and_press flushes each controller internally (exactly once
            # per frame, matching play.py). Do NOT add a second flush here: with
            # emulator_ss's per-pipe flush ledger a double-flush skews input
            # delivery by a frame for BOTH controllers -> systematic mistiming
            # (dead L-cancel, both bots mush). This was the batched-harness bug.
            prev_a, _, _ = decode_and_press(ctrl_a, pa, prev_a)
            prev_b, _, _ = decode_and_press(ctrl_b, pb, prev_b)

        # write result json (schema matches rank_h2h.py: a_wins for first name)
        rep = {"a_char": a_char, "b_char": b_char, "n_matches": len(results),
               "a_wins": aw, "b_wins": bw, "draws": dr,
               "a_win_rate": (aw / len(results)) if results else 0.0,
               "matches": results}
        outp = Path(out_dir) / f"{a_char}__vs__{b_char}.json"
        outp.write_text(json.dumps(rep, indent=1))
        conn.send(('RESULT', env_id, a_char, b_char, aw, bw, dr))

    conn.send(('DONE', env_id))
    try:
        con.stop()
    except Exception:
        pass


# ----------------------------- central --------------------------------
def run(pairings, n_envs, n_matches, seed, out_dir, replay_dir=None):
    import torch
    from tools.inference_utils import (
        load_mimic_model, load_inference_context, build_mock_frame)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # chars actually needed by the schedule
    chars = sorted({c for pr in pairings for c in pr})
    print(f"loading {len(chars)} models onto {device} ...", flush=True)
    models, ctxs, seq_len = {}, {}, None
    for c in chars:
        m, cfg = load_mimic_model(f"{char_dir(c)}/model.pt", device); m.eval()
        models[c] = m
        ctx = dict(load_inference_context(char_dir(c)))
        ctx["combo_map"] = {}; ctx["n_combos"] = cfg.n_controller_combos
        ctxs[c] = ctx
        seq_len = cfg.max_seq_len if seq_len is None else seq_len
        assert cfg.max_seq_len == seq_len, "mixed seq_len across checkpoints"
    print(f"models loaded, seq_len={seq_len}", flush=True)

    ctxmp = mp.get_context("spawn")
    stop_ev = ctxmp.Event()
    jobs_q = ctxmp.Queue()
    for pr in pairings:
        jobs_q.put(tuple(pr))

    conns, procs = [], []
    for i in range(n_envs):
        pc, cc = ctxmp.Pipe()
        p = ctxmp.Process(target=env_proc,
                          args=(i, cc, 51600 + i, jobs_q, stop_ev, n_matches,
                                seed, str(out_dir), replay_dir))
        p.start(); conns.append(pc); procs.append(p)
    conn_env = {pc: i for i, pc in enumerate(conns)}

    # 2 slots per env: 2*i = A-side, 2*i+1 = B-side. each slot has a char +
    # rolling (seq_len) window buffer. The buffers live ON THE GPU so central's
    # per-frame host->device cost is a single new frame per push (~1x features),
    # not the full 180-frame window x 24 slots every cycle. That transfer was
    # the round-trip bottleneck that pushed Dolphin past emulator_ss's 250 ms
    # combined-frame-wait and served stale pads (mistiming). The in-place shift
    # is now a GPU op; the batched-forward stack needs no .to(device).
    n_slots = 2 * n_envs
    slot_char = [None] * n_slots
    bufs = [None] * n_slots

    def reset_slot(s, char):
        slot_char[s] = char
        mock = build_mock_frame(ctxs[char])
        bufs[s] = {k: v.expand(seq_len, *v.shape[1:]).to(device).clone()
                   for k, v in mock.items()}

    def push(s, frame):
        b = bufs[s]
        for k, v in frame.items():
            b[k][:-1] = b[k][1:].clone()
            b[k][-1] = v[0]  # v already on device (moved in the F handler)

    done = set()
    t0 = time.time()
    n_pairs = len(pairings)
    completed = 0
    frames_done = 0
    while len(done) < n_envs:
        ready = wait(conns, timeout=1.0)
        active = []  # (env_id, side) needing a forward this cycle
        for c in ready:
            try:
                msg = c.recv()
            except EOFError:
                done.add(conn_env[c]); continue
            tag = msg[0]
            if tag == 'F':
                _, env_id, payload = msg
                for side, sl in (('a', 2 * env_id), ('b', 2 * env_id + 1)):
                    fr = {k: torch.from_numpy(a).to(device, non_blocking=True)
                          for k, a in payload[side].items()}
                    push(sl, fr)
                active.append(env_id)
                frames_done += 1
            elif tag == 'START':
                _, env_id, a_char, b_char = msg
                reset_slot(2 * env_id, a_char)
                reset_slot(2 * env_id + 1, b_char)
            elif tag == 'RESULT':
                _, env_id, a, b, aw, bw, dr = msg
                completed += 1
                print(f"[{completed}/{n_pairs}] {a} vs {b}: {aw}-{bw} "
                      f"(dr {dr}) | {frames_done/(time.time()-t0):.0f} fps agg",
                      flush=True)
            elif tag == 'DONE':
                done.add(msg[1])
        if not active:
            continue
        # group ready slots by char, one batched forward per model
        by_char = {}
        for env_id in active:
            for sl in (2 * env_id, 2 * env_id + 1):
                by_char.setdefault(slot_char[sl], []).append(sl)
        slot_out = {}
        for char, slots in by_char.items():
            keys = bufs[slots[0]].keys()
            mega = {k: torch.stack([bufs[s][k] for s in slots], dim=0)
                    for k in keys}  # buffers already on device
            with torch.no_grad():
                out = models[char](mega)
            for j, s in enumerate(slots):
                slot_out[s] = {k: out[k][j:j + 1, -1:, :].contiguous().cpu().numpy()
                               for k in HEADS}
        for env_id in active:
            conns[env_id].send({'a': slot_out[2 * env_id],
                                'b': slot_out[2 * env_id + 1]})

    dt = time.time() - t0
    print(f"\nDONE {completed}/{n_pairs} pairings in {dt/60:.1f} min | "
          f"{frames_done} frames | {frames_done/dt:.0f} fps agg "
          f"({frames_done/dt/60:.1f}x realtime)", flush=True)
    for p in procs:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairings", required=True, help="JSON list of [a,b] pairs")
    ap.add_argument("--n-envs", type=int, default=12)
    ap.add_argument("--n-matches", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="eval_results/h2h/rerun_batched")
    ap.add_argument("--replay-dir", default=None,
                    help="If set, envs save .slp here for canary validation.")
    args = ap.parse_args()
    pairings = json.load(open(args.pairings))
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    replay_dir = None
    if args.replay_dir:
        replay_dir = str(ROOT / args.replay_dir)
        os.makedirs(replay_dir, exist_ok=True)
    # cluster the queue by character so concurrent slots share models (better
    # batching): order pairings by first char, then second.
    pairings.sort(key=lambda p: (p[0], p[1]))
    print(f"{len(pairings)} pairings, {args.n_envs} envs, n_matches={args.n_matches}",
          flush=True)
    run(pairings, args.n_envs, args.n_matches, args.seed, out_dir, replay_dir)


if __name__ == "__main__":
    main()
