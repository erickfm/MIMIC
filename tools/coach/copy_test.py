#!/usr/bin/env python3
"""Does a model branched from a human's game inherit the human's play (in-context copying)?

Two ways to reach the same branch frame B of a real game (a template .slp the sim replays
exactly: current Slippi, online RNG):
  you: the sim replays the recording's inputs up to B, so the models' 180-frame history holds
       the humans' states and inputs;
  bot: the models play both sides from the start, so at B the history is their own.
From B both sides are played by their models (T=1) for --horizon frames. Per frame after B it
records the policy's entropy (sum over its heads; for cond-heads models each head's entropy is
conditioned on the earlier heads' samples) and the damage each side took.

If the model copies the human, `you` starts above `bot` and falls to it as its own inputs fill
the context (by ~180 frames). A gap that persists is the situation, not the history.

  python3 tools/coach/copy_test.py --replays ~/Slippi --code WAVE --char FALCO --n 64 \
      --out data/coach_rank/copy_test_falco.json
"""
import argparse, glob, json, os, random, struct, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools")); sys.path.insert(0, str(ROOT / "tools" / "coach"))

POLICY = {"FALCO": "hf_eval/falco", "FOX": "runs/fox-M-bc2-mlr01-s1/policy", "LUIGI": "hf_eval/luigi",
          "CPTFALCON": "hf_eval/cptfalcon", "MARTH": "hf_eval/marth", "SHEIK": "hf_eval/sheik",
          "JIGGLYPUFF": "hf_eval/puff", "PEACH": "hf_eval/peach", "SAMUS": "hf_eval/samus",
          "PIKACHU": "hf_eval/pikachu", "YOSHI": "hf_eval/yoshi", "GANONDORF": "hf_eval/ganondorf",
          "DOC": "hf_eval/doc", "MARIO": "hf_eval/mario", "LINK": "hf_eval/link", "YLINK": "hf_eval/ylink",
          "ROY": "hf_eval/roy", "DK": "hf_eval/dk", "BOWSER": "hf_eval/bowser", "NESS": "hf_eval/ness",
          "MEWTWO": "hf_eval/mewtwo", "GAMEANDWATCH": "hf_eval/gameandwatch", "ZELDA": None,
          "KIRBY": None, "PICHU": None, "POPO": None}


def candidates(replays, code, char, lo, horizon):
    """(slp, my port, last usable frame) of 1v1 current-Slippi online games where `code` plays `char`
    and the opponent has a policy."""
    from peppi_py import read_slippi
    from sim_play import EXT_TO_MELEE
    out = []
    for p in sorted(glob.glob(os.path.join(replays, "**", "*.slp"), recursive=True)):
        try:
            g = read_slippi(p, skip_frames=True)
        except Exception:
            continue
        pls = [(i, pl) for i, pl in enumerate(g.start.players) if pl]
        if len(pls) != 2 or g.start.slippi.version[:2] < (3, 18):
            continue
        mine = [i for i, pl in pls if pl.netplay and pl.netplay.code.startswith(code)]
        if len(mine) != 1:
            continue
        me = mine[0]
        chars = {i: EXT_TO_MELEE.get(pl.character) for i, pl in pls}
        opp = next(i for i, _ in pls if i != me)
        if chars[me] != char or not POLICY.get(chars[opp]):
            continue
        n = (g.metadata or {}).get("lastFrame")
        if n and n > lo + horizon + 60:
            out.append((p, me, int(n)))
    return out


def prev_from_controller(cs):
    import melee
    B = melee.enums.Button
    d = {"main_x": float(cs.main_stick[0]), "main_y": float(cs.main_stick[1]),
         "c_x": float(cs.c_stick[0]), "c_y": float(cs.c_stick[1]),
         "l_shldr": float(cs.l_shoulder), "r_shldr": float(cs.r_shoulder)}
    for b in ["BUTTON_A", "BUTTON_B", "BUTTON_X", "BUTTON_Y", "BUTTON_Z", "BUTTON_L", "BUTTON_R"]:
        d[f"btn_{b}"] = int(bool(cs.button.get(getattr(B, b), False)))
    return d


def entropy(preds):
    import torch
    import torch.nn.functional as F
    hs = []
    for k in ("shoulder_val", "c_dir_logits", "main_xy", "btn_logits"):
        lp = F.log_softmax(preds[k][0, -1].float(), -1)
        hs.append(float(-(lp.exp() * lp).sum()))
    return hs


def run(job):
    import torch
    from inference_utils import (PlayerState, build_frame, build_frame_p2, decode_and_press,
                                 load_inference_context, load_mimic_model, set_cond_temperature)
    from play import _make_ctx
    from rlvr.sim.controller import SimController
    from rlvr.sim.gamestate import SlippiParser
    from rlvr.sim.melee_sim import MeleeSim, Pad
    from sim_play import template_players
    from tools.sim_env_check import recorded
    from tools.sim_replay_check import slp_events, physics_diff
    from pads import pad_from_pre

    slp, my_port, B, mode, seed, horizon = job
    torch.manual_seed(seed)
    dev = "cuda"
    players = template_players(slp)
    pre, post = recorded(slp)
    rec_seed = {struct.unpack(">i", p[:4])[0]: struct.unpack(">I", p[4:8])[0]
                for c, p in slp_events(slp) if c == 0x3A}
    s0 = rec_seed[min(rec_seed)]
    sides = []
    for (port, char), build in zip(players, (build_frame, build_frame_p2)):
        d = ROOT / POLICY[char]
        model, cfg = load_mimic_model(str(d / "model.pt"), dev)
        set_cond_temperature(model, 1.0)
        ctx = _make_ctx(load_inference_context(str(d)), cfg)
        sides.append(dict(port=port, char=char, build=build, ctrl=SimController(), ctx=ctx,
                          state=PlayerState(model, cfg.max_seq_len, dev, ctx=ctx), me=port == my_port))
    seq = sides[0]["state"].seq_len
    rec_pads = lambda f: [pad_from_pre(pre[(f, q, 0)]) if (f, q, 0) in pre else Pad() for q in range(4)]
    sim = MeleeSim(slp, seed=s0, rng="online")
    parser = SlippiParser(slp)
    frame = sim.reset(initial_pads=rec_pads(-123))
    ent, dmg, exact = [], [], None
    start_pct = None
    while not sim.done and frame is not None and frame.index < B + horizon:
        f = frame.index
        live = mode == "bot" or f >= B           # models choose this frame's next input
        need_state = live or f > B - seq
        pads = rec_pads(f + 1) if not live else [Pad()] * 4
        if need_state:
            gs = parser.parse(frame)
            for s in sides:
                if mode == "you" and f <= B:   # history from the recording: the human's input on this frame
                    s["state"].prev_sent = prev_from_controller(gs.players[s["port"] + 1].controller_state)
                s["state"].push_frame(s["build"](gs, s["state"].prev_sent, s["ctx"]))
            if f == B:
                exact = all(not physics_diff(post[(B, s["port"], 0)], frame.post[(s["port"], 0)])
                            for s in sides) if mode == "you" else None
                start_pct = {s["port"]: float(gs.players[s["port"] + 1].percent) for s in sides}
            if live:
                for s in sides:
                    preds = s["state"].predict()
                    if f >= B and s["me"]:
                        ent.append(entropy(preds))
                    s["state"].prev_sent, _, _ = decode_and_press(s["ctrl"], preds, s["state"].prev_sent)
                    pads[s["port"]] = s["ctrl"].pad()
                if f >= B:
                    dmg.append({str(s["port"]): float(gs.players[s["port"] + 1].percent) for s in sides})
        frame = sim.step(pads)
    sim.close()
    return dict(slp=slp, my_port=my_port, B=B, mode=mode, exact_at_branch=exact, ent=ent, pct=dmg)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replays", required=True)
    ap.add_argument("--code", required=True, help="connect-code prefix of the human, e.g. WAVE")
    ap.add_argument("--char", required=True, help="libmelee character name the human plays")
    ap.add_argument("--n", type=int, default=64, help="branch points (each run in both modes)")
    ap.add_argument("--horizon", type=int, default=300)
    ap.add_argument("--min-frame", type=int, default=600)
    ap.add_argument("--max-frame", type=int, default=3600)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    cands = candidates(os.path.expanduser(a.replays), a.code, a.char, a.min_frame, a.horizon)
    rng.shuffle(cands)
    print(f"{len(cands)} usable games; branching {min(a.n, len(cands))}", flush=True)
    jobs = []
    for i, (slp, port, n) in enumerate(cands[:a.n]):
        B = rng.randint(a.min_frame, min(a.max_frame, n - a.horizon - 60))
        for mode in ("you", "bot"):
            jobs.append((slp, port, B, mode, a.seed * 100003 + i, a.horizon))
    import multiprocessing as mp
    res = []
    with ProcessPoolExecutor(a.jobs, mp_context=mp.get_context("spawn")) as ex:
        for r in ex.map(run, jobs):
            res.append(r)
            print(f"  {r['mode']:3s} B={r['B']:5d} exact={r['exact_at_branch']} frames={len(r['ent'])}", flush=True)
    json.dump(res, open(a.out, "w"))
    summarize(res, a.horizon)


def summarize(res, horizon):
    bad = {(r["slp"], r["B"]) for r in res if r["mode"] == "you" and not r["exact_at_branch"]}
    pairs = {}
    for r in res:
        if (r["slp"], r["B"]) in bad or len(r["ent"]) < horizon:
            continue
        pairs.setdefault((r["slp"], r["B"]), {})[r["mode"]] = np.array(r["ent"]).sum(1)
    pairs = [p for p in pairs.values() if len(p) == 2]
    print(f"\n{len(pairs)} branch points with both modes ({len(bad)} dropped: sim not exact at branch)")
    if not pairs:
        return
    you = np.stack([p["you"] for p in pairs]); bot = np.stack([p["bot"] for p in pairs])
    d = you - bot
    print(f"{'frames after branch':>20s} {'you':>8s} {'bot':>8s} {'you-bot':>16s}")
    for lo, hi in [(0, 10), (10, 30), (30, 60), (60, 120), (120, 180), (180, 240), (240, 300)]:
        if hi > horizon:
            break
        x = d[:, lo:hi].mean(1)
        print(f"{f'{lo}-{hi}':>20s} {you[:, lo:hi].mean():8.4f} {bot[:, lo:hi].mean():8.4f} "
              f"{x.mean():+8.4f}±{x.std(ddof=1) / np.sqrt(len(x)):.4f}")


if __name__ == "__main__":
    main()
