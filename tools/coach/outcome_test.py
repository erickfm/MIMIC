#!/usr/bin/env python3
"""At a human's decision points, do the model's choices lead to better outcomes than the human's?

The sim replays a real game (current Slippi, exact) to a decision point D: the first frame of a
can-act window that the human ended by committing to a move (tools/coach/decision_points.py).
From the same state, arms differ only in who plays the human's side for the decision stretch
(D to the human's commit + --tail frames, at most --max-stretch): `you` (the recorded inputs)
and each --arm NAME=POLICY_DIR[@CTX] (a policy, optionally with only the last CTX frames of
context). After the stretch the --cont policy (default: the char's BC policy) plays the human's
side in every arm, and the opponent's policy plays the opponent from D in every arm. Outcome over --horizon frames from D:
damage dealt and taken (sums of percent increases, so deaths don't reset it) and stocks.

  python3 tools/coach/outcome_test.py --replays ~/Slippi --code WAVE --char FALCO --n 160 \
      --samples 4 --out data/coach_rank/outcome_falco.json --arm model=hf_eval/falco --arm short=hf_eval/falco@30
"""
import argparse, json, os, random, struct, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools")); sys.path.insert(0, str(ROOT / "tools" / "coach"))
from copy_test import POLICY, candidates, prev_from_controller   # noqa: E402



def pick_point(slp, port, lo, hi, max_delay, rng):
    """A random can-act window the human ended by committing within max_delay frames."""
    import decision_points as D
    g, frames, st, f1, f4, hl, age, keep = D.load(slp, port)
    ok, _ = D.can_act(st, f1, f4, hl, age)
    ws = [w for w in D.windows(frames, st, ok)
          if w["ended_by"] == "player" and w["commit"] is not None and w["length"] <= max_delay
          and lo <= w["frame"] <= hi]
    return rng.choice(ws) if ws else None


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

    slp, my_port, D, stretch, arm, arm_spec, cont, seed, horizon = job
    torch.manual_seed(seed)
    dev = "cuda"
    pre, post = recorded(slp)
    rec_seed = {struct.unpack(">i", p[:4])[0]: struct.unpack(">I", p[4:8])[0]
                for c, p in slp_events(slp) if c == 0x3A}
    def load(d):
        model, cfg = load_mimic_model(str(d / "model.pt"), dev)
        set_cond_temperature(model, 1.0)
        return model, cfg, _make_ctx(load_inference_context(str(d)), cfg)

    sides = []
    for (port, char), build in zip(template_players(slp), (build_frame, build_frame_p2)):
        me = port == my_port
        model, cfg, ctx = load(ROOT / (cont if me else POLICY[char]))
        s = dict(port=port, build=build, ctrl=SimController(), ctx=ctx, me=me,
                 state=PlayerState(model, cfg.max_seq_len, dev, ctx=ctx), prev=None)
        if me and arm_spec:
            d, _, c = arm_spec.partition("@")
            am, acfg, actx = load(ROOT / d)
            s["arm"] = PlayerState(am, int(c) if c else acfg.max_seq_len, dev, ctx=actx)
            s["arm_ctx"] = actx
        sides.append(s)
    seq = sides[0]["state"].seq_len
    rec_pads = lambda f: [pad_from_pre(pre[(f, q, 0)]) if (f, q, 0) in pre else Pad() for q in range(4)]
    sim = MeleeSim(slp, seed=rec_seed[min(rec_seed)], rng="online")
    parser = SlippiParser(slp)
    frame = sim.reset(initial_pads=rec_pads(-123))
    exact, last, out = None, None, {"dealt": 0.0, "taken": 0.0, "stocks_lost": 0, "stocks_taken": 0}
    while not sim.done and frame is not None and frame.index < D + horizon:
        f = frame.index
        pads = rec_pads(f + 1)
        if f > D - seq:
            gs = parser.parse(frame)
            for s in sides:
                # The input that landed on this frame: recorded before D, and for the human's side
                # in arm A through the stretch; otherwise what the policy sent (already in prev).
                recorded_here = f <= D or (s["me"] and arm == "you" and f <= D + stretch)
                if recorded_here:
                    s["prev"] = prev_from_controller(gs.players[s["port"] + 1].controller_state)
                fr = s["build"](gs, s["prev"], s["ctx"])
                s["state"].push_frame(fr)
                if "arm" in s:
                    s["arm"].push_frame(s["build"](gs, s["prev"], s["arm_ctx"]))
            if f == D:
                exact = all(not physics_diff(post[(D, s["port"], 0)], frame.post[(s["port"], 0)]) for s in sides)
            if f >= D:
                for s in sides:
                    if s["me"] and arm == "you" and f < D + stretch:
                        continue   # the human's next input comes from the recording
                    use = s["arm"] if ("arm" in s and f < D + stretch) else s["state"]
                    s["prev"], _, _ = decode_and_press(s["ctrl"], use.predict(), s["prev"])
                    pads[s["port"]] = s["ctrl"].pad()
                cur = {s["port"]: (float(gs.players[s["port"] + 1].percent), int(gs.players[s["port"] + 1].stock))
                       for s in sides}
                if last is not None:
                    for s in sides:
                        dp = max(0.0, cur[s["port"]][0] - last[s["port"]][0])
                        ds = max(0, last[s["port"]][1] - cur[s["port"]][1])
                        if s["me"]:
                            out["taken"] += dp; out["stocks_lost"] += ds
                        else:
                            out["dealt"] += dp; out["stocks_taken"] += ds
                last = cur
        frame = sim.step(pads)
    sim.close()
    return dict(slp=slp, D=D, arm=arm, seed=seed, exact=exact, **out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replays", required=True)
    ap.add_argument("--code", required=True)
    ap.add_argument("--char", required=True)
    ap.add_argument("--n", type=int, default=160, help="decision points (one per game)")
    ap.add_argument("--samples", type=int, default=4, help="runs per arm per point")
    ap.add_argument("--horizon", type=int, default=300)
    ap.add_argument("--tail", type=int, default=20, help="frames after the human's commit in the stretch")
    ap.add_argument("--max-stretch", type=int, default=60)
    ap.add_argument("--max-delay", type=int, default=40, help="max frames from window start to commit")
    ap.add_argument("--arm", action="append", default=[], help="NAME=POLICY_DIR[@CTX], repeatable")
    ap.add_argument("--cont", default=None, help="policy for the human's side after the stretch")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    arms = [("you", None)] + [tuple(x.split("=", 1)) for x in a.arm]
    cont = a.cont or POLICY[a.char]
    rng = random.Random(a.seed)
    cands = candidates(os.path.expanduser(a.replays), a.code, a.char, 600, a.horizon)
    rng.shuffle(cands)
    jobs, points = [], 0
    for slp, port, n in cands:
        if points >= a.n:
            break
        w = pick_point(slp, port, 600, n - a.horizon - 60, a.max_delay, rng)
        if w is None:
            continue
        stretch = min(a.max_stretch, w["length"] + a.tail)
        for arm, spec in arms:
            for k in range(a.samples):
                jobs.append((slp, port, w["frame"], stretch, arm, spec, cont,
                             a.seed * 1000003 + points * 100 + k, a.horizon))
        points += 1
    print(f"{points} decision points x {len(arms)} arms x {a.samples} samples = {len(jobs)} runs", flush=True)
    import multiprocessing as mp
    res = []
    with ProcessPoolExecutor(a.jobs, mp_context=mp.get_context("spawn")) as ex:
        for i, r in enumerate(ex.map(run, jobs, chunksize=1)):
            res.append(r)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(jobs)} runs", flush=True)
                json.dump(res, open(a.out, "w"))
    json.dump(res, open(a.out, "w"))
    summarize(res, [n for n, _ in arms])


def summarize(res, ARMS):
    by = {}
    bad = {(r["slp"], r["D"]) for r in res if not r["exact"]}
    for r in res:
        key = (r["slp"], r["D"])
        if key in bad:
            continue
        net = r["dealt"] - r["taken"]
        by.setdefault(key, {}).setdefault(r["arm"], []).append(
            (net, r["dealt"], r["taken"], r["stocks_taken"] - r["stocks_lost"]))
    pts = [v for v in by.values() if all(a in v for a in ARMS)]
    print(f"\n{len(pts)} decision points ({len(bad)} dropped: sim not exact at the point)")
    m = {a: np.array([np.mean(p[a], 0) for p in pts]) for a in ARMS}   # (points, 4)
    print(f"{'arm':6s} {'net dmg':>9s} {'dealt':>8s} {'taken':>8s} {'net stocks':>11s}")
    for a in ARMS:
        x = m[a].mean(0)
        print(f"{a:6s} {x[0]:9.2f} {x[1]:8.2f} {x[2]:8.2f} {x[3]:11.3f}")
    print("paired over decision points (mean ± se):")
    import itertools
    for b, a in itertools.combinations(ARMS, 2):
        d = m[a] - m[b]
        se = d.std(0, ddof=1) / np.sqrt(len(d))
        print(f"  {a} - {b}:  net dmg {d[:, 0].mean():+.2f}±{se[0]:.2f}   net stocks {d[:, 3].mean():+.3f}±{se[3]:.3f}"
              f"   {a} better at {(d[:, 0] > 0).mean():.0%} of points")


if __name__ == "__main__":
    main()
