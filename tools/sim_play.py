#!/usr/bin/env python3
"""MIMIC policy vs MIMIC policy in the native sim (rlvr/sim): the same inference code as
tools/play.py (PlayerState, build_frame / build_frame_p2, decode_and_press), with the sim's
libmelee-parsed GameStates in place of Dolphin's and SimController in place of the pipe.

Each match is entered from a template .slp (characters, costumes, stage, rules); the two
policies are assigned to the template's ports by character. Every match is written as a .slp,
so the usual replay tools (tools/validate_run.py, canaries) apply.

  python3 tools/sim_play.py --templates replays_doc_rt --policy DOC=hf_eval/doc \
      --policy FOX=hf_eval/fox-master --n-matches 12 --jobs 6 --out-dir sim_replays/doc_fox
"""
import argparse, json, os, random, struct, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))

# Libmelee Character value for each Slippi external character id in a Game Start event.
EXT_TO_MELEE = {0: "CPTFALCON", 1: "DK", 2: "FOX", 3: "GAMEANDWATCH", 4: "KIRBY", 5: "BOWSER",
                6: "LINK", 7: "LUIGI", 8: "MARIO", 9: "MARTH", 10: "MEWTWO", 11: "NESS",
                12: "PEACH", 13: "PIKACHU", 14: "POPO", 15: "JIGGLYPUFF", 16: "SAMUS",
                17: "YOSHI", 18: "ZELDA", 19: "SHEIK", 20: "FALCO", 21: "YLINK", 22: "DOC",
                23: "ROY", 24: "PICHU", 25: "GANONDORF"}


def template_players(slp):
    """[(port 0-3, CHARACTER name)] of a template's Game Start."""
    from sim_replay_check import slp_events
    for c, p in slp_events(slp):
        if c == 0x36:
            return [(i, EXT_TO_MELEE[p[0x64 + 0x24 * i]]) for i in range(4) if p[0x65 + 0x24 * i] != 3]
    raise ValueError(f"{slp}: no Game Start")


def play_match(job):
    import torch
    from inference_utils import (PlayerState, build_frame, build_frame_p2, decode_and_press,
                                 load_inference_context, load_mimic_model, set_cond_temperature)
    from play import _make_ctx
    from rlvr.sim.controller import SimController
    from rlvr.sim.gamestate import SlippiParser, SlpWriter
    from rlvr.sim.melee_sim import MeleeSim

    template, policies, seed, out_path, temperature = job
    device = "cuda" if torch.cuda.is_available() else "cpu"
    players = template_players(template)
    if len(players) != 2:
        raise ValueError(f"{template}: sim_play needs a 1v1 template")
    sides = []
    for (port, char), build in zip(players, (build_frame, build_frame_p2)):
        ckpt_dir = policies[char]
        model, cfg = load_mimic_model(str(Path(ckpt_dir) / "model.pt"), device)
        set_cond_temperature(model, temperature)
        ctx = _make_ctx(load_inference_context(str(ckpt_dir)), cfg)
        sides.append(dict(port=port, char=char, build=build, ctrl=SimController(),
                          state=PlayerState(model, cfg.max_seq_len, device, ctx=ctx), ctx=ctx))
    sim = MeleeSim(template, seed=seed)
    parser, writer = SlippiParser(template), SlpWriter(template)
    t0 = time.time()
    frame = sim.reset()
    n = 0
    while not sim.done:
        writer.add(frame)
        gs = parser.parse(frame)
        pads = [None] * 4
        for s in sides:
            fr = s["build"](gs, s["state"].prev_sent, s["ctx"])
            s["state"].push_frame(fr)
            s["state"].prev_sent, _, _ = decode_and_press(s["ctrl"], s["state"].predict(),
                                                          s["state"].prev_sent, temperature=temperature)
            pads[s["port"]] = s["ctrl"].pad()
        from rlvr.sim.melee_sim import Pad
        frame = sim.step([p or Pad() for p in pads])
        n += 1
        if frame is None:
            break
    sim.close()
    writer.write(out_path)
    last = {p: (int(gs.players[p + 1].stock), round(float(gs.players[p + 1].percent), 1)) for p, _ in players}
    return dict(template=str(template), out=str(out_path), seed=seed, frames=n,
                fps=round(n / (time.time() - t0)), final={players[i][1]: last[players[i][0]] for i in range(2)})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--templates", required=True, help="dir of 1v1 template .slp files (cycled)")
    ap.add_argument("--policy", action="append", required=True, metavar="CHAR=CKPT_DIR",
                    help="checkpoint dir per character, e.g. FOX=hf_eval/fox-master")
    ap.add_argument("--n-matches", type=int, default=6)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    policies = dict(p.split("=", 1) for p in a.policy)
    templates = sorted(Path(a.templates).glob("*.slp"))
    os.makedirs(a.out_dir, exist_ok=True)
    rng = random.Random(a.seed)
    jobs = [(str(templates[i % len(templates)]), policies, rng.getrandbits(32),
             os.path.join(a.out_dir, f"sim_{i:04d}.slp"), a.temperature) for i in range(a.n_matches)]
    ctx = __import__("multiprocessing").get_context("spawn")   # CUDA in workers
    with ProcessPoolExecutor(a.jobs, mp_context=ctx) as ex:
        results = []
        for r in ex.map(play_match, jobs):
            results.append(r)
            print(json.dumps(r), flush=True)
    Path(a.out_dir, "results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
