#!/usr/bin/env python3
"""Throughput of the batched native-sim runner (rlvr/sim/vec_env.py): env-frames per second.

  python3 tools/sim_bench.py --templates replays_doc_rt --policy DOC=hf_eval/doc \
      --policy FOX=hf_eval/fox-master --envs 64 --workers 16 --steps 600
"""
import argparse, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--templates", required=True)
    ap.add_argument("--policy", action="append", required=True, metavar="CHAR=CKPT_DIR")
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--rng", default="online", choices=["online", "offline"])
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--record-dir")
    ap.add_argument("--episodes", type=int, default=0, help="run until this many episodes finish "
                    "(instead of --steps); with --record-dir every episode is written as a .slp")
    a = ap.parse_args()
    from rlvr.sim.vec_env import VecSim
    templates = sorted(Path(a.templates).glob("*.slp"))
    policies = dict(p.split("=", 1) for p in a.policy)
    t0 = time.time()
    v = VecSim(templates, policies, a.envs, a.workers, rng=a.rng, compile=not a.no_compile,
               amp=not a.fp32, record_dir=a.record_dir)
    for _ in range(5):   # compile / warm-up
        v.step()
    print(f"startup + warm-up {time.time() - t0:.0f}s", flush=True)
    t1, f1 = time.time(), v.frames
    steps = 0
    while (len(v.results) < a.episodes) if a.episodes else (steps < a.steps):
        v.step()
        steps += 1
    a.steps = steps
    dt = time.time() - t1
    print(f"envs {a.envs} workers {a.workers} compile={not a.no_compile} bf16={not a.fp32} rng={a.rng}: "
          f"{(v.frames - f1) / dt:.0f} env-frames/s ({a.steps / dt:.0f} steps/s, {(v.frames - f1) / dt / 60:.0f}x realtime), "
          f"episodes finished {len(v.results)}")
    if a.episodes:
        import json
        wins = {}
        for r in v.results[:a.episodes]:
            (c1, (s1, _)), (c2, (s2, _)) = r["final"].items()
            w = c1 if s1 > s2 else c2 if s2 > s1 else "draw"
            wins[w] = wins.get(w, 0) + 1
        print("results:", json.dumps(wins))
        if a.record_dir:
            Path(a.record_dir, "results.json").write_text(json.dumps(v.results, indent=1))
    v.close()


if __name__ == "__main__":
    main()
