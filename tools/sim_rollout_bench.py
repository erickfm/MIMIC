#!/usr/bin/env python3
"""Rollout throughput of the native-sim RL loop: VecSim with one policy on both sides (as sim_rl's
learner vs frozen copy), optional VR modules, a fixed number of steps after warm-up. Prints env
frames per second. Profile it with `py-spy record --subprocesses -o out.svg -- python3 tools/sim_rollout_bench.py ...`.

  python3 tools/sim_rollout_bench.py --policy-dir checkpoints/smallfox/base --envs 64 --workers 16 \
      --vr stock_delta,damage_delta --steps 2000
"""
import argparse, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy-dir", required=True)
    ap.add_argument("--templates", default="replays_doc_rt")
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--vr", default="", help="comma-separated VR ids computed in the workers")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=300)
    a = ap.parse_args()
    import tempfile
    from rlvr.sim.vec_env import VecSim
    from sim_h2h import make_ditto
    tdir = Path(tempfile.mkdtemp(prefix="sim_bench_"))
    templates = []
    for src in sorted(Path(a.templates).glob("*.slp")):
        make_ditto(src, "FOX", tdir / src.name)
        templates.append(tdir / src.name)
    pol = dict(dir=a.policy_dir, amp=True, compile=True)
    v = VecSim(templates, {"A": pol, "B": dict(pol)}, a.envs, a.workers, seed=0,
               assign=lambda e, side, ch: ("A", "B")[(e + side) % 2],
               vr_ids=[x for x in a.vr.split(",") if x])
    for _ in range(a.warmup):
        v.step()
    f0, t0 = v.frames, time.time()
    for _ in range(a.steps):
        v.step()
    dt = time.time() - t0
    print(f"envs {a.envs} workers {a.workers} vr [{a.vr}]: {(v.frames - f0) / dt:.0f} env-frames/s "
          f"({dt / a.steps * 1e3:.1f} ms/step)")
    v.close()


if __name__ == "__main__":
    main()
