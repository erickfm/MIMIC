#!/usr/bin/env python3
"""A model of one specific player: fine-tune a character's BC model on that player's own games, then
measure whether it predicts that player's real inputs on held-out games better than the generic
master model does. The coach offers it only if it's clearly better.

Data: shards that keep only that player's perspective (tools/slp_to_shards.py --keep-code), built
twice from a game-level split of their replays:
  - a training set whose own val shards pick the checkpoint (train.py's bestloss), and
  - a test set (--val-frac 1.0) that no choice is made on: the reported numbers.

  python3 tools/coach/player_model.py finetune --parent falco-S-bc2-hf28k-s1 \\
      --data data/stuckey/falco_train --name stuckey-falco-S-ft1 --steps 2000
  python3 tools/coach/player_model.py eval --data data/stuckey/falco_test \\
      --policies runs/falco-S-bc2-hf28k-s1/policy runs/falco-S-bc2-hf28k-s1.rl2-s1/policy \\
                 runs/stuckey-falco-S-ft1/policy
Both use the GPU: run them only when nothing else does.
"""
import argparse, json, re, shutil, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from run import NORM_JSONS, export_policy, git_commit  # noqa: E402

# train.py flags that shape the model or the data, from the parent's command (the rest are set here)
DROP = {"--data-dir", "--run-name", "--ckpt-dir", "--seed", "--lr", "--muon-lr", "--max-samples", "--max-steps",
        "--resume", "--val-frac", "--batch-size", "--grad-accum-steps"}


def parent_args(parent):
    cmd = json.loads((ROOT / "runs" / parent / "run.json").read_text())["recipe_resolved"]
    out = []
    for k, v in cmd["args"].items():
        f = f"--{k}"
        if f in DROP:
            continue
        out += [f] if v is True else [f, str(v)]
    return out, cmd


def ensure_meta(data, src):
    """train.py reads the normalization files from the data dir; the shard builder copies only some."""
    for f in NORM_JSONS:
        if not (Path(data) / f"{f}.json").exists():
            shutil.copy(Path(src) / f"{f}.json", Path(data) / f"{f}.json")


def finetune(a):
    args, rec = parent_args(a.parent)
    prd = ROOT / "runs" / a.parent
    ckpt = (prd / "policy" / "model.pt").resolve()
    ensure_meta(a.data, prd / "policy")
    rd = ROOT / "runs" / a.name
    if rd.exists():
        sys.exit(f"{rd} exists")
    (rd / "logs").mkdir(parents=True)
    cmd = ["python3", "train.py", *args, "--data-dir", str(Path(a.data).resolve()), "--run-name", a.name,
           "--ckpt-dir", str(rd / "ckpt"), "--seed", str(a.seed), "--resume", str(ckpt), "--no-load-optim",
           "--warm-restart", "--lr", str(a.lr), "--muon-lr", str(a.muon_lr), "--max-steps", str(a.steps),
           "--val-frac", "0.025", "--batch-size", str(a.local_batch or rec["max_local_batch"]),
           "--grad-accum-steps", str(1 if a.local_batch else rec["eff_batch"] // rec["max_local_batch"]), *a.extra.split()]
    info = dict(name=a.name, stage="player-ft", parent=a.parent, data=a.data, seed=a.seed, commit=git_commit(),
                started=datetime.now(timezone.utc).isoformat(timespec="seconds"), status="running", cmd=cmd,
                steps=a.steps, lr=a.lr, muon_lr=a.muon_lr)
    (rd / "run.json").write_text(json.dumps(info, indent=1))
    t0 = time.time()
    with open(rd / "logs" / "train.log", "w") as f:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
    best = next((rd / "ckpt").glob("*_bestloss.pt"), None)
    info.update(status="done" if rc == 0 and best else "failed", wall_min=round((time.time() - t0) / 60, 1))
    if best:
        export_policy(rd, best, prd / "policy")
    (rd / "run.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


def evaluate(a):
    rows = []
    for pol in a.policies:
        pol = Path(pol)
        run = pol.parent.name
        rj = json.loads((pol.parent / "run.json").read_text())
        base = rj["parent"] if rj.get("stage") in ("rl", "player-ft") else run
        while json.loads((ROOT / "runs" / base / "run.json").read_text()).get("stage") != "bc":
            base = json.loads((ROOT / "runs" / base / "run.json").read_text())["parent"]
        args, _ = parent_args(base)
        ensure_meta(a.data, pol)
        cmd = ["python3", "train.py", *args, "--data-dir", str(Path(a.data).resolve()), "--seed", "1",
               "--eval-only", str((pol / "model.pt").resolve()), "--run-name", f"eval-{run}", "--ckpt-dir",
               str(ROOT / "checkpoints" / "player_eval"), *a.extra.split()]
        out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        m = re.search(r"EVAL_ONLY .*val/total=([\d.]+) main_f1=([\d.]+) btn_f1=([\d.]+) batches=(\d+)", out.stdout)
        if not m:
            print(out.stdout[-2000:], out.stderr[-2000:])
            rows.append(dict(policy=str(pol), error="no EVAL_ONLY line"))
            continue
        rows.append(dict(policy=str(pol), loss=float(m[1]), main_f1=float(m[2]), btn_f1=float(m[3]), batches=int(m[4])))
        print(rows[-1], flush=True)
    out = Path(a.out or Path(a.data) / "player_eval.json")
    out.write_text(json.dumps(rows, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("finetune")
    f.add_argument("--parent", required=True, help="BC run to start from")
    f.add_argument("--data", required=True)
    f.add_argument("--name", required=True)
    f.add_argument("--steps", type=int, default=2000)
    f.add_argument("--lr", type=float, default=1e-4)
    f.add_argument("--muon-lr", type=float, default=0.002)
    f.add_argument("--seed", type=int, default=1)
    f.add_argument("--local-batch", type=int, default=None, help="smoke tests only (the recipe's batch otherwise)")
    f.add_argument("--extra", default="", help="more train.py flags, one string (smoke tests: '--no-amp --no-compile')")
    e = sub.add_parser("eval")
    e.add_argument("--data", required=True)
    e.add_argument("--policies", nargs="+", required=True)
    e.add_argument("--out")
    e.add_argument("--extra", default="", help="more train.py flags, one string (on the CPU: '--no-amp --no-compile')")
    a = ap.parse_args()
    finetune(a) if a.cmd == "finetune" else evaluate(a)


if __name__ == "__main__":
    main()
