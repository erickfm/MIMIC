#!/usr/bin/env python3
"""After tools/allchar_queue.py: switch the coach to the RL policies and retrain its value models.

A value model reads its policy's trunk, so each character's value model is retrained on the RL
policy's features (and re-validated against sim rollouts, tools/coach/value_check.py) before the
coach uses that policy. Per character:
  1. the policy: Falco's M RL run, Fox's M RL run (fox-master), else the character's S RL run;
  2. replays: ~/Slippi when it has enough games of that character, else master-vs-master archives
     from erickfm/melee-ranked-replays (smallest first, about 1.5 GB);
  3. value.py extract --policy (GPU) -> data/coach_rank/vfeat_<char>_rl.npz; train -> value_<char>.pt
     (the BC one is kept as value_<char>_bc.pt);
  4. value_check.py --n 40 -> data/coach_rank/vcheck_<char>_rl.json;
  5. tools/coach/policies.json gets the character (the coach reads it at start).
Needs the GPU to itself: run it only once the queue has finished.

  python3 tools/coach/after_queue.py --dry-run     # what it would do
  python3 tools/coach/after_queue.py               # all characters with an RL run
  python3 tools/coach/after_queue.py --chars FALCO FOX
"""
import argparse, json, os, shutil, subprocess, sys, tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "tools", ROOT / "tools" / "coach"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
DATA = ROOT / "data" / "coach_rank"
POLICIES = ROOT / "tools" / "coach" / "policies.json"
RANKED = "erickfm/melee-ranked-replays"
LOCAL_ENOUGH = 600        # perspectives of the character in ~/Slippi to train on them alone
FIXED = {"FOX": "runs/fox-M-bc2-mlr01-s1.rl2-s1/policy"}   # fox-master (M BC + RL), already promoted
HF_BUCKET = {"JIGGLYPUFF": "JIGGLYPUFF", "GAMEANDWATCH": "GAMEANDWATCH"}


def run_status(name):
    p = ROOT / "runs" / name / "run.json"
    return json.loads(p.read_text()).get("status") if p.exists() else None


def rl_policy(char):
    """The RL policy the coach should use for `char`, as a path under the repo, or None."""
    from serve import RUN_CHAR
    if char in FIXED and (ROOT / FIXED[char] / "model.pt").exists():
        return FIXED[char]
    key = RUN_CHAR.get(char)
    if not key:
        return None
    for scale in ("M", "S"):
        runs = sorted((ROOT / "runs").glob(f"{key}-{scale}-*.rl2-s*"), key=lambda q: q.stat().st_mtime)
        runs = [r for r in runs if run_status(r.name) == "done" and (r / "policy" / "model.pt").exists()]
        if runs:
            return str((runs[-1] / "policy").relative_to(ROOT))
    return None


def local_count(char):
    lib = Path.home() / ".cache" / "mimic-coach" / "library.json"
    if not lib.exists():
        return 0
    n = 0
    for v in json.loads(lib.read_text()).values():
        ps = v.get("players") or []
        if len(ps) == 2:
            n += sum(1 for p in ps if p.get("char") == char)
    return n


def ranked_replays(char, budget=1.5e9):
    """Master-vs-master archives of `char` from the ranked dataset, extracted; returns the folder."""
    from huggingface_hub import HfApi, hf_hub_download
    out = DATA / "ranked" / char.lower()
    if out.exists() and any(out.rglob("*.slp")):
        return out
    bucket = HF_BUCKET.get(char, char)
    files = [f for f in HfApi().list_repo_tree(RANKED, path_in_repo=bucket, repo_type="dataset")
             if hasattr(f, "size") and "master-master" in f.path and f.size > 1e6]
    files.sort(key=lambda f: f.size)
    got = 0
    out.mkdir(parents=True, exist_ok=True)
    for f in files:
        if got > budget:
            break
        p = hf_hub_download(RANKED, f.path, repo_type="dataset", local_dir=str(DATA / "ranked" / "_dl"))
        with tarfile.open(p) as t:
            t.extractall(out, filter="data")
        os.remove(p)
        got += f.size
    return out


def sh(cmd, log):
    print("$", " ".join(cmd), flush=True)
    with open(log, "a") as f:
        return subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chars", nargs="*")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    from serve import RUN_CHAR
    chars = a.chars or sorted(RUN_CHAR)
    done = json.loads(POLICIES.read_text()) if POLICIES.exists() else {}
    report = {}
    for char in chars:
        pol = rl_policy(char)
        if not pol:
            print(f"{char}: no RL run yet", flush=True)
            continue
        n_local = local_count(char)
        src = "~/Slippi" if n_local >= LOCAL_ENOUGH else "ranked master-master"
        print(f"{char}: {pol}; replays from {src} ({n_local} local)", flush=True)
        if a.dry_run or done.get(char) == pol:
            continue
        log = DATA / f"after_queue_{char.lower()}.log"
        replays = str(Path.home() / "Slippi") if n_local >= LOCAL_ENOUGH else str(ranked_replays(char))
        feats = DATA / f"vfeat_{char.lower()}_rl.npz"
        vpath = DATA / f"value_{char.lower()}.pt"
        if sh(["python3", "tools/coach/value.py", "extract", "--char", char, "--replays", replays, "--out", str(feats),
               "--policy", pol, "--max-games", "1500"], log):
            report[char] = "extract failed"
            continue
        if vpath.exists() and not (DATA / f"value_{char.lower()}_bc.pt").exists():
            shutil.copy(vpath, DATA / f"value_{char.lower()}_bc.pt")
        if sh(["python3", "tools/coach/value.py", "train", "--data", str(feats), "--out", str(vpath)], log):
            report[char] = "train failed"
            continue
        # the coach uses the RL policy from here on (value_check reads POLICY through serve)
        done[char] = pol
        POLICIES.write_text(json.dumps(done, indent=1) + "\n")
        rc = sh(["python3", "tools/coach/value_check.py", "--char", char, "--value", str(vpath), "--n", "40",
                 "--out", str(DATA / f"vcheck_{char.lower()}_rl.json"), "--replays", replays], log)
        report[char] = "ok" if rc == 0 else "value_check failed"
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
