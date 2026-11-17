#!/usr/bin/env python3
"""After tools/allchar_queue.py: switch the coach to the RL policies and retrain its value models.

A value model reads its policy's trunk, so each character's value model is retrained on the RL
policy's features (and re-validated against sim rollouts, tools/coach/value_check.py) before the
coach uses that policy. Per character:
  1. the policy: the character's S RL run (owner, 2026-11-06: the coach uses S RL models throughout,
     the same ones live play uses): its rl-v4 run (200 updates, owner 2026-11-16), else its rl2 run, else
     for Fox SF7u100 (fox-S-bc1-s1.rl1-s0, rl-v1);
  2. replays: ~/Slippi when it has enough games of that character, else ranked archives from
     erickfm/melee-ranked-replays (master-vs-master first, about 1.5 GB; see ranked_replays);
  3. value.py extract --policy (GPU) -> data/coach_rank/vfeat_<char>_rl.npz; train -> value_<char>.pt
     (the BC one is kept as value_<char>_bc.pt);
  4. value_check.py --n 40 on ~/Slippi (the sim is exact on current Slippi, not on the older ranked
     archives) -> data/coach_rank/vcheck_<char>_rl.json;
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
FIXED = {"FOX": "runs/fox-S-bc1-s1.rl1-s0/policy"}   # SF7u100 (beat fox-master 109-83 in the sim): fallback
HF_BUCKET = {"JIGGLYPUFF": "JIGGLYPUFF", "GAMEANDWATCH": "GAMEANDWATCH"}


def run_status(name):
    p = ROOT / "runs" / name / "run.json"
    return json.loads(p.read_text()).get("status") if p.exists() else None


def rl_policy(char):
    """The RL policy the coach should use for `char`, as a path under the repo, or None."""
    from serve import RUN_CHAR
    key = RUN_CHAR.get(char)
    if not key:
        return None
    # rl-v4 (200 updates, owner 2026-11-16) first, then the 100-update rl2 runs; newest of each
    for tag in ("rl4", "rl2"):
        runs = sorted((ROOT / "runs").glob(f"{key}-S-*.{tag}-s[0-9]"), key=lambda q: q.stat().st_mtime)
        runs = [r for r in runs if run_status(r.name) in ("done", "legacy") and (r / "policy" / "model.pt").exists()]
        if runs:
            return str((runs[-1] / "policy").relative_to(ROOT))
    if char in FIXED and (ROOT / FIXED[char] / "model.pt").exists():
        return FIXED[char]
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
    """Ranked replays of `char` from the ranked dataset, extracted; returns the folder. Master-vs-master
    archives first (smallest first); when a character has too few of those (Mewtwo, Ness, Zelda),
    master-vs-diamond and master-vs-platinum fill the budget, the same master-* pairs its BC model
    was trained on. Resumable: `.archives` lists what is extracted."""
    from huggingface_hub import HfApi, hf_hub_download
    out = DATA / "ranked" / char.lower()
    bucket = HF_BUCKET.get(char, char)
    files = [f for f in HfApi().list_repo_tree(RANKED, path_in_repo=bucket, repo_type="dataset")
             if hasattr(f, "size") and f.size > 1e6]
    pair = lambda f: next((i for i, k in enumerate(("master-master", "master-diamond", "master-platinum"))
                           if f"_{k}_" in f.path), None)
    mm = sorted((f for f in files if pair(f) == 0), key=lambda f: f.size)
    files = sorted((f for f in files if pair(f) is not None), key=lambda f: (pair(f), f.size))
    mark = out / ".archives"
    if mark.exists():
        have = set(mark.read_text().split())
    elif out.exists() and any(out.rglob("*.slp")):   # extracted before .archives existed: master-master only
        have, got = set(), 0
        for f in mm:
            if got > budget:
                break
            have.add(f.path)
            got += f.size
    else:
        have = set()
    out.mkdir(parents=True, exist_ok=True)
    got = sum(f.size for f in files if f.path in have)
    for f in files:
        if got > budget:
            break
        if f.path in have:
            continue
        p = hf_hub_download(RANKED, f.path, repo_type="dataset", local_dir=str(DATA / "ranked" / "_dl"))
        with tarfile.open(p) as t:
            t.extractall(out, filter="data")
        os.remove(p)
        got += f.size
        have.add(f.path)
        mark.write_text("\n".join(sorted(have)) + "\n")
    if not mark.exists():
        mark.write_text("\n".join(sorted(have)) + "\n")
    return out


def sh(cmd, log):
    print("$", " ".join(cmd), flush=True)
    with open(log, "a") as f:
        return subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chars", nargs="*")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--device", default=None,
                    help="cpu: run it while the GPU is busy (slower; start it with CUDA_VISIBLE_DEVICES= and nice)")
    a = ap.parse_args()
    from serve import RUN_CHAR
    chars = a.chars or sorted(RUN_CHAR)
    done = json.loads(POLICIES.read_text()) if POLICIES.exists() else {}
    dev = ["--device", a.device] if a.device else []
    report = {}
    for char in chars:
        pol = rl_policy(char)
        if not pol:
            print(f"{char}: no RL run yet", flush=True)
            continue
        n_local = local_count(char)
        src = "~/Slippi" if n_local >= LOCAL_ENOUGH else "ranked master-*"
        print(f"{char}: {pol}; replays from {src} ({n_local} local)", flush=True)
        if a.dry_run or done.get(char) == pol:
            continue
        log = DATA / f"after_queue_{char.lower()}.log"
        replays = str(Path.home() / "Slippi") if n_local >= LOCAL_ENOUGH else str(ranked_replays(char))
        feats = DATA / f"vfeat_{char.lower()}_rl.npz"
        vpath = DATA / f"value_{char.lower()}.pt"
        if sh(["python3", "tools/coach/value.py", "extract", "--char", char, "--replays", replays, "--out", str(feats),
               "--policy", pol, "--max-games", "1500", *dev], log):
            report[char] = "extract failed"
            continue
        if vpath.exists() and not (DATA / f"value_{char.lower()}_bc.pt").exists():
            shutil.copy(vpath, DATA / f"value_{char.lower()}_bc.pt")
        if sh(["python3", "tools/coach/value.py", "train", "--data", str(feats), "--out", str(vpath), *dev], log):
            report[char] = "train failed"
            continue
        # the coach uses the RL policy from here on (value_check reads POLICY through serve)
        done[char] = pol
        POLICIES.write_text(json.dumps(done, indent=1) + "\n")
        # checked on ~/Slippi either way: the sim replays current Slippi exactly, not the older ranked
        # archives; a V trained on ranked replays has never seen any of these games
        rc = sh(["python3", "tools/coach/value_check.py", "--char", char, "--value", str(vpath), "--n", "40",
                 "--out", str(DATA / f"vcheck_{char.lower()}_rl.json"), "--replays", str(Path.home() / "Slippi"),
                 *([] if n_local >= LOCAL_ENOUGH else ["--all-held-out"]), *dev], log)
        report[char] = "ok" if rc == 0 else "value_check failed"
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
