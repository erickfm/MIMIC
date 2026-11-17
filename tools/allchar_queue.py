#!/usr/bin/env python3
"""Train RL models for every character, one after another, on this machine's GPU.

For each job in order: download the character's master shards from erickfm/mimic-shards-v2 (if not
here), register them in recipes/datasets.yaml, train BC with tools/run.py (the step budget scales
with the data, see steps_for), delete the shards once no later BC job needs them, then RL (rl-v2,
which runs the 192-game h2h vs the BC parent). The next character's shards download while the
current one trains. A failed step is recorded and the queue moves on.

State: runs/allchar_queue.json (one entry per job: status, run names, times, error).
Log:   runs/allchar_queue.log

  nohup python3 tools/allchar_queue.py > /dev/null 2>&1 &
  python3 tools/allchar_queue.py --status
"""
import argparse, datetime as dt, hashlib, json, os, shutil, subprocess, sys, threading, time
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

REPO = Path(__file__).resolve().parents[1]
STATE = REPO / "runs" / "allchar_queue.json"
LOG = REPO / "runs" / "allchar_queue.log"
DATASETS = REPO / "recipes" / "datasets.yaml"
PT = ZoneInfo("America/Los_Angeles")

# (char, scale): Falco M first (asked for), then S for every character with shards, biggest data
# first. Ice Climbers don't reproduce in the sim (0/36), so no RL. Fox S and Zelda S added at the end
# (owner, 2026-11-06: the coach uses S RL throughout; Fox's only S RL was rl-v1, Zelda had none).
# Zelda has no shards on HF: they're built locally from its master-* ranked archives into data_dir.
JOBS = [("falco", "M"), ("falco", "S"), ("marth", "S"), ("sheik", "S"), ("puff", "S"), ("cptfalcon", "S"),
        ("peach", "S"), ("samus", "S"), ("luigi", "S"), ("pikachu", "S"), ("dk", "S"), ("doc", "S"),
        ("yoshi", "S"), ("ganondorf", "S"), ("link", "S"), ("bowser", "S"), ("gameandwatch", "S"),
        ("mario", "S"), ("roy", "S"), ("ness", "S"), ("ylink", "S"), ("mewtwo", "S"),
        ("fox", "S"), ("zelda", "S")]
RECIPE = {"M": "recipes/bc-M-v3.yaml", "S": "recipes/bc-S-v2.yaml"}


def now():
    return dt.datetime.now(PT).strftime("%a %-I:%M %p")


def log(msg):
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def load():
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save(st):
    STATE.write_text(json.dumps(st, indent=1) + "\n")


def data_dir(char):
    return REPO / "data" / f"{char}_hf" / char


def download(char):
    d = data_dir(char)
    if (d / "tensor_manifest.json").exists():
        m = json.loads((d / "tensor_manifest.json").read_text())
        if all((d / s).exists() for s in m["train_shards"] + m["val_shards"]):
            return
    from huggingface_hub import snapshot_download
    snapshot_download("erickfm/mimic-shards-v2", repo_type="dataset", allow_patterns=[f"{char}/*"],
                      local_dir=str(d.parent), max_workers=8)


def register(char):
    """recipes/datasets.yaml entry <char>-hf for the downloaded shards; returns its id."""
    d = data_dir(char)
    sha = hashlib.sha1((d / "tensor_manifest.json").read_bytes()).hexdigest()
    ds = yaml.safe_load(DATASETS.read_text())
    key = f"{char}-hf"
    if key not in ds or ds[key]["manifest_sha1"] != sha:
        m = json.loads((d / "tensor_manifest.json").read_text())
        with open(DATASETS, "a") as f:
            f.write(f"{key}:\n  note: All of erickfm/mimic-shards-v2 {char}/ ({m['n_train_games']:,} train / "
                    f"{m['n_val_games']:,} val games, {m['n_train_frames'] / 1e6:.0f}M train frames).\n"
                    f"  manifest_sha1: {sha}\n  paths: [{d.relative_to(REPO)}]\n")
    return key


def steps_for(char):
    """The 2026-07 all-character retrain's plateau fit (val argmin vs train frames, eff batch 512):
    steps = 53.3 * Mframes^1.19, here at this recipe's eff batch 768 and clipped to [3000, 56000].
    It gives fox-hf-647 51k (its recipe uses 56k) and Falco 28k."""
    m = json.loads((data_dir(char) / "tensor_manifest.json").read_text())
    s = 53.3 * (m["n_train_frames"] / 1e6) ** 1.19 * 512 / 768
    return int(min(56000, max(3000, round(s / 1000) * 1000)))


def run(cmd, tag):
    log(f"$ {' '.join(cmd)}")
    with open(REPO / "runs" / f"allchar_queue.{tag}.out", "a") as f:
        return subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT).returncode


def run_status(name):
    p = REPO / "runs" / name / "run.json"
    return json.loads(p.read_text()).get("status") if p.exists() else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    st = load()
    if a.status:
        for c, sc in JOBS:
            j = st.get(f"{c}-{sc}", {})
            print(f"{c + '-' + sc:<16} {j.get('status', 'queued'):<10} {j.get('bc', ''):<28} {j.get('rl', '')}  {j.get('error', '')}")
        return
    log(f"queue start: {len(JOBS)} jobs")
    prefetch = {}

    def fetch(char):
        try:
            download(char)
        except Exception as e:   # retried in the main loop
            log(f"prefetch {char} failed: {e}")

    for n, (char, scale) in enumerate(JOBS):
        key = f"{char}-{scale}"
        j = st.setdefault(key, {})
        if j.get("status") in ("done", "failed"):
            continue
        j.update(status="running", started=now())
        save(st)
        try:
            if char in prefetch:
                prefetch.pop(char).join()
            log(f"{key}: data")
            download(char)
            ds = register(char)
            steps = steps_for(char)
            tag = f"hf{steps // 1000}k"
            recipe = yaml.safe_load((REPO / RECIPE[scale]).read_text())
            bc = f"{char}-{scale}-{recipe['tag']}-{tag}-s{a.seed}"
            j.update(bc=bc, steps=steps)
            save(st)
            if run_status(bc) != "done":
                if (REPO / "runs" / bc).exists():          # a failed or interrupted earlier attempt
                    shutil.move(str(REPO / "runs" / bc), str(REPO / "runs" / f"{bc}.failed-{int(time.time())}"))
                log(f"{key}: BC {bc} ({steps} steps)")
                run(["python3", "tools/run.py", "launch", RECIPE[scale], "--char", char, "--seed", str(a.seed),
                     "--tag", tag, "--set", f"data={ds}", "--set", f"steps={steps}"], key)
                if run_status(bc) != "done":
                    raise RuntimeError(f"BC {bc} did not finish (runs/{bc}/logs/train.log)")
            # the next character downloads while this one runs RL
            nxt = next((c for c, _ in JOBS[n + 1:] if c != char), None)
            if nxt and nxt not in prefetch:
                prefetch[nxt] = threading.Thread(target=fetch, args=(nxt,), daemon=True)
                prefetch[nxt].start()
            if not any(c == char for c, _ in JOBS[n + 1:]):
                shutil.rmtree(data_dir(char).parent, ignore_errors=True)
                log(f"{key}: deleted {char} shards")
            rl = f"{bc}.rl2-s{a.seed}"
            j.update(rl=rl)
            save(st)
            if run_status(rl) != "done":
                if (REPO / "runs" / rl).exists():
                    shutil.move(str(REPO / "runs" / rl), str(REPO / "runs" / f"{rl}.failed-{int(time.time())}"))
                log(f"{key}: RL {rl}")
                run(["python3", "tools/run.py", "launch", "recipes/rl-v2.yaml", "--parent", bc, "--seed", str(a.seed)], key)
                if run_status(rl) != "done":
                    raise RuntimeError(f"RL {rl} did not finish (runs/{rl}/logs/train.log)")
            h2h = sorted((REPO / "runs" / rl / "eval").glob("*.result.json"))
            if h2h:
                r = json.loads(h2h[-1].read_text())
                j["h2h"] = f"{r.get('a_wins')}-{r.get('b_wins')} vs BC"
            j.update(status="done", finished=now())
            log(f"{key}: done {j.get('h2h', '')}")
        except Exception as e:
            j.update(status="failed", error=str(e)[:300], finished=now())
            log(f"{key}: FAILED {e}")
        save(st)
    log("queue finished")


if __name__ == "__main__":
    main()
