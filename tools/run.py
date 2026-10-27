#!/usr/bin/env python3
"""Recipe launcher, run registry and h2h bookkeeping (recipes/*.yaml -> runs/<name>/).

A recipe is a frozen YAML file in recipes/: every flag its stage needs, the dataset by id (pinned by
manifest hash in recipes/datasets.yaml), the step budget. Change anything -> new recipe version.

Run names (built here, never typed by hand):
    BC:  {char}-{scale}-{recipe tag}[-{tag}]-s{seed}            fox-S-bc1-s1, fox-S-bc1-14k-s1
    RL:  {parent}.{recipe tag}[-{tag}]-s{seed}                  fox-S-bc1-s1.rl2-s1
--set overrides need --tag, so a name always says how the run differs from its recipe.

Each run lives in runs/<name>/: run.json (recipe, overrides, resolved command, commit, host, times),
logs/, ckpt/ (BC) or rl/ (RL), policy/ (model.pt + norm JSONs: what sim_h2h / sim_rl / play load)
and eval/ (one small *.result.json per h2h). runs/registry.tsv is rebuilt from those files.

  tools/run.py launch recipes/bc-S-v1.yaml --char fox --seed 1 [--tag 14k --set steps=14000] [--gpus N]
  tools/run.py launch recipes/rl-v2.yaml --parent fox-S-bc1-s1 --seed 1
  tools/run.py h2h fox-S-bc1-s1.rl2-s1 fox-S-bc1-s1 [--games 192 --seed 41]
  tools/run.py register fox-S-bc1-s1 --recipe recipes/bc-S-v1.yaml --policy checkpoints/smallfox/base_full56k
  tools/run.py registry
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"
DATASETS = REPO / "recipes" / "datasets.yaml"
NORM_JSONS = ("cat_maps", "controller_combos", "mimic_norm", "norm_stats", "stick_clusters")
CHARS = {"fox": "FOX"}          # run-name char -> libmelee name for sim_h2h; add as needed
H2H_DEFAULTS = dict(games=192, seed=41, envs=48, workers=16, templates="replays_doc_rt")


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def git_commit():
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True,
                         text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO,
                           capture_output=True, text=True).stdout.strip()
    return sha + ("-dirty" if dirty else "")


def load_recipe(path, sets):
    r = yaml.safe_load(Path(path).read_text())
    r["file"] = str(Path(path).resolve().relative_to(REPO))
    for kv in sets:                                  # top-level keys (steps, data, ...) or args.<flag>
        k, v = kv.split("=", 1)
        v = yaml.safe_load(v)
        if k in r and k != "args":
            r[k] = v
        else:
            r.setdefault("args", {})[k.removeprefix("args.")] = v
    return r


def flags(args):
    out = []
    for k, v in args.items():
        if v is True:
            out.append(f"--{k}")
        elif v is False:
            out.append(f"--no-{k}")
        elif v is not None:
            out += [f"--{k}", str(v)]
    return out


def resolve_data(data_id):
    d = yaml.safe_load(DATASETS.read_text())[data_id]
    for p in d["paths"]:
        m = (REPO / p) / "tensor_manifest.json"
        if m.exists() and hashlib.sha1(m.read_bytes()).hexdigest() == d["manifest_sha1"]:
            return str(REPO / p) if not Path(p).is_absolute() else p
    sys.exit(f"dataset {data_id}: no path in {d['paths']} has manifest sha1 {d['manifest_sha1']}")


def run_dir(name):
    return RUNS / name


def read_run(name):
    p = run_dir(name) / "run.json"
    if not p.exists():
        sys.exit(f"no run {name} ({p})")
    return json.loads(p.read_text())


def write_run(name, info):
    (run_dir(name) / "run.json").write_text(json.dumps(info, indent=1) + "\n")


def export_policy(rd, model_path, norm_src):
    pol = rd / "policy"
    pol.mkdir(exist_ok=True)
    for f in NORM_JSONS:
        shutil.copy(Path(norm_src) / f"{f}.json", pol / f"{f}.json")
    link = pol / "model.pt"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(Path(model_path).resolve())


def sh(cmd, log, env=None):
    print(f"$ {' '.join(cmd)}\n  > {log}", flush=True)
    with open(log, "a") as f:
        return subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT, env=env).returncode


# ── launch ──────────────────────────────────────────────────────────────────

def cmd_launch(a):
    r = load_recipe(a.recipe, a.set)
    if a.set and not a.tag:
        sys.exit("--set changes the recipe: give the run a --tag that says how (e.g. --tag 14k)")
    commit = git_commit()
    if r.get("requires_commit") and not commit.startswith(r["requires_commit"]):
        sys.exit(f"{r['file']} only runs at commit {r['requires_commit']} (HEAD is {commit})")
    sfx = (f"-{a.tag}" if a.tag else "") + f"-s{a.seed}"
    if r["stage"] == "bc":
        if not a.char:
            sys.exit("BC runs need --char")
        name = f"{a.char}-{r['scale']}-{r['tag']}{sfx}"
    else:
        if not a.parent:
            sys.exit("RL runs need --parent (the BC run they start from)")
        name = f"{a.parent}.{r['tag']}{sfx}"
    rd = run_dir(name)
    if (rd / "run.json").exists():
        sys.exit(f"{name} already exists ({rd}); pick another --tag or --seed")
    rd.mkdir(parents=True)
    (rd / "logs").mkdir()
    info = dict(name=name, stage=r["stage"], recipe=r["file"], tag=a.tag, overrides=a.set, seed=a.seed,
                parent=a.parent, char=a.char or read_run(a.parent)["char"], commit=commit,
                host=socket.gethostname(), started=now(), status="running", recipe_resolved=r)
    if a.dry_run:
        shutil.rmtree(rd)
    t0 = time.time()
    try:
        ok = (launch_bc if r["stage"] == "bc" else launch_rl)(r, a, name, rd, info)
    except KeyboardInterrupt:
        ok = False
    if a.dry_run:
        return
    info.update(status="done" if ok else "failed", finished=now(), wall_min=round((time.time() - t0) / 60, 1))
    write_run(name, info)
    print(f"{name}: {info['status']} in {info['wall_min']} min")
    if ok and r["stage"] == "rl":
        h2h(name, a.parent, **H2H_DEFAULTS)
    cmd_registry(None)


def launch_bc(r, a, name, rd, info):
    gpus = a.gpus or (int(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
                          .stdout.count("GPU ")))
    per_gpu = r["eff_batch"] // gpus
    accum = max(1, -(-per_gpu // r["max_local_batch"]))
    local = per_gpu // accum
    if local * accum * gpus != r["eff_batch"]:
        sys.exit(f"eff_batch {r['eff_batch']} doesn't split over {gpus} GPUs x accum {accum}")
    data = resolve_data(r["data"])
    ckpt = rd / "ckpt"
    launcher = (["torchrun", f"--nproc_per_node={gpus}"] if gpus > 1 else ["python3"])
    cmd = launcher + ["train.py", *flags(r["args"]), "--data-dir", data, "--run-name", name,
                      "--ckpt-dir", str(ckpt), "--seed", str(a.seed), "--batch-size", str(local),
                      "--grad-accum-steps", str(accum), "--max-samples", str(r["eff_batch"] * r["steps"])]
    info.update(cmd=cmd, data_dir=data, gpus=gpus)
    if a.dry_run:
        print(" ".join(cmd))
        return True
    write_run(name, info)
    if sh(cmd, rd / "logs" / "train.log") != 0:
        return False
    steps = sorted(ckpt.glob(f"{name}_step*.pt"))
    exp = r.get("export", "last")
    if exp == "last":
        model = steps[-1]
    else:
        model = ckpt / f"{name}_swa{exp['swa_last']}.pt"
        if sh(["python3", "tools/average_checkpoints.py", "--out", str(model), "--run", str(ckpt / name),
               "--last", str(exp["swa_last"])], rd / "logs" / "export.log") != 0:
            return False
    export_policy(rd, model, data)
    info["policy_from"] = str(model.relative_to(REPO))
    if r.get("eval_only"):
        e = r["eval_only"]
        ev = ["python3", "train.py", *flags(r["args"]), "--data-dir", data, "--eval-only", str(model),
              "--seed", str(e["seed"]), "--batch-size", str(e["batch-size"])]
        env = {**os.environ, "WANDB_MODE": "disabled"}
        if sh(ev, rd / "logs" / "eval.log", env) != 0:
            return False
        m = re.findall(r"^EVAL_ONLY .*val/total=([0-9.]+)", (rd / "logs" / "eval.log").read_text(), re.M)
        info["val_fixed"] = float(m[-1]) if m else None
    return True


def launch_rl(r, a, name, rd, info):
    parent = run_dir(a.parent) / "policy"
    if not (parent / "model.pt").exists():
        sys.exit(f"parent {a.parent} has no policy/model.pt")
    out = rd / "rl"
    cmd = ["python3", "tools/sim_rl.py", "--policy-dir", str(parent), "--templates", r["templates"],
           "--out-dir", str(out), "--seed", str(a.seed), *flags(r["args"])]
    info.update(cmd=cmd)
    if a.dry_run:
        print(" ".join(cmd))
        return True
    write_run(name, info)
    if sh(cmd, rd / "logs" / "train.log") != 0:
        return False
    final = sorted(out.glob("rl_u*.pt"))[-1]
    export_policy(rd, final, parent)
    info["policy_from"] = str(final.relative_to(REPO))
    return True


# ── h2h ─────────────────────────────────────────────────────────────────────

def h2h(a_name, b_name, games, seed, envs, workers, templates):
    ra, rb = read_run(a_name), read_run(b_name)
    char = CHARS[ra["char"]]
    ev = run_dir(a_name) / "eval"
    ev.mkdir(exist_ok=True)
    stem = f"h2h_vs_{b_name}_g{games}_s{seed}"
    full = ev / f"{stem}.json"
    cmd = ["python3", "tools/sim_h2h.py", "--char", char,
           "--a", f"A={run_dir(a_name)}/policy:bf16", "--b", f"B={run_dir(b_name)}/policy:bf16",
           "--templates", templates, "--games", str(games), "--envs", str(envs), "--workers", str(workers),
           "--seed", str(seed), "--out", str(full)]
    if sh(cmd, ev / f"{stem}.log") != 0:
        print(f"h2h {a_name} vs {b_name} FAILED (see {ev / (stem + '.log')})")
        return
    rep = json.loads(full.read_text())["report"]
    res = dict(a=a_name, b=b_name, games=games, seed=seed, precision="bf16", templates=templates,
               a_wins=rep["a_wins"], b_wins=rep["b_wins"], draws=rep["draws"],
               a_win_rate=round(rep["a_win_rate"], 4), wilson95=[round(x, 4) for x in rep["wilson95"]],
               p_two_sided=rep["p_two_sided"], stock_margin_a=round(rep["mean_stock_margin_a"], 3),
               seconds=rep["seconds"], commit=git_commit(), finished=now())
    (ev / f"{stem}.result.json").write_text(json.dumps(res, indent=1) + "\n")
    print(f"{a_name} vs {b_name}: {res['a_wins']}-{res['b_wins']} ({res['a_win_rate']:.1%}, "
          f"p={res['p_two_sided']:.3g})")


def cmd_h2h(a):
    h2h(a.a, a.b, a.games, a.seed, a.envs, a.workers, a.templates)
    cmd_registry(None)


# ── register (runs made before this tool) ───────────────────────────────────

def cmd_register(a):
    rd = run_dir(a.name)
    pol = Path(a.policy).resolve()
    if (rd / "run.json").exists():                   # e.g. pulled via git on another machine: relink only
        link = rd / "policy"
        if link.is_symlink():
            link.unlink()
        link.symlink_to(pol)
        print(f"{a.name}: policy -> {pol}")
        return
    rd.mkdir(parents=True)
    (rd / "policy").symlink_to(pol)
    r = yaml.safe_load(Path(a.recipe).read_text()) if a.recipe else {}
    write_run(a.name, dict(name=a.name, stage=r.get("stage", a.stage), recipe=a.recipe, seed=a.seed,
                           parent=a.parent, char=a.char, status="legacy", legacy_policy=str(pol),
                           note=a.note, registered=now()))
    cmd_registry(None)


# ── registry ────────────────────────────────────────────────────────────────

def pt(iso):
    if not iso:
        return ""
    t = dt.datetime.fromisoformat(iso).astimezone(ZoneInfo("America/Los_Angeles"))
    return t.strftime("%Y-%m-%d %I:%M %p")


def cmd_registry(_):
    rows = []
    for p in sorted(RUNS.glob("*/run.json")):
        i = json.loads(p.read_text())
        h = []
        for e in sorted(p.parent.glob("eval/*.result.json")):
            x = json.loads(e.read_text())
            h.append(f"vs {x['b']}: {x['a_wins']}-{x['b_wins']} ({x['a_win_rate']:.1%})")
        rows.append([i["name"], i.get("stage") or "", Path(i.get("recipe") or "").stem, i.get("parent") or "",
                     str(i["seed"]) if i.get("seed") is not None else "", i["status"], pt(i.get("started")), str(i.get("wall_min", "")),
                     i.get("host", ""), i.get("commit", ""), str(i.get("val_fixed") or ""), "; ".join(h)])
    hdr = ["name", "stage", "recipe", "parent", "seed", "status", "started_pt", "wall_min", "host",
           "commit", "val_fixed", "h2h"]
    RUNS.mkdir(exist_ok=True)
    (RUNS / "registry.tsv").write_text("\n".join("\t".join(r) for r in [hdr] + rows) + "\n")
    w = [max(len(r[k]) for r in [hdr] + rows) for k in range(len(hdr))]
    for r in [hdr] + rows:
        print("  ".join(c.ljust(w[k]) for k, c in enumerate(r)).rstrip())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("launch")
    p.add_argument("recipe")
    p.add_argument("--char")
    p.add_argument("--parent")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--tag")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="recipe key (steps, data, ...) or a train/sim_rl flag; needs --tag")
    p.add_argument("--gpus", type=int)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_launch)
    p = sub.add_parser("h2h")
    p.add_argument("a")
    p.add_argument("b")
    for k, v in H2H_DEFAULTS.items():
        p.add_argument(f"--{k}", type=type(v), default=v)
    p.set_defaults(fn=cmd_h2h)
    p = sub.add_parser("register", help="add a run made before this tool, or relink an existing run's policy")
    p.add_argument("name")
    p.add_argument("--policy", required=True, help="existing policy dir (model.pt + norm JSONs)")
    p.add_argument("--recipe")
    p.add_argument("--stage", default="bc")
    p.add_argument("--char", default="fox")
    p.add_argument("--seed", type=int)
    p.add_argument("--parent")
    p.add_argument("--note")
    p.set_defaults(fn=cmd_register)
    p = sub.add_parser("registry")
    p.set_defaults(fn=cmd_registry)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
