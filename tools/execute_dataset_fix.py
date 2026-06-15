#!/usr/bin/env python3
"""Execute the in-place fix of erickfm/melee-ranked-replays.

Builds the corrected dataset under the `_fixed/` prefix (non-destructive), in
resumable phases. The destructive swap is a SEPARATE subcommand, gated.

Subcommands:
  renames  - server-side CommitOperationCopy of all non-entangled tarballs to
             _fixed/{true_char}/ (free). a3 + fixed points copy unchanged.
  retar    - download the 3 entangled buckets, re-sort .slp by TRUE character
             (frame-majority for Zelda vs Sheik), upload split tarballs to
             _fixed/.
  metadata - regenerate metadata/metadata_a{N}.json into _fixed/metadata/.
  verify   - sample _fixed/ folders, check contents vs folder name via libmelee.
  swap     - DESTRUCTIVE: delete old folders, promote _fixed/* to top level.
             Requires --i-understand-this-is-destructive.

All HF writes go to _fixed/ except `swap`. Safe to re-run; per-item markers.
"""
import sys, os, glob, json, subprocess, tempfile, argparse, signal, time
from pathlib import Path
from collections import defaultdict
from huggingface_hub import HfApi, hf_hub_download, CommitOperationCopy, CommitOperationAdd, CommitOperationDelete
from melee import Character
import peppi_py

REPO = "erickfm/melee-ranked-replays"
ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "data" / "ranked_index"
STAGE = "_fixed"
CORRECT_BATCH = 3

# external (peppi) -> libmelee value
PEPPI_TO_LIBMELEE = {
    0: 2, 1: 3, 2: 1, 3: 24, 4: 4, 5: 5, 6: 6, 7: 17, 8: 0, 9: 18,
    10: 16, 11: 8, 12: 9, 13: 12, 14: 10, 15: 15, 16: 13, 17: 14,
    18: 19, 19: 7, 20: 22, 21: 20, 22: 21, 23: 26, 24: 23, 25: 25,
}
CHAR_NAME = {c.value: c.name for c in Character}
CHAR_NAME[Character.ZELDA.value] = "ZELDA_SHEIK"
CHAR_NAME[Character.SHEIK.value] = "ZELDA_SHEIK"
CHAR_NAME[Character.POPO.value] = "ICE_CLIMBERS"
CHAR_NAME[Character.NANA.value] = "ICE_CLIMBERS"

# peppi external picks of interest
PICK_LUIGI, PICK_ZELDA, PICK_SHEIK = 7, 18, 19
PICK_MEWTWO, PICK_NESS = 10, 11
FORM_ZELDA, FORM_SHEIK = 19, 7  # per-frame in-game char ids

ENTANGLED_FOLDERS = {"ZELDA_SHEIK", "ICE_CLIMBERS", "MARTH"}

import re
TARBALL_RE = re.compile(r"([A-Z_]+)/\1_(\w+-\w+)_a(\d+)\.tar\.gz")


def folder_perm():
    """non-collapsed buggy folder -> true char (clean rename)."""
    pre = defaultdict(list)
    for ext in PEPPI_TO_LIBMELEE:
        pre[CHAR_NAME[ext]].append(ext)
    m = {}
    for label, exts in pre.items():
        trues = {CHAR_NAME[PEPPI_TO_LIBMELEE[e]] for e in exts}
        if len(trues) == 1:
            m[label] = next(iter(trues))
    return m


def list_tarballs(api):
    return [f for f in api.list_repo_files(REPO, repo_type="dataset")
            if f.endswith(".tar.gz") and TARBALL_RE.match(f)]


# ---------------- renames ----------------
def phase_renames(api, dry):
    perm = folder_perm()
    ops = []
    skipped_retar = 0
    for f in list_tarballs(api):
        folder, rank, arch = TARBALL_RE.match(f).group(1, 2, 3)
        arch = int(arch)
        # routing
        if folder == "ZELDA_SHEIK":
            skipped_retar += 1; continue                      # retar (all batches)
        if folder == "ICE_CLIMBERS":
            if arch == CORRECT_BATCH:
                dst = f"{STAGE}/ICE_CLIMBERS/ICE_CLIMBERS_{rank}_a{arch}.tar.gz"
            else:
                skipped_retar += 1; continue                  # retar split
        elif folder == "MARTH":
            if arch == CORRECT_BATCH:
                dst = f"{STAGE}/MARTH/MARTH_{rank}_a{arch}.tar.gz"
            else:
                skipped_retar += 1; continue                  # retar (zelda)
        elif arch == CORRECT_BATCH:
            dst = f"{STAGE}/{folder}/{folder}_{rank}_a{arch}.tar.gz"   # a3 correct
        else:
            true = perm[folder]
            dst = f"{STAGE}/{true}/{true}_{rank}_a{arch}.tar.gz"        # clean rename
        ops.append((f, dst))

    print(f"renames: {len(ops)} copies, {skipped_retar} deferred to retar")
    done = _load(STATE / "renames_done.txt")
    todo = [(s, d) for s, d in ops if d not in done]
    print(f"  {len(todo)} remaining")
    if dry:
        for s, d in todo[:12]:
            print(f"    {s} -> {d}")
        return
    BATCH = 100
    for i in range(0, len(todo), BATCH):
        chunk = todo[i:i + BATCH]
        cops = [CommitOperationCopy(src_path_in_repo=s, path_in_repo=d) for s, d in chunk]
        api.create_commit(REPO, repo_type="dataset", operations=cops,
                          commit_message=f"build _fixed/: clean renames {i}-{i+len(chunk)}")
        with open(STATE / "renames_done.txt", "a") as fh:
            for _, d in chunk:
                fh.write(d + "\n")
        print(f"  committed {i+len(chunk)}/{len(todo)}", flush=True)


# ---------------- retar ----------------
def majority_form(gf, port):
    ch = gf.frames.ports[port].leader.post.character.to_numpy()
    return "ZELDA" if int((ch == FORM_ZELDA).sum()) >= int((ch == FORM_SHEIK).sum()) else "SHEIK"


def route_game(path, bucket):
    """Return set of TRUE target folders this game belongs to (entangled only)."""
    g = peppi_py.read_slippi(path, skip_frames=True)
    players = [(i, p.character) for i, p in enumerate(g.start.players) if p is not None]
    if len(players) != 2:
        return set()
    targets = set()
    gf = None
    for i, pick in players:
        if bucket == "ZELDA_SHEIK":
            if pick == PICK_LUIGI:
                targets.add("LUIGI")
            elif pick == PICK_SHEIK:
                targets.add("SHEIK")
            elif pick == PICK_ZELDA:
                if gf is None: gf = peppi_py.read_slippi(path)
                targets.add(majority_form(gf, i))
        elif bucket == "ICE_CLIMBERS":
            if pick == PICK_MEWTWO: targets.add("MEWTWO")
            elif pick == PICK_NESS: targets.add("NESS")
        elif bucket == "MARTH":
            if pick == PICK_ZELDA:
                if gf is None: gf = peppi_py.read_slippi(path)
                targets.add(majority_form(gf, i))
    return targets


def retar_bucket(api, bucket, dry):
    """Process all (or buggy) tarballs of `bucket`, splitting into _fixed/."""
    all_tb = [f for f in list_tarballs(api) if f.startswith(bucket + "/")]
    if bucket in ("ICE_CLIMBERS", "MARTH"):
        all_tb = [f for f in all_tb if f"_a{CORRECT_BATCH}." not in f]   # a3 handled by renames
    done = _load(STATE / f"retar_{bucket}_done.txt")
    todo = [f for f in all_tb if f not in done]
    print(f"retar {bucket}: {len(all_tb)} tarballs, {len(todo)} remaining")
    if dry:
        print("  (dry-run, no work)"); return
    arch_prefix = "m" if bucket == "MARTH" else "a"   # avoid SHEIK archive collision
    for rp in todo:
        folder, rank, arch = TARBALL_RE.match(rp).group(1, 2, 3)
        tmp = Path(tempfile.mkdtemp(prefix="retar_"))
        try:
            os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
            tp = hf_hub_download(REPO, rp, repo_type="dataset", local_dir=str(tmp / "dl"))
            (tmp / "ex").mkdir(parents=True, exist_ok=True)
            subprocess.run(["tar", "-xzf", tp, "-C", str(tmp / "ex")], check=True, capture_output=True)
            files = glob.glob(str(tmp / "ex" / "**" / "*.slp"), recursive=True)
            groups = defaultdict(list)   # target_char -> [slp paths]
            for sf in files:
                try:
                    for t in route_game(sf, folder):
                        groups[t].append(sf)
                except Exception:
                    pass
            ops = []
            for target, slps in groups.items():
                outdir = tmp / "out" / target
                outdir.mkdir(parents=True, exist_ok=True)
                for sf in slps:
                    os.link(sf, outdir / Path(sf).name)
                tarname = f"{target}_{rank}_{arch_prefix}{arch}.tar.gz"
                tarpath = tmp / tarname
                subprocess.run(["tar", "-czf", str(tarpath), "-C", str(outdir), "."],
                               check=True)
                ops.append(CommitOperationAdd(
                    path_in_repo=f"{STAGE}/{target}/{tarname}", path_or_fileobj=str(tarpath)))
            if ops:
                ok = commit_robust(api, ops,
                                   f"build _fixed/: retar {folder} {rank} a{arch}")
                if not ok:
                    print(f"  {rp}: upload FAILED after retries — leaving for resume", flush=True)
                    continue
            with open(STATE / f"retar_{bucket}_done.txt", "a") as fh:
                fh.write(rp + "\n")
            print(f"  {rp.split('/')[1]}: { {k: len(v) for k,v in groups.items()} }", flush=True)
        finally:
            subprocess.run(["rm", "-rf", str(tmp)], check=False)


def commit_robust(api, ops, msg, timeout=900, retries=6):
    """create_commit with a hard timeout + retry — huggingface_hub hangs on
    dead CLOSE-WAIT sockets on this network; SIGALRM breaks the stuck call."""
    for attempt in range(retries):
        def _to(signum, frame):
            raise TimeoutError("commit timed out")
        old = signal.signal(signal.SIGALRM, _to)
        signal.alarm(timeout)
        try:
            api.create_commit(REPO, repo_type="dataset", operations=ops,
                              commit_message=msg)
            return True
        except Exception as e:
            print(f"    upload attempt {attempt+1}/{retries} failed "
                  f"({type(e).__name__}); retry in 15s", flush=True)
            time.sleep(15)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)
    return False


def _load(p):
    return set(p.read_text().split()) if p.exists() else set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["renames", "retar", "metadata", "verify", "swap"])
    ap.add_argument("--bucket", help="for retar: ZELDA_SHEIK | ICE_CLIMBERS | MARTH")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--i-understand-this-is-destructive", action="store_true")
    a = ap.parse_args()
    api = HfApi()
    STATE.mkdir(parents=True, exist_ok=True)
    if a.phase == "renames":
        phase_renames(api, a.dry_run)
    elif a.phase == "retar":
        assert a.bucket in ("ZELDA_SHEIK", "ICE_CLIMBERS", "MARTH"), "need --bucket"
        retar_bucket(api, a.bucket, a.dry_run)
    elif a.phase == "swap":
        phase_swap(api, a.dry_run, a.i_understand_this_is_destructive)
    else:
        print(f"phase {a.phase} not yet implemented in this file")


def phase_swap(api, dry, confirmed):
    files = api.list_repo_files(REPO, repo_type="dataset")
    fixed = [f for f in files if f.startswith(STAGE + "/") and f.endswith(".tar.gz")]
    copies = [(f, f[len(STAGE) + 1:]) for f in fixed]   # _fixed/FOX/.. -> FOX/..
    dst = {d for _, d in copies}
    old_tar = [f for f in files if not f.startswith(STAGE + "/")
               and f.endswith(".tar.gz") and TARBALL_RE.match(f)]
    orphans = [f for f in old_tar if f not in dst]       # old paths not overwritten
    old_meta = [f for f in files if f.startswith("metadata/")]
    fixed_all = [f for f in files if f.startswith(STAGE + "/")]   # incl non-tar (none)

    print(f"SWAP plan:")
    print(f"  copy  _fixed/* -> top level : {len(copies)} (overwrites same-named old)")
    print(f"  delete orphan old tarballs  : {len(orphans)}")
    print(f"  delete old metadata         : {len(old_meta)}")
    print(f"  delete _fixed/ staging      : {len(fixed_all)}")
    print(f"  orphan folders: { sorted({o.split('/')[0] for o in orphans}) }")
    if dry or not confirmed:
        if not confirmed:
            print("  (need --i-understand-this-is-destructive to execute)")
        return

    def commit_batches(ops, msg):
        B = 100
        for i in range(0, len(ops), B):
            commit_robust(api, ops[i:i + B], f"{msg} {i}-{i+min(B,len(ops)-i)}")
            print(f"    {msg}: {min(i+B, len(ops))}/{len(ops)}", flush=True)

    # 1. promote: copy _fixed -> top level (overwrite), delete orphans + old metadata
    commit_batches([CommitOperationCopy(src_path_in_repo=s, path_in_repo=d)
                    for s, d in copies], "promote copy")
    commit_batches([CommitOperationDelete(path_in_repo=f)
                    for f in orphans + old_meta], "delete old")
    # 2. remove staging
    commit_batches([CommitOperationDelete(path_in_repo=f) for f in fixed_all],
                   "delete _fixed")
    print("SWAP COMPLETE")


if __name__ == "__main__":
    main()
