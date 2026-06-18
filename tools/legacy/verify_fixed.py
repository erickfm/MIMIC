#!/usr/bin/env python3
"""Verify _fixed/ folders contain their named character (libmelee ground truth).
For ZELDA/SHEIK, checks MAJORITY in-game form. Samples one tarball per folder."""
import glob, subprocess, tempfile, sys
from pathlib import Path
from collections import Counter
from huggingface_hub import HfApi, hf_hub_download
from melee import Console, Character
import peppi_py

REPO = "erickfm/melee-ranked-replays"
FORM_ZELDA, FORM_SHEIK = 19, 7
N = 30

# folder -> (libmelee names that count as a match)
COLLAPSE = {"ZELDA": "ZELDA", "SHEIK": "SHEIK", "POPO": "ICE_CLIMBERS", "NANA": "ICE_CLIMBERS"}


def libmelee_names(path):
    c = Console(is_dolphin=False, path=path, allow_old_version=True); c.connect()
    gs = c.step()
    if gs is None:
        c.stop(); return None
    names = [COLLAPSE.get(p.character.name, p.character.name) for p in gs.players.values()]
    c.stop()
    return names


def majority_form_has(path, want_form):
    g = peppi_py.read_slippi(path)
    for i in range(4):
        try:
            ch = g.frames.ports[i].leader.post.character.to_numpy()
        except Exception:
            continue
        z = int((ch == FORM_ZELDA).sum()); s = int((ch == FORM_SHEIK).sum())
        if z + s == 0:
            continue
        form = "ZELDA" if z >= s else "SHEIK"
        if form == want_form:
            return True
    return False


def verify_folder(api, folder, sample_tarball):
    tmp = Path(tempfile.mkdtemp(prefix="vf_"))
    try:
        tp = hf_hub_download(REPO, sample_tarball, repo_type="dataset", local_dir=str(tmp / "dl"))
        subprocess.run(["tar", "-xzf", tp, "-C", str(tmp)], check=True, capture_output=True)
        files = glob.glob(str(tmp / "**" / "*.slp"), recursive=True)[:N]
        hit = 0; n = 0; seen = set(); dup = 0
        for f in files:
            fn = Path(f).name
            if fn in seen: dup += 1
            seen.add(fn)
            try:
                names = libmelee_names(f)
                if not names: continue
                n += 1
                if folder in ("ZELDA", "SHEIK"):
                    ok = majority_form_has(f, folder)
                else:
                    ok = folder in names
                if ok: hit += 1
            except Exception:
                pass
        flag = "OK" if n and hit == n else "** CHECK **"
        print(f"  {folder:14s} {hit}/{n} contain {folder}  dups={dup}  {flag}")
    finally:
        subprocess.run(["rm", "-rf", str(tmp)], check=False)


def main():
    api = HfApi()
    files = api.list_repo_files(REPO, repo_type="dataset")
    # pick one representative tarball per requested folder
    folders = sys.argv[1:] or ["ZELDA", "SHEIK", "LUIGI", "MEWTWO", "NESS",
                               "ICE_CLIMBERS", "MARTH", "FOX", "SAMUS", "PIKACHU"]
    for folder in folders:
        cands = [f for f in files if f.startswith(f"_fixed/{folder}/")]
        # prefer a master-tier tarball with decent size
        cands.sort(key=lambda x: ("master-master" not in x, x))
        if cands:
            verify_folder(api, folder, cands[0])
        else:
            print(f"  {folder}: no tarballs")


if __name__ == "__main__":
    main()
