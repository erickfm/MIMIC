#!/usr/bin/env python3
"""PLAN (dry-run by default) the in-place correction of erickfm/melee-ranked-replays.

The scramble is a fixed permutation of folder labels in the buggy batches
(a1,a2,a4,a5,a6); batch a3 is already correct. Because each character bucket
is "all games containing that character," every scrambled tarball is wholly
the character it maps to -> most of the fix is server-side renames
(CommitOperationCopy + Delete, no data transfer). The two collapsed labels
(ZELDA_SHEIK, ICE_CLIMBERS) are genuine mixes and need download/split/reupload.

This script ONLY plans + verifies. It performs NO writes. It:
  1. derives the buggy-folder -> true-character map from the verified tables,
  2. classifies every tarball into: keep(a3) / clean-rename / split,
  3. checks for destination collisions,
  4. spot-checks a3-correctness on a few untested folders via libmelee.
"""
import re, glob, subprocess, tempfile, random
from pathlib import Path
from collections import defaultdict, Counter
from huggingface_hub import HfApi, hf_hub_download
from melee import Console, Character

REPO = "erickfm/melee-ranked-replays"
CORRECT_BATCH = 3
COLLAPSE = {"ZELDA": "ZELDA_SHEIK", "SHEIK": "ZELDA_SHEIK",
            "POPO": "ICE_CLIMBERS", "NANA": "ICE_CLIMBERS"}

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


def build_folder_map():
    """buggy folder name -> ('rename', true) or ('split', {true: ext,...})."""
    preimages = defaultdict(list)   # buggy label -> [ext ids]
    for ext in PEPPI_TO_LIBMELEE:
        preimages[CHAR_NAME[ext]].append(ext)
    fmap = {}
    for label, exts in preimages.items():
        trues = {CHAR_NAME[PEPPI_TO_LIBMELEE[e]]: e for e in exts}
        if len(trues) == 1:
            fmap[label] = ("rename", next(iter(trues)))
        else:
            fmap[label] = ("split", trues)
    return fmap


TARBALL_RE = re.compile(r"([A-Z_]+)/\1_(\w+-\w+)_a(\d+)\.tar\.gz")


def main():
    api = HfApi()
    files = [f for f in api.list_repo_files(REPO, repo_type="dataset")
             if f.endswith(".tar.gz")]
    fmap = build_folder_map()

    print("=== derived buggy-folder -> true-character permutation ===")
    for label in sorted(fmap):
        kind, val = fmap[label]
        if kind == "rename":
            tag = "" if val == label else "  <-- moves"
            print(f"  {label:14s} -> {val:14s}{tag}")
        else:
            print(f"  {label:14s} -> SPLIT into {sorted(val)}")
    print()

    keep, renames, splits = [], [], []
    dest_paths = Counter()
    for f in files:
        m = TARBALL_RE.match(f)
        if not m:
            continue
        folder, rank, arch = m.group(1), m.group(2), int(m.group(3))
        if arch == CORRECT_BATCH:
            keep.append(f); continue
        kind, val = fmap[folder]
        if kind == "rename":
            if val == folder:
                keep.append(f)          # fixed point (e.g. KIRBY, GANONDORF)
            else:
                dst = f"{val}/{val}_{rank}_a{arch}.tar.gz"
                renames.append((f, dst)); dest_paths[dst] += 1
        else:
            splits.append(f)

    collisions = {d: n for d, n in dest_paths.items() if n > 1}
    a3_collisions = [d for d in dest_paths if d.endswith("_a3.tar.gz")]

    print(f"=== plan over {len(files)} tarballs ===")
    print(f"  keep as-is (a3 + fixed points): {len(keep)}")
    print(f"  clean server-side renames:      {len(renames)}")
    print(f"  split (download/re-tar):        {len(splits)}  "
          f"(only ZELDA_SHEIK + ICE_CLIMBERS buggy batches)")
    print(f"  destination collisions:         {len(collisions)}")
    if collisions:
        for d, n in list(collisions.items())[:10]:
            print(f"    COLLISION {d} <- {n} sources")
    print(f"  rename dsts landing on a3 paths: {len(a3_collisions)} (should be 0)")
    print()
    print("  sample renames:")
    for src, dst in renames[:8]:
        print(f"    {src}  ->  {dst}")
    print("  split tarballs:")
    for f in splits[:6]:
        print(f"    {f}")
    print(f"    ... ({len(splits)} total)")

    # spot-check a3 correctness on a few untested folders
    print("\n=== a3 correctness spot-check (libmelee) ===")
    a3 = [f for f in files if f.endswith("_a3.tar.gz")]
    random.seed(5); random.shuffle(a3)
    checks = [f for f in a3 if f.split("/")[0] in
              ("MARIO", "GANONDORF", "PIKACHU", "LINK")][:3]
    for repo_path in checks:
        folder = repo_path.split("/")[0]
        tmp = Path(tempfile.mkdtemp(prefix="a3chk_"))
        try:
            tp = hf_hub_download(REPO, repo_path, repo_type="dataset",
                                 local_dir=str(tmp / "dl"))
            subprocess.run(["tar", "-xzf", tp, "-C", str(tmp)],
                           check=True, capture_output=True)
            slps = glob.glob(str(tmp / "**" / "*.slp"), recursive=True)
            random.shuffle(slps)
            hit = 0; n = 0
            for s in slps[:25]:
                try:
                    c = Console(is_dolphin=False, path=s, allow_old_version=True)
                    c.connect(); gs = c.step()
                    if gs is None: c.stop(); continue
                    names = [COLLAPSE.get(p.character.name, p.character.name)
                             for p in gs.players.values()]
                    c.stop()
                    if folder in names: hit += 1
                    n += 1
                except Exception:
                    pass
            print(f"  {folder:12s} a3: {hit}/{n} contain real {folder} "
                  f"({'OK' if n and hit==n else 'CHECK'})")
        finally:
            subprocess.run(["rm", "-rf", str(tmp)], check=False)


if __name__ == "__main__":
    main()
