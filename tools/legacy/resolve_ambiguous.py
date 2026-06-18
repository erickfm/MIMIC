#!/usr/bin/env python3
"""Resolve the collapsed-label ambiguity in data/ranked_index/index.jsonl.

Every ambiguous index row (true char unknown because the buggy label was the
collapsed ZELDA_SHEIK or ICE_CLIMBERS) lives in the ZELDA_SHEIK or ICE_CLIMBERS
bucket. This streams those buckets (master-tier by default), parses each .slp
with libmelee for ground-truth characters, then patches the index in place.

Streams one tarball at a time (download -> extract -> parse -> delete) so peak
disk stays small. Resumable: a per-tarball done-marker skips finished work.

Output: rewrites index.jsonl with resolved true_p1/true_p2 + ambiguous=False
for every file found; prints exact per-character master-tier counts after.
"""
import json, glob, subprocess, tempfile, collections, os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from huggingface_hub import hf_hub_download, HfApi
from melee import Console, Character

REPO = "erickfm/melee-ranked-replays"
ROOT = Path(__file__).resolve().parent.parent
IDX = ROOT / "data" / "ranked_index" / "index.jsonl"
RES = ROOT / "data" / "ranked_index" / "resolved.jsonl"
DONE = ROOT / "data" / "ranked_index" / "resolved_done.txt"
BUCKETS = ["ZELDA_SHEIK", "ICE_CLIMBERS"]
TIER_PREFIX = "_master-"   # master-tier only; set "" for all tiers
WORKERS = 24

COLLAPSE = {"ZELDA": "ZELDA_SHEIK", "SHEIK": "ZELDA_SHEIK",
            "POPO": "ICE_CLIMBERS", "NANA": "ICE_CLIMBERS"}


def _parse_one(path):
    try:
        c = Console(is_dolphin=False, path=path, allow_old_version=True)
        c.connect()
        gs = c.step()
        if gs is None:
            c.stop(); return (path, None)
        names = sorted(COLLAPSE.get(p.character.name, p.character.name)
                       for p in gs.players.values())
        c.stop()
        if len(names) != 2:
            return (path, None)
        return (path, names)
    except Exception:
        return (path, None)


def target_tarballs():
    api = HfApi()
    files = [f for f in api.list_repo_files(REPO, repo_type="dataset")
             if f.endswith(".tar.gz")]
    out = []
    for b in BUCKETS:
        out += sorted(f for f in files
                      if f.startswith(b + "/") and TIER_PREFIX in f)
    return out


def load_done():
    return set(DONE.read_text().split()) if DONE.exists() else set()


def main():
    done = load_done()
    tarballs = target_tarballs()
    print(f"{len(tarballs)} target tarballs ({len(done)} already done)")
    res_fh = open(RES, "a")
    for i, repo_path in enumerate(tarballs):
        if repo_path in done:
            continue
        tmp = Path(tempfile.mkdtemp(prefix="resolve_"))
        try:
            os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
            tarp = hf_hub_download(REPO, repo_path, repo_type="dataset",
                                   local_dir=str(tmp / "dl"))
            subprocess.run(["tar", "-xzf", str(tarp), "-C", str(tmp)],
                           check=True, capture_output=True)
            files = glob.glob(str(tmp / "**" / "*.slp"), recursive=True)
            n_ok = 0
            with ProcessPoolExecutor(max_workers=WORKERS) as ex:
                for path, names in ex.map(_parse_one, files, chunksize=32):
                    if names is None:
                        continue
                    res_fh.write(json.dumps(
                        {"filename": Path(path).name,
                         "true_p1": names[0], "true_p2": names[1]}) + "\n")
                    n_ok += 1
            res_fh.flush()
            with open(DONE, "a") as d:
                d.write(repo_path + "\n")
            print(f"[{i+1}/{len(tarballs)}] {repo_path.split('/')[1]}: "
                  f"{n_ok}/{len(files)} resolved", flush=True)
        finally:
            subprocess.run(["rm", "-rf", str(tmp)], check=False)
    res_fh.close()
    patch_index()


def patch_index():
    resolved = {}
    if RES.exists():
        for line in open(RES):
            e = json.loads(line)
            resolved[e["filename"]] = (e["true_p1"], e["true_p2"])
    print(f"\nresolved lookup: {len(resolved)} files")
    n_patched = 0
    rows = []
    for line in open(IDX):
        e = json.loads(line)
        if e.get("ambiguous") and e["filename"] in resolved:
            t1, t2 = resolved[e["filename"]]
            e["true_p1"], e["true_p2"] = t1, t2
            e["cand_p1"], e["cand_p2"] = [t1], [t2]
            e["ambiguous"] = False
            e["resolved_by"] = "libmelee"
            n_patched += 1
        rows.append(e)
    tmp = IDX.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as fh:
        for e in rows:
            fh.write(json.dumps(e) + "\n")
    tmp.replace(IDX)
    print(f"patched {n_patched} ambiguous rows")

    master = collections.Counter()
    still_amb = 0
    for e in rows:
        if e.get("ambiguous"):
            still_amb += 1
        if not (e["rank"] or "").startswith("master"):
            continue
        for k in ("true_p1", "true_p2"):
            if e.get(k):
                master[e[k]] += 1
    print(f"remaining ambiguous rows (non-master tiers): {still_amb}")
    print("\n=== EXACT master-tier games per character (post-resolution) ===")
    for name, n in master.most_common():
        print(f"  {name:14s} {n:7d}")


if __name__ == "__main__":
    main()
