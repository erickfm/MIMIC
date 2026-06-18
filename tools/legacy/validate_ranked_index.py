#!/usr/bin/env python3
"""Validate data/ranked_index/index.jsonl against libmelee ground truth.

Downloads a stratified sample of tarballs (all archive batches, all rank
tiers, many characters), parses each .slp with libmelee, and checks the
index's recovered characters match. For ambiguous (collapsed-label) entries,
checks the libmelee truth is contained in the candidate set.

Pass tarball repo-paths as argv, or use the built-in curated set. Local
tarballs (FOX a1/a3) can be passed as filesystem paths.
"""
import sys, json, glob, random, tempfile, subprocess, collections
from pathlib import Path
from huggingface_hub import hf_hub_download
from melee import Console, Character

REPO = "erickfm/melee-ranked-replays"
IDX = Path(__file__).resolve().parent.parent / "data" / "ranked_index" / "index.jsonl"
SAMPLE_PER_TARBALL = 60

COLLAPSE = {"ZELDA": "ZELDA_SHEIK", "SHEIK": "ZELDA_SHEIK",
            "POPO": "ICE_CLIMBERS", "NANA": "ICE_CLIMBERS"}

CURATED = [
    "MARTH/MARTH_master-diamond_a2.tar.gz",
    "JIGGLYPUFF/JIGGLYPUFF_diamond-diamond_a2.tar.gz",
    "FALCO/FALCO_master-platinum_a4.tar.gz",
    "PEACH/PEACH_platinum-platinum_a4.tar.gz",
    "CPTFALCON/CPTFALCON_master-master_a5.tar.gz",
    "SAMUS/SAMUS_diamond-platinum_a5.tar.gz",
    "DOC/DOC_master-master_a6.tar.gz",
    "FOX/FOX_master-diamond_a6.tar.gz",
]


def load_index():
    idx = {}
    for line in open(IDX):
        e = json.loads(line)
        idx[e["filename"]] = e
    return idx


def libmelee_chars(path):
    c = Console(is_dolphin=False, path=path, allow_old_version=True)
    c.connect()
    gs = c.step()
    if gs is None:
        c.stop(); return None
    names = [p.character.name for p in gs.players.values()]
    c.stop()
    return sorted(COLLAPSE.get(n, n) for n in names)


def index_matches_truth(entry, truth):
    """truth: sorted 2-list of collapsed libmelee names.
    entry slots: true_p1/true_p2 (name or None) + cand_p1/cand_p2 (candidate list).
    Returns (ok, kind) where kind in {'exact','ambig_ok','MISMATCH'}."""
    slots = []
    for k_true, k_cand in (("true_p1", "cand_p1"), ("true_p2", "cand_p2")):
        t = entry.get(k_true)
        slots.append([t] if t is not None else list(entry.get(k_cand) or []))
    # bipartite match 2x2
    t0, t1 = truth
    def can(slot, name): return name in slot
    matched = ((can(slots[0], t0) and can(slots[1], t1)) or
               (can(slots[0], t1) and can(slots[1], t0)))
    if not matched:
        return False, "MISMATCH"
    kind = "exact" if entry.get("true_p1") and entry.get("true_p2") else "ambig_ok"
    return True, kind


def validate_tarball(repo_path, idx):
    tmp = Path(tempfile.mkdtemp(prefix="idxval_"))
    try:
        if Path(repo_path).exists():
            tarp = repo_path
        else:
            tarp = hf_hub_download(REPO, repo_path, repo_type="dataset",
                                   local_dir=str(tmp / "dl"))
        subprocess.run(["tar", "-xzf", str(tarp), "-C", str(tmp)],
                       check=True, capture_output=True)
        files = glob.glob(str(tmp / "**" / "*.slp"), recursive=True)
        random.seed(13); random.shuffle(files)
        files = files[:SAMPLE_PER_TARBALL]
        stats = collections.Counter()
        misses = []
        for f in files:
            fn = Path(f).name
            if fn not in idx:
                stats["not_in_index"] += 1; continue
            truth = libmelee_chars(f)
            if truth is None or len(truth) != 2:
                stats["unparseable"] += 1; continue
            ok, kind = index_matches_truth(idx[fn], truth)
            stats[kind] += 1
            if not ok and len(misses) < 5:
                misses.append((fn, truth, idx[fn].get("true_p1"),
                               idx[fn].get("true_p2"), idx[fn].get("archive")))
        return stats, misses
    finally:
        subprocess.run(["rm", "-rf", str(tmp)], check=False)


def main():
    idx = load_index()
    print(f"index: {len(idx)} replays\n")
    targets = sys.argv[1:] or CURATED
    total = collections.Counter()
    for t in targets:
        stats, misses = validate_tarball(t, idx)
        tag = Path(t).name
        ok = stats["exact"] + stats["ambig_ok"]
        n = ok + stats["MISMATCH"]
        rate = f"{100*ok/n:.1f}%" if n else "n/a"
        print(f"{tag:48s} ok={ok:3d}/{n:3d} ({rate})  "
              f"exact={stats['exact']} ambig={stats['ambig_ok']} "
              f"MISMATCH={stats['MISMATCH']} skip={stats['unparseable']+stats['not_in_index']}")
        for m in misses:
            print(f"    MISMATCH {m[0]} truth={m[1]} idx=({m[2]},{m[3]}) arch={m[4]}")
        total.update(stats)
    ok = total["exact"] + total["ambig_ok"]
    n = ok + total["MISMATCH"]
    print(f"\nTOTAL ok={ok}/{n} ({100*ok/max(n,1):.2f}%)  "
          f"exact={total['exact']} ambig_ok={total['ambig_ok']} MISMATCH={total['MISMATCH']}")


if __name__ == "__main__":
    main()
