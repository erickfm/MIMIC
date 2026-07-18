"""Generalized master-rank perspective filter (supersedes the Fox-only
filter_masterfox.py for multi-character pipelines; the Fox script is kept
for the historical box scripts).

Scan .slp pools and keep only games where EVERY port playing the target
character is MASTER rank. Per-port rank is recoverable from the ranked
dataset's anonymizer, which writes it into netplay.name ("Master Player" /
"Diamond Player" / "Platinum Player"). Keeping a game iff all target-char
ports are master guarantees every perspective slp_to_shards extracts for
that character is master, and drops mixed-rank dittos whole.

Character is given as the HF bucket name. Note the Zelda/Sheik caveat: the
.slp header records the CSS pick, which is ZELDA for both SHEIK and ZELDA
buckets (Sheik is entered in-game) — both buckets therefore filter on
Zelda's header ID. The buckets themselves are already split by majority
in-game form, so this is correct for both.

Usage:
    python3 tools/filter_masterchar.py --char FALCO data/falco_ranked_slp \
        --out data/falco_master_keep.txt
"""
import argparse
import glob
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import melee
from peppi_py import read_slippi

# peppi external (CSS-order) id -> libmelee Character value. Same verified
# table as tools/shard_and_upload_ranked.py / rlvr/state/peppi_adapter.py.
PEPPI_TO_LIBMELEE = {
    0: 2, 1: 3, 2: 1, 3: 24, 4: 4, 5: 5, 6: 6, 7: 17, 8: 0, 9: 18,
    10: 16, 11: 8, 12: 9, 13: 12, 14: 10, 15: 15, 16: 13, 17: 14,
    18: 19, 19: 7, 20: 22, 21: 20, 22: 21, 23: 26, 24: 23, 25: 25,
}
LIBMELEE_TO_PEPPI = {v: k for k, v in PEPPI_TO_LIBMELEE.items()}

# HF bucket name -> libmelee Character enum NAME whose header ID to match.
# Identity for most; the exceptions are documented above / in CLAUDE.md.
BUCKET_TO_ENUM = {
    "JIGGLYPUFF": "JIGGLYPUFF",
    "ICE_CLIMBERS": "POPO",   # header records Popo for IC games
    "SHEIK": "ZELDA",         # header records the CSS pick (Zelda)
    "ZELDA": "ZELDA",
}


def bucket_ext_ids(bucket: str) -> frozenset:
    enum_name = BUCKET_TO_ENUM.get(bucket, bucket)
    lib_val = melee.Character[enum_name].value
    ids = {LIBMELEE_TO_PEPPI[lib_val]}
    # Zelda/Sheik hardening: headers normally record the CSS pick (Zelda,
    # ext 18), but shard_and_upload_ranked defensively bucketed BOTH Zelda
    # and Sheik values — match both so an ext-19 header can't silently drop
    # a game from either bucket.
    if bucket in ("SHEIK", "ZELDA"):
        ids.add(LIBMELEE_TO_PEPPI[melee.Character.SHEIK.value])
    return frozenset(ids)


def classify(ext_ids: frozenset, path: str):
    try:
        g = read_slippi(path, skip_frames=True)
        ranks = []
        for pl in g.start.players:
            if pl is None:
                continue
            if pl.character in ext_ids:
                nm = (pl.netplay.name if pl.netplay else "") or ""
                ranks.append(nm.replace(" Player", "").strip().lower())
        if ranks and all(r == "master" for r in ranks):
            return path
    except Exception:
        return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--char", required=True, help="HF bucket name, e.g. FALCO")
    ap.add_argument("dirs", nargs="+", help=".slp directories to scan")
    ap.add_argument("--out", required=True, help="keep-list output path")
    ap.add_argument("--workers", type=int, default=96)
    args = ap.parse_args()

    ext_ids = bucket_ext_ids(args.char)
    files = []
    for d in args.dirs:
        files += glob.glob(d.rstrip("/") + "/*.slp")
    keep = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(partial(classify, ext_ids), files, chunksize=32):
            if r:
                keep.append(r)
    with open(args.out, "w") as f:
        f.write("\n".join(keep) + ("\n" if keep else ""))
    print(f"char={args.char} ext_ids={sorted(ext_ids)} scanned={len(files)} "
          f"master_keep={len(keep)} "
          f"({100.0 * len(keep) / max(1, len(files)):.1f}% of pool)")


if __name__ == "__main__":
    main()
