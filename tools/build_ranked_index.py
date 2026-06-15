#!/usr/bin/env python3
"""Build a TRUE-character index over erickfm/melee-ranked-replays.

Background: the dataset's per-character buckets are partially mislabeled. The
uploader (tools/shard_and_upload_ranked.py) built character names as
`CHAR_NAME[peppi.character]` where CHAR_NAME is keyed by libmelee enum *value*
but peppi returns *external* (CSS-order) character IDs. Most archive batches
(a1,a2,a4,a5,a6) were uploaded with this bug; batch a3 was uploaded with
correct code (verified empirically via libmelee on FOX + SAMUS, all 6 batches).

The per-replay sidecars `metadata/metadata_a{N}.json` record the (possibly
buggy) p1/p2 names. This script recovers the TRUE characters with no bulk
download:
  - archive == 3  -> names are already correct, trust as-is.
  - archive != 3  -> reverse the bug: the recorded name N was produced as
    CHAR_NAME[external_id]; invert to the external_id(s), then map external ->
    libmelee via the verified table (rlvr/state/peppi_adapter) -> true name.

Collapsed labels (ZELDA_SHEIK, ICE_CLIMBERS) can have two external-ID
preimages under the bug, so their reversal is ambiguous; those entries are
written with `ambiguous=True` and a candidate list instead of a single name.

Output: data/ranked_index/index.jsonl  (one row per replay) and a printed
per-character true-game-count report.
"""
import json
import collections
from pathlib import Path

from huggingface_hub import hf_hub_download
from melee import Character

REPO = "erickfm/melee-ranked-replays"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "ranked_index"

# Verified peppi external -> libmelee enum value (rlvr/state/peppi_adapter.py).
PEPPI_TO_LIBMELEE = {
    0: 2, 1: 3, 2: 1, 3: 24, 4: 4, 5: 5, 6: 6, 7: 17, 8: 0, 9: 18,
    10: 16, 11: 8, 12: 9, 13: 12, 14: 10, 15: 15, 16: 13, 17: 14,
    18: 19, 19: 7, 20: 22, 21: 20, 22: 21, 23: 26, 24: 23, 25: 25,
}

# The buggy uploader's name table: keyed by libmelee value, with collapses.
CHAR_NAME = {c.value: c.name for c in Character}
CHAR_NAME[Character.ZELDA.value] = "ZELDA_SHEIK"
CHAR_NAME[Character.SHEIK.value] = "ZELDA_SHEIK"
CHAR_NAME[Character.POPO.value] = "ICE_CLIMBERS"
CHAR_NAME[Character.NANA.value] = "ICE_CLIMBERS"

# Canonical true-name for a libmelee value (same collapses as the dataset uses).
NAME_BY_LIBVAL = dict(CHAR_NAME)

# Build the reverse-bug table: buggy_name -> set of true names.
# The bug computed name = CHAR_NAME[external_id]; so for each external_id, the
# buggy label is CHAR_NAME[external_id] and the true char is
# NAME_BY_LIBVAL[PEPPI_TO_LIBMELEE[external_id]].
REVERSE = collections.defaultdict(set)
for ext_id, lib_val in PEPPI_TO_LIBMELEE.items():
    buggy_label = CHAR_NAME.get(ext_id)
    true_name = NAME_BY_LIBVAL.get(lib_val)
    if buggy_label is not None and true_name is not None:
        REVERSE[buggy_label].add(true_name)


def true_names(name: str, archive: str):
    """Return (resolved_name_or_None, candidates_list, ambiguous_bool)."""
    if str(archive) == "3":
        return name, [name], False
    cands = sorted(REVERSE.get(name, []))
    if len(cands) == 1:
        return cands[0], cands, False
    if len(cands) == 0:
        return None, [], False  # unknown label
    return None, cands, True


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for a in range(1, 7):
        p = hf_hub_download(REPO, f"metadata/metadata_a{a}.json", repo_type="dataset")
        rows.extend(json.load(open(p)))
    print(f"loaded {len(rows)} metadata rows")

    out_path = OUT_DIR / "index.jsonl"
    per_char = collections.Counter()      # true games per character (unambiguous)
    ambiguous_rows = 0
    unknown_rows = 0
    with open(out_path, "w") as fh:
        for e in rows:
            arch = e.get("archive")
            rec = {
                "filename": e["filename"],
                "rank": e.get("rank"),
                "archive": arch,
                "meta_p1": e["p1"],
                "meta_p2": e["p2"],
            }
            r1, c1, amb1 = true_names(e["p1"], arch)
            r2, c2, amb2 = true_names(e["p2"], arch)
            rec["true_p1"] = r1
            rec["true_p2"] = r2
            rec["cand_p1"] = c1
            rec["cand_p2"] = c2
            rec["ambiguous"] = amb1 or amb2
            if amb1 or amb2:
                ambiguous_rows += 1
            for r in (r1, r2):
                if r is None:
                    unknown_rows += 1
                else:
                    per_char[r] += 1
            fh.write(json.dumps(rec) + "\n")

    print(f"\nwrote {out_path}")
    print(f"ambiguous rows (collapsed-label reversal): {ambiguous_rows} "
          f"({100*ambiguous_rows/len(rows):.1f}%)")
    print(f"unresolved character slots: {unknown_rows}")
    print("\n=== TRUE games per character (counts a character slot per game) ===")
    for name, n in per_char.most_common():
        print(f"  {name:14s} {n:7d}")


if __name__ == "__main__":
    main()
