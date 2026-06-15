# Research notes — 2026-06-15

Fixed the `erickfm/melee-ranked-replays` dataset **in place** on HuggingFace:
every character folder now contains only that character's games, and the
combined `ZELDA_SHEIK` bucket is split into clean `SHEIK` + `ZELDA` folders.
This note records the bug, the multi-day investigation that surfaced it, the
fix, and the obstacles — because the same class of bug (parser ID conventions)
keeps biting and the fix recipe is reusable.

## The bug — peppi external IDs vs libmelee enum values

`tools/shard_and_upload_ranked.py` builds character buckets by reading each
replay's character with `peppi_py` and naming it via
`CHAR_NAME = {c.value: c.name for c in melee.Character}`. But **peppi returns
EXTERNAL (CSS-order) character IDs** (Falcon=0, Fox=2, Samus=16, Pikachu=13)
while **libmelee's `Character` enum uses different VALUES** (Falcon=2, Fox=1,
Samus=13, Pikachu=12). The code indexed the libmelee-value table with peppi's
external id, so every bucket was misnamed:

| real character | peppi ext | libmelee val | filed under |
|---|---|---|---|
| Pikachu | 13 | 12 | **SAMUS** (CHAR_NAME[13]) |
| Samus | 16 | 13 | **MEWTWO** (CHAR_NAME[16]) |
| Fox | 2 | 1 | **CPTFALCON** |
| DK | 1 | 3 | **FOX** |
| Roy | 23 | 26 | **PICHU** |
| Zelda | 18 | 19 | **MARTH** |

It's a fixed permutation. The 2026-04-21 note first flagged this against the
FOX bucket; it sat unfixed because training is self-correcting —
`slp_to_shards.py --character <id>` re-filters with libmelee and keeps only the
true character, so the BC models trained on real data despite the misnamed
folders (verified: re-sharding a broken SAMUS tarball of 1,183 files kept
exactly the 51 real Samus). The cost was silent: ~5× wasted download per
character, and the public folders were wrong for anyone else.

## Not uniform — batch a3 was clean

Critically, the scramble is **not in every archive**. The dataset uploads in
6 archive batches (a1–a6) × 6 rank pairs. Batch **a3 was uploaded with correct
code**; a1/a2/a4/a5/a6 are scrambled. Verified on FOX + SAMUS (libmelee on all
6 batches: a3 = 100% the named character, others ~96% wrong) and then on a
554-replay stratified sample across all batches / all 6 rank tiers / 9
characters. The "SAMUS" master-master a1 tarball was 95% Pikachu; a3 was 100%
Samus.

## The translation layer — no 800 GB download

Rather than re-download the whole dataset, recovered the truth from the
`metadata/metadata_a{N}.json` sidecars (850,005 rows of
`{filename, p1, p2, rank, archive}`, ~100 MB). Those labels are scrambled too,
but **reversibly** (same fixed permutation): trust a3 as-is, invert the bug on
the rest. `tools/build_ranked_index.py` produced
`data/ranked_index/index.jsonl`; `tools/validate_ranked_index.py` checked it
against libmelee at **554/554** across every batch/tier. The only ambiguity is
the two collapsed labels (`ZELDA_SHEIK`, `ICE_CLIMBERS`), whose reversal has
two preimages; those were resolved by direct libmelee parse
(`tools/resolve_ambiguous.py`).

The payoff: the 2026-06-12 models had used only ~16–20% of the available
master-tier data per character (e.g. Samus 4,035 of 25,227 games).

## Zelda/Sheik — split by majority frame, not CSS pick

Zelda and Sheik share one character slot in Melee (same CSS pick; transform
via down-B mid-match). The dataset lumps them in `ZELDA_SHEIK`. We split into
separate `SHEIK` and `ZELDA` folders, where **a game counts as Zelda only if
the player is in Zelda form for the majority of frames** — otherwise a Zelda
picker who transforms to Sheik and plays Sheik all game pollutes the Zelda
folder (`tools/classify_zelda_sheik.py`, `tools/classify_marth_zelda.py`).

Data reality: pure Zelda is vanishingly rare (master-master: 0.3% of the slot;
rises to ~3% at platinum). Total real-Zelda games (majority frames, all ranks):
**3,436** — 649 in the `ZELDA_SHEIK` bucket + **2,787 hiding in the scrambled
`MARTH` folder** (real Zelda → buggy "MARTH"). That's a shippable Zelda bot,
but diamond/platinum-skewed; master Zelda essentially doesn't exist. Sheik is
99.7% of the slot and abundant.

## The fix — mostly free server-side renames

Because each scrambled tarball is *wholly* one character (just wrong-named),
most of the fix is `CommitOperationCopy` (server-side, no transfer):

- **772 clean renames** → `_fixed/{true}/...` via the validated permutation.
- **Re-tar only the 3 entangled buckets** (`ZELDA_SHEIK` 157 GB, `ICE_CLIMBERS`
  22 GB, buggy `MARTH` 3 GB): download → re-sort `.slp` by true character →
  re-upload. SHEIK/ZELDA/LUIGI/MEWTWO/NESS are born here.
- Verified `_fixed/` (all 26 folders vs libmelee, zero dups), then an
  atomic-ish swap: promote `_fixed/*` to top level, delete old folders +
  scrambled metadata, delete `_fixed/`.

`tools/execute_dataset_fix.py` (phases `renames` / `retar` / `swap`),
`tools/verify_fixed.py`.

## Obstacles

- **Home uplink ~4 MB/s AND huggingface_hub hangs on dead CLOSE-WAIT sockets**
  (same hang seen on downloads earlier). Uploads stalled at 0 B/s indefinitely.
  Fixed with a `SIGALRM` timeout+retry wrapper (`commit_robust`, 15-min cap)
  that ground the 140 GB SHEIK upload overnight, auto-recovering ~8 hangs.
  `HF_HUB_ENABLE_HF_TRANSFER=1` made it **worse** — hung outright. Don't use it
  on this connection.
- **My routing double-counted cross-bucket games.** A Sheik-vs-Zelda game
  physically sits in both the `ZELDA_SHEIK` bucket (Sheik) and the `MARTH`
  bucket (Zelda); `route_game` routed every entangled player, so the Zelda
  perspective landed in `ZELDA` twice (once as `_a{N}`, once as `_m{N}`).
  Verification caught it (ZELDA had 63 tarballs, should be 36; `a1` games were
  a strict subset of `m1`). Fixed by deleting the redundant a/m-suffix
  duplicates (55 tarballs). Lesson: each game-perspective must be routed from
  exactly the bucket whose label matches that player.

## Root cause fixed

`tools/shard_and_upload_ranked.py:44` now has the verified
`PEPPI_TO_LIBMELEE` remap and `_parse_one` maps external → libmelee value
before naming. Future rebuilds won't re-scramble. (The script still collapses
Zelda+Sheik → `ZELDA_SHEIK` at header-parse time; splitting them needs frame
data, so it stays a documented post-step via the `classify_*` tools.)

## Result + follow-ups

All 26 folders content-verified against libmelee on the **live** dataset
(SAMUS=Samus, FOX=Fox, SHEIK=Sheik, ZELDA=Zelda), zero duplicates. ROY
recovered 6→36 tarballs. HF git history retains the pre-fix state.

Open: (1) regenerate clean `metadata/` sidecars from the corrected folders —
deleted the scrambled ones, not yet rebuilt. (2) retrain the underfed models
(Samus/Doc/Pikachu/Yoshi) + a now-viable Zelda bot on the corrected data.
