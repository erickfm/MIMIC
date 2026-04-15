# Ranked dataset pipeline (`erickfm/melee-ranked-replays`)

How raw ranked .slp archives get sharded and uploaded to HuggingFace.

## Source

Six anonymized archives of Slippi ranked replays (platinum+), ~850k total
replays, at `/home/erick/Documents/melee/ranked-anonymized-N-*.{7z,zip}`.

Each archive has flat files of the form `{rank1}-{rank2}-{hash}.slp` where
each rank ∈ {platinum, diamond, master}. Rank pairs appear in the fixed
orderings `diamond-diamond`, `diamond-platinum`, `master-diamond`,
`master-master`, `master-platinum`, `platinum-platinum` — higher rank first
(M > D > P), same-rank pairs use the same token twice. There are 6 possible
combos total.

## Layout on HF

```
shards/
  {CHAR}_{rank_pair}_a{N}.tar.gz    # one per (character, rank_pair, archive)
  metadata_a{N}.json                # flat list, one entry per replay in archive N
```

Each tarball contains raw `.slp` files (no preprocessing), named by their
original hash-suffixed filename. Up to **25 characters × 6 rank pairs × 6
archives = 900 shards** in the fully populated case (rare char × rank combos
may be empty for some archives).

## Tool

`tools/shard_and_upload_ranked.py` processes one archive end-to-end.

```bash
python tools/shard_and_upload_ranked.py \
    --archive /home/erick/Documents/melee/ranked-anonymized-1-116248.7z \
    --archive-id 1 \
    --workdir /home/erick/Documents/melee/staging_a1
```

It sets a global `socket.setdefaulttimeout(600)` so wedged HF uploads bubble
up as exceptions and hit the retry loop instead of hanging the process
forever (a hang-in-the-middle-of-upload is otherwise silent with
`huggingface_hub`).

## Phases

### 1. Extract

`7z x` (.7z) or `unzip` (.zip) decompresses the whole archive to
`{workdir}/extracted/`. For archive 1 this was 67 GB → 441 GB of .slp files.
Needs enough free disk for the largest archive: ~600 GB uncompressed.

### 2. Per-file parse (parallel, ProcessPoolExecutor across all CPUs)

For each .slp file:

1. **Read header only** via `peppi_py.read_slippi(path, skip_frames=True)` —
   skipping frames makes it fast (ms per file) since we only need the Start
   event, not the ~10k frames of gameplay.
2. **Pull the 2 players** out of `game.start.players`, reject if not exactly 2.
3. **Map each player's character int to a name** via a lookup built from
   `melee.Character` enum, with two collapses:
   - ZELDA (19) and SHEIK (7) → `ZELDA_SHEIK` (they're the same fighter mid-match)
   - POPO (10) and NANA (11) → `ICE_CLIMBERS` (the two climbers are one unit)
4. **Reject junk characters**: WIREFRAME_MALE/FEMALE, GIGA_BOWSER, SANDBAG,
   UNKNOWN — these aren't legal tournament characters; replays featuring
   them are debug/test files.
5. **Parse rank from filename** via regex: the `{rank1}-{rank2}` prefix.

Per-file output: `(filename, p1_name, p2_name, rank_pair, error_or_None)`.

### 3. Bucketing

Each successful replay enters up to two buckets keyed by `(character, rank_pair)`:

- One for player 1's character
- One for player 2's character (**skipped if same char** — no double-counting dittos)

So a MARTH vs FALCO `diamond-platinum` replay lands in both
`MARTH_diamond-platinum` and `FALCO_diamond-platinum` buckets. A FOX ditto
only goes into `FOX_diamond-diamond` once.

This matches the HF layout (`shards/{CHAR}_{combo}_aN.tar.gz`) where every
bucket holds all the replays featuring that character at that rank combo —
you can download "all Marth games at master-master" by pulling one tarball.

### 4. Metadata

A flat list of `{filename, p1, p2, rank, archive}` entries — one row per
**replay** (not per bucket) — schema matching the existing `metadata_a3.json`
on HF exactly. Archive ID is a string (`"1"`, `"2"`, ...) for consistency
with the a3 baseline.

### 5. Tar + upload

For each bucket, in deterministic sorted order:

1. Check `api.list_repo_files()` snapshot — **skip if `shards/{name}` already exists**
   (cheap resume after a crash or a kill).
2. `tarfile.open(..., "w:gz", compresslevel=6)` to `{workdir}/tars/{name}`.
3. Upload via `api.upload_file(...)`, with 5-attempt exponential backoff on
   exceptions (`wait = min(300, 2^attempt * 10)`).
4. Delete the local tar.

After all bucket uploads, `metadata_aN.json` goes up and the extract dir
is removed. Workdir shrinks back to ~tens of MB (just logs + metadata).

## Resume semantics

- **Mid-upload crash:** the next run re-parses headers (fast, ~20s), then
  skips every bucket whose tar already exists on HF. Only the unfinished
  buckets get rebuilt. Requires `--skip-extract` if the extract dir is still
  present so it doesn't blow up trying to re-extract into a non-empty dir.
- **Stuck upload:** the socket timeout will eventually raise and the retry
  loop will reattempt. If retries are exhausted the process dies and you
  resume as above.
- **Already-uploaded archive:** running again is a no-op — all 144 (or so)
  buckets get the `skip (already uploaded)` path.

## Per-archive stats (as processed)

| Archive | Size (compressed) | Replays | Extracted | Buckets | Status |
|---------|-------------------|---------|-----------|---------|--------|
| a1 | 67 GB | 116,248 (116,149 good) | 441 GB | 144 | ✓ uploaded 2026-04-14 |
| a2 | 88 GB | 151,807 | ~620 GB | ? | in progress |
| a3 | 93 GB | 128,787 | ? | 150 | ✓ (pre-existing, different tool) |
| a4 | 108 GB | 148,358 | ? | ? | pending |
| a5 | 97 GB | 133,261 | ? | ? | pending |
| a6 | 126 GB | 171,694 | ? | ? | pending |
