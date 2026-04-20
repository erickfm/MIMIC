# Research Notes — 2026-04-20 b (docs cleanup snapshot)

Snapshots of point-in-time state that got dropped from `CLAUDE.md` during
the 2026-04-20 docs cleanup. The cleanup reworked `CLAUDE.md` into a
timeless quick reference (no changelog-style sections, no dated
headers, no per-note docs index), folded a couple of bespoke docs in,
and moved one-off debug artifacts into `docs/archive/`. The items below
are the snapshots that didn't have a natural home anywhere else, so
they land here so nothing is lost.

## Un-promoted per-character pipeline runs

Wandb runs from the 2026-04-17/18 per-character pipeline cycle, trained
with `--mimic-minimal-features` (pre-fullfeat). Val losses below are
the wandb best; underlying `_best.pt` / `_bestloss.pt` files are still
present on the training box but were never renamed to the
`{char}-{YYYYMMDD}-{descriptor}-{steps}k.pt` convention or uploaded to
HF.

| char       | run id      | val     | notes         |
|------------|-------------|---------|---------------|
| fox        | `qeka6rq8`  | ~0.7081 | pre-fullfeat  |
| falco      | `zb1vhjxs`  | ~0.7448 | pre-fullfeat  |
| sheik      | `jc4xe4dv`  | ~0.6611 | pre-fullfeat  |
| cptfalcon  | `6k1x8xdi`  | ~0.7356 | pre-fullfeat  |
| marth      | `eo8yjem4`  | ~0.6746 | pre-fullfeat  |

All relpos / `--model mimic`. Expect ~1–3% val improvement when
re-run without `--mimic-minimal-features` (see
`research-notes-2026-04-19.md` for the fullfeat measurements).

## Superseded Fox checkpoints still on disk

These files still exist under `checkpoints/` but should not be used
for inference. They're kept for audit only:

- `fox-20260415-rope-25k.pt` — old Fox best (RoPE, val 0.7358). RoPE
  variants are deprecated (see `research-notes-2026-04-14.md`).
- `fox-20260413-rope-32k.pt` — Fox tainted by the seq_len=60 bug
  (see `research-notes-2026-04-15.md`).
- `fox-20260411-relpos-noself-28k.pt` — Fox trained without
  `--self-inputs`.

## Ranked dataset pipeline — per-archive status

Processing status of the six `ranked-anonymized-N-*.{7z,zip}`
archives via `tools/shard_and_upload_ranked.py`. Replay counts are
from archive extraction; buckets are (character × rank_pair)
groupings, up to 144–150 per archive:

| Archive | Size (compressed) | Replays                 | Extracted | Buckets | Status |
|---------|-------------------|-------------------------|-----------|---------|--------|
| a1      | 67 GB             | 116,248 (116,149 good)  | 441 GB    | 144     | uploaded 2026-04-14 |
| a2      | 88 GB             | 151,807                 | ~620 GB   | ?       | in progress |
| a3      | 93 GB             | 128,787                 | ?         | 150     | uploaded (pre-existing, different tool) |
| a4      | 108 GB            | 148,358                 | ?         | ?       | pending |
| a5      | 97 GB             | 133,261                 | ?         | ?       | pending |
| a6      | 126 GB            | 171,694                 | ?         | ?       | pending |

## What else changed in the 2026-04-20 cleanup

For the record, so someone reading this note later can reconstruct
the scope without digging through the diff:

- `CLAUDE.md` rewritten to strip changelog-style framing — no dated
  section headers (`Shard Alignment (Critical — 2026-04-11)` →
  `Shard alignment`), no "Fixed on DATE" prose, no historical result
  tables, no "Superseded / kept for audit only" bullets (now above),
  no trailing per-note docs index. Same operational content, stated
  in present tense.
- `GPUS.md` → folded into `CLAUDE.md § Environment`.
- `docs/ranked-dataset-pipeline.md` → folded into
  `CLAUDE.md § Data § Ranked dataset pipeline` (procedural content);
  the per-archive status table (above) moved here.
- `docs/gamecube_input_collapse.md` →
  `docs/archive/research-notes-2026-04-09-input-collapse.md`. The
  resolution rules themselves stay in `CLAUDE.md` pitfall #10.
- `docs/hal-mimic-diff-audit.md` →
  `docs/archive/research-notes-2026-04-02-hal-mimic-diff-audit.md`.
  Inconclusive debug artifact; superseded by the 2026-04-13 TRIG fix
  in `research-notes-2026-04-13.md`.
- `docs/results-2026-03-17.md` →
  `docs/archive/research-notes-2026-03-17-results.md`. Historical
  closed-loop eval, pre-v2, pre-TRIG-fix.

Nothing in `docs/research-notes-*.md` was modified — research notes
are frozen-history by convention, so stale references inside them
(e.g. to the now-archived `gamecube_input_collapse.md`) stay as-is.
