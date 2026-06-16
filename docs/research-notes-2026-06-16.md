# Research notes — 2026-06-16

Trained rank-specific Fox behavior-cloning models (`fox-master`, `fox-diamond`,
`fox-platinum`) and verified a clean skill gradient head-to-head. This was the
first use of the freshly-corrected `erickfm/melee-ranked-replays` dataset for
something beyond reproducing the existing models.

## Per-player rank is in `netplay.name`

The dataset buckets by rank *pair* (`master-diamond`, …) but the filename
doesn't say which port is which rank. Turns out the anonymizer preserved each
player's rank in `start.players[*].netplay.name` = `"Master Player"` /
`"Diamond Player"` / `"Platinum Player"` (validated 200/200 vs the filename
rank tokens, zero mismatch). So we can use **mixed-rank pairs** but train only
the **target-rank Fox's perspective** — e.g. a master-diamond game feeds the
master model only through the master player (if Fox).

`tools/partition_fox_by_rank.py` routes each Fox `.slp` into
`data/fox_<rank>_slp/` by the Fox player's rank. Mixed-rank Fox dittos (~6.6%
of Fox games — not "rare") are excluded, since `slp_to_shards --character 1`
alone can't separate two same-character perspectives by port. Then the existing
sharder + trainer run unchanged (zero rank-awareness in the model code).

## Data + pipeline

Mixed pairs roughly doubled the data vs same-rank-only: master 102k / diamond
82k / platinum 154k Fox games (vs 48.6k / 23.5k / 99.1k). That's far more than
fits on disk once sharded (~4× expansion → ~5 TB) *and* more than training uses
(capped at `--max-samples 16777216` windows), so each rank was **subsampled to
25k games** (still > production Fox's ~17k; diamond still beats its 23.5k
same-rank floor). Shared metadata (norm/clusters/combos) built once and reused
so the three models are directly comparable. `tools/run_rank_fox.sh` does
download→partition→shard→train; `tools/finish_rank_fox.sh` auto-uploads each to
`erickfm/MIMIC/fox-<rank>/` (additive) and runs the benchmark.

Val losses ended close (master ~0.733, diamond 0.733, platinum 0.748) — val
measures fit to each rank's *own* data, not cross-rank skill, so the head-to-
head is the real test.

## Result — monotonic skill gradient

Rank-ladder head-to-head, 15 matches/matchup, realtime, alternate ports,
FD, Fox ditto (`tools/bench_rank_fox.sh`, reports in `reports/rankfox_*.json`):

| matchup | winner | win rate | avg stocks |
|---|---|---|---|
| master vs platinum | master | 67% (10–5) | 1.27 vs 0.47 |
| master vs diamond  | master | 67% (10–5) | 1.20 vs 0.53 |
| diamond vs platinum| diamond| 60% (9–6)  | 1.33 vs 0.80 |

**master > diamond > platinum**, transitive, win-rate and stock margin agree.
Rank-conditioned BC reproduces the real Slippi skill ladder: higher training
rank → stronger bot. (A single match-1 platinum 2–0 was pure variance.)

## Gotchas hit

- Output dir `data/fox_master_v2` collided with a stale pre-existing 152 GB
  dir (legacy 22-col schema, rejected by the current encoder) → used
  `data/foxrank_<rank>_v2`.
- `META="$(build_shared_meta)"` captured the function's log output, not a path
  → metadata copy failed. Fixed to a fixed `SHARED_META` path.
- Forgot `cat_maps.json` in the metadata copy (a `_load_prereqs` requirement).
- `play.py --character` takes the enum NAME (`FOX`), not the index `1` that
  `slp_to_shards` uses.

## Open

- Discord wiring (`!fox-master` etc.) left for explicit go — changes live bot
  behavior, not purely additive.
- `data/foxrank_<rank>_v2` shards (~1.1 TB) can be freed; keep if retraining
  (training logged `--cosine-decay-steps 32373/30738` for a tuned next run).
