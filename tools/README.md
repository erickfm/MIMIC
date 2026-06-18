# `tools/` — index

Utilities around the MIMIC training/inference pipeline. Grouped by purpose.
Canonical orientation is `../CLAUDE.md`; this file is just a map of what's here.

## Data pipeline (raw replays → training shards)
- `download_fox.py` — pull Fox `.slp` from the legacy `slippi-public-dataset-v3.7`.
- `shard_and_upload_ranked.py` — ranked `.slp` archive → HF tarballs, bucketed by
  (character, rank_pair). Holds the `PEPPI_TO_LIBMELEE` external-ID remap.
- `slp_to_shards.py` — `.slp` → v2 `.pt` shards with `target[i] = buttons[i+1]`.
- `slp_to_ranked_shards.py` — ranked-source variant.
- `split_by_character.py` — partition a mixed `.slp` dir by character.
- `partition_fox_by_rank.py` — symlink Fox `.slp` into per-rank dirs by the Fox
  player's `netplay.name` (master/diamond/platinum).
- `reshard.py` — re-split shards to a target size (`--target-mb 800`).
- `build_norm_stats.py`, `build_mimic_norm.py`, `build_clusters.py`,
  `build_controller_combos.py` — generate the metadata a shard build needs
  (`norm_stats.json`, `mimic_norm.json`, `stick_clusters.json`,
  `controller_combos.json`).

## Ranked-dataset correction toolchain (2026-06-15 in-place fix; reusable)
See `../docs/research-notes-2026-06-15.md`. Reverse the peppi-ID scramble,
validate, split collapsed labels, execute the server-side fix, verify.
- `build_ranked_index.py` · `validate_ranked_index.py` · `resolve_ambiguous.py`
- `classify_zelda_sheik.py` · `classify_marth_zelda.py`
- `execute_dataset_fix.py` (phases `renames`/`retar`/`swap`) · `verify_fixed.py`

## Training runners
- `run_local_chars.sh` — local single-4090 all-character pipeline.
- `run_all_chars.sh` — remote 2×5090 all-character pipeline.
- `run_rank_fox.sh` + `finish_rank_fox.sh` + `bench_rank_fox.sh` — the
  master/diamond/platinum Fox rank ladder (train → upload → h2h).
- `run_long_fox.sh` — crash-resilient ~480k-step master-only long run (auto-resume).
- `chain_warmrestart.sh` — warm-restart a run from its best checkpoint (keep Adam,
  reset LR); see `train.py --warm-restart`.
- `run_allranks_fox.sh` — all-ranks-pooled vs master experiment.
- `retrain_all_baseline.sh` — older all-character baseline retrain (see 2026-04-20 note).

## Inference
- `play.py` — bot vs CPU or bot vs bot, N back-to-back matches, optional win-rate JSON.
- `play_netplay.py` — Slippi Online Direct Connect (persistent session).
- `discord_bot.py` — Discord front-end (`!play`, `!<character>`, queue).
- `inference_utils.py` — **shared** decode pipeline; new entry points import from here.
- `run_hal_model.py` — reference reimpl of HAL's 5-class inference (loads Eric Gu's
  original HAL checkpoints only; not on any MIMIC path — do not delete).

## Promotion / upload
- `upload_char.py` — upload a `{char}` checkpoint + metadata to `erickfm/MIMIC/{char}/`.
- `upload_models_to_hf.py` — bulk model upload helper.

## Diagnostics / eval / plots
- `validate_checkpoint.py` — per-head CE on a val set.
- `validate_pipeline.py` — end-to-end pipeline sanity.
- `diagnose.py` — train-vs-inference tensor compare.
- `inspect_frame.py` · `inference_diag.py` — per-frame I/O dumps.
- `extract_wavedashes.py` — wavedash-only windows for overfit checks.
- `gameplay_health.py` — gameplay health metrics.
- `plot_val_vs_winrate.py` — val loss vs h2h win rate scatter (master-Fox thread).

## `legacy/`
Spent, single-use orchestration scripts from past sessions (character-specific
finish/continue/queue runners). Kept for history; nothing in the active path
references them.
