# `tools/` — index

Utilities around the MIMIC training/inference pipeline — the durable
infrastructure only. Spent single-use scripts are removed once their job is
done (recoverable from git history), not archived in-tree. Canonical
orientation is `../CLAUDE.md`; this file is just a map.

## Data pipeline (raw replays → training shards)
- `download_fox.py` — pull Fox `.slp` from the legacy `slippi-public-dataset-v3.7`.
- `shard_and_upload_ranked.py` — ranked `.slp` archive → HF tarballs, bucketed by
  (character, rank_pair). Holds the `PEPPI_TO_LIBMELEE` external-ID remap.
- `slp_to_shards.py` — `.slp` → v2 `.pt` shards with `target[i] = buttons[i+1]`.
- `split_by_character.py` — partition a mixed `.slp` dir by character.
- `reshard.py` — re-split shards to a target size (`--target-mb 800`).
- `build_norm_stats.py`, `build_mimic_norm.py`, `build_clusters.py`,
  `build_controller_combos.py` — generate shard-build metadata (`norm_stats.json`,
  `mimic_norm.json`, `stick_clusters.json`, `controller_combos.json`).

## Training
- `run_local_chars.sh` — local single-4090 all-character pipeline.
- `run_all_chars.sh` — remote 2×5090 all-character pipeline.
- `retrain_all_baseline.sh` — all-character baseline retrain (see 2026-04-20 note).
- (`train.py --warm-restart` lives at repo root: keep Adam, reset the LR schedule.)

## Inference
- `play.py` — bot vs CPU or bot vs bot, N back-to-back matches, optional win-rate JSON.
- `play_netplay.py` — Slippi Online Direct Connect (persistent session).
- `discord_bot.py` — Discord front-end (`!play`, `!<character>`, queue).
- `inference_utils.py` — **shared** decode pipeline; new entry points import from here.
- `run_hal_model.py` — reference reimpl of HAL's 5-class inference (loads Eric Gu's
  original HAL checkpoints only; not on any MIMIC path — do not delete).

## Promotion / upload
- `upload_char.py` — upload a `{char}` checkpoint + metadata to `erickfm/MIMIC/{char}/`.
- `upload_models_to_hf.py` — bulk package + push to HuggingFace.

## Diagnostics / eval / plots
- `validate_checkpoint.py` — per-head CE on a val set.
- `diagnose.py` — train-vs-inference tensor compare.
- `inspect_frame.py` — per-frame I/O dump.
- `extract_wavedashes.py` — wavedash-only windows for overfit checks.
- `plot_val_vs_winrate.py` — val loss vs h2h win rate scatter (master-Fox thread).

## Removed (recoverable from git history)
Two spent one-off clusters were deleted once their work shipped — pull them
from git history if a rebuild ever needs them:
- **Ranked-dataset fix toolchain** (2026-06-15 in-place correction; root cause
  now lives in `shard_and_upload_ranked.py`) — see
  `../docs/research-notes-2026-06-15.md`.
- **Rank-Fox experiment runners** (2026-06-16/18 — shipped) — see
  `../docs/research-notes-2026-06-16.md` / `-2026-06-18.md`.
