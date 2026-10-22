# `tools/` — index

Utilities around the MIMIC training/inference pipeline — the durable
infrastructure only. Spent single-use scripts are removed once their job is
done (recoverable from git history), not archived in-tree. Canonical
orientation is `../CLAUDE.md`; this file is just a map.

## Data pipeline (raw replays → training shards)
- `shard_and_upload_ranked.py` — ranked `.slp` archive → HF tarballs, bucketed by
  (character, rank_pair). Holds the `PEPPI_TO_LIBMELEE` external-ID remap.
- `slp_to_shards.py` — `.slp` → v2 `.pt` shards with `target[i] = buttons[i+1]`.
- `filter_masterfox.py` — keep only perspectives where the target char is
  master-rank (rank read from `netplay.name`; hardcodes Fox, generalizes).
- `reshard.py` — re-split shards to a target size (`--target-mb 800`).
- `build_norm_stats.py`, `build_mimic_norm.py` — generate shard-build metadata
  (`norm_stats.json`, `mimic_norm.json`); `stick_clusters.json` /
  `controller_combos.json` come from `tools/meta/`.

## Training
- `run_local_chars.sh` — local single-4090 all-character pipeline.
- `run_pod_chars.sh` — remote-box all-character pipeline (stages, never pushes).
- `fast_proxy.sh` / `proxy_run.sh` — ~5-min and ~45-min local research proxies.
- `build_masterfox_train_box.sh` / `build_full_masterfox_box.sh` — remote-box
  filter→reshard→train pipelines for the rank-filtered master-Fox set.
- `average_checkpoints.py` — checkpoint averaging (tail-SWA is part of the
  production recipe; also used for WiSE-FT-style interpolation).
- (`train.py --warm-restart` lives at repo root: keep Adam, reset the LR schedule.)

## Inference
- `play.py` — bot vs CPU or bot vs bot, N back-to-back matches, optional win-rate JSON.
- `play_netplay.py` — Slippi Online Direct Connect (persistent session).
- `discord_bot.py` — Discord front-end (`!play`, `!<character>`, queue).
- `inference_utils.py` — **shared** decode pipeline; new entry points import from here.
- `ffw_batch_mp.py` — multiprocess FFW rollout harness (one env per process,
  central batched forward) for Dolphin A/B evals.
- `h2h_batch_mp.py` — batched cross-character Dolphin h2h (both pads, one central process).
- `sim_h2h.py` / `sim_rl.py` / `sim_play.py` — head-to-head, RL, and play in the
  native sim (`rlvr/sim/`).
- `run_hal_model.py` — reference reimpl of HAL's 5-class inference (loads Eric Gu's
  original HAL checkpoints only; not on any MIMIC path — do not delete).

## Emulator builds
- `build_savestate_dolphin.sh` — build the patched savestate fork
  (`erickfm/slippi-Ishiiruka@mimic-savestates`, pinned in
  `patches/slippi-ishiiruka-commit.txt`) into `emulator_ss/` (netplay variant,
  SAVESTATE/LOADSTATE pipe verbs + dual-pad FFW fix) and `emulator_pb/`
  (playback variant, replay-seeded savestate harvest). `emulator_ss/` is the
  dual-pad FFW build for Dolphin h2h; the netplay bot uses the stock Slippi
  build in `emulator/`.
- `patches/` — the savestate patch + pinned fork commit (backup for the fork).

## Promotion / upload
- `upload_char.py` — upload a `{char}` checkpoint + metadata to `erickfm/MIMIC/{char}/`.

## Diagnostics / eval / plots
- `lcancel_analysis.py`, `ctrl_canary.py`, `state_canary.py` —
  inference-faithfulness canaries (L-cancel rate / stick distribution /
  action-state distribution vs the training corpus).
- `validate_checkpoint.py` — per-head CE on a val set (legacy: assumes
  5-class buttons + reaction-delay 1; only valid for old checkpoints).
- `inspect_frame.py` — per-frame I/O dump (legacy: applies the pre-v2
  controller offset unconditionally).
- `extract_wavedashes.py` — wavedash-only windows for overfit checks.
- `plot_val_vs_winrate.py` — val loss vs h2h win rate scatter (master-Fox thread).

## Removed (recoverable from git history)
Spent one-off clusters deleted once their work shipped — pull them from git
history if a rebuild ever needs them:
- **Ranked-dataset fix toolchain** (2026-06-15 in-place correction; root cause
  now lives in `shard_and_upload_ranked.py`) — see
  `../docs/research-notes-2026-06-15.md`.
- **Rank-Fox experiment runners** (2026-06-16/18 — shipped) — see
  `../docs/research-notes-2026-06-16.md` / `-2026-06-18.md`.
- **June/July per-run box scripts + probes** (masterfox/allmaster/width/warmstart
  builds, reshard one-offs, rank probes, L-cancel probes, `ffw_batch_bench.py`,
  `diagnose.py`, `split_by_character.py`, `upload_models_to_hf.py` — superseded
  by `upload_char.py`) — removed 2026-07-17.
- **2026-10-22 legacy cleanup** — the Dolphin RL loop (`rlvr/online`) and offline
  RLVR (`rlvr/train`, `rollout`, `eval`, `sampler`, `tagger`, `tasks`);
  `download_fox.py`, `build_clusters.py`,
  `build_controller_combos.py`, `run_all_chars.sh`, `retrain_all_baseline.sh`,
  `gen_h2h_schedule.py`, `plot_h2h.py`, `plot_winrate_steps.py`
  (see `../docs/research-notes-2026-10-22.md`).
