# CLAUDE.md — Agent Orientation for MIMIC

## What This Project Is

MIMIC is a behavior-cloning bot for Super Smash Bros. Melee. It watches human
replays and learns to predict controller inputs from game state. At inference
it drives a controller through Dolphin (the GameCube emulator) via libmelee.

The reference implementation is **HAL** (by Eric Gu), a separate project at
`/home/erick/projects/hal`. HAL's architecture and training pipeline are the
target we're reproducing. HAL can 4-stock a level 9 CPU. MIMIC is our
reimplementation that trains on the same data and aims to match or exceed
HAL's gameplay quality.

## Critical: Naming Confusion

**Read this first. This will save you hours.**

- **"hal-*" prefixed checkpoints and run names are MIMIC models**, not HAL.
  They use MIMIC's codebase (`train.py`, `mimic/model.py`) with HAL's
  architecture config (`--model hal`). The only actual HAL checkpoint is
  `/home/erick/projects/hal/checkpoints/000005242880.pt`.

- **`tools/run_hal_model.py` is NOT HAL's code.** It's our reimplementation
  of HAL's inference pipeline. HAL's actual inference is `hal/eval/play.py`
  in the HAL repo. These two scripts produce different results (ours had
  bugs that were fixed on 2026-04-07).

- **`data/fox_hal_norm`, `data/fox_hal_local`, etc. are MIMIC-format shards**
  built with HAL's normalization. They are NOT HAL's MDS data.

- **Two `stats.json` files exist in the HAL repo** with completely different
  values. See "Stats Files" section below. Using the wrong one silently
  breaks inference.

## Architecture (HAL-Matching Config)

When using `--model hal`, MIMIC matches HAL's GPTv5Controller:

- **Params:** ~19.95M (not 26.3M — earlier research notes were wrong about this)
- **Transformer:** d_model=512, 6 layers, 8 heads, block_size=1024
- **Position encoding:** Relative position (Shaw et al.) with skew matrix
- **Input:** Linear(164, 512) from concatenated [stage_emb(4) + 2*char_emb(12) + 2*action_emb(32) + gamestate(18) + controller(54)]
- **Output heads (autoregressive with detach):** shoulder(3) -> c_stick(9) -> main_stick(37) -> buttons(5)
- **Head hidden dim:** `input_dim // 2` (NOT a fixed 256 — each head has different hidden size)
- **Sequence length:** 256 frames (~4.3 seconds)
- **Dropout:** 0.2

The gamestate is 9 features per player (ego + opponent = 18):
`percent, stock, facing, invulnerable, jumps_left, on_ground, shield_strength, position_x, position_y`

The controller is a 54-dim one-hot: main_stick(37) + c_stick(9) + buttons(5) + shoulder(3).

## Stats Files (Critical)

Two `stats.json` files exist in the HAL repo:

| File | Source | p1_percent max | Frames |
|------|--------|---------------|--------|
| `hal/data/stats.json` | Full multi-character dataset | 362 | 222M |
| `hal/checkpoints/stats.json` | Fox training subset | 236 | 27M |

**Which one to use:** `hal/checkpoints/stats.json`. Despite HAL's `play.py`
code appearing to override to `hal/data/stats.json`, the Preprocessor actually
loads from `checkpoints/stats.json` (verified: `pp.stats["p1_percent"].max == 236.0`).
The override mechanism (`override_stats_path`) changes the config field but the
Preprocessor resolves to the checkpoint stats anyway.

**Impact of using the wrong file:** Every normalized feature value shifts. For
example, percent=50 normalizes to -0.576 (correct) vs -0.724 (wrong). This
makes the model see garbage inputs and play terribly. This bug was found and
fixed on 2026-04-07 after multiple rounds of debugging.

## Shard Alignment (Critical — 2026-04-11)

melee-py's `console.step()` returns **post-frame** game state (action,
position, percent — after engine processes inputs) alongside **pre-frame**
controller inputs (the buttons themselves). This means the game state at
frame `i` already reflects button[i] — e.g., action=KNEE_BEND appears on the
same frame as button=JUMP.

**v2 shards** (`data/fox_hal_v2`) fix this by shifting targets forward by 1
frame: `target[i] = buttons[i+1]`. The model sees the current game state and
predicts what to press NEXT. This matches inference exactly.

**Do NOT use `--controller-offset` or `--reaction-delay 1` with v2 shards.**
The alignment is already correct. Adding offsets would double-shift the data.

**Old shards** (`data/fox_hal_full`, `data/fox_hal_match_shards`) have the
leak. With those shards, use `--reaction-delay 1` to achieve the same effect
at dataloader time (this is what HAL does).

## Training

### Command (current best config — v2 shards)

```bash
python3 train.py \
  --model hal --encoder hal_flat \
  --hal-mode --hal-minimal-features --hal-controller-encoding \
  --stick-clusters hal37 --plain-ce \
  --lr 3e-4 --batch-size 512 \
  --max-samples 16777216 \
  --data-dir data/fox_hal_v2 \
  --reaction-delay 0 \
  --run-name <name> \
  --no-warmup --cosine-min-lr 1e-6
```

### Command (old shards, HAL-compatible)

```bash
torchrun --nproc_per_node=8 train.py \
  --model hal --encoder hal_flat \
  --hal-mode --hal-minimal-features --hal-controller-encoding \
  --stick-clusters hal37 --plain-ce \
  --lr 3e-4 --batch-size 64 \
  --max-samples 16777216 \
  --data-dir data/fox_hal_full \
  --controller-offset --self-inputs \
  --reaction-delay 1 \
  --run-name <name> \
  --nccl-timeout 7200 --no-warmup --cosine-min-lr 1e-6
```

For single-GPU training, add `--grad-accum-steps 8` to match the effective
batch size of 512. Remove `torchrun --nproc_per_node=8`.

**Throughput notes (RTX 4090):** BF16 AMP and torch.compile are enabled by
default. BF16 AMP is enabled by default with FP32 attention upcast in the
relpos path (prevents BF16 overflow in manual attention score computation).
For strict HAL reproduction, use `--model hal --no-amp --no-compile` instead.

**BF16 + relpos stability:** The Shaw relpos attention computes Q@K^T + Srel
manually. In BF16 this overflows due to limited mantissa precision (7 bits).
The fix is an `autocast(enabled=False)` block around the attention math in
`CausalSelfAttentionRelPos.forward()`, keeping Q/K/Er in FP32 for the dot
products while the rest of the model (FFN, embeddings, heads) stays in BF16.
Do NOT use GradScaler with BF16 — it's only needed for FP16.

### max_steps Bug (Fixed 2026-04-07)

`train.py` previously computed `max_steps = max_samples // BATCH_SIZE` using
local batch size (64) instead of effective batch size (64 * n_gpus * grad_accum).
With 8 GPUs this meant 262K steps / 134M samples instead of 32K steps / 16.7M
samples — 8x too much training. This caused severe overfitting (val loss 1.57).
Now fixed to divide by effective batch size.

### Training Results (2026-04-07)

| Run | Data | Best Val Loss | Overfitting |
|-----|------|--------------|-------------|
| HAL original | 2,830 games | 1.03 (on our val) | Mild |
| MIMIC 3.2K games | 3,229 games | 1.05 | +13% |
| MIMIC 12K games | 12,153 games | 0.977 | None (+1.6%) |

More data eliminates overfitting. The full HF dataset has 46K Fox games;
only 12K were used due to disk constraints.

### Training Results (2026-04-08)

| Run | Data | Filtering | Best Val Loss | vs HAL gameplay |
|-----|------|-----------|--------------|-----------------|
| HAL original | 2,830 games | Yes (HAL filters) | 1.089 (on filtered val) | — |
| hal-fixed-pipeline | 7,600 games | No | 1.038 | HAL 1-0 (close) |
| hal-filtered | 6,898 games | Yes (HAL filters) | 1.054 | **MIMIC 3-0** |

Data filtering (min 1500 frames, damage check, completion check) was more
important than val loss for gameplay quality. The unfiltered model had lower
val loss but worse gameplay because the val set contained the same junk.

### Do NOT run inference while training on the same GPU

GPU contention causes frame drops in Dolphin, making the model appear
unresponsive and miss inputs. Always suspend or kill training before
running inference. Verified 2026-04-07: same inference code went from
"less responsive, can't combo" to working correctly after freeing the GPU.

## Inference

### Running HAL's Original Checkpoint

```bash
python3 tools/run_hal_model.py \
  --checkpoint /home/erick/projects/hal/checkpoints/000005242880.pt \
  --dolphin-path /home/erick/projects/hal/emulator/squashfs-root/usr/bin/dolphin-emu \
  --iso-path "/home/erick/Downloads/Super Smash Bros. Melee (USA) (En,Ja) (Rev 2).iso" \
  --character FOX --cpu-character FOX --cpu-level 9 --stage FINAL_DESTINATION
```

### Running MIMIC Checkpoints

Use `tools/run_mimic_via_hal_loop.py` (not `run_hal_model.py`) for MIMIC
checkpoints, since they use MIMIC's model class (`mimic/model.py`) not the
minimal HAL reimplementation.

### Inference Bug History (2026-04-07)

`tools/run_hal_model.py` is a from-scratch reimplementation of HAL's inference.
It had 4 bugs that were found and fixed through systematic comparison against
HAL's actual `play.py`:

1. **Wrong stats file** — Was using `hal/data/stats.json` (max=362) instead of
   `checkpoints/stats.json` (max=236). Fixed to use checkpoint stats.

2. **Context window mock values** — Was filling with raw `torch.ones`. HAL
   fills with `torch.ones` then preprocesses them (normalizing values, encoding
   controller as proper one-hot). Mock sticks at (1.0, 1.0) map to cluster 23
   (up-right), not cluster 0 (neutral). Fixed to compute preprocessed mock.

3. **Player ordering** — Was using `list(gs.players.items())` (dict order).
   HAL uses `sorted(gs.players.items())` (port order). Fixed.

4. **Button release scope** — Was only releasing 4 buttons (A, B, X, Z). HAL
   releases all 7 (A, B, X, Y, Z, L, R) every frame. Without this, buttons
   can get stuck. Fixed.

### Verification

`tools/verify_hal_pipeline.py` compares our preprocessing output against HAL's
actual Preprocessor on the same input. All values match to <1e-5. Run this
after any changes to `run_hal_model.py` to catch regressions.

### HAL's Own Inference (Ground Truth)

If our reimplementation breaks, HAL's own code always works:

```bash
cd /home/erick/projects/hal
python3 -m hal.eval.play --artifact_dir checkpoints --character FOX
```

This requires `hal/local_paths.py` to have correct paths for the emulator and
ISO. The `MAC_*` path aliases were added for this purpose.

## Data Directories

| Directory | Contents | Target alignment | Status |
|-----------|----------|-----------------|--------|
| `data/fox_hal_v2` | ~17K Fox games, HAL-normalized, 800MB shards, quality-filtered, **next-frame targets** | target[i] = buttons[i+1] (clean) | **Active — use with rd=0, no offset** |
| `data/fox_hal_full` | ~10K Fox games, HAL-normalized, 800MB shards, quality-filtered | target[i] = buttons[i] (leaked) | Legacy — use with rd=1 |
| `data/fox_hal_800m` | 7,600 Fox games, HAL-normalized, 800MB shards | target[i] = buttons[i] (leaked) | Legacy |
| `data/fox_hal_local` | 7,600 Fox games, HAL-normalized, 3.8GB shards | target[i] = buttons[i] (leaked) | Legacy |

Use 800MB shards with `mmap=True` in DataLoader for optimal throughput. The
`tools/reshard.py` script can split large shards: `python tools/reshard.py --src <dir> --dst <dir> --target-mb 800`.

To build new shards, use `tools/slp_to_shards.py`
with `--hal-norm` and a metadata dir containing 5-combo `controller_combos.json`.

**Game quality filters (added 2026-04-08, matching HAL):** `slp_to_shards.py`
now filters replays the same way HAL's `process_replays.py` does:
- Minimum 1,500 frames (~25 seconds) — rejects disconnects and junk
- Damage check — both players must take at least some damage
- Completion check — one player must lose all stocks (no ragequits)
Existing `fox_hal_local` shards were built WITHOUT these filters and contain
low-quality games. Rebuild shards from .slp source to get clean data.

**btns_single encoding (fixed 2026-04-08):** The `btns_single` field in shards
encodes multi-hot buttons as single-label using early-release logic (match HAL's
`convert_multi_hot_to_one_hot_early_release`). When buttons change but nothing
new is pressed (partial release), the label is NO_BUTTON (4). Previously MIMIC
kept the surviving held button — this was fixed in both `slp_to_shards.py` and
the existing shard data (524K frames affected, 0.71%).

### HuggingFace Dataset

`erickfm/slippi-public-dataset-v3.7` — 95K unique replays organized by
character. Fox folder has 45,854 .slp files. This is the raw replay source.
Shards must be built from these using `tools/slp_to_shards.py` with
appropriate metadata and `--hal-norm`.

### Building HAL-Normalized Shards

Requires a metadata directory with: `norm_stats.json`, `cat_maps.json`,
`stick_clusters.json`, `controller_combos.json` (5-combo version), and
`hal_norm.json`. The `controller_combos.json` MUST have 5 combos
(A, B, Jump, Z, None) for HAL mode. Using the 32-combo version produces
shards with 81-dim controller vectors instead of 54-dim, which crashes
the HAL model.

## Checkpoints

### The Only HAL Checkpoint

`/home/erick/projects/hal/checkpoints/000005242880.pt` — HAL's best at 5.2M
samples. This is the one that plays well. State dict with `module.` prefix
(from DDP). 101MB.

`checkpoints/hal_original.pt` is a DIFFERENT checkpoint (different md5sum,
different weights). Do not confuse them.

### MIMIC Checkpoints (all in `checkpoints/`)

All `hal-*` prefixed checkpoints are MIMIC models trained with `--model hal`.
The naming is confusing but intentional (they use HAL's architecture config).

Currently training: `hal-local_*` — local training run on fox_hal_local data.

## File Map

### Core
- `train.py` — Training loop (DDP, gradient accumulation, HAL mode, cosine LR)
- `mimic/model.py` — Model architecture (FramePredictor, HAL presets, attention variants)
- `mimic/dataset.py` — StreamingMeleeDataset (per-game and pre-windowed shards)
- `mimic/frame_encoder.py` — Input encoders (HALFlatEncoder for HAL mode)
- `mimic/features.py` — Feature encoding (cluster centers, controller one-hot, normalization)
- `eval.py` — Offline evaluation (validation metrics)
- `inference.py` — Legacy inference script (use tools/run_hal_model.py instead for HAL mode)

### Tools
- `tools/run_hal_model.py` — **Our reimplementation** of HAL's inference. Loads HAL checkpoints and plays via Dolphin. Fixed 2026-04-07.
- `tools/run_mimic_via_hal_loop.py` — Runs MIMIC checkpoints through HAL-style inference loop. Fixed stats path and player ordering 2026-04-08.
- `tools/head_to_head.py` — Runs two checkpoints (HAL/MIMIC) against each other in Dolphin. Added 2026-04-08.
- `tools/validate_checkpoint.py` — Evaluates checkpoint(s) on val data, reports per-head CE loss
- `tools/verify_hal_pipeline.py` — Compares our preprocessing against HAL's. Run after inference changes.
- `tools/slp_to_shards.py` — Raw .slp replays -> .pt tensor shards (core pipeline)
- `tools/slp_to_ranked_shards.py` — Ranked replay sharding by character
- `tools/gameplay_health.py` — Analyze inference log for gameplay quality metrics
- `tools/diagnose.py` — Pipeline debugging (tensor-level train vs inference comparison)
- `tools/inference_diag.py` — Offline inference diagnostics (output distribution stats)
- `tools/validate_pipeline.py` — Verify slp_to_shards output matches old pipeline
- `tools/split_by_character.py` — Split dataset by character

### Docs
- `docs/research-notes-2026-04-08.md` — HAL vs MIMIC pipeline audit, button encoding, shoulder analog/digital
- `docs/research-notes-2026-04-07.md` — max_steps bug fix, inference bug fixes, training results
- `docs/archive/research-notes-*.md` — Historical research journal (2026-03-14 through 2026-04-06).
  **Warning:** these contain claims that were believed true at the time but later
  proven wrong (e.g., "HAL doesn't overfit", "26.3M params", specific stats file
  claims). Treat as historical context, not ground truth.
- `docs/results-2026-03-17.md` — Early training result summaries
- `docs/hal-mimic-diff-audit.md` — HAL vs MIMIC architecture comparison (historical, 2026-04-02)
- `GPUS.md` — Remote GPU machine addresses and status

## Research Notes Warning

The `docs/research-notes-*.md` files are a chronological journal spanning
2026-03-14 to 2026-04-08. They record what was believed true at each point
in time. Several claims in the notes were later found to be incorrect:

- "HAL's val loss is stable" — Actually HAL overfits too (val rises from 0.744 to 0.802 after 5.2M samples)
- "Architecture: 26,274,803 params" — Actually ~19,950,000 params
- "HAL uses `hal/data/stats.json` for inference" — The Preprocessor actually loads `checkpoints/stats.json`
- Various "this matches HAL" claims that were later found to have subtle differences

The notes are still valuable for understanding the project's evolution and
the reasoning behind decisions. Just don't treat specific numbers or "verified"
claims as current truth without checking the code.

## The HAL Repo (`/home/erick/projects/hal`)

This is Eric Gu's original HAL codebase. Key files:

- `hal/eval/play.py` — Ground-truth inference script (always works)
- `hal/preprocess/preprocessor.py` — Preprocessing (normalization, controller encoding)
- `hal/preprocess/transformations.py` — Feature transforms (one-hot encoding, sampling)
- `hal/preprocess/input_configs.py` — Input feature configuration
- `hal/preprocess/postprocess_configs.py` — Output decoding configuration
- `hal/training/models/gpt.py` — Model architecture (GPTv5Controller)
- `hal/constants.py` — Cluster centers, button lists, character/stage/action indices
- `hal/emulator_helper.py` — Dolphin controller interface
- `hal/gamestate_utils.py` — Gamestate extraction from melee-py
- `hal/data/stats.json` — Full dataset stats (222M frames, DO NOT use for inference)
- `checkpoints/stats.json` — Fox training stats (27M frames, USE THIS ONE)
- `checkpoints/config.json` — Training config
- `checkpoints/000005242880.pt` — Best checkpoint (5.2M samples)
- `hal/local_paths.py` — Local machine paths (emulator, ISO, replay dir)

## Common Pitfalls for Agents

1. **Don't trust file/variable names.** `hal_original.pt` is not HAL's best
   checkpoint. `run_hal_model.py` is not HAL's code. `hal-exact-8gpu_best.pt`
   is a MIMIC model.

2. **Don't trust research notes as current truth.** Always verify against code.

3. **Don't use `hal/data/stats.json` for inference.** Use `checkpoints/stats.json`.

4. **Don't run inference while training on the same GPU.** Frame drops make
   gameplay look broken when the model is actually fine.

5. **Don't assume `max_samples` means total samples.** With DDP, it's divided
   by effective batch size (local_batch * n_gpus * grad_accum).

6. **Don't mix normalization schemes.** HAL-mode training needs HAL-normalized
   shards with 5-combo controller encoding. The `ranked_fox` data uses old
   normalization with 32 combos — incompatible.

7. **Don't hardcode head hidden dims as 256.** HAL's heads use `input_dim // 2`
   which varies per head (256, 257, 262, 280).

8. **Check `sorted()` on player dicts.** melee-py's `gamestate.players` dict
   order is not guaranteed to match port order. Always `sorted()`.

9. **Use `blocking_input=True` for inference.** This makes Dolphin wait for
   controller input before advancing each frame. Without it, slow model
   inference causes frame drops (the game advances without receiving input).
   In head-to-head, non-blocking mode systematically disadvantages whichever
   model's inputs are flushed second. Fixed 2026-04-08.

11. **Shoulder is analog-only.** Neither HAL nor MIMIC triggers the digital L/R
   click (`press_button(BUTTON_L)`). Only analog values are sent via
   `press_shoulder(BUTTON_L, value)`. The game's tech input appears to use the
   analog threshold, not the digital click. See research notes 2026-04-08.

12. **Button encoding is single-label.** The 5-class button head (A, B, Jump,
    Z, None) cannot represent two simultaneous action buttons. Multi-button
    overlaps (2.65% of frames) are collapsed via early-release encoding: the
    newest button (0→1 transition) gets the label. Shoulder+button combos ARE
    representable since shoulder is a separate head.

13. **Post-frame game state leak (fixed 2026-04-11).** melee-py returns
    post-frame game state (action already reflects button press) with
    pre-frame controller inputs. Old shards (`fox_hal_full`) have target[i]
    = buttons[i], so the model can read the answer from self_action. v2
    shards (`fox_hal_v2`) shift targets to buttons[i+1], fixing the leak.
    **Do NOT use `--controller-offset` or `--reaction-delay 1` with v2 shards.**

14. **Don't compare val loss across shard versions.** v2 shards produce
    higher val loss because the model can no longer cheat via action→button
    memorization. A val loss of ~1.0 on v2 shards may correspond to better
    gameplay than 0.74 on old shards.
