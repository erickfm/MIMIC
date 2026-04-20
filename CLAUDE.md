# CLAUDE.md — Agent Orientation for MIMIC

## What This Project Is

MIMIC is a behavior-cloning bot for Super Smash Bros. Melee. It watches human
replays and learns to predict controller inputs from game state. At inference
it drives a controller through Dolphin (the GameCube emulator) via libmelee.

MIMIC started as an independent BC-for-Melee project, cycled through a lot
of ideas, and at one point re-bootstrapped its architecture and data
pipeline from [HAL](https://github.com/ericyuegu/hal) (by Eric Gu) to get a
known-good baseline. From there it diverged again: 7-class button head
(adding TRIG / A_TRIG for airdodge-wavedash-tech), v2 shard target
alignment, RoPE as an alternative to Shaw relpos, netplay + Discord bot
frontend. The active code path is MIMIC's own; HAL is just historically
where the transformer backbone came from.

## Lineage hazards (things that look like HAL)

- `tools/run_hal_model.py` — **still loads actual HAL checkpoints** (Eric
  Gu's original weights). Kept as a reference implementation. Not used by
  any production code path. Don't rename it; don't delete it.
- `tools/validate_checkpoint.py` has an inner `HALModel` class — also a
  legacy HAL-compat reimplementation used only for validating old HAL
  checkpoints. Leave alone.
- Research notes in `docs/` reference `--hal-mode`, `hal_norm.json`, the
  "HAL preset", etc. They're frozen snapshots of when those names existed.
  Don't sweep them.
- Legacy on-disk data directories (`data/fox_hal_full`, `data/fox_hal_local`,
  `data/fox_hal_800m`) keep their names. Nothing in the active code path
  references them anymore, they're frozen.

## Architecture

MIMIC's canonical config (preset name `mimic`, bootstrapped from HAL's
GPTv5Controller and later diverged):

- **Params:** ~19.95M (minimal-features) / ~20.00M (full-features; +18K
  projection weights for 36 additional input scalars)
- **Transformer:** d_model=512, 6 layers, 8 heads, block_size=1024
- **Position encoding:** Shaw relative-position attention (`mimic`).
  RoPE variants (`mimic-rope*`) are deprecated — see Pitfalls.
- **Input:**
  - **Minimal features** (historical default, `--mimic-minimal-features`):
    Linear(166 → 512) from `[stage_emb(4) + 2*char_emb(12) + 2*action_emb(32) + gamestate(18) + controller(56)]`.
  - **Full features** (new default as of 2026-04-19): Linear(202 → 512) —
    same categorical embeddings, but the 18-dim `gamestate` becomes
    54-dim (27 per player: 22 numeric + 5 flags). Drop
    `--mimic-minimal-features` from the CLI to enable.
- **Output heads (autoregressive with detach):** shoulder(3) → c_stick(9) → main_stick(37) → buttons(7)
- **Head hidden dim:** `input_dim // 2` (NOT a fixed 256 — each head has different hidden size)
- **Sequence length:** 180 frames (~3 seconds)
- **Dropout:** 0.2 (mimic / modern-relpos), 0.1 (mimic-xl and other
  post-bugfix presets)

### Gamestate columns per player

**Minimal (9):** `percent, stock, facing, invulnerable, jumps_left,
on_ground, shield_strength, position_x, position_y` (exact HAL order
is preserved by a reindex in the encoder's minimal path).

**Full (27):** 22 numeric + 5 flags in native shard order:

    numeric[0-4]   : pos_x, pos_y, percent, stock, jumps_left
    numeric[5-9]   : speed_air_x_self, speed_ground_x_self,
                     speed_x_attack, speed_y_attack, speed_y_self
    numeric[10-12] : hitlag_left, hitstun_left, invuln_left
    numeric[13]    : shield_strength
    numeric[14-21] : ecb_{bottom,left,right,top}_{x,y}
    flags[0-4]     : on_ground, off_stage, facing, invulnerable,
                     moonwalkwarning

The controller is a 56-dim one-hot: main_stick(37) + c_stick(9) +
buttons(7) + shoulder(3).

### Preset variants (in `mimic/model.py:MODEL_PRESETS`)

- `mimic` — canonical (relpos, LN, GELU, full attention). Production.
- `mimic-rope`, `mimic-rope-lt`, `mimic-rope-lf`, `mimic-rope-deep`,
  `mimic-xpos`, `mimic-selrope`, `mimic-ropenope`, `mimic-xpos-64`,
  `mimic-flex`, `mimic-ropeflex`, `mimic-learned` — position-encoding
  experiments. Mostly historical; RoPE-family underperforms. Not used
  in production.
- `mimic-xl` (2026-04-18) — width+FFN scale-up of `mimic`. d_model=768,
  nhead=12, num_layers=6, dim_feedforward=3072, dropout=0.1, SwiGLU
  FFN, relpos. ~44M params. Adding `--num-layers 10` gives a ~73M
  deeper variant. See `research-notes-2026-04-19.md` for the full
  scale sweep.
- `mimic-xl-rms` (2026-04-19) — `mimic-xl` plus RMSNorm. Tested at
  10-layer depth; did not meaningfully differ from LN variant.
- `modern-relpos`, `modern-relpos-gelu`, `modern` — `mimic`-sized
  (~20M) with GQA (n_kv_heads=2) + SwiGLU + RMSNorm. At puff scale
  none of these beat the plain `mimic` recipe, in ties to within
  0.001 val-loss across all three runs.
- `tiny`, `small`, `medium`, `base`, `deep`, `shallow`,
  `wide-shallow`, `xlarge`, `xxlarge`, `huge`, `giant` — legacy
  generic presets without mimic-specific keys. **Avoid for mimic_mode
  training** — they don't set `max_seq_len`, which triggers the
  seq_len-fallback-to-60 path.

### Optional: learned input gate (feature-importance diagnostic)

`ModelConfig.use_input_gate: bool` (default False). Enable per-run with
`--input-gate-l1 <lambda>` (typical: 0.01). When on,
`MimicFlatEncoder` adds a per-input-column sigmoid gate on the final
projection, and the training loop adds `lambda * sigmoid(gate).mean()`
to the loss as an L1 sparsity penalty. End-of-training writes
`checkpoints/{run_name}_gate_report.json` ranking every input scalar
by its learned gate value (0 = model pruned it, 1 = model kept it).

Primarily a diagnostic tool, not a production regularizer — at
λ=0.01 it costs ~0 val-loss (and in our one measurement on puff
actually *helped* slightly, likely through a small implicit
regularization effect). Leave `--input-gate-l1` unset (default 0) for
standard production runs.

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

**v2 shards** (`data/fox_v2`) fix this by shifting targets forward by 1
frame: `target[i] = buttons[i+1]`. The model sees the current game state and
predicts what to press NEXT. This matches inference exactly.

**Do NOT use `--controller-offset` or `--reaction-delay 1` with v2 shards.**
The alignment is already correct. Adding offsets would double-shift the data.

**Old shards** (`data/fox_hal_full`, `data/fox_hal_match_shards`) have the
leak. With those shards, use `--reaction-delay 1` to achieve the same effect
at dataloader time (this is what HAL does).

## Training

### Command (current best config — v2 shards, full features)

```bash
torchrun --nproc_per_node=2 train.py \
  --model mimic --encoder mimic_flat \
  --mimic-mode --mimic-controller-encoding \
  --stick-clusters hal37 --plain-ce \
  --lr 3e-4 --batch-size 256 --grad-accum-steps 1 \
  --max-samples 16777216 \
  --data-dir data/<char>_v2 \
  --self-inputs \
  --reaction-delay 0 \
  --run-name <name> \
  --no-warmup --cosine-min-lr 1e-6 \
  --nccl-timeout 3600
```

**Key 2026-04-19 change: DROP `--mimic-minimal-features`.** Prior to
that date the flag was silently ignored by `MimicFlatEncoder` — every
training run was using a 9-scalar-per-player gamestate (6 numeric + 3
flags) regardless of the flag, because the encoder sliced the
`self_numeric` tensor based on shape rather than honoring the config.
The encoder now honors the flag. Dropping it exposes the full 22-col
numeric + 5-flag per-player gamestate (velocity, hitlag, hitstun,
invuln_left, ECB, off_stage, moonwalkwarning) and is worth ~1.5-3.5%
val-loss reduction at no wall-clock cost. See
`docs/research-notes-2026-04-19.md` for the measurements.

Single-GPU variant: swap `torchrun --nproc_per_node=2` for `python3`
and use `--batch-size 64 --grad-accum-steps 8` to keep effective batch
at 512.

The legacy `--hal-*` flags and `--model hal` / `--encoder hal_flat` names
still work as aliases — saved checkpoints from before the 2026-04-13
rename load unchanged.

**`--model mimic-rope*` presets are deprecated as of 2026-04-18.** Use
`--model mimic` (Shaw relative-position attention). See "Common
Pitfalls" below for the RoPE note.

**`--self-inputs` is required even on v2 shards.** Earlier v2 runs that
dropped `--self-inputs` had val loss 2.3 and main stick F1 of 15%. With
`--self-inputs` on, val loss drops to 0.67 and main F1 to 57%. The flag
tells the encoder to use `self_controller` as a feature — without it, the
model has no controller history input at all. `--controller-offset` is
still not needed (v2 alignment is baked into the shard), but `--self-inputs`
is critical.

### Command (old leaked shards — legacy reproduction only)

```bash
torchrun --nproc_per_node=8 train.py \
  --model mimic --encoder mimic_flat \
  --mimic-mode --mimic-minimal-features --mimic-controller-encoding \
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
For bit-exact reproduction of pre-AMP runs, use `--no-amp --no-compile`.

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

### Model preset → seq_len gating (Critical — 2026-04-15)

`train.py` sets the global `SEQUENCE_LENGTH` for the dataset by looking
up `MODEL_PRESETS[model_preset]["max_seq_len"]`. Before the 2026-04-15
fix, it was hardcoded as `elif model_preset == "hal": SEQUENCE_LENGTH = 256`.
That string check was missed during the 2026-04-13 HAL→MIMIC rename,
so `--model mimic*` runs silently fell back to the module default
(`SEQUENCE_LENGTH = 60`) instead of 256 — a 4.3× reduction in temporal
context per sample. Every rename-era run was training on stunted
windows. See `docs/research-notes-2026-04-15.md` for the full story.

**Do not add new `if model_preset == "X"` gates.** If behavior depends
on a preset, read the value off `MODEL_PRESETS[model_preset]` so any
future preset/alias automatically picks it up.

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

Use `tools/play_vs_cpu.py` (not `run_hal_model.py`) for MIMIC
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

(Removed: `tools/verify_hal_pipeline.py` was a one-time preprocessing-vs-HAL
comparison tool. Deleted during the 2026-04-13 MIMIC rename since MIMIC has
diverged far enough from HAL that the equivalence check is no longer useful.)

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
| `data/fox_v2` | ~17K Fox games, 800MB shards, quality-filtered, **next-frame targets** | target[i] = buttons[i+1] (clean) | **Active — use with rd=0, no offset** |
| `data/falco_v2` | ~9K Falco games, same format | clean | Active |
| `data/cptfalcon_v2` | ~9K CptFalcon games, same format | clean | Active |
| `data/luigi_v2` | ~2K Luigi games, same format | clean | Active |
| `data/fox_hal_full` | ~10K Fox games, 800MB shards, quality-filtered | target[i] = buttons[i] (leaked) | Legacy — use with rd=1 |
| `data/fox_hal_800m` | 7,600 Fox games, 800MB shards | leaked | Legacy |
| `data/fox_hal_local` | 7,600 Fox games, 3.8GB shards | leaked | Legacy |

Legacy dirs keep `hal_*` in their names because nothing references them
from the active code path anymore — they're frozen. New data dirs always
drop the `hal_` prefix.

Use 800MB shards with `mmap=True` in DataLoader for optimal throughput. The
`tools/reshard.py` script can split large shards: `python tools/reshard.py --src <dir> --dst <dir> --target-mb 800`.

To build new shards, use `tools/slp_to_shards.py`
with `--mimic-norm` and a metadata dir containing 7-combo `controller_combos.json`.

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

### HuggingFace Datasets

Two sources of raw replays. The ranked one is canonical for new training.

- **`erickfm/melee-ranked-replays`** (canonical, since 2026-04-17 retrain).
  Ranked Slippi replays stored as **.tar.gz per (character, rank_pair,
  archive)**: `shards/{CHAR}_{rank_pair}_aN.tar.gz`. Rank pairs are
  `master-master`, `master-diamond`, `master-platinum`, `diamond-*`,
  `platinum-*`. Higher-rank token first (M>D>P). Per-char master-tier
  training pulls **all three `master-*` pairs** (master-master has both
  players master; master-diamond/platinum mixes in games where the other
  player is sub-master and we can't tell from the .slp which port is
  master — accepted as a data-quality/quantity tradeoff). See
  `docs/ranked-dataset-pipeline.md`.
- **`erickfm/slippi-public-dataset-v3.7`** — legacy. 95K replays by
  character; Fox folder has 45,854 .slp. Used for the 2026-04-12/13/14
  runs. `tools/download_fox.py` is hardcoded to this source.

### Building MIMIC-normalized Shards

Requires a metadata directory with: `norm_stats.json`, `cat_maps.json`,
`stick_clusters.json`, `controller_combos.json`, and `mimic_norm.json`.
The `controller_combos.json` MUST have 7 combos (A, B, Z, Jump, TRIG,
A_TRIG, None) for the 7-class button head. The old 5-combo version
(A, B, Jump, Z, None — HAL's scheme) still works via backcompat but
cannot represent airdodge / wavedash / L-cancel.

### Retraining a character from scratch (ranked master-* data)

Summary stats are preserved — **you don't need to regenerate them**. All 5
inference-relevant JSONs live on `huggingface.co/erickfm/MIMIC/<char>/`;
the 6th (`norm_minmax.json`) stays local in `data/<char>_v2/` and is only
needed if you want to rebuild `mimic_norm.json` from scratch.

**What you have on HF** per char (`erickfm/MIMIC/<char>/`):
`model.pt`, `config.json`, `metadata.json`, and the 5 JSONs —
`cat_maps.json`, `controller_combos.json`, `mimic_norm.json`,
`norm_stats.json`, `stick_clusters.json`.

**Retrain recipe** (also lives in `tools/run_all_chars.sh` for the full
7-char pipeline; the snippet below is the single-char equivalent):

```bash
C=fox; HF_BUCKET=FOX; IDX=1   # adjust per char

# 1. Raw .slp — all three master-* pairs
hf download erickfm/melee-ranked-replays --repo-type dataset \
  --include "shards/${HF_BUCKET}_master-master_a*.tar.gz" \
  --include "shards/${HF_BUCKET}_master-diamond_a*.tar.gz" \
  --include "shards/${HF_BUCKET}_master-platinum_a*.tar.gz" \
  --local-dir data/${C}_ranked_slp/_tars
for t in data/${C}_ranked_slp/_tars/shards/*.tar.gz; do
  tar -xzf "$t" -C data/${C}_ranked_slp/
done

# 2. Metadata — pull from HF if data/${C}_v2 is missing
mkdir -p data/${C}_v2
hf download erickfm/MIMIC --include "${C}/*.json" --local-dir data/${C}_v2
mv data/${C}_v2/${C}/*.json data/${C}_v2/ && rmdir data/${C}_v2/${C}

# 3. Re-shard using existing mimic_norm.json
python3 tools/slp_to_shards.py \
  --slp-dir data/${C}_ranked_slp \
  --meta-dir data/${C}_v2 \
  --mimic-norm data/${C}_v2/mimic_norm.json \
  --character ${IDX} \
  --staging-dir data/${C}_v2 \
  --repo erickfm/mimic-${C}-v2 --no-upload --keep-staging \
  --shard-gb 4.0 --val-frac 0.1 --seed 42

# 4. Train (relpos, full features, eff_batch 512, 32,768 steps)
torchrun --nproc_per_node=2 train.py \
  --model mimic --encoder mimic_flat \
  --mimic-mode --mimic-controller-encoding \
  --stick-clusters hal37 --plain-ce \
  --lr 3e-4 --batch-size 256 --grad-accum-steps 1 \
  --max-samples 16777216 --data-dir data/${C}_v2 \
  --self-inputs --reaction-delay 0 \
  --run-name ${C}-retrain-$(date -u +%Y%m%d) \
  --no-warmup --cosine-min-lr 1e-6 --nccl-timeout 3600
```

(Note: `--mimic-minimal-features` is intentionally absent — see
"Training" section above and `docs/research-notes-2026-04-19.md`.)

Wall time per char (2×RTX 5090): ~10 min download/extract + 30-90 min
shard + ~50 min train = ~1.5-2.5 hours. Disk peak: 200-800 GB (chars with
bigger master populations like sheik and cptfalcon are closer to the top).

Character index + HF bucket name per char (bucket ≠ display name for
Sheik and Puff because the ranked pipeline collapses Zelda+Sheik into one
bucket and Jigglypuff is the full name):

| char       | HF bucket     | idx |
|------------|---------------|-----|
| fox        | `FOX`         | 1   |
| falco      | `FALCO`       | 22  |
| marth      | `MARTH`       | 18  |
| sheik      | `ZELDA_SHEIK` | 7   |
| cptfalcon  | `CPTFALCON`   | 2   |
| puff       | `JIGGLYPUFF`  | 15  |
| luigi      | `LUIGI`       | 17  |

## Checkpoints

### The Only Actual HAL Checkpoint

`/home/erick/projects/hal/checkpoints/000005242880.pt` — HAL's (Eric Gu's)
best at 5.2M samples. State dict with `module.` prefix (from DDP). 101MB.
Loaded via `tools/run_hal_model.py` for comparison runs only. Not used
by any MIMIC production code path.

### MIMIC Checkpoints (all in `checkpoints/`)

**Naming convention (new runs from 2026-04-13 onward):**

    {char}-{YYYYMMDD}-{descriptor}-{steps}k.pt

The `{descriptor}` slot is a free-form tag for whatever is most distinctive
about the run — usually the position encoding (`rope`, `relpos`), but it
can also flag a non-standard recipe: `overfit`, `wavedash`, `small`,
`longrun`, `lowdropout`, etc. Pick whatever makes this checkpoint
recognizable next to others for the same character on the same day.

Current best checkpoints (all in `checkpoints/`):

**Active per-character bests:**
- `puff-20260419-mimic-fullfeat-gate01-33k.pt` — current best Puff
  (mimic + full features + input-gate λ=0.01, val 0.6641). First
  character trained with `--mimic-minimal-features` dropped; ~3.6%
  below the 0.6890 minimal-features baseline at the same param count.
- `falco-20260412-relpos-28k.pt` — current best Falco (relpos, val 0.7374)
- `cptfalcon-20260412-relpos-27k.pt` — current best CptFalcon (relpos, val 0.71)
- `luigi-20260412-relpos-5k.pt` — current best Luigi (relpos, early-stopped, val ~1.0)

**Recent pipeline runs (wandb but not yet promoted to
`{char}-{date}-{tag}-{steps}k.pt`; underlying `_best.pt` /
`_bestloss.pt` still present):**
- Fox ~0.7081 (run `qeka6rq8`), Falco ~0.7448 (`zb1vhjxs`),
  Sheik ~0.6611 (`jc4xe4dv`), CptFalcon ~0.7356 (`6k1x8xdi`),
  Marth ~0.6746 (`eo8yjem4`) — all relpos / `--model mimic` runs from
  the 2026-04-17/18 per-character pipeline cycle, trained with
  `--mimic-minimal-features` (pre-fullfeat). Expect ~1-3% val
  improvement when re-run without that flag.

**Scaling / architecture sweep checkpoints (puff, 2026-04-18/19) — see
`research-notes-2026-04-19.md`:**
- `puff-20260418-relpos-xl-30k.pt` — `mimic-xl` 6L, 44M, val 0.6766.
- `puff-20260419-relpos-xl-deep-31k.pt` — `mimic-xl` 10L, 73M, val 0.6698.
- `puff-20260419-modern-relpos-32k.pt` — modern block (GQA+SwiGLU+RMS), 20M, val 0.6882.
- `puff-20260419-modern-relpos-gelu-29k.pt` — modern block, GELU FFN, 20M, val 0.6883.

**Superseded / kept for audit only:**
- `fox-20260415-rope-25k.pt` — old Fox best (RoPE, val 0.7358); RoPE
  variants are deprecated as of 2026-04-18.
- `fox-20260413-rope-32k.pt` — Fox tainted by the seq_len=60 bug.
- `fox-20260411-relpos-noself-28k.pt` — Fox trained without
  `--self-inputs`.

Rationale: the old names (`hal-7class-v2-long`, `falco-7class-v2-full`,
etc.) carried no date, no encoder info, and the "7class-v2" prefix was
universal noise. All four legacy production checkpoints were retroactively
renamed 2026-04-13 and all live references (`tools/discord_bot.py`,
`tools/upload_models_to_hf.py`, `tools/play_netplay.py`, `README.md`,
`docs/discord-bot-setup.md`) updated in the same commit. Research notes
were left alone — they're historical records.

**Set `--run-name` to `{char}-{YYYYMMDD}-{descriptor}`** when starting a new
run (the step suffix gets appended when renaming the finished `_best.pt`).

**Promotion policy.** When a finished wandb run beats the current best
`val/total` for its character, the candidate promotion is: pull the
checkpoint, rename it to `{char}-{YYYYMMDD}-{descriptor}-{steps}k.pt`,
upload it to `erickfm/MIMIC` on HF, and update the Discord bot's default
for that character. Skip runs whose name starts with `SWEEP-`, `DBG`,
`DEBUG-`, `FIX-`, or `BENCH-` — those are infra/debug and not production
candidates even if their loss is lower. A promotable run is a
named-character training like `fox-20260416-master-relpos`. Always
surface candidates to the user and ask before pushing — don't
auto-promote. If a new character appears (e.g. first Marth run), flag it
separately since the bot's character list needs wiring too.

**Bot startup has an orphan sweep.** `_cleanup_orphan_processes()` runs
before `bot.run()` and SIGTERMs (then SIGKILLs) any `play_netplay.py`
and our-path `dolphin-emu` processes still alive from a prior bot run.
These would otherwise be reparented to init and burn CPU/GPU forever —
we had a 10+ hour Dolphin at 85% CPU with no replays being written. If
you ever need a play_netplay.py to survive a bot restart, this is where
to carve an exception, not by disabling the sweep.

**Bot-startup audit.** Whenever you (re)start `tools/discord_bot.py`, do
a promotion audit first. The authoritative source is HF
`erickfm/MIMIC` — checkpoints live there after promotion, and the GPU
boxes are ephemeral so wandb checkpoints aren't retrievable once the
machine dies. Procedure:

1. List files under `erickfm/MIMIC` via `HfApi.list_repo_files`. For each
   character dir (`fox/`, `falco/`, `cptfalcon/`, `luigi/`, plus any new
   ones), read `metadata.json` for `run_name`, `global_step`, and
   `val_loss`.
2. Compare each character's HF `run_name` to the filename the bot's
   `CHARACTERS` dict in `tools/discord_bot.py` currently points at. If
   HF has a newer run (different `run_name` / higher step / better
   `val_loss`) than what's wired into the bot, it's a replacement
   candidate.
3. Also flag any HF character directory the bot doesn't know about —
   that's a new character the bot needs wired up (extend `CHARACTERS`
   dict + aliases).
4. Report candidates to the user and ask whether to pull + swap before
   bringing the bot up. On approval: `snapshot_download` the new
   character dirs into `hf_checkpoints/`, copy the `.pt` into
   `checkpoints/` with the project naming convention, edit
   `tools/discord_bot.py` `CHARACTERS` dict, and restart the bot.

Secondary (supplementary) check: wandb `erickfm/MIMIC` for finished
non-debug runs newer than HF's `last_modified` — useful to remind the
user "you have trained X but it hasn't been uploaded yet," but the
checkpoints themselves have to come from the GPU box (or be re-trained)
since wandb doesn't store the `.pt`.

## File Map

### Core
- `train.py` — Training loop (DDP, gradient accumulation, mimic_mode, cosine LR)
- `mimic/model.py` — Model architecture (FramePredictor, mimic presets, attention variants)
- `mimic/dataset.py` — StreamingMeleeDataset (per-game and pre-windowed shards)
- `mimic/frame_encoder.py` — Input encoders (MimicFlatEncoder for
  mimic_mode). As of 2026-04-19: honors `mimic_minimal_features`
  (previously ignored — minimal path unchanged for back-compat, full
  path exposes 22 numeric + 5 flags per player). Also hosts the
  optional L1-gated input projection (`use_input_gate`) used for
  feature-importance diagnostics.
- `mimic/features.py` — Feature encoding (cluster centers, controller one-hot, normalization)
- `eval.py` — Offline evaluation (validation metrics)
- `inference.py` — Legacy inference script (use `tools/play_vs_cpu.py` or `tools/play_netplay.py` in modern code paths)

### Tools
**Inference (local):**
- `tools/play_vs_cpu.py` — Runs MIMIC checkpoints vs CPU in Dolphin. Uses shared `inference_utils.decode_and_press`.
- `tools/head_to_head.py` — Runs two checkpoints against each other in the same Dolphin instance (watchable ditto).
- `tools/run_hal_model.py` — Our reimplementation of HAL's 5-class inference. Loads HAL checkpoints. Structurally can't wavedash (no TRIG class).

**Inference (online, Slippi netplay):**
- `tools/play_netplay.py` — Joins a Slippi Online Direct Connect lobby. Uses `MenuHelper.menu_helper_simple(connect_code=...)`, detects bot's port via the `connectCode` field on `PlayerState` (handles dittos and palette swaps). **Persistent-session mode** (see below): plays up to `--max-matches N` back-to-back matches in one Dolphin process, emitting a per-match stdout block (`MATCH_START:`, `RESULT:`, `SCORE:`, `REPLAY:`) and a single `SESSION_END:` on exit. Default `--max-matches 1` preserves the old one-shot CLI behavior; the Discord bot passes `-1` for unlimited.
- `tools/discord_bot.py` — Discord front-end (prefix commands: `!play`, `!queue`, `!cancel`, `!info`). Single-session FIFO queue via `asyncio.Queue`. Spawns `play_netplay.py` once per user and streams its stdout: each `MATCH_START` → `▶️ Match N starting` post, each `RESULT/SCORE/REPLAY` triplet → result announcement + replay upload. Uploads the saved replay back to the channel. Config via `.env` (see `.env.example`). Per-character checkpoint labels render consistently (run name + `val`, step count, `btn` F1 pulled from each char's HF `metadata.json`) across `!info`, session-starting, and match-result messages via `_ckpt_label_for(char)`.

**Persistent-session model (2026-04-16).** Each Discord `!play` spawns
one `play_netplay.py` that keeps Dolphin alive across multiple matches
in the same Slippi Direct Connect lobby. After each match, `MenuHelper`
drives through POSTGAME_SCORES back to CSS; the opponent has
`--rematch-timeout` seconds (default 30) to pick a character and press
Start for the next match. The session ends when (a) opponent DCs or
idles, (b) another user calls `!play` (the bot writes `STOP\n` to the
subprocess's stdin, which finishes the current match then exits), or
(c) the current player `!cancel`s (same STOP path). **Do NOT
reintroduce per-match subprocess spawning** — the 30–60 s of Dolphin
relaunch dead air was the exact UX problem this refactor fixed.

Stdout protocol emitted by `play_netplay.py`:
```
MATCH_START: <1-based idx>     # emitted when IN_GAME is first reached
RESULT: win|loss|draw|disconnect|no-opponent|timeout|failed
SCORE: bot=Xstk/Y% opp=Xstk/Y%
REPLAY: /abs/path/Game_YYYYMMDDThhmmss.slp
# (above four repeat per match)
SESSION_END: max-matches|opponent-gone|opponent-timeout|stopped|error|signal|hard-timeout
```

Per-match state reset lives in the outer match loop of `play_netplay.py`
(MenuHelper `stage_selected` / `frozen_stadium_selected`, `PlayerState`
recreated on each IN_GAME transition). `opponent_last_seen` and
`opponent_ever_seen` deliberately persist across matches for DC
detection. STOP polling uses `select.select([sys.stdin], [], [], 0)` —
non-blocking, checked once per console step.

**Inference (shared):**
- `tools/inference_utils.py` — Shared inference pipeline: `load_mimic_model`, `load_inference_context`, `build_frame`, `build_frame_p2`, `PlayerState`, `decode_and_press`. **This is where the L-button fix (2026-04-13) lives.** Any new inference entry point should import from here, not reimplement.

**Diagnostics:**
- `tools/inspect_frame.py` — Shows exactly what goes into and out of the model for a single frame. Takes `--shard 0 --frame 534 --context 2` style args.
- `tools/extract_wavedashes.py` — Extracts wavedash-only training windows for overfit sanity checks. Used to prove the architecture can represent wavedashes (2026-04-13).
- `tools/validate_checkpoint.py` — Evaluates checkpoint(s) on val data, reports per-head CE loss.
- `tools/diagnose.py` — Pipeline debugging (tensor-level train vs inference comparison).

**Data:**
- `tools/slp_to_shards.py` — Raw .slp replays → .pt tensor shards. Produces **v2 shards** (target[i] = buttons[i+1]) by default since 2026-04-11c.
- `tools/split_by_character.py` — Split dataset by character.

### Docs
- `docs/discord-bot-setup.md` — Full setup guide for the Discord bot (Slippi account, Dolphin, ISO, env vars, troubleshooting).
- `docs/research-notes-2026-04-19.md` — **Scale sweep + the
  `--mimic-minimal-features` silent-ignore fix.** Shows that 2×/4×
  width/depth scale-ups plateau at ~2-3% val drop vs baseline, that
  modern-block (GQA/SwiGLU/RMS) changes are null at this size, and
  that exposing the full gamestate (dropping `--mimic-minimal-features`)
  on a baseline-sized model beats every scaled variant. Also
  documents the L1 input-gate feature-importance tooling.
- `docs/research-notes-2026-04-13.md` — **The TRIG L-button bug debug story.** The most important note for anyone touching `decode_and_press` or questioning why BC bots can't wavedash.
- `docs/research-notes-2026-04-12.md` — Multi-character v2 training results (Falco/CptFalcon/Luigi) + bot-vs-bot ditto analysis.
- `docs/research-notes-2026-04-11c.md` — v2 shard target shift. Post-frame gamestate leaks answers to the button head; fixed by shifting targets forward by 1 frame in the shard itself.
- `docs/research-notes-2026-04-11b.md` — Bistable inference analysis (found earlier, partially superseded by the TRIG fix).
- `docs/research-notes-2026-04-11.md` — 7-class button encoding, first round of v2 training, relpos retrain.
- `docs/research-notes-2026-04-09.md` — GPU throughput optimization, RoPE vs relpos experiments.
- `docs/research-notes-2026-04-08.md` — HAL vs MIMIC pipeline audit.
- `docs/research-notes-2026-04-07.md` — max_steps bug, inference bug fixes, training results.
- `docs/archive/research-notes-*.md` — Historical journal (2026-03-14 through 2026-04-06). **Warning:** contains claims that were later proven wrong. Treat as historical context only.
- `GPUS.md` — Remote GPU machine addresses and status.

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

1. **`tools/run_hal_model.py` loads actual HAL weights — MIMIC checkpoints
   use `tools/play_vs_cpu.py` / `play_netplay.py` / `head_to_head.py`.**
   `run_hal_model.py` is the reference-implementation path for Eric Gu's
   original HAL checkpoints; it's not used by any MIMIC production code.

2. **Don't trust research notes as current truth.** Always verify against code.

3. **Don't run inference while training on the same GPU.** Frame drops make
   gameplay look broken when the model is actually fine.

4. **Don't assume `max_samples` means total samples.** With DDP, it's divided
   by effective batch size (local_batch * n_gpus * grad_accum).

5. **Don't mix normalization schemes.** `mimic_mode` training needs
   `mimic_norm.json` + MIMIC controller combos (7-combo for current models,
   5-combo for legacy HAL-compat). The `ranked_fox` data uses old
   normalization with 32 combos — incompatible.

6. **Don't hardcode head hidden dims as 256.** The autoregressive heads use
   `input_dim // 2` which varies per head (256, 257, 262, 280).

7. **Check `sorted()` on player dicts.** melee-py's `gamestate.players` dict
   order is not guaranteed to match port order. Always `sorted()`.

8. **Use `blocking_input=True` for inference.** This makes Dolphin wait for
   controller input before advancing each frame. Without it, slow model
   inference causes frame drops (the game advances without receiving input).
   In head-to-head, non-blocking mode systematically disadvantages whichever
   model's inputs are flushed second. Fixed 2026-04-08.

9. **The TRIG (L/R) class must call `press_button`, not just `press_shoulder`.**
   Melee's shoulder events split on analog-vs-digital:
   - **Shield**: analog threshold (any shoulder value above ~0.3)
   - **L-cancel**: analog threshold, rising edge during the L-cancel window
   - **Tech**: digital L/R press
   - **Airdodge**: digital L/R press
   - **Wavedash**: airdodge into ground → digital press required

   So `press_shoulder(BUTTON_L, 1.0)` alone is enough for shield + L-cancel,
   but tech / airdodge / wavedash need `press_button(BUTTON_L)`. The 7-class
   button head's TRIG (class 4) and A_TRIG (class 5) classes call
   `ctrl.press_button(BUTTON_L)` to cover all four cases at once.

   The real bug in `decode_and_press` before 2026-04-13 was that the TRIG
   branch did *nothing* at all — no analog, no digital — so the model could
   predict "press L" at 99% confidence and the game received a neutral
   shoulder. See `tools/inference_utils.py:decode_and_press` and research
   notes 2026-04-13 for the full debug story. HAL's 5-class button head has
   no TRIG class, so HAL-lineage bots can't produce a shoulder event even in
   principle.

10. **Button encoding is single-label.** The 5-class button head (A, B, Jump,
    Z, None) cannot represent two simultaneous action buttons. Multi-button
    overlaps (2.65% of frames) are collapsed via early-release encoding: the
    newest button (0→1 transition) gets the label. Shoulder+button combos ARE
    representable since shoulder is a separate head.

11. **Post-frame game state leak (fixed 2026-04-11).** melee-py returns
    post-frame game state (action already reflects button press) with
    pre-frame controller inputs. Old shards (`fox_hal_full`) have target[i]
    = buttons[i], so the model can read the answer from self_action. v2
    shards (`fox_v2`) shift targets to buttons[i+1], fixing the leak.
    **Do NOT use `--controller-offset` or `--reaction-delay 1` with v2 shards.**

12. **Don't compare val loss across shard versions.** v2 shards produce
    higher val loss because the model can no longer cheat via action→button
    memorization. A val loss of ~1.0 on v2 shards may correspond to better
    gameplay than 0.74 on old shards.

12a. **RoPE presets are deprecated (2026-04-18).** The `mimic-rope*`
    family — `mimic-rope`, `mimic-rope-lt`, `mimic-rope-lf`,
    `mimic-rope-deep`, `mimic-rope-xxx` aliases — underperforms the
    relpos baseline. Default to `--model mimic`. `fox-20260415-rope-25k.pt`
    is kept for audit only. Do not start new production runs on a RoPE
    preset unless you're specifically testing RoPE and know what
    you're looking at. The bug is in the positional-encoding path
    itself, not the training recipe.

12b. **`--mimic-minimal-features` silently degrades loss (fixed
    2026-04-19, flag repurposed).** Prior to 2026-04-19,
    `MimicFlatEncoder.__init__` accepted the
    `mimic_minimal_features` kwarg and discarded it; the forward pass
    sliced the numeric tensor based on shape (`sn.shape[-1] > 7`), so
    every run used the 9-scalar-per-player minimal gamestate
    regardless of the flag. As of 2026-04-19 the encoder honors the
    flag: minimal path is bit-identical to pre-fix behavior (for
    back-compat with old checkpoints), full path (flag omitted)
    exposes 27 scalars per player (22 numeric + 5 flags). The
    full-features path is the new default and is worth 1.5-3.5%
    val-loss reduction at no wall-clock cost. **Drop
    `--mimic-minimal-features` from all new training recipes** unless
    deliberately reproducing a pre-fix baseline. Old checkpoints
    trained with the flag still load unchanged — the flag flows
    through the pickled config.

12c. **Inference must produce the full 22-col numeric + 5-flag tensor
    (fixed 2026-04-20).** `tools/inference_utils.py:build_frame` /
    `build_frame_p2` / `build_mock_frame` used to produce a 7-col
    numeric + 5-flag tensor matching the pre-fix minimal encoder's
    expectations. When `--mimic-minimal-features` was dropped as
    the new default, `MimicFlatEncoder(mimic_minimal_features=False)`
    started raising `ValueError: requires 22-col self_numeric shards`
    at inference time for any fullfeat checkpoint. Fixed by:
    (1) extending `MIMIC_NUM` → `MIMIC_NUM_FULL` (22 columns in exact
    shard order — see `mimic/features.py:numeric_state` for the
    schema), (2) reading speeds / hitlag / hitstun / invuln_left /
    ECB via libmelee's `PlayerState.speed_*`, `.hitlag_left`,
    `.hitstun_frames_left`, `.invulnerability_left`, and
    `.ecb_{bottom,left,right,top}` attrs, (3) normalizing extras by
    z-score using `norm_stats.json` mean/std (mirrors
    `slp_to_shards.py:549-554`), (4) loading `norm_stats.json` in
    `load_inference_context`. **This works for BOTH minimal and
    fullfeat checkpoints** because the minimal encoder path slices
    the 22-col input to 6 cols internally. Old checkpoints still
    play correctly; new fullfeat checkpoints now play too. If
    `norm_stats.json` is missing from a data dir (shouldn't happen
    with any properly-staged char), extras pass through as 0.0 and
    minimal checkpoints still work (the encoder slices the zeros
    away); fullfeat checkpoints will see zeroed-out features and
    perform worse.

13. **Discord bot portability: keep paths relative in `.env`.** The Discord
    bot's `.env` uses relative paths (`./emulator/...`, `./melee.iso`,
    `./slippi_home`) that `_resolve_path` in `tools/discord_bot.py` converts
    to absolute paths against the repo root at runtime. This makes the repo
    `scp`-able to any machine that has run `setup.sh`. Don't hardcode
    absolute paths in `.env` — it defeats portability.

14. **Slippi credentials live at `./slippi_home/Slippi/user.json`** (gitignored).
    Not at `~/.config/SlippiOnline/Slippi/user.json` — libmelee IS pointed
    at the bundled dir explicitly via `dolphin_home_path=SLIPPI_HOME` in
    `tools/play_netplay.py`. Place user.json in the repo so that uploading
    the repo to a new machine carries the bot's Slippi login with it.
    Never commit `slippi_home/` — it contains the bot's `playKey`.

15. **Dolphin needs runtime shared libraries.** The AppImage-extracted
    `dolphin-emu` binary links against `libasound2`, `libusb-1.0-0`,
    `libgtk-3-0`, `libbluetooth3`, `libhidapi-hidraw0`, and friends. Missing
    any of them makes the binary exit 127, which libmelee surfaces as
    `RuntimeError: Unexpected return code 127 from dolphin` inside
    `Console.__init__` — `play_netplay.py` then exits 1 with an empty
    `RESULT:` line and the Discord bot reports `result=failed score=`.
    `setup.sh` installs the full list; on existing machines, run
    `ldd emulator/squashfs-root/usr/bin/dolphin-emu | grep 'not found'` to
    see what's missing.

16. **Setup Xvfb for headless machines.** Dolphin crashes at startup with
    "Unable to initialize GTK+, is DISPLAY set properly?" if no display
    server is available. `setup.sh` installs and starts Xvfb on `:99` and
    adds `export DISPLAY=:99` to `~/.bashrc`. On existing machines, check
    `DISPLAY` is set in the environment the Discord bot / play_netplay.py
    inherits.

17. **Use `gfx_backend="Vulkan"` on headless/containerized hosts.** Xvfb has
    no GPU passthrough, so Dolphin's default OpenGL backend falls back to
    llvmpipe software rasterization and burns ~6 CPU cores (Dolphin pegs
    ~590% CPU) rendering a framebuffer nobody is watching. Vulkan bypasses
    the GLX/X11 path entirely — the NVIDIA Vulkan ICD talks directly to the
    GPU device node and only uses Xvfb as a trivial presentation surface.
    On this container Vulkan dropped Dolphin CPU from ~590% → ~68% with
    GPU memory allocated and non-zero GPU utilization. Slippi Ishiiruka
    has Vulkan compiled in on Linux even though most community guides
    don't mention it (Windows users use D3D, macOS uses Metal). Do NOT
    use `gfx_backend="Null"` on Ishiiruka — libmelee rejects it with
    `ValueError('Null video requires mainline or ExiAI Ishiiruka.')` and
    the `ENABLE_HEADLESS` cmake flag is broken on this fork anyway
    (project-slippi/Ishiiruka#209).
