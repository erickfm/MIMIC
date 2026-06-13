# CLAUDE.md — Agent Orientation for MIMIC

## What this project is

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

- **Params:** ~19.95M (minimal-features) / ~19.99M (full-features).
- **Transformer:** d_model=512, 6 layers, 8 heads, block_size=1024.
- **Position encoding:** Shaw relative-position attention (`mimic`).
  RoPE variants (`mimic-rope*`) are deprecated — see Pitfalls.
- **Input:**
  - **Minimal features** (legacy, `--mimic-minimal-features`):
    Linear(166 → 512) from
    `[stage_emb(4) + 2*char_emb(12) + 2*action_emb(32) + gamestate(18) + controller(56)]`.
  - **Full features** (default — drop `--mimic-minimal-features` from the CLI):
    Linear(184 → 512) — same categorical embeddings, but the 18-dim
    `gamestate` becomes 36-dim (18 per player: 13 numeric + 5 flags).
- **Output heads (autoregressive with detach):**
  shoulder(3) → c_stick(9) → main_stick(37) → buttons(7).
- **Head hidden dim:** `input_dim // 2` (NOT a fixed 256 — each head has
  different hidden size).
- **Sequence length:** 180 frames (~3 seconds).
- **Dropout:** 0.2 (mimic / modern-relpos), 0.1 (mimic-xl and other scaled
  presets).

### Gamestate columns per player

**Minimal (9):** `percent, stock, facing, invulnerable, jumps_left,
on_ground, shield_strength, position_x, position_y` (exact HAL order
is preserved by a reindex in the encoder's minimal path).

**Full (18):** 13 numeric + 5 flags in native shard order:

    numeric[0-4]   : pos_x, pos_y, percent, stock, jumps_left
    numeric[5-9]   : speed_air_x_self, speed_ground_x_self,
                     speed_x_attack, speed_y_attack, speed_y_self
    numeric[10-11] : hitlag_left, hitstun_left
    numeric[12]    : shield_strength
    flags[0-4]     : on_ground, off_stage, facing, invulnerable,
                     moonwalkwarning

`invuln_left` and all 8 ECB corners are intentionally not in the schema:
`libmelee`'s `.slp` parser does NOT populate them. The attrs exist on
`PlayerState` but are never written by any parser (library-level dead
field), and ECB bytes live past where `console.py` reads the event
payload for the `.slp` format we work with (`console.py` silently falls
back to 0 if `event_bytes` is too short, which it is). They always
carried constant zero — an L1 input-gate run independently pruned them
to the sparsity floor. Some older fullfeat checkpoints (`puff xxl`,
`peach baseline`, `falco xxl continue`, `ice_climbers xxl`) were trained
on the 22-col layout and cannot load into the current 184→512 encoder;
minimal-features checkpoints still load correctly because the minimal
path is shard-width aware and reduces to the same 9-slot layout
internally.

### Per-feature normalization

Transform definitions in `tools/build_mimic_norm.py:MIMIC_TRANSFORMS`;
implementations in `mimic/features.py:mimic_normalize` (docstring lists
formulas); inference mirror in `tools/inference_utils.py:XFORM`.

In brief: `percent / stock / jumps_left / flags → normalize` (min-max
to [-1, +1]); `pos_x / pos_y → standardize` (z-score);
`shield_strength → invert_normalize` ("broken" is +1); velocities →
`tanh_scale` (preserves sign, saturates extremes); `hitlag_left →
linear_max(20)`; `hitstun_left → log_max(120)` (compresses long tail
while keeping low-value resolution). Old `mimic_norm.json` files
without `tanh_scale / linear_max / log_max` entries fall back to
z-score; regenerate norm to pick up the new transforms.

Controller one-hot: main_stick(37) + c_stick(9) + buttons(7) +
shoulder(3) = 56 dims.

### Preset variants (`mimic/model.py:MODEL_PRESETS`)

- **`mimic`** — production. d_model=512, 6 layers, 8 heads, relpos
  attention, LN, GELU. ~20M params.
- **`mimic-xl`** — width+FFN scale-up, SwiGLU, ~44M params.
  `--num-layers 10` makes it ~73M. Not currently in production.
- **`modern*` and `mimic-xl-rms`** — GQA + RMSNorm + SwiGLU
  variants; tied or slightly below `mimic` at puff scale; parked.
- **`mimic-rope*` family** — RoPE / xPos / ALiBi / FlexAttention
  position-encoding experiments. **Underperforms relpos. Don't use
  in production.**
- **Generic presets** (`tiny`, `small`, `medium`, `base`, `deep`,
  `shallow`, `wide-shallow`, `xlarge`, `xxlarge`, `huge`, `giant`)
  — legacy, no mimic-specific keys. **Avoid for `--mimic-mode`** —
  they don't set `max_seq_len` so the dataset silently falls back
  to 60 frames.

### Learned input gate (diagnostic, opt-in)

`--input-gate-l1 <λ>` (typical 0.01) adds a per-input sigmoid gate
to `MimicFlatEncoder` and an L1 sparsity penalty. End of training
writes `checkpoints/{run_name}_gate_report.json` ranking all input
scalars by learned gate value. λ=0.01 costs ~0 val-loss. Off by
default; not a production regularizer.

## Stats files (HAL legacy)

Only relevant when running `tools/run_hal_model.py`. HAL has two
`stats.json` files (`hal/data/` and `hal/checkpoints/`); the
Preprocessor always loads `hal/checkpoints/stats.json` (Fox training
subset, p1_percent max 236) regardless of `override_stats_path` —
the override mechanism changes the config field but not the
resolved path. Using the wrong file shifts every normalized
percent and the model plays terribly.

## Shard alignment

melee-py's `console.step()` returns **post-frame** game state (action,
position, percent — after the engine processes inputs) alongside
**pre-frame** controller inputs (the buttons themselves). So the game
state at frame `i` already reflects `button[i]` — e.g., `action=KNEE_BEND`
appears on the same frame as `button=JUMP`. If you train on the naïve
alignment, the model can read the answer from `self_action`.

**v2 shards** (`data/<char>_v2`) fix this by shifting targets forward
by one frame: `target[i] = buttons[i+1]`. The model sees the current
game state and predicts what to press NEXT. This matches inference
exactly.

**Rule: do NOT use `--controller-offset` or `--reaction-delay 1` with
v2 shards.** The alignment is already correct; adding an offset
double-shifts the data.

**Old shards** (`data/fox_hal_full`, `data/fox_hal_match_shards`) have
the leak. With those shards, use `--reaction-delay 1` to achieve the
same effect at dataloader time (this is what HAL does).

Val loss is not comparable across shard versions: v2 shards produce
higher val loss because the model can no longer cheat via
action→button memorization. A v2 val loss of ~1.0 can correspond to
better gameplay than 0.74 on old shards.

## Training

### Current best command (v2 shards, full features)

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

**Full features are the default** — omit `--mimic-minimal-features`
so the encoder exposes the full 13-numeric + 5-flag per-player
gamestate. Worth ~1.5–3.5% val-loss reduction over the minimal path
at no wall-clock cost. Old checkpoints trained with the flag still
load unchanged (the flag flows through the pickled config and the
minimal-path behavior is bit-identical for back-compat).

**`--self-inputs` is required even on v2 shards.** Without it the
encoder has no controller-history input at all and val loss climbs
~3.5× (main-stick F1 drops from ~57% to ~15%). `--controller-offset`
is still not needed (v2 alignment is baked into the shard), but
`--self-inputs` is critical.

**Use `--model mimic` (Shaw relpos).** `mimic-rope*` presets are
deprecated — see Pitfalls.

Single-GPU variant: swap `torchrun --nproc_per_node=2` for `python3`
and use `--batch-size 64 --grad-accum-steps 8` to keep effective batch
at 512.

The legacy `--hal-*` flags and `--model hal` / `--encoder hal_flat`
names still work as aliases; checkpoints from before the HAL→MIMIC
rename load unchanged.

### Legacy leaked-shards command

For reproducing pre-v2 runs against `data/fox_hal_*`: same as above
but add `--mimic-minimal-features --controller-offset --reaction-delay 1`.
Rarely needed.

### BF16 + relpos stability

BF16 AMP and torch.compile are enabled by default. The Shaw relpos
attention computes `Q@K^T + S_rel` manually, which overflows in BF16
due to limited mantissa precision (7 bits). The fix is an
`autocast(enabled=False)` block around the attention math in
`CausalSelfAttentionRelPos.forward()`, keeping Q/K/Er in FP32 for the
dot products while the rest of the model (FFN, embeddings, heads)
stays in BF16. Do NOT use `GradScaler` with BF16 — it's only needed
for FP16. For bit-exact reproduction of pre-AMP runs, use
`--no-amp --no-compile`.

### Effective batch size (`max_steps`)

`train.py` computes `max_steps = max_samples // effective_batch_size`,
where `effective_batch_size = local_batch * n_gpus * grad_accum_steps`.
Do not reintroduce a divide-by-local-batch shortcut: on 8 GPUs that
would train 8× the requested sample budget and severely overfit.

### Model preset → seq_len gating

`train.py` sets the global `SEQUENCE_LENGTH` for the dataset by
reading `MODEL_PRESETS[model_preset]["max_seq_len"]`. The `mimic`
preset sets 256; presets without `max_seq_len` (the legacy generic
presets — `tiny`, `small`, etc.) fall back to the module default
`SEQUENCE_LENGTH = 60`, a 4.3× reduction in temporal context per
sample.

**Do not add `if model_preset == "X"` gates in this lookup.** Read
the value off the preset dict so any new preset/alias automatically
picks it up. String checks silently fall through for renamed or
aliased presets.

### Do NOT run inference while training on the same GPU

GPU contention causes frame drops in Dolphin, making the model appear
unresponsive and miss inputs. Always suspend or kill training before
running inference.

## Inference

### Running MIMIC checkpoints

Use `tools/play.py` for bot-vs-CPU and bot-vs-bot matches (one
Dolphin, N back-to-back matches, optional win-rate JSON report) and
`tools/play_netplay.py` for Slippi Online Direct Connect. Both
import from `tools/inference_utils.py`, which holds the shared decode
pipeline (`load_mimic_model`, `load_inference_context`, `build_frame`,
`build_frame_p2`, `PlayerState`, `decode_and_press`). Any new
inference entry point should import from there rather than
reimplement — the L-button / TRIG decode (see Pitfalls #9) lives in
one place on purpose.

### Running HAL's original checkpoint (rare)

`tools/run_hal_model.py` is a from-scratch reimplementation of HAL's
5-class inference, used only for Eric Gu's original HAL weights at
`/home/erick/projects/hal/checkpoints/000005242880.pt`. It does NOT
share `inference_utils` with the MIMIC play tools. If our
reimplementation breaks, HAL's own code always works:

```bash
cd /home/erick/projects/hal
python3 -m hal.eval.play --artifact_dir checkpoints --character FOX
```

This requires `hal/local_paths.py` to have correct paths for the
emulator and ISO. The `MAC_*` path aliases are for this purpose.

## Data

### Directories

| Directory | Contents | Target alignment | Status |
|-----------|----------|-----------------|--------|
| `data/fox_v2` | ~17K Fox games, 800MB shards, quality-filtered, next-frame targets | `target[i] = buttons[i+1]` (clean) | **Active — use with rd=0, no offset** |
| `data/falco_v2` | ~9K Falco games, same format | clean | Active |
| `data/cptfalcon_v2` | ~9K CptFalcon games, same format | clean | Active |
| `data/luigi_v2` | ~2K Luigi games, same format | clean | Active |
| `data/fox_hal_full` | ~10K Fox games, 800MB shards, quality-filtered | `target[i] = buttons[i]` (leaked) | Legacy — use with rd=1 |
| `data/fox_hal_800m` | 7,600 Fox games, 800MB shards | leaked | Legacy |
| `data/fox_hal_local` | 7,600 Fox games, 3.8GB shards | leaked | Legacy |

Legacy dirs keep `hal_*` in their names because nothing references
them from the active code path anymore — they're frozen. New data
dirs always drop the `hal_` prefix.

Use 800MB shards with `mmap=True` in DataLoader for optimal
throughput. `tools/reshard.py` can split large shards:
`python tools/reshard.py --src <dir> --dst <dir> --target-mb 800`.

### Game quality filters

`tools/slp_to_shards.py` filters replays the same way HAL's
`process_replays.py` does:

- Minimum 1,500 frames (~25 seconds) — rejects disconnects and junk
- Damage check — both players must take at least some damage
- Completion check — one player must lose all stocks (no ragequits)

Existing `fox_hal_local` shards were built without these filters and
contain low-quality games. Rebuild from .slp source to get clean
data.

### Button encoding (7-class priority collapse)

Current shards (v2, 7-class) collapse multi-hot buttons to a single
label **statelessly per frame** via Melee's input-resolution rules
(`mimic/features.py:_collapse_buttons_7class_np`): B overrides
everything → A+TRIG (no B) → A_TRIG → else highest priority of
Z > A ≈ TRIG > JUMP → NONE. No dependence on the previous frame's
buttons. X and Y both map to JUMP; L and R both map to TRIG.

**Legacy 5-class shards only** (`btns_single`, HAL-compat): encoded
with HAL's early-release state machine
(`convert_multi_hot_to_one_hot_early_release`): when buttons change
but nothing new is pressed (partial release), the label is
`NO_BUTTON` (4). Do not "keep the surviving held button" — that
diverges from the training target. The 7-class encoding replaced
this in commit `ed87d7c` (2026-04-11); it applies only when working
with old 5-combo data.

### HuggingFace datasets

Two sources of raw replays. Ranked is canonical for new training.

- **`erickfm/melee-ranked-replays`** (canonical). Ranked Slippi
  replays stored as `.tar.gz` per (character, rank_pair, archive):
  `{CHAR}/{CHAR}_{rank_pair}_a{N}.tar.gz`. Rank pairs are
  `master-master`, `master-diamond`, `master-platinum`, `diamond-*`,
  `platinum-*`. Higher-rank token first (M>D>P). Per-char
  master-tier training pulls all three `master-*` pairs
  (master-master has both players master; master-diamond/platinum
  mixes in games where the other player is sub-master and we can't
  tell from the .slp which port is master — accepted as a
  data-quality/quantity tradeoff).
- **`erickfm/slippi-public-dataset-v3.7`** — legacy. 95K replays by
  character; Fox folder has 45,854 .slp. `tools/download_fox.py` is
  hardcoded to this source.

### Ranked dataset pipeline (`tools/shard_and_upload_ranked.py`)

Processes one archive end-to-end: `7z`/`unzip` → header-parse via
`peppi_py.read_slippi(skip_frames=True)` (parallel ProcessPoolExecutor)
→ bucket by (character, rank_pair) → tar+upload per bucket with skip
on HF-already-present + 5-attempt exponential backoff + 600s socket
timeout.

Filename convention `{rank1}-{rank2}-{hash}.slp` (ranks higher-first,
M > D > P). HF layout `{CHAR}/{CHAR}_{rank_pair}_a{N}.tar.gz` plus
`metadata/metadata_a{N}.json`. Resumable: re-runs skip uploaded
tarballs; if `extracted/` still exists from a prior run, pass
`--skip-extract`.

Character mapping notes: Zelda+Sheik collapsed to bucket
`ZELDA_SHEIK`; Popo+Nana to `ICE_CLIMBERS`; debug chars
(WIREFRAME/GIGA/SANDBAG) rejected.

### Building MIMIC-normalized shards

Requires a metadata directory with: `norm_stats.json`, `cat_maps.json`,
`stick_clusters.json`, `controller_combos.json`, and `mimic_norm.json`.
The `controller_combos.json` MUST have 7 combos (A, B, Z, Jump, TRIG,
A_TRIG, None) for the 7-class button head. The old 5-combo version
(A, B, Jump, Z, None — HAL's scheme) still works via backcompat but
cannot represent airdodge / wavedash / L-cancel.

### Retraining a character from scratch

Full pipeline (download master-* ranked replays from HF → extract →
re-shard with existing `mimic_norm.json` → train using the current
best command) is wired into `tools/run_all_chars.sh` (remote 2×5090
box) and `tools/run_local_chars.sh` (local single-4090 box; builds
fresh per-char metadata, prefetches char N+1 during char N's
training, uploads in the background). Per-char wall time on 2×RTX
5090 is ~1.5–2.5 hr; on the local 4090 ~2 hr of GPU time per char
(bs 256 × grad-accum 2, ~4.5 step/s).

**`hf download --include` gotcha:** repeating the `--include` flag
overwrites earlier patterns (argparse `nargs='*'`) — all glob
patterns must follow a SINGLE `--include`. `run_all_chars.sh` has
the repeated-flag bug and silently fetched only `master-platinum`
(6 of 18 tarballs per char); the 2026-06-12 local runs initially
trained on that ⅓ data and early-stopped at 1.6k–7.8k steps before
the fix. Any model whose data was pulled through the buggy path
should be considered platinum-only until verified.

Character index + HF bucket map (Zelda/Sheik bucket as `ZELDA_SHEIK`;
Jigglypuff bucket is the full name):

| char | HF bucket | idx |
|---|---|---|
| fox | `FOX` | 1 |
| falco | `FALCO` | 22 |
| marth | `MARTH` | 18 |
| sheik | `ZELDA_SHEIK` | 7 |
| cptfalcon | `CPTFALCON` | 2 |
| puff | `JIGGLYPUFF` | 15 |
| luigi | `LUIGI` | 17 |

## Checkpoints

**HAL's only actual checkpoint**:
`/home/erick/projects/hal/checkpoints/000005242880.pt` (101 MB, DDP-prefixed
state dict). Eric Gu's best at 5.2M samples. Loaded only by
`tools/run_hal_model.py`. Not on any MIMIC code path.

**MIMIC checkpoints** live in `checkpoints/`. Naming:
`{char}-{YYYYMMDD}-{descriptor}-{steps}k.pt`. Descriptor is a
free-form tag (`relpos`, `rope`, `fullfeat`, `gate01`, `xxl`, etc).
Set `--run-name` to `{char}-{YYYYMMDD}-{descriptor}`; step suffix is
appended when promoting the `_best.pt`.

Current per-character production:
- `puff-20260419-mimic-fullfeat-gate01-33k.pt` (val 0.66)
- `falco-20260412-relpos-28k.pt` (val 0.74)
- `cptfalcon-20260412-relpos-27k.pt` (val 0.71)
- `luigi-20260412-relpos-5k.pt` (val ~1.0, early-stopped)

### Promotion policy

When a finished wandb run beats current val for its character: pull
the `.pt`, rename to the convention, upload to `erickfm/MIMIC` on HF,
update `tools/discord_bot.py:CHARACTERS`. **Skip names starting with
`SWEEP-`/`DBG-`/`DEBUG-`/`FIX-`/`BENCH-`** — infra/debug runs aren't
production candidates regardless of val. **Always confirm with user
before pushing** — never auto-promote.

### Discord-bot (re)start: audit + orphan sweep

Before `bot.run()`:
1. **Promotion audit.** List `erickfm/MIMIC` via `HfApi.list_repo_files`,
   read each char's `metadata.json` (`run_name`/`global_step`/`val_loss`),
   compare to the `CHARACTERS` dict in `tools/discord_bot.py`. Flag any
   HF entry newer than what's wired up + any HF char the bot doesn't
   know about (needs new alias). Report to user, get approval, then
   `snapshot_download` + copy into `checkpoints/` + edit the dict.
2. **Orphan sweep.** `_cleanup_orphan_processes()` already runs before
   `bot.run()` and SIGTERMs (then SIGKILLs) leftover `play_netplay.py`
   and our-path `dolphin-emu` processes. Without it they reparent to
   init and burn CPU/GPU indefinitely (seen: 10+ hr Dolphin at 85% CPU
   writing no replays). Don't disable.

Secondary supplementary check: wandb `erickfm/MIMIC` for finished
non-debug runs newer than HF — useful to remind the user "you trained
X but didn't upload it." The `.pt` itself must come from the GPU box
(wandb doesn't store checkpoints) or a retrain.

## File map

### Core
- `train.py` — training loop (DDP, gradient accumulation, mimic_mode, cosine LR).
- `mimic/model.py` — model architecture (FramePredictor, mimic presets, relpos+other attention variants).
- `mimic/dataset.py` — StreamingMeleeDataset (per-game + pre-windowed shards).
- `mimic/frame_encoder.py` — MimicFlatEncoder; honors `mimic_minimal_features` (slices the shard numeric tensor to 9 cols in HAL order) vs full (13 numeric + 5 flags). Also hosts the optional `use_input_gate` L1 diagnostic.
- `mimic/features.py` — feature schema + normalization. `numeric_state(full=True)` returns 13 cols.
- The old root-level `eval.py` / `inference.py` were removed (2026-06); inference lives in `tools/play.py` / `tools/play_netplay.py`.

### Tools

**Inference (local):**
- `tools/play.py` — Run a MIMIC checkpoint vs a CPU (`--opponent cpu:9`) or a second checkpoint (`--opponent <ckpt>`) in one Dolphin instance: N back-to-back matches, `--n-matches 1` for a watchable game, optional win-rate JSON via `--out`. Uses shared `inference_utils.decode_and_press`.
- `tools/run_hal_model.py` — Reimplementation of HAL's 5-class inference. Loads HAL checkpoints. Structurally can't wavedash (no TRIG class).

**Inference (online, Slippi netplay):**
- `tools/play_netplay.py` — Joins a Slippi Direct Connect lobby and plays N
  back-to-back matches in one Dolphin process (persistent-session mode —
  don't reintroduce per-match spawning, the relaunch dead air was the UX bug
  this fixed). Stdout protocol: per-match block of `MATCH_START` /
  `RESULT: win|loss|draw|disconnect|no-opponent|timeout|failed` / `SCORE` /
  `STAGE` / `REPLAY`, plus a single `SESSION_END:` on exit. STOP polling on
  stdin via non-blocking `select.select`.
- `tools/discord_bot.py` — Discord front-end. Prefix commands `!play`,
  `!queue`, `!cancel`, `!info`, `!reload`, plus one `!<character>` shortcut
  per loaded character (registered dynamically via
  `_register_char_shortcuts`; rebuilt on `!reload` so newly-uploaded chars
  are callable without a bot restart). FIFO queue via `asyncio.Queue`,
  single-session. `!reload` uses a fingerprint tuple
  `(path, run_name, global_step, val_loss)` to detect retrains that reuse
  `hf_checkpoints/{char}/model.pt` paths.

**Multi-instance** on one machine is supported via N `.env` files (distinct
Slippi accounts, `SLIPPI_HOME` dirs, replay dirs). ~3–5 concurrent
sessions fit on a 24 GB GPU + 8-core CPU before 60fps deadlines tighten.

**Inference (shared)**: `tools/inference_utils.py` —
`load_mimic_model`, `load_inference_context`, `build_frame`,
`build_frame_p2`, `PlayerState`, `decode_and_press`. Single place
where the L-button / TRIG decode lives — new inference entry points
must import from here, not reimplement. Produces the full 13-numeric +
5-flag tensor; reads `PlayerState.speed_*` / `.hitlag_left` /
`.hitstun_frames_left`; normalizes via `mimic_norm.json` (z-score
fallback for entries missing newer transform keys). Both minimal and
fullfeat checkpoints work — minimal encoder slices internally.

**Diagnostics**: `tools/inspect_frame.py` (per-frame I/O dump),
`tools/extract_wavedashes.py` (wavedash-only windows for overfit
checks), `tools/validate_checkpoint.py` (per-head CE on val),
`tools/diagnose.py` (train-vs-inference tensor compare).

**Data**: `tools/slp_to_shards.py` (.slp → v2 .pt shards with
`target[i] = buttons[i+1]` alignment); `tools/shard_and_upload_ranked.py`
(ranked .slp archive → HF tarballs); `tools/split_by_character.py`.

## Porting slippistats logic (live-streaming variants)

Several VR modules under `rlvr/online/vr/` need streaming-state-
machine versions of logic that already exists in the slippistats
library (`~/.local/lib/python3.12/site-packages/slippistats/`).
slippistats is **batch on parsed .slp**; our actor needs
**streaming on live `libmelee.PlayerState`**. Same predicates,
different data shape + interface.

The streaming-slippistats module (`rlvr/online/slippi_stream.py`) is
the canonical home for these ports — the `is_*`/`in_punish_state`
predicates plus the `MoveCounter` / `ComboTracker` / `TechTracker` /
`RecoveryTracker` state machines, consumed by the VR modules in
`rlvr/online/vr/`. Its combo logic went through ~6 rounds of "I caught
another missed case" (in the now-retired `combo_extend_online.py`,
since superseded by `slippi_stream.py` + the `combo_length` VR). Each
round caught a real bug. Pattern: the obvious predicates get ported on
the first pass, the specialized ones get missed.

**Default to paranoid faithfulness when porting** any slippistats
logic. Concretely:

1. Read the WHOLE function in `slippistats/stats/*` you're
   porting (start to end), not just the parts that look relevant.
2. Enumerate every conditional branch (`if X: ...`) and check
   that the port has an equivalent — including the seemingly
   incidental ones like `player_did_lose_stock` mid-combo.
3. Enumerate every state predicate referenced (`is_X(...)`) and
   verify the range constants match libmelee Action enum values
   (don't rely on memory or assumed ranges).
4. Compare start / keep-alive / termination sets line-by-line.
5. Don't add safety nets / hard caps / extra filters that
   slippistats DOESN'T have unless the reason is documented. They
   handled the edge cases we'd think to add.

Specific gotchas already found while porting the combo logic into
`slippi_stream.py` (all are slippistats behaviors a naive port misses):

- **THROWN range (239-243)** is separate from CAPTURE (223-232).
  Throws are NOT in the "in opp's grab" set on their own — a
  grab→throw combo gets fragmented without explicit THROWN.
- **DYING (0-10)** must be in the keep-alive set so the kill blow
  stays in-episode long enough for the stock decrement to register
  before the K-gap closes.
- **`player_did_lose_stock`** (bot dies mid-combo) is a termination
  condition alongside opp-died and K-gap.
- **Command grabs (266-304, 327-338)** are separate from regular
  grabs and must be in start + keep-alive predicates.
- **`hitlag_left > 0`** is a keep-alive condition independent of
  `hitstun_frames_left`. Both are populated by libmelee.
- **`COMBO_LENIENCY = 45 frames`**, not 20 or 30. Tech-chase /
  re-grab / wait-DI sequences need the full 45.
- **`action_changed_since_hit`** uses the BOT's action state, not
  opp's. Move counting requires tracking self.action + action_frame.
- **No frame-count hard cap** in slippistats. The three termination
  conditions (opp died / K-gap / bot died) are sufficient.
- **The `stock` counter decrements ~1.5 s *after* the kill** — the KO'd
  character runs the DEAD animation (action 0-10) first. A fixed
  backward look-back window from the stock-decrement frame lands inside
  those DEAD frames and misses the killing hit. This broke
  `stock_delta`'s SD-gate — every kill scored as an opponent
  self-destruct, inverting the reward. Gate hit-recency *statefully*
  (`OppHitRecencyTracker`: a decay counter that DEAD frames pass through
  untouched), not with a backward scan. `low_percent_kill` still has
  this bug in both its SD-gate and its death-percent peak.

When porting new slippistats logic (edgeguard, shield-escape,
pressure, etc.), file a research note specifically calling out
what was ported and what was intentionally skipped, with file
references to the slippistats functions used. Past notes:
`docs/research-notes-2026-05-14b.md`.

## Pitfalls for agents

1. **`tools/run_hal_model.py` loads actual HAL weights.** MIMIC
   checkpoints go through `tools/play.py` / `play_netplay.py`.
   `run_hal_model.py` is the
   reference-implementation path for Eric Gu's original HAL
   checkpoints and is not used by any MIMIC production code.

2. **Don't trust research notes as current truth.** Always verify
   against code. See the Research Notes section below.

3. **Don't run inference while training on the same GPU.** Frame
   drops make gameplay look broken when the model is fine.

4. **`max_samples` is total, not per-GPU.** `train.py` divides by
   effective batch size (`local_batch * n_gpus * grad_accum`) to
   compute `max_steps`. Don't reintroduce a divide-by-local-batch
   shortcut — on 8 GPUs that 8×-overtrains.

5. **Don't mix normalization schemes.** `mimic_mode` training needs
   `mimic_norm.json` + MIMIC controller combos (7-combo current,
   5-combo legacy). Old data like `ranked_fox` uses old normalization
   with 32 combos — incompatible.

6. **Don't hardcode head hidden dims as 256.** The autoregressive
   heads use `input_dim // 2`, which varies per head (256, 257, 262,
   280).

7. **`sorted()` player dicts.** melee-py's `gamestate.players` dict
   order is not guaranteed to match port order. Always `sorted()`.

8. **Use `blocking_input=True` for inference.** Dolphin waits for
   controller input before advancing each frame. Without it, slow
   model inference causes frame drops (the game advances without
   receiving input). In head-to-head, non-blocking mode systematically
   disadvantages whichever model's inputs are flushed second.

9. **TRIG (L/R) must call `press_button`, not just `press_shoulder`.**
   Melee's shoulder events split on analog vs digital:
   - **Shield**: analog threshold (any shoulder value above ~0.3).
   - **L-cancel**: analog threshold, rising edge during the L-cancel window.
   - **Tech**: digital L/R press.
   - **Airdodge**: digital L/R press.
   - **Wavedash**: airdodge into ground → digital press required.

   So `press_shoulder(BUTTON_L, 1.0)` alone is enough for shield +
   L-cancel, but tech / airdodge / wavedash need
   `press_button(BUTTON_L)`. The 7-class button head's TRIG (class 4)
   and A_TRIG (class 5) classes call `ctrl.press_button(BUTTON_L)` to
   cover all four cases at once. `tools/inference_utils.py:decode_and_press`
   is the single place this lives — new inference entry points must
   import from it rather than reimplement. HAL's 5-class button head
   has no TRIG class, so HAL-lineage bots are structurally incapable
   of teching, airdodging, or wavedashing.

10. **Button encoding is single-label.** The 5-class button head (A,
    B, Jump, Z, None) cannot represent two simultaneous action
    buttons; the 7-class head adds TRIG + A_TRIG for the one emergent
    combo that matters (airdodge/wavedash + an A-attack shield-grab
    interaction). Multi-button overlaps (2.65% of frames) are
    collapsed via a stateless per-frame priority cascade mirroring
    Melee's input resolution (B overrides all; A+TRIG → A_TRIG; else
    Z > A ≈ TRIG > JUMP) — see "Button encoding" in the Data section.
    Early-release encoding is legacy 5-class only. Shoulder+button
    combos ARE representable since shoulder is a separate head.

11. **RoPE (`mimic-rope*`) presets are deprecated.** They underperform
    the relpos baseline — the bug is in the positional-encoding path
    itself, not the training recipe. Default to `--model mimic` (Shaw
    relpos). Use a RoPE preset only when specifically testing it and
    you know what you're looking at.

12. **Full features are the default.** Omit `--mimic-minimal-features`
    unless deliberately reproducing a minimal-path baseline. The
    minimal path is bit-identical for back-compat with old
    checkpoints; the full path exposes 13 numeric + 5 flags per
    player and is worth ~1.5–3.5% val-loss reduction at no wall-clock
    cost.

13. **Discord bot portability: keep paths relative in `.env`.** The
    bot's `.env` uses relative paths (`./emulator/...`, `./melee.iso`,
    `./slippi_home`) that `_resolve_path` in `tools/discord_bot.py`
    converts to absolute against the repo root at runtime. This makes
    the repo `scp`-able to any machine that has run `setup.sh`. Don't
    hardcode absolute paths.

14. **Slippi credentials live at `./slippi_home/Slippi/user.json`**
    (gitignored). Not at `~/.config/SlippiOnline/Slippi/user.json` —
    libmelee is pointed at the bundled dir explicitly via
    `dolphin_home_path=SLIPPI_HOME` in `tools/play_netplay.py`. Place
    `user.json` in the repo so uploading the repo to a new machine
    carries the bot's Slippi login. Never commit `slippi_home/` — it
    contains the bot's `playKey`.

15. **Dolphin needs runtime shared libraries.** The AppImage-extracted
    `dolphin-emu` binary links against `libasound2`, `libusb-1.0-0`,
    `libgtk-3-0`, `libbluetooth3`, `libhidapi-hidraw0`, and friends.
    Missing any of them makes the binary exit 127, which libmelee
    surfaces as `RuntimeError: Unexpected return code 127 from
    dolphin` inside `Console.__init__` — `play_netplay.py` then exits
    1 with an empty `RESULT:` line and the Discord bot reports
    `result=failed score=`. `setup.sh` installs the full list; on
    existing machines run `ldd emulator/squashfs-root/usr/bin/dolphin-emu
    | grep 'not found'` to see what's missing.

16. **Setup Xvfb for headless machines.** Dolphin crashes at startup
    with "Unable to initialize GTK+, is DISPLAY set properly?" if no
    display server is available. `setup.sh` installs and starts Xvfb
    on `:99` and adds `export DISPLAY=:99` to `~/.bashrc`. On
    existing machines, check `DISPLAY` is set in the environment the
    Discord bot / `play_netplay.py` inherits.

17. **Use `gfx_backend="Vulkan"` on headless/containerized hosts.**
    Xvfb has no GPU passthrough, so Dolphin's default OpenGL backend
    falls back to llvmpipe software rasterization and burns ~6 CPU
    cores (~590% CPU) rendering a framebuffer nobody is watching.
    Vulkan bypasses the GLX/X11 path entirely — the NVIDIA Vulkan ICD
    talks directly to the GPU device node and only uses Xvfb as a
    trivial presentation surface. In this container Vulkan dropped
    Dolphin CPU from ~590% → ~68% with GPU memory allocated and
    non-zero GPU utilization. Slippi Ishiiruka has Vulkan compiled in
    on Linux even though most community guides don't mention it
    (Windows uses D3D, macOS uses Metal). Do NOT use
    `gfx_backend="Null"` on Ishiiruka — libmelee rejects it with
    `ValueError('Null video requires mainline or ExiAI Ishiiruka.')`
    and the `ENABLE_HEADLESS` cmake flag is broken on this fork
    anyway (project-slippi/Ishiiruka#209).

18. **FFW (`emulator_ffw/` + `--use-exi-inputs --enable-ffw`) is not
    gameplay-faithful.** It produces matches ~4× shorter than realtime
    from the same models (h2h: ~2,400 vs ~9,700 frames/match) — the bots
    lose stocks ~4× faster, likely the EXI input path mistiming
    controller inputs under fast-forward. Do NOT use FFW for RL training
    or h2h eval; run realtime (`emulator/`). A win-rate that looks
    consistent FFW-vs-realtime is a false positive — win-rate survives
    *symmetric* degradation (both bots equally hobbled); compare **match
    length** (frame count) to detect emulator-fidelity problems. The
    separate inter-update PPO-pause disconnect (`EnetDisconnected`, enet
    unserviced > ~20 s) is fixed by the `dolphin_actor.py` keepalive
    thread, but that fix is moot while FFW gameplay is unfaithful. See
    `docs/research-notes-2026-05-18.md`.

19. **Netplay headless: use the *mainline* Slippi Dolphin + `gfx_backend="Null"`,
    NOT the bundled Ishiiruka.** libmelee's `choose_direct_online` only selects
    Direct when `menu_selection` is 2/3 (mainline's online-submenu layout); the
    Ishiiruka 3.5.2 build reports `sel=6` there, so the bot stalls forever at
    `MAIN_MENU/ONLINE_PLAY_SUBMENU` and never goes online (zero matchmaking
    traffic). `setup.sh` now also fetches mainline into `emulator_mainline/` and
    `DOLPHIN_PATH` defaults to it. Mainline supports `Null` video → no rendering,
    so #16/#17 (Xvfb/Vulkan) don't apply to the netplay bot. `play_netplay.py`
    also forces the `fork` mp start-method (macOS/Windows `spawn` re-imports the
    unguarded script and double-runs it). And the host must pass **inbound P2P
    UDP** — RunPod-style container pods drop it (STUN falsely reports "cone");
    use a real VM (e.g. GCP).

## Research notes

The chronological dev journal lives in `docs/research-notes-*.md`,
one file per day (sometimes with `b` / `c` / `d` suffixes when
multiple notes land on the same date). They capture experiment
results, design decisions, and debug stories in the present tense of
when they were written — they're the historical reference for "why
is X the way it is?" Older notes live in `docs/archive/`.

Ops setup docs for the Discord bot (Slippi account, `.env`, install
flow, troubleshooting) live in `docs/discord-bot-setup.md` and are
linked from `README.md`.

**Research Notes Warning.** The notes record what was believed true
at each point in time. Several claims were later found wrong — for
example:

- "HAL's val loss is stable" — actually HAL overfits too (val rises
  from 0.744 to 0.802 after 5.2M samples).
- "Architecture: 26,274,803 params" — actually ~19,950,000 params.
- "HAL uses `hal/data/stats.json` for inference" — the Preprocessor
  actually loads `checkpoints/stats.json`.
- Various "this matches HAL" claims that later turned out to have
  subtle differences.

The notes are still valuable for understanding the project's
evolution and the reasoning behind decisions. Just don't treat
specific numbers or "verified" claims as current truth without
checking the code.

## The HAL repo (`/home/erick/projects/hal`)

Eric Gu's original codebase. Only need to touch this for `hal/eval/play.py`
(ground-truth inference fallback when MIMIC reimplementation breaks). The
canonical checkpoint is `hal/checkpoints/000005242880.pt`; **use
`hal/checkpoints/stats.json`** (27M-frame Fox stats), not
`hal/data/stats.json` (222M multi-char), per "Stats files" section above.

## Environment

### Remote GPU

| Machine | Host | Port | User | GPU | Storage | Status |
|---------|------|------|------|-----|---------|--------|
| A | 194.14.47.19 | 22877 | root | RTX 5090 | 3 TB SSD | Active |

```bash
ssh -p 22877 root@194.14.47.19   # Machine A
```
