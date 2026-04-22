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

Defined by `tools/build_mimic_norm.py:MIMIC_TRANSFORMS` and applied by
`mimic/features.py:mimic_normalize` / `mimic_normalize_array` at shard
time; mirrored in `tools/inference_utils.py:XFORM` for live inference.

| transform | formula | used for |
|---|---|---|
| `normalize` | `2(x-min)/(max-min) - 1` → [-1, +1] | percent, stock, jumps_left, facing, invulnerable, on_ground |
| `standardize` | `(x - mean) / std` | pos_x, pos_y |
| `invert_normalize` | `2(max-x)/(max-min) - 1` | shield_strength (so "shield broken" is +1) |
| `tanh_scale` | `tanh(x / scale)` | 5 velocities (scale=5 for self, scale=10 for attack) |
| `linear_max` | `x / max` | hitlag_left (max=20) |
| `log_max` | `log1p(clamp(x,0,max)) / log1p(max)` | hitstun_left (max=120) |

The bottom three transforms target zero-inflated / heavy-tailed columns
that z-score handled poorly: velocities are signed heavy-tailed (sign
is load-bearing, tail dominates std and wastes dynamic range on
typical-motion frames); hitlag is bounded 0–20 where the in-hitlag
binary is what matters; hitstun has a long tail where 60/80/100 are
all "still combo'd" but low-value resolution (0 / 5 / 15) must stay
crisp. `tanh` preserves sign and saturates extremes; `x/max` works for
hitlag's bounded range; `log_max` compresses the hitstun tail while
keeping low-value resolution.

**Backward compat:** old checkpoints' `mimic_norm.json` files predate
the `tanh_scale` / `linear_max` / `log_max` entries, so at inference
the new XFORM entries are only active if the loaded char's
`mimic_norm` has them — old chars fall back to the z-score path
automatically. Retraining with the new transforms requires
regenerating `mimic_norm.json` (delete the file; shard pipeline
rebuilds it).

The controller is a 56-dim one-hot: main_stick(37) + c_stick(9) +
buttons(7) + shoulder(3).

### Preset variants (in `mimic/model.py:MODEL_PRESETS`)

- `mimic` — canonical (relpos, LN, GELU, full attention). Production.
- `mimic-rope`, `mimic-rope-lt`, `mimic-rope-lf`, `mimic-rope-deep`,
  `mimic-xpos`, `mimic-selrope`, `mimic-ropenope`, `mimic-xpos-64`,
  `mimic-flex`, `mimic-ropeflex`, `mimic-learned` — position-encoding
  experiments. RoPE family underperforms the relpos baseline. Not used
  in production.
- `mimic-xl` — width+FFN scale-up of `mimic`. d_model=768, nhead=12,
  num_layers=6, dim_feedforward=3072, dropout=0.1, SwiGLU FFN, relpos.
  ~44M params. `--num-layers 10` gives a ~73M deeper variant.
- `mimic-xl-rms` — `mimic-xl` plus RMSNorm. Did not meaningfully
  differ from LN variant at 10-layer depth.
- `modern-relpos`, `modern-relpos-gelu`, `modern` — `mimic`-sized
  (~20M) with GQA (n_kv_heads=2) + SwiGLU + RMSNorm. At puff scale
  none of these beat the plain `mimic` recipe, tying to within 0.001
  val-loss across all three runs.
- `tiny`, `small`, `medium`, `base`, `deep`, `shallow`,
  `wide-shallow`, `xlarge`, `xxlarge`, `huge`, `giant` — legacy
  generic presets without mimic-specific keys. **Avoid for mimic_mode
  training** — they don't set `max_seq_len`, which triggers the
  seq_len-fallback-to-60 path (see Training).

### Optional: learned input gate (feature-importance diagnostic)

`ModelConfig.use_input_gate: bool` (default False). Enable per-run with
`--input-gate-l1 <lambda>` (typical: 0.01). When on,
`MimicFlatEncoder` adds a per-input-column sigmoid gate on the final
projection, and the training loop adds `lambda * sigmoid(gate).mean()`
to the loss as an L1 sparsity penalty. End-of-training writes
`checkpoints/{run_name}_gate_report.json` ranking every input scalar
by its learned gate value (0 = model pruned it, 1 = model kept it).

Primarily a diagnostic tool, not a production regularizer — at λ=0.01
it costs ~0 val-loss (and in one measurement on puff actually helped
slightly, likely via small implicit regularization). Leave
`--input-gate-l1` unset (default 0) for standard production runs.

## Stats files (HAL legacy)

Two `stats.json` files exist in the HAL repo. Relevant when running
`tools/run_hal_model.py` or comparing against HAL:

| File | Source | p1_percent max | Frames |
|------|--------|---------------|--------|
| `hal/data/stats.json` | Full multi-character dataset | 362 | 222M |
| `hal/checkpoints/stats.json` | Fox training subset | 236 | 27M |

**Use `hal/checkpoints/stats.json`.** HAL's `play.py` looks like it
overrides to `hal/data/stats.json` via `override_stats_path`, but the
Preprocessor actually loads from `checkpoints/stats.json` (verified:
`pp.stats["p1_percent"].max == 236.0`). The override mechanism changes
the config field but the Preprocessor resolves to the checkpoint stats
anyway. Using the wrong file shifts every normalized feature value
(e.g. percent=50 normalizes to -0.576 correctly vs -0.724 wrongly),
which makes the model see garbage inputs and play terribly.

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

### Legacy command (old leaked shards — reproduction only)

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

Single-GPU: drop the `torchrun` wrapper and add `--grad-accum-steps 8`
to match effective batch 512.

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

Use `tools/play_vs_cpu.py` for bot-vs-CPU matches,
`tools/play_netplay.py` for Slippi Online Direct Connect, and
`tools/head_to_head.py` for bot-vs-bot (watchable ditto). All three
import from `tools/inference_utils.py`, which holds the shared decode
pipeline (`load_mimic_model`, `load_inference_context`, `build_frame`,
`build_frame_p2`, `PlayerState`, `decode_and_press`). Any new
inference entry point should import from there rather than
reimplement — the L-button / TRIG decode (see Pitfalls #9) lives in
one place on purpose.

### Running HAL's original checkpoint

```bash
python3 tools/run_hal_model.py \
  --checkpoint /home/erick/projects/hal/checkpoints/000005242880.pt \
  --dolphin-path /home/erick/projects/hal/emulator/squashfs-root/usr/bin/dolphin-emu \
  --iso-path "/home/erick/Downloads/Super Smash Bros. Melee (USA) (En,Ja) (Rev 2).iso" \
  --character FOX --cpu-character FOX --cpu-level 9 --stage FINAL_DESTINATION
```

`run_hal_model.py` is MIMIC's from-scratch reimplementation of HAL's
5-class inference, for running Eric Gu's original HAL weights only.
It does not share `inference_utils` with the MIMIC play tools.

### HAL's own inference (ground truth)

If our reimplementation breaks, HAL's own code always works:

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

### `btns_single` encoding

Shards encode multi-hot buttons as single-label using early-release
logic (matching HAL's `convert_multi_hot_to_one_hot_early_release`):
when buttons change but nothing new is pressed (partial release), the
label is `NO_BUTTON` (4). Do not "keep the surviving held button" —
that diverges from the training target.

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

### Ranked dataset pipeline

Source: six anonymized archives of Slippi ranked replays (platinum+),
~850k total, at `/home/erick/Documents/melee/ranked-anonymized-N-*.{7z,zip}`.
Each archive has flat files named `{rank1}-{rank2}-{hash}.slp` where
each rank ∈ {platinum, diamond, master}. Rank pairs appear in the
fixed orderings `diamond-diamond`, `diamond-platinum`,
`master-diamond`, `master-master`, `master-platinum`,
`platinum-platinum` — higher rank first (M > D > P).

HF layout:

```
{CHAR}/
  {CHAR}_{rank_pair}_a{N}.tar.gz    # one per (character, rank_pair, archive)
metadata/
  metadata_a{N}.json                # flat list, one entry per replay in archive N
```

Tarballs contain raw `.slp` files, no preprocessing. Up to 25
characters × 6 rank pairs × 6 archives = 900 shards in the fully
populated case (rare char × rank combos may be empty for some
archives).

`tools/shard_and_upload_ranked.py` processes one archive end-to-end:

```bash
python tools/shard_and_upload_ranked.py \
    --archive /home/erick/Documents/melee/ranked-anonymized-1-116248.7z \
    --archive-id 1 \
    --workdir /home/erick/Documents/melee/staging_a1
```

It sets `socket.setdefaulttimeout(600)` so wedged HF uploads raise
instead of hanging the process forever.

Phases:

1. **Extract** — `7z x` (.7z) or `unzip` (.zip) to
   `{workdir}/extracted/`. Needs ~600 GB free for the largest archive
   (uncompressed).
2. **Per-file parse** (ProcessPoolExecutor across all CPUs) — read
   header only via `peppi_py.read_slippi(path, skip_frames=True)`,
   pull the two players out of `game.start.players`, map character
   int → name (ZELDA and SHEIK → `ZELDA_SHEIK`; POPO and NANA →
   `ICE_CLIMBERS`), reject `WIREFRAME_MALE/FEMALE / GIGA_BOWSER /
   SANDBAG / UNKNOWN` (debug/test characters, not legal tournament
   picks), parse the rank pair from the filename.
3. **Bucketing** — each replay enters up to two buckets keyed by
   (character, rank_pair). P1 and P2 both contribute, except dittos
   (same char for both players) which land once.
4. **Metadata** — flat list of
   `{filename, p1, p2, rank, archive}` entries per replay, schema
   matching the existing `metadata_a3.json` on HF.
5. **Tar + upload** — for each bucket in sorted order, check
   `api.list_repo_files()` and skip if `{CHAR}/{name}` already exists;
   otherwise `tarfile.open(..., "w:gz", compresslevel=6)`, upload via
   `api.upload_file(...)` with 5-attempt exponential backoff
   (`wait = min(300, 2^attempt * 10)`), delete the local tar. Metadata
   goes up after all bucket uploads; extract dir is removed; workdir
   shrinks back to tens of MB.

Resume semantics:

- **Mid-upload crash:** next run re-parses headers (~20s), then skips
  every bucket whose tar already exists on HF. Only unfinished buckets
  get rebuilt. Requires `--skip-extract` if the extract dir is still
  present so it doesn't blow up trying to re-extract into a non-empty
  dir.
- **Stuck upload:** socket timeout eventually raises and the retry
  loop reattempts. If retries exhaust, process dies and you resume as
  above.
- **Already-uploaded archive:** running again is a no-op — every
  bucket hits the "already uploaded" skip path.

### Building MIMIC-normalized shards

Requires a metadata directory with: `norm_stats.json`, `cat_maps.json`,
`stick_clusters.json`, `controller_combos.json`, and `mimic_norm.json`.
The `controller_combos.json` MUST have 7 combos (A, B, Z, Jump, TRIG,
A_TRIG, None) for the 7-class button head. The old 5-combo version
(A, B, Jump, Z, None — HAL's scheme) still works via backcompat but
cannot represent airdodge / wavedash / L-cancel.

### Retraining a character from scratch (ranked master-* data)

Summary stats are preserved — you don't need to regenerate them. All
5 inference-relevant JSONs live on
`huggingface.co/erickfm/MIMIC/<char>/`; the 6th (`norm_minmax.json`)
stays local in `data/<char>_v2/` and is only needed to rebuild
`mimic_norm.json` from scratch.

**What you have on HF** per char (`erickfm/MIMIC/<char>/`):
`model.pt`, `config.json`, `metadata.json`, and the 5 JSONs —
`cat_maps.json`, `controller_combos.json`, `mimic_norm.json`,
`norm_stats.json`, `stick_clusters.json`.

**Retrain recipe** (also lives in `tools/run_all_chars.sh` for the
full 7-char pipeline; the snippet below is the single-char equivalent):

```bash
C=fox; HF_BUCKET=FOX; IDX=1   # adjust per char

# 1. Raw .slp — all three master-* pairs
hf download erickfm/melee-ranked-replays --repo-type dataset \
  --include "${HF_BUCKET}/${HF_BUCKET}_master-master_a*.tar.gz" \
  --include "${HF_BUCKET}/${HF_BUCKET}_master-diamond_a*.tar.gz" \
  --include "${HF_BUCKET}/${HF_BUCKET}_master-platinum_a*.tar.gz" \
  --local-dir data/${C}_ranked_slp/_tars
for t in data/${C}_ranked_slp/_tars/${HF_BUCKET}/*.tar.gz; do
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

(Note: `--mimic-minimal-features` is intentionally absent — full
features are the default.)

Wall time per char (2×RTX 5090): ~10 min download/extract + 30–90 min
shard + ~50 min train = ~1.5–2.5 hours. Disk peak: 200–800 GB (chars
with bigger master populations like sheik and cptfalcon are closer to
the top).

Character index + HF bucket name per char (bucket ≠ display name for
Sheik and Puff because the ranked pipeline collapses Zelda+Sheik into
one bucket and Jigglypuff is the full name):

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

### The only actual HAL checkpoint

`/home/erick/projects/hal/checkpoints/000005242880.pt` — HAL's (Eric
Gu's) best at 5.2M samples. State dict with `module.` prefix (from
DDP). 101 MB. Loaded via `tools/run_hal_model.py` for comparison runs
only. Not used by any MIMIC production code path.

### MIMIC checkpoints (all in `checkpoints/`)

**Naming convention:**

    {char}-{YYYYMMDD}-{descriptor}-{steps}k.pt

The `{descriptor}` slot is a free-form tag for whatever is most
distinctive about the run — usually the position encoding (`rope`,
`relpos`), but can also flag a non-standard recipe: `overfit`,
`wavedash`, `small`, `longrun`, `lowdropout`, etc. Pick whatever makes
the checkpoint recognizable next to others for the same character on
the same day. Set `--run-name` to `{char}-{YYYYMMDD}-{descriptor}`
when starting a new run; the step suffix is appended when renaming
the finished `_best.pt`.

Current per-character actives:

- `puff-20260419-mimic-fullfeat-gate01-33k.pt` — Puff (mimic + full
  features + input-gate λ=0.01, val 0.6641).
- `falco-20260412-relpos-28k.pt` — Falco (relpos, val 0.7374).
- `cptfalcon-20260412-relpos-27k.pt` — CptFalcon (relpos, val 0.71).
- `luigi-20260412-relpos-5k.pt` — Luigi (relpos, early-stopped, val ~1.0).

### Promotion policy

When a finished wandb run beats the current best `val/total` for its
character: pull the checkpoint, rename it to
`{char}-{YYYYMMDD}-{descriptor}-{steps}k.pt`, upload it to
`erickfm/MIMIC` on HF, and update the Discord bot's default for that
character. Skip runs whose name starts with `SWEEP-`, `DBG`, `DEBUG-`,
`FIX-`, or `BENCH-` — those are infra/debug, not production
candidates, even if their loss is lower. A promotable run is a
named-character training like `fox-20260416-master-relpos`. Always
surface candidates to the user and ask before pushing — don't
auto-promote. If a new character appears (e.g. first Marth run), flag
it separately since the bot's character list needs wiring too.

### Bot-startup audit

Whenever you (re)start `tools/discord_bot.py`, do a promotion audit
first. The authoritative source is HF `erickfm/MIMIC` — checkpoints
live there after promotion, and the GPU boxes are ephemeral so wandb
checkpoints aren't retrievable once the machine dies. Procedure:

1. List files under `erickfm/MIMIC` via `HfApi.list_repo_files`. For
   each character dir (`fox/`, `falco/`, `cptfalcon/`, `luigi/`, plus
   any new ones), read `metadata.json` for `run_name`, `global_step`,
   and `val_loss`.
2. Compare each character's HF `run_name` to the filename the bot's
   `CHARACTERS` dict in `tools/discord_bot.py` currently points at.
   If HF has a newer run (different `run_name` / higher step / better
   `val_loss`) than what's wired into the bot, it's a replacement
   candidate.
3. Also flag any HF character directory the bot doesn't know about —
   that's a new character the bot needs wired up (extend
   `CHARACTERS` dict + aliases).
4. Report candidates to the user and ask whether to pull + swap
   before bringing the bot up. On approval: `snapshot_download` the
   new character dirs into `hf_checkpoints/`, copy the `.pt` into
   `checkpoints/` with the project naming convention, edit
   `tools/discord_bot.py` `CHARACTERS` dict, and restart the bot.

Secondary (supplementary) check: wandb `erickfm/MIMIC` for finished
non-debug runs newer than HF's `last_modified` — useful to remind the
user "you have trained X but it hasn't been uploaded yet," but the
checkpoints themselves have to come from the GPU box (or be
re-trained) since wandb doesn't store the `.pt`.

### Bot startup orphan sweep

`_cleanup_orphan_processes()` runs before `bot.run()` and SIGTERMs
(then SIGKILLs) any `play_netplay.py` and our-path `dolphin-emu`
processes still alive from a prior bot run. Without this they get
reparented to init and burn CPU/GPU forever — we've seen a 10+ hour
Dolphin at 85% CPU with no replays being written. If you ever need
a `play_netplay.py` to survive a bot restart, carve an exception
here rather than disabling the sweep.

## File map

### Core
- `train.py` — Training loop (DDP, gradient accumulation, mimic_mode, cosine LR).
- `mimic/model.py` — Model architecture (FramePredictor, mimic presets, attention variants).
- `mimic/dataset.py` — StreamingMeleeDataset (per-game and pre-windowed shards).
- `mimic/frame_encoder.py` — Input encoders (MimicFlatEncoder for mimic_mode). Honors `mimic_minimal_features`: minimal path slices the shard's numeric tensor down to 9 slots (HAL-reorder); full path expects 13 numeric + 5 flags per player. Also hosts the optional L1-gated input projection (`use_input_gate`) for feature-importance diagnostics.
- `mimic/features.py` — Feature encoding (cluster centers, controller one-hot, normalization). `numeric_state(full)` returns 13 cols.
- `eval.py` — Offline evaluation (validation metrics).
- `inference.py` — Legacy inference script. Use `tools/play_vs_cpu.py` or `tools/play_netplay.py` in modern code paths.

### Tools

**Inference (local):**
- `tools/play_vs_cpu.py` — Runs MIMIC checkpoints vs CPU in Dolphin. Uses shared `inference_utils.decode_and_press`.
- `tools/head_to_head.py` — Two checkpoints against each other in one Dolphin instance (watchable ditto).
- `tools/run_hal_model.py` — Reimplementation of HAL's 5-class inference. Loads HAL checkpoints. Structurally can't wavedash (no TRIG class).

**Inference (online, Slippi netplay):**
- `tools/play_netplay.py` — Joins a Slippi Online Direct Connect lobby. Uses `MenuHelper.menu_helper_simple(connect_code=...)`, detects bot's port via the `connectCode` field on `PlayerState` (handles dittos and palette swaps). **Persistent-session mode** (see below): plays up to `--max-matches N` back-to-back matches in one Dolphin process, emitting a per-match stdout block (`MATCH_START:`, `RESULT:`, `SCORE:`, `REPLAY:`) and a single `SESSION_END:` on exit. Default `--max-matches 1` preserves the one-shot CLI behavior; the Discord bot passes `-1` for unlimited.
- `tools/discord_bot.py` — Discord front-end (prefix commands: `!play`, `!queue`, `!cancel`, `!info`, `!reload`, plus one `!<character>` shortcut per loaded character/alias). Single-session FIFO queue via `asyncio.Queue`. Spawns `play_netplay.py` once per user and streams its stdout: each `MATCH_START` → `▶️ Match N starting` post, each `RESULT/SCORE/REPLAY` triplet → result announcement + replay upload. Config via `.env` (see `.env.example`). Per-character checkpoint labels render consistently (run name + `val`, step count pulled from each char's HF `metadata.json`) across `!info`, session-starting, and match-result messages via `_ckpt_label_for(char)`.

`!<character>` shortcut: `!fox`, `!marth`, `!falcon`, etc. are
registered dynamically via `_register_char_shortcuts()` — one command
per key in `CHAR_ALIASES` (character keys + hardcoded aliases
`falcon`/`cpt`/`cf`), skipping the reserved set
(`play`/`queue`/`cancel`/`info`/`reload`/`help`). The shortcut extracts
a `TAG#NUMBER` substring from `ctx.author.display_name` via
`_DISPLAY_NAME_CODE_RE` (case-insensitive, takes the first match) and
invokes `cmd_play` with it. Shortcuts are rebuilt on `!reload` so a
newly-uploaded character is immediately callable without a bot
restart. If the caller's display name has no code the shortcut replies
with a redirect to the long-form `!play`.

`!reload` diff: `cmd_reload` compares a fingerprint tuple `(path,
run_name, global_step, val_loss)` rather than just the local path,
because a retrain reuses `hf_checkpoints/{char}/model.pt` — only the
`metadata.json` fields actually change. Without the fingerprint,
re-uploading a new checkpoint for an existing character would silently
report "no changes."

**Persistent-session model.** Each Discord `!play` spawns one
`play_netplay.py` that keeps Dolphin alive across multiple matches in
the same Slippi Direct Connect lobby. After each match, `MenuHelper`
drives through POSTGAME_SCORES back to CSS; the opponent has
`--rematch-timeout` seconds (default 30) to pick a character and
press Start for the next match. The session ends when (a) opponent
DCs or idles, (b) another user calls `!play` (the bot writes `STOP\n`
to the subprocess's stdin, which finishes the current match then
exits), or (c) the current player `!cancel`s (same STOP path). **Do
NOT reintroduce per-match subprocess spawning** — the 30–60 s of
Dolphin relaunch dead air was the exact UX problem this refactor
fixed.

Stdout protocol emitted by `play_netplay.py`:

```
MATCH_START: <1-based idx>     # emitted when IN_GAME is first reached
RESULT: win|loss|draw|disconnect|no-opponent|timeout|failed
SCORE: bot=Xstk/Y% opp=Xstk/Y%
STAGE: <melee.Stage enum name>   # observed gs.stage, not the CLI preference
REPLAY: /abs/path/Game_YYYYMMDDThhmmss.slp
# (above five repeat per match)
SESSION_END: max-matches|opponent-gone|opponent-timeout|stopped|error|signal|hard-timeout
```

Per-match state reset lives in the outer match loop of
`play_netplay.py` (MenuHelper `stage_selected` /
`frozen_stadium_selected`, `PlayerState` recreated on each IN_GAME
transition). `opponent_last_seen` and `opponent_ever_seen`
deliberately persist across matches for DC detection. STOP polling
uses `select.select([sys.stdin], [], [], 0)` — non-blocking, checked
once per console step.

**Multi-instance on one machine.** Running N independent MIMIC bots
on the same box is feasible; limits are accounts + process isolation,
not hardware. Each instance needs its own Slippi account (unique
`connectCode` + `playKey` — Slippi rejects duplicate logins), own
`SLIPPI_HOME` dir, own Discord bot token + `BOT_SLIPPI_CODE`, and own
replay dir. Resource budget on a 24 GB GPU + 8-core CPU: ~3–5
concurrent sessions before 60 fps frame deadlines tighten (20M-param
bf16 model is ~200 MB VRAM, ~2 ms/frame inference; Dolphin on Vulkan
is ~50–70% of one core). Simplest deployment: N copies of the bot
with distinct `.env` files — zero code change. Nicer deployment: one
Discord front-end routing to a worker pool of N slots; requires
replacing `match_worker` with a parallel pool and turning
`current_proc` into a per-slot dict — ~100 lines.

**Inference (shared):**
- `tools/inference_utils.py` — Shared inference pipeline: `load_mimic_model`, `load_inference_context`, `build_frame`, `build_frame_p2`, `PlayerState`, `decode_and_press`. Produces the full 13-numeric + 5-flag tensor (via `MIMIC_NUM_FULL` in exact shard-schema order); reads speeds / hitlag / hitstun via libmelee's `PlayerState.speed_*`, `.hitlag_left`, `.hitstun_frames_left`; normalizes extras via the transforms in `mimic_norm.json` (z-score for entries missing newer transform keys). Works for both minimal and fullfeat checkpoints because the minimal encoder path slices the 13-col input to 9 cols internally.

**Diagnostics:**
- `tools/inspect_frame.py` — Shows exactly what goes into and out of the model for a single frame. Takes `--shard 0 --frame 534 --context 2` style args.
- `tools/extract_wavedashes.py` — Extracts wavedash-only training windows for overfit sanity checks.
- `tools/validate_checkpoint.py` — Evaluates checkpoint(s) on val data, reports per-head CE loss.
- `tools/diagnose.py` — Pipeline debugging (tensor-level train vs inference comparison).

**Data:**
- `tools/slp_to_shards.py` — Raw .slp replays → .pt tensor shards. Produces v2 shards (`target[i] = buttons[i+1]`). Writes the 13-col schema driven by `mimic/features.py:numeric_state`.
- `tools/shard_and_upload_ranked.py` — Ranked-replay archive → HF tarballs. See Data § Ranked dataset pipeline.
- `tools/split_by_character.py` — Split dataset by character.

## Pitfalls for agents

1. **`tools/run_hal_model.py` loads actual HAL weights.** MIMIC
   checkpoints go through `tools/play_vs_cpu.py` / `play_netplay.py`
   / `head_to_head.py`. `run_hal_model.py` is the
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
    collapsed via early-release encoding: the newest button (0→1
    transition) gets the label. Shoulder+button combos ARE
    representable since shoulder is a separate head.

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

Eric Gu's original HAL codebase. Key files:

- `hal/eval/play.py` — Ground-truth inference script (always works).
- `hal/preprocess/preprocessor.py` — Preprocessing (normalization, controller encoding).
- `hal/preprocess/transformations.py` — Feature transforms (one-hot encoding, sampling).
- `hal/preprocess/input_configs.py` — Input feature configuration.
- `hal/preprocess/postprocess_configs.py` — Output decoding configuration.
- `hal/training/models/gpt.py` — Model architecture (GPTv5Controller).
- `hal/constants.py` — Cluster centers, button lists, character/stage/action indices.
- `hal/emulator_helper.py` — Dolphin controller interface.
- `hal/gamestate_utils.py` — Gamestate extraction from melee-py.
- `hal/data/stats.json` — Full dataset stats (222M frames, DO NOT use for inference).
- `checkpoints/stats.json` — Fox training stats (27M frames, USE THIS ONE).
- `checkpoints/config.json` — Training config.
- `checkpoints/000005242880.pt` — Best checkpoint (5.2M samples).
- `hal/local_paths.py` — Local machine paths (emulator, ISO, replay dir).

## Environment

### Remote GPU

| Machine | Host | Port | User | GPU | Storage | Status |
|---------|------|------|------|-----|---------|--------|
| A | 194.14.47.19 | 22877 | root | RTX 5090 | 3 TB SSD | Active |

```bash
ssh -p 22877 root@194.14.47.19   # Machine A
```
