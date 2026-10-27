# CLAUDE.md — Agent Orientation for MIMIC

## What this project is

MIMIC is a behavior-cloning bot for Super Smash Bros. Melee. It watches human
replays and learns to predict controller inputs from game state. At inference
it drives a controller through Dolphin (the GameCube emulator) via libmelee.

The transformer backbone and early data pipeline started from
[HAL](https://github.com/ericyuegu/hal) (Eric Gu); everything on the active
code path is MIMIC's own. Leftover `hal` names and the HAL reference
implementation are covered in one place: "HAL leftovers" near the end.

## Architecture

MIMIC's canonical config (preset `M`):

- **Params:** ~19.99M.
- **Transformer:** d_model=512, 6 layers, 8 heads, block_size=1024.
- **Position encoding:** Shaw relative-position attention — the only one
  implemented (RoPE/xPos/ALiBi/learned were removed 2026-10-22; see Pitfalls).
- **Input:**
  Linear(184 → 512) from
  `[stage_emb(4) + 2*char_emb(12) + 2*action_emb(32) + gamestate(36) + controller(56)]`,
  gamestate = 18 per player (13 numeric + 5 flags). The older 9-col
  "minimal features" input (Linear 166 → 512) was removed 2026-10-22;
  minimal-features checkpoints no longer load (git history).
- **Output heads (autoregressive with detach):**
  shoulder(3) → c_stick(9) → main_stick(37) → buttons(7).
- **Head hidden dim:** `input_dim // 2` (NOT a fixed 256 — each head has
  different hidden size).
- **Sequence length:** 180 frames (~3 seconds).
- **Dropout:** 0.2.

### Gamestate columns per player

**18 per player:** 13 numeric + 5 flags in native shard order:

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
on the 22-col layout and cannot load into the current 184→512 encoder.

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

**C-stick history encoding provenance:** shards built from 2026-07-17
onward encode the c-stick *history* one-hot as nearest-of-9 clusters
from raw coords (`encode_controller_onehot(c_xy=...)`), matching both
the c_dir targets and inference. Shards built earlier used a 5-class
direction collapse (deadzone 0.15, cardinals only) mapped into the 9
slots — their history never activates the 4 diagonal slots, so models
trained on them see a small OOD input at inference whenever the bot
samples a diagonal c-stick (~0.7% of frames). Fixed by rebuilding
shards; the 5-class path survives only as the fallback for old shards
without raw c coords.

### Model names (`mimic/model.py:MODEL_PRESETS`)

Two choices name a model: a size preset and an attention mode.

- **Size:** **`M`** (`--model M`, default) — production, d_model=512, 6 layers, 8
  heads, ~20.0M params. **`S`** (`--model S`) — half width (256, 4 heads), ~5.4M
  params, the fast research proxy. Older checkpoints record `model_preset: "mimic"`
  (= M) or `"mimic-w256"` (= S); it's only a label, loading reads the config fields.
  The 11.5M–154M presets (`mimic-w384/w768/w1024/xl/xxl`) were removed 2026-10-23
  (git history).
- **Attention: full vs stream.** **Full** (default): every frame the model re-reads
  the last 180 frames (~3 s) from scratch. **Stream** (`--attn-window 180`, plus a
  longer `--seq-len`): each layer reads 180 frames of the layer below, so 6 layers
  reach 1,075 frames (~18 s), and inference caches earlier work (`mimic/kv_cache.py`):
  one step per frame, ~22× faster batched. **Stream is sunset (owner, 2026-10-25):**
  at matched data and steps S-stream lost to S-full 77–115 despite lower loss, and
  neither RL nor a shorter window closed the gap (`docs/research-notes-2026-10-25.md`).
  The code still works (every inference path streams window models through the cache)
  but don't build on it.
- **Training budget** (separate from the model): **proxy** = S, 1,900 steps (~5 min);
  **cell** = M, 6,500 steps on 5% of master Fox (~45 min); **prod** = M, full data,
  ~280k steps.
- The relpos block also takes `use_rmsnorm` / `use_swiglu`. Everything
  else (RoPE/xPos/ALiBi/flexbias/learned position encodings, the generic
  block and GQA, the `modern*` and generic `tiny`…`giant` presets, the
  `hal*` preset aliases, the MLP backbone) was removed 2026-10-22; it's
  in git history.

### Learned input gate (diagnostic, opt-in)

`--input-gate-l1 <λ>` (typical 0.01) adds a per-input sigmoid gate
to `MimicFlatEncoder` and an L1 sparsity penalty. End of training
writes `checkpoints/{run_name}_gate_report.json` ranking all input
scalars by learned gate value. λ=0.01 costs ~0 val-loss. Off by
default; not a production regularizer.

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

**Rule: do NOT use `--reaction-delay 1` with v2 shards** (the default
is 0; `--controller-offset` was removed 2026-10-22). The alignment is already correct; adding an offset
double-shifts the data.

**Old shards** (`data/fox_hal_full`, `data/fox_hal_match_shards`) have
the leak. With those shards, use `--reaction-delay 1` to achieve the
same effect at dataloader time.

Val loss is not comparable across shard versions: v2 shards produce
higher val loss because the model can no longer cheat via
action→button memorization. A v2 val loss of ~1.0 can correspond to
better gameplay than 0.74 on old shards.

### Why val loss is a weak proxy for strength (error consequence is non-uniform)

The deeper reason val loss keeps failing to predict head-to-head
strength (empirically confirmed repeatedly — see the 2026-06 h2h runs
where −0.012 val "wins" were in-game washes): **cross-entropy weights
each frame's error by how *often/confidently* it occurs, never by how
much that input *matters to winning*.** But human input consequence is
wildly non-uniform:

- On the vast majority of frames the exact input barely matters — a
  slightly-off stick angle, a **missed L-cancel** (minor tempo loss).
  These dominate the loss (they're common) but are nearly free in-game.
- A tiny fraction of inputs are **decisive**, and among those some
  mistakes are **catastrophic**: pressing **L/R off-stage** (airdodge
  into the blastzone = a free stock), a missed tech (comboed to death),
  an SD. These are rare, so they barely move the loss — but they can
  lose the game outright.

So a minuscule val-loss difference (0.001) can correspond to *anything*
from "nothing" (a few L-cancels) to "loses the match" (a few off-stage
shoulder presses). The loss literally cannot tell them apart. Two
corollaries:

1. **Don't trust small val deltas as strength.** Confirm with h2h. This
   is why the promotion/eval discipline leans on head-to-head, not val.
2. **This is a core argument for RL.** A game-outcome reward
   (stock/win) is *consequence-weighted by construction* — it learns
   that pressing L off-stage costs a stock no matter how rare, an
   asymmetry cross-entropy can never see. (A future BC direction:
   consequence-weight the loss near critical states — edge, high-%,
   tech situations.) It also explains the determinism fragility: near
   the edge, the *modal* action can itself be the SD, so argmax/low-T
   walks off and dies (see the 2026-06-27 decode sweep: T<1.0 →
   4-stocked).

## Training

### Current best command (v2 shards, full features)

```bash
torchrun --nproc_per_node=2 train.py \
  --model M --encoder mimic_flat \
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

Since 2026-10-22 most of these flags are defaults (`--model M`,
`--encoder mimic_flat`, `--mimic-controller-encoding`, `--stick-clusters
hal37`, `--self-inputs`, `--reaction-delay 0`, `--lr 3e-4`, no warmup,
`--cosine-min-lr 1e-6`) and `--mimic-mode` / `--plain-ce` are accepted
no-ops (mimic mode and plain CE are the only mode and loss; focal loss
and the multi-label heads were removed). A bare `train.py --data-dir D
--run-name N` trains the production recipe.

**Muon is the default optimizer (since 2026-10-22).** `--optimizer muon`
(implicit): Muon at `--muon-lr 0.02` on the hidden 2D matrices,
aux-AdamW at `--lr` on embeddings / heads / relpos tables / norms /
biases, one cosine schedule over all groups. On the 20M proxy it beat
AdamW by −1.6% val, and a 3,500-step Muon run beat a 6,500-step AdamW
run (≥1.8× fewer steps, ~4% more time per step). lr 0.04 was worse
than AdamW; SOAP (lr 3e-3) much worse. Proxy-scale, single seed — not
yet h2h-validated at production scale. **Every run before 2026-10-22
used AdamW:** pass `--optimizer adamw` to reproduce or `--resume`
one (a mismatched resume now fails with a message saying so). See
`docs/research-notes-2026-10-22.md`.

**Sliding-window (stream) models: sunset 2026-10-25.** `--attn-window 180` with a longer
`--seq-len` (≥ the 1,075-frame receptive field, e.g. 1440) trains them; they stream through
`mimic/kv_cache.py` in every inference path, and the sim RL update scores them with an exact
packed forward (`ppo.pack_streams`). They lose to full attention in play at matched compute
(see Model names above), so don't build on them. Score any window model with
`tools/eval_last_frame.py`, not train.py's window-averaged val. Opt-in speed flag:
`--relpos-impl sdpa` (~14% faster step, equal val on one run; its bf16 sibling diverged, so
watch a long run before relying on it).

**Architecture variants (`--arch key=value,...`, 2026-10-25).** Sets `ModelConfig` fields:
`attn_gate`, `qk_norm`, `mlp_act=relu2`, `zero_init_proj`, `logit_softcap`, `value_residual`,
`unet_skips`, `canon`, `use_rmsnorm`, `use_swiglu`, `encoder_type=mimic_entity` (two tokens per
frame), and size fields (`d_model`, `num_layers`, ...). All off by default; full-causal only.
The proxy sweep (`docs/research-notes-2026-10-25.md`) found only gated attention better at proxy
scale (−0.006 val, 62–34 in a proxy h2h, +7% params, ~9% slower); the rest were noise or worse, so
don't re-run them. **The gate does not scale:** at S 56k (full56k's recipe + `attn_gate=1`) it led
until ~10k steps, then overfit (train loss 0.022 below full56k, last-frame val 0.022 above) and lost
34–62 to full56k. Not a default. Variants without a hand-written fast path fall back to the plain forward in
`fast_policy.forward_last` (correct, slower sim inference); the gate has a fast path.

**`--cond-heads` is the default (since 2026-10-22).** Each head is
conditioned on the earlier heads' classes, so the four heads model the
joint distribution; same parameters and training cost. It beat the
same-recipe control 611–445 in the native sim (−4.2% val) and passed
the Dolphin canaries (2026-10-22: 2–0 vs CPU-9, center 41%, L-cancel
100%). `--no-cond-heads` for the older heads. `--resume` and `--eval-only`
both build the model from the CLI flags, so they now refuse a
checkpoint whose cond-heads / attn-window / patch / mtp settings differ
from the flags (an old checkpoint needs `--no-cond-heads`, and
`--optimizer adamw` to resume its optimizer state); `--warm-restart`
may switch architecture on purpose.

**Also default since 2026-10-22:** `--lean-metrics` (metrics only on
log steps; `--no-lean-metrics` to disable) and validation every 2.5%
of a run instead of 1% (`--val-frac 0.01` restores the old cadence;
`tools/proxy_run.sh` pins 0.01 so proxies stay comparable with older
ones). The weight EMA (`--ema-decay`) was removed 2026-10-22: promotion
uses the tail-SWA average (`tools/average_checkpoints.py`).

**`--self-inputs` is required even on v2 shards** (it's the default;
`--no-self-inputs` turns it off). Without it the encoder has no
controller-history input at all and val loss climbs ~3.5× (main-stick
F1 drops from ~57% to ~15%).

`--model M` is the default.

Single-GPU variant: swap `torchrun --nproc_per_node=2` for `python3`
and use `--batch-size 64 --grad-accum-steps 8` to keep effective batch
at 512.

The `--encoder hal_flat` alias still works; checkpoints from before the
HAL→MIMIC rename load unchanged (the `hal_*` config fields are migrated
in `ModelConfig.__post_init__`).

### Legacy leaked-shards runs

Reproducing pre-v2 runs against `data/fox_hal_*` (minimal features,
`--controller-offset`, rd=1) needs the code from before the 2026-10-22
legacy cleanup (git history).

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
reading `MODEL_PRESETS[model_preset]["max_seq_len"]` (180 for every
preset; `--seq-len` overrides it, e.g. for sliding-window chunks).

**Do not add `if model_preset == "X"` gates in this lookup.** Read
the value off the preset dict so any new preset/alias automatically
picks it up. String checks silently fall through for renamed or
aliased presets.

### Local research proxies

- **Fast (~5 min, the research loop):** `tools/fast_proxy.sh NAME S 1900 SEED [flags]`
  — 5M model, Muon, cond-heads (since 2026-10-25), val at the end, tail-SWA, fixed-subset val in
  `checkpoints/fastproxy/results.tsv`. Seed gap ~0.0007; reproduces cond-heads (−3.9% vs −4.2%);
  exaggerates optimizer/LR effects ~5× (right sign). Calibration in
  `docs/research-notes-2026-10-22.md`.
- **C cell (~45 min, confirmation):** `tools/proxy_run.sh NAME M data/mastfox_c9_half 6500 SEED`
  — 20M production architecture, pinned to the pre-2026-10-22 settings (AdamW, no cond-heads,
  val every 1%) for comparability with older proxies.
- **Warm start (~15 min, shape-compatible changes only):** `--resume hf_eval/fox-master/model.pt
  --no-load-optim --warm-restart --lr 1e-4`, 2k steps.
- Score sliding-window models with `tools/eval_last_frame.py`.

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
reimplement — the L-button / TRIG decode (see Pitfalls #8) lives in
one place on purpose.

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
  mixes in games where the *other* player is sub-master). **Per-port
  rank IS recoverable from the .slp** — the dataset's anonymizer wrote
  each player's rank into the netplay name field
  (`start.players[i].netplay.name` = `"Master Player"` /
  `"Diamond Player"` / `"Platinum Player"`; connect codes are blanked).
  So we can tell exactly which port is which rank — for any character and
  any rank tier; nothing below is Fox-specific (Fox is just the current
  test char). **Think in *perspectives*, not games.** `slp_to_shards.py`
  extracts both players of every game and keeps each perspective where
  `self` is the target char (the `character_filter` at
  slp_to_shards.py:1215): a target-vs-other game yields **one** perspective;
  a same-char ditto yields **two** — and those two are *different humans*,
  distinct trajectories, **not** a double-count. Every distinct target-char
  perspective is extracted exactly once. The rank gotcha: in a mixed-rank
  pair (e.g. master-diamond) we keep whichever player is the target char,
  and that player is the *higher*-ranked one only ~half the time — the
  named "master" is often the *opponent* we discard. Measured 2026-06-25 for
  Fox over the master-* pool: only **61.6%** of Fox perspectives are
  master-rank (100% in master-master; ~51-52% in master-diamond/platinum).
  The same dilution applies to any char pulled from mixed-rank tiers.
  `tools/filter_masterfox.py` filters to perspectives where the target
  player is master (keep a game iff every target-char port is master, so no
  sub-master perspective survives and mixed-rank dittos are dropped whole);
  it hardcodes Fox (`FOX_EXT=2`) but the approach generalizes — swap the
  char id / rank. `tools/build_masterfox_train_box.sh` does the full
  filter→reshard→train, `tools/build_full_masterfox_box.sh` does it uncapped
  (all ~102k master-Fox perspectives). The earlier "can't tell which port is
  master" claim was wrong.
  **Folders are now correctly
  labeled** — as of 2026-06-15 the dataset was re-sorted in place so
  every folder contains only its named character (a long-standing
  peppi-external-vs-libmelee-value ID scramble was fixed; see
  `docs/research-notes-2026-06-15.md` and
  [[project_ranked_dataset_mixed_labeling]]). The old combined
  `ZELDA_SHEIK` bucket is now split into separate **`SHEIK`** and
  **`ZELDA`** folders (`ZELDA` = games that are majority-Zelda-frames
  only; ~3.4k games, diamond/platinum-skewed). The old scrambled
  `metadata/metadata_a{N}.json` sidecars were deleted in the fix and
  not yet regenerated.
- **`erickfm/slippi-public-dataset-v3.7`** — legacy. 95K replays by
  character; Fox folder has 45,854 .slp. (Its downloader,
  `tools/download_fox.py`, was removed 2026-10-22.)

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

**peppi external-ID remap (the root-cause fix).** peppi returns
EXTERNAL (CSS-order) character IDs (Falcon=0, Fox=2, Samus=16) while
libmelee's `Character` enum uses different VALUES (Falcon=2, Fox=1,
Samus=13). The original code indexed the libmelee-value name table
with peppi's external id, scrambling every bucket (real Pikachu →
"SAMUS" folder, etc.). Fixed 2026-06-15: `PEPPI_TO_LIBMELEE` remaps
external→libmelee value before naming (`_parse_one`). Do NOT drop
that remap — it's the same verified table as
`rlvr/state/peppi_adapter.py`.

Character mapping notes: Popo+Nana collapsed to `ICE_CLIMBERS`; debug
chars (WIREFRAME/GIGA/SANDBAG) rejected. Zelda+Sheik are still
collapsed to a single `ZELDA_SHEIK` bucket at header-parse time (this
script can't tell them apart without frame data); the canonical
dataset splits them into separate `SHEIK`/`ZELDA` folders as a
post-step keyed on majority in-game form via the `classify_zelda_sheik` /
`classify_marth_zelda` scripts (deleted with `tools/legacy/` in commit
`5ecd366` — recover from git history if the split ever needs redoing).

**Dataset-fix toolchain** (used for the 2026-06-15 in-place correction).
The scripts were **deleted once the fix shipped** — the root cause is now
fixed upstream in `shard_and_upload_ranked.py`, so they only matter if the
dataset is ever rebuilt and re-scrambled. Recover them from git history
(commit `f08df05`, under `tools/`) if needed:
`build_ranked_index.py` (reverse the bug from metadata → true chars),
`validate_ranked_index.py` (check vs libmelee),
`resolve_ambiguous.py` (split collapsed labels by parse),
`execute_dataset_fix.py` (phases `renames`/`retar`/`swap` —
server-side `CommitOperationCopy` for clean renames, re-tar only the
entangled buckets; `commit_robust` wraps uploads in a SIGALRM
timeout+retry because huggingface_hub hangs on dead sockets and the
home uplink is ~4 MB/s — do NOT use `HF_HUB_ENABLE_HF_TRANSFER`, it
hangs outright here), `verify_fixed.py`.

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
best command) is wired into `tools/run_pod_chars.sh` (remote box;
stages, never pushes) and `tools/run_local_chars.sh` (local single-4090
box; builds fresh per-char metadata, prefetches char N+1 during char
N's training, uploads in the background). The v2 shard sets for every
character are also on HF (`erickfm/mimic-shards-v2`), usable directly
as `--data-dir`. Per-char wall time on 2×RTX
5090 is ~1.5–2.5 hr; on the local 4090 ~2 hr of GPU time per char
(bs 256 × grad-accum 2, ~4.5 step/s).

**`hf download --include` gotcha:** repeating the `--include` flag
overwrites earlier patterns (argparse `nargs='*'`) — all glob
patterns must follow a SINGLE `--include`. The (since removed)
`run_all_chars.sh` HAD the repeated-flag bug and silently fetched only
`master-platinum` (6 of 18 tarballs per char); the 2026-06-12 local
runs initially trained on that ⅓ data and early-stopped at 1.6k–7.8k
steps before the fix. Any model whose data was pulled through the buggy
path should be considered platinum-only until verified.

Character index + HF bucket map (`idx` = the libmelee `--character`
value passed to `slp_to_shards.py`). Post-2026-06-15, Sheik and Zelda
are separate folders (`SHEIK`/`ZELDA`); Jigglypuff bucket is the full
name `JIGGLYPUFF`:

| char | HF bucket | idx |
|---|---|---|
| fox | `FOX` | 1 |
| falco | `FALCO` | 22 |
| marth | `MARTH` | 18 |
| sheik | `SHEIK` | 7 |
| zelda | `ZELDA` | 19 |
| cptfalcon | `CPTFALCON` | 2 |
| puff | `JIGGLYPUFF` | 15 |
| luigi | `LUIGI` | 17 |

## Checkpoints

**MIMIC checkpoints** live in `checkpoints/`. Naming:
`{char}-{YYYYMMDD}-{descriptor}-{steps}k.pt`. Descriptor is a
free-form tag (`relpos`, `rope`, `fullfeat`, `gate01`, `xxl`, etc).
Set `--run-name` to `{char}-{YYYYMMDD}-{descriptor}`; step suffix is
appended when promoting the `_best.pt`.

Current per-character production:
- `fox-master` — `fox-mastfox-20260717-cstick9-nomirror` (tail-SWA of steps
  224k–280k = `AVG_c9nm.pt`), promoted 2026-07-18. Champion recipe (280k
  steps, eff batch 768, seed 42) on freshly rebuilt v2 shards with the
  **nearest-of-9 c-stick history encoding** (commit `fb7bb1a`), **no mirror
  aug** (dropped by owner; also sidesteps the mirror facing-flag bug), and
  the **full uncapped 102,301-game master-Fox pool** (97,492 train
  perspectives — the prior ~77k cap was a disk artifact). **Beat the prior
  champ 16–5 (76.2%, p≈0.026) over 21 realtime matches** — the first h2h in
  the fox line to clear binomial noise — with clean canaries (93.4%
  aggregate L-cancel, n=1,103). Attribution caveat: three changes at once
  (c-stick fix / no mirror / more data); the win is not attributable to any
  single one. Prior champ `fox-mastfox-20260625` retained on disk as
  `AVG_mastfox.pt` (the RLVR frozen-BC reference). `metadata.json:val_loss`
  (0.6788) is in-training best on its own val split — not comparable across
  data builds; h2h is the cross-build metric.
- `fox-diamond` (val 0.7325) / `fox-platinum` (val 0.7479) — the rank ladder
  (`docs/research-notes-2026-06-16.md`).

All three load into the Discord bot automatically — it has no hardcoded
character list; `_sync_hf_to_local` discovers every `erickfm/MIMIC/{dir}/`
with a `model.pt` + `metadata.json:melee_enum`, so the rank dirs surface as
`!fox-master` / `!fox-diamond` / `!fox-platinum` (and `!play fox-master CODE`)
after a `!reload` or restart. `upload_char.py` already writes `melee_enum:
"FOX"` and bundles each dir's norm files, which is what makes this work.
- `puff-20260419-mimic-fullfeat-gate01-33k.pt` (val 0.66)
- `falco-20260412-relpos-28k.pt` (val 0.74)
- `cptfalcon-20260412-relpos-27k.pt` (val 0.71)
- `luigi-20260412-relpos-5k.pt` (val ~1.0, early-stopped)

HF `erickfm/MIMIC` also holds folders for `marth`, `peach`, `pikachu`,
`samus`, `sheik`, `yoshi`, `ganondorf` (older promotions); audit their
`metadata.json` (`run_name`/`val_loss`) before relying on them.

### Promotion policy

When a finished wandb run beats current val for its character: pull
the `.pt`, rename to the convention, upload via `tools/upload_char.py`
(bundles the norm JSONs + `metadata.json:melee_enum` that the bot's
autodiscovery needs — there is no dict to edit; the runtime-populated
`CHARACTERS` picks the new dir up on `!reload`/restart). **Skip names
starting with `SWEEP-`/`DBG-`/`DEBUG-`/`FIX-`/`BENCH-`** — infra/debug
runs aren't production candidates regardless of val. **Always confirm
with user before pushing** — never auto-promote.

### Discord-bot (re)start: audit + orphan sweep

Before `bot.run()`:
1. **Promotion audit.** List `erickfm/MIMIC` via `HfApi.list_repo_files`,
   read each char's `metadata.json` (`run_name`/`global_step`/`val_loss`),
   compare to what the bot's `_sync_hf_to_local` autodiscovery will load
   from `hf_checkpoints/`. Flag any HF entry newer than the local copy.
   Report to user, get approval — the sync itself is automatic (the
   `CHARACTERS` dict is populated at runtime, nothing to edit).
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
- `mimic/model.py` — model architecture (FramePredictor, mimic presets, Shaw relpos attention incl. sliding window and the sdpa/flex training kernels).
- `mimic/dataset.py` — StreamingMeleeDataset (per-game v2 shards).
- `mimic/frame_encoder.py` — MimicFlatEncoder (13 numeric + 5 flags per player, embeddings, controller one-hot → one Linear). Also hosts the optional `use_input_gate` L1 diagnostic.
- `mimic/features.py` — feature schema + normalization. `numeric_state(full=True)` returns 13 cols.
- The old root-level `eval.py` / `inference.py` were removed (2026-06); inference lives in `tools/play.py` / `tools/play_netplay.py`.

### Tools

**Inference (local):**
- `tools/play.py` — Run a MIMIC checkpoint vs a CPU (`--opponent cpu:9`) or a second checkpoint (`--opponent <ckpt>`) in one Dolphin instance: N back-to-back matches, `--n-matches 1` for a watchable game, optional win-rate JSON via `--out`. Uses shared `inference_utils.decode_and_press`.

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
fallback for entries missing newer transform keys).

**Diagnostics**: `tools/inspect_frame.py` (per-frame I/O dump),
`tools/extract_wavedashes.py` (wavedash-only windows for overfit
checks), `tools/validate_checkpoint.py` (per-head CE on val),
`tools/diagnose.py` (train-vs-inference tensor compare).

**Inference-faithfulness canaries** (cheap, dense, replay-only behavioral
checks against the training corpus — confirm the live Dolphin pipeline isn't
silently mistimed/corrupt; see `docs/research-notes-2026-06-29.md`):
`tools/lcancel_analysis.py` (per-move L-cancel rate, healthy ~88%; catches
systematic timing), `tools/ctrl_canary.py` (center-stick %, ~34% healthy →
100% broken, catches dropped/idle inputs; + stick-dist JS for general
corruption), `tools/state_canary.py` (action-state distribution JS). Bot is
isolated by player type (model = `type=0` human-controller, CPU = `type=1`).
Validate any canary with `tools/play.py --drop-prob P` (overwrites the bot's
input with neutral on fraction P of frames; P=1.0 = known-broken control).
The L-cancel target is **realized avoidable lag** = `max(0, realized_landing_lag
− cancelled_min[move])` (`cancelled_min` {NAIR:7,FAIR:11,BAIR:10,UAIR:9,DAIR:9}),
NOT the binary `post.l_cancel` flag (which counts ledge-cancels/early-hits as
misses) and NOT the old offset=−4 press proxy (which diverged from the truth).

**Native sim (decomp Melee, no Dolphin)**: `tools/build_native_sim.sh` (run by
`setup.sh`; builds into gitignored `sim/` — never commit or upload anything there),
`rlvr/sim/melee_sim.py` (RL env: one game per process, `reset()`/`step(pads)`, ~2.2k fps,
inputs land on f+1 like Dolphin), `tools/sim_replay_check.py` / `tools/sim_env_check.py`
(byte-exact fidelity checks vs .slp). Status and limits: `docs/research-notes-2026-10-20b.md`.
Sim RL: `tools/vr_rl_run.sh NAME "stock_delta:1,damage_delta:1" 100`. The opponent is the frozen
base. The recipe is the sim_rl dense defaults: PPO, 16,384 decisions per update × 100 updates, 4 epochs, lr 1e-5,
a critic baseline (V(s) on the frozen BC's features; `--critic-lr`, `--critic-epochs`) and a per-decision KL to BC
(β 0.03). The global KL anchor was removed 2026-10-27, the one change a greedy ablation kept. That recipe scored
180–204 vs SF7u100 over h2h seeds 41/42 at 107.3 min; a retrain of SF7's own recipe scored 181–203 at 111.3 min.
Single-run noise vs SF7u100 is ±5–10 points, so the ablation's pass/fail calls were mostly noise. The best pooled
record is no critic + anchor only, 406–362 over 2 runs × 2 seeds at 103.5 min (commit `1a91438`'s defaults).
Clearly worse: 1 PPO epoch at 4× lr (153–231 over 2 runs); probably worse: 50 updates at 2× lr (172–212).
Results are from one training seed; commit `c2df04b` reproduces SF7 (its dense defaults are SF7's recipe).
SF7 on S-full 56k won 140–52 vs the base and beat fox-master 109–83 in the sim and 130–70 in Dolphin; gains stop by
~100 updates (u150/u200 tied u100). `--resume rl_uNNNN.pt` restores the policy, AdamW and the critic; the KL reference
stays the BC model. See `docs/research-notes-2026-10-26.md`, `docs/research-notes-2026-10-27.md`. Sim h2h:
`tools/sim_h2h.py`, bf16.

**Data**: `tools/slp_to_shards.py` (.slp → v2 .pt shards with
`target[i] = buttons[i+1]` alignment); `tools/shard_and_upload_ranked.py`
(ranked .slp archive → HF tarballs).

## Porting slippistats logic

The streaming slippistats ports (`rlvr/online/slippi_stream.py`) and the
VR reward modules that used them were removed 2026-10-22 with the Dolphin
RL path; they're in git history, with port notes in
`docs/research-notes-2026-05-14b.md`. If you port slippistats logic again
(e.g. a sim reward): read the whole slippistats function, enumerate every
branch and state predicate, check enum ranges against libmelee, and don't
add filters slippistats doesn't have. The past ports kept missing the
specialized cases: the THROWN range (239-243) separate from CAPTURE, DYING
(0-10) as keep-alive, command grabs, `hitlag_left > 0` as keep-alive,
the 45-frame combo leniency, and the stock counter decrementing ~1.5 s
after the kill (gate hit-recency statefully, not with a backward scan).

## Pitfalls for agents

1. **Don't trust research notes as current truth.** Always verify
   against code. See the Research Notes section below.

2. **Don't run inference while training on the same GPU.** Frame
   drops make gameplay look broken when the model is fine.

3. **`max_samples` is total, not per-GPU.** `train.py` divides by
   effective batch size (`local_batch * n_gpus * grad_accum`) to
   compute `max_steps`. Don't reintroduce a divide-by-local-batch
   shortcut — on 8 GPUs that 8×-overtrains.

4. **Don't mix normalization schemes.** `mimic_mode` training needs
   `mimic_norm.json` + MIMIC controller combos (7-combo current,
   5-combo legacy). Old data like `ranked_fox` uses old normalization
   with 32 combos — incompatible.

5. **Don't hardcode head hidden dims as 256.** The autoregressive
   heads use `input_dim // 2`, which varies per head (256, 257, 262,
   280).

6. **`sorted()` player dicts.** melee-py's `gamestate.players` dict
   order is not guaranteed to match port order. Always `sorted()`.

7. **Use `blocking_input=True` for inference.** Dolphin waits for
   controller input before advancing each frame. Without it, slow
   model inference causes frame drops (the game advances without
   receiving input). In head-to-head, non-blocking mode systematically
   disadvantages whichever model's inputs are flushed second.

8. **TRIG (L/R) must call `press_button`, not just `press_shoulder`.**
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
   import from it rather than reimplement. A 5-class button head has
   no TRIG class, so a bot using one can't tech, airdodge, or wavedash.

9. **Button encoding is single-label.** The 5-class button head (A,
    B, Jump, Z, None) cannot represent two simultaneous action
    buttons; the 7-class head adds TRIG + A_TRIG for the one emergent
    combo that matters (airdodge/wavedash + an A-attack shield-grab
    interaction). Multi-button overlaps (2.65% of frames) are
    collapsed via a stateless per-frame priority cascade mirroring
    Melee's input resolution (B overrides all; A+TRIG → A_TRIG; else
    Z > A ≈ TRIG > JUMP) — see "Button encoding" in the Data section.
    Early-release encoding is legacy 5-class only. Shoulder+button
    combos ARE representable since shoulder is a separate head.

10. **Only Shaw relpos attention exists.** RoPE variants underperformed
    relpos (−1–2% val at matched settings, 2026-10-21) and were removed
    with the other position encodings on 2026-10-22; recover them from
    git history if a future experiment needs them.

11. **Full features are the only input.** The full path (13 numeric +
    5 flags per player) was worth ~1.5–3.5% val over the old 9-col
    minimal path, which was removed 2026-10-22.

12. **Discord bot portability: keep paths relative in `.env`.** The
    bot's `.env` uses relative paths (`./emulator/...`, `./melee.iso`,
    `./slippi_home`) that `_resolve_path` in `tools/discord_bot.py`
    converts to absolute against the repo root at runtime. This makes
    the repo `scp`-able to any machine that has run `setup.sh`. Don't
    hardcode absolute paths.

13. **Slippi credentials live at `./slippi_home/Slippi/user.json`**
    (gitignored). Not at `~/.config/SlippiOnline/Slippi/user.json` —
    libmelee is pointed at the bundled dir explicitly via
    `dolphin_home_path=SLIPPI_HOME` in `tools/play_netplay.py`. Place
    `user.json` in the repo so uploading the repo to a new machine
    carries the bot's Slippi login. Never commit `slippi_home/` — it
    contains the bot's `playKey`.

14. **Dolphin needs runtime shared libraries.** The AppImage-extracted
    `dolphin-emu` binary links against `libasound2`, `libusb-1.0-0`,
    `libgtk-3-0`, `libbluetooth3`, `libhidapi-hidraw0`, and friends.
    Missing any of them makes the binary exit 127, which libmelee
    surfaces as `RuntimeError: Unexpected return code 127 from
    dolphin` inside `Console.__init__` — `play_netplay.py` then exits
    1 with an empty `RESULT:` line and the Discord bot reports
    `result=failed score=`. `setup.sh` installs the full list; on
    existing machines run `ldd emulator/squashfs-root/usr/bin/dolphin-emu
    | grep 'not found'` to see what's missing.

15. **Setup Xvfb for headless machines.** Dolphin crashes at startup
    with "Unable to initialize GTK+, is DISPLAY set properly?" if no
    display server is available. `setup.sh` installs and starts Xvfb
    on `:99` and adds `export DISPLAY=:99` to `~/.bashrc`. On
    existing machines, check `DISPLAY` is set in the environment the
    Discord bot / `play_netplay.py` inherits.

16. **Use `gfx_backend="Vulkan"` on headless/containerized hosts.**
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

17. **FFW (`emulator_ffw/` + `--use-exi-inputs --enable-ffw`) IS
    gameplay-faithful — the old "unfaithful" claim was a misdiagnosis
    (corrected 2026-06-30).** With `blocking_input=True` (our default,
    pitfall #7) the canary suite passes: L-cancel rate **91.5%** (n=588)
    vs realtime 88% / corpus 90.4%, match length **normal** (~6,400
    frames), at **~2.9× realtime**. Faithful by construction: Fizzi's
    `AI/LoopMainEngine/*` gecko codes run the *real* engine loop N× per
    video frame (NOT frame-skipping), and blocking input makes Dolphin
    wait for the bot on every engine frame → one decision per game
    frame, just faster wall-clock. Our setup is byte-for-byte
    slippi-ai's (`use_exi_inputs + enable_ffw + blocking_input`, same
    Fizzi codeset in `emulator_ffw/.../GALE01r2.ini`). The 2026-05-18
    "matches ~4× short" result was inferred from **match length in a
    bot-vs-bot run** — the symmetric-degradation trap (both bots equally
    hobbled → faster mutual deaths → short matches, no actual
    infidelity; most likely that run wasn't blocking-input). **Do NOT
    diagnose emulator fidelity from match length or win-rate** (both
    survive symmetric degradation) — use the canaries
    (`tools/lcancel_analysis.py` etc.), which measure input timing
    directly. **Parallel caveat:** N independent `play.py` processes do
    NOT scale (GPU-bound on batch-1 inference — 1→4 instances buys only
    2.9→3.5× total, GPU pins at 90% while CPU idles). Real throughput
    needs **batched multi-env inference** (one process, N Dolphins, one
    batched forward) = slippi-ai's `AsyncBatchedEnvironment` = the RLVR
    rollout harness (the Dolphin RL actors were removed 2026-10-22; RL
    now runs in the native sim). See
    `docs/research-notes-2026-06-30.md` (supersedes the FFW claims in
    `docs/research-notes-2026-05-18.md`). **RNG caveat (2026-10-20):**
    FFW does not reproduce realtime Melee's RNG draw sequence — FFW h2h
    replays desync RNG within ~100s of frames where realtime games stay
    in step all match (measured with the native sim). Timing/physics
    canaries are unaffected; don't treat FFW replays as bit-exact.

18. **Netplay runs on the CURRENT Slippi Ishiiruka in `emulator/`** —
    `setup.sh` downloads the latest release at install time (the old vendored
    `emulator.tar.gz` was removed 2026-07-17) and `DOLPHIN_PATH` defaults to
    `emulator/squashfs-root/usr/bin/dolphin-emu`. Historical note: an earlier
    version of this pitfall recommended *mainline* Dolphin because the stale
    Ishiiruka 3.5.2 build stalled libmelee's `choose_direct_online` menu
    navigation — that advice is dead (see #19: mainline crashes under our
    libmelee EXI protocol or is gate-rejected; current Ishiiruka navigates
    fine and is what the Phillip benchmark ran on). Still true:
    `play_netplay.py` forces the `fork` mp start-method (macOS/Windows
    `spawn` re-imports the unguarded script and double-runs it), and the host
    must pass **inbound P2P UDP** — RunPod-style container pods drop it
    (STUN falsely reports "cone"); use a real VM (e.g. GCP).

19. **"required update available" on netplay = a stale Slippi build, NOT a
    Dolphin-fork problem.** Slippi gates Online on a server-side *minimum
    version*. When the bundled build falls behind (the 3.6.x Ishiiruka line
    shipped after our 3.5.2 and became required), Online is hard-gated. Fix =
    update the build **in place** to the current Slippi release — download
    `Slippi_Online-x86_64.AppImage` from the latest `project-slippi/Ishiiruka`
    release, `--appimage-extract` it over `emulator/squashfs-root`. Do **not**
    chase mainline beta to dodge this: mainline `beta.19` *crashes* under our
    `libmelee 0.45.1` EXI protocol (`EnetDisconnected`), and `beta.17` runs but
    is itself gate-rejected. **To watch a netplay session on a GUI box** (e.g.
    supervising the PHAI#591/Phillip benchmark), pass `--gfx-backend OGL`
    (Ishiiruka rejects the mainline-only `Null`; the flag defaults to `Null` so
    the headless deployment is unchanged); Ishiiruka opens its own window — no
    `--platform` (mainline-only; its Qt X11 plugin is named `xcb`, not `x11`).
    **Debugging trap:** a launch script that runs `pkill -f play_netplay.py`
    SIGKILLs its *own* shell (the script's argv contains that string), dies
    before launching, and leaves a stale log you'll misread as a bot crash. Only
    `pkill -x dolphin-emu` (exact NAME) is safe. See the memory note
    `reference_netplay_slippi_setup` for the full recipe.

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

- "Architecture: 26,274,803 params" — actually ~19,950,000 params.
- Various "this matches the reference" claims that later turned out to
  have subtle differences.

The notes are still valuable for understanding the project's
evolution and the reasoning behind decisions. Just don't treat
specific numbers or "verified" claims as current truth without
checking the code.

## HAL leftovers

MIMIC's backbone came from [HAL](https://github.com/ericyuegu/hal) (Eric Gu). What's left:

- **`tools/run_hal_model.py`** reimplements HAL's 5-class inference and loads only Eric Gu's
  original checkpoint, `/home/erick/projects/hal/checkpoints/000005242880.pt` (101 MB,
  DDP-prefixed, best at 5.2M samples). Not on any MIMIC path; it doesn't use `inference_utils`.
  Keep it; don't rename or delete it. It needs `hal/checkpoints/stats.json` (Fox subset,
  p1_percent max 236), not `hal/data/stats.json`: HAL's Preprocessor always loads the former
  regardless of `override_stats_path`, and the wrong file shifts every normalized percent. If it
  breaks, HAL's own code works: `cd /home/erick/projects/hal && python3 -m hal.eval.play
  --artifact_dir checkpoints --character FOX` (needs correct paths in `hal/local_paths.py`).
- **`tools/validate_checkpoint.py`** has an inner `HALModel` class for validating old HAL
  checkpoints. Leave it.
- **Names:** the `--stick-clusters hal37` value, the `--encoder hal_flat` alias, `hal_*` config
  fields (migrated in `ModelConfig.__post_init__`), and legacy data dirs `data/fox_hal_*` (frozen,
  leaked alignment) keep their names. Research notes mention `--hal-mode`, `hal_norm.json`, the
  "HAL preset": frozen snapshots, don't sweep them.

## Environment

### Remote GPU

| Machine | Host | Port | User | GPU | Storage | Status |
|---------|------|------|------|-----|---------|--------|
| A | 74.2.96.39 | 11639 | root | 2× RTX 5090 (32 GB ea) | 2 TB `/workspace` | Active |

```bash
ssh mimic-train                       # Machine A (alias in ~/.ssh/config)
ssh -p 11639 root@74.2.96.39          # Machine A (2× RTX 5090)
```

Port changes on each reprovision; the box also needs your SSH **public key**
re-added to `~/.ssh/authorized_keys` after a fresh provision (publickey auth
fails otherwise).

**Persistent storage is `/workspace` ONLY** — everything outside `/workspace`
(e.g. `/root`, `/tmp`) is **ephemeral** and wiped on container restart. Clone the
repo, store data, checkpoints, and logs under `/workspace/MIMIC`. On a fresh box,
CUDA may report "unknown error" / `is_available()=False` until `nvidia-uvm` is
loaded — run `nvidia-modprobe -u -c=0` (creates `/dev/nvidia-uvm`).
