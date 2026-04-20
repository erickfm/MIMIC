# Research notes — 2026-04-20

## tl;dr

1. **Capacity ceiling confirmed.** `mimic-xxl` (154M params, 8× the
   baseline's weight footprint, 4× the wall-time) buys ~2-3% val-loss
   on puff / ice_climbers vs the plain `mimic` 20M baseline. Decision:
   production recipe pivots back to `mimic` baseline. xxl is parked in
   `MODEL_PRESETS` for future ablations but not used in the 8-char
   retrain.
2. **Full feature set exposed** (`--mimic-minimal-features` dropped). The
   encoder now honors the flag after the 2026-04-19 bug fix; dropping
   it gives the model the 18-dim per-player gamestate (velocity, hitlag,
   hitstun, flags) instead of the 9-dim minimal. Worth ~1-3% val-loss
   per char on paper.
3. **Schema pruned to what's actually in the data.** Dropped
   `invulnerability_left` and all 8 ECB corners (`ecb_{bottom,left,
   right,top}_{x,y}`) — libmelee never populates them for our .slp
   format. Verified directly from shard tensors: 2.5M frames all
   identically zero. L1-input-gate ranking independently drove these
   features to the sparsity floor. Numeric schema: **22 → 13 cols**.
4. **Principled transforms** for the features that used to be z-scored
   against a near-identity fallback (velocity, hitlag, hitstun). New
   transforms: `tanh_scale` (signed, heavy-tailed), `linear_max`
   (bounded nonneg), `log_max` (zero-inflated heavy-tailed nonneg).
5. **Infrastructure cleanup** — `build_norm_stats.py` parallelized via
   `mp.Pool` (30 min → 11-20 min on the 112-core box);
   `retrain_all_baseline.sh` with HF_TOKEN export, patience=12
   watchdog, mimic_norm.json rebuild gate independent of norm_stats.
6. **8-character retrain** running on the new schema + transforms.
   First 4 results mixed — large-dataset chars slightly regressed,
   mid-dataset chars improved. See results table at bottom.

## What actually changed on disk

All changes are coupled — the schema drop, the new transforms, and
the encoder dimension change all have to land together, because the
model's input projection dimension is fixed at load time and any mismatch
between training and inference fails loudly.

### `mimic/features.py`

`numeric_state(full=True)` now returns 13 columns (was 22). The 9
dropped columns are the ones libmelee can't fill:

- `invuln_left` — `PlayerState.invulnerability_left` is declared in
  libmelee but never assigned by any parser. Dead at the library level.
- 8 ECB corners — `console.py` reads them from Slippi event bytes at
  offsets 0x4D-0x65, but our `.slp` files don't have those bytes, so
  libmelee silently falls back to 0. Dead at the data level.

`mimic_normalize` / `mimic_normalize_array` pick up three new transform
types alongside the HAL-inherited `normalize` / `standardize` /
`invert_normalize`:

- `tanh_scale` — `tanh(x / scale)`. Signed, saturates smoothly,
  preserves sign. Used for velocities.
- `linear_max` — `x / max`. Bounded nonneg. Used for hitlag.
- `log_max` — `log1p(clamp(x, 0, max)) / log1p(max)`. Heavy-tailed
  nonneg where exact low values matter but the tail doesn't. Used for
  hitstun.

### `tools/build_mimic_norm.py`

`MIMIC_TRANSFORMS` is now a dict-of-dicts with per-transform parameters.
The new entries with their justifications inline:

| feature | transform | parameter | reason |
|---|---|---|---|
| `speed_air_x_self`, `speed_ground_x_self`, `speed_y_self` | `tanh_scale` | `scale=5.0` | covers 2× Fox dash (2.5). walk (0.3) → 0.06, run (2.5) → 0.46, fast-fall (3) → 0.54, extreme (>8) → saturated toward 1 |
| `speed_x_attack`, `speed_y_attack` | `tanh_scale` | `scale=10.0` | preserves discrimination across full hit-severity range. small (5) → 0.46, medium (10) → 0.76, kill (25) → 0.99. scale=5 would saturate before kill-range |
| `hitlag_left` | `linear_max` | `max=20.0` | Melee caps hitlag at ~20 frames. "in hitlag" binary matters more than exact frame count |
| `hitstun_left` | `log_max` | `max=120.0` | values go up to ~100 on kill-level hits. log compresses the tail (60 vs 80 vs 100 all "still combo'd") while keeping 0 / 5 / 15 distinguishable |

### `mimic/frame_encoder.py`

`MimicFlatEncoder.__init__` now actually honors `mimic_minimal_features`
(was silently ignored before 2026-04-19). `per_player = 9 if minimal
else 18` (was 27 before the schema drop). The minimal path stays
shard-width-aware (`_IDX` dispatch for 22/13/7-col shards) so
pre-schema-drop minimal-features checkpoints still load unchanged.

Fullfeat path requires a 13-col numeric input and asserts loudly on
22-col (since those old fullfeat shards encode the now-dropped cols
inline and the projection dimension would be wrong).

### `tools/slp_to_shards.py`

Schema writes dropped for `invuln_left` and the 8 ECB corners, both on
the self and nana numeric groups. Everything else preserved.

### `tools/build_norm_stats.py`

Parallelized via `mp.Pool(min(cpu_count, 64))`. The 5000-file metadata
pass drops from ~30 min (single-core on the 112-core box — oops) to
~11-20 min depending on per-char .slp throughput. New `--workers`
flag; `--workers 0` = auto.

### `tools/inference_utils.py`

`MIMIC_NUM_FULL` shrunk to 13 entries (was 22). `XFORM` gained
`_tanh`, `_linear_max`, `_log_max` helpers mirroring
`MIMIC_TRANSFORMS`. `_player_numeric_full` reads the live libmelee
`PlayerState` fields (`.speed_air_x_self`, `.hitlag_left`,
`.hitstun_frames_left`, etc.) and normalizes them through the per-char
`mimic_norm.json`, falling back to z-score via `norm_stats.json` for
any feature the loaded char's mimic_norm doesn't cover (for backward
compat with checkpoints from before the transforms landed).

### `tools/retrain_all_baseline.sh`

The orchestrator. Queues 8 chars, per-char pipeline downloads from
`erickfm/melee-ranked-replays`, extracts, builds metadata
(norm_stats + mimic_norm), writes 7-class controller_combos, shards
with the current schema, trains with patience=12 val-plateau watchdog,
uploads `_bestloss.pt` to `erickfm/MIMIC/<char>/`, cleans raw + shards.
Resume state in `checkpoints/retrain_baseline_state`. HF_TOKEN
exported from `.env` to avoid anon rate limits on the ~150 tar
downloads.

## Methodology: why we believe the schema drop is safe

Three independent lines of evidence pointed at `invuln_left` + 8 ECB
corners being dead:

1. **Direct tensor inspection.** `torch.load('data/luigi_v2/train_shard_000.pt')` →
   2.5M frames, every one of the 9 cols all zero, across both players.
2. **libmelee source audit.** `PlayerState.invulnerability_left` declared
   but never assigned. `console.py` reads ECB bytes at offsets that
   don't exist in our .slp format; silently zeroes them.
3. **L1 input-gate report** from the 2026-04-19 puff run at
   λ=0.01. The input projection has a per-column sigmoid gate penalized
   to sparsity; these 9 cols' gates all collapsed to the sparsity floor
   within the first eval, confirming the model was getting zero signal
   from them.

All three agree. Dropping them means the input projection shrinks from
202→512 to 184→512 (a 5.6% parameter reduction in the projection
layer) with no loss of information.

## Methodology: why these specific transforms

The old pipeline handled features outside the 9-entry `mimic_norm.json`
via a z-score fallback computed from `norm_stats.json`'s per-column
`(mean, std)` — but with `std` floored to 1.0 to avoid division blowup.
For velocity that floor meant the z-score collapsed to near-identity:
typical values are 0-5, the floored "std" is 1, so `x/1 ≈ x` and the
dynamic range the model saw was whatever the raw unit was. Not a
normalization at all.

For velocity specifically we want:

- **Sign preservation.** Which direction the unit is moving is
  load-bearing for combo routing; sign-lossy transforms (e.g., `|x|`)
  throw away structure.
- **Soft saturation.** Dash speeds top out at ~2.5, knockback
  velocities in the 5-20 range depending on hit severity. A linear
  scale picked for dash speeds would saturate before knockback is
  meaningfully represented; a scale picked for knockback would quantize
  dash too coarsely. `tanh` saturates *smoothly* instead of clipping —
  small differences stay distinguishable even in the tail.
- **Different scale for self-motion vs hit-motion.** Self-velocity sits
  well under 5 in 99% of frames; `scale=5` puts walk at 0.06, run at
  0.46. Hit-velocity routinely hits 10-20 on kill-level hits; `scale=10`
  puts medium knockback at 0.76, kill-range at 0.99 without clipping.

For `hitlag_left`, Melee's engine caps it at ~20 frames by construction.
`x/20` is fine — the value is bounded, the frame count is roughly
linear in the feature we care about, and the "in hitlag" signal
(anything > 0) is just as useful as the exact number.

For `hitstun_left`, real-world values run 0-100+. The distinction
between hitstun=60 and hitstun=80 isn't actionable — both say "still
combo'd, don't input anything meaningful yet." But the distinction
between 0 / 5 / 15 is very actionable (first-actionable-frame timing).
Log compression gets us both: the low end stays linear-ish, the tail
saturates.

These pair of choices — drop the dead cols, give the live cols
real normalizations — should be a strict improvement over the
z-score fallback. Whether they're enough of an improvement to beat the
old minimal-features baseline on every character is a separate
question, addressed below.

## Retrain queue results (in progress)

All 8 chars run on **identical HPs**: `mimic` preset (20M, relpos,
d=512/L=6/H=8, GELU, LN), seq_len=180, dropout=0.2, bs=256 × 2 GPUs ×
grad_accum=1 = eff-batch 512, LR 3e-4 cosine→1e-6 no-warmup,
max-samples=16.78M (≈ max_steps 32768), BF16 AMP + torch.compile,
patience=12 val-plateau watchdog, v2 shards with
`--self-inputs --reaction-delay 0`.

Data for each char is the same *source* as the old baselines —
`erickfm/melee-ranked-replays` master-master + master-diamond +
master-platinum tars, same quality filters (≥1500 frames, damage,
completion) — but reshardedto the new 13-col schema. Train/val split
is 0.9/0.1 seeded 42, regenerated per char.

Old baselines (all `--mimic-minimal-features`, pre-schema-drop,
pre-transforms, pre-watchdog):

| char | dataset (train games / frames) | old run | old val | old step |
|---|---|---|---|---|
| fox | 31,030 / 290.4M | `qeka6rq8` | 0.7081 | ~30k |
| falco | 20,882 / 197.6M | `zb1vhjxs` | 0.7448 | ~28k |
| marth | 11,759 / 118.5M | `eo8yjem4` | 0.6746 | ~30k |
| sheik | 51,751 / 552.5M | `jc4xe4dv` | 0.6611 | ~28k |
| cptfalcon | 17,557 / 158.2M | `6k1x8xdi` | 0.7356 | ~27k |
| luigi | ~2K / ~20M | legacy | ~1.00 | early-stopped |
| puff | ~15K / ~140M | 2026-04-19 gate01 | 0.6641 | fullfeat |
| ice_climbers | ~5K / ~50M | xxl, overfit-killed | 0.6817 | ~3k |

New baselines as they land:

| char | new run | new val | old val | delta | final step | notes |
|---|---|---|---|---|---|---|
| fox | `fox-20260420-baseline` | 0.7144 | 0.7081 | **+0.9%** | 32768 | full run, no watchdog |
| falco | `falco-20260420-baseline` | 0.7487 | 0.7448 | **+0.5%** | 31392 | near-end stop |
| marth | `marth-20260420-baseline` | **0.6664** | 0.6746 | **−1.22%** ✓ | 31065 | |
| sheik | `sheik-20260420-baseline` | **0.6566** | 0.6611 | **−0.68%** ✓ | 26160 | early finish, likely watchdog |
| cptfalcon | `cptfalcon-20260420-baseline` | (training) | 0.7356 | — | — | |
| luigi | (queued) | | | | | |
| puff | (queued) | | ~0.6641 | | | peach-pattern comparison only |
| ice_climbers | (queued) | | ~0.6817 | | | |

### What the first four results seem to say

The pattern so far — two mild regressions (fox +0.9%, falco +0.5%) and
two real improvements (marth −1.22%, sheik −0.68%) — is counter to
the plan's prediction of uniform 1-3% improvement across chars. A few
hypotheses, in decreasing order of probability:

1. **Sample variance.** Each char's "old baseline" is a single run.
   With patience=12 and the old runs not having a watchdog, which
   eval-step the old best landed on is noisy. A 0.5% delta might just
   be one eval cycle of luck. Without repeated seeds we can't tell.
2. **Large-dataset diminishing returns.** Fox (31k games) and falco
   (21k) have enough data that the old minimal encoder was already
   extracting most of the learnable signal from the 9-dim gamestate.
   The extra features (velocity, hitlag, hitstun) help less when
   implicit correlates (action-state + position deltas across 180-frame
   windows) already carry similar information. Marth (12k) and sheik
   (52k) don't fit this clean — sheik is the largest and it still won.
3. **Transform choice mismatched to char.** The tanh scales were
   picked from per-character physics (Fox dash speed 2.5 → scale 5).
   Other chars have slower run speeds (marth 1.5, puff 0.8), so the
   scale=5 put their self-velocity at low-end-of-tanh range with more
   headroom. Chars with faster natural movement might be losing
   information to saturation. Not sure this is actually the story
   — scale=5 hits tanh(2.5/5) = 0.46, well inside the linear regime
   — but plausible to check.
4. **Peach was anomalously good.** The −8% peach win that motivated
   this retrain was on a different data vintage, pre-schema-drop (22
   cols). Dropping 9 dead cols doesn't hurt, but maybe the pre-drop
   run *benefited* from the extra noise as implicit regularization,
   and peach's −8% wouldn't survive resharding. Can't test without
   retraining peach under the new schema (which is the plan's
   follow-on item).

Three results outstanding (cptfalcon, luigi, puff, ice_climbers). If
the remaining results track the pattern of the first four,
interpretation stays mixed — we've made the pipeline cleaner and more
correct, but the val-loss gain isn't a slam-dunk.

### What's definitely better regardless of val-loss

- **Inference path is cleaner.** Nine permanently-zero columns no
  longer flowing into the encoder means nine fewer places for a subtle
  normalization bug to hide.
- **New transforms are more defensible than the z-score fallback** for
  the features that used to hit it. Even if marginal val-loss is
  mixed, future work on velocity / hitstun features has a sane
  baseline to iterate from.
- **Watchdog caught sheik at step 26160** (vs 32768 full runs on
  others), which saves some compute and suggests we could push
  patience lower if we want faster iteration. Not reducing patience in
  this cycle.
- **Infrastructure**. `build_norm_stats.py` is now fast enough that
  re-bootstrapping a character from scratch is under 20 min for the
  stats step (was 30+ min).

## Files touched

    mimic/features.py
    mimic/frame_encoder.py
    mimic/model.py
    tools/slp_to_shards.py
    tools/build_mimic_norm.py
    tools/build_norm_stats.py
    tools/inference_utils.py
    tools/retrain_all_baseline.sh
    tools/upload_char.py
    tools/discord_bot.py
    CLAUDE.md

Related earlier doc: `docs/research-notes-2026-04-19.md` (scale sweep +
minimal-features bug fix that set up this cycle).

## Follow-ons (not in this cycle)

- Ablation: same char with fullfeat + new transforms vs fullfeat + old
  z-score fallback. Isolates whether the transforms are pulling their
  weight independent of the schema drop.
- Retrain peach under the new schema + transforms. Currently peach is
  the only char on the pre-drop 22-col schema; all 9 should sit on the
  same basis for fair comparison.
- Repeated seeds on one char (fox or marth) to get a variance estimate
  on the "0.5% regression" noise floor.
- Re-measure L1 input-gate rankings on the new schema. If velocities
  now rank higher (as the new transforms should enable), that's
  independent confirmation.
- Push seq_len > 180 once the schema/data pipeline has fully settled.
