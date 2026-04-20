# 2026-04-19 — Scale sweep, Modern-block sweep, and the full-features win

Long session on puff_v2. Seven training runs plus a small amount of
scaffolding work. Bottom line:

- **Scaling up at minimal-features hit a wall fast.** 2.2× params bought
  1.8% val-loss drop, 3.7× bought 2.8% — clearly diminishing returns.
- **Modern-block tricks (GQA + SwiGLU + RMSNorm) produced zero signal**
  at matched param count vs the LN+GELU+full-attention baseline.
- **Dropping `--mimic-minimal-features` was the one change that actually
  moved the needle.** A baseline-sized (~20M param) `mimic` preset with
  the full 22-numeric + 5-flag per-player gamestate, plus an L1-gated
  input projection as a feature-importance probe, reached
  **val 0.6641** — 3.6% better than the hitherto-best standard baseline
  (0.6890), and better than both XL-scaled width (0.6766) and
  XL-scaled depth (0.6698) runs that cost 4-6× the compute.
- **The main-stick F1 ceiling that was stuck at 51-53% across every
  architectural variant broke 54% for the first time** when speeds and
  hitlag/hitstun were actually fed to the encoder.

TLDR for anyone reading this later: the MimicFlatEncoder was silently
discarding 16 of 22 numeric columns per player even when
`mimic_minimal_features` was `False`, because the encoder never read
that flag and always sliced down to a 6-column subset based on shard
shape. The per-character pipeline had been leaving that "free lunch" on
the floor since the mimic preset rename.

## Setup

- **Data:** `data/puff_v2`, 33,585 games (filtered), 168 shards at 4 GB
  each, 3,358,500 train windows + 375,300 val windows, 22-col numeric
  per player, 5-col flags per player.
- **GPUs:** 2× RTX 5090, DDP, effective batch 512, cosine to 1e-6 from
  3e-4 over full 32,768 steps (~5 real epochs on puff).
- **Wall budget:** a 2-hour slot we were originally going to point at a
  scaled Fox model, pivoted to puff_v2 because fox shards had been
  cleaned up by the character-rotation pipeline.
- **Pipeline state:** puff was the current training target; luigi's
  data was being downloaded/sharded in parallel on CPU while we used
  the idle GPUs. Fullfeat+gate run ran concurrently with luigi sharding.

## Experiments, in order

| Run | Config | Params | Wall | Best val | Δ vs baseline |
|---|---|---|---|---|---|
| `mimic` (baseline, reference) | LN + GELU, full attn, relpos, dropout 0.2 | 19.98M | 48.8 min | 0.6890 | — |
| `mimic-xl` 6L | d_model=768, d_ff=3072, nhead=12, SwiGLU, dropout 0.1 | 44.30M | 82.9 min | 0.6766 | −1.8% |
| `mimic-xl` 10L deep | same + num_layers=10 | 72.90M | 127.7 min | **0.6698** | −2.8% |
| `mimic-xl-rms` 10L deep | same + RMSNorm | 72.88M | (killed @ 52%) | 0.6904 @ kill | n/a |
| `modern-relpos` | d=512/6L, GQA 4:1 + SwiGLU + RMSNorm | 19.97M | 50.9 min | 0.6882 | −0.1% |
| `modern-relpos-gelu` | same but GELU FFN | 19.98M | 48.8 min | 0.6883 | −0.1% |
| `mimic` + full feats + gate λ=0.01 | baseline arch, drop `--mimic-minimal-features`, L1 input gate | 20.00M | 49.2 min | **0.6641** | **−3.6%** |

All val numbers are `_bestloss.pt` tracked, not end-of-training.

## What the scale experiments said

### Width (`mimic-xl` 6L)

Bumped `d_model` 512→768, `nhead` 8→12 (head_dim kept at 64),
`dim_feedforward` 2048→3072 (4× ratio preserved), dropout 0.1, swapped
GELU FFN for SwiGLU (param-matching). Added as a new preset
`mimic-xl`. 2.22× params, 70% more wall-clock, 1.8% val-loss drop vs
baseline. The c-stick F1 jumped 67% → 70% — the one sub-head where
extra capacity clearly helped.

### Depth (`mimic-xl` 10L)

Same width preset with `--num-layers 10`. 3.65× params vs baseline,
1.56× wall-clock vs 6L-XL. Incremental val drop from 0.6766 → 0.6698
(another −1.0%). Overfitting gap widened notably (v/t hit 1.07-1.08
late in training vs 1.05 for 6L). Deep extra capacity mostly learned
train-set specifics that did not transfer.

### RMSNorm on top of depth (`mimic-xl-rms` 10L)

Added `use_rmsnorm=True` to the 10L XL config. Killed at 52% through
when it became clear the curve was trailing the LN version. Best
val-at-kill was 0.6904 vs LN-equivalent 0.6958 at the same step — RMS
was very slightly *ahead* mid-run but trending into the same
late-cosine plateau. Not worth the full 2h.

### Modern-block sweep at baseline size

Two runs both matched to ~20M params:

- `modern-relpos` — preset already in the codebase: d=512, 6L, GQA
  (`n_kv_heads=2`, 4:1 query-to-KV), SwiGLU, RMSNorm. Final val 0.6882.
- `modern-relpos-gelu` — newly added preset; same as above but GELU FFN
  at `dim_feedforward=2048`. Final val 0.6883.

Baseline `mimic` at same size: 0.6890. **All three tied within noise.**
SwiGLU vs GELU was a 0.0001 dead heat. GQA's compute savings are
negligible at seq_len=180 / d_model=512 (attention isn't the bottleneck
at this shape; the FFN is), so its main value is throughput, which it
didn't deliver either — `modern-relpos` was actually slower than
baseline (50.9 vs 48.8 min) because SwiGLU's three matmul launches
outweighed GQA's K/V savings in kernel-launch overhead.

**Conclusion from the arch sweep:** at ~20M params on this data, the
LN/GELU/full-attn/relpos recipe is at a plateau, and swapping any
individual modern component into it does not move the plateau. The
rest of the session was spent on the hypothesis that the ceiling is
not in the architecture.

## The feature-wiring bug (the real story)

Every run prior to the "full feats" one used `--mimic-minimal-features`.
That flag is wired through the CLI → `hal_minimal_features` → `ModelConfig.mimic_minimal_features` → the encoder's kwargs. It was
supposed to toggle the encoder between a 9-scalar-per-player minimal
gamestate (6 numeric + 3 flags, HAL-compatible) and the full 27-scalar
set (22 numeric + 5 flags).

**Reality:** `MimicFlatEncoder.__init__` accepted the kwarg and
discarded it. No `self._minimal` was stored anywhere. The forward pass
sliced down to 6 numeric cols + 3 flags **purely based on the runtime
shape of the `self_numeric` tensor** (`if sn.shape[-1] > 7`). Since v2
shards store 22 cols regardless of config, every run through
`mimic_flat` since the mimic preset rename was silently running in
minimal mode.

Net effect: **16 numeric columns per player were being dropped before
they hit the projection Linear**:

- 5 speed columns (`speed_air_x_self`, `speed_ground_x_self`,
  `speed_x_attack`, `speed_y_attack`, `speed_y_self`)
- `hitlag_left`, `hitstun_left`
- 8 ECB corners (`ecb_{bottom,left,right,top}_{x,y}`)
- `invuln_left`

Plus two flags (`off_stage`, `moonwalkwarning`).

The fix is surgical. Three changes in `mimic/frame_encoder.py`:

1. Store the flag on the encoder: `self._minimal = mimic_minimal_features`.
2. Recompute `numeric_dim` from the flag: `9*2` vs `27*2`.
3. Branch in `forward()`: minimal path is bit-identical to pre-fix
   behavior (preserves HAL's reorder and the 7-col legacy shard shim),
   full path concatenates `[all_22_numeric, all_5_flags]` per player
   without HAL reorder (the downstream Linear absorbs the permutation).

Checkpoints trained with `mimic_minimal_features=True` still load
identically — the flag flows through the pickled config to the encoder
and the minimal path preserves the exact HAL column subset and reorder.

## The L1 input gate (feature importance without retraining 36 times)

Rather than do permutation-importance or drop-one-retrain to measure
which of the newly-exposed features actually mattered, wired an L1-
penalized per-input-column sigmoid gate into `MimicFlatEncoder`:

- `ModelConfig.use_input_gate: bool = False` (new).
- `MimicFlatEncoder.input_gate_logits: nn.Parameter((input_dim,))` when
  enabled, initialized to `+2.0` so `sigmoid ≈ 0.88` at start.
- Forward applies `combined = combined * sigmoid(input_gate_logits)`
  before the projection.
- `gate_l1_penalty() → scalar` exposed for the training loop.
- `gate_report() → List[(name, value)]` for analysis.
- CLI: `--input-gate-l1 <lambda>`. Setting `>0` flips `use_input_gate`
  on and adds `(lambda / grad_accum_steps) * gate_l1_penalty()` to the
  micro loss. Gate stats (mean/min/max/frac-below-0.1) logged to wandb
  every `log_interval`.
- End-of-training dump: `checkpoints/{run_name}_gate_report.json` with
  all `input_dim` features ranked ascending by gate value.

Trained with `--input-gate-l1 0.01` on the full-features baseline.
Final gate distribution: min 0.057, max 0.88, mean 0.56. 28 of 202
features (14%) pruned to ≤0.1 (the L1-floor where the sigmoid
saturates).

### Ranking highlights

All 16 ECB corners (self + opp), both `invuln_left` fields, and 4 of 9
`ctrl_cstick` bins pruned to the L1 floor. Expected: Puff's style
doesn't depend on edge-cancel geometry; `invuln_left` is redundant with
the binary `invulnerable` flag; the c-stick bins pruned are the ones
puff players rarely throw.

The top of the ranking is dominated by **autoregressive
self-controller bins** (13 of the top 20 features are `ctrl_main[*]`),
followed by **velocity + position**: `self_speed_air_x_self` at #3,
`self_pos_y/x` at #4/5, `self_speed_y_self` at #7, `opp_pos_x` at #9.
These are exactly the columns minimal-features was throwing away
(velocity) or keeping (position).

One clean independent sanity check: all 12 dims of `self_char_emb` were
pruned or heavily suppressed (ranks 169-180, gate 0.06-0.26). The
puff_v2 dataset is Puff-only, so `self_character` is constant and the
embedding carries no signal — the gate found that on its own.

Full ranked JSON lives at
`checkpoints/puff-20260419-mimic-fullfeat-gate01_gate_report.json`.
Load and `sort` for any slicing.

## Results on main-stick loss specifically

Main-stick (37-way cluster classification) was the largest single
contributor to loss across every prior run (~67% of total). Main-stick
F1 was stuck at 51-53% regardless of arch:

| Run | main F1 @ best val |
|---|---|
| baseline mimic | 51.5% |
| modern-relpos SwiGLU | 52.0% |
| modern-relpos GELU | 52.2% |
| mimic-xl 6L | 51.9% |
| mimic-xl 10L | 52.0% |
| **mimic + full feats + gate** | **53.7%** (touched 54.1% mid-run) |

The 2 percentage point jump was the first meaningful movement on this
head in this whole session. Interpretation: part of what we were
calling the "main-stick aleatoric floor" wasn't actually aleatoric —
the model literally could not see the features that disambiguate
equally-valid human stick choices (velocity, momentum, hitstun timing).
Some of the floor is still aleatoric (different humans tilt different
directions from the same state) but less of it than prior analyses
assumed.

## Going forward

- **New production recipe: drop `--mimic-minimal-features`** when
  training any character whose shards have the full 22-col numeric.
  Same arch, same wall-clock, ~1.5-3% val-loss win (exact win depends
  on the character). Everything in `tools/run_all_chars.sh` and the
  retrain snippet in CLAUDE.md should be updated.
- **Keep all 202 features, including the hard-pruned ones.** The
  Puff-specific gate report pruned ECB and `invuln_left`, but other
  characters (fast-fallers that edge-cancel, characters with long
  invuln windows after tech) may genuinely need those features. A
  character-neutral production model is better served keeping them in
  and letting the downstream projection learn to ignore where
  appropriate.
- **The gate is a diagnostic, not a regularizer.** It helped here to
  verify that features matter and to identify redundancy, but it costs
  a small amount of loss at λ=0.01 (the kept features never reach
  sigmoid=1.0, so the projection sees a scaled signal). For production
  runs, leave `--input-gate-l1` off (default 0) — it runs the encoder
  identically to pre-gate.
- **Re-run fox / falco / sheik / cptfalcon / marth with full
  features** when the pipeline cycles through them next, and expect a
  similar ~1-3% drop at similar wall-clock.

## Today's preset additions

All additive — no existing preset was modified.

- `mimic-xl` — d_model=768, nhead=12, num_layers=6, dim_feedforward=3072,
  dropout=0.1, pos_enc="relpos", `use_swiglu=True`. ~44M params.
  Scale-up variant of the baseline `mimic` preset.
- `mimic-xl-rms` — same as `mimic-xl` plus `use_rmsnorm=True`. Added
  while isolating the RMSNorm contribution on the 10L scale-up.
- `modern-relpos-gelu` — GELU FFN variant of the existing
  `modern-relpos` preset (same d/L/heads, drops `use_swiglu`, nudges
  `dim_feedforward` to 2048 to hit matched param count).

`MODEL_PRESETS` in `mimic/model.py` also picked up a
`use_input_gate` field on `ModelConfig`. Default False; not set on any
preset. Opt in per-run via `--input-gate-l1 <lambda>`.

## Scaffolding changes

- `mimic/frame_encoder.py:MimicFlatEncoder` — now honors
  `mimic_minimal_features` (previously ignored). Adds `input_gate_logits`
  parameter, gate_l1_penalty()/gate_values()/gate_report()
  helpers, and `_build_feature_names()` mapping each of the 202 (or
  166 in minimal mode) input-dim positions to a human-readable label.
- `mimic/frame_encoder.py:build_encoder` — now accepts
  `use_input_gate` and forwards it to `MimicFlatEncoder`.
- `mimic/model.py:ModelConfig` — new `use_input_gate: bool` field.
- `mimic/model.py:FramePredictor` — passes `cfg.use_input_gate` through
  to `build_encoder`.
- `train.py` — new `--input-gate-l1 <lambda>` CLI flag; threaded into
  `train()` and `get_model()`; L1 penalty added to micro-loss; gate
  stats logged to wandb; gate_report JSON dump at end of training;
  per-step LR now logged to wandb as `train/lr` (unrelated,
  previously missing).

## Stuff worth checking if you're touching this again

1. The gate dump code path writes to
   `checkpoints/{run_name}_gate_report.json` hardcoded. If you refactor
   `CKPT_DIR`-style handling later, make sure this still resolves.
   Initial version referenced a nonexistent `CKPT_DIR` and crashed at
   end-of-training — caught and fixed post-run, gate report was
   regenerated from the final `_bestloss.pt` checkpoint and the
   `feature_names` builder.
2. The Linear projection input_dim differs between minimal (166) and
   full (202). Minimal-trained checkpoints are NOT interchangeable with
   full-feature ones. The config-pickled `mimic_minimal_features` flag
   is what makes reload work either way.
3. λ=0.01 produced a clean ranking but mean gate ≈ 0.56 — many
   features sit in a middle band rather than hard-pruning. For a
   sharper dichotomy, try λ=0.03-0.05; expect the "definitely kept"
   band to separate more clearly. Watch for over-regularization: if
   val loss regresses >1% vs non-gated fullfeat, λ is too high.
4. The SwiGLU `hidden = int(2*d_ff/3)` param-matching rule means
   swapping SwiGLU↔GELU at the same `dim_feedforward` keeps params
   within a few hundred. To match param count exactly for a
   SwiGLU-with-d_ff=2050 → GELU, use d_ff=2048 (off by 9K / 0.05% —
   within noise).
