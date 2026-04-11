# Research Notes — 2026-04-11

## Overview

This session covered: departing from HAL's 5-class button encoding to a new
7-class encoding based on Melee's actual input resolution rules, building the
full data pipeline on a new machine, training multiple models, and debugging
inference issues. The 7-class encoding is implemented and produces correct
training metrics, but closed-loop inference exhibits a NONE-bias feedback loop
that remains unresolved.

---

## 1. Environment Setup

Set up MIMIC on a new machine with an RTX 5090 (32GB VRAM). Key components:
- PyTorch 2.8.0 + CUDA 12.8
- melee-py, Dolphin emulator (squashfs AppImage), Melee ISO
- No existing checkpoints or data on this machine

### Dataset Download

Downloaded Fox .slp files from `erickfm/slippi-public-dataset-v3.7` on
HuggingFace. The dataset has ~55K Fox replays across 5 batch directories plus
a top-level directory.

Initial downloads were severely rate-limited (429 errors) because we weren't
authenticated. After `hf auth login`, parallel `snapshot_download` calls
achieved ~150 files/sec. Downloaded **20,009 files (43GB)** before stopping.

After deduplication via symlinks (filenames collide across batches), **18,192
unique .slp files** available in `data/fox_all_slp/`.

### Metadata Generation

Built metadata from the .slp files:
- `norm_stats.json` + `norm_minmax.json`: from 5,000-file sample via
  `tools/build_norm_stats.py` (restored from git history, commit 70de567)
- `stick_clusters.json`: HAL's hardcoded 37 main stick clusters + 3 shoulder
  bins ([0.0, 0.4, 1.0])
- `controller_combos.json`: 7-class scheme (see section 2)
- `hal_norm.json`: generated from norm_stats + HAL's transform types per
  feature (normalize, invert_normalize, standardize)

**Bug found:** `build_norm_stats.py` crashed on `Stage.NO_STAGE` (value 0,
truthy in Python). Fixed by checking `stage in stages.BLASTZONES`.

---

## 2. 7-Class Button Encoding

### Motivation

HAL uses a 5-class button encoding (A=0, B=1, Jump=2, Z=3, NONE=4) derived
from early-release logic that tracks frame-to-frame button state transitions.
Shoulder is a separate 3-class analog head.

The document `docs/gamecube_input_collapse.md` shows that Melee resolves all
31 possible button combinations into exactly 7 distinct in-game behaviors via
three priority rules:

1. **B pressed → B** (hard override, beats everything)
2. **A + TRIG both pressed (no B) → A+TRIG** (shield grab, emergent behavior)
3. **Else highest priority: Z > A ≈ TRIG > JUMP**

This gives 7 classes: A=0, B=1, Z=2, JUMP=3, TRIG=4, A_TRIG=5, NONE=6.

### Design Decisions

- **Keep 3-class analog shoulder head** — TRIG in the button head handles the
  digital press, shoulder head predicts analog pressure independently. The game
  sees these separately.
- **Change both input AND output encoding** — controller one-hot goes from
  54-dim (37+9+5+3) to 56-dim (37+9+7+3)
- **Include A+TRIG** — shield grab is a core competitive technique
- **Use 9-class c-stick** (not 5)
- **Replace early-release entirely** — each frame labeled independently by
  applying the 3 priority rules. No temporal state tracking needed.
- **Clean break** — old 5-class shards are incompatible, must re-shard

### Why Early-Release Is No Longer Needed

For the common cases (button held, button released, nothing pressed), the
7-class collapse and early-release produce identical labels. Verified by
simulation: a simple A-press sequence (NONE→A→A→A→NONE) is identical under
both schemes. The 7-class collapse applies per-frame priority rules that
subsume the early-release logic.

The only difference is in multi-button overlaps (2.6% of frames), where the
7-class collapse resolves by game behavior priority (B overrides all, A+TRIG
is emergent) while early-release resolves by temporal ordering (newest button
wins).

### Implementation

**`mimic/features.py`:**
- Added `_collapse_buttons_7class_np(btns)` — vectorized, uses bottom-up
  assignment where later rules overwrite earlier (B last = highest priority)
- Added `_collapse_buttons_7class_single(buttons)` — single-frame version
  for inference
- Updated `encode_controller_onehot` and `encode_controller_onehot_single`
  to branch on `n_combos == 7` and call the 7-class collapse directly
- Renamed old `_collapse_buttons_np` → `_collapse_buttons_5class_np`
- Updated `load_controller_combos` to return `(combos, combo_to_idx, n_combos)`
  and handle the 7-class JSON format (no `combos` field)

**`tools/slp_to_shards.py`:**
- Replaced 30-line early-release state machine with single
  `_collapse_buttons_7class_np(raw_btns)` call
- Fixed `n_combos_local = len(combo_map_local)` bug — was overwriting the
  correct n_combos from the JSON with len of empty dict (0) for 7-class
- Added dual-path: checks `_W["n_combos"]` to decide between 7-class collapse
  and 5-class early-release

**`tools/run_mimic_via_hal_loop.py`:**
- Added 7-class decode branch in `decode_and_press`
- Fixed flag double-normalization bug (see section 4)

**`tools/head_to_head.py`:**
- Fully rewritten to use shared `tools/inference_utils.py`

**`tools/inference_utils.py`:** (new file)
- Shared module for `build_frame`, `decode_and_press`, `PlayerState`,
  `load_mimic_model`, `load_inference_context`
- Single code path for frame construction, normalization, and decoding
- Both `run_mimic_via_hal_loop.py` and `head_to_head.py` import from here

**`data/fox_meta/controller_combos.json`:**
- Updated to 7-class scheme: `{"button_names": [...], "n_combos": 7,
  "class_scheme": "melee_7class"}`

### Verification

All 31 button combinations from the doc verified against expected labels
(64 tests: 32 combos × 2 variants, vectorized + single-frame). 100% pass.

Shard data verified:
- `self_controller` dim = 56 (37+9+7+3) ✓
- `btns_single` range [0, 6] with all 7 classes present ✓
- One-hot sections all sum to 1.0 ✓
- No NaN/Inf ✓

---

## 3. Dataset Audit

### Shard Contents vs Training Usage

The HAL-mode encoder (`HALFlatEncoder`) only reads 12 state keys:
`stage, self_character, opp_character, self_action, opp_action, self_numeric,
opp_numeric, self_flags, opp_flags, self_controller, self_port, opp_port`

The shards contain 68 state tensors (projectiles, nana data, opponent buttons,
action elapsed, costume, etc.) — **66% of shard data is unused** in HAL mode.
Per 800MB shard, only ~248MB is actually read. Not a correctness issue but
significant I/O waste.

### Controller Offset

The controller offset (`--controller-offset`) is applied at **dataloader time**
in `mimic/dataset.py:227`, NOT in the shard. The shard stores same-frame
alignment: `input_controller[t] == target_btns[t]` (100% match). The dataset
shifts `self_controller` by -1 so position i sees frame i-1's controller.

This is correct behavior — do not be alarmed by same-frame alignment in raw
shards.

### Button Label Distribution (7-class)

From 3-shard sample (1.5M frames):

| Class | Count | Percent |
|-------|-------|---------|
| A | 58,325 | 3.8% |
| B | 71,988 | 4.7% |
| Z | 11,945 | 0.8% |
| JUMP | 93,243 | 6.1% |
| TRIG | 195,864 | 12.9% |
| A_TRIG | 3,743 | 0.2% |
| NONE | 1,083,193 | 71.3% |

Real human Fox replays show ~70.8% NONE (from 20-game sample), closely
matching the shard distribution.

### Per-Action Button Distribution

The NONE dominance is correct per game state:

| Action State | NONE % | Notes |
|-------------|--------|-------|
| STANDING | 98.9% | Correct — idle |
| CROUCHING | 96.7% | Correct — holding down |
| DASHING | 99.8% | Correct — no buttons during dash |

Buttons only fire on **transition frames** — the 1-3 frames where a player
initiates an action. The vast majority of gameplay is continuation states.

---

## 4. Bugs Found and Fixed

### Flag Double-Normalization (inference)

**Location:** `tools/run_mimic_via_hal_loop.py` (old version, pre-refactor)

**Bug:** `build_frame` applied XFORM normalization to boolean flags (on_ground,
facing, invulnerable), mapping 0→-1 and 1→1. The `HALFlatEncoder.forward()`
then applied `* 2.0 - 1.0` again. Result: facing=0 became -3.0 instead of -1.0.

**Fix:** Pass raw 0/1 flags in `build_frame`. The encoder normalizes them.
Training shards store raw flags.

**Scope:** Only affected `run_mimic_via_hal_loop.py`. The old `head_to_head.py`
used `_InferenceModel` wrapper that bypassed the encoder's `forward()`, so it
was not affected. After refactoring both scripts to use shared
`inference_utils.py`, all inference uses the same correct path.

### Reaction Delay Mismatch

**Bug:** Training used `REACTION_DELAY = 1` (default in train.py:53). This
means `target[t] = shard[t+1]` — the model learns to predict the next frame's
action. At inference, the prediction is applied to the current frame, creating
a 1-frame timing mismatch.

**Evidence from prior research:** `docs/archive/research-notes-2026-04-02-b.md`
Finding 23: "With `reaction_delay=1`, the model predicts frame i+1's button
from frame i's gamestate. At inference, the prediction gets applied on frame i
(immediately), not frame i+1."

**Fix:** Train with `--reaction-delay 0 --controller-offset`. Target is
same-frame, controller shifted by -1 to avoid leaking the answer.

**Impact:** Val loss dropped from 1.167 (rd=1) to **0.743** (rd=0). Button F1
improved from 77.2% to **90.2%**. Main stick F1 from 37.2% to **55.4%**.

### n_combos_local Override Bug (slp_to_shards.py)

**Bug:** After calling `load_controller_combos(meta_dir)` which correctly
returned `n_combos=7`, line 1161 overwrote it with
`n_combos_local = len(combo_map_local)`. For 7-class, `combo_map_local` is an
empty dict (rule-based, no lookup), so n_combos became 0. This produced shards
with 49-dim controllers (37+9+0+3) instead of 56-dim (37+9+7+3).

**Fix:** Removed the `len(combo_map_local)` override. Same bug existed in the
streaming upload path (line 1286).

### Context Window Prefill

**Bug:** `PlayerState.push_frame()` prefilled the context window with 255
copies of the first real frame. At game start, this is a spawn animation with
no buttons — essentially 255 frames of "doing nothing." The model took this
literally and predicted NONE for the first ~4 seconds.

**Fix:** Prefill with HAL-style mock values (all features set to 1.0 then
normalized), matching HAL's context initialization. The mock values are
deliberately nonsensical so the model learns to ignore/override them quickly.

**Impact:** This improved early-game behavior but did not resolve the
persistent NONE-bias in closed-loop inference (see section 6).

---

## 5. Training Runs

### hal-7class (17K games, rd=1) — First 7-class run

- **Data:** 17,319 filtered games, 162.8M frames, 324 train shards
- **Config:** `--model hal-learned --encoder hal_flat --hal-mode
  --hal-minimal-features --hal-controller-encoding --stick-clusters hal37
  --plain-ce --lr 3e-4 --batch-size 64 --max-samples 16777216
  --grad-accum-steps 8 --controller-offset --self-inputs`
- **Issue:** Used default `reaction_delay=1`
- **Results:** Val loss 1.192, btn F1 79.1%, main F1 37.1%
- **wandb:** https://wandb.ai/erickfm/mimic/runs/k4rb1ybz
- **Gameplay:** Sporadic, 95%+ NONE in replays

### hal-7class-matched (2.6K games, rd=1) — HAL-sized dataset

- **Data:** 2,647 filtered games (~HAL's 2,830), 24.9M frames, 50 train shards
- **Config:** Same as above
- **Results:** Val loss 1.167, btn F1 77.2%, main F1 37.2%
- **wandb:** https://wandb.ai/erickfm/mimic/runs/ip6lb4ka
- **Gameplay:** Same NONE-bias issue

### hal-7class-rd0 (2.6K games, rd=0) — Fixed reaction delay

- **Data:** Same 2,647-game dataset
- **Config:** Added `--reaction-delay 0`, increased to `--batch-size 512`
  (no grad accum) for better GPU utilization (46% → 86% on RTX 5090)
- **Results:** Val loss **0.743**, btn F1 **90.2%**, main F1 **55.4%**,
  btn acc **95.4%**
- **wandb:** https://wandb.ai/erickfm/MIMIC/runs/51igt4dt
- **Gameplay:** Still exhibits NONE-bias in closed-loop inference

### Training Depth Comparison

| Run | Games | Samples | Steps | Reps/Game |
|-----|-------|---------|-------|-----------|
| HAL original | 2,830 | 5.2M | 10,156 | ~1,837 |
| hal-filtered (prev best) | 6,898 | 16.7M | 32,768 | ~2,422 |
| hal-7class-matched | 2,647 | 5.2M | 10,240 | ~1,964 |

Our HAL-matched run has comparable per-game repetition to HAL's original.

---

## 6. The NONE-Bias Problem (Unresolved)

### Symptoms

The model achieves excellent offline metrics (90%+ button accuracy, 95%+ on
action frames) but in closed-loop Dolphin inference, it predicts NONE 95-99%
of the time. Fox stands still, crouches, or barely moves.

### What Has Been Verified

1. **Forward pass correctness:** Feeding training data through `PlayerState`
   frame-by-frame produces **identical** encoder proj inputs (max diff = 0.0)
   and identical predictions as batch training forward pass.

2. **Data correctness:** 7-class shards have matching input/target
   distributions (72.3% NONE for both input controller and target btns_single).
   One-hot sections valid. No NaN/Inf. Controller offset correctly applied at
   dataloader time.

3. **Model capability:** On real human replay data fed through the inference
   code path (building frames from .slp via `build_frame` with actual
   controller feedback), the model correctly predicts JUMP at 97% confidence
   during jumps, A during attacks, etc.

4. **Normalization match:** Numeric features produced by `build_frame` match
   shard values within floating-point precision. Flags are passed raw (0/1)
   and normalized by the encoder, matching training.

### The Mechanism

The model relies heavily on the **previous frame's controller input** to decide
the current action. Evidence:

| Condition | Action-Frame Accuracy |
|-----------|----------------------|
| With controller input (normal) | **95.5%** |
| Controller zeroed out | **31.8%** (all predictions become NONE) |

Without seeing a button in the controller input, the model defaults to NONE
regardless of game state. At inference, the model's own NONE predictions feed
back as the next frame's controller input, creating a self-reinforcing loop:

1. Frame 0: no prior controller → predict NONE
2. Frame 1: controller shows NONE → predict NONE
3. Frame 2: controller shows NONE → predict NONE
4. (indefinitely)

The model never learned to **initiate** actions from game state alone — it
learned to **continue** actions from controller history (teacher forcing).

### Why HAL Doesn't Have This Problem

Unknown. HAL uses the same architecture, same controller feedback, same
NONE-heavy distribution, and achieves similar offline metrics. Possible
explanations to investigate:

1. **Early-release vs per-frame collapse** — while labels are identical for
   common cases, the early-release encoding tracks state transitions which
   may create different learning dynamics
2. **Self-input dropout (si_drop)** — the codebase has a `si_drop_prob`
   mechanism that stochastically zeros the controller input during training,
   forcing the model to learn to initiate actions without controller history.
   Our training did not use this. It's unclear whether previous successful
   MIMIC runs used it.
3. **Different training data** — HAL trained on its own curated dataset which
   may have different characteristics
4. **Something in HAL's inference pipeline** — HAL's `play.py` may handle the
   feedback loop differently

### 5-Class Baseline Attempt (Invalid)

Attempted to train a 5-class early-release baseline for comparison. This
required restoring the early-release code in `slp_to_shards.py` (dual path
gated on `n_combos`) and creating a 5-class `controller_combos.json`.

**Critical finding:** The 5-class combo file we created had the wrong index
ordering. Our file mapped `[0,0,0,0,0]` (no buttons) to index 0. But the
early-release target encoding uses A=0, B=1, Jump=2, Z=3, **NONE=4**. HAL's
actual combo map (from `tools/run_hal_model.py`) maps:

```python
COMBO_MAP = {
    (1, 0, 0, 0, 0): 0,  # A
    (0, 1, 0, 0, 0): 1,  # B
    (0, 0, 1, 0, 0): 2,  # Jump
    (0, 0, 0, 1, 0): 3,  # Z
    (0, 0, 0, 0, 0): 4,  # NONE
    (0, 0, 0, 0, 1): 4,  # shoulder only → NONE
    (1, 0, 0, 0, 1): 0,  # A + shoulder → A
    ...
}
```

So `(0,0,0,0,0)` should map to index **4**, not 0. With our wrong file, the
input controller encoded "no buttons" as class 0 (A position) while the target
encoded it as class 4 (NONE). The shard data confirmed this: **85% of input
controller frames were class 0 (A) while 83% of targets were class 4 (NONE)**.

The 5-class model trained on this data achieved similar metrics (val loss 0.745,
btn F1 89.5%) but the results are meaningless — the model learned contradictory
input/target mappings.

**The 7-class encoding avoids this entirely** because both input and target use
`_collapse_buttons_7class_np` which returns the same indices. No combo file
lookup needed. This is a structural advantage of the 7-class approach.

To produce a valid 5-class baseline, the combo file must match HAL's exact
ordering, including the shoulder-combo entries. This was not completed.

### Verified: Not an Encoding Issue

The controller dependency experiment (95.5% accuracy with controller, 31.8%
without) combined with the replay analysis (model correctly predicts actions
when given real human replay context) proves the model has learned the task
correctly. The NONE-bias is a closed-loop deployment problem, not a training
or encoding problem.

The model can predict every action class with high confidence when the input
context contains real gameplay. It fails at inference because its own NONE
predictions create a degenerate context that perpetuates NONE.

### Next Steps

1. **Self-input dropout (si_drop)** — the codebase has `--si-drop-start`,
   `--si-drop-end`, `--si-drop-max` flags that stochastically zero the
   controller input during training. This explicitly forces the model to learn
   to initiate actions from game state alone, breaking the controller
   dependency. Most promising approach.
2. **Fix 5-class combo file** — use HAL's exact combo map ordering (including
   shoulder combos) for a valid 5-class baseline comparison.
3. **Compare HAL's inference pipeline** — trace what HAL's `play.py` does
   differently for controller feedback. HAL may have different feedback
   dynamics that avoid the NONE loop.
4. **Scheduled sampling** — during training, occasionally replace the teacher
   controller input with the model's own prediction. This directly addresses
   the exposure bias.

---

## 7. Infrastructure Notes

### Dolphin Inference

- Dolphin instances **must be killed manually** and **run sequentially**. 
  Concurrent instances cause spectator server port conflicts and EOFError
  crashes.
- Use `Xvfb :99` as a persistent virtual display. Start it once in its own
  background command, not chained with the inference command.
- `pkill -f dolphin` can kill the calling bash shell if "dolphin" appears in
  the command line. Use `pkill -f dolphin-emu` instead, or get PIDs first.
- `blocking_input=True` is critical — without it, Dolphin advances frames
  without waiting for controller input, causing frame drops.

### Replay Saving

Added `save_replays=True, replay_dir=<path>` to Dolphin Console construction
in both inference scripts. Replays save to `replays/` directory.

### GPU Utilization

RTX 5090 (32GB VRAM). With `--batch-size 64 --grad-accum-steps 8`: 46% GPU,
1.6GB VRAM. With `--batch-size 512` (no accum): **86% GPU**, 5.6GB VRAM.
Effective batch size 512 in both cases. The larger batch is ~3.5x faster
throughput.

### Code Refactoring

Eliminated bespoke inference code. Previously `head_to_head.py` had its own
`_InferenceModel` wrapper (~400 lines) that bypassed the encoder's `forward()`
and reimplemented the model's forward pass manually. This meant any encoder
change had to be mirrored in two places, and bugs (like the flag
double-normalization) affected the scripts differently.

Now both `run_mimic_via_hal_loop.py` and `head_to_head.py` import from
`tools/inference_utils.py`. One code path for frame construction,
normalization, and decoding.

---

## 8. File Changes Summary

### New Files
- `tools/inference_utils.py` — shared inference module
- `tools/build_hal_norm.py` — generates hal_norm.json from norm_stats
- `tools/download_fox.py` — batch HF dataset downloader

### Restored from Git History
- `tools/build_norm_stats.py` — compute norm_stats.json from .slp files
- `tools/build_clusters.py` — stick cluster generation
- `tools/build_controller_combos.py` — combo discovery

### Modified
- `mimic/features.py` — 7-class collapse functions, updated encode functions,
  updated `load_controller_combos` return signature
- `tools/slp_to_shards.py` — 7-class target encoding, n_combos fixes
- `tools/run_mimic_via_hal_loop.py` — rewritten to use inference_utils
- `tools/head_to_head.py` — rewritten to use inference_utils
- `tools/build_norm_stats.py` — Stage.NO_STAGE fix, added min/max tracking
- `eval.py`, `inference.py`, `train.py`, `tools/validate_checkpoint.py` —
  updated `load_controller_combos` call sites for 3-tuple return

### Data
- `data/fox_meta/` — metadata files (norm_stats, cat_maps, stick_clusters,
  hal_norm, controller_combos)
- `data/fox_hal_match_shards/` — 2,647-game 7-class shards
- `data/fox_hal_full/` — 17,319-game 7-class shards
- `data/fox_all_slp/` — 18,192 unique .slp symlinks
- `replays/` — inference replay .slp files
