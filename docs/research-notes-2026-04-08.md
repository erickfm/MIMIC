# Research Notes — 2026-04-08: HAL vs MIMIC Pipeline Audit

## Head-to-Head Result

Ran HAL Original (port 1) vs MIMIC hal-local_best.pt (port 2) on FD using
`tools/head_to_head.py`. **HAL won 2-0 in stocks** (216 seconds). Both models
have identical architecture (19,983,347 trainable params).

## Exhaustive Pipeline Comparison

### What matches exactly

- Architecture: same shapes at every layer, same head dims, same RelPos attention
- Optimizer: AdamW, lr=3e-4, betas=(0.9,0.999), weight_decay=0.01, fused
- LR schedule: CosineAnnealingLR, T_max=32768, eta_min=1e-6, no warmup
- Gradient clipping: max_norm=1.0
- Batch size: 64 local × 8 GPUs = 512 effective
- Loss: plain F.cross_entropy, equal weighting, no label smoothing
- Mixed precision: none (pure FP32)
- Weight init: Normal(0, 0.02), residual scaling 0.02/sqrt(2*6)
- Er (relative position): torch.randn(1024, 64) — standard normal
- Normalization stats: hal_norm.json = checkpoints/stats.json (verified exact match)
- Controller encoding: 54-dim one-hot (37+9+5+3)

### Differences found and fixed

**1. btns_single encoding (slp_to_shards.py line 652)**

When buttons change but nothing new is pressed (partial release):
- HAL: NO_BUTTON (index 4)
- MIMIC was: min(curr_buttons) — kept the surviving button

Fixed to match HAL. 524,482 frames affected (0.71% of 73.6M).

**2. run_mimic_via_hal_loop.py wrong stats file (line 60)**

Was loading `hal/data/stats.json` (p1_percent.max=362) instead of
`checkpoints/stats.json` (max=236). Causes percent normalization errors up to
0.44. Fixed.

**3. run_mimic_via_hal_loop.py unsorted player dict (lines 100, 227)**

Was `list(gs.players.items())` instead of `sorted()`. Can swap ego/opponent on
some frames. Fixed.

**4. _InferenceModel stage index (run_hal_model.py line 202)**

Was doing `(stage - 1).clamp(min=0)` on already 0-based indices. No-op on FD
(index 0) but wrong for other stages. Fixed.

### Differences accepted (non-impactful)

**5. Windowing: 100 random windows/game vs 1/game**

HAL samples 1 random window per game per __getitem__. MIMIC samples 100 per
game per shard visit. Over 16.7M samples both achieve similar unique window
coverage (~22% vs ~28% of possible windows per game). Net total unique windows:
MIMIC 14.8M vs HAL 12.2M. Not a meaningful advantage for either.

**6. Data size: 7,600 games vs 2,830 games**

MIMIC has 2.7× more training data from the same source. More data previously
shown to reduce overfitting (12K games: val +1.6% vs 3.2K: val +13%). This is
a confounding variable but likely helps, not hurts.

**7. Perspective: pre-stored both vs random per access**

MIMIC shards store both p1 and p2 perspectives (3,800 replays × 2 = 7,600
games). HAL stores one per replay and picks randomly. Using `--random-perspective`
at training time matches HAL's behavior.

### Shoulder head: analog vs digital

The GameCube L/R triggers have separate analog (smooth range) and digital
(click at bottom) signals. libmelee confirms: `press_shoulder(L, 1.0)` sets
analog only; digital needs separate `press_button(BUTTON_L)`.

**HAL also never triggers digital L/R.** Its `send_controller_inputs` iterates
`ORIGINAL_BUTTONS` and presses only what the button head predicts (A/B/X/Z).
The shoulder head only calls `press_shoulder` (analog). Digital L/R are always
released.

HAL still plays well (4-stocking CPUs), so the analog trigger value appears
sufficient for teching and shielding in Melee. The digital click is not needed.
Both HAL and MIMIC are in the same boat here.

### Button encoding: how it works

The model has a 5-class button head: A, B, Jump(X|Y), Z, None. It can only
predict one button per frame. When .slp shows multiple buttons held
simultaneously, the early-release encoding picks the one that just appeared
(went from 0→1). Buttons already held from previous frames are ignored.

The encoding converts button STATE (what's held) to button EVENTS (what was
just pressed). Single buttons held alone are labeled correctly every frame.
The information loss is only when two action buttons overlap (2.65% of frames).

Shoulder (L/R) is a separate 3-class analog head, so shoulder+button combos
(shield+jump, L-cancel, wavedash) are fully representable.

### Training results (hal-fixed-pipeline)

Trained on fixed shards with `--random-perspective`, bs=128, accum=4, single 4090.
Stopped at step ~13,100 (6.7M samples). Best val loss **1.038** (vs HAL's 1.089).

Head-to-head results against HAL Original (Fox vs Fox, FD):
- hal-fixed-pipeline_best (2.5M samples, val 1.03): **HAL 4-0**
- hal-fixed-pipeline_step9828 (5.0M samples, val 1.05): **HAL 2-0**
- For reference, pre-fix hal-local_best: **HAL 2-0**

Lower val loss did not translate to better gameplay.

### Data filtering — CRITICAL FINDING (identified after training)

HAL aggressively filters replays. MIMIC does not. This is likely the dominant
factor in the gameplay gap.

| Filter | HAL | MIMIC |
|--------|-----|-------|
| Min game length | 1,500 frames (25 sec) | 2 frames |
| Damage check | Skip if 0 damage dealt | None |
| Completion check | Require someone loses all stocks | None |
| Val split | 98/1/1 | 90/10 |

MIMIC trains on ragequits, disconnects, zero-damage games, and 2-frame games.
These teach the model poor behavior that doesn't show up in val loss (the val
set contains the same junk). HAL only trains on complete, competitive games.

### blocking_input — INFERENCE BUG (fixed)

All three inference scripts used `blocking_input=False`. This means Dolphin
advances frames without waiting for controller input. If model inference takes
>16.67ms, the model misses frames. In head-to-head, P2's inputs are flushed
after P1's, systematically disadvantaging P2 (always MIMIC in our tests).

Fixed all scripts to `blocking_input=True`. Dolphin now waits for input before
advancing, eliminating frame drops.

### Forward pass — confirmed identical

Architecture, attention, heads, dropout, initialization all verified
byte-compatible between HAL and MIMIC. No forward pass differences.

### Remaining differences (lower priority)

1. **torch.compile**: HAL uses `torch.compile(model, backend="aot_eager")` for
   inference. MIMIC does not. Affects speed, not numerical output.

2. **Target format**: HAL passes one-hot float targets to F.cross_entropy. MIMIC
   passes integer class indices. Mathematically equivalent.

3. **MDS vs .pt**: Different data loading formats but equivalent semantics.

## Next Steps

1. Add HAL's data filters to slp_to_shards.py (min 1500 frames, damage check,
   completion check)
2. Rebuild shards with filtering
3. Retrain and re-evaluate gameplay
4. Run head-to-head with ports swapped (MIMIC as P1) to verify blocking_input fix
