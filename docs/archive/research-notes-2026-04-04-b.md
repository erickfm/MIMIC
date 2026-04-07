# Research Notes — 2026-04-04 (Part 2)

## Summary

Two critical preprocessing bugs in `tools/run_hal_model.py` were identified and fixed, enabling HAL's trained model to play intentionally through MIMIC's inference pipeline for the first time. A systematic comparison methodology and automated gameplay health checker were also developed.

---

## Bugs Found and Fixed

### Bug 1: Wrong Normalization Statistics (Critical)

**Problem:** `data/fox_public_shards/hal_norm.json` contained normalization statistics computed from an unknown source (likely Fox-only replay data), not from HAL's actual training statistics at `hal/data/stats.json` (221M frames across all characters).

**Key mismatches:**

| Feature | HAL's stats.json | MIMIC's hal_norm.json | Impact |
|---------|-----------------|----------------------|--------|
| percent max | 362 (p1), 421 (p2) | 236 | 100% damage: -0.45 vs -0.15 |
| pos_x mean | -0.517 (p1) | 1.118 | All positions shifted |
| pos_x std | 57.468 (p1) | 56.671 | Positions rescaled |
| pos_y mean | 12.294 (p1) | 11.492 | All positions shifted |
| shield min | 0.024 (p1) | 0.0 | Minor |

**Additional issue:** HAL uses player-specific stats (p1 for ego, p2 for opponent). MIMIC's hal_norm.json was player-agnostic, using a single set for both. The p1/p2 stats differ meaningfully — e.g., percent max is 362 for p1 vs 421 for p2.

**Fix:** Modified `run_hal_model.py` to load HAL's actual `stats.json` directly, with player-specific stats (p1 for ego, p2 for opponent) and HAL's exact transform functions (normalize, invert_and_normalize, standardize) per feature.

### Bug 2: Button Encoding Order Mismatch (Critical)

**Problem:** The `COMBO_MAP` in `run_hal_model.py` assigned button class indices in the wrong order.

**HAL's encoding** (from `encode_buttons_one_hot_no_shoulder`):
```
stacked = [button_a, button_b, jump, button_z, no_button]
→ A=0, B=1, Jump=2, Z=3, NONE=4
```

**MIMIC's old COMBO_MAP:**
```
(0,0,0,0,0): 0  → NONE=0  (should be 4)
(1,0,0,0,0): 1  → A=1     (should be 0)
(0,1,0,0,0): 2  → B=2     (should be 1)
(0,0,1,0,0): 3  → Jump=3  (should be 2)
(0,0,0,1,0): 4  → Z=4     (should be 3)
```

**Impact:** Every frame's controller feedback one-hot vector had the button section garbled. When no button was pressed, the model saw class 0 (which HAL interprets as "A is pressed"). This corrupted the model's understanding of what controller input it had sent on the previous frame.

**Fix:** Changed COMBO_MAP to match HAL's ordering: A=0, B=1, Jump=2, Z=3, NONE=4. Also added entries for shoulder+button combos (shoulder maps to NONE in the button section since HAL handles shoulder separately).

---

## Verification

### Preprocessing Tensor Comparison

Wrote `tools/compare_hal_mimic_preprocessing.py` which processes synthetic gamestates through HAL's actual `Preprocessor` (imported from HAL's codebase) and MIMIC's fixed `build_frame()`, then compares every tensor value.

**4 test cases**, varying damage, position, buttons, shield, stocks:
- All categorical indices: **exact match**
- Gamestate vector (18 floats): **max diff 6e-8** (float32 rounding)
- Controller vector (54 floats): **max diff 0.0** (exact match)

**Result: ALL TESTS PASSED**

### Additional Verifications (During Planning)

| Check | Result |
|-------|--------|
| Character mapping (27 chars) | 0 mismatches |
| Action mapping (395 actions) | 0 mismatches |
| Stage mapping (6 stages) | 0 mismatches |
| Stick clusters (37 main) | Exact match (values and order) |
| C-stick clusters (9) | Exact match |
| Shoulder clusters (3) | Exact match |
| Controller concat order | main(37)+c(9)+btn(5)+shldr(3) = 54 — matches |
| Feature order (18-dim gamestate) | Same 9 features per player in same order |

---

## Live Gameplay Results

Ran `tools/run_hal_model.py` with fixed preprocessing against level 9 CPU Fox on Battlefield.

### Before Fix (Bugged)
- Bot stands completely idle
- 0 button presses per match
- NONE probability: 99.9%+
- Model never initiates actions

### After Fix
- Bot actively moves, attacks, jumps, shields, lasers
- 13,855 frames (~231 seconds) of gameplay logged

### Gameplay Health Metrics

| Metric | Bugged | Fixed | Threshold |
|--------|--------|-------|-----------|
| Button press rate | ~0/sec | **1.4/sec** | >= 0.5 |
| Unique buttons | 0-1 | **4** | >= 3 |
| Stick at neutral | ~100% | **34.2%** | <= 80% |
| Unique stick positions | 1-3 | **37** | >= 10 |
| Mean NONE prob | 0.999+ | **0.827** | <= 0.95 |
| NONE < 0.99 | <1% | **38.2%** | >= 10% |
| Non-NONE top prediction | ~0% | **16.8%** | >= 5% |
| Shoulder active | ~0% | **15.8%** | >= 1% |

**Result: 8/8 health checks passed — HEALTHY**

---

## New Tools

### `tools/compare_hal_mimic_preprocessing.py`
Processes synthetic gamestates through both HAL's actual preprocessor and MIMIC's reimplementation. Compares categoricals, gamestate vector, and controller vector. Reports per-component diffs.

### `tools/gameplay_health.py`
Parses inference log output (MAIN/C/L/BTN/top3 lines) and computes 8 diagnostic metrics with pass/fail thresholds. Can be piped from inference or run on saved logs. Exit code 0 = healthy, 1 = unhealthy.

```bash
# From saved log:
python tools/gameplay_health.py game.log

# Piped:
python tools/run_hal_model.py ... 2>&1 | tee game.log
python tools/gameplay_health.py game.log
```

---

## Next Steps

1. **Apply fixes to `inference.py`** — The same normalization and button encoding fixes need to be applied to MIMIC's main inference pipeline so MIMIC-trained models benefit too.

2. **Retrain with correct HAL preprocessing** — Now that we know the exact normalization transforms and button ordering, retrain a MIMIC model using HAL's actual stats. This should close the STANDING calibration gap (94.8% HAL vs 99% MIMIC).

3. **Investigate remaining calibration gap** — Even with correct preprocessing, MIMIC's architecture differs from HAL's (RoPE vs relative position encoding, per-group MLPs vs single concat→Linear). These may affect training calibration.
