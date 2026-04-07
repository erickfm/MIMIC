# Research Notes — 2026-04-04 (Part 3)

## Summary

After fixing 4 inference preprocessing bugs (normalization stats, button encoding order, feature name aliases, numpy uint8 type bypass), we verified preprocessing is correct and ran extensive diagnostics. The MIMIC model still doesn't play effectively despite 89.8% training F1. Root cause analysis revealed: (1) the model is NOT copying controller feedback (zeroing it barely changes predictions), (2) the model IS learning from game state (contextual predictions on training data), (3) the prediction gap between training and inference is driven by `self_action` and `self_controller` distributions, and (4) there are **9 unmatched architectural/training differences** between HAL and MIMIC that were never controlled for.

---

## Preprocessing Bugs Fixed (This Session)

| Bug | Effect | Fix |
|-----|--------|-----|
| hal_norm.json wrong stats | percent normalized to -0.15 instead of -0.45 at 100% | Load HAL's stats.json directly |
| COMBO_MAP button order | NONE=class 0 instead of class 4, garbled controller feedback | Reorder to A=0,B=1,Jump=2,Z=3,NONE=4 |
| Feature name aliases | pos_x/pos_y/invuln_left not in HAL's stats dict → skipped | Add aliases position_x→pos_x etc |
| numpy uint8 type | ps.stock returns uint8, fails isinstance(v, (int, float)) | Use try/except float() conversion |

**Verification**: Same raw values through both pipelines produce identical normalized outputs (22 test cases, all match within 1e-8).

---

## Diagnostic Results

### Model Prediction Analysis

| Test | Result |
|------|--------|
| Predictions on training data | 26.2% non-NONE argmax, NONE std=0.39 (contextual) |
| Predictions on inference data | 8.3% non-NONE argmax, NONE std=0.26 (weaker) |
| Zero controller feedback | 23.2% non-NONE (model barely relies on controller) |
| Feature ablation | self_action (+11.7%) and self_controller (+15.0%) drive the gap |

### Gameplay Health (Level 9 CPU)

| Metric | HAL Original | MIMIC Latest |
|--------|-------------|-------------|
| Damage dealt | 335% | 11-18% |
| Stocks taken | 3 | 0 |
| Non-NONE argmax | 18.1% | 6.7-10.1% |
| Button press rate | 1.4/sec | 0.7-0.9/sec |

---

## Confirmed Differences: HAL vs MIMIC

Rigorous side-by-side comparison of HAL's original model against our latest MIMIC model. Every difference verified by inspecting checkpoint weights and model code.

| # | Feature | HAL | MIMIC | Impact |
|---|---------|-----|-------|--------|
| 1 | **Context length** | 256 frames (4.27s) | 60 frames (1.00s) | 4.3x less temporal context |
| 2 | **Position encoding** | Relative position (skew matrix) | RoPE | Different temporal attention patterns |
| 3 | **C-stick output** | 9 clusters (2D positions with diagonals) | 5 directions (categorical, no diagonals) | No diagonal c-stick attacks |
| 4 | **Dropout** | 0.2 | 0.1 | Less regularization |
| 5 | **Features per player** | 9 numeric | 10 numeric (+invuln_left) | Extra feature, different order |
| 6 | **Feature order** | percent,stock,facing,...,pos_x,pos_y | pos_x,pos_y,percent,stock,... | Different positions in projection |
| 7 | **Embedding vocab** | stage=6, char=27, action=396 | stage=8, char=32, action=395 | Different embedding table sizes |
| 8 | **Param count** | 26.3M | 19.6M | 6.7M fewer parameters |
| 9 | **Training data** | 221M frames | 59M frames (1 epoch) | 3.7x less data |

### Confirmed matches:
- Block structure: both pre-norm (LN → attn → + → LN → MLP → +)
- Core dimensions: d_model=512, nhead=8, num_layers=6
- Head order: shoulder → c_stick/c_dir → main_stick → buttons
- Head structure: LN → Linear(d→d/2) → GELU → Linear(d/2→out)
- Embedding dims: stage=4, character=12, action=32
- Button classes: 5 (A, B, Jump, Z, NONE)
- Main stick clusters: 37 (HAL's hand-designed)
- Shoulder: 3 classes [0.0, 0.4, 1.0]
- Loss: plain cross-entropy on all heads
- Optimizer: AdamW, LR=3e-4
- AMP: FP32 (no mixed precision)

---

## Next Steps

Eliminate all 9 differences to reproduce HAL. Priority order based on likely impact:

1. **Context length 256** — most impactful, gives 4.3x more temporal context
2. **Relative position encoding** — HAL's skew-based attention, replaces RoPE
3. **C-stick 9 clusters** — enables diagonal smash attacks
4. **Match exact features** — 9 per player in HAL's order, remove invuln_left
5. **Dropout 0.2** — match HAL's regularization
6. **Embedding vocab sizes** — stage=6, char=27, action=396
7. **Training on full data** — 221M frames or multiple epochs
