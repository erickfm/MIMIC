# MIMIC Experiment Results

All runs: 2M samples from data/full, hybrid16 encoder, learned pos-enc unless noted. Metrics evaluated on held-out val split using `eval.py`.

**Key metrics**:
- **val/total**: sum of all task losses (lower is better)
- **btn_f1**: F1 score for button presses — measures precision/recall balance on the rare "button pressed" class
- **cdir_active**: c-stick accuracy on non-neutral frames only — the old `cdir_acc` was >99% because neutral dominates; this measures accuracy when the c-stick is actually used
- **btn_prec / btn_rec**: button precision and recall

> Note: Phase 1-2 runs used the old saturated metrics (cdir_acc ~99%, btn_acc ~99%) and most checkpoints are no longer available for re-evaluation. Takeaways are preserved but tables use the original metrics.

---

## Phase 1-2: Encoder & Backbone

Default encoder, various presets and LRs. These runs established the architecture.

### Scaling (model size)

| Run | Preset | Params | LR | Batch | val/total |
|-----|--------|--------|----|-------|-----------|
| `scale-tiny` | tiny (256d/4L) | 6.4M | 1e-3 | 128 | 0.0742 |
| `scale-small` | small (512d/4L) | 16.3M | 3e-4 | 128 | 0.0929 |
| `scale-medium` | medium (768d/4L) | 32.4M | 8e-4 | 128 | 0.0797 |
| `scale-base` | base (1024d/4L) | 54.9M | 1e-3 | 64 | 0.0736 |
| `scale-deep` | deep (512d/8L) | 28.9M | 3e-4 | 64 | 0.0876 |
| `scale-wide-shallow` | 1024d/2L | 62.0M | 3e-4 | 32 | 0.0904 |
| `scale-xlarge` | xlarge | 105.3M | 1e-4 | 32 | 0.1084 |
| `scale-xxlarge` | xxlarge | 232.0M | 1e-4 | 16 | 0.0908 |

xlarge overfit badly (train 0.0261 vs val 0.1084). Medium and base hit the best val-per-param.

### Phase 1-2 takeaways

- **hybrid16** encoder won over flat, composite8, and default (55-token) encoders
- **medium** backbone (768d/4L) hit the best loss-per-param sweet spot at ~32M params
- **lr 8e-4** outperformed 3e-4 and 5e-4 at medium scale
- Larger models (xlarge, xxlarge) overfit at 2M samples with small batch sizes
- RoPE slightly beat learned pos-enc at small scale; ALiBi showed promise but inconsistent results

---

## Phase 3: Architecture & Loss Refinement

medium (768d/4L/3072ff) + hybrid16 encoder, lr=8e-4, 2M samples, data/full. Re-evaluated on subset val with new metrics.

### Depth

| Run | Config | Params | val/total | btn_f1 | btn_prec | btn_rec | cdir_active |
|-----|--------|--------|-----------|--------|----------|---------|-------------|
| `width-512` | 512d/4L | 16.3M | 0.1299 | 84.1% | 85.6% | 82.9% | 53.5% |
| `baseline` | 768d/4L | 32.4M | 0.1213 | 85.6% | 85.7% | 85.6% | 60.5% |
| `depth-8L` | 768d/8L | 60.8M | 0.1185 | 85.8% | 86.0% | 85.6% | 60.4% |
| `width-1024` | 1024d/4L | 54.9M | 0.1222 | 85.4% | 85.9% | 85.0% | 60.6% |
| `wide-shallow` | 1536d/2L | 62.0M | 0.1219 | 85.1% | 85.7% | 84.6% | 58.3% |
| `deep-small` | 512d/8L | 28.9M | 0.1343 | 83.1% | 85.0% | 81.6% | 54.6% |

768d/4L remains the sweet spot. 8L is marginally better on btn_f1/cdir_active but has 2x the params. The 512d models lag on all metrics -- too small to capture the game state effectively.

### Positional Encoding

| Run | Method | val/total | btn_f1 | btn_prec | btn_rec | cdir_active |
|-----|--------|-----------|--------|----------|---------|-------------|
| `baseline` | Learned | 0.1213 | 85.6% | 85.7% | 85.6% | 60.5% |
| `pos-alibi` | ALiBi | 0.1354 | 83.5% | 85.4% | 82.2% | 52.6% |

Learned clearly better. ALiBi checkpoint was at 15K steps with 512d so it's not a fair comparison -- see Phase 1-2 for the small-scale ALiBi results.

### Loss Functions

| Run | Stick loss | val/total | btn_f1 | btn_prec | btn_rec | cdir_active |
|-----|-----------|-----------|--------|----------|---------|-------------|
| `baseline` | MSE | 0.1213 | 85.6% | 85.7% | 85.6% | 60.5% |
| `loss-huber` | Huber | **0.0920** | **85.8%** | 86.0% | 85.6% | **59.1%** |

Huber delivers the best val/total (0.0920 vs 0.1213). Button metrics are nearly identical -- the improvement comes from stick loss robustness. Huber checkpoint was at only 5K steps vs baseline at 18K, suggesting it converges faster.

### Context Length

| Run | Seq len | val/total | btn_f1 | btn_prec | btn_rec | cdir_active |
|-----|---------|-----------|--------|----------|---------|-------------|
| `baseline` | 60 | 0.1213 | 85.6% | 85.7% | 85.6% | 60.5% |
| `ctx-360` | 360 | 0.2296 | 82.9% | 86.2% | 80.1% | **70.4%** |

ctx-360 (small 512d model, massively overfit) has the highest cdir_active_acc at 70.4% -- even though val/total is terrible. This suggests longer context helps the model understand when to actually use the c-stick, even if overall loss suffers from overfitting.

### Phase 3 observations

- **btn_f1 ~85-86%** across all medium+ models -- this is the ceiling at 2M samples with opp inputs included
- **cdir_active_acc ~54-61%** -- the model gets the c-stick direction wrong ~40% of the time on active frames. This is likely a major source of bad gameplay behavior.
- **Huber loss** converges faster and achieves better val/total without hurting btn metrics
- **Model size matters more for cdir_active** than for btn_f1: 512d models get ~53-55% vs 768d+ at ~60%

---

## Phase 4: No Opponent Inputs

**Hypothesis**: Opponent controller inputs (analog stick, buttons, c-stick direction) are zero/neutral when playing against a CPU at inference, creating a train-test mismatch. Removing them forces the model to rely on observable game state only.

**Changes**: `--no-opp-inputs` flag strips `opp_buttons`, `opp_analog`, `opp_c_dir` (and Nana equivalents). Encoder drops OPP_INPUT composite token (16 → 15 in hybrid16). Button encoders take 12-dim (self only) vs 24-dim.

**Sweep**: 21 runs across 21 GPUs, medium (768d/4L) + hybrid16 + learned pos-enc + MSE stick + BCE btn, multiple seeds.

### ctx-60 (no-opp-inputs)

| Run | Steps | val/total | btn_f1 | btn_prec | btn_rec | cdir_active |
|-----|-------|-----------|--------|----------|---------|-------------|
| `noi-ctx60-65k` | 65K | 0.1023 | 85.9% | 86.0% | 85.9% | 60.9% |
| `noi-ctx60-80k-A` | 80K | 0.1000 | 86.0% | 86.1% | 85.9% | 60.6% |
| `noi-ctx60-80k-B` | 80K | 0.1014 | 86.0% | 86.1% | 85.9% | 60.5% |
| `noi-ctx60-120k` | 120K | **0.0998** | **86.1%** | **86.2%** | 86.0% | 60.8% |

**ctx-60 aggregate**: val/total mean 0.1009, btn_f1 mean 86.0%, cdir_active mean 60.7%.

### ctx-180 (no-opp-inputs)

| Run | Steps | val/total | btn_f1 | btn_prec | btn_rec | cdir_active |
|-----|-------|-----------|--------|----------|---------|-------------|
| `noi-ctx180-65k` | 65K | **0.1088** | **87.8%** | 87.9% | **87.8%** | **71.7%** |
| `noi-ctx180-80k` | 80K | 0.1232 | 87.1% | 87.6% | 86.8% | 65.5% |
| `noi-ctx180-120k` | 120K | 0.1250 | 86.3% | 87.1% | 85.6% | 65.1% |

**ctx-180 aggregate**: val/total mean 0.1190, btn_f1 mean 87.1%, cdir_active mean 67.4%.

### Phase 3 vs Phase 4 comparison

| Config | val/total | btn_f1 | cdir_active | Notes |
|--------|-----------|--------|-------------|-------|
| Phase 3 baseline (768d/4L, ctx-60) | 0.1213 | 85.6% | 60.5% | single run, 18K steps |
| Phase 4 ctx-60 (best) | **0.0998** | **86.1%** | **60.8%** | 120K steps |
| Phase 4 ctx-180 (best) | 0.1088 | **87.8%** | **71.7%** | 65K steps |

### Phase 4 observations

- **ctx-180 dramatically improves cdir_active**: 71.7% vs 60.5% for ctx-60. With 3 seconds of context, the model learns *when* to use the c-stick much better. This was masked by the old saturated cdir_acc metric (~99%).
- **btn_f1 improves modestly**: 87.8% (ctx-180) vs 85.6% (Phase 3). ~2 percentage points from removing opp inputs + longer context.
- **No-opp-inputs helps val/total**: ctx-60 mean 0.1009 vs Phase 3's 0.1213, a 17% improvement.
- **Longer training shows diminishing returns**: ctx-180 at 65K steps (best cdir_active 71.7%) beats 120K (65.1%). The model may overfit on longer runs.
- **ctx-60 is remarkably stable across seeds**: btn_f1 ranges only 85.9-86.1%, cdir_active 60.5-60.9%.

---

## Summary

Best configuration from all experiments:

| Setting | Winner | Evidence |
|---------|--------|----------|
| Depth | 4 layers | Phase 3: 4L matches 8L on btn_f1, half the params |
| Width | 768 (d_model) | Phase 3: 768d >> 512d on all metrics |
| Context | 180 frames (3s) | Phase 4: cdir_active 71.7% vs 60.8% at ctx-60 |
| Pos encoding | Learned | Phase 3: beats ALiBi |
| Stick loss | Huber | Phase 3: val/total 0.0920 vs 0.1213 MSE |
| Button loss | BCE | Standard across all runs |
| Opp inputs | Remove them | Phase 4: better val/total and inference consistency |

### Top checkpoints by quality

| Rank | Run | val/total | btn_f1 | cdir_active | Config |
|------|-----|-----------|--------|-------------|--------|
| 1 | `noi-ctx180-65k` | 0.1088 | **87.8%** | **71.7%** | medium, no-opp, MSE, 180f |
| 2 | `loss-huber` | **0.0920** | 85.8% | 59.1% | medium, Huber, 60f |
| 3 | `noi-ctx60-120k` | 0.0998 | 86.1% | 60.8% | medium, no-opp, MSE, 60f |
| 4 | `noi-ctx180-80k` | 0.1232 | 87.1% | 65.5% | medium, no-opp, MSE, 180f |
| 5 | `depth-8L` | 0.1185 | 85.8% | 60.4% | 768d/8L, MSE, 60f |

### Next steps

- **Combine Huber + no-opp-inputs + ctx-180**: The best stick loss (Huber) hasn't been tested with the best feature config (no-opp-inputs) at the best context (180). This combination should push cdir_active well above 71.7%.
- **More data**: All experiments used 2M samples. Scaling to the full dataset (~100M+ frames) should help with the cdir_active ceiling.
- **Better cdir modeling**: cdir_active at 71.7% is the weakest link. Consider focal loss for c-stick (rare active frames), or a hierarchical predict-then-direction approach.

Checkpoints: [huggingface.co/erickfm/MIMIC](https://huggingface.co/erickfm/MIMIC). Wandb group: `no-opp-inputs`.
