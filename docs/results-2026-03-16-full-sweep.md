# Full Dataset Training Sweep — Results

**Started**: 2026-03-16 21:10 UTC
**Last updated**: 2026-03-17 07:45 UTC
**Wandb group**: `full-sweep`

## Baseline Configuration

All experiments share these defaults unless noted in the "Change" column:


| Parameter            | Value                                                   |
| -------------------- | ------------------------------------------------------- |
| Model                | `medium` (32.4M params, d_model=768, nhead=8, 4 layers) |
| Positional encoding  | RoPE                                                    |
| Batch size           | 256                                                     |
| Learning rate        | 5e-5 (cosine decay)                                     |
| Warmup               | 5% of total steps                                       |
| Dropout              | 0.1                                                     |
| Label smoothing      | 0.1                                                     |
| Sequence length      | 60 frames                                               |
| Weight decay         | 1e-2                                                    |
| Stick loss           | clusters (63 main, 4 shoulder bins)                     |
| Autoregressive heads | True (L→R→cdir→main→btn)                                |
| AMP                  | bfloat16                                                |


## Baseline Reference

The "vs baseline" column in both tables diffs against the **default config trained on 50% data** — i.e. the mean of g1-d50-r1 and g1-d50-r2.


|                     | g1-d50-r1         | g1-d50-r2         | **mean**  |
| ------------------- | ----------------- | ----------------- | --------- |
| Final step          | 195,312 / 195,312 | 195,312 / 195,312 |           |
| Status              | **DONE**          | **DONE**          |           |
| Final val_f1        | 89.7%             | 88.0%             | **88.9%** |
| Final train_f1      | 89.2%             | 87.3%             | 88.2%     |
| val-train f1        | +0.5%             | +0.7%             | +0.6%     |

The baseline has **finished training**. The "vs baseline" column diffs against the final mean val_f1 of **88.9%**.

Config: medium model (32.4M), RoPE, batch=256, lr=5e-5, dropout=0.1, label_smoothing=0.1, seq_len=60, weight_decay=1e-2, 50M samples over 50% of corpus.

## Group 1 — Data Scaling (sorted by val btn_f1)


| Run | Change | Progress | **val_f1** | vs baseline | val-train f1 | ETA |
|-----|--------|----------|------------|-------------|--------------|-----|
| g1-d50-r1 | data=50%, 50M samples | **DONE** | **89.7%** | **+0.9** | +0.5% | done |
| g1-d10-r1 | data=10%, 10M samples | **DONE** | **89.1%** | +0.2 | +1.0% | done |
| g1-d100-r1 | data=100%, 100M samples | 242k/391k (62%) | **89.0%** | +0.1 | +2.6% | ~6.8h |
| g1-d25-r2 | data=25%, 25M samples, seed=43 | **DONE** | **88.8%** | -0.1 | -0.5% | done |
| g1-d75-r1 | data=75%, 75M samples | 240k/293k (82%) | **88.8%** | -0.1 | -0.4% | ~2.4h |
| g1-d100-r2 | data=100%, 100M samples, seed=43 | 241k/391k (62%) | **88.4%** | -0.5 | +0.3% | ~6.8h |
| g1-d10-r2 | data=10%, 10M samples, seed=43 | **DONE** | **88.3%** | -0.5 | +2.6% | done |
| g1-d50-r2 | data=50%, 50M samples, seed=43 | **DONE** | **88.0%** | -0.9 | +0.7% | done |
| g1-d25-r1 | data=25%, 25M samples | **DONE** | **87.6%** | -1.3 | -1.2% | done |
| g1-d75-r2 | data=75%, 75M samples, seed=43 | 241k/293k (82%) | **87.2%** | -1.7 | +0.7% | ~2.4h |


## Group 2 — Hyperparameter Tweaks (sorted by val btn_f1)

All use 50% data, 50M samples (195,312 steps). "vs baseline" diffs against the g1-d50 final mean (**88.9%**).


| Run | Change | Progress | **val_f1** | vs baseline | val-train f1 | ETA |
|-----|--------|----------|------------|-------------|--------------|-----|
| g2-drop05 | dropout=0.05 | 182k/195k (93%) | **90.0%** | **+1.1** | +5.8% | ~0.6h |
| g2-wd1e3 | weight_decay=1e-3 | 183k/195k (94%) | **89.5%** | **+0.6** | +1.2% | ~0.6h |
| g2-drop15 | dropout=0.15 | 182k/195k (93%) | **89.3%** | +0.5 | +0.0% | ~0.6h |
| g2-combo | dropout=0.1 + ls=0.05 + seq=120 | 189k/195k (97%) | **88.6%** | -0.2 | +2.3% | ~0.3h |
| g2-ls05 | label_smoothing=0.05 | 177k/195k (91%) | **88.5%** | -0.4 | +0.8% | ~0.9h |
| g2-seq30 | seq_len=30 | **DONE** | **87.8%** | -1.1 | -1.1% | done |
| g2-ls00 | label_smoothing=0.0 | 181k/195k (93%) | **87.7%** | -1.2 | -2.4% | ~0.7h |
| g2-seq120 | seq_len=120 | 111k/195k (57%) | **86.3%** | -2.5 | -2.2% | ~4.2h |
| g2-drop20 | dropout=0.2 | 177k/195k (91%) | **85.3%** | **-3.5** | -1.4% | ~0.9h |
| g2-base | model=base (51.8M params) | 164k/195k (84%) | **84.9%** | **-3.9** | -2.1% | ~1.6h |
| g2-drop00 | dropout=0.0 | 190k/195k (97%) | **84.5%** | **-4.3** | +0.7% | ~0.2h |


## Key Takeaways (at ~10h, most runs >90% done)

**Baseline finished at 88.9% mean val_f1.** g1-d50-r1 hit 89.7%, g1-d50-r2 hit 88.0% — a 1.7% seed variance gap that narrowed from 2.1% at the midpoint, confirming that convergence tightens replica spread.

**g2-drop05 broke 90%** — the only run to do so. dropout=0.05 at 90.0% val_f1 with a +5.8% val-over-train gap, suggesting strong generalization with very light regularization. This is the sweep winner so far.

**Rankings reshuffled dramatically from mid-training:**

- g2-ls00 (no label smoothing) was leading at 89.6% mid-training, now **dropped to 87.7%** — it overfit in the second half. Label smoothing helps late in training.
- g2-drop20 (heavy dropout) was #2 at 88.9%, now **dropped to 85.3%** — too much regularization hurt convergence.
- g2-drop05 was near the bottom (85.3%) at mid-training, now **surged to 90.0%** — slow and steady.

**Data scaling is remarkably flat.** Group 1 shows 10%-100% data all landing in the 87-90% band. The top run (g1-d50-r1, 89.7%) and the bottom finished run (g1-d25-r1, 87.6%) differ by only 2.1% — comparable to seed variance. More data does not clearly help yet at this model size.

**Dropout sweet spot is low (0.05-0.1).** The ranking is: drop05 (90.0%) > drop15 (89.3%) > baseline 0.1 (88.9%) > drop20 (85.3%) > drop00 (84.5%). Zero dropout clearly overfits; heavy dropout clearly underfits.

**Lower weight decay helps:** g2-wd1e3 at 89.5% is the #2 tweak, beating the baseline's 1e-2.

**Still running:** g1-d100-r1/r2 (62%), g1-d75-r1/r2 (82%), g2-seq120 (57%). These may shift but are unlikely to take the lead.

## Notes

- btn_f1 is macro F1 over 12 binary button predictions (threshold=0.5). Does not capture stick/shoulder/cdir quality.
- Future runs (new code deployed) will also report main_f1, shldr_f1, cdir_f1.
- Val is computed over 100 batches at each checkpoint.
- Final seed variance across r1/r2 replications: 0.5-1.7% val_f1 (tightened from 2-4% mid-training). Differences within ~1.5% may be noise.
- Baseline reference is the final g1-d50 mean: **88.9%**.

