# FRAME Experiment Results

## Phase 1-2: Encoder & Backbone (completed)

Ran on full dataset. Key findings that informed Phase 3 defaults:

- **hybrid16** encoder won over flat, composite8, and default (55-token) encoders
- **medium** backbone (768d/4L) hit the best loss-per-param sweet spot at ~32M params
- **lr 8e-4** outperformed 3e-4 and 5e-4 at this scale
- Longer context helped (ctx-180, ctx-240 beat ctx-60) but with diminishing returns
- ALiBi positional encoding showed promise but had a bug (fixed mid-run)

## Phase 3: Architecture & Loss Refinement (in progress)

Baseline: medium (768d/4L/3072ff) + hybrid16 encoder, lr=8e-4, 2M samples, data/full

### Axis 1: Depth


| Run        | Layers | Params | Batch | step/s | val/total | val/cdir_acc | val/btn_acc | Notes   |
| ---------- | ------ | ------ | ----- | ------ | --------- | ------------ | ----------- | ------- |
| `depth-2L` | 2      | 18.3M  | 512   | 2.3    |           |              |             | Running |
| `baseline` | 4      | 32.4M  | 384   | 3.0    |           |              |             | Running |
| `depth-6L` | 6      | 46.6M  | 256   | 4.1    |           |              |             | Running |
| `depth-8L` | 8      | 60.8M  | 192   | 4.8    |           |              |             | Running |


### Axis 2: Width


| Run          | d_model | Params | Batch | step/s | val/total | val/cdir_acc | val/btn_acc | Notes   |
| ------------ | ------- | ------ | ----- | ------ | --------- | ------------ | ----------- | ------- |
| `width-512`  | 512     | 16.3M  | 512   | 2.5    |           |              |             | Running |
| `baseline`   | 768     | 32.4M  | 384   | 3.0    |           |              |             | Running |
| `width-1024` | 1024    | 54.9M  | 256   | 4.4    |           |              |             | Running |


### Axis 3: Context Length


| Run        | Seq len | Time | Batch | step/s | val/total | val/cdir_acc | val/btn_acc | Notes   |
| ---------- | ------- | ---- | ----- | ------ | --------- | ------------ | ----------- | ------- |
| `ctx-30`   | 30      | 0.5s | 768   | 0.8    |           |              |             | Running |
| `baseline` | 60      | 1.0s | 384   | 3.0    |           |              |             | Running |
| `ctx-90`   | 90      | 1.5s | 256   | 1.9    |           |              |             | Running |
| `ctx-120`  | 120     | 2.0s | 192   | 2.3    |           |              |             | Running |
| `ctx-180`  | 180     | 3.0s | 128   | 2.9    |           |              |             | Running |


### Axis 4: Positional Encoding


| Run        | Method  | Batch | step/s | val/total | val/cdir_acc | val/btn_acc | Notes   |
| ---------- | ------- | ----- | ------ | --------- | ------------ | ----------- | ------- |
| `baseline` | Learned | 384   | 3.0    |           |              |             | Running |
| `pos-rope` | RoPE    | 384   | 1.4    |           |              |             | Running |


### Axis 5: Loss Functions


| Run              | Stick loss     | Btn loss  | Batch | step/s | val/total | val/cdir_acc | val/btn_acc | Notes                                         |
| ---------------- | -------------- | --------- | ----- | ------ | --------- | ------------ | ----------- | --------------------------------------------- |
| `baseline`       | MSE            | BCE       | 384   | 3.0    |           |              |             | Running                                       |
| `loss-huber`     | Huber          | BCE       | 384   | 1.6    |           |              |             | Running                                       |
| `loss-discrete`  | Discrete 32x32 | BCE       | 384   | 1.4    |           |              |             | Running; train loss scale differs (CE vs MSE) |
| `loss-focal-btn` | MSE            | Focal BCE | 384   | 1.5    |           |              |             | Running                                       |


### Observations (early, pre-validation)

- **Deeper models train faster per-step** (depth-8L: 4.8 step/s vs depth-2L: 2.3 step/s) -- smaller batch sizes mean less work per step
- **ctx-30 struggles**: higher loss and lower cdir_acc early on (97.5% vs 99%+), suggesting 0.5s of context is insufficient
- **loss-discrete** has much higher total loss (~2.5 vs ~0.09) because CE over 1024 bins is a harder objective than MSE -- need to compare on val metrics not raw loss
- **loss-focal-btn** shows lower btn_acc (~~97%) compared to baseline BCE (~~99.4%) early on -- focal loss intentionally focuses on hard examples, so easy-button accuracy drops while it learns the rare ones
- **pos-rope** converging slower than learned pos-enc early on (cdir_acc 99.3% vs 98-99%), may need more warmup

---

## Phase 4: Champion (planned)

Combine best settings from Phase 3 and train longer on full data.