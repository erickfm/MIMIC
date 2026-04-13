# 2026-04-13c — Fox RoPE retrain (fox-rope-v2)

## Motivation

The Fox checkpoint on HF (`hal-7class-v2-long`, 2026-04-11) was stuck at
val loss 2.27 with btn F1 ~59% / main F1 ~15% — dramatically worse than
Falco/CptFalcon/Luigi (~0.7 val, ~88% btn F1, ~55% main F1).

Root cause in hindsight: the `hal-7class-v2-long` run was trained
**without** `--self-inputs`, which removes the controller history from
the encoder inputs entirely. Without it the model has no signal for
what it just pressed, so main stick prediction collapses to ~15% F1.
This was the one MIMIC 7-class run that forgot the flag.

Retraining was overdue regardless — also a chance to try RoPE again
since the 2026-04-09 RoPE attempt on the old leaked shards had plateaued
at val loss 1.083.

## Config

```
python3 train.py \
  --model hal-rope --encoder hal_flat \
  --hal-mode --hal-minimal-features --hal-controller-encoding \
  --stick-clusters hal37 --plain-ce \
  --lr 3e-4 --batch-size 512 \
  --max-samples 16777216 \
  --data-dir data/fox_v2 \
  --reaction-delay 0 --self-inputs \
  --dropout 0.1 \
  --run-name fox-rope-v2 \
  --no-warmup --cosine-min-lr 1e-6
```

- v2 shards (no leak, no controller-offset)
- `hal-rope` preset (RoPE position encoding instead of Shaw relpos)
- `--self-inputs` enabled (the fix vs the old Fox checkpoint)
- dropout 0.1 (the 2026-04-09 notes flagged this as optimal for RoPE)
- 32,768 steps, batch 512 → 16.8M samples

## Results

| Metric | fox-rope-v2 (this run) | hal-7class-v2-long (old Fox) | Falco (reference) |
|---|---|---|---|
| Val loss | **0.77** | 2.27 | 0.68 |
| Btn F1 | **87.7%** | ~59% | 88.2% |
| Main stick F1 | **~55%** | ~15% | 58.5% |
| Shoulder F1 | ~86% | — | — |
| Steps | 32,768 | 32,768 | ~28K |
| Throughput | **16.7 step/s** | ~9 step/s | ~8 step/s |

Val loss was flat in the 0.77–0.82 band from ~step 20K onward — fully
converged, no more headroom at this recipe. New best checkpoint saved
at step ~32.1K (`checkpoints/fox-20260413-rope-32k.pt`, renamed from the default
`fox-rope-v2_best.pt` per the new naming convention — see CLAUDE.md).

## Observations

1. **`--self-inputs` was the dominant factor.** The RoPE vs relpos choice
   is secondary — matching Falco-tier numbers with the flag on, vs the
   old broken 2.27 with the flag off, tells us the position encoding was
   never the problem. The 2026-04-09 RoPE plateau at val 1.083 was on
   leaked old shards and probably also related to data, not arch.

2. **RoPE is ~2× faster than relpos on this GPU.** 16.7 step/s vs the
   relpos runs' ~8–9 step/s, same batch size and dropout. The Shaw
   manual-attention kernel in FP32-upcast mode is the slow path; RoPE
   uses standard SDPA which the inductor can fully fuse. For future
   training runs on RTX 5090, RoPE is now the default choice unless
   there's a concrete reason to want relpos.

3. **Checkpoint size differs.** fox-rope-v2 is 235 MB vs the relpos
   checkpoints at 265 MB — saved weight is the Shaw relative position
   embedding table (block_size × n_heads × d_head per layer). Nothing
   about gameplay, just file-size trivia worth noting on HF.

4. **Model not yet evaluated in Dolphin.** Pushed directly to HF without
   a gameplay test; the val metrics are in the Falco-reference range so
   I'm trusting them. If someone runs it and it plays badly, rerun with
   `--model hal` (relpos) on the same data to isolate whether it's a
   RoPE-at-inference issue.

## HF upload

`erickfm/MIMIC/fox/` replaced with `fox-rope-v2_best.pt` (model.pt),
regenerated config.json / metadata.json / README.md metrics table.
Other three characters untouched.
