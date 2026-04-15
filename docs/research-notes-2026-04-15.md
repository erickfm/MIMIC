# 2026-04-15 — torch.compile BF16 precision bug + train.py enhancements

Two threads today. First: new train.py features for monitoring and schedule
control. Second: discovery and investigation of a `torch.compile`
precision bug that gives a 2× speedup at the cost of landing in a worse
training basin — currently believed to be character-specific (Falco hit,
Fox clean, CptFalcon/Luigi in progress).

## train.py — new features

**`val/train_ratio` metric.** Each val eval now logs `val/total` divided by
a rolling 100-step mean of `train/total`. A ratio >1 and rising is the
classic overfitting signature. Also logs `train/rolling_total` alongside.
Zero-cost (just a `deque(maxlen=100)` append per step).

**`_bestloss.pt` checkpoint.** Separate from `_best.pt` (tracked by button
F1), the new checkpoint tracks the lowest `val/total` across training. Both
live in `checkpoints/` with the run name as prefix. Lets us recover the
lowest-loss point even if the run overtrains or F1 peaks at a different
step than val loss.

**`--cosine-decay-steps N`.** Decouples the cosine decay length from total
training length. Old behavior: cosine stretched across all `max_steps` →
a 65k-step run keeps LR too high at step 32k (where shorter runs would
already be cooling off). New: cosine decays to `eta_min` over `N` steps,
then holds flat at `eta_min` for the remainder. Implemented via
`SequentialLR([cosine, flat])` where `flat = ConstantLR(factor=eta_min/peak_lr)`.

Usage for a 65k-step Falco run matching a 32k cosine recipe:

```
--max-samples 33554432 --cosine-decay-steps 32768
```

## torch.compile BF16 precision bug — investigation in flight

### The symptom

Falco run `falco-20260414-relpos-flattail` (torch 2.8, default compile)
landed at val loss 0.7651 vs the old `falco-20260412-relpos-28k` run at
0.7374. Train loss also shifted up by ~0.07 nats from step 500 onward —
a stable parallel offset, not a schedule/seed issue.

### The mechanism (so far)

- Eager BF16: good basin, 8.2 step/s.
- Compile default (torch 2.8): **bad basin, 20 step/s**.
- Compile + `allow_bf16_reduced_precision_reduction=False`: good, 8.8 step/s.
- Compile + `fullgraph=True`: good, 9.6 step/s.
- torch 2.11 default compile: good, 8.4 step/s.
- FP32 FFN (via explicit autocast-disable): good, 5.4 step/s.

The FP32-FFN fix confirms the bug lives in the FFN linears (not attention —
attention's Q/K/Er was already FP32 via explicit `.float()` casts, verified
by inspecting Inductor's `output_code` dump). On torch 2.8, Inductor picks
an aggressive cuBLAS BF16-GEMM algorithm that accumulates in BF16 instead
of FP32. K=2048 BF16 accumulations lose enough precision that training
lands in a worse basin after 32k steps.

Eager cuBLAS picks a more conservative algo despite the same flag state.
The `allow_bf16_reduced_precision_reduction=False` flag restricts algo
selection to FP32-accumulate — fixes compile but also costs compile's speed.
No per-op precision knob in cuBLAS/Inductor — either all BF16 GEMMs use
BF16-accum or none do.

### Character dependence (in progress)

| Character   | broken-fast compile | good basin? | speedup? |
|-------------|---------------------|-------------|----------|
| Falco       | bad (1.02 @ step 1600) | no       | 20 step/s |
| Fox         | fine (0.89 @ step 1600) | yes      | 6 step/s  |
| CptFalcon   | in progress         | ?           | ?         |
| Luigi       | not yet tested      | ?           | ?         |

Fox broken-fast and Fox fullgraph produce identical train loss to 4
decimals — Fox doesn't trigger Inductor's aggressive kernel path at all
(throughput is 6-7 step/s, same as fullgraph). Possibly shape-dependent
kernel selection: the inductor dump on Falco showed specialized kernels
at seq_len=60, suggesting the compiled shape depends on the data
distribution of the first few batches. Different character datasets may
produce different batch shape distributions during warmup → different
kernel selection → different precision tradeoff.

### Current recommendation

Until character sweep is complete:

- **Falco**: use `torch.compile(model, fullgraph=True)`. Known-working, 9.6 step/s.
  Do not use default compile.
- **Fox**: default compile is fine. Same speed, same result.
- **CptFalcon, Luigi**: TBD pending probes.

If in doubt, `allow_bf16_reduced_precision_reduction=False` globally is
the most mechanistically understood fix and works for all characters at
~8.8 step/s.

### Related: falco-20260414-relpos-flattail is tainted

That run's `_bestloss.pt` (0.7651) landed in the bad basin due to
default compile. Should be discarded and retrained with `fullgraph=True`
once the character sweep confirms the picture. Old `falco-20260412-relpos-28k`
(val 0.7374) remains the production Falco checkpoint.

## Open questions

1. Does CptFalcon/Luigi trigger the aggressive kernel? If yes, same fix.
   If no, compile is safe by default for those characters.
2. Why does Fox compile at 6 step/s while Falco compiles at 20? Same hw,
   same model, same config. Only data-dir differs. Possibly linked to
   Dynamo's shape specialization during warmup.
3. Is the bad-basin Falco actually worse in gameplay, or just a val loss
   artifact? Not tested. Gameplay comparison deferred.
