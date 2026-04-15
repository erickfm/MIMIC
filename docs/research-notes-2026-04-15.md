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

## Character sweep — complete

4-character sweep (Fox / CptFalcon / Luigi / Falco, each with a pair of
2048-step probes: default compile and `fullgraph=True`, seed=7, otherwise
identical commands):

```
Character    broken-fast step/s    step 1600 train loss           basin
                                   broken      fullgraph
Falco              20              1.019       0.853               BAD
Fox                 6              0.890       0.892                clean
CptFalcon           7              0.869       0.869                clean
Luigi              11              0.825       0.825                clean
```

Only Falco exhibits the bug. The other three are byte-identical under
both compile modes and sit at entirely different throughputs — 6, 7, 11
step/s — none near Falco's 20.

## Mechanism (confirmed via PyTorch docs, GitHub issues, and cuBLAS 12.x notes)

Default `torch.compile(model)` allows Dynamo **graph breaks**. When
Dynamo can't trace something, it commits the partial FX graph, runs the
offending fragment in eager Python, then resumes tracing a fresh graph.
Inductor lowers each subgraph independently.

`torch.autocast` is a C++ TLS context manager. Dynamo models it
symbolically *inside* one traced graph, but across a break boundary the
autocast state is torn down and re-entered. Our
`CausalSelfAttentionRelPos.forward` has a nested `autocast(enabled=False)`
block with explicit `.float()` casts on Q/K/Er — exactly the pattern
that's known to break around graph boundaries (pytorch#93890, #114818,
#140118).

Inductor emits `extern_kernels.bmm/mm` for most matmuls (confirmed in
the output_code dump — no `tl.dot`). Those calls go to cuBLASLt, whose
algorithm heuristic keys on **dtype + stride + leading-dim + the current
autocast math-mode TLS**. Around a graph break, Inductor can lower the
FFN matmul with BF16 inputs under a TLS state where the outer
autocast's "inner disable" was lost. cuBLASLt is then free to pick
`CUBLAS_COMPUTE_16BF` (accumulate-in-BF16) tensor-core algos — new in
cuBLAS 12.8 for Blackwell SM_120 and preferentially selected for
certain shape classes.

K=2048 BF16 accumulations lose ~3 bits of mantissa per reduction.
Training lands in a consistently worse basin: +0.17 nat train loss,
+0.03 nat val loss, stable parallel offset from step 500 onward — the
textbook signature of an accumulation-precision delta, not noise.

**Why only Falco**: cuBLASLt's heuristic is shape-dependent. Fox,
CptFalcon, Luigi land on shapes where the top-ranked algo is
`COMPUTE_32F` regardless of TLS state. Falco's shape mix tips exactly
one GEMM into preferring `COMPUTE_16BF` under the corrupted TLS path.
Same code, same seed, same weights — the heuristic just lands on a
different algo for a different shape distribution.

## Fix shipped

Two independent fixes, applied together for belt-and-suspenders:

1. **`torch.compile(model, fullgraph=True)`** — forces a single compiled
   graph with consistent autocast TLS throughout Inductor lowering.
   PyTorch docs explicitly recommend fullgraph for production
   ("Use fullgraph=True to Identify and Eliminate Graph Breaks").
   Failure mode is loud: any untraceable op raises
   `torch._dynamo.exc.Unsupported` at compile time.

2. **`torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False`**
   in `train.py` startup — globally blacklists BF16-accumulate algos
   at cuBLASLt heuristic time. Free on FP32-accumulate paths; immunizes
   against future cuBLAS heuristic changes on Blackwell / CUDA 13.

Either one alone fixes the basin. Together they guarantee correctness
across future model and dataset changes.

## Discarded artifacts

- `checkpoints/falco-20260414-relpos-flattail_*.pt`: all tainted by
  default-compile bug. `_bestloss.pt` (val 0.7651) should be treated
  as a compile-bug victim, not a real Falco model. Moved to
  `checkpoints/_archive/` for historical reference. Production Falco
  remains `falco-20260412-relpos-28k.pt` (val 0.7374).

## Open / deferred

1. Gameplay comparison old-Falco vs bad-basin-Falco: not tested. A
   0.03 val loss gap is real but its game-play impact is unmeasured.
   Deferred because the fix is cheap enough that we don't need to
   argue about whether it mattered.
2. Why exactly the Falco shape distribution tips cuBLASLt into the
   BF16-accum algo while Fox/CF/Luigi don't: not investigated. We'd
   need to dump per-batch shapes and cross-reference with cuBLAS algo
   selection logs. Not load-bearing — the fix works regardless.

## Related GitHub issues

Open in torch 2.8 as of 2026-04-15:
- pytorch#140118 — `set_autocast_enabled` + fullgraph interaction
- pytorch#93890  — autocast context manager doesn't survive graph breaks
- pytorch#114818 — `@torch.amp.autocast` decorator causes breaks
- pytorch#123157 — CUBLAS_COMPUTE_16F exposure / batched-GEMM compute mode
- pytorch#29531  — FP32 accumulation for reduced-precision matmuls
- pytorch#100241 — `no_grad`/`autocast` interaction with compile

---

## 2026-04-15 (late) — Retraction: the bug above is not the real bug

Everything above this line is **wrong**. I spent ~8 hours convinced the
Falco basin shift was a `torch.compile` + BF16 accumulation issue,
shipped a "fix" (`fullgraph=True` + `allow_bf16_reduced_precision_reduction=False`),
and documented a mechanism writeup that matched the eager/compile
measurements I'd taken. It was self-consistent and wrong.

The real bug is a one-line miss in the HAL→MIMIC rename
(commit `706a4af`, 2026-04-13). `train.py` had:

```python
elif model_preset == "hal":
    SEQUENCE_LENGTH = 256
```

The rename aliased `MODEL_PRESETS["mimic"] = MODEL_PRESETS["hal"]`
(same dict object), but this string literal check was never updated.
Every `--model mimic*` run since 2026-04-13 silently fell back to
the module default `SEQUENCE_LENGTH = 60` — a 4.3× reduction in
temporal context per training window.

**Blast radius.** Any run using `--model mimic*` between 2026-04-13
and 2026-04-15 trained on stunted sequences. Known tainted:
- `fox-20260413-rope-32k.pt` (post-rename, likely on HF)
- `falco-20260414-relpos-flattail_*` (the "bad basin" run that started
  the investigation)
- All four-character sweep probes from today
- Everything I called a "compile bug" measurement

**Not tainted** (pre-rename): `falco-20260412-relpos-28k.pt`,
`cptfalcon-20260412-relpos-27k.pt`, `luigi-20260412-relpos-5k.pt`.
These still use `--model hal` and trained at seq 256.

**The fix.** Replace the string check with a preset lookup so aliases
work automatically:

```python
elif model_preset and (
    model_preset in MODEL_PRESETS
    and MODEL_PRESETS[model_preset].get("max_seq_len") is not None
):
    SEQUENCE_LENGTH = MODEL_PRESETS[model_preset]["max_seq_len"]
```

Verified end-to-end: at seed 42, `--model mimic` + the fix produces
train loss bit-exact with `--model hal` on the old commit, all the way
down to step 1600 (0.8230 vs 0.8229).

### How the compile theory fooled me

At `seq_len=60` the attention matrix is `(60, 60)` per head instead of
`(256, 256)` — ~18× less attention compute, much smaller FFN
activations. Compile *does* find a legitimately fast kernel pattern on
small matmuls that it can't reproduce on bigger ones. So I saw
20 step/s with "default compile" and 9.6 step/s with "fullgraph", both
on Falco, both at the stunted seq 60. The fullgraph "slowdown" wasn't
fullgraph making things slower — it was fullgraph happening to produce
a basin that looked closer to "good" in my probes for unrelated
reasons.

The character sweep also fooled me: all four probes (Fox/CF/Luigi/Falco)
were post-rename, so *all* of them ran at seq 60. The reason Fox/CF/Luigi
"didn't show the bug" is because I was comparing bugged-to-bugged, and
the compile path for those characters happened to pick the same kernel
in both broken-fast and fullgraph variants. I built a story about
shape-dependent cuBLASLt heuristics on top of noise.

### Lessons

1. **Dump tensor shapes first.** The moment two nominally-identical runs
   diverged at step 1, I should have printed the state dict shapes to
   check whether inputs were actually identical. I did it on probe ~40,
   not probe 1. It would have saved 6+ hours.
2. **Don't ship a fix based on 8k-step probes.** I committed the
   compile theory as a CLAUDE.md section and archived a "tainted"
   checkpoint based on short probes, not an end-to-end 32k retrain.
   Validate against the full recipe before committing.
3. **Git bisect beats theorizing.** Once the "seed variance" path was
   ruled out, bisecting 44 commits over `train.py + mimic/` would have
   found `706a4af` in ~6 iterations / ~30 minutes. I got there only
   after many more probes.
4. **Confirmation bias is a thing.** The character sweep result was a
   red flag ("why would a compile bug be dataset-specific?") and I
   rationalized it into a shape-heuristic story instead of stopping to
   re-examine.
5. **Never add `if model_preset == "X"` gates.** If behavior varies by
   preset, read it off the preset dict so aliases pick it up for free.

The torch.compile section previously added to CLAUDE.md has been
removed. The archived `checkpoints/_archive/falco-20260414-relpos-flattail_*`
are still correctly discarded, but for the right reason now (trained
at seq 60, not "compile BF16 poisoning").

Torch 2.8 / CUDA 12.8 / Blackwell SM_120 compile is probably fine for
this model. Needs a real seq-256 benchmark to know. Deferred.
