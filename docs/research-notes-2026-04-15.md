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

## Post-fix Fox retrain — 2026-04-15

Retrained Fox on the corrected seq_len=256 path with default compile,
identical to the 2026-04-13 recipe otherwise:

```
--model mimic-rope --encoder mimic_flat
--mimic-mode --mimic-minimal-features --mimic-controller-encoding
--stick-clusters hal37 --plain-ce
--lr 3e-4 --batch-size 512 --max-samples 16777216
--data-dir data/fox_v2 --reaction-delay 0 --self-inputs
--no-warmup --cosine-min-lr 1e-6
```

Result at step 25,179 (best val): **val_loss=0.7358**. Saved as
`checkpoints/fox-20260415-rope-25k.pt` (from `_bestloss.pt`).

Comparison to the historical Fox checkpoints:
- `fox-20260415-rope-25k.pt`:       val 0.7358 (seq 256, post-fix)
- `fox-20260413-rope-32k.pt`:       val ~0.77 (seq 60, tainted — superseded)
- `fox-20260411-relpos-noself-28k`: val ~0.77 (no self-inputs, pre-rename)

~0.04 nat improvement over the tainted seq-60 run and the older
no-self-inputs baseline. Consistent with the Falco seed=42 probe
(0.8229 → 0.8230 post-fix) — the fix recovers the full training
context and the models land where they should.

Steady-state throughput was ~5.5 step/s for the full Fox run at seq
256 on the RTX 5090, not the ~2 step/s I'd benchmarked earlier in the
investigation. Earlier benchmark was contaminated by stale GPU state;
Fox running clean on an otherwise-idle GPU is quite fast.

Fox retrain total wall clock: ~100 min for 32k steps.

## Window-size sweep — how much does seq length actually buy us

Once the fix was confirmed, natural follow-up: how much of the ~0.04
nat gap between seq 60 and seq 256 comes from each doubling, and is
there a bigger win above 256?

Setup: Falco, seed=42, batch 512, compile default, 4096 steps per
probe, seeds/data identical except `--seq-len`. Seq values:
32 / 64 / 128 / 256 / 384 / 512. Sequential run via
`/tmp/run_sweep.sh` on an otherwise idle GPU.

Results:

```
seq   step   train    best_val   rate (step/s)   OOM (halved?)
 32   4096   0.8621   0.8421     8.60             0  clean batch 512
 64   4096   0.8190   0.8170     6.40             0  clean batch 512
128   4096   0.8113   0.7949     4.20             0  clean batch 512
256   4096   0.7840   0.7757     2.20             0  clean batch 512   ← production
384   4096   0.7566   0.7728     1.70             2  OOM'd, batch 256
512   4096   0.7866   0.7753     1.40             4  OOM'd, batch 128
```

### Clean region (seq 32 → 256, batch 512 throughout)

Best val is monotonic and shows a textbook diminishing-returns curve:

```
seq doubling    Δ best val
 32 → 64       −0.025
 64 → 128      −0.022
128 → 256      −0.019
```

Each doubling buys ~0.02 nat of val loss improvement at this training
length. The sum (seq 32 → 256) is −0.066 nats, consistent with the
0.045–0.065 gap we observed when the rename silently dropped the
window from 256 to 60 and every `--model mimic*` run trained stunted.

### Above 256 — tainted by OOM-halving, not directly comparable

seq 384 and 512 cannot fit batch 512 on a 32 GB 5090 at this model
size — the attention activations are quadratic in seq. train.py's
OOM-retry halves the batch on the first OOM, so:
- seq 384 effectively ran at batch 256
- seq 512 effectively ran at batch 128

Smaller batch = more optimizer steps per sample = lower train loss at
the same step count, independently of the window change. The seq 384
train loss (0.7566) is noticeably lower than seq 256 (0.7840) partly
because of that, not because seq 384 "learns better" per se. Can't
separate the two effects from one run.

Seq 512 is suspicious: its train loss bounces *up* vs seq 384 (0.7866
vs 0.7566), probably because batch 128 is too small to offset the seq
penalty. Best val tracks it: 0.7753 vs 0.7728.

### Conclusion

**seq 256 is on the sweet spot of the curve.** Every doubling before
256 gives a clear win; every doubling after is either noise or tainted
by batch-size side effects.

If you want to push past 256 properly, you need either gradient
accumulation (keep batch 512 logical, split physical), a smaller model
(fewer params → fits longer sequences), or a larger GPU. Not worth
doing blind — benchmark first.

Throughput follows the attention-is-quadratic law reasonably well:

```
seq   step/s   rel to seq 256
 32   8.60     3.9×
 64   6.40     2.9×
128   4.20     1.9×
256   2.20     1.0×
384   1.70     0.77×  (but batch 256)
512   1.40     0.64×  (but batch 128)
```

The 32 → 256 throughput ratio is ~4×, not the 64× pure attention math
would suggest — FFN / overhead / memory bandwidth dominate at small
seq lengths. At larger seq the crossover makes attention more
important and throughput drops steeper.

### Intermediate points: seq 180 (3s) and seq 192 (3.2s)

Filled in two extra points between 128 and 256 to probe the region
where Melee-semantic window sizes live. Both at seed=42, same config
as the main sweep:

```
seq   wall     best_val
180   22.8m    0.7840
192   23.5m    0.7858
```

seq 180 beat seq 192 by ~0.0018 nats — small but surprising given the
otherwise smooth curve. Reran both at seed=7 to check if it was
pure variance:

```
seq    seed42   seed7
180    0.7840   0.7842
192    0.7858   0.7856
```

**Both seeds reproduce the ordering**: seq 180 < seq 192 by ~0.0015
nat regardless of seed. Not noise — small and reproducible.

Most likely mechanism: **leftover-frame waste**. The pre-windowed
shards contain games of varying length, and each game contributes
`floor(game_frames / seq_len)` windows with the tail dropped. At seq
192 vs seq 180, more of each game's tail lands below the window
threshold and gets discarded, so the effective dataset is slightly
smaller at 192 than at 180. Over 4k training steps with batch 512,
the 192 run has seen marginally fewer unique windows than the 180
run, which costs a hair of val loss.

This is an artifact of shard construction, not a model-architecture
effect. The practical lesson is that val-loss curves across seq
lengths won't be perfectly smooth — sequence lengths that divide
typical game lengths evenly will look slightly better per training
step than ones that don't. The overall diminishing-returns shape
still holds.

If you ever need a sub-256 "cheap but good" sequence length, prefer
seq 180 over seq 192 — same wall time, same step count, slightly
better val, same seed variance.

### New default: seq 180

Bumped all `MODEL_PRESETS` entries in `mimic/model.py` from
`max_seq_len=256` to `max_seq_len=180`, and the module-level
`SEQUENCE_LENGTH = 60` fallback in `train.py` to `180`.

Rationale: seq 180 gives ~92% of the basin improvement of seq 256
(0.7840 vs 0.7757 best val at 4k steps, both from seq 32's 0.8421)
at **70% of the wall time** (22.8m vs 32.5m per 4k-step probe).
The extra 0.008 nat at seq 256 is real but not load-bearing given
the throughput cost. Production Fox retrain from earlier today
(`fox-20260415-rope-25k.pt`) is at seq 256; next retrain can drop
to 180 if that margin matters less than training speed.

Note: existing checkpoints trained at seq 256 (e.g.
`fox-20260415-rope-25k.pt`, `falco-20260412-relpos-28k.pt`) continue
to load fine because `max_seq_len` is a buffer dimension at load
time — shorter inference-time sequences just use the first 180
positions of the learned/registered position stuff. Longer than 180
would need the old preset, which we've now overwritten. If a legacy
checkpoint needs seq 256 decoding, override with `--seq-len 256`
on the CLI at training/eval time.

## Auto-converge recipe hunt — WSD vs cosine + patience cleanup

**The problem.** Different characters have wildly different optimal
training lengths (Fox 17K games → ~25–28K steps; Luigi 2K → ~5K).
Picking `--max-samples` blind either under-trains or over-trains. Goal:
**one recipe that converges to the best achievable val loss on any
character without prior knowledge** — launch, walk away, get a finished
model at the right plateau.

The conversation cycled through three drafts before landing on the
boring answer.

### Draft 1 — patience-based knee detector (EMA + min_delta)

Added an EMA-smoothed val loss, `--patience N`, `--patience-min-delta`,
`--patience-ema-alpha`, plus `--patience-stop` (vs free-mode logging).
Five wandb metrics. Three hyperparameters. The pitch: "smoothed val
loss tells you when to stop."

Tested with `--patience 5 --patience-ema-alpha 0.3` on the new
master-master Fox dataset (see below). Knee fired at step 8502 reporting
`best_val_ema=0.8273 @ step 6867`. **The detector was wrong.** Free-mode
showed raw val loss continuing to fall to 0.7599 at step ~29k — the
"knee" was a noise trough on a smoothed signal that was lagging the
actual minimum by ~20k steps.

Lesson: smoothing a noisy signal then thresholding the smoothed version
hides the real minimum behind the smoother's lag. The end-of-run
summary did correctly print the raw `best_val_loss_step` because that
state was tracked separately for `_bestloss.pt`, so the diagnostic was
salvageable, but the in-loop fire was unusable.

### Draft 2 — strip the EMA, just count raw evals

Dropped `--patience-min-delta`, `--patience-ema-alpha`, all four EMA
metrics. Counter ticks against raw `best_val_loss`; `--patience N`
decides when to fire. Still a knob for halt vs free-mode. Three lines
of code instead of forty.

### Draft 3 — WSD (Warmup-Stable-Decay), patience triggers the decay

The cosine schedule has a fundamental problem: it needs to know the
endpoint to shape the LR curve, but we don't know the endpoint. WSD
solves this on paper — flat LR until val plateaus, then a short cosine
decay produces the final refined checkpoint. Patience fires the
stable→decay transition; run halts after the decay phase.

Implemented as a manual LR setter (no scheduler), `--wsd` and
`--wsd-decay-steps 2000`. Smoke test passed. Then three real probes
on `data/fox_master_v2` at seq=60 (pre-fix taint, see below):

| Run     | Schedule                              | Steps   | Best val_loss |
|---------|---------------------------------------|---------|---------------|
| Probe-1 | cosine 3e-4 → 1e-6 across 32k         | 32,768  | **0.7599**    |
| Probe-2 | WSD, ceiling clipped (no decay phase) | ~32,768 | 0.7677        |
| Probe-3 | WSD: stable 34,749 + decay 2,000      | 36,749  | 0.7727        |

Probe-3 trained ~15% **longer** than probe-1 and got a **worse** result.
Worse: every "new best val_loss" line in probe-3's log came from the
stable phase (last one at step 26,169). The decay phase from step
34,749 → 36,749 produced **zero** new bests.

Why WSD lost here: stable LR held at 3e-4 for the entire run. Cosine
had already annealed to ~1e-4 by step 26k where the model was doing
its productive work, and to ~3e-5 by step 32k for final refinement.
The literature consensus is that the WSD decay phase should be
**10–20% of the stable phase length** — for probe-3 that would have
been 3,500–7,000 steps, not 2,000. The fixed 2,000 was ~5.7%, way
too short. The fix is `--wsd-decay-frac` (proportional decay) but
that's another knob, and at this point we're three drafts deep on a
recipe that the existing cosine schedule already handles cleanly.

**The headline finding: WSD works, the loss penalty is intolerable.**
The mechanism — patience watches val, fires the stable→decay transition,
decay phase produces the final checkpoint, run halts cleanly — all
worked end-to-end on probe-3. The plumbing is sound. The problem is
the *result*: probe-3 trained 15% **longer** than cosine probe-1 and
landed at val_loss 0.7727 vs probe-1's 0.7599 — a **+0.013 nat penalty**.
For a BC bot whose entire output quality is downstream of val loss,
that's not a wash, that's a real regression. WSD's pitch was "no prior
knowledge needed about the knee," but the cost is a model that's
measurably worse at predicting controller inputs. Even with a
better-tuned `--wsd-decay-frac`, the structural gap (peak LR for the
entire stable phase vs cosine annealing through the productive zone)
is going to cost *something*. Cheaper to just run a probe and use
cosine.

### Draft 4 (final) — drop WSD, drop patience, keep cosine + two-run pattern

The honest realization: with cosine, `--patience` adds nothing. The
LR schedule is locked at startup and doesn't react to anything during
training. Free-mode patience just prints what `_bestloss.pt` already
records. Halt-mode patience *fights* cosine — cosine's whole value is
the final low-LR refinement window, and halting before that window
throws the schedule's whole point away. Probe-1 vs probe-2/3 showed
exactly this: cosine across the full 32k beat both WSD attempts.

So the recipe collapsed back to:

1. **Probe**: run cosine across a generous `--max-samples`. End-of-run
   prints `Best val_loss=X @ step Y. Use --cosine-decay-steps Y for
   next run on this data.` (Same info also lives on disk as the
   modification time of `_bestloss.pt`.)
2. **Production**: same recipe but `--cosine-decay-steps Y`. Cosine
   anneals exactly to the knee, then flat-tails.

Two runs per character. Boring. Works. The "auto-converge in one run"
ambition wanted a single magic schedule that adapts to the data
without prior knowledge; what we found instead is that prior knowledge
is cheap (one probe per character, ~1 hour) and worth way more than
any clever adaptive schedule we could rig up.

### Code state after the cleanup

- Removed: `--wsd`, `--wsd-decay-steps`, `--patience`, `--patience-stop`,
  `--patience-min-delta`, `--patience-ema-alpha`, all EMA state, all WSD
  manual-LR plumbing, the WSD-vs-cosine branch in scheduler construction.
- Kept: existing cosine + flat-tail builder, `_bestloss.pt`, `val/total`,
  `val/train_ratio`. Added two new wandb metrics for visibility:
  `val/evals_since_improve` (counter against raw best_val_loss, ticks
  every val without an improvement) and `val/best_val_loss_step` (the
  step where the current best was set). Both are diagnostic-only — no
  control flow uses them.
- Added: end-of-run `_log` line that prints
  `Best val_loss=X @ step Y. Use --cosine-decay-steps Y for next run`
  whenever a finite best_val_loss exists.

Net change in `train.py`: -50 LOC, -4 CLI flags. Three drafts of
features got built and ripped out. The cleanup is the actual ship.

### `data/fox_master_v2` — new dataset built today

To probe the WSD recipe I needed a clean Fox dataset I wasn't already
training production models on. Built from
`erickfm/melee-ranked-replays` shards (`tools/shard_and_upload_ranked.py`,
documented in `docs/ranked-dataset-pipeline.md`):

- Downloaded `FOX_master-master_a1.tar.gz` (0.26 GB, 284 replays) and
  `FOX_master-master_a3.tar.gz` (7.4 GB, 9018 replays). Only the
  master-master bucket — `master-platinum`, `master-diamond` etc. would
  pollute the skill mix because the tarball name doesn't tell you which
  Fox is which rank.
- Extracted to `data/raw_slp/fox_master_master/` (9302 .slp files, 33 GB).
- `tools/slp_to_shards.py --character 1 --shard-gb 0.8 --val-frac 0.05`
  produced **9866 train games / 96M frames in 191 shards**, plus
  **516 val games / 5M frames in 11 shards**. ~15 min wall clock.
  152 GB total on disk.

Pass rate from raw .slp to kept frames: ~9866/9302 = >100%. The >1
ratio is the "both perspectives" expansion in `slp_to_shards.py` —
each Fox-vs-Fox replay produces two training perspectives. The 5%
val_frac at file granularity gave 11 val shards which is a perfectly
fine eval pool.

### IMPORTANT: probe results above are tainted

All three WSD-era probes used `--model mimic`, which on the
pre-88eecf1 code path silently set `SEQUENCE_LENGTH=60` (see top of
this note for the bug). The probe val losses (0.7599 / 0.7677 / 0.7727)
are all ~0.04 nat too high relative to the post-fix world. The
**relative ordering** of cosine vs WSD almost certainly survives the
fix — cosine's full schedule will still beat WSD with a 5.7% decay
window — but the absolute numbers are wrong. The cleanup decision
(drop WSD, drop patience, keep cosine + two-run) doesn't depend on
the absolute numbers, so it stands.

A clean post-fix knee probe on `fox_master_v2` should land somewhere
in the 0.69–0.71 range based on the upstream Fox seq-256 retrain
(0.7358 on `fox_v2` — `fox_master_v2` is higher-skill data, lower
diversity, plausibly slightly higher floor). Deferred — the recipe
itself is now boring enough that I don't need a clean probe to know
how to use it.

### Discarded artifacts

All 71 checkpoint files from today's six runs were deleted (~17.7 GB
reclaimed):

- `checkpoints/smoke-patience_*.pt` (3 files) — 16-step crash test
- `checkpoints/smoke-wsd_*.pt` (3 files) — 16-step WSD code-path test
- `checkpoints/fox-master-knee-probe-1_*.pt` (23 files, 5.8 GB)
- `checkpoints/fox-master-knee-probe-2_*.pt` (23 files, 5.8 GB)
- `checkpoints/fox-master-knee-probe-3_*.pt` (18 files, 4.6 GB)

Every probe ran on the seq_len=60 path and is therefore not directly
comparable to post-fix Fox numbers. The actual scientific output of
the day is this research note plus the train.py simplification commit
(924f98e) — both preserved. Regenerating any of the probes is a
one-line train.py invocation if needed; storing the artifacts cost
more than reproducing them ever will.

The intermediate `_step{NNNNNN}.pt` snapshots in particular were the
real bulk of the waste — the default save-every-N-steps cadence
produces ~20 files per 32k-step run, almost all of which serve no
purpose post-hoc once `_best.pt` and `_bestloss.pt` are in hand. A
future `train.py` cleanup could plausibly drop the step snapshots
entirely (or gate them behind an opt-in flag), since they're
load-bearing for resume only and resume is rarely used in this
project.
