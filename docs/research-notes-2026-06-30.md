# 2026-06-30 — FFW re-examined: it's faithful. Parallel Dolphins need batched inference.

## TL;DR

- **FFW is faithful.** The 2026-05-18 "FFW is unfaithful (matches ~4× short)"
  finding was a **misdiagnosis**. Under `--use-exi-inputs --enable-ffw` with
  `blocking_input=True` (our default), the canary suite passes: L-cancel rate
  **91.5%** (n=588) vs realtime 88% / corpus 90.4%, match length **normal**
  (6,385 frames ≈ realtime 6,600), at **~2.9× realtime** wall-clock.
- **Our setup is byte-for-byte slippi-ai's** (vladfi1, Phillip's successor):
  same Fizzi gecko codeset, same `use_exi_inputs + enable_ffw + blocking_input`.
- **Naive parallelism (N independent `play.py`) does NOT scale.** 1→4 instances
  moved throughput only 2.9× → 3.5× realtime; the GPU pins at 90% while 28/32
  CPU cores idle. Bottleneck = **batch-1 model inference**, not Dolphin/CPU.
- **The fix is batched multi-env inference** (one process, N Dolphins, one
  batched policy forward), exactly what slippi-ai's `AsyncBatchedEnvironment`
  does. That's also the RLVR rollout harness — build once, reuse. Built two:
  threaded (`tools/ffw_batch_bench.py`, 9.6×, GIL-bound) and **multiprocess async
  (`tools/ffw_batch_mp.py`, ~30× realtime, ~1,800 fps at N=32)** — the machine
  ceiling, ~10× the single-instance baseline, CPU-bound. Fizzi hit ~32× only by
  adding a CPU-overclock; we get ~30× from parallelism alone.

## How we settled faithfulness (canaries, not match length)

The 2026-05-18 note inferred "EXI input path mistiming" purely from **match
length** in a **bot-vs-bot** run (FFW 2,430 vs realtime 9,687 frames). That is
exactly the *symmetric-degradation* trap that note itself warned about: anything
that hobbles both bots equally makes them trade stocks faster → short matches →
looks "unfaithful" with zero actual infidelity. Match length can't distinguish
"both bots worse" from "emulator broken."

The 2026-06-29 canaries **directly measure the bot's input timing/fidelity**, so
they close that gap. Ran the champion (`AVG_mastfox`) vs CPU-7 under FFW,
headless (`gfx=Null`, `:99`), 24 matches, and scanned the replays:

| signal | FFW | realtime baseline | corpus | verdict |
|---|---|---|---|---|
| **L-cancel rate** | **91.5%** (n=588) | 88% | 90.4% | healthy — timing preserved |
| match length | 6,385 frames | ~6,600 | — | normal, not 4×-short |
| wall-clock | **2.9× realtime** | 1× | — | speedup real |

L-cancel is the *timing* sentinel; if FFW mistimed inputs it would crater.
Instead it sits at the corpus value. FFW is faithful.

**Why it's faithful (mechanism).** Fizzi's codeset (`AI/LoopMainEngine/*`) runs
the game's *real* engine loop multiple times per video frame — it is NOT
frame-skipping. With `blocking_input=True`, Dolphin **waits for the bot's input
on every engine frame** (CLAUDE.md pitfall #8). So it's one bot decision per game
frame, just faster wall-clock. `PreventControllerReads` + `OverwriteInputs`
(EXI) supply those inputs; `SkipSounds` trims audio work. The faithfulness hinges
entirely on blocking input — with non-blocking, the looped frames would reuse a
stale input and the bot really would be hobbled (probably the true root of the
05-18 result, if that run wasn't blocking).

## Research: what slippi/melee modders do for FFW

- **Fizzi's "Fast forward codeset for bots"** is the canonical approach (the
  gecko codes are already in our `emulator_ffw/.../Sys/GameSettings/GALE01r2.ini`):
  `OverwriteInputs` (bot input injection), `LoopMainEngine/ForceContinueLoop` +
  `ForceStartLoop` (run engine N× per frame), `PreventControllerReads`,
  `SkipSounds`, `IncrementPadIndex`/`PadAlwaysUseMasterIndex`. Fizzi's benchmarks:
  an 8-min CPU-vs-CPU match in ~15 s (≈32×) — but that stacked a Dolphin
  **CPU-overclock** ("200–400% emulated CPU") on top of the gecko loop. Our
  `OverwriteInputs` is a *modified/longer* variant (`0000003A` vs Fizzi's stub
  `0000001D`) because ours injects real EXI inputs, not a stub.
- **slippi-ai** (`vladfi1/slippi-ai`, the main public Melee RL project) drives
  FFW with the identical flags — `slippi_ai/dolphin.py`:
  `console_kwargs.setdefault('use_exi_inputs', True)` /
  `setdefault('enable_ffw', True)` / `blocking_input=True`. It leaves
  `emulation_speed` at default (commented out) and gets the gecko loop alone
  (≈ our 2.9×). It runs `num_envs=8` via `AsyncBatchedEnvironmentMP` — N Dolphins,
  **batched** policy forward. It also ships `scripts/test_ffw.py` because *some
  character/stage combos crash under FFW* — worth a sweep before broad reliance.

Two speed knobs we're NOT using yet: (1) Dolphin CPU-overclock (`emulation_speed`)
stacked on the gecko loop; (2) batched multi-env (below). They compound.

## Parallel Dolphins — measured (independent `play.py` processes)

Champion vs CPU-7, FFW, headless. Per-instance fps computed from each instance's
own summed match-wall (robust to staggered finish).

| config | per-instance | aggregate | GPU util | CPU load (of 32) |
|---|---|---|---|---|
| N=1 | 176 fps (2.9×) | 176 fps (2.9×) | ~45% | ~2 |
| N=4 | **53 fps (0.88×)** | **210 fps (3.5×)** | **90%** | 8 |

All four N=4 instances returned identical 53 fps → clean, GPU-bound. 1→4
instances bought only 2.9→3.5× total. N=8 would split the GPU finer
(~26 fps each, aggregate ≈ flat), so it wasn't worth running to completion.

**Diagnosis: GPU, not CPU/Dolphin.** GPU pinned at 90% with 28/32 cores idle.
Each `play.py` runs its own model doing **batch-1** inference; batch-1 forwards
are launch-latency-bound (a 20M model is trivial FLOPs for a 4090), so N
independent processes saturate the GPU *scheduler* and serialize. (Note: an
earlier N=4 run that also read 53 fps was blamed on a concurrent canary stealing
cores — that was wrong; the bottleneck was the GPU the whole time.)

## The fix: batched multi-env rollout (the real throughput lever)

One process, N Dolphins, **one batched policy forward across all N envs**. A
single batch-16 forward is far cheaper than 16 batch-1 forwards, so the GPU stops
being the wall. This is slippi-ai's `AsyncBatchedEnvironment` design and why
`num_envs=8+` works there. It is also precisely the RLVR rollout harness (gather
transitions from many envs into one batched inference), so building it serves
both goals.

**Prototype: `tools/ffw_batch_bench.py`.** One process, N Dolphins, N worker
threads (libmelee's `console.step()` blocks in a GIL-releasing C socket read, so
steps overlap), synchronized each frame by two `threading.Barrier`s into a single
batched `model(mega)` forward, per-env decode via slicing `preds[k][i:i+1]` into
the existing `decode_and_press`. Measured (champion vs CPU-7, FFW headless,
1,200 frames/env):

| N | aggregate | per-env | vs independent play.py |
|---|---|---|---|
| 1 | 224 fps (3.7×) | 224 (3.73×) | — |
| 4 | **563 fps (9.4×)** | 141 (2.34×) | independent N=4 = 210 fps → **2.7×** |
| 8 | **577 fps (9.6×)** | 72 (1.20×) | peak |
| 12 | 537 fps (9.0×) | 45 (0.75×) | declining |
| 16 | 518 fps (8.6×) | 32 (0.54×) | declining |

Batched inference does exactly what it should: at N=4 GPU util is **37%** (vs 90%
for independent processes), so GPU is no longer the wall. Peak **9.6× realtime at
N=8** — 2.7× the independent-process peak (3.5×), 3.3× the single `play.py` (2.9×).

**But the prototype plateaus at N=8 then declines — that ceiling is a GIL wall,
not GPU/Dolphin.** During the N=4 timing, GPU=37% *and* CPU load≈0.93 (≈one core
busy of 32). Neither resource is saturated; a single core is. The per-round work
that's GIL-serialized — building each env's `(1,seq_len,F)` batch (`torch.cat`) and
`decode_and_press` (Python sampling + controller press) for all N envs, plus the
lockstep barrier that paces every round to the slowest env — grows linearly with
N on one thread. That caps aggregate around ~577 fps regardless of how many
Dolphins/GPU headroom exist.

**Multiprocess async batched — the machine ceiling (`tools/ffw_batch_mp.py`).**
Each env in its own process (no shared GIL); the central process owns the ONLY
model copy (on GPU), keeps each env's rolling window, and does one batched forward
across whatever envs are ready each cycle (async — never blocks on the slowest).
Wire payloads are tiny: env→central sends `build_frame`'s 16 small arrays,
central→env sends only the 4 head logits at the last timestep. `spawn` start
method so central inits CUDA before the (CUDA-free) env procs launch. Decode
(`decode_and_press`) runs in each env process, so it's genuinely parallel.

Two implementation notes that mattered: (1) the central window must use a
**pre-allocated rolling buffer** (one vectorized shift per key, then `stack`
across active envs) — the naive `torch.cat` of 180 tiny tensors ×16 keys per env
per frame made N=2 *slower* than threaded (74 fps/env); the fix took it to 226.
(2) all per-env CPU work (build_frame, decode, sampling) is off the central
thread, so central only does stack+forward+scatter.

Measured (champion vs CPU-7, FFW headless, 12 s/point):

| N | aggregate | ×realtime | per-env |
|---|---|---|---|
| 4 | 859 fps | 14.3× | 215 (3.58×) |
| 8 | 1,410 fps | 23.5× | 176 (2.94×) |
| 16 | 1,743 fps | 29.1× | 109 (1.82×) |
| 24 | 1,794 fps | 29.9× | 75 |
| 32 | **1,812 fps** | **30.2×** | 57 |

**~30× realtime on one box, single 4090.** The plateau past N≈16 (29→30×) is CPU
saturation: 32 Dolphins + 32 env procs + central oversubscribe 32 cores, so
per-env craters (215→57) while aggregate barely moves. The **knee is N≈8–16** —
N=8 gives 23.5× with per-env still at the full single-instance rate (176 fps),
N=16 gives 29× at 109 fps/env. For RL, pick N by whether you want max aggregate
(N≈16–24) or healthy per-env latency (N≈8).

**Throughput ladder on this box (4090 / 32-core), single GPU, no CPU-overclock:**

| architecture | peak | bottleneck |
|---|---|---|
| single `play.py` | 2.9× | — |
| N independent `play.py` procs | 3.5× | GPU (N× batch-1 forwards) |
| threaded batched (`ffw_batch_bench.py`) | 9.6× | GIL (serial per-round Python) |
| **multiprocess async batched (`ffw_batch_mp.py`)** | **30.2×** | **CPU (Dolphins + env procs)** |

That's ~10× the single-instance baseline. Fizzi's ~32× came from stacking a
Dolphin CPU-overclock on the gecko loop; we reach ~30× from **parallelism alone**.
On a CPU-saturated box the overclock knob won't add aggregate — the remaining
lever is more/bigger boxes (this harness is the per-box unit) or a lighter env
process. `tools/ffw_batch_mp.py` is the seed of the RLVR rollout collector.

## Bottom line for the RLVR plan

FFW is unparked and faithful, and one box does **~30× realtime** (~1,800 fps,
`tools/ffw_batch_mp.py`, N≈16–32) — the machine ceiling, CPU-bound. That
transforms RLVR feasibility: a ~17 h realtime run becomes hours, and the harness
IS the rollout collector. Next steps, in order: (1) wire `ffw_batch_mp.py` into
the RLVR loop as the rollout collector (swap CPU opponent → frozen-BC self-play
per [[feedback_rlvr_training_opponent]]; return transitions, not just frame
counts); (2) run slippi-ai's char/stage FFW stability sweep
(`scripts/test_ffw.py` analogue) before depending on it broadly — some combos
crash under FFW; (3) the Dolphin CPU-overclock knob (`emulation_speed`, Fizzi's
route to 32×) won't help on a CPU-saturated box, so scale is more boxes, not more
envs per box.
