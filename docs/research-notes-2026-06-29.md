# 2026-06-29 — Inference-faithfulness canaries (a verification system)

## Problem

We need to know, cheaply and reliably, whether the live inference pipeline is
**faithful** — i.e., the deployed model is actually outputting what it learned,
with no silent timing / decode / dropped-input / normalization bug. Val loss
doesn't cover this (it's offline; says nothing about the live Dolphin path), and
"watch a game" is subjective. The motivating scare: a replay looked slow and we
couldn't tell if the model was performing correctly.

## Approach: behavioral canaries

A **canary** = a cheap, dense, replay-only statistic of the bot's behavior that
has a **known reference value** from the training corpus. Faithful inference →
the canary matches the reference; a broken pipeline → it deviates. "Dense" so a
few games give a tight estimate; "known reference" so deviation is meaningful.

All canaries isolate the bot by **player type**: the model drives a virtual
controller and records as `type=0` (human) in the `.slp`; a CPU records `type=1`.
So model-vs-CPU runs measure only the bot. (All-human corpus is unaffected.)

Validated against a **deliberately-broken control**: `tools/play.py --drop-prob P`
overwrites the bot's input with neutral on a fraction `P` of frames (input "lost
in transit"; the model's internal state stays unaware — like a real frame drop).
`P=1.0` = fully inert bot = known-broken pipeline.

## The suite (three complementary canaries)

| canary | tool | healthy | broken (100% drop) | catches |
|---|---|---|---|---|
| **L-cancel rate** | `lcancel_analysis.py` | ~88% | (n→0 aerials) | systematic timing / frame-alignment |
| **center-stick %** | `ctrl_canary.py` | ~34% | 100% | dropped / idle inputs (sharp, ~linear) |
| **stick-dist JS** | `ctrl_canary.py` | ~0.01–0.05 | 0.45 | general "wrong but active" corruption |
| (action-state JS) | `state_canary.py` | ~0.02–0.05 | 0.60 | general corruption (coarser) |

They're **complementary by failure mode** (validated, not assumed):
- **L-cancel rate** caught the timing case but is *blind to random drops* — the
  model presses the trigger on multiple frames in the 7-frame window, so dropping
  a random 20% rarely kills all the presses (rate flat: 88→86→87% at 0/5/20%
  drop). Good *timing-bug* sentinel, not a *drop* sentinel.
- **center-stick %** is the drop sentinel: a dropped frame *is* a center stick,
  so it climbs near-linearly: 34% (clean) → 42% (5%) → 51% (20%) → 100% (broken).
- **stick / action-state JS** are the general distributional check — they'd catch
  a decode bug that presses the *wrong* thing (which center-stick % would miss).

## Thresholds (from the data)

- L-cancel rate: healthy ≈ corpus **90.4%**; the BC model sits ~2 pts low at
  **~88%** (normal BC under-fit, per-move shape matches — NAIR/DAIR ~92%,
  BAIR/UAIR ~84%). Far below 88% → timing/decode regression.
- center-stick %: healthy **~34%**; > ~45% → suspect dropped/idle inputs.
- stick-JS: healthy **< ~0.05**; > ~0.15–0.2 → suspect corruption. (0.45 = broken.)

## The L-cancel objective, properly defined (bonus from building the sentinel)

The engine's binary `post.l_cancel` (1=success/2=miss) is *not* faithful to lag
suffered — ~20–27% of "misses" cost nothing (slid off a ledge, or got hit before
the lag exceeded the cancelled minimum). The right signal, validated on 35,717
real Fox landings, is **realized avoidable lag**:

    cancelled_min = {NAIR:7, FAIR:11, BAIR:10, UAIR:9, DAIR:9}  # frames
    avoidable_lag = max(0, realized_landing_lag − cancelled_min[move])

Apply uniformly across all exits (a got-hit landing still counts iff it ate lag
past the cancelled minimum). Use this, never the binary flag or the old
offset=−4 press proxy (which diverged from the truth and was never validated).

## Tools

- `tools/lcancel_analysis.py <dir>` — per-move L-cancel rate + avoidable-lag breakdown.
- `tools/ctrl_canary.py [n]` — controller canary: stick-dist JS + center-stick %.
- `tools/state_canary.py [n]` — action-state distribution JS.
- `tools/compare_lcancel.py [n]` — model-vs-corpus L-cancel rate with CIs.
- `tools/play.py --drop-prob P` — fault injection for validating any canary.

## Next: this is exactly the tool FFW debugging needed

FFW (`emulator_ffw/`, `--use-exi-inputs --enable-ffw`) was parked 2026-05-18 as
**unfaithful** — matches run ~4× short (2430 vs 9687 frames), bots lose stocks
~4× faster, "almost certainly the EXI input path mistiming controller inputs."
But that root cause was never *pinned down* — it was inferred from match length,
because win-rate and frame-delta histograms were both false positives (win-rate
survives symmetric degradation; histograms check stepping, not input fidelity).

The canaries **directly measure input timing and fidelity** — exactly what FFW's
diagnosis lacked. Running the bot under FFW + the canaries should: (a) confirm the
input mistiming (L-cancel rate craters if timing is broken), (b) help localize it
(L-cancel vs center-stick vs distribution distinguish frame-offset from
dropped/garbled inputs), and (c) give a *fast* FFW-fidelity pass/fail instead of
slow match-length comparisons. FFW (if fixable) is the throughput unlock for the
whole RLVR plan — worth re-examining now that we can verify it.
