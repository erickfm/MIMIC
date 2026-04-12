# Research Notes — 2026-04-12: Multi-Character v2 Training Results

## Overview

Trained four character-specific models on v2 shards (next-frame targets) with
relpos position encoding and `--self-inputs`. All four play actively in
Dolphin — no stuck modes — confirming the gamestate leak fix from 04-11
generalizes across characters.

## Training Results

All runs: `--model hal --reaction-delay 0 --self-inputs`, batch 512, 32K steps
unless noted.

| Character | Games | Steps | Val btn F1 | Val main F1 | Val loss | Notes |
|-----------|-------|-------|-----------|------------|----------|-------|
| Fox (v2) | 17,319 | 10K | 59.4% | 15.5% | 2.27 | rd=0, no self-inputs (legacy) |
| Falco (small) | 1,748 | 32K | 82.1% | 45.0% | 1.27 | self-inputs |
| Falco (full) | 9,110 | 32K | 88.2% | 58.5% | 0.68 | self-inputs |
| CptFalcon | 9,404 | 32K | 89.9% | 52.2% | 0.71 | self-inputs |
| Luigi (32K) | 1,951 | 32K | 84.6% | 52.4% | 1.02 | self-inputs |
| **Luigi (long)** | 1,951 | 5,242 (best) | — | — | — | overfits past ~5K with 1.9K games |

The Fox v2 run was an early test without `--self-inputs`. Adding `--self-inputs`
to the other three character runs (Falco, CptFalcon, Luigi) gave dramatically
better convergence — Falco hit 88% btn F1 in 32K steps vs Fox's 59%.

## Bot v Bot Results (sane Dolphin gameplay)

All ditto matches on Final Destination, temperature=1.0, both sides using the
same checkpoint:

| Match | Result | Duration | Top P1 actions |
|-------|--------|----------|----------------|
| Falco ditto | P1 **3-0** P2 | 106s | jumps, dairs, aerials |
| CptFalcon ditto | P2 **3-0** P1 | 169s | dashes, aerials, jumps |
| Luigi (32K) ditto | P1 **1-0** P2 | 295s | aerials, jumps |
| Luigi (long, step 5242) | P2 **2-0** P1 | 321s | active, full game |

All games go to completion with diverse actions. No stalling. Stochasticity
determines which side wins.

## v2 vs old shards: button F1 inflation

The old shards (target[i] = button[i]) inflate button F1 because the model
can copy `self_action` → button. v2 shards (target[i] = button[i+1]) require
predicting from pre-action game state, which is harder but matches inference.

Old shard 7-class run (hal-7class-rd0): val btn F1 90.2%, val loss 0.74 — but
gameplay was bistable (sometimes stuck in WAIT/SQUAT_WAIT attractors).
v2 shard runs: val btn F1 84-90%, val loss 0.7-1.3 — no stuck modes.

## Luigi long-run overfitting

Luigi has only 1,951 games. Training 262K steps (134M samples = 68 epochs per
game) gives near-perfect train metrics (btn F1 92.7%, main F1 74.7% at step
~111K) but val plateaus much earlier. The auto-saved `_best.pt` was at
**step 5242** — exactly where overfitting starts.

Lesson: with small per-character datasets (<5K games), early stopping matters.
The checkpoint saver watches val F1 and saves the best, so `_best.pt` is the
right one to use regardless of how long training runs.

## Wavedash observation

User noted the Luigi model never wavedashes in inference. Wavedash requires:
- Jump button press
- Within ~3 frames, analog shoulder press (L/R > threshold)
- Stick at down-left or down-right simultaneously

Three things to investigate:
1. Are wavedashes present in the Luigi training data at all? Some Luigi mains
   never wavedash. Could detect by looking for KNEE_BEND → LANDING_FALL_SPECIAL
   transitions in self_action.
2. The button head and shoulder head are independent — the model might not
   have learned to coordinate them on the right frames.
3. Stick + button + shoulder coordination across 3 consecutive frames is hard
   to express with the current head structure.

Not investigated yet.

## Best Checkpoints (zipped)

Bundled the best checkpoint per character into `mimic_best_checkpoints.zip`
(828MB, gitignored):

- `hal-7class-v2-long_best.pt` — Fox (Eric's reference reproduction)
- `falco-7class-v2-full_best.pt` — Falco (10K games)
- `cptfalcon-7class-v2_best.pt` — CptFalcon
- `luigi-7class-v2-long_best.pt` — Luigi (early-stop best from long run)

Each .pt contains the full ModelConfig in `ckpt['config']`, plus `global_step`,
`norm_stats`, `stick_centers`, `shoulder_centers`, `model_state_dict`.
