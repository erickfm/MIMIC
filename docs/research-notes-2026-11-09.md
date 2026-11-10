# 2026-11-09 — where S sim-RL time goes (timing only)

`rl-v2` at S (`fox-S-bc2-hf51k-s1.rl2-s1`, 4090): ~62 s per update = ~30 s rollout + ~32 s update
(4 PPO epochs ~29.5 s); 100 updates ~104 min. Nothing below has been tested for strength.

## Rollout: one lockstep step of 64 envs (profiled outside sim_rl, 1,000 steps after warm-up)

| part | ms/step |
|---|---|
| learner forward + sample (CUDA graph, not compiled) | 4.64 |
| opponent forward + sample (CUDA graph, compiled) | 2.38 |
| sim, 16 worker processes | 4.18 |
| pads to the wire | 0.21 |
| total | 11.43 (5,600 env-frames/s) |

The parts run one after another. In sim_rl the rollout runs ~4,350 env-frames/s (~14.7 ms/step);
the ~3 ms difference is presumably sim_rl's per-slot Python return bookkeeping (not measured directly).
Only 1 frame in 8 is trained on (`--sample-every 8`).

- Compiling the learner like the opponent: 4.64 → 2.54 ms, step 11.43 → 9.35 ms (6,850 env-frames/s).
  The compiled forward follows in-place weight updates (trunk features vs eager after a 1e-2 weight
  perturbation: max diff 0.035 at bf16, outputs moved 0.29). Not wired into sim_rl.
- Untried: overlap the sim (CPU) with the model (GPU) by stepping two env groups alternately.

## Update: `--ppo-amp` at S (3 updates each, seed 1, same args as rl-v2)

| | update_s | epochs_s | rollout_s |
|---|---|---|---|
| fp32 (rl-v2) | 30.8 | 28.3 | 28.4–29.8 |
| bf16 (`--ppo-amp`) | 21.8 | 19.9 | 28.7–29.7 |

Anchor KL and clip fraction in the same range over these 3 updates (says nothing about strength).
At M (2026-11-04) the same flag took the update 83 → 52 s.

Both together would be roughly 59 → 46 s per update (estimated by adding the two measured savings).
Before using either: one 100-update run with them, h2h vs `fox-S-bc2-hf51k-s1.rl2-s1`.

## Test run: `fox-S-bc2-hf51k-s1.rl2-fast-s1` (rl-v2 + `--ppo-amp` + `--compile-learner`, same parent and seed)

- Wall time: 82.0 min for 100 updates vs 104.2 min for `fox-S-bc2-hf51k-s1.rl2-s1` (−21%); ~49 s per update.
- vs `fox-S-bc2-hf51k-s1.rl2-s1` (the same recipe without the two flags): **96–96 (50.0%, 95% CI 43–57%)**,
  stock margin −0.01. vs its BC parent: 164–28 (85.4%; the original run: 150–42, 78.1%).
- Anchor KL per update varies a lot: 0.016 vs 0.009 at update 64, 0.011 vs 0.023 at update 100.
- One run, one 192-game match: no loss of strength detected, and it can't rule out a few points either way.

## Is sim RL noisy or untuned? (analysis of existing runs, no new training)

**Match noise vs training noise.** The 13 Oct 27 matches vs SF7u100 from recipes within noise of each other
(1-epoch 4x lr excluded): mean 48.0%, sd 4.8 pts; a 192-game match alone gives 3.6. Excess sd 3.1 pts
(chi2 21.1 on 12 df, p 0.049), and that excess also holds the real recipe differences. The same checkpoint at
h2h seeds 41 vs 42 moved −3.1, +6.2, −2.6, +3.1 (expected sd of a difference 5.1). Today: same parent and seed,
two numerics, 96–96. So most of the spread in single results is the 192-game measurement; run-to-run training
variation is at most ~3 pts sd.

**Training curves, all 29 rl2 runs** (rollout win rate vs the frozen BC opponent, 20-update bins, ~180–360
games per bin per run):

| updates | 1–20 | 21–40 | 41–60 | 61–80 | 81–100 |
|---|---|---|---|---|---|
| mean over runs | 56.0% | 63.9% | 68.9% | 72.5% | 74.8% |

Gains per bin shrink (+7.9, +5.0, +3.6, +2.3) but have not stopped: 20 of 29 runs were higher in 81–100 than
61–80. Anchor KL keeps growing (median ~0.01 at u20, ~0.02 at u50, ~0.03 at u100). SF7 (rl-v1) plateaued
by ~u70–100 (u150 94–98, u200 96–96 vs u100), but rl-v2 policies move ~2x further from BC, so that may not carry
over.

**Clip fraction** averages 0–0.0003 per run (max 0.008): PPO's ±0.2 ratio clip essentially never binds; the
step size is set by lr alone.

**Tested so far:** decisions per update (4k–64k at 30 updates), number of updates (30–200, rl-v1), critic, KL
terms, 1 epoch at 4x lr (worse), 50 updates at 2x lr (probably worse), lr 3e-5 with weak KL (drifts worse).
**Never tested:** discount (γ 0.99 per frame = 1.15 s half-life; Phillip uses ~8 s), return horizon (5 s),
--sample-every (8), lr at 100 updates or an lr schedule, more than 4 epochs, a per-state baseline other than the
online critic, opponents other than the frozen BC.
