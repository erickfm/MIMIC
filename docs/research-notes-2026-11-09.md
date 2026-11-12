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

## Research queue results (finished Tue 11/10 4:22 AM)

Fox S, parent fox-S-bc2-hf51k-s1, training seed 1 unless noted. "vs refs" pools three 192-game h2h:
vs rl2-fast-s1 (u100) at seeds 41 and 42, and vs rl2-s1 at 41 (576 games).

| variant | vs refs | vs BC parent |
|---|---|---|
| rl2-fast continued to u200 | 381–195 (66.1%) | — |
| rl2-fast u150 (vs u100, seed 41 only) | 115–77 (59.9%) | — |
| lr 2e-5 | 330–246 (57.3%) | 161–31 |
| ridge baseline | 313–263 (54.3%) | 155–37 |
| sample-every 2 | 307–269 (53.3%) | 150–42 |
| plain rl-v3, training seed 2 | 289–287 (50.2%) | 143–49 |
| gamma 0.997, horizon 600 | 258–318 (44.8%) | 133–59 |

- More updates is the clear result: u200 beats u100 by ~16 points on every reference; u150 sits between.
  This differs from SF7 (rl-v1), where u150/u200 tied u100.
- The seed-2 rerun scores 50.2%, so the reference set is calibrated. lr 2e-5 (+7) is the only other
  variant past ~3 points; ridge and sample-every 2 are within ~2 sd of noise; gamma 0.997 is worse.
- lr 2e-5 at 100 updates may be doing the same thing as more updates (bigger total policy movement).

## Follow-up queue (Tue 11/10, 6:22 AM – 2:13 PM)

Second training seeds for the two results past noise, a stacking test, and u300. Same references and
h2h seeds as above (192 games each).

| run | opponent | result |
|---|---|---|
| rl-v3 seed 2, u200 | refs (u100 ×2 seeds, rl2-s1) | 331–245 (57.5%) |
| rl-v3 seed 2, u200 | seed 1 u200 | 80–112 (41.7%) |
| lr 2e-5 seed 2 (u100) | refs | 347–229 (60.2%) |
| lr 2e-5 seed 1, u200 | seed 1 u200 (lr 1e-5), h2h seeds 41/42 | 246–138 (64.1%) |
| lr 2e-5 seed 1, u200 | seed 1 u100 | 151–41 (78.6%) |
| rl-v3 seed 1, u300 | seed 1 u200, h2h seeds 41/42 | 228–156 (59.4%) |

- More updates replicates on seed 2, smaller: u100 → u200 moves seed 2 from 50.2% to 57.5% vs refs (seed 1: +16).
- lr 2e-5 replicates: 57.3% (seed 1) and 60.2% (seed 2); pooled 677–475 (58.8%).
- The two stack: lr 2e-5 at u200 beats lr 1e-5 at u200 64.1%.
- u300 still beats u200 (59.4%, 126–66 and 102–90 at the two h2h seeds). No plateau yet at lr 1e-5.
- Reading: at rl-v2/v3 settings the policy is undertrained; total step (lr × updates) is the lever.
  rl-v3 at 100 updates is 82 min, so u200 is ~2.7 h and u300 ~4 h at Fox S.
- Not yet tested: lr 2e-5 past u200, lr 4e-5, and whether the gain holds against fox-master / in Dolphin.

## Queue 3: where do gains stop? (Tue 11/10 2:50 PM – Wed 11/11 3:39 AM)

Fox S, seed 1, rl-v3 except lr and updates. Each endpoint vs fixed opponents: u100 = rl2-fast-s1 (lr 1e-5, u100)
at h2h seeds 41+42 (384 games), fox-master (fox-M-bc2-mlr01-s1.rl2-s1, M, 20M params) at 41+42, BC parent at 41.

| lr | updates | vs u100 | vs fox-master | vs BC |
|---|---|---|---|---|
| 1e-5 | 200 | 253–131 (65.9%) | 124–260 (32.3%) | 170–22 |
| 1e-5 | 300 | 276–108 (71.9%) | 151–233 (39.3%) | 168–24 |
| 1e-5 | 400 | 315–69 (82.0%) | 175–209 (45.6%) | 180–12 |
| 2e-5 | 100 | 214–170 (55.7%) | 109–275 (28.4%) | 161–31 |
| 2e-5 | 200 | 291–93 (75.8%) | 171–213 (44.5%) | 173–19 |
| 2e-5 | 300 | 302–82 (78.6%) | 175–209 (45.6%) | 177–15 |
| 2e-5 | 400 | 307–77 (79.9%) | 193–191 (50.3%) | 177–15 |
| 4e-5 | 100 | 245–139 (63.8%) | 122–262 (31.8%) | 157–35 |
| 4e-5 | 200 | 290–94 (75.5%) | 175–209 (45.6%) | 175–17 |

Direct matches:
- lr 2e-5: u300 vs u200 114–78 (59.4%); u400 vs u200 218–166 (56.8%, 2 seeds).
- Equal total step: lr 1e-5 u400 vs lr 2e-5 u200 173–211 (45.1%, 2 seeds); lr 1e-5 u400 vs u300 106–86.
- lr 4e-5 vs 2e-5: at u200 195–189 (50.8%); at u100 170–214 (44.3%).

Anchor KL to BC (per learner frame, logged each update):
lr 1e-5 0.011 (u100) → 0.025 (u200) → 0.024 (u300) → 0.041 (u400);
lr 2e-5 0.042 → 0.078 → 0.090 → 0.072; lr 4e-5 0.087 (u100) → 0.116 (u200). Clip fraction stays ≤ 0.003.

Reading:
- Gains slow but haven't stopped. At lr 2e-5, u200 → u400 adds ~6 points vs fox-master and wins 56.8% directly;
  vs u100 it has nearly saturated (75.8 → 79.9%).
- lr 2e-5 at u400 ties fox-master (193–191): an S model (5.4M) + 400 updates matching the M production model.
- lr 4e-5 buys nothing over 2e-5 at 200 updates. lr 2e-5 at u200 ≈ lr 1e-5 at u400 vs fixed opponents
  and slightly better head-to-head, at half the updates: lr and updates mostly trade off.
- Inferred, not tested: lr 2e-5's anchor KL levels off at ~0.08 after u200, so the β 0.01 anchor may be what
  slows the gains. Testing that needs a smaller β.

## Assessment of queue 3 (own review + an independent review of the raw results, Wed 11/11)

Corrections to the readings above:
- "lr 2e-5 at u400 ties fox-master" is not established. 193–191 has a 95% CI of 45–55%, and it is the best of
  9 checkpoints played against fox-master, so it is likely inflated a few points. The four u300/u400 checkpoints
  pool to 46.7% vs fox-master. It also isn't a size comparison: fox-master had 100 RL updates at lr 1e-5; this
  S model had 400 at 2e-5, and more total compute (S BC 116 min + ~360 min RL vs M 189 + 219 min).
- "lr and updates trade off" is weak. It holds at 2e-5×200 vs 1e-5×400 (54.9% direct, both ~45% vs fox-master)
  and fails at 4e-5×100 (31.8% vs fox-master, against 44.5% for 2e-5×200). Anchor KL rises with lr
  (~0.04 / 0.08 / 0.11), so β 0.01 is a different constraint at each lr.
- "Gains slowing" holds for lr 2e-5 only (u400 vs u200 56.8%, no more than u300 vs u200; training return flat
  over u301–400). lr 1e-5 is still rising vs fox-master (32 → 39 → 46%). Win rate vs u100 is near its ceiling (~80%).

What holds: more updates help up to u200 (both seeds); lr 2e-5 beats 1e-5 at u100 (both seeds, 55.7% and 60.2%
vs u100 vs 50.2% for 1e-5 seed 2); lr 4e-5 = 2e-5 at u200.

Measurement: across 48 checkpoint pairs played at both h2h seeds, the 41−42 difference has sd 4.2 pts (binomial
expects 5.1), so h2h seed noise is binomial and the excess spread is training seed: lr 1e-5 seed 1 vs seed 2 at u100
beat BC 85.4% vs 74.5%. Everything past u200 is seed 1 only. All references share one lineage (the u100 reference
is a seed-1 lr 1e-5 checkpoint); fox-master is the only outside opponent. Most of the reward gain is damage
(learner damage 0.6 → 3.1–3.6 per update); stocks change hands more in both directions.

## Queue 4: second seeds, round-robin, M at lr 2e-5 (Wed 11/11 3:43 PM – Thu 11/12 9:12 AM)

Round-robin: 13 models, every pair 96 games at h2h seed 43 (78 matches, 7,488 games), fitted with Bradley-Terry
(`tools/rl_ratings.py rr`). Rating = log-odds vs fox-master (fox-M-bc2-mlr01-s1.rl2-s1: M, lr 1e-5, u100).
Fit chi2 71.1 on 66 df: the results are consistent with one transitive scale. Using all matches between these
models (122 matches, 15,936 games; `tools/rl_ratings.py all`) gives the same order (± ≈ 0.05).

| model | rating (rr) | ≈ win % vs fox-master |
|---|---|---|
| M, lr 2e-5, u200 | +1.09 ± 0.09 | 75 |
| S, lr 2e-5, seed 2, u400 | +0.05 ± 0.08 | 51 |
| fox-master | 0 | 50 |
| S, lr 2e-5, seed 1, u400 | −0.03 | 49 |
| S, lr 4e-5, seed 1, u200 | −0.06 | 48 |
| S, lr 2e-5, seed 2, u200 | −0.07 | 48 |
| S, lr 1e-5, seed 2, u400 | −0.12 | 47 |
| S, lr 2e-5, seed 1, u200 | −0.20 | 45 |
| S, lr 1e-5, seed 1, u400 | −0.22 | 44 |
| S, lr 1e-5, seed 1, u200 | −0.80 | 31 |
| S, lr 2e-5, seed 1, u100 | −0.87 | 30 |
| S, lr 1e-5, seed 2, u200 | −0.96 | 28 |
| S, lr 1e-5, seed 1, u100 | −1.30 | 21 |

Direct matches vs fox-master (h2h seeds 41+42, 384 games):
- S lr 2e-5 u400: seed 1 193–191, seed 2 207–177; pooled 400–368 (52.1%).
- S lr 1e-5 u400: seed 1 175–209, seed 2 181–203 (46.4% pooled).
- M lr 2e-5: u100 170–214 (44.3%), u200 279–105 (72.7%). M vs its BC parent 179–13.
- M lr 2e-5 × 200 updates took 299 min with the rl-v3 speed flags (they work at M; fox-master's 100 rl-v2 updates took 219 min).
- M anchor KL: 0.047 (u100), 0.089 (u200), same range as S at lr 2e-5.

Reading (now on two training seeds at S, one at M):
- Training seeds now agree within ~0.1–0.25 logits at u200 and u400.
- More updates help at both lrs; at lr 2e-5 u200 → u400 is +0.15 logits (both seeds), at lr 1e-5 +0.6–0.8.
- lr 2e-5 beats 1e-5 clearly at u200 (both seeds, ~0.7 logits), slightly at u400 (~0.2).
- lr 2e-5 × 200 ≈ lr 1e-5 × 400 (−0.14 vs −0.17 average): the trade-off holds on two seeds. lr 4e-5 × 200 ≈ lr 2e-5 × 200.
- S levels off at about fox-master's strength (~50%).
- The large result: M with lr 2e-5 × 200 updates beats fox-master 72.7% and rates ~1.1 logits above every S model.
  fox-master was under-trained by RL, not limited by size. One M seed; M at u100 lost to fox-master 44%.
