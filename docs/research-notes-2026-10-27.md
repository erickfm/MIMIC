# 2026-10-27 — Simplifying sim RL: a greedy ablation, and how noisy one run is

Night of 10-26 through the afternoon of 10-27. Goal: the fewest moving parts in `tools/sim_rl.py` dense mode that are
at least as strong as the SF7 recipe and faster. **All results are from one training seed** (sim_rl `--seed 0`);
several recipes were trained twice with that seed, and the runs differ (GPU / rollout nondeterminism), so those
pairs are two samples of the recipe, not a replication with a second seed. Nothing promoted.

**Reproducing SF7:** commit `c2df04b` is the last one whose sim_rl dense defaults are SF7's recipe (critic, both KL
terms); its first update reproduces SF7's to every logged digit. Commit `1a91438` has the first pass's recipe (no
critic, anchor only). The current defaults are the second pass's (critic, per-decision KL only).

## Setup

- Start: `checkpoints/smallfox/base_full56k` (S, full attention, cond-heads, 56k steps). Opponent: the frozen base.
  Reward `stock_delta:1,damage_delta:1`. Local RTX 4090, one GPU job at a time.
- SF7 recipe: 16,384 decisions × 100 updates, critic, two KL terms (per-decision 0.03 on the sampled actions, global
  anchor 0.01 on 512 random learner frames), 4 PPO epochs, lr 1e-5. Reference model: SF7u100
  (`checkpoints/smallfox/SF7-full56k_16k_u100/eval_final`).
- Greedy ablation, one change per run, stacked on the last kept recipe: (1) no critic, (2) 1 PPO epoch at lr 4e-5,
  (3) one KL term (anchor only; if that loses, per-decision only), (4) 50 updates at 2× lr, then 33 at 3× if 4a
  passes. Keep rule: the final checkpoint vs SF7u100 in a 192-game bf16 sim h2h at seed 41 has win rate ≥ 47%
  (draws half) and one-sided exact binomial p > 0.05 against 50%, and training wall-clock is lower than the last
  kept recipe's. Final check: the stacked recipe vs SF7u100 at h2h seed 42.
- Baselines: without the critic the advantage is (return − batch mean) / batch std over the update's 16,384
  decisions (`rlvr/sim/ppo.py:_compute_advantages`); with it, (return − V(s)) z-scored.
- Scripts and logs: `checkpoints/smallfox/ablate/` (`step.sh`, `score.py`, `retime.sh`, `results.txt`).

## Results

Every match is 192 games vs SF7u100; p is one-sided against 50%.

**First pass** (from 12:00 AM):

| Step | Change | Seed 41 | Train | vs base | Kept |
|---|---|---|---|---|---|
| 1 | no critic | 94–98 (49.0%, p 0.414) | 107.3 min | 143–49 | yes |
| 2 | + 1 PPO epoch, lr 4e-5 | 82–110 (42.7%, p 0.026) | 69.8 | 139–53 | no |
| 3 | + anchor only (run 1) | 100–92 (52.1%, p 0.742) | 115.7* | 148–44 | — |
| 3 | same recipe, run 2 | 106–86 (55.2%, p 0.935) | 103.5 | 153–39 | yes |
| 4a | + 50 updates, lr 2e-5 | 83–109 (43.2%, p 0.035) | 51.5 | 137–55 | no |

Final check at seed 42: run 2 94–98, run 1 106–86. Both passed, and this recipe was committed (`1a91438`).

\*Run 1 overlapped external CPU load (a browser and a headless Chromium) from about update 30.

**Second pass.** The no-critic step was retrained (A1r) after stating its baseline before the run, and lost 78–114
(40.6%, p 0.006; 107.4 min). Pooled with run 1 it is 172–212 (44.8%, p ≈ 0.02), so step 1 was reverted and steps 2–4
were re-tested on SF7 (critic kept):

| Step | Change on SF7 | Seed 41 | Train | vs base | Kept |
|---|---|---|---|---|---|
| 2 | 1 PPO epoch, lr 4e-5 | 71–121 (37.0%, p < 0.001) | 74.4 min | 124–68 | no |
| 3 | anchor only | 86–105–1 (45.1%, p 0.097) | 107.5 | 132–60 | no |
| 3 reverse | per-decision KL only | 100–92 (52.1%, p 0.742) | 107.3 | 148–44 | yes |
| 4a | + 50 updates, lr 2e-5 | 89–103 (46.4%, p 0.174) | 53.6 | 139–53 | no |
| 4b | 33 updates, lr 3e-5 | not run: 4a failed | | | |

**Final check, seed 42: SF7 minus the anchor 80–112 (41.7%, one-sided p 0.013). Failed.**

**SF7 retrained** (same recipe, code and seed; update 1 identical to the original): 146–46 vs base (original
140–52); vs SF7u100 88–104 (45.8%, p 0.139) at seed 41 and 93–99 (48.4%, p 0.359) at seed 42; 111.3 min (original
111.6).

## Pooled by recipe, against the SF7 retrain as the yardstick

| Recipe | vs SF7u100, pooled | Train |
|---|---|---|
| SF7 retrain | 181–203 (47.1%), 1 run × 2 seeds | 111.3 min |
| No critic, anchor only | 406–362 (52.9%), 2 runs × 2 seeds | 103.5 |
| Critic, per-decision only (current default) | 180–204 (46.9%), 1 run × 2 seeds | 107.3 |
| Critic, anchor only | 86–105–1 (45.1%), 1 match | 107.5 |
| No critic, both KL | 172–212 (44.8%), 2 runs, seed 41 | 107.3–107.4 |
| 50 updates at 2× lr | 172–212 (44.8%), 2 runs (critic and not) | 51.5–53.6 |
| 1 epoch at 4× lr | 153–231 (39.8%), 2 runs (critic and not) | 69.8–74.4 |

## Notes

- **Noise.** A retrain of SF7's own recipe scores ~47% against SF7u100, so SF7u100 is a slightly lucky draw and the
  keep rule's 47% bar sits near the recipe's own median: a change exactly as good as SF7 fails the strength test
  about half the time. One checkpoint at two h2h seeds moved up to 10 points (52.1% vs 41.7%); two runs of one
  recipe moved up to 8 (49.0% vs 40.6%). A 192-game match's own binomial noise is ±3.6 points (1 sd). Greedy single-run
  decisions at this noise level are close to coin flips for effects of a few points, and the two passes ended on
  different recipes for that reason.
- **Wall-clock reproduces** (SF7 111.6 vs 111.3 min): dropping any one of the critic, the anchor or the per-decision
  KL saves ~4 min (~107.4); dropping the critic and the per-decision KL together saves ~8 (103.5). Without the
  per-decision KL the frozen reference's scoring pass over the 16,384 decisions is skipped; without the critic its
  forward and 20-epoch fit are skipped.
- **What held up in both passes.** 1 PPO epoch at 4× lr is clearly worse (both runs ~37–43%): same per-update
  lr × steps, but the later epochs re-evaluate the gradient at the moved parameters, and clip_frac is ≤ 0.0004 with
  4 epochs, so the gain is not from clipping. 50 updates at 2× lr is probably worse (both runs 43–46%), consistent
  with 10-26's finding that update count drives the gain.
- **Critic and KL terms** change strength by less than the noise. The one recipe with an edge is no critic + anchor
  only: ~6 points above the SF7 retrain over four matches from two runs (p ≈ 0.06 on game noise alone, weaker with
  run noise). Its policies also moved furthest from BC (anchor KL 0.019 and 0.024 at u100 vs 0.006–0.013 for the
  others).
- **Cheap predictors.** Over the 10 ablation runs, the vs-SF7u100 win rate correlates 0.75 with the rollout record of
  the last 10 updates (free) and 0.74 with the 192-game vs-base screen.
- **Logging.** The `anchor_kl` field and `--global-kl-beta` / `--anchor-frames` are gone with the anchor;
  `vr_rl_run.sh`'s summary prints `kl`.
- Open: the default is the greedy procedure's output, which failed its final check; switching to the no-critic +
  anchor-only recipe (better pooled record, faster) is the owner's call.
