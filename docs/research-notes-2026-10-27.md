# 2026-10-27 — Simplifying sim RL: no critic, one KL term, same strength, less time

Night of 10-26 → 10-27. Goal: the fewest moving parts in `tools/sim_rl.py` dense mode that are at least as strong
as the SF7 recipe and faster. **All results are from one training seed** (sim_rl `--seed 0`) per run; no run was
repeated. Nothing promoted.

**Reproducing SF7:** commit `c2df04b` is the last one whose sim_rl dense defaults are SF7's recipe (critic, both KL
terms). The next commit removes the critic code and sets the per-decision KL to 0 in dense mode.

## Setup

- Start: `checkpoints/smallfox/base_full56k` (S, full attention, cond-heads, 56k steps). Opponent: the frozen base.
  Reward `stock_delta:1,damage_delta:1`. Local RTX 4090, one GPU job at a time.
- SF7 recipe: 16,384 decisions × 100 updates, critic, two KL terms (per-decision 0.03 on the sampled actions, global
  anchor 0.01 on 512 random learner frames), 4 PPO epochs, lr 1e-5. Reference model: SF7u100
  (`checkpoints/smallfox/SF7-full56k_16k_u100/eval_final`).
- Greedy ablation, one change per run, stacked on the last kept recipe. Keep rule: the final checkpoint vs SF7u100 in
  a 192-game bf16 sim h2h at seed 41 has win rate ≥ 47% (draws half) and one-sided exact binomial p > 0.05 against
  50%, and training wall-clock is lower than the last kept recipe's.
- Scripts and logs: `checkpoints/smallfox/ablate/` (`step.sh`, `score.py`, `retime.sh`, `results.txt`).

## Results

| Step | Change | vs SF7u100 (192, seed 41) | One-sided p | Train time | vs base (192) | Kept |
|---|---|---|---|---|---|---|
| — | SF7 (reference) | — | — | 111.6 min | 140–52 (72.9%) | — |
| 1 | no critic | 94–98 (49.0%) | 0.414 | 107.3 min | 143–49 (74.5%) | yes |
| 2 | + 1 PPO epoch, lr 4e-5 | 82–110 (42.7%) | 0.026 | 69.8 min | 139–53 (72.4%) | no |
| 3 | + no per-decision KL (anchor only) | 100–92 (52.1%) | 0.742 | 115.7 min* | 148–44 (77.1%) | yes |
| 4a | + 50 updates, lr 2e-5 | 83–109 (43.2%) | 0.035 | 51.5 min | 137–55 (71.4%) | no |
| 4b | 33 updates, lr 3e-5 | not run: 4a failed | | | | |

Final check, fresh seed: **step 3 recipe vs SF7u100, 192 games, seed 42: 106–86 (55.2% [48.1, 62.1], two-sided
p = 0.170, one-sided p = 0.935).**

\*Step 3's full run overlapped external CPU load (a browser and a headless Chromium on the same box) from about
update 30: rollouts went from ~29.6 s to ~33 s and the update phase from ~32.5 s to ~40 s. Over updates 1–25,
before the load, its update phase was 32.5 s vs step 1's 34.7 s with equal rollouts. Re-timed under the same load,
8 updates each in A1, A3, A3, A1 order: step 1 8.6 / 8.6 min, step 3 8.3 / 8.3 min (update phase 275 s vs 256 s
per 8 updates, rollouts equal). That ≈ 3.5% (~3.5 min per 100 updates) is what kept it on the time rule.

Final recipe (sim_rl dense defaults since this commit): 16,384 decisions × 100 updates, 4 PPO epochs, lr 1e-5,
advantage = (return − batch mean) / batch std, global KL anchor 0.01, no critic, no per-decision KL.

## Notes

- **Critic.** Removing it changed nothing measurable (49.0%), as the negative explained variance predicted
  (`docs/research-notes-2026-10-26.md`). It saved ~2 s per update (the critic forward and its 20-epoch fit).
- **Per-decision KL.** With it off, sim_rl now skips the frozen reference's scoring pass over the 16,384 decisions
  (nothing uses it); that is where the time saving comes from. The anchor alone lets the policy move further from BC:
  anchor KL 0.0191 at u100 vs 0.0070 (step 1) and 0.0113 (SF7). Its training reward was the highest of the runs (mean over
  updates 91–100: stock +0.39 / damage +1.59 per min vs step 1 +0.24 / +1.25 and SF7 +0.32 / +1.48; rollout games
  in those updates 118–33 vs 102–45 and 106–41).
- **PPO epochs.** The 4 epochs are not interchangeable with a 4× lr single step: same number of updates, same
  per-update lr × steps, but 1 epoch at 4e-5 lost 82–110. The 4-epoch runs' clip_frac is ≤ 0.0004, so the gain from
  multiple epochs is not from PPO's clipping; the later epochs re-evaluate the gradient at the moved parameters.
- **Fewer updates.** 50 updates at 2× lr lost 83–109, consistent with 10-26's finding that more updates (not more
  data per update or more lr) drive the gain. Its training reward over updates 41–50 (+0.25 stock / +1.02 damage per
  min) was at or above the 100-update runs' over the same updates (+0.12 to +0.22 stock), but below where they
  ended (updates 91–100: +0.24 to +0.39 stock, +1.25 to +1.59 damage).
- **Logging.** In dense mode the `kl` field (per-decision KL to BC) is now `null`; the anchor KL is in `anchor_kl`.
  `vr_rl_run.sh`'s summary line prints whichever are present.
- One seed throughout, and the two strength checks of the final recipe (seed 41: 100–92, seed 42: 106–86) use the
  same trained checkpoint; they vary the evaluation games, not the training run.
