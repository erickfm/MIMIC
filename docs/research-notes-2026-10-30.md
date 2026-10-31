# 2026-10-30 — L size (44M): BC + RL at 24k and 56k steps

Local 4090. New preset `L` in `mimic/model.py` (d_model 768, 12 heads, 6 layers, FFN 3072, 44,336,165 params;
same head dim 64 and depth as M) and recipe `recipes/bc-L-v1.yaml` (`bc-M-v3` at L, eff batch 768 as 192 × 4
accumulation, 24k steps). Measured on the 4090: BC ~2.0 steps/s (~12 GB), sim RL ~229 s per update (rollout ~80 s,
PPO ~149 s), vs M's ~131 s. h2h in the native sim, bf16, 192 games, seed 41, one training seed per model.

| Run | BC steps | BC time | RL time | Final val (100 batches) |
|---|---|---|---|---|
| `fox-L-bc1-s1` → `.rl2-s1` | 24k | 199 min | 376 min | 0.681 |
| `fox-L-bc1-56k-s1` → `.rl2-s1` | 56k | ~506 min | 376 min | 0.675 (best 0.662 @ 40.6k) |
| `fox-M-bc2-mlr01-s1` → `.rl2-s1` (reference) | 56k | 189 min | 219 min | 0.684 |

24k was chosen only so BC + RL + h2h fit one night (owner: "go till 9am"); the 56k run (owner: "run L at 56k with
RL") is the matched comparison with M. Partial checkpoints of a longer run can't stand in for a shorter one: the
cosine schedule depends on the step budget.

| Match | Result (first named) | Win rate [95%] | p |
|---|---|---|---|
| L 24k RL vs its base | 152–40 | 79.2% | |
| L 24k RL vs `fox-M-bc2-mlr01-s1.rl2-s1` (fox-master) | 103–89 | 53.6% [46.6, 60.6] | 0.35 |
| L 56k RL vs its base | 160–32 | 83.3% | 1e-21 |
| L 56k RL vs `fox-M-bc2-mlr01-s1.rl2-s1` (fox-master) | 92–100 | 47.9% [41.0, 55.0] | 0.61 |
| M RL (`fox-M-bc2-mlr01-s1.rl2-s1`) vs S RL (`fox-S-bc1-s1.rl1-s0`, SF7u100) | 152–40 | 79.2% [72.9, 84.3] | 1e-16 |

M RL vs S RL: stock margin +1.18 for M. Plots (train/val loss, RL reward and anchor KL per update, all three runs):
`checkpoints/L56k_20261030/` (not tracked). Not promoted; `fox-master` is unchanged.

At the matched 56k budget, L + RL ties M + RL (92–100, stock margin −0.05), using 2.2× the parameters, ~2.7× the
BC time and ~1.7× the RL time; the 24k L's 53.6% was within noise too. One seed per model, one 192-game match.
