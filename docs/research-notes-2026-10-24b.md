# 2026-10-24b — second 8-hour budget: stream RL speedup, base comparisons, RL on both small bases

Owner gave another 8 h (11:10–19:10 PDT), small models only, nothing promoted. Scripts:
`checkpoints/smallfox/budget_{b,c,d,e}.sh`; every result line is in `checkpoints/smallfox/results.txt`.
All h2h in the native sim, bf16, Fox dittos, `replays_doc_rt` templates; 192-game screens at seed 41,
480-game confirmations at seeds 51 and 61.

## Stream RL ~2.6× faster (commit e3c1fa0)

- **Update (`rlvr/sim/ppo.py:last_features`).** A decision only needs the top layer's output at its
  own position. For a sliding-window model that depends on 1 position in the top layer, 180 in the one
  below, and so on (W − 1 more per layer). The update now computes only those positions, not all
  1,075 at every layer. Rows are right-aligned first, and the frames before an episode start are
  masked as keys. Exact: `rlvr/tests/test_rl_context.py` still passes at ~1e-7, with mixed-length
  batches. Forward + backward per decision: 7.3 → 2.5 ms (bf16 autocast added only ~10%, because
  attention runs in fp32 regardless, so the update numerics are unchanged).
- **Rollout (`BatchedPolicy.context`).** One batched gather for all slots, replacing one per slot.
- **In the run (S-stream, 64 envs):** update 142 → 57 s, rollout 22 → 6.6 s (env 1.5k → 5k fps).

## Base comparisons (480 games, seed 51)

| A | B | A's record | A win rate [95%] | Stock margin |
|---|---|---|---|---|
| S-stream 56k (`checkpoints/smallfox/base_stream`) | S-full 5-minute proxy (`base`) | 452–28 | 94.2% [91.7, 95.9] | +2.09 |
| S-stream 56k | fox-master (M-full, production) | 121–359 | 25.2% [21.5, 29.3] | |
| S-stream step 2,800 (192 games, seed 41) | S-full proxy | 116–76 | 60.4% [53.4, 67.1] | |

The first row mixes stream attention, 30× the steps and ~10× the data, so it isn't a stream-vs-full
test.

## RL: stock 1 + damage 1 (S2's recipe), 30 updates unless noted

| Run | Base | Change | Screen | p | Confirmations | Total |
|---|---|---|---|---|---|---|
| S2 (yesterday) | proxy | — | 122–70 | <0.001 | 289–190, 293–187 | 61.2% / 1,152 |
| **S11** | proxy | seed 1 | **122–70** | <0.001 | **289–191, 278–202** | **59.8%** / 1,152 |
| S13 | proxy | no critic | 118–74 | 0.002 | 268–212, 273–207 | 57.2% / 1,152 |
| S12 | proxy | critic 2 epochs (not 20) | 89–103 | 0.35 | — | |
| S10 | proxy | 60 updates | 97–95 | 0.94 | — | |
| S10 at u30 / u45 | proxy | same run's checkpoints | 109–83 / 104–88 | 0.071 / 0.28 | — | |
| SS3 | stream 56k | 60 updates | 104–88 | 0.28 | — | |
| SS4 | stream 56k | seed 1 | 93–99 | 0.72 | — | |
| SS5 | stream 56k | lr 3e-5 | 106–86 | 0.17 | — | |
| SS6 | stream 56k | lr 1e-4 | 57–135 | <0.001 (worse) | — | |
| SS7 | stream step 2,800 | — | 83–109 | 0.071 (worse) | — | |

Mean KL per update over a run (PPO KL to BC): proxy runs 0.004–0.006; stream 56k at lr 1e-5
0.0015–0.0017, 3e-5 0.0072, 1e-4 0.048; stream step 2,800 0.0028.

### What this says

- **The recipe replicates on the proxy base.** Two seeds, 61.2% and 59.8% over 1,152 games each.
  Run-to-run spread is large, though: S10 is the same recipe and seed as S2 (the runs aren't
  deterministic), and its u30 screened 56.8% against S2's 63.5%. Going past 30 updates didn't help
  (S10: 56.8 → 54.2 → 50.5% at u30/45/60, none significant).
- **The critic is optional.** Its explained variance is negative in every 20-epoch run (−0.1 to −0.4,
  falling over the run): fit 20 epochs on the last 3 batches, it predicts worse than the mean on the
  next batch, so it adds noise to the advantages. With 2 epochs it reaches +0.1, but that run screened
  46.4% (one screen). With no critic: 57.2% confirmed, against 60.5% for the two 20-epoch runs (about
  1.6 standard errors, not significant).
- **No stream base improved under RL.** That covers 4 runs on the 56k base (two seeds, 60 updates, 3×
  lr) and 1 on an early checkpoint, step 2,800, which is only slightly stronger than the proxy base.
  None gained; the early checkpoint went backwards (43.2%). In every stream run, the learner's
  stock_delta in rollout games fell over training (e.g. SS4 +0.07 → −0.34/min, SS7 −0.07 → −0.84),
  whereas it rose in the proxy runs (S11 −0.35 → +0.78). At equal KL (SS5 0.0072 vs proxy
  0.004–0.006) the stream policy still didn't improve. So **the cause is not base strength and not
  step size; something in stream RL itself is off.**
- **lr 1e-4 collapses** a stream base (29.7%, KL 0.048).

## Open: why stream RL doesn't learn

Re-scoring is exact (the test compares the update's fp32 logits with the logits the action was
sampled from, all in fp32). What the test does *not* cover is the production rollout path:
- bf16 autocast, a bf16 KV cache and a CUDA graph;
- the policy is sampled from bf16 logits computed through the cache;
- the update then scores the same actions in fp32 from scratch.

For full models the rollout is also bf16, but the window is recomputed every frame. A stream model's
cache carries bf16 keys/values for up to 1,075 frames, so the behaviour policy and the scored policy
may differ more. The first check for next time: on real rollout states, the KL between the rollout's
sampled logits and the update's fp32 re-score, for stream vs full models. If that's large, PPO's ratio
is off-policy by that much.
