# 2026-10-28 — XS model, best S vs best M, short S runs, RL on XS, XS snapshot ladder

Overnight queue on the local 4090, 10-27 11:41 PM → 10-28 7:05 AM Pacific, owner-approved
(`checkpoints/overnight_20261028/overnight.sh`, log next to it). One training seed per model, nothing
promoted. All h2h in the native sim, bf16, Fox dittos, `replays_doc_rt`, 192 games per match; "3 seeds" =
h2h seeds 41/42/43 pooled (576 games). Run dirs and result files under `runs/`; `runs/registry.tsv` lists them.

## XS: `fox-XS-bc1-s1` (`recipes/bc-XS-v1.yaml`)

`bc-S-v1` with `--arch d_model=128,nhead=2,dim_feedforward=512` (6 layers): 1.66M params (S 5.34M, M 19.97M).
Same data, 56k steps, batch 768, Muon, cond-heads, seed 1. 77.5 min (12.0 step/s averaged; S's 56k took 125 min, 7.5 step/s
on this box, so 3.2× fewer params buys 1.6× speed). Weights-only snapshots at steps
0, 1, 5, 10, 50, 100, 500, 1k, 5k, 10k, 50k (`--save-steps`), final at 56k.

Final in-training val (100 random batches, same val split): XS 0.7582, S 0.7190, M (`fox-M-bc2-s1`) 0.6870.

| Matchup (3 seeds) | Record | Win rate [95%] |
|---|---|---|
| XS vs S (`fox-S-bc1-s1`) | 126–450 (45–147, 46–146, 35–157) | 21.9% [18.7, 25.4] |
| XS vs M (`fox-M-bc2-s1`) | 54–522 (18–174, 12–180, 24–168) | 9.4% [7.3, 12.0] |
| S vs M (10-27, for reference) | 196–380 | 34.0% |

## Best S vs best M: SF7u100 (`fox-S-bc1-s1.rl1-s0`) vs cond-heads M (`fox-M-bc1-legacy`)

95–97, 97–95, 74–118 = 266–310, SF7u100 46.2% [42.1, 50.3], p ≈ 0.07. Seed 43 is the outlier (38.5%).
Seed 41's `sim_h2h` printed and wrote its full report but exited nonzero at teardown (no traceback in the
log); its `result.json` was written by hand from the report (marked with a `note` field).

## Short S runs with their own full LR schedule (`bc-S-v1 --set steps=…`)

`fox-S-bc1-14k-s1` (36.9 min) and `fox-S-bc1-28k-s1` (67.4 min) vs the 56k final (`fox-S-bc1-s1`, 125 min),
next to the 56k run's own mid-run checkpoints at the same step (10-27, LR still at 0.85 / 0.50 of peak):

| Step | Separate run, own schedule | Mid-run checkpoint | Val: separate run / mid-run |
|---|---|---|---|
| 14k | 197–379, 34.2% [30.4, 38.2] | 33.2% | 0.7280 / 0.7381 |
| 28k | 268–308, 46.5% [42.5, 50.6] | 39.4% | 0.7167 / 0.7318 |

Seeds: 14k 63–129, 71–121, 63–129; 28k 95–97, 82–110, 91–101. The 28k run's last val (0.7167) is below the
56k final's (0.7190), within the 100-batch noise, and it still loses 46.5%.

## RL on XS: `fox-XS-bc1-s1.rl2-s1` (`rl-v2`, 100 updates, 78.2 min)

First RL launch through `tools/run.py` (worked end to end, including the export and the auto h2h vs parent).

| Matchup | Record | Win rate |
|---|---|---|
| XS+RL vs XS (seed 41 only) | 143–49 | 74.5% [67.9, 80.1] |
| XS+RL vs S, 3 seeds | 217–359 (72–120, 80–112, 65–127) | 37.7% [33.8, 41.7] |
| XS BC vs S, 3 seeds | 126–450 | 21.9% |

For reference SF7 (S + RL) beat its own base 140–52 (72.9%).

## XS snapshot ladder (`runs/fox-XS-bc1-s1/eval/snaps/`, not tracked)

Each snapshot vs the next one, 192 games, seed 41. Win rate is the later snapshot's; game length is mean
frames per game / 60.

| Pair | Later wins | Later win rate | Mean game length |
|---|---|---|---|
| 0 → 1 | 100–92 | 52.1% | 81 s |
| 1 → 5 | 96–96 | 50.0% | 82 s |
| 5 → 10 | 93–99 | 48.4% | 84 s |
| 10 → 50 | 89–103 | 46.4% | 84 s |
| 50 → 100 | 25–167 | 13.0% | 57 s |
| 100 → 500 | 183–9 | 95.3% | 53 s |
| 500 → 1k | 149–43 | 77.6% | 108 s |
| 1k → 5k | 161–31 | 83.9% | 127 s |
| 5k → 10k | 122–70 | 63.5% | 148 s |
| 10k → 50k | 130–62 | 67.7% | 151 s |
| 50k → 56k | 96–96 | 50.0% | 156 s |

Step 100 loses 167–25 to step 50 with shorter games; not looked at in replays yet (why it regresses is
unknown). The last 6k steps (LR near the floor) add nothing; 10k → 50k still does.

## XS deprecated (owner, 2026-10-28)

XS loses to S (21.9%) and M (9.4%) at the same recipe, trains only 1.6× faster than S on the 4090, and XS
BC + RL (156 min, 37.7% vs S BC) is weaker than S BC alone (125 min). `recipes/bc-XS-v1.yaml` stays as a
frozen record; don't launch it. The runs (`fox-XS-bc1-s1`, `fox-XS-bc1-s1.rl2-s1`) and the snapshot
ladder stay on disk for reference.

Dolphin watch games (realtime, one match per adjacent snapshot pair, `runs/fox-XS-bc1-s1/eval/watch/`,
not tracked): the later snapshot won 8 of 11. The exceptions were step 50 over step 100 (as in the sim),
step 500 over step 1k (against the sim's 43–149), and step 50k over step 56k (sim 96–96). Single games.

## Does the S recipe transfer to M? One-knob variants of `bc-M-v2` (vast box, 2× RTX 5090)

Owner's question: with data and steps fixed, does the config tuned at S (`bc-S-v1`: Muon 0.02, batch 768, 56k
steps, 647-shard data, cond-heads) carry over to M unchanged? Each run is `bc-M-v2` (= `bc-S-v1` at M) with one
change, one run per GPU (384 local × accum 2; 4.8–5.0 step/s each, 9.6 combined vs 8.6 for one 2-GPU DDP run),
~3 h 10 m each, commit 8290988. Each plays `fox-M-bc2-s1` (Muon 0.02, seed 1) at h2h seeds 41/42/43 on the
local 4090.

| Run | Change | Seeds 41 / 42 / 43 | Total | Win rate [95%] | Final val |
|---|---|---|---|---|---|
| `fox-M-bc2-mlr01-s1` | Muon LR 0.01 | 109–83, 113–79, 110–82 | 332–244 | 57.6% [53.6, 61.6] | 0.6844 |
| `fox-M-bc2-adamw-s1` | AdamW (lr 3e-4) | 86–106, 91–101, 92–100 | 269–307 | 46.7% [42.7, 50.8] | 0.6877 |
| `fox-M-bc2-s2` | training seed 2 | 89–103, 74–118, 87–105 | 250–326 | 43.4% [39.4, 47.5] | 0.6874 |
| `fox-M-bc2-mlr04-s1` | Muon LR 0.04 | 0–192, 0–192, 1–191 | 1–575 | 0.2% | 0.7972 |

`fox-M-bc2-s1` final val 0.6870.

- The seed-2 rerun (same recipe) loses 43.4%: training-seed noise alone moves the win rate ~6–7 points, more than
  the ~±4 of a 576-game match. One training seed per variant only resolves differences larger than that.
- AdamW at fixed data and steps is inside that noise, so AdamW is not what makes `bc-M-v1` (AdamW, 2× data,
  5× steps) beat `fox-M-bc2-s1` 62.5%; data and/or steps are (untested: whether AdamW gains more on a long schedule).
- Muon 0.04 is unstable at M: val rose from 0.80 to 0.83 over the first ~20k steps and never recovered.
- Muon 0.01 wins all three seeds by a similar margin, 14 points above the seed-2 rerun. One training seed.
  Muon LR 0.02 came from the 20M C cell at 6,500 steps (vs AdamW and 0.04; 0.01 was never tried there) and
  0.01 has not been tried at S.

**Defaults changed (owner):** `train.py --muon-lr` now defaults per size, M 0.01 / S 0.02
(`MUON_LR_DEFAULT`); new recipe `recipes/bc-M-v3.yaml` = `bc-M-v2` with `muon-lr: 0.01`. So S and M now differ in
one knob. Box run dirs (all checkpoints, logs, offline wandb, queue/setup logs) copied back with every file's
SHA-256 matched (`checkpoints/box_20261028/`), then the box was destroyed.
