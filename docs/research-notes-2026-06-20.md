# Research notes — 2026-06-20

**In-game verification of the less-reg + tail-SWA recipe** (the 2026-06-19
study's open item). Plus a methodology fix to the val-vs-winrate plot: re-score
every point's val on a fixed seed so the x-axis is actually comparable.

## TL;DR

- The recipe (less-reg + tail-SWA, **32k steps**) reaches the **same in-game
  strength as the 442k-step long run** vs the common opponent — **76.5%** vs
  **76.2%**. The "fewer steps" goal is met: long-run quality at ~14× fewer steps.
- But the recipe does **NOT** beat the warm-restart it improves on in a direct
  h2h: **42.9% (9–12 over 21)** — a coin flip, not the projected +11 pp. The
  −0.0087 val edge did not convert to a measurable in-game edge.
- Net: the recipe's value is **training efficiency**, not raw strength over the
  warm-restart. This is the 2026-06-18 caution made concrete — tiny val gaps buy
  tiny, noisy edges.

## X-axis fix: re-score every point on fixed seed-42 `--eval-only`

The old plot mixed in-training "best val" numbers, which the 06-19 study showed
are selection-biased (the warm-restart's "0.7176" was really 0.7270 on a fixed
subset). Re-scored all four checkpoints through `train.py --eval-only --seed 42`
on `data/foxrank_master_v2` (same val loop, 100 batches):

| checkpoint | old plotted val | **seed-42 val** | steps |
|---|---|---|---|
| original (`fox-master-20260615`) | 0.7334 | **0.7438** | 32k |
| warm-restart (`…0616-warmrestart`) | 0.7176 | **0.7270** | ~32k |
| **recipe** (`AVG_R3_last5`) | — | **0.7183** | **32k** |
| long run (`…0616-long`) | 0.7130 | **0.7174** | ~442k |

Sanity: warm-restart → 0.7270 and AVG_R3 → 0.7183 reproduced the 06-19 numbers
exactly, confirming the eval path. On this honest axis the recipe (0.7183) sits
within **0.0009** of the long run (0.7174) — and beats the warm-restart by
**0.0087**, as claimed.

## H2H results (realtime, Fox ditto, FD, alternate ports, common-opponent rig)

All against the same fair rig validated 06-18 (`selfplay_control.json` = 47.6%).

| matchup | A | B | result (A) | report |
|---|---|---|---|---|
| **#1 recipe vs original** | `AVG_R3_last5` | `…0615` | **76.5%** (13–3–1, **17** matches*) | `lessreg_swa_vs_original.json` |
| **#2 recipe vs warm-restart** | `AVG_R3_last5` | `…warmrestart` | **42.9%** (9–12, 21 matches) | `lessreg_swa_vs_warmrestart.json` |

\* H2H #1 hung at match 18/21 (Dolphin stall during an interrupted run) and was
stopped at 17; 13–3–1 is already conclusive and matches the long run's 76.2%.

For reference (existing reports, same rig):
- long run vs original: 76.2% (16–5, 21)
- warm-restart vs original: 71.4% (15–6, 21)

## Interpretation — the projection overpredicted

The 06-19 note projected ~+11 pp for the recipe over the warm-restart, from the
−0.0087 val gap × the ~1.3 pp/0.001 slope measured on the *vs-original* points.
The direct h2h says otherwise: **42.9%, within noise of even**. Two reasons the
extrapolation broke:

1. **The 1.3 pp/0.001 slope is a far-field fit.** It was measured over val gaps
   of 0.016–0.020 (original → warm-restart → long). Extrapolating it down to a
   0.009 gap assumes linearity into a regime where the 06-18 data already warned
   of a noise floor (long vs warm-restart was only 57% at 0.0046 apart, itself
   within noise). Small val gaps don't buy proportional strength.
2. **H2H isn't transitive, and the vs-original edge is itself noisy.** Recipe
   76.5% vs warm-restart 71.4% looks like a recipe edge, but at N≈17–21 that
   ~5 pp difference is within noise — consistent with the recipe and warm-restart
   being ~even, which is exactly what the direct #2 shows.

**This does not undercut the recipe.** Its real value was never "beat the
warm-restart" — it's "reach long-run-quality (76% vs original) in 32k steps
instead of 442k, for free (post-hoc averaging, less reg)." That holds: #1 ≈ the
long run. The recipe is a cheap path to a strong model, not a stronger model than
the warm-restart.

## Plot

`tools/plot_val_vs_winrate.py` now plots the seed-42 x-axis and draws the recipe
as a distinct orange star (vs the blue continued-training trend line), annotated
with step counts. `reports/val_vs_winrate.png` regenerated. The star lands on top
of the long run — the "same strength, 14× fewer steps" story reads off the chart.

## Open / next

- Recipe is **validated as an efficiency win**, not a strength win over the
  warm-restart. Use it as the default cheap fine-tune; don't expect it to out-duel
  a model it's only ~0.009 val better than.
- The 1.3 pp/0.001 val→winrate slope should be treated as **non-linear / far-field
  only**. Below ~0.01 val gap, treat models as even pending a large-N h2h.
- No promotion: the long run (0.7174, shipped) is still the champion by a hair and
  the recipe doesn't beat it. The recipe is a *method* to apply when retraining
  other characters cheaply.
- If we ever want to resolve recipe-vs-warmrestart for real, it needs N≫21
  (the 9–12 split has a ~±20 pp CI).
