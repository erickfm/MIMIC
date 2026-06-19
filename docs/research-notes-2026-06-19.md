# Research notes — 2026-06-19

**Goal:** a *reproducible recipe* (not a one-off model) that reaches a lower val
than the warm-restart's "0.7176" in the **same ~32,768-step / eff-batch-512**
budget. Motivation: our h2h data shows ~**1.3 pp win-rate per 0.001 val loss**,
so even −0.003 to −0.005 val is worth several points in real games.

## The methodology fix that reframed everything

The warm-restart's headline **0.7176 was selection bias.** Validation during
training samples 100 windows whose subset *varies across evals* (the global RNG
advances with training), so "best val over training" is the **max of noisy
samples → biased low**. Scored on a **fixed seed-42 val subset** via a new
`train.py --eval-only`, the warm-restart bestloss is actually **0.7270**.

Noise floor (eval the same ckpt under different seeds):
- **Same seed → exactly reproducible** (0.7270 == 0.7270): zero measurement
  noise, so at a fixed seed we can trust differences as small as 0.001.
- **Cross-seed spread ~0.004** (0.7270 / 0.7237 / 0.7230 at seeds 42/123/7): the
  absolute level shifts by subset, so a real recipe must win on **multiple seeds**.

**Rule going forward: never compare in-training "best val" — always
`--eval-only` on a fixed seed.** In-training bests are selection-biased.

## The result: tail-SWA is the recipe (free, reproducible)

Post-hoc **Stochastic Weight Averaging** — element-wise mean of a run's last ~5
step-checkpoints (`tools/average_checkpoints.py`) — reproducibly beats the
bestloss by **−0.005**, on every seed, with **zero training**:

| seed | warm-restart bestloss | SWA last-5 | Δ |
|---|---|---|---|
| 42  | 0.7270 | 0.7219 | −0.0051 |
| 123 | 0.7237 | 0.7186 | −0.0051 |
| 7   | 0.7230 | 0.7180 | −0.0050 |

Mechanism: the warm-restart's last ~10k steps *overfit* (raw val rose 0.7176→0.72
as LR→1e-6). Those tail checkpoints are each overfit, but **averaging cancels the
overfit into a flatter, better-generalizing point** — recovering value from the
otherwise-wasted tail. Projected win-rate gain ≈ **+6.5 pp** (Δval × 1.3 pp/0.001).
Best SWA window = the **last few converged checkpoints**; wider windows that reach
back into the high-LR phase are worse (last-5 0.7219 < w18–26 0.7237 < all 0.7260).

## Seeded training runs (all warm-start from cont 0.7267, seed 42, eff-batch 512, EMA dual-logged)

Scored at fixed seed 42. Floor to beat = warm-restart SWA **0.7216**.

| run | change vs control | raw bestloss | EMA bestloss | SWA last-5 |
|---|---|---|---|---|
| warm-restart | — | 0.7270 | — | 0.7219 |
| **R1 control** | defaults (WD 1e-2, dropout 0.2) | 0.7228 | 0.7228 | 0.7216 |
| **R2** | WD 3e-2 + dropout 0.3 (more reg) | 0.7294 | 0.7294 | 0.7282 ❌ |
| **R3** | **WD 3e-3 + dropout 0.1 (less reg)** | 0.7195 | 0.7190 | **0.7183** ✅ |

**Findings:**
1. **Online EMA (decay 0.999) ties raw best but does NOT beat post-hoc tail-SWA**
   (R1: EMA 0.7228 vs SWA 0.7216). EMA averages the whole trajectory incl. the
   high-LR noise; tail-SWA averages only the converged late checkpoints. Prefer
   tail-SWA; it's also free (no online cost, applies to existing runs).
2. **More regularization HURTS, LESS regularization HELPS.** The default reg
   (dropout 0.2, WD 1e-2) was *underfitting* the 32k warm-start fine-tune. R2
   (more reg) +0.007; **R3 (less reg) −0.0033 vs the control**, confirmed across
   all three seeds (below). The "overfit tail" was a red herring — the regime is
   underfit-prone, not overfit-prone.

### Cross-seed confirmation (the recipe is reproducible)

R3 (less-reg) SWA beats R1 (control) SWA by a consistent **−0.0032 to −0.0033**,
and the original warm-restart bestloss by **−0.0086 to −0.0087**, on every seed:

| seed | warm-restart bestloss | R1 control SWA | **R3 less-reg SWA** | Δ vs baseline |
|---|---|---|---|---|
| 42  | 0.7270 | 0.7216 | **0.7183** | −0.0087 |
| 123 | 0.7237 | 0.7183 | **0.7151** | −0.0086 |
| 7   | 0.7230 | 0.7176 | **0.7143** | −0.0087 |

(R4 = a *fresh* less-reg run trained at seed 123 confirms robustness to training
order, not just val subset — results appended on completion.)

## Recipe

> **Warm-start from a decent checkpoint + the standard 32k cosine, but
> REDUCE regularization (dropout 0.2→0.1, weight_decay 1e-2→3e-3), then promote
> the SWA of the last ~5 step-checkpoints** (not the bestloss, not the EMA).

vs the warm-restart's shipped bestloss: **−0.0087 val, ~+11 pp** projected
win-rate (Δval × 1.3 pp/0.001), reproducible on 3 seeds, **same 32k-step /
batch-512 budget**. Decomposition: less-reg ≈ −0.0033 (raw fit), tail-SWA ≈
−0.0012 on top of a well-selected bestloss; the rest is recovering the
warm-restart's badly-selected (selection-biased) bestloss.

## Instrumentation added (this study)
- `train.py`: EMA **dual-logs** raw `val/total` + `val_ema/total` and saves a
  separate `{run}_ema_bestloss.pt` (EMA never affects training → clean A/B);
  `--eval-only <ckpt>` scores any checkpoint with the exact training val loop.
- `tools/average_checkpoints.py`: SWA/LAWA averaging of N checkpoints (`--run … --last K`).

## Open / next
- Verify the SWA win in-game with one h2h vs the warm-restart bestloss (the
  +6.5 pp projection).
- The SWA recipe is free and general — apply it to every promoted model.
