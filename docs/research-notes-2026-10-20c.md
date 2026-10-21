# Lower val loss for a fixed dataset and compute budget

MIMIC research memo, 2026-10-20.

This memo draws on three sources:

- every research note in `docs/`;
- a read of the production training path;
- outside literature.

No repo files were changed.

Tags:

- **[measured]**: run on a checkpoint, or read directly in code.
- **[inferred]**: an estimate, not verified.
- **[lit]**: a published result, not tested here.

## Recommended order

1. Fix the val measurement so 0.5–2% gains can be detected.
2. Make the autoregressive heads condition on the real earlier-head classes. A probe measured −0.017 nats.
3. Fix weight-decay grouping, and run Muon and SOAP. Both are already implemented and have never been run.
4. Make each step cheaper (per-step metric syncs, FP32 relpos attention), then spend the savings on more steps or longer context.
5. Try seq 256.
6. Feed the encoder the shard features it currently ignores, projectiles first.
7. Data-limited levers:
   - all-character/all-rank pretraining with per-character fine-tuning;
   - ensemble → distill;
   - a joint retune of weight decay × step budget.

## Corrections to current beliefs

### The RoPE "deprecated" verdict compared seq 60 against seq 256

**[measured]** Before the 04-15 fix, `train.py` set seq 256 only when `model_preset == "hal"`. Any other preset fell back to `SEQUENCE_LENGTH = 60` (`git show 706a4af:train.py`, lines 59 and 660–663).

Which runs that affected:

- **04-13 Fox RoPE run:** it used `--model hal-rope` with no `--seq-len` (`docs/research-notes-2026-04-13c.md:22-31`), so it trained at seq 60. This run was even before the rename.
- **04-14 Falco sweep:** it used `mimic-rope*` presets inside the bug window, so it also ran at seq 60. The note says seq_len=256 (`docs/research-notes-2026-04-14.md:5-8`), but that is the setting it intended, not what ran.
- **The relpos reference:** the 04-12 `--model hal` run, at seq 256 (`docs/research-notes-2026-04-15.md:239-241`).

How much of the gap the bug explains:

- The 04-15 seq sweep measured seq 32→256 at −0.066 nats. The rename gap was 0.045–0.065 (`docs/research-notes-2026-04-15.md:376-377`).
- So most of the "structural ~0.08 gap" (`docs/research-notes-2026-04-14.md:13-17`) is plausibly the seq-length bug.
- The "RoPE is 2× faster" figure (16.7 vs ~8 step/s, `docs/research-notes-2026-04-13c.md:50`) has the same confound.

There is a second confound. The generic block used by the RoPE and flex presets has an extra dropout after GELU that the relpos block lacks (`mimic/model.py:614-619` vs `mimic/model.py:423-430`). RoPE and relpos have never been compared at matched seq length and dropout.

### The "autoregressive" heads don't condition on anything

**[measured]** Each head receives the detached logits of the earlier heads (`mimic/model.py:782-789`), not a true or sampled class. Those logits are a deterministic function of `h`, so the chain adds no information. In effect the four heads are independent.

Inference also samples each head independently (`tools/inference_utils.py:507-525`). That matches training, but it can't represent the joint distribution: for example TRIG together with a diagonal wavedash stick.

```
current:  h → shoulder(3);  [h, logits_shldr] → c_stick(9);  … → main(37);  … → buttons(7)
proposed: h → shoulder(3);  [h, onehot(shldr*)] → c_stick(9);  [h, onehot(shldr*, c*)] → main(37);  …
          * = ground truth while training, the sampled class at inference
```

### Weight decay hits LayerNorm gains, embeddings and relpos tables

**[measured]** The exclusion rule is `n.endswith("bias") or "norm" in n.lower()` (`train.py:862`). Neither norm name matches it:

- block norms are named `ln_1`/`ln_2` (`mimic/model.py:408,412`);
- head norms are named `heads.*_head.0` (`mimic/model.py:769`).

The only non-bias parameter excluded is `final_norm.weight`. So weight decay applies to:

- 12 block LayerNorm gains;
- 4 head LayerNorm gains;
- the stage/char/action embeddings;
- all six `Er` tables.

This likely explains part of why the 06-19 study found that lower weight decay helps (`docs/research-notes-2026-06-19.md:53-74`).

### `--plain-ce` and the label-smoothing/focal flags do nothing

**[measured]** In mimic mode, `compute_loss` returns `_compute_loss_hal` before it reads any of these flags (`train.py:329-330`). The loss is always the unweighted sum of four plain cross-entropies, averaged over all B×180 positions.

### Val covers about 256 games

**[measured]** `MAX_VAL_BATCHES = 100` (`train.py:77`), and the dataset draws up to 100 windows per game (`mimic/dataset.py:215`). So 100 batches × 256 windows ≈ 256 games, or 1–2 shards.

- `--eval-only` uses the same loop and cap (`train.py:1109-1145`).
- Same-seed comparisons are reproducible and paired, but they sit on a narrow slice of games.
- The ~0.004 cross-seed spread (`docs/research-notes-2026-06-19.md:11-23`) is mostly which-games variance.
- In-training "best" selection and the early-stop watchdog use this same estimate.

## Recommendations

### 1. Fix the measurement first

- Use a fixed, seeded set of a few windows from **every** val game, instead of 100 windows from each of ~256 games.
- Report late-position loss separately. Inference always predicts at position 179 with full context, but the current val counts every position equally.

**[measured]** Loss by context position on the c9nm checkpoint, over 1,536 val windows:

| Positions | Mean loss (nats) |
|---|---:|
| 0–9 | 0.850 |
| 10–59 | 0.761 |
| 60–169 | 0.734 |
| 170–178 | 0.704 |
| All (current val definition) | 0.7465 |

About 0.015–0.02 of today's val comes from short-context positions the bot never uses at inference. Comparisons stay valid, since every run carries the same offset. But noise in those positions dilutes the signal you care about.

### 2. Make the heads actually conditional

**[measured]** Probe on the c9nm checkpoint:

- **Setup:** freeze the model's marginal logits, add a learned offset table indexed by the true earlier-head classes, fit on half of 276k val frames, evaluate on the other half.
- **Button CE:** 0.1670 → 0.1465.
- **Total:** −0.017 nats. Main stick and c-stick didn't improve with additive offsets.
- **Control** (earlier-head classes shuffled): −0.003.

**[inferred]** A conditional head that interacts with `h` should do at least as well as an additive table.

The change touches two places:

- **`MimicPredictionHeads.forward`:** feed true one-hots during training.
- **`decode_and_press`:** sample shoulder → c-stick → main → buttons in order, feeding each sample forward.

The summed loss then becomes a true joint NLL. Rerun the baseline under the new definition instead of comparing to older numbers.

### 3. Weight-decay grouping and optimizer

- Exclude LayerNorm gains, embeddings and `Er` from weight decay.
- Run `--optimizer muon` and `--optimizer soap`. Both are implemented (`train.py:871-909`, `mimic/muon.py`, `mimic/soap.py`) and have never been run (`docs/research-notes-2026-07-05.md:115-122`). The baseline to beat there is 0.7213 / 0.7178 / 0.7163 on seeds 42 / 123 / 7.

**[lit]**

- **Melee BC at ~20M parameters:** Faynt reports Muon at 0.8463 val NLL vs 0.8605 for tuned AdamW. That is −1.65%, single seed, with Muon lr 0.01, aux-AdamW lr 1e-3 and momentum 0.95 Nesterov [1].
- **Large tuned-baseline benchmark:** matrix optimizers were 1.3–1.4× faster at 130M, with the advantage shrinking as scale grows. At ≥8× Chinchilla data per parameter, SOAP and Kron overtook Muon [2].
- **Why this matters here:** Fox runs see each frame ~41 times, which is that high-data regime. So run SOAP as well as Muon.

### 4. Make each step cheaper

- **[measured]** Metrics run every step, with about 40 GPU→CPU syncs:
  - four `.item()` losses;
  - `_multiclass_prf`, with boolean-mask indexing and an `.item()` per head (`train.py:142-161`);
  - `grad_norm.item()` (`train.py:1237`).

  Compute them only on log steps.
- **[measured]** Relpos attention disables autocast. It materializes FP32 (B, 8, T, T) tensors in every layer for `Q@Kᵀ`, `Q@Erᵀ`, the skew, the mask and the softmax (`mimic/model.py:379-391`). At bs 256 and T=180, each one is about 265 MB.
- **[inferred]** Two options for that attention:
  - BF16 matmuls with an FP32 softmax;
  - FlexAttention with a `score_mod` that reads a precomputed `QEr` (torch 2.11 is installed).

  BF16 has FP32's exponent range, so overflow is unlikely to be the real constraint. Test it.
- **[inferred]** Training costs about 127 MFLOPs per token. The documented 4090 rate (4.5 step/s at 256 × 2) is roughly 53 TFLOP/s, about 32% of peak. A profiler run would show where the rest goes.
- One clean RoPE-vs-relpos run, at matched seq length and dropout, would settle whether RoPE's speed is worth having.

### 5. Longer context

**[measured]** Loss is still falling at positions 170+ (table above).

Prior evidence that longer context helps:

- **04-15 sweep:** about −0.02 nats per doubling, up to 256. The 384/512 points are confounded, because the batch was halved to avoid running out of memory (`docs/research-notes-2026-04-15.md:353-407`).
- **[lit] HAL:** 128→256 was a large gain; 256→384 a smaller one [11].
- **[lit] Faynt:** uses 256 [1].
- **[lit] slippi-ai:** carries LSTM state across batches, so its effective context is unbounded [13].

Try 256 after step 4. Use gradient accumulation to keep the effective batch at 512–768.

### 6. Shard features the encoder ignores

**[measured]** These are in the shards but unused by `MimicFlatEncoder`:

- `self/opp_action_elapsed`: populated and normalized;
- 8 projectile slots: present on 24% of frames;
- Nana (Ice Climbers' second climber);
- the 20-wide global numeric vector, which includes stage geometry.

Projectiles are the most likely to matter, for Falco, Fox, Samus, Link and similar characters. The opponent's controller stays out, by design.

### 7. Data-limited levers

**Pretrain on all characters and ranks, then fine-tune per character.**

- **[lit]** Pretraining acts as a multiplier on fine-tuning data, and it helps most when fine-tuning data is small [7].
- **[lit]** In multilingual LMs, data from related languages helps low-resource languages until model capacity binds [8].
- **[lit]** An alternative to filtering by rank: keep all ranks and condition on a rank token. A single rating-conditioned chess model beat separate per-rating models [9]. Per-port rank is recoverable from the ranked replays.
- Nobody has published the clean ablation: scratch vs pretrain→fine-tune on the same target data.
- Expect the most gain for small characters and little for Fox. This is also one place a larger model could pay off, because the pretraining set is far bigger than any single character's.

**Ensemble, then distill.**

- **[lit]** The closest published setting to MIMIC's (fixed data, ample compute): distilling an 8-member ensemble into one same-size student kept 83% of the ensemble gain. Self-distillation beat the best single model [3].
- **[lit]** On Melee, Faynt's 75M→10M distillation gave −0.0046, then −0.0024 NLL [1].
- **[lit]** Distillation loses at fixed total compute when data is plentiful [4]. MIMIC is data-limited, the opposite regime.
- Cheap first check: average the logits of 3 seed runs and score that on val. The gap to a single model is the ceiling, before any distillation code is written.

**Retune weight decay and step budget together for long runs.**

- The 06-19 "less regularization" result came from a 32k-step warm-start fine-tune (`docs/research-notes-2026-06-19.md:53-74`).
- **[lit]** In the long multi-epoch regime, Kim et al. found optimal weight decay around 30× the standard value, tuned jointly with epochs [3]. That's a different regime from the 06-19 study, so the two results don't contradict.
- **[lit]** Dropout was the one regularizer that reliably reduced multi-epoch degradation [6]. Repeating data up to ~4 epochs is worth about as much as fresh data [5].
- **[measured]** The production Fox run is heavily multi-epoch:
  - data: 97.5k perspectives × ~9,650 frames ≈ 941M frames;
  - training: 280k steps × 768 × 180 ≈ 38.7B positions;
  - so each frame is seen about 41 times.
- This matters most for the small characters. Run a small grid there: weight decay × dropout × {1×, 1.5×} the plateau-fit step budget.

## What to skip

| Idea | Why |
|---|---|
| Bigger presets at fixed data | Diminishing returns: 2.2× params bought −1.8%, 3.7× bought −2.8%, and 154M bought ~2–3% at 4× the wall time (`docs/research-notes-2026-04-19.md:44-54`, `docs/research-notes-2026-04-20.md:5-10`). Faynt's fitted loss exponents: 0.20 for parameters vs 0.49 for data [1]. |
| GQA / SwiGLU / RMSNorm swaps | Tied in-house at matched size (`docs/research-notes-2026-04-19.md:83-104`). |
| Label smoothing | HAL found Gaussian smoothing over stick centers didn't help [11]. It also distorts the hard-label NLL being measured. Distillation gives soft targets that carry real information. |
| Multi-token prediction | Worse than next-token prediction at 0.3B and 0.6B in the original paper [12]. |
| WSD schedule for loss | A cooldown matches cosine [10], and averaging after WSD gave nothing in Faynt [1]. Its one use here: a cheap step-budget search (one stable run, branching cooldowns). |
| QK-norm, gated attention, value residual | Each is likely ≤0.5% at this scale, and none was tested with repeated data [14][15][16]. Revisit if Muon's higher LR needs stabilizing. |

## Smaller code issues

- **`mimic/dataset.py:215`:** `windows_per_game` reads `max_w` from the last game in the shard and applies it to every game. It doesn't trigger on current data, because every game is at least 2,612 frames.
- **Correlated batches:** each training batch comes from a single shard, about 170 games out of roughly 280 (`mimic/dataset.py:157-256`). So batches are mildly correlated, not i.i.d. [inferred]
- **Init scale:**
  - `Er` is initialized with `torch.randn` at std 1 (`mimic/model.py:364`).
  - Categorical embeddings start at std 0.02, while controller one-hots are 1. So action identity enters about 50× weaker at the start, and weight decay keeps pulling it down. [inferred]
- **EMA and DDP val:** EMA updates every step in a Python loop over parameters; `torch._foreach_lerp_` is cheaper. Under DDP, every rank runs the full val pass redundantly.

## What's already been tried

Condensed from the full notes sweep. Val numbers from different shard builds aren't comparable.

| Change | Result | Source |
|---|---|---|
| `--self-inputs` on v2 shards | 2.27 → 0.67 val; critical | `research-notes-2026-04-12b.md:50-58` |
| Data 10M / 51M / 305M frames | 0.82 / 0.73 / 0.671; small sets overfit by step 2.5k | `research-notes-2026-04-12b.md:62-66` |
| Seq 32 → 256 | About −0.02 nats per doubling; 180 chosen as ~92% of the gain | `research-notes-2026-04-15.md:353-479` |
| WSD vs cosine | Cosine better, but tainted by the seq-60 bug and a short decay | `research-notes-2026-04-15.md:545-579` |
| Full features (13 numeric + 5 flags) | 0.6641 vs 0.6890 baseline on puff (in-training best) | `research-notes-2026-04-19.md:44-54` |
| New normalization transforms | Mixed by character, single seed, never ablated | `research-notes-2026-04-20.md:16-73` |
| Fox 32k → 442k steps | 0.7176 → 0.7130 in-training best (0.7174 fixed-seed); 32k was badly undertrained | `research-notes-2026-06-18.md:10-21` |
| Tail-SWA of last 5 checkpoints | −0.005 on 3 seeds, free | `research-notes-2026-06-19.md:25-42` |
| Less regularization (dropout 0.1, wd 3e-3) | −0.0033 vs control; more regularization was +0.007 | `research-notes-2026-06-19.md:48-74` |
| Mirror augmentation | −0.004 val, measured with a facing-flag bug present; dropped by owner | memory: mirror-augmentation note |
| c9nm (c-stick fix + no mirror + 102k pool) | Won 16–5 h2h over the prior champion; the gain can't be attributed to one change | memory: fox-c9nm note |
| Per-character plateau-fit step budgets | steps ≈ 53.3·Mframes^1.19; EMA and SWA each won about half the time | memory: all-char retrain note |

Main stick is about 67% of total loss in every run (`research-notes-2026-04-19.md:196-198`). So changes that help the main-stick head dominate the total.

## Probe scripts

These ran CPU-only, against `checkpoints/AVG_c9nm_32k.pt` and the held-out shards in `data/mastfox_c9_local`. They live in a temporary scratch folder that may be cleared:

- `/tmp/claude-1000/-home-erick-projects-MIMIC/9e6ebbaa-88ab-41fa-9220-3d2b4d7c7de9/scratchpad/condprobe.py`: the conditional-head probe.
- `/tmp/claude-1000/-home-erick-projects-MIMIC/9e6ebbaa-88ab-41fa-9220-3d2b4d7c7de9/scratchpad/lastpos.py` and `/tmp/claude-1000/-home-erick-projects-MIMIC/9e6ebbaa-88ab-41fa-9220-3d2b4d7c7de9/scratchpad/perpos.py`: loss by context position.

## Sources

1. Janati, Kuzmin, Swamy, Niu. *Faynt: Scaling and Optimizing Policies for Competitive Melee.* arXiv:2610.02144, Oct 2026. https://arxiv.org/abs/2610.02144. Single seed per setting. Its absolute NLL isn't comparable to MIMIC's, because it uses a different controller codec; only the relative deltas carry over.
2. Wen et al. *Fantastic Pretraining Optimizers and Where to Find Them.* arXiv:2509.02046, 2025. https://arxiv.org/abs/2509.02046
3. Kim, Kotha, Liang, Hashimoto. *Pre-training under infinite compute.* arXiv:2509.14786, 2025. https://arxiv.org/abs/2509.14786
4. Busbridge et al. *Distillation Scaling Laws.* arXiv:2502.08606, 2025. https://arxiv.org/abs/2502.08606
5. Muennighoff et al. *Scaling Data-Constrained Language Models.* arXiv:2305.16264, 2023. https://arxiv.org/abs/2305.16264
6. Xue et al. *To Repeat or Not To Repeat: Insights from Scaling LLM under Token-Crisis.* arXiv:2305.13230, 2023. https://arxiv.org/abs/2305.13230
7. Hernandez et al. *Scaling Laws for Transfer.* arXiv:2102.01293, 2021. https://arxiv.org/abs/2102.01293
8. Chang et al. *When Is Multilinguality a Curse?* arXiv:2311.09205, EMNLP 2024. https://arxiv.org/abs/2311.09205
9. Tang et al. *Maia-2: A Unified Model for Human-AI Alignment in Chess.* arXiv:2409.20553, NeurIPS 2024. https://arxiv.org/abs/2409.20553
10. Hägele et al. *Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations.* arXiv:2405.18392, 2024. https://arxiv.org/abs/2405.18392
11. Eric Gu. *Behavior cloning Melee* (HAL blog). https://ericyuegu.com/behavior-cloning-melee
12. Gloeckle et al. *Better & Faster Large Language Models via Multi-token Prediction.* arXiv:2404.19737, 2024. https://arxiv.org/abs/2404.19737
13. vladfi1. *slippi-ai* (Phillip). Config `slippi_ai/jax/scripts/imitation_tx_like.py`. https://github.com/vladfi1/slippi-ai
14. Wortsman et al. *Small-scale proxies for large-scale Transformer training instabilities.* arXiv:2309.14322, 2023. https://arxiv.org/abs/2309.14322
15. Qiu et al. *Gated Attention for Large Language Models.* arXiv:2505.06708, 2025. https://arxiv.org/abs/2505.06708
16. Zhou et al. *Value Residual Learning (ResFormer).* arXiv:2410.17897, 2024. https://arxiv.org/abs/2410.17897
