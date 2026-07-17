# Research notes — 2026-07-16 (evening) → 07-17

## Run k: first dual-rollout FFW RLVR (policy vs frozen BC)

The payoff run for the three-agent day (research-notes-2026-07-15): the
L-cancel RLVR loop, but with the opponent = frozen BC (`AVG_mastfox`)
instead of CPU-9, under FFW on the fixed `emulator_ss` binary. This is
the configuration [[feedback_rlvr_training_opponent]] called for and the
dual-pad fix unblocked.

**Setup:** `rlvr/online/loop.py --opponent-ckpt checkpoints/AVG_mastfox.pt`,
8 actors on distinct slippi ports, `emulator_ss` FFW, tuned recipe
(lr 3e-5, kl_beta 0.003, ppo-epochs 4). Both models batch-1 on one 4090.
Throughput ~43–46 fps per actor at 8 actors ≈ **5.7× realtime aggregate**
— dual-model FFW self-play at scale worked first try, no
`EnetDisconnected`, no desyncs. 40 updates in **60 minutes** wall.

**In-run trajectory:** reward ~0.05 throughout (the opponent being BC not
CPU doesn't change miss-rate much — misses are self-inflicted, not
opponent-driven). KL: u10 0.098 → u20 0.095 → u30 0.090 → u40 0.145.
clip_frac 0.05–0.12.

### Skill ladder (vs-CPU FFW rollouts, avoidable-lag L-cancel)

| checkpoint | L-cancel | n |
|---|---|---|
| BC baseline (`eval_patched`) | 95.2% | 931 |
| k-u10 | 95.6% | 909 |
| k-u20 | 95.8% | 954 |
| k-u30 | 96.2% | 897 |
| k-u40 | 97.8% | 895 |

Deflated read: u10–u30 are within binomial noise of the baseline
(±~1.4 pts at these ns). Only u40 is a clear gain (+2.6 pts, ~3.5σ).
The earlier "monotonic ladder" framing oversold it — the honest summary
is "flat until u30, real gain at u40."

### Strength (h2h vs BC, realtime, the metric that matters)

- **k-u40: 1–11** over 12 matches. Degraded, unambiguously. Dual
  rollouts did **not** preserve h2h strength at this depth.
- Per-port L-cancel *inside those h2h games*: k-u40 **96.6%** (n=328)
  vs BC **93.0%** (n=302) — the skill gain is real and survives into
  the deployment context even while the checkpoint loses the matches.
  Skill/strength decoupling confirmed again, now in-context.
- **k-u20: 7–10 over 17 matches — parity** (same record as r2-u30's
  7–10; two-sided binomial p≈0.63). Per-port L-cancel inside those
  games: k-u20 **95.7%** (n=513) vs BC **93.6%** (n=471) — +2.1 pts,
  same direction as u40's +3.6 but not individually significant
  (CIs ±1.8/±2.2). So within run k the ladder reads: u20 (KL 0.095) =
  parity strength + marginal skill edge; u40 (KL 0.145) = collapsed
  strength + clear skill edge. The strength cliff for this run sits
  between KL 0.095 and 0.145.

### What this does and doesn't say about the dual-rollout theory

The coverage theory (on-policy KL only constrains visited states; BC
opponent visits deployment-like states) predicted dual rollouts would
hold strength better than vs-CPU at matched drift. The evidence is
**not clean support**:

- run-k u40 hit KL 0.145 and collapsed to 1–11; the vs-CPU r2-u30
  (KL 0.130) held parity (7–10). At *similar* KL the dual-rollout run
  did **worse** on strength, not better — though single runs at
  different depths, so not a controlled comparison.
- The KL trajectory was not self-regulating: it sat ~0.09 for 30
  updates then jumped 60% in the last 10.

What run-k *does* establish: the dual-rollout infrastructure works at
5.7× realtime, the skill transfers into deployment context, and the
destroy-and-learn cost model ([[feedback_rlvr_destroy_and_learn]]) got
its data point — u40 bought +3.6 pts in-context L-cancel for a
strength collapse.

### WiSE-FT interpolation (post-hoc recombination)

The checkpoints sit ~0.2% apart in weight space, so the cheap
recombination lever is linear interpolation
(θ = BC + α(θ_RL − BC), WiSE-FT). Built
`checkpoints/DBG-wiseft-ku40-a{25,50,75}.pt` from u40
(handling `AVG_mastfox`'s `_orig_mod.` compile-prefix keys); all load
through `load_mimic_model`. Question: does any α keep most of u40's
skill gain at h2h parity?

Skill ladder (same vs-CPU FFW protocol as above):

| checkpoint | L-cancel | n |
|---|---|---|
| BC (α=0) | 95.2% | 931 |
| a25 | 95.3% | 884 |
| a50 | 95.8% | 925 |
| a75 | 96.9% | 1037 |
| u40 (α=1) | 97.8% | 895 |

Skill interpolates smoothly (≈ linear in α). a25/a50 are within noise
of BC; a75 keeps ~⅔ of u40's gain (+1.7 pts, ~1.8σ).

a75 h2h vs BC (17 realtime matches): **5–12**, in-context L-cancel
97.1% (n=512) vs 95.7% (n=494). Comparison across the three h2h'd
points of the α/checkpoint family:

| point | h2h vs BC | in-context L-cancel edge |
|---|---|---|
| k-u20 | 7–10 (parity) | +2.1 pts |
| wiseft a75 | 5–12 (below parity) | +1.4 pts |
| k-u40 | 1–11 (collapsed) | +3.6 pts |

Verdict: **interpolation moved along the same skill-strength frontier,
not above it.** a75 recovered some of u40's strength loss but gave up
skill roughly proportionally, and is dominated (weakly, at these ns) by
simply taking the u20 training checkpoint. At ~17 matches per point
none of the pairwise differences among u20/a75 are individually
significant — the clean claims are only that u40's skill gain is real
and that u40's strength collapse is real. No free lunch from post-hoc
recombination at this scale; the cheap-recombination hope from
[[feedback_rlvr_destroy_and_learn]] is not refuted (one objective, one
run) but gets no support here.

## Code review of the drilling harness (4ce353e) — 2 reward-corruption bugs fixed

The agent-written miss-drilling harness was pushed without human review;
an adversarial review agent went over it against the canonical
avoidable-lag rule, the PPO path, and the fork's flush-ledger caveat.
The PPO/group-advantage path and constant reuse verified correct. Bugs
found and fixed (in that order of severity):

1. **No frame-discontinuity guard during warmup/control.** A straggler
   LOADSTATE host job (e.g. from a timed-out rollout — the job has no
   cancellation) could fire mid-episode and silently poison a
   matched-context group with a reward belonging to a different state.
   Fix: any frame jump ≤0 or >5 during warmup/control aborts the
   rollout (`frame_discontinuity`).
2. **Run length counted delivered frames, not frame ids.** A dropped
   frame inside a landing run undercounted realized lag by 1, flipping
   a marginal miss (avoidable_lag 1) to reward=1.0 — an inverted
   gradient on exactly the near-miss population the drill targets.
   Fix: `length = exit_frame − start_frame` (the canonical replay rule
   is a frame-index span).
3. **SAVESTATE/LOADSTATE verbs sent with their own FLUSH**, violating
   the dual-pad fix's ledger caveat (1-frame input skew around every
   save/load). Fix: verbs are now written without a flush and ride the
   same frame's input flush.
4. **Restore off-by-one**: the re-delivered capture frame was pushed
   into the context window again (duplicate frame, wrong
   controller-history feature). Fix: consume it with a neutral press;
   warmup starts on the next frame.
5. Orphan-Dolphin guard on failed session start; bounded keepalive
   join that refuses to hand a possibly-still-driven console back to
   the main loop.

Post-fix re-smoke (2 updates, 6 states, N=8): mechanics pass — 48
loads, latency p50 0.19 s, rewind frame delta 0 vs sidecar (confirming
the no-own-flush verbs still fire), zero `frame_discontinuity` aborts
(no false positives from the new guard). All 6 groups were degenerate
(all-success or no-landing), so no gradient step ran — mechanics are
validated, training efficacy is not, and the v1 finding that N=8
groups are mostly zero-variance stands. Drill v2 needs bigger N and/or
mixed-outcome state curation before it produces signal.


## Leash post-mortem + direction discussion (owner session, 07-17)

### Drift measurements for run k (rel-L2 vs BC, same metric as the r-series ladder)

| checkpoint | KL (rollout states) | rel-L2 | h2h |
|---|---|---|---|
| k-u10 | 0.098 | 0.00075 | — |
| k-u20 | 0.095 | 0.00106 | 7–10 parity |
| k-u30 | 0.090 | 0.00131 | — |
| k-u40 | 0.145 | 0.00150 | 1–11 collapsed |
| wiseft a75 | — | 0.00113 | 5–12 |

r-series reference: parity at 0.00134, collapse at 0.00199. So run k
collapsed at *less* weight drift than the r-series collapse rung, and
a75 underperformed at *less* drift than the r-series parity rung.
Neither KL-on-rollouts nor weight-space distance transfers across runs
as a strength threshold — direction of drift matters, not magnitude.
(Noise caveat: at n=17 matches, only u40's 1–11 clears binomial noise,
p≈0.006; 7–10 vs 5–12 are not distinguishable from each other or from
parity. All decode at T=1.0 sampling, fixed ports, FD — the noise
floor is wide.)

### Owner conclusions

- One run is enough to lose confidence in constraint-style leashes;
  no need to prove the point further.
- **Average KL is wrong in principle, not just underpowered**: BC's
  action span is wide and consequence is non-uniform — drift in a few
  specific actions (shoulder off-stage, ledge SDs) is catastrophic
  while most drift is free. A mean over action-probability mass cannot
  see the difference. This is the action-space twin of the
  val-loss-vs-strength argument ([[project_error_consequence_nonuniform]]).
- The same reasoning casts suspicion on full-temperature sampling as
  the deployment decode; wants a temperature/decode investigation.

### Human-miss mining (owner proposal, adopted direction)

The human replay corpus already contains labeled L-cancel mistakes
(~10% of landings at the ~90.4% corpus rate). Two uses considered:

1. **RLVR on harvested human-miss windows** (GRPO-style matched-context
   groups) — the adopted direction. All infrastructure exists: playback
   harvest (savestate at any human replay frame), the drill loop, and
   the dual-pad fix for the port-2 frozen-BC driver a restored human
   state needs. Two advantages over bot-miss drilling: no discovery
   cost (millions of pre-labeled windows), and human-miss contexts are
   plausibly hard contexts, so bot success there should sit well below
   the 95% ceiling that made bot-miss N=8 groups degenerate.
   **Cheap gate before committing: harvest ~50 human-miss states, roll
   8 per state, measure group variance.** Caveat: miss-only contexts
   are a skewed training distribution; mix with regular rollouts.
2. **"Fix the data" (edit misses into successes, retrain BC)** —
   rejected. Owner's objection: doesn't generalize to multi-turn
   skills (edgeguard, punish) with no single fixable window. Stronger
   objection: the edit is *counterfactual* — the frames after a fixed
   miss still show the lag animation that actually happened, so the
   edited buttons contradict the surrounding observed gamestates even
   for the single-frame task.

### Decode/temperature investigation (queued)

Prior data point: 2026-06-27 sweep, T<1.0 → 4-stocked (near the edge
the modal action can be the SD; greedy also biases toward the most
common class — center-stick ~34% — i.e. passive drift). But T=1.0 has
the mirror problem: it faithfully samples BC's rare catastrophic
actions at their natural rate. Untested middle ground: nucleus/top-p
or per-head temperature (buttons low-T, sticks full-T) — cut the
low-probability disaster tail without collapsing the multimodality
greedy destroys. The top-k/top-p plumbing was already designed in the
June decode-tuning plan (never executed); `_safe_sample` in
`tools/inference_utils.py` is the single place to change; the
common-opponent h2h rig is the measure. No training required.

Ranked next steps by information value: (1) group-variance probe on
human-miss states, (2) decode/temp sweep, (3) full human-miss GRPO
run. Owner caveat on (3): it inherits the select-by-h2h gate, and h2h
is wall-time expensive — cheaper now that the dual-pad fix lets h2h
run under FFW instead of realtime, but still the dominant cost (tens
of matches per checkpoint for records that clear binomial noise).

## RL progress snapshot (as of 2026-07-17)

**Proven:**
- The RLVR mechanism works end-to-end on a real skill: L-cancel
  94.2% → 99.2% vs CPU (r7), with a tuned recipe (lr 3e-5, kl_beta
  0.003, 4 PPO epochs, batch ∝ 1/miss-rate) and an
  engine-confirmed metric (realized avoidable lag).
- Skill transfers into deployment context and survives strength
  collapse (u40: +3.6 pts in-context L-cancel while losing 1–11) —
  skill and strength are decoupled in the weights.
- Infrastructure: dual-rollout FFW self-play at ~5.7× realtime
  (8 actors, one 4090), savestate pipe verbs (patched fork, pinned),
  replay-seeded savestate harvest from human .slp, miss-drilling
  harness (mechanics validated, reward-corruption bugs found by
  review and fixed).

**Not solved:**
- Strength preservation. Every constraint-style leash tried or
  measured (KL on rollout states, weight-space distance, WiSE-FT
  recombination, BC-opponent rollouts) failed to predict or prevent
  h2h collapse. Current working position: train aggressively,
  checkpoint densely, and *select* by h2h — measurement over
  constraint.
- Drill training efficacy (groups degenerate at N=8 on bot misses).
- Only one objective attempted; multi-RLVR balancing untested.
