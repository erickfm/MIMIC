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
- k-u20 h2h: in progress (17 matches total). This locates whether the
  KL~0.095 shelf is inside the parity band.

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
skill gain at h2h parity? Results below when run.

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

The harness's smoke run predates these fixes; re-smoke before any real
drill run.

<!-- PENDING: u20 h2h result, WiSE-FT ladder + h2h -->
