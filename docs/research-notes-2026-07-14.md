# Research notes — 2026-07-14

## L-cancel RLVR: first working engine-confirmed-metric RL result

Goal (from the RLVR roadmap): use L-cancel as the test case to prove the
online RLVR loop end-to-end with an engine-confirmed metric, then tune the
recipe for time-to-target. Result: **the loop works and improves the
policy** — BC 94.2% → 96.8% avoidable-lag success in 100 updates (recipe 1),
and the recipe search below collapsed the wall clock further.

### Bugs found and fixed before anything could run

Review of the (already-built, May-era) `rlvr/online/` stack found one known
defect and three unknown ones. All four would have corrupted or destroyed
the L-cancel training signal:

1. **Reward was the raw `post.l_cancel` flag**
   (`rlvr/online/tasks/l_cancel_online.py:enrich_with_replay`). The flag
   miscounts ledge-slides and hit-interrupted landings as misses
   (research-notes-2026-06-29). Replaced with **realized avoidable lag**:
   forward-scan the landing-state run in the post-match .slp,
   `reward = 1.0 iff max(0, lag − cancelled_min[move]) == 0`,
   `CANCELLED_MIN = {NAIR:7, FAIR:11, BAIR:10, UAIR:9, DAIR:9}` (Fox).
   Landings exiting into damage/dead states carry no input-timing signal
   and are dropped. Verified against an independent reimplementation of
   `tools/lcancel_analysis.py` on 1,246 real landings (25 human master
   games + 10 bot games): **0 mismatches**; mean reward 0.909 on the human
   corpus vs the documented 90.4% baseline, 0.918 on bot replays vs ~92%.

2. **Mid-match drain destroyed deferred-reward episodes**
   (`dolphin_actor.py:collect` + `actor_pool.py:_actor_collect_loop`).
   Every mid-match episode close called `_finalize_match_episodes()`, which
   enriches against `_last_replay_path` — mid-match that is either None
   (first match → pending episodes dropped) or the **previous match's
   .slp**. Frame ids restart every match, so the landing-frame lookup
   "succeeds" in the wrong game and scores silent garbage. Essentially
   every L-cancel episode would have been dropped or mis-scored. Fix: new
   `_drain_ready_episodes()` surfaces only final-reward episodes mid-match;
   NaN-pending episodes stay buffered until the match-end finalize.
   Behavior identical for non-deferred tasks.

3. **All actors shared one replay dir** (`actor_pool.py`). With
   `--n-actors > 1`, `_find_latest_replay` grabs the globally newest .slp —
   often another actor's match. Fix: per-actor subdirs
   (`replay_dir/actor{i:02d}`). Also added a stale-replay mtime guard in
   `_find_latest_replay` (match-start wall-clock stamp) so a failed replay
   save drops episodes instead of scoring them against the previous match.

4. **"Ineligible" episodes were punished as misses**
   (`l_cancel_online.py:compute_outcome`). Aerials ending in
   GENERIC_LANDING (**autocancels — optimal play**) or interrupted by the
   opponent were scored `terminal_reward=0.0`. With group-normalized
   advantages that trains *against* autocancels. In the first smoke run
   these were **half the batch**. Fix: NaN + `pending=False` = unscoreable;
   the actor discards them at close (they no longer reach PPO or count
   toward the collect quota).

### FFW faithfulness re-verified inside the training harness

Smoke run (2 actors, FFW vs CPU-9, `gfx_backend=Null`, blocking input) —
canaries on the run's own replays:

| canary | run | healthy ref | broken ref |
|---|---|---|---|
| L-cancel rate | 90.9% (150/165) | 88–91.5% | ~23% (self-play FFW) |
| measured `cancelled_min` | 7/11/10/9/9 | exact table match | — |
| match length | 6,866 mean | ~6,400 | ~1,600 |
| center-stick | 38.8% | ~34%, suspect >45% | 100% |

The frame-exact `cancelled_min` match is the strongest signal: engine input
timing under FFW+EXI (1 injected bot) is identical to realtime. Throughput:
~235–260 fps/actor at n=2 (~4× realtime each), ~13× aggregate at n=4.

### Recipe 1 (baseline): it works, but slow

`--task l_cancel_online`, CPU-9 opponent, lr 1e-5, kl_beta 0.01, clip 0.2,
1 PPO epoch, 4 actors, 64 eps/update, T=1.0, γ=0.998, ref = BC.
100 updates ≈ 85 min wall (run in 4 pause/resume segments; KL re-anchored
to the original BC ref on each resume).

Endpoint A/B eval (identical conditions both sides: `ffw_batch_mp --ckpt
… --replay-dir …`, 8 FFW envs × 300 s, CPU-7, then `lcancel_analysis`):

| | BC (`AVG_mastfox`) | RLVR 100 upd |
|---|---|---|
| avoidable-lag success | 94.2% (n=830) | **96.8%** (n=869) |
| engine-flag rate | 93.3% | 96.5% |
| match length | 6,323 | 6,619 |
| center-stick | 39.5% | 38.0% |

Miss rate 5.8% → 3.2% (−45%), z ≈ 2.6, p ≈ 0.01. Gains concentrated where
BC was weakest: BAIR 90.0→96.6, FAIR 88.3→94.2, UAIR 93.1→95.4,
DAIR 94.5→98.2 (NAIR flat within noise). Canaries clean — no trigger-spam,
no play-style collapse. KL(π‖BC) ended at 0.010.

Two lessons about measurement:
- **The in-run success curve was useless for steering.** Per-update batches
  (~75 eps) have ±2–3% binomial noise; 10-update pooled bins stayed
  "flat" 93.7–95.6% for all 100 updates while the endpoint eval showed an
  unambiguous +2.6-pt gain. Compare recipes at eval, steer in-run by
  KL/clip_frac only.
- **In-run rate ≠ eval rate.** The training measure drops
  opponent-interrupted landings (reward design), the eval counts more
  categories; they sit ~1 pt apart. Only like-for-like endpoint A/B counts.

### Recipe search (time-to-target)

Recipe 1's own telemetry showed where the wall-clock was going:
clip_frac 0.00 for all 100 updates (policy far inside the trust region →
lr headroom), one gradient pass per rollout batch (collection is ~95% of
wall time → epochs are nearly-free reuse), kl_beta 0.01 anchoring to a
5.8%-miss prior (caps attainable success).

Recipe 2 (`fox-lcancel-rlvr-20260714e-fast`): lr 3e-5 (~3×),
kl_beta 0.003 (~3× weaker anchor), **ppo_epochs 4** (new `--ppo-epochs`
flag in `rlvr/online/loop.py` — repeated `ppo_update` calls against the
same stored `logprob_old` = textbook multi-epoch PPO), 8 actors.
≈12× learning-per-rollout. In-run: KL settled into a 0.03–0.05 band
(β-balanced, not runaway), clip_frac 0.04–0.06 (clip active, far from
saturation), t_ppo 7 s vs t_collect ~40 s.

Recipe 2 was stopped at update 50 (KL hit 0.28 with flat batch success —
the stop-loss case) and evaluated as a **checkpoint ladder** against the
reusable BC eval set:

| ckpt | KL(π‖BC) | wall | success% | n | m_len | cstick% |
|---|---|---|---|---|---|---|
| BC | 0 | — | 94.2 | 830 | 6323 | 39.5 |
| recipe-1 final | 0.010 | ~85 min | 96.8 | 869 | 6619 | 38.0 |
| recipe-2 u20 | 0.077 | ~17 min | 96.0 | 1108 | 6440 | 36.8 |
| **recipe-2 u30** | 0.130 | ~25 min | **98.1** | 1171 | 7088 | 31.5 |
| recipe-2 u50 | 0.280 | ~41 min | 96.6 | 1179 | **8873** | **24.3** |

Findings:
- **u30 beats recipe-1's endpoint in ~30% of the wall clock** (98.1 vs
  96.8). The 12×-learning-per-rollout recipe works.
- **Success is non-monotonic in KL.** Peak ≈ KL 0.13; by KL 0.28 success
  regresses AND both canaries break (match length +40%, center-stick
  collapses to 24% — playstyle wander, not L-cancel gain). kl_beta 0.003
  was too weak to cap drift at the peak; the checkpoint ladder + eval is
  what catches it. Always keep `--checkpoint-every 10` and evaluate the
  ladder, not just the final.
- **Diagnosis of the drift vector:** ~95% of episodes succeed and get a
  small positive advantage smeared (γ=0.998) over every frame of the
  episode — reinforcing whatever playstyle happened, not just the
  trigger press. The L-cancel decision lives in the ~7 frames at landing.

Recipe 3 (`fox-lcancel-rlvr-20260714f-g09`) tried **γ=0.9** to concentrate
credit on press-adjacent frames. **Stopped-lossed at update 15 — γ decay is
the WRONG mechanism.** Under whole-batch z-scored advantages, decayed
returns make the *early* frames of *successful* episodes carry negative
advantage (their return ≈ 0.9^30 ≈ 0.04, below the batch mean), so the
update actively suppresses starting aerials. Measured: batch success fell
96→89.9→92.8% within 15 updates while KL burned 2× faster than recipe 2
(0.093 vs 0.046 at u15). Negative result worth keeping: **don't lower γ on
terminal-reward tasks with group-normalized advantages.**

Recipe 4 (`fox-lcancel-rlvr-20260714g-tail15`) tried the "right way" to
concentrate credit: **`--episode-tail-frames 15`** (new flag) — train only
the last 15 frames of each episode, γ back at 0.998 so advantages stay
balanced. **Also a negative result:** ladder eval u20/u30/u40 =
95.7/96.3/95.9% — all below recipe-2's u30 (98.1%), at similar KL burn.
Conclusion (with recipe 3): **the whole-episode training was load-bearing.**
L-cancel success is not just the trigger press — the early-aerial frames
(fastfall timing, aerial height, spacing) set up the landing, and recipe 2
was improving those too. Its "drift" was partly on-objective adaptation;
it only turns harmful past KL ≈ 0.13. Don't cut credit windows on tasks
whose setup matters.

### What the residual ~2% actually is (miss autopsy on r2-u30's eval)

All 22 misses inspected: **20/22 had a full trigger press within the 10
frames before landing** — the policy presses, but *too early* (the window
is the last ~7 frames; an early press held through landing has no rising
edge in-window — the classic human miss mode). 0/22 platform cases (FD
eval), 2/22 no-attempt. Misses skew to the late-hitting aerials
(UAIR 10, BAIR 6, FAIR 5, NAIR 1, DAIR 0). So the tail is a
**timing-precision problem, trainable in principle** — not an irreducible
environment quirk.

Recipe 5 (`fox-lcancel-rlvr-20260715h-trust2`): **iterated trust region** —
restart from the r2-u30 champion with `--ref-ckpt` re-anchored to r2-u30
itself and kl_beta back to 0.01. The u30→u50 degradation happened because
KL-to-BC grows unboundedly once the policy is far from the anchor; re-
anchoring turns optimization into a sequence of bounded trust-region
rounds around the current best.

Recipe-5 ladder (KL now measured from the r2-u30 anchor):

| ckpt | KL(π‖r2u30) | success% | n | m_len | cstick% |
|---|---|---|---|---|---|
| r5 u10 | 0.080 | **98.5** | 1175 | 7456 | 29.8 |
| r5 u20 | 0.141 | 98.4 | 1092 | 7090 | 24.5 |
| r5 u30 | 0.217 | 98.1 | 1237 | 6128 | 28.2 |

Round 2 works but with hard diminishing returns: +0.4 pt at u10 (inside
noise vs 98.1, z ≈ 0.75), then flat while center-stick keeps eroding.
**Diagnosis of the plateau: rare-event gradient starvation.** At 98.5%
success a 64-episode batch holds 0–2 misses; many updates see zero
(all-success ⇒ zero advantage variance ⇒ the update is a pure KL step).
The signal per update collapses exactly as the policy improves —
batch size must scale with 1/miss-rate to keep the negative examples per
update roughly constant.

Recipe 6 (`fox-lcancel-rlvr-20260715i-trust3-b192`): trust-region round 3
from r5-u10 (best + least drift), **episodes-per-update 192** (~3–5 misses
per batch), kl_beta 0.01, otherwise recipe-2 hypers.

Recipe-6 ladder: u5 97.6 / u10 97.4 / **u15 98.5%** — round 3 ended level
with its own anchor (r5-u10, 98.5). Two independent continuation
strategies (tight KL, big batch) stalled at the same **~98.5% attractor**.
Interpretation: not gradient magnitude (misses get −5σ) but **sample
coverage** — the residual miss contexts are rare and heterogeneous (each
appears ~once per several matches), so per-context learning is starved.
Time-to-99 scales with context frequency, not step count.

Recipe 7 (`fox-lcancel-rlvr-20260715j-long-b192`): single long run instead
of anchor-hopping (trust-round resets change the KL geometry every 30
updates; a fixed anchor gives consistent pressure). From r5-u10, ref fixed
at r5-u10, kl_beta 0.003 (headroom), b192 (precision), 60 updates,
checkpoint ladder every 5.

Recipe-7 in-run: after ~20 updates the big-batch success rate jumped to a
sustained 99.5–100% (u21–23: 213/214, 202/203, 197/198; u30: 205/205) —
the first sub-1%-miss stretch of the search. Stopped at u30 (KL 0.49,
deep past every prior degradation threshold). Ladder eval:

| ckpt | KL(π‖r5u10) | success% | n | m_len | cstick% |
|---|---|---|---|---|---|
| r7 u20 | 0.20 | 98.9 | 961 | 7484 | 28.5 |
| **r7 u25** | ~0.27 | **99.4** | 939 | 6775 | 29.5 |
| r7 u30 | 0.49 | 99.3 | 866 | **4971** | **18.2** |

u30 shows the cliff: success holds but canaries break (matches 21% short,
stick collapse) — the ladder catches exactly where to stop. **Confirmation
eval on r7-u25 at 2× size: 99.2% on n=2,230 (95 games), 95% CI
[98.8, 99.6]; pooled u25 evidence 3,169 landings @ 99.2%.** Canaries
clean: m_len 7126 (healthy range 6.1–7.5k across all non-degraded
checkpoints; the broken signature is ~5k), cstick 28.8 (<45), flag rate ≡
avoidable rate (no trigger-spam). Per-move ≥98.8%, NAIR 100% (n=379).

### Result and the recipe that got there

**BC 94.2% → 99.2% avoidable-lag success (miss rate 5.8% → 0.8%, −86%).**
Candidate checkpoint: `fox-lcancel-rlvr-20260715j-long-b192_update0025.pt`.
NOT promoted — promotion stays h2h-vs-BC + user approval (the KL(π‖BC) is
substantial and center-stick sits ~10 pts below BC, so strength must be
adjudicated in-game).

Winning path (wall clock is training only; each round's ladder eval adds
~15–20 min):

| leg | recipe | updates | wall | eval result |
|---|---|---|---|---|
| BC → r2u30 | lr 3e-5, kl 3e-3, epochs 4, b64, n8 | 30 | ~25 min | 98.1% |
| → r5u10 | + re-anchor ref at champion, kl 1e-2 | 10 | ~9 min | 98.5% |
| → r7u25 | + ref fixed, kl 3e-3, **b192** | 25 | ~63 min | **99.2%** |

Reference: recipe-1 (lr 1e-5, kl 1e-2, 1 epoch, b64, n4) spent 85 min to
reach 96.8% and plateaued — no visible path to 99.

### What each lever bought (the reusable part, for the other VRs)

1. **ppo_epochs 4** — the single biggest win. Collection is ~95% of wall
   time; reusing each rollout 4× is ~free. logprob_old stays fixed →
   textbook multi-epoch PPO; clip_frac never exceeded ~0.15.
2. **lr 3e-5** — clip_frac ~0.00 at lr 1e-5 meant the trust region was
   never touched; 3× was stable everywhere.
3. **kl_beta 0.003** — the anchor caps attainable success (a 5.8%-miss
   prior pulls back); but success is **non-monotonic in KL** (peak ≈0.13
   from anchor in round 1; cliff by 0.49 in round 3). The kl_beta doesn't
   reliably cap drift — the **checkpoint ladder + endpoint eval** is the
   actual control mechanism. Never trust the final checkpoint.
4. **Batch scaling with success** (b64 → b192) — at 98.5% success a
   64-episode batch holds 0–2 misses; all-success batches are pure KL
   steps. Batch must grow ~1/miss-rate to keep the rare-event gradient
   alive. This is what broke the 98.5% attractor.
5. **Re-anchoring** (trust rounds) — works but with hard diminishing
   returns (+3.9 → +0.4 → +0.0 per round). The long fixed-anchor run
   (recipe 7) outperformed anchor-hopping.
6. **Negative results:** γ decay (recipe 3) actively suppresses aerial
   starts under group-normalized advantages; `--episode-tail-frames`
   (recipe 4) cuts off the setup-frames learning path (aerial timing/
   fastfall) that L-cancel success depends on. Whole-episode training at
   γ≈1 is correct for this task shape.
7. **Measurement discipline:** in-run success is unusable for steering at
   b64 (±2–3%/batch); compare recipes only at endpoint eval with n≳1000,
   and confirm any threshold claim at n≳2000. Steer in-run by clip_frac
   and KL only.

### Session infra fixes (committed with this note)

- `--ppo-epochs`, `--episode-tail-frames`, lr default 1e-6→1e-5, tuned
  recipe in the module docstring (`rlvr/online/loop.py`).
- The four L-cancel-blocking bugs (avoidable-lag reward, mid-match drain,
  per-actor replay dirs + stale-replay guard, ineligible-episode NaN
  discard).
- `tools/ffw_batch_mp.py --ckpt` for endpoint A/B evals.

### Next direction: miss-targeted savestate drilling

Uniform rollouts collect signal at the miss rate — at 99.2% a miss appears
once per ~130 landings, so holding ~5 misses per batch means b≈650
(~9 min/update), doubling for every further halving. The fix is to target
the failures: harvest savestates ~1–2 s before each detected miss, then
roll **N stochastic completions per savestate** — which forms a
matched-context GRPO group (same state, advantage = reward − group mean;
the LLM N-completions-per-prompt structure), far lower-variance credit
than z-scoring across heterogeneous episodes. Mix drilled mini-episodes
with regular rollouts; h2h gate adjudicates as usual.

This is the roadmap's "Option C" savestate harness (designed 2026-04-21,
never built) in miss-prioritized form. The real payoff is the next
objectives: tech/recovery trigger states occur a handful of times per
match (vs ~25 aerial landings), so uniform collection is near-hopeless
there — the savestate harness is the enabling infrastructure for the
high-consequence skills. Untested prerequisites: savestate behavior under
the FFW/EXI headless build, and save/restore of the policy's 180-frame
context + controller state.

### H2h vs BC (2026-07-15 afternoon): skill transfers, strength collapses

r7-u25 vs `AVG_mastfox`, realtime `tools/play.py`, fixed ports (RLVR p1 /
BC p2), stopped after 9 games — the verdict didn't need 17:

- **Strength: catastrophic.** 0 wins, **0 stocks taken in 9 games** (BC
  won 3-0/3-0/2-0/4-0/3-0/4-0/3-0/3-0/…). The vs-CPU eval showed none of
  this (r7-u25 beat CPU-9 every match with near-normal canaries) —
  vs-CPU strength saturates and cannot adjudicate (the roadmap's
  two-sided h2h gate exists precisely for this).
- **L-cancel: the gain is real and context-robust.** In those same games:
  RLVR **98.8%** (n=162, CI ±1.7) vs BC **92.7%** (n=179) — a +6.1-pt gap
  under real pressure. BC's h2h rate matches its vs-CPU rate (92.7 vs
  94.2), validating the metric across contexts. The skill gain and the
  strength loss are separable facts.

### How much drift bought this (parameter-space ladder from BC)

Relative L2 `‖θ−θ_BC‖/‖θ_BC‖` per checkpoint (state-dict note: BC saves
carry `_orig_mod.` compile prefixes; strip before comparing — a silent
key mismatch reads as 0 drift):

| ckpt | rel-L2 from BC | L-cancel (vs-CPU eval) | h2h strength |
|---|---|---|---|
| r1_final | 0.00028 | 96.8% | not tested |
| r2_u30 | 0.00134 | 98.1% | pending |
| r5_u10 | 0.00154 | 98.5% | pending |
| r7_u25 | 0.00199 | 99.2% | **0 stocks in 9 games** |
| r7_u30 | 0.00207 | 99.3% | (canaries already broken) |

Two lessons: **(1) 0.2% relative parameter movement is enough to erase
head-to-head strength** while vs-CPU play still looks normal; **(2)
behavior is violently non-linear in parameter distance** — u25 (clean
vs-CPU canaries) and u30 (broken) differ by only ~4% of an already tiny
distance. No parameter-norm or KL threshold substitutes for playing the
pre-patch model. The practical promotion recipe for RLVR checkpoints is
therefore three-tiered: engine metric (cheap, every ladder step) →
canaries (cheap, every ladder step) → **h2h vs base (mandatory before
promotion; canaries under-detect strength loss)**.

### The full ladder (h2h complete)

r2-u30 (17 games) and r5-u10 (stopped at 9 — verdict clear) vs BC, same
realtime rig:

| ckpt | rel-L2 from BC | L-cancel h2h (n) | h2h W-L | verdict |
|---|---|---|---|---|
| BC | — | 92.7–94.5% per set | — | baseline |
| **r2-u30** | 0.00134 | **96.9%** (523) | **7–10 (41%)** | parity (rig control 47.6%); skill gain intact |
| r5-u10 | 0.00154 | 97.3% (186) | 1–7 (12%) | degraded — mostly swept |
| r7-u25 | 0.00199 | 98.8% (162) | 0–9, **zero stocks** | collapsed |

Findings:
- **The strength cliff is razor-sharp in parameter space:** +15% more
  distance (0.00134→0.00154) took win rate from parity to 12%; +30% more
  to zero. No smooth trade-off curve to ride — the usable region ends
  abruptly, and only h2h sees the edge (vs-CPU play and canaries looked
  near-normal well past it).
- **The L-cancel gain survives at every rung** (96.9/97.3/98.8 vs BC's
  ~93), including in games the policy loses badly — skill acquisition
  and strength retention are fully decoupled along the RLVR path.
- BC's h2h L-cancel (92.7–94.5 across three replay sets) matches its
  vs-CPU 94.2 — the avoidable-lag metric is stable across contexts.

**Promotable profile: r2-u30** (`fox-lcancel-rlvr-20260714e-fast_
update0030.pt`) — the 25-minute checkpoint: +2.4-pt L-cancel under
pressure (96.9 vs 94.5 in-game paired), strength at parity within n=17
noise. Confirming "not down" more tightly would need a larger h2h (n≈50
for ±14%). r5-u10/r7-u25 are research artifacts, not candidates.

RLVR validation status: reward works, transfers to real opponents,
strength cost is measurable and localized on the drift ladder — the loop
plus its three-tier eval (metric → canaries → h2h) is validated
end-to-end.

### Loose ends

- h2h gate: DONE for r7-u25 / r5-u10 / r2-u30 (see "The full ladder").
  Remaining: a larger-n h2h on r2-u30 (n≈50) before any promotion, and
  the promotion itself is a user decision.
- ~15 intermediate RLVR checkpoints in `checkpoints/` (runs
  `20260714{,b,c,d,e}-`, `20260715{h,i,j}-`) — prune once a champion is
  picked.
- Recommended-default verification: the winning path was stitched
  (r2→r5→r7); whether a single from-BC run at lr 3e-5 / kl 3e-3 /
  epochs 4 / b192 reproduces 99+ in one shot is untested.

(Work spans 2026-07-14 evening through 2026-07-15 early morning.)


### Infra notes

- `tools/ffw_batch_mp.py` grew `--ckpt` (default unchanged) so endpoint
  A/B evals roll out any checkpoint under identical conditions.
- Importing `tools/ctrl_canary.py` executes a module-level 800-replay
  corpus scan — don't import it for its helpers; inline them.
- ~13 of 54 eval .slp per side are truncated (envs killed at the wall-time
  deadline mid-write); peppi raises (or Rust-panics) on them — skip with
  `except BaseException`. Symmetric across A/B, doesn't bias.
- The eval BC baseline set is reusable across recipe iterations (same
  generator, same conditions) — only the candidate side needs regenerating.
