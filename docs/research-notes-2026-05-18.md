# Research notes — 2026-05-18

Validating the 7-VR RLVR suite end-to-end: run online RL with the VRs, then
head-to-head the trained model against the pre-RL BC baseline. The suite
itself (`rlvr/online/slippi_stream.py`, `rlvr/online/vr/`, `CompositeVRTask`,
`combo_extend` retirement) was implemented over the prior days; this is the
first real validation run. It surfaced two bugs and one dead end.

## Bug 1 — the stock-delta SD-gate inverted the reward

The first RL run trained on a broken signal. `stock_delta` credits an
opponent stock loss as `+1` only if the opponent was "recently in hitstun /
a DAMAGE state" — a gate meant to filter opponent self-destructs. The gate
(`recently_in_hitstun_or_damage`) scanned a fixed 20-frame window back from
the **stock-decrement frame**.

That window is in the wrong place. The `stock` counter decrements a long,
*variable* time after the killing hit — the KO'd character sits in a DEAD
action state (range 0-10) through the whole blast-zone / star-KO animation,
which can run 1.5+ seconds, *before* `stock` ticks down. So a 20-frame
look-back from the decrement frame lands entirely inside DEAD frames and
never reaches the hit. Every kill was classified as an opponent
self-destruct: reward became `−1` per bot death, `+0` per kill — it
**punished winning**. `EVT_EP_VR` on a won match: `kills=0, deaths=3,
filtered_opp_sds=4`.

Instrumenting the death sequence made it obvious — at the stock-decrement
frame the opponent had been in action state 7 (`DEAD_UP_FALL_HIT_CAMERA`)
for the entire 90-frame dump.

This lag was *already documented* in CLAUDE.md's slippistats-porting section
("DYING (0-10) ... the stock decrement registers late") — the SD-gate just
didn't apply the known gotcha. Classic: the obvious predicate gets ported,
the timing subtlety gets missed.

**Fix:** `OppHitRecencyTracker` (`slippi_stream.py`) — a stateful recency
counter, not a backward scan. Each frame: hitstun/hitlag/DAMAGE → set the
counter to `hit_memory` (90 alive-frames); a DEAD frame → leave it
*untouched* (transparent); else decay by 1. On a stock loss, gate on
`counter > 0`. Because DEAD frames are transparent, the
hit → fly → dead → (long wait) → decrement sequence stays gated True
regardless of how long the death animation runs; a genuine SD (no hit in
the 90 alive-frames before death) still gates False. Two regression tests
lock in the star-KO and late-fall cases. Confirmed live: won matches now
score `kills=2..4`, positive reward.

## Bug 2 — PPO learning rate default too low

After the gate fix the run trained but the policy barely moved — checkpoint
weight-delta ~`6.5e-7` per 2 updates (the noise floor), KL-to-ref pinned at
`0`, and an h2h would have come back ~50%. Cause: `rlvr/online/loop.py`
defaults `--lr 1e-6`. At `1e-5` (10×) the weight movement scaled ~10×, KL
climbed smoothly to ~`0.0025` and plateaued. Use `--lr 1e-5` for VR runs;
the `kl_beta=0.01` penalty still guards drift. Diagnose a stuck run by
diffing checkpoint weights — the `update=` line's `reward` field is a
per-frame mean over ~9k frames and rounds to ~0 regardless.

## Result — the VR suite works

Realtime run: 50 updates, `lr=1e-5`, `stock_delta+damage_delta`, from the
Fox BC baseline. KL climbed `0 → 0.0025` and plateaued; clean single chunk,
no hangs. Head-to-head, trained model vs the BC baseline (Final Destination,
alternate-ports):

- **26–16 → 61.9% win-rate**, 42 matches
- avg stocks remaining: trained 1.12 vs baseline 0.57 (+0.55 margin)

A real but modest improvement. 26–16 at n=42 is only ~1.5σ (`p ≈ 0.16`) —
*probably* better, not statistically nailed. One run, one stage, opponent =
the baseline only. The RLVR loop is functioning; the effect size wants more
realtime matches to pin down.

## Dead end — FFW is unfaithful

Chased FFW (`emulator_ffw/`, `--use-exi-inputs --enable-ffw`) for the ~8×
speedup. The inter-update disconnect was fixed — during the PPO update
nothing pumps `console.step()`, so enet goes unserviced and the FFW
slippstream peer drops (`EnetDisconnected`). Fix: a keepalive thread in
`dolphin_actor.py` (`start_keepalive`/`stop_keepalive`) that idle-steps
Dolphin during PPO. That works.

But FFW itself is **not faithful**. FFW matches run ~4× shorter than
realtime — h2h: FFW mean 2430 frames/match vs realtime 9687; training
episodes ~3000 vs ~9000 — from the *same* models. FFW degrades gameplay
(the bots lose stocks ~4× faster), almost certainly the EXI input path
mistiming controller inputs under fast-forward.

The methodology trap, worth remembering: an FFW h2h gave a win-rate
*statistically consistent* with realtime (56.7% vs 61.9%, `p=0.60`) plus
clean `{1:N}` frame-delta histograms — and that was first read as "FFW
validated." Both signals are false positives. Win-rate survives *symmetric*
degradation (FFW hobbles both bots equally, so who-wins is preserved while
both play far worse); the histogram only checks the emulator's frame
stepping, not input fidelity. **Match length (frame count) is the metric
that exposes emulator-fidelity problems** — check it when comparing any two
emulator paths. The "FFW eval validated" claim was retracted.

FFW is parked. RL training and h2h eval run realtime (`emulator/`) — slow
(~17 h per 50-update run) but faithful. The keepalive fix is kept (correct,
harmless) but moot until FFW's input path is debugged.

## Open items

- `low_percent_kill` has the *same* stock-decrement-lag bug class — both its
  SD-gate and its death-percent peak look back from the decrement frame. Not
  fixed (it wasn't in this validation run); needs the `OppHitRecencyTracker`
  treatment + a fix for the death-percent window.
- The 61.9% result wants more **realtime** matches for a tight CI (FFW is
  out as a shortcut).
- Making FFW usable needs a focused EXI-input-path investigation: run the
  `emulator_ffw/` build with `--use-exi-inputs` but *without* `--enable-ffw`
  and compare match lengths — isolates whether the culprit is the EXI
  injection itself or the fast-forward timing.
