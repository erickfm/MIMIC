# Research notes — 2026-06-18

Pushed the master-only Fox model a lot harder via continued training, and —
importantly — pinned down **when val loss actually predicts in-game strength
for us**. Also added a `--warm-restart` flag to `train.py` (and fixed a real
LR bug in it). The 0.7130 long-run checkpoint is now the shipped `fox-master`.

## The training thread

`fox-master` (rank-master, val 0.7334) was badly *under*-trained: the original
32,768-step run saw only ~7% of one epoch over the 25k-game master set
(`data/foxrank_master_v2`; ~245M windows, 16.7M-window budget). So more steps
had lots of room. Sequence:

1. **Continuation** (resume latest step, *existing* cosine to its end): val
   0.7334 → **0.7267**. (Predicted it'd plateau; it didn't — undertrained.)
2. **Warm-restart** (resume the 0.7267 best, keep Adam, **reset LR**): →
   **0.7176**.
3. **Long run** (fresh, ~480k steps ≈ 1 epoch, single long cosine): best val
   **0.7130** recorded at step **441,600** (~92% of the 480k target) before
   being stopped. New champion. (The `_bestloss.pt` global_step is 441,600.)

## `train.py --warm-restart` + the LR bug it exposed

Added `--warm-restart`: on resume, keep the Adam optimizer state but reset the
LR schedule + step counter (fresh cosine from warm weights). The first cut was
buggy — the "fresh" cosine trained at ~eta_min (3.8e-6) instead of 3e-4.
Cause: **`CosineAnnealingLR` updates *recurrently* from the optimizer's
current lr**, and `optimizer.load_state_dict` had just set lr to the
checkpoint's end-of-cosine value; resetting only the scheduler (`base_lrs` was
fine) wasn't enough. Fix: after loading the optimizer, reset each param group's
`lr`/`initial_lr` to the base **and rebuild the scheduler** (`_make_scheduler`).
Verified in isolation + a live run (lr=3.00e-04 at step 1, Adam state intact).

## The real finding — when val loss predicts strength

Head-to-head each improved checkpoint vs the original master (21 matches,
realtime, alternate ports, Fox ditto, FD; `tools/bench_rank_fox.sh`):

| model (same master data) | val | win rate vs original |
|---|---|---|
| original | 0.7334 | — |
| warm-restart | 0.7176 | 71% |
| long run | 0.7130 | **76%** |

**Lower val → stronger bot, monotonically — but only because it's the *same
data*.** Across *different* data the relationship breaks: the rank ladder had
near-equal val (0.733/0.733/0.748) but a clear h2h gradient, and `fox-allranks`
(val 0.744) tied master 53% despite "worse" val. So the operating rule:

> **Val loss is a reliable *relative* metric for in-game strength only within
> the same data distribution** (picking the best checkpoint of a run,
> training-length variants). Across datasets (ranks/pools/characters), the loss
> scales aren't comparable — h2h is the only arbiter.

The magnitude tracks the gap, too: the two best models head-to-head (long
0.7130 vs warm-restart 0.7176, only 0.0046 apart) split **12–9 = 57.1%** for
the long model (stocks 1.10 vs 0.71). Same direction (lower val wins), but a
much smaller edge than the 0.016–0.020-apart vs-original gaps — and at N=21
that 57% is within noise of even. So: lower val ⇒ stronger holds, but tiny val
differences buy only tiny, noisy edges; don't over-read a 0.005 val improvement.

## Confound check (the part that almost bit us)

Every h2h above put the new model as A (`--ckpt`) and the original as B. In
`play.py`, **A is always decoded/pressed first each frame** (port is alternated
but processing order is not), and N=21 is odd — both plausible A-side
advantages. Control: ran the original vs an *identical copy of itself*
(`reports/selfplay_control.json`). Result: **A-slot 10–11 = 47.6%**, balanced
across ports (5/11 on port 1, 5/10 on port 2). So **the rig is fair** — under
`blocking_input=True` both controllers register before the frame advances, so
"A first" confers no edge, and `--alternate-ports` cancels the port effect. The
76%/71% are genuine strength, not artifacts.

## Result / shipped

`fox-master-20260616-long_bestloss.pt` (val 0.7130) promoted to
`erickfm/MIMIC/fox-master/` (overwrites the 0.7334; HF git history retains it).
Tools: `tools/run_long_fox.sh` (crash-resilient long run, auto-resume from
latest checkpoint), `tools/chain_warmrestart.sh`, `tools/run_allranks_fox.sh`.

## Open / notes
- `data/foxrank_*_v2` shards (~1.1 TB) can be freed; keep master if iterating.
- Same continued-training trick likely helps the other rank/character models —
  they're all under-trained at 32k steps.
