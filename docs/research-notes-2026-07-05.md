# Research notes — 2026-07-05

Wrap-up of the L-cancel-baseline / FFW-rollout arc and a status pin on the two
parked BC-optimization levers (alternative optimizers, decode tuning). This note
documents work that until now lived only in memory + the working tree; the code
is committed alongside it.

## TL;DR

- **L-cancel baseline finally measured** (the never-done backfill): the production
  master-Fox model L-cancels at **~90–93% engine-confirmed** live — human-master
  level, corpus is ~90.4%. The old offline "98.2% solve" was a meaningless proxy
  (press-trigger-at-offset −4), ~uncorrelated with the real `post.l_cancel`. BAIR
  and FAIR are the consistent weak moves (~88–89%).
- **Decode tuning is dead** (closes Phase A of the optimizer/decode plan): full
  sampling at **T=1.0 is optimal**. Every lower temperature 4-stocks (argmax walks
  off the edge / SDs — determinism fragility near the ledge). The default is
  unchanged; the top-k/top-p plumbing is committed but the sweep found nothing to
  promote.
- **FFW-vs-CPU rollout generator built and validated at scale**: `ffw_batch_mp.py`
  with `--replay-dir` produces replays at **~28× realtime**, and fidelity holds —
  **93.6% L-cancel over 2,515 landings / 118 games** in a single 10-minute run.
- **FFW fidelity depends on the number of EXI-injected controllers**: faithful
  vs-CPU (1 injected bot), **broken in self-play** (2 injected bots → both ports
  degrade symmetrically to ~23% L-cancel, games 4× short). This is the first real
  thing the L-cancel test case surfaced and it constrains the RLVR opponent choice.
- **Muon + SOAP optimizer paths landed but are UNEXERCISED** (Phase B code-complete,
  never run). Committed as infra; not a production candidate until run + eval'd.

## L-cancel baseline (engine-confirmed, live)

Measured with `tools/lcancel_analysis.py` on live replays of `AVG_mastfox.pt`
(the master-Fox tail-SWA). Metric = **realized avoidable lag** = `max(0,
realized_landing_lag − cancelled_min[move])`, NOT the binary `post.l_cancel` flag
(which miscounts ledge-cancels/early-hits) and NOT the old offset=−4 press proxy.

| Sample | condition | L-cancel | n |
|---|---|---|---|
| Realtime self-play (7 games) | baseline | 90.1% | 394 |
| FFW vs CPU (8 games) | 1 injected bot | 92% | 141 |
| FFW vs CPU batched (32 games) | 1 injected bot | 93.3% | 505 |
| FFW vs CPU batched (118 games, 10 min) | 1 injected bot | **93.6%** | 2,515 |

Per-move on the big sample: NAIR 96.3%, DAIR 95.0%, UAIR 91.5%, BAIR 89.5%,
FAIR 88.5%. BAIR/FAIR are the durable weak spots across every sample.

Takeaway: the model is already at human-master L-cancel rates. L-cancel remains a
**low-consequence** objective (a missed L-cancel is minor tempo loss) — its value
in the RLVR roadmap is as the **test case** to build the rollout harness and prove
the metric isn't lying, not for the strength payoff.

## Decode tuning — swept, dead, default unchanged

Phase A of the plan (`.claude/plans/glistening-stargazing-owl.md`) added top-k/top-p
to the sampler and swept temperature × top-k head-to-head. Result (see
`reports/seq_decode_T{0.0,0.5,0.7,1.0}.json`, local):

- **T=1.0 full sampling** — self-play win 43.8% (a wash vs the fixed opponent; the
  rig's self-play control is ~47.6%). This is the shipped default.
- **T=0.7 / 0.5 / 0.0** — **0% win, 4-stocked** (avg_a_stocks 0.0). Lower temperature
  collapses to argmax-like decode which, near the ledge, walks off and SDs because
  the *modal* action can itself be the mistake ([[project_error_consequence_nonuniform]]).

So T=1.0 sampling is **load-bearing**, not a default nobody tuned. Decode tuning
buys nothing; the levers table for holding-data-model-task-fixed is exhausted
(this + the optimizer path below were the last two). Focus moved to RL.

## FFW rollout generator (vs-CPU) — built + validated

`tools/ffw_batch_mp.py` gained `--replay-dir`: every env's Dolphin saves `.slp`,
turning the throughput bench into a rollout/replay collector. One 10-minute run at
N=16:

- 28.4× realtime, 1,703 fps aggregate, **1.02M in-game frames**, 136 games, 0 crashes.
- 118 games parse clean; **18 truncated** — the last game per env, cut mid-write at
  the wall-clock deadline. **peppi panics (Rust `pyo3_runtime.PanicException`,
  uncatchable across a `ProcessPoolExecutor` worker) on truncated `.slp`** — the
  real collector must either let each env finish its current game before stopping,
  or parse defensively (serial + `except BaseException`).
- Fidelity sustained: **93.6% L-cancel / 2,515 landings** (table above).

This validates the vs-CPU rollout path end-to-end: generate at ~28×, replays land
on disk, metric extraction works, fidelity confirmed. Usable foundation for an
L-cancel (or any per-frame-reward) RLVR loop **with a CPU opponent**.

## FFW fidelity depends on # of EXI-injected bots

Controlled 3-way test (same model, T=1.0, FD, FFW/self-play the only variables):

| Condition | L-cancel | Match length | Verdict |
|---|---|---|---|
| Realtime self-play | 90% (n=394) | 9,410 f | faithful |
| FFW **vs CPU** (1 injected bot) | 92% (n=141) | 6,212 f | **faithful** |
| FFW **self-play** (2 injected bots) | 23% (n=60) | 2,100 f | **BROKEN** |

Per-port split of the FFW self-play games: **both** bots break (port1 21%, port2
26%) — a *global* engine-loop timing desync when two controllers are EXI-injected
under FFW, not a "one master pad" issue. Do NOT use match length as a fidelity
metric (opponent-confounded; pitfall #18) — only the L-cancel canary measures it.

**Reconciles the two prior notes** (`2026-05-18` "FFW unfaithful" was a self-play
run → right about self-play, over-generalized; `2026-06-30` "FFW faithful" was a
vs-CPU run → right about vs-CPU, over-generalized). Both were half-right.

**RLVR consequence:** the roadmap's recommended frozen-BC self-play opponent
([[feedback_rlvr_training_opponent]]) is a SECOND injected bot = the broken mode.
Cannot naively swap CPU→frozen-BC under FFW without fixing dual-pad EXI injection
first, or run frozen-BC self-play in realtime, or accept a CPU/scripted opponent.
The precise gecko-code mechanism is unverified; whether it's a fundamental FFW
limit or a fixable tooling bug is unknown (slippi-ai does self-play RL and may have
solved dual-agent FFW — not checked). See [[project_rlvr_ffw_unfaithful]].

## Parked BC levers (committed as infra, not run)

- **Muon + SOAP** (`mimic/muon.py`, `mimic/soap.py`, `train.py --optimizer`): the
  "better optimizer at equal compute" experiment. Muon on the ~40 hidden 2D
  matrices + aux-AdamW on embeddings/heads/relpos/norms; SOAP as a drop-in AdamW
  replacement. **Code-complete, NEVER RUN** — decode being dead and the RL pivot
  deprioritized it. Not a production candidate until de-risked (grad norms /
  Newton-Schulz NaN), LR-tuned, and seed-matched `--eval-only` beats the AdamW
  knee-cap baseline (0.7213/0.7178/0.7163 at seeds 42/123/7). The DDP box is down,
  so `SingleDeviceMuonWithAuxAdam` (single-GPU) is what's vendored.
- **Mirror augmentation** (`--mirror-aug`, `mimic/features.py:mirror_window`): the
  one *validated* win from the earlier arc (−0.004 val on all 3 seeds; should be
  default). Exact L-R involution on the 37/9 stick clusters. Committed here.
- **Width presets** `mimic-w768` / `mimic-w1024` (d_model-only scale-ups) for the
  does-extra-width-help sweep.

## Artifacts

Replay corpora live under `reports/` (gitignored — large `.slp`, local to the box
this ran on; they do NOT travel with the repo): `rollouts_ffw_cpu/` (136),
`lcancel_ffw_vs_cpu_batched/` (32), `lcancel_ffw_vs_cpu/` (8),
`lcancel_baseline_ffw/` (20, kept as FFW-self-play defect evidence),
`baseline_replays/` (7 realtime self-play). Regenerate on the new machine with
`tools/ffw_batch_mp.py --sweep 16 --seconds 600 --replay-dir <dir>` then
`tools/lcancel_analysis.py <dir>`.
