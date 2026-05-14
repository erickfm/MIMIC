# 2026-05-14 — RLVR pipeline end-to-end + first PPO + win-rate eval

Yesterday's note (`research-notes-2026-05-13.md`) ended with the V(s)
discovery list — candidate VRs ranked by matched-pair z-score on
human shard data. Today operationalized one candidate
(`combo_on_opp_damage`) and ran the first PPO-against-Dolphin
end-to-end comparison.

## What ran

Three sequential PPO runs from the same Fox BC starting checkpoint
(`hf_checkpoints/fox/model.pt`, val 0.7144, fox-20260420-baseline).
All FFW Exi-AI Dolphin, all 32 episodes / update × 50 updates, same
hyperparameters (`lr=1e-6, temperature=1.0, clip_eps=0.2,
kl_beta=0.01`).

| Run | Task | Wall time | Episodes/update (avg) | Combo / success rate (final 5 updates) |
|---|---|---|---|---|
| A | `l_cancel_online` (control) | 73 min | 47 | 84-100% L-cancel success |
| B | `shield_escape_online` (control) | killed at 47 min | 0 | **never started — see negative finding** |
| C | `combo_extend_online` (candidate) | 83 min | 47 | 24-46% combo rate |

Then a win-rate eval: 30 matches each vs CPU-9 Fox on FD,
FFW-headless, `rlvr/eval/winrate_vs_cpu.py`.

## Headline result — win-rate saturates, stock margin separates

Win-rate alone says nothing — the BC baseline already 30-0's CPU-9
Fox 100% of the time, so RLVR has nowhere to go up. But average
stocks remaining at match end (max=4) shows a real gap:

| Run | Win-rate | Avg stocks remaining | Stock margin |
|---|---|---|---|
| BC baseline | 100% (30-0) | 2.40 ± 0.86 | 60.0% |
| l_cancel RLVR | 100% (30-0) | 2.47 ± 0.86 | 61.7% |
| **combo_extend RLVR** | 100% (30-0) | **2.77 ± 0.73** | **69.2%** |

The diff is 0.37 stocks (BC → combo_extend); with sample SE ≈
0.86/√30 ≈ 0.157 the difference is ~2.3 SE — borderline significant
on N=30 alone, but the right direction and big enough to be worth
chasing with a properly powered head-to-head run.

l_cancel barely moves (+0.07 stocks). It's a control — the policy
already L-cancels at ~90% from BC, so the marginal task signal is
small. That's expected.

**Important caveat**: this run is a single seed per task. Per-policy
seed variance can plausibly account for ~0.1-0.2 stocks. So
"combo_extend > BC by 0.37 stocks" is suggestive but not yet a
rigorous causal claim. The head-to-head head-to-head plan below is
the proper validation.

## Negative finding — shield_escape doesn't trigger vs CPU-9 Fox

`shield_escape_online.should_start` requires:
1. Self in `SHIELD_ACTIONABLE` action-state, AND
2. `shield_strength < SMALL_SHIELD_THRESHOLD`, AND
3. A big shield-strength drop in the last `DAMAGE_DELTA_LOOKBACK`
   frames (i.e. someone hit our shield hard).

Against CPU-9 Fox on FD, 0 episodes triggered in 47 minutes. CPU-9
Fox doesn't pressure shield aggressively enough to drive
`shield_strength` down past the threshold within lookback. This
isn't a bug in the task — it's a real property of the opponent.
**The task is correctly written but unsuitable for CPU-9-Fox
training data**; would work fine against a human or a
shield-pressuring bot, but you need an opponent that *applies*
shield pressure to learn shield escapes.

Implication for future RLVR setup: opponent choice has to be matched
to the task. A "do everything against CPU-9 Fox" universal-trainer
strategy will silently no-op for any task whose triggering
condition CPU-9 doesn't produce.

## FFW Dolphin — measured 2.28×

Today wired in the Exi-AI Ishiiruka Dolphin fork (vladfi1's
prebuilt AppImage at `vladfi1/slippi-Ishiiruka/releases/exi-ai-0.2.0`)
at `emulator_ffw/`, separate from the netplay-capable `emulator/`.
With `use_exi_inputs=True, enable_ffw=True, gfx_backend="Null",
DISPLAY=:99`, measured **2.28× faster than realtime** on our actor
pipeline. Polling-mode was slower (56fps vs 62fps blocking) and
wastes ~94% of poll cycles — kept default `blocking_input=True,
polling_mode=False`.

The Python step loop is now the bottleneck at ~17ms/frame
(inference + state-machine + Episode bookkeeping). Dolphin can go
faster; we cannot consume frames faster. A GPU machine + maybe
torch.compile on the inference path would unlock real headroom.

## Two bugs found + fixed

**1. `.slp` enrichment race**: `DolphinActor` finalizes per-match
episodes on the same Dolphin frame as the match-end menu transition.
`libmelee` has just closed the .slp, but the OS hasn't flushed all
dirty pages. peppi-py opens the file and hits `I/O error: failed to
fill whole buffer`, dropping every pending episode that needed
post-match scoring. Fix: file-size-stabilization wait in
`_find_latest_replay` (up to 2s, breaks early on stable size) +
peppi-parse retry with exponential backoff in
`l_cancel_online.enrich_with_replay` (6 retries, ~3s max). Without
this, l_cancel produced near-zero training signal because every
other match's enriched rewards were silently dropped.

**2. `cfg_snapshot` NameError**: `rlvr/online/loop.py:171,179`
referenced a `cfg_snapshot` variable that was never defined.
End-of-training checkpoint save crashed. Fixed by renaming to
`model_cfg_snapshot = asdict(cfg)` and initializing at the top of
`train()`.

Both committed in `179d2b0` and `3ddfb7a`.

## KL spike observations

PPO with `kl_beta=0.01` saw rare single-batch KL spikes during
training:
- l_cancel update=25: KL=0.486 (16× normal)
- l_cancel update=48: KL=0.153 (5× normal)
- combo_extend update=21: KL=2.40 (100× normal)
- combo_extend update=35: KL=**9.80** (400× normal)

In every case the next update came back to KL≈0.02-0.04. The
combination of `clip_eps=0.2` and the KL penalty absorbs the spike
before it propagates — clip_frac stays in 0.14-0.22 range
throughout, and the policy doesn't collapse. Combo / l_cancel rates
don't crater after spikes. So these aren't training failures —
they're outlier batches where the sampled actions happened to
correspond to high-information regions, and PPO's safety
mechanisms work as advertised.

## What to do next — head-to-head bot-vs-bot

CPU-9 Fox saturates win-rate. The discriminating eval is
**RLVR-policy vs BC-baseline ditto**: two MIMIC instances in one
Dolphin, both Fox, FD. Win-rate has a meaningful range (50% baseline,
not 100%); discrimination scales with effect size.

Plan:
1. Build `rlvr/eval/head_to_head_winrate.py` — two-policy single-Dolphin
   driver based on the existing `tools/head_to_head.py` watchable.
2. Smoke: 3 matches, `DISPLAY=:0` (real X server, watchable),
   regular `emulator/` (not FFW), Vulkan backend. Sanity-check the
   adapter works.
3. Bulk run: N=200 matches per matchup (BC vs l_cancel, BC vs
   combo_extend), alternating ports to eliminate port-handedness
   bias. Wilson 95% CI half-width at N=200 is ±7pp — good enough
   to resolve a 10pp effect.
4. Stage variance check: small follow-up run on
   Battlefield + YS + DL64 + FoD if FD result is meaningful.

Cost: ~80s/match realtime × N=200 = ~4.5hr/matchup × 2 = ~9hr.
Or FFW-headless ~4hr total but unwatchable. User has flagged
watching is qualitatively useful, so default to watchable-realtime
for the bulk run.

## Notes for future me

- **Eval saturation against CPU-9 Fox** is a problem for any task
  that incrementally improves an already-strong BC. Default eval
  going forward should be head-to-head, not vs-CPU.
- **Stock-margin > win-rate** as a sensitivity metric when the
  policy dominates. Average stocks remaining (or %-of-stocks-taken)
  separates policies even when both are at 100%.
- **VR-by-VR opponent matching** matters. shield_escape needs a
  shield-pressuring opp, edgeguard needs an opp that goes off-stage
  often, combo_extend works against any opp that lets you land hits.
  Don't assume one fixed eval opponent works for the full VR list.
- **Per-task variance from one seed** is unbounded. Don't draw firm
  conclusions from N=1 PPO seed comparisons. Either run multiple
  seeds per task or use head-to-head with large N.
