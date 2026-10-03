# 2026-07-22 — The h2h campaign ran on the broken FFW build (and two bugs behind it)

## TL;DR

The 2026-07-20 all-character head-to-head campaign is **invalid**. It ran
bot-vs-bot matches on the **stock `emulator_ffw`** AppImage, which has a known
dual-pad keep-alive-FLUSH race (fixed 2026-07-16 in `emulator_ss`, fork
`241c13f`). Two EXI-injected controllers → both bots get stale pads → symmetric
input mistiming. Effect measured directly, same two checkpoints
(`hf_eval/bowser`, `hf_eval/fox-master`), only the emulator varied:

| emulator | L-cancel (bowser / fox) | avg match frames | outcome |
|---|---|---|---|
| realtime OGL (`emulator/`) | 79% / 90% | ~9,500 | Fox 5–1 |
| **`emulator_ss` (fixed FFW)** | **85% / 91%** | **7,956** | **Fox 5–0** |
| `emulator_ffw` (stock, buggy) | 13% / 33% | 2,907 | **Bowser 6–0** |

The stock build doesn't just add noise — it **inverts the winner**. The
campaign's headline ("bot-vs-bot Elo uncorrelated with CPU-9 win rate; heavies
top, fast chars bottom") is an artifact of the broken build: under symmetric
mistiming nobody L-cancels, everybody eats full landing lag, games are 3× short,
and slow heavyweights that don't need precise tech win the mush. On a faithful
build fox beats bowser, matching realtime.

## Two bugs, found in sequence

**Bug 1 — character selection with `--alternate-ports`.** ~13.5% of campaign
games (99/731 replays) came out as **unscheduled mirror matchups** (marth/marth,
gameandwatch/gameandwatch, …) or even 3-player replays — none were scheduled.
Cause: `tools/play.py` passed `autostart=True` to port_b's `menu_helper_simple`
unconditionally. After a port flip the previous match's coins are still down on
the OLD characters; the game's `ready_to_start` flag flips to 0 the instant both
coins are down *regardless of which character*, so port_b pressed START while a
port was mid-switch. Fix: gate autostart on **both physical ports being locked
onto their intended characters (coin down)** before allowing START (`_css_locked`
check in the CSS branch of `play.py`; SHEIK compared as ZELDA per the CSS
convention). Verified: 6 matches / 5 port flips → 0 wrong-character replays.

**Bug 2 — the broken emulator (the big one).** After fixing bug 1, clean
bowser-vs-fox FFW replays *still* read 13%/33% L-cancel vs 79%/90% realtime.
That's the stock-`emulator_ffw` dual-pad race. Fix: point the dual-pad tooling at
`emulator_ss/Binaries/dolphin-emu`. Re-verified with the canary above.

## Why this wasn't caught earlier — and what the canaries were for

**The canary caught it. I overrode it.** On the original campaign the L-cancel
canary printed `LCANCEL-LOW(mistimed?)` for every character. That is the alarm
firing correctly. The failure was human/agent judgment: the low reading was
rationalized as a "chaotic-brawl context artifact" and that rationalization was
written into memory (`project_h2h_vs_cpu9_uncorrelated`: *"Don't trust the
corpus-calibrated L-cancel canary in high-pressure h2h context; center-stick% is
the robust drop detector there"*). That note taught the wrong lesson and would
have perpetuated it.

Three compounding mistakes:

1. **A prior memory note already said not to do this.**
   `project_rlvr_ffw_unfaithful` states plainly: stock `emulator_ffw` is broken
   for dual-pad/self-play, **use `emulator_ss`**. The h2h tooling
   (`run_h2h_lane.py`) hardcoded `emulator_ffw` for bot-vs-bot anyway. The
   knowledge existed; it wasn't applied to the new tool.

2. **The canary fired and was dismissed** instead of being connected to that
   existing note — even though "L-cancel low in dual-pad FFW" is *exactly* the
   documented signature.

3. **The wrong sentinel was promoted.** The dual-pad bug serves *stale* pads
   (off by ~1 frame); it does **not** drop inputs to neutral. So center-stick%
   stays normal (31–61% across the campaign) and only *timing-sensitive* signals
   (L-cancel) move. center-stick is structurally blind to a mistiming bug — yet
   it was elevated to "the robust drop detector" while the canary that actually
   worked was demoted. **A green center-stick canary is not evidence of timing
   fidelity; only the L-cancel canary measures timing.**

## What was NOT affected

The **CPU-9 win-rate evals** used a CPU opponent = a *single* EXI-injected bot.
Single-injected-bot FFW is faithful even on the stock build (documented: 92%
L-cancel vs CPU). So the per-character CPU-9 numbers stand. Only the **bot-vs-bot
h2h campaign** is invalid.

The **all-character retrain + upload** (2026-07-19/20) is unaffected — training
never touches the emulator.

## Prevention (make it not happen again)

**The rule — a canary failure under FFW means FIX FFW until the canaries pass
UNDER FFW. Realtime is a diagnostic control, never the destination. We do not
fall back to slow.** FFW throughput is the whole point (RLVR rollouts, large
h2h/eval sweeps); retreating to realtime "because it's faithful" abandons the
throughput we need and is a surrender, not a fix. Use realtime (or a known-good
FFW build) only to prove the model itself is faithful and to give the canary a
target — then go make FFW match it. That is exactly what happened here: the fix
was `emulator_ss`, which restores 85–91% L-cancel *under FFW*, not a switch to
realtime for the real workload.

- **Discipline:** an `LCANCEL-LOW` canary result is **blocking**, not advisory,
  and it is a **bug in the FFW harness to be root-caused and fixed** (emulator
  build, dual-pad EXI race, keep-alive/flush ledger, blocking-input, gecko
  codeset) — then re-run the canary under FFW and require it to pass before
  trusting any strength number. Never explain a canary failure away with a
  "context artifact" story; never quietly accept a slower path.
- **Tooling:** both `run_h2h_lane.py` and `eval_cpu9_lane.sh` now point at
  `emulator_ss/Binaries/dolphin-emu`. No MIMIC tool references the stock
  `emulator_ffw` build for bot-vs-bot anymore.
- **Sentinel choice:** center-stick% detects *dropped/neutralized* inputs, not
  *mistimed* ones. For any FFW / dual-pad / timing question the L-cancel canary
  is the one that matters.
- **Memory:** `feedback_ffw_validity_nonnegotiable` records this rule.
  `project_h2h_vs_cpu9_uncorrelated` is retracted (campaign on the broken build).
  `project_inference_canaries` updated so the "center-stick is robust in h2h"
  line no longer stands unqualified.

## Rerun

Campaign re-run with both fixes (character-select gate + `emulator_ss`) supersedes
the 2026-07-20 results. Old `eval_results/h2h/` (stock-FFW) is discarded, not
aggregated.
