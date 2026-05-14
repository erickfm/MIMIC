# 2026-05-14b — combo-extension rebuild, web HUD, PPO perf, GRPO future work

`research-notes-2026-05-14.md` covered the first 3-way PPO comparison
and the discovery that win-rate vs CPU-9 saturates. The second session
that day did three things:

1. **Rebuilt the combo-extension task** to be a faithful streaming
   port of `slippistats.stats.combo_computer` (plus a `n_moves >= 2`
   guard that slippistats doesn't have).
2. **Replaced the pygame HUD with a stdlib-HTTP + SSE web dashboard**
   that's actually OBS-stream-friendly and design-coherent.
3. **Moved the reference-model forward pass off the per-frame
   critical path** to recover ~5 ms/frame during open episodes.

Plus a future-work writeup of GRPO via Dolphin savestates.

## Combo detection — what changed

The original `ComboExtendOnlineTask` was a thin sketch ported from
`value/derived_features.py`. Watching it run during bvb training
surfaced a sequence of real bugs. The full corrected logic now lives
in `rlvr/online/tasks/combo_extend_online.py` and is the faithful
slippistats port for live PlayerState streams.

### Trigger correctness

- **Pre-hit start percent.** `should_start` snapshots `prev.percent`
  (the frame before the punish-state transition), not `curr.percent`
  (the frame opp's percent has already gone up). Without this, a
  single-hit punish reads damage=0 because curr==end. Matches
  slippistats's `start_percent = prev_opponent_frame.post.percent`.
- **Damage-or-grab safeguard.** A trigger only counts as an episode
  start if opp actually took damage (curr.percent > prev.percent) OR
  was put into a grab/capture state. Shield-stun edge cases or
  hitstun-flag glitches that put opp into a damage-like state
  without taking damage no longer open episodes.

### Punish-state set (full slippistats parity)

`_in_punish_state` now matches `combo_computer.py:260-277`:

- DAMAGE (75-91)
- CAPTURE (223-232)
- COMMAND_GRAB ranges (266-304, 327-338) — character-specific grabs
- THROWN (239-243) — the throw animations themselves
- hitstun_frames_left > 0
- hitlag_left > 0
- DYING (0-10) — keeps the kill blow inside the window so the
  stock decrement registers in-episode
- DOWN (183-198) — knockdown / missed tech
- TECH (199-204) — tech in place / roll / jump
- DODGE (233-236) — spot dodge / roll / airdodge
- GUARD (178-182) — shielding
- GUARD_BREAK (205-211) — shield broken
- FALL_SPECIAL (35-37) — post-up-B helpless
- LEDGE_ACTION (252-263) — at the ledge / ledge option
- off_stage flag

The K-frame gap (`COMBO_END_GAP`) is now 45 frames (was 20), matching
slippistats's `COMBO_LENIENCY = 45`.

### Peak-percent scoring

Damage is computed against `max(opp.percent observed during the
episode) - start_percent`, not `end_opp.percent - start_percent`.
Two reasons:

1. When a combo kills, opp's percent resets to 0 on respawn. Without
   peak tracking, a real kill reads damage=0 (end_pct=0, start_pct>0).
2. Without it we'd want a flat stock-taken bonus, but that rewards
   opp self-destructs that happen mid-window for free. Peak-based
   scoring handles kills naturally: a 0-to-death drives peak to
   ~130%, clipped to 80 → reward ≈ 1.0. A 0% SD with no damage from
   us leaves peak at 0 → reward 0.

The slippistats analog is `combo.current_percent = post.percent`
per frame, updated only when `did_lose_stock` is False.

### `n_moves >= 2` (the actual new contribution)

Slippistats has a "moves" abstraction (`MoveLanded`) — hits sharing
the same bot action_state are one move. Their batch combo-computer
records this for accounting but doesn't gate combo VALIDITY on it.

We need to gate, because RLVR rewards behavior and a single Fox dair
drill = 6-8 percent increases. Without a gate, the bot could
reward-hack via "spam dair as the only move." So we count **moves**,
not hits, and require ≥2 for a real combo classification.

A move advances when:
- bot's action_state changes from one hit to the next, OR
- bot's action_frame (state_age) decreases — same action restarted
  (e.g. jab1 → cancel → jab1 again)

`compute_outcome` returns `single_hit` with 0 reward when `n_moves <
2`. `test_outcome_multihit_same_move_is_single_hit` specifically
guards against the dair-spam case.

### What we deliberately skip

Slippistats checks we don't implement:

- `is_maybe_juggled` — needs stage geometry + position. Marginal
  effect; combos with explicit air-DAMAGE actions cover the same
  cases.
- `is_upb_lag` — needs prev_state diff; rare effect; FALL_SPECIAL
  range already covers most of it.

### Late additions to the port (after observing 6-8s episodes)

Two MORE slippistats-parity bugs surfaced after the user audited
the port against the reference. Both real, both shipped:

- **`player_did_lose_stock` termination.** slippistats
  `combo_computer.py:295`:
  `if reset_counter > COMBO_LENIENCY or player_did_lose_stock:
  should_terminate = True`. If the BOT dies mid-combo (opp
  counter-KOs us), slippistats terminates the combo immediately.
  My port had no equivalent — the episode stayed open until the
  K-gap fired even though the bot had literally died. Now I track
  `_episode_start_self_stock` at `should_start` and terminate when
  `self_curr.stock < start`.
- **Command-grab in start safeguard.** slippistats's start check
  (line 191) is `is_damaged OR is_grabbed OR is_cmd_grabbed OR
  is_in_hitstun`. My damage-or-grab safeguard checked only the
  regular CAPTURE range (223-232), missing both command-grab
  ranges (266-304 and 327-338). Opp entering Bowser side-B / Kirby
  inhale wouldn't start a combo. Fixed by expanding the safeguard.

These two fixes brought the port to true parity with slippistats
on combo START + combo TERMINATION conditions (the keep-alive set
was already correct). 22/22 tests pass.

Also: slippistats has **NO frame-count hard cap**. Their three
termination conditions are `opp_did_lose_stock`, K-gap, and
`player_did_lose_stock`. My earlier addition of a 360-frame safety
cap (`MAX_EPISODE_FRAMES`) was a deviation from the reference and
was reverted.

The 6-8s episode durations I observed in the log ARE legitimate
under slippistats's permissive semantics — combos that include
off-stage chase + edgeguard sequences naturally last that long.
Not a bug; intentional design of the reference.

### Note for future slippistats ports

We kept catching missed cases days/hours after writing the port.
The pattern: I'd port the "obvious" predicates and miss the more
specialized ones that didn't fit my mental model of "what a combo
is." For streaming versions of any other slippistats logic (the
upcoming edgeguard / shield-escape / pressure VRs are all
candidates), we should default to **paranoid faithfulness**:

1. Read the WHOLE function in `slippistats/stats/*` we're porting,
   not just the parts that look relevant.
2. Enumerate every conditional branch (`if X: ...`) and check
   that our port has an equivalent.
3. Enumerate every state predicate referenced (`is_X(...)`) and
   verify range constants.
4. Compare termination/start/keep-alive sets line-by-line.

This is encoded into `CLAUDE.md` under the new "Porting slippistats
logic" section so the next port doesn't repeat the same mistakes.

## The web HUD

`rlvr/eval/training_web/server.py` replaces `rlvr/eval/training_hud.py`
(pygame) as the auto-launched dashboard when training runs in
viewable mode.

- **stdlib only** — `http.server.ThreadingHTTPServer` + a tiny SSE
  endpoint at `/events`. No FastAPI, no npm, no build step.
- Single HTML file embedded as a Python string. Inline CSS + JS.
- `LogTailer` thread reads new lines from the training log, parses
  the same `EVT_EP_OPEN` / `EVT_EP_TICK` / `EVT_EP` / `EVT_MATCH_END`
  / `update=` patterns the pygame HUD parsed, broadcasts each as a
  JSON event over SSE.
- On new connection, server sends a `snapshot` of the current state
  first so reconnects / late-joiners aren't blank.
- Frontend renders an SVG live trajectory plot, viridis-colored
  recent-extensions bars, dark monochrome palette, Inter + JetBrains
  Mono from Google Fonts.

New event we needed to emit: **`EVT_MATCH_END`**. Existed nowhere
before. `DolphinActor._step_one_frame` now snapshots the latest
in-game stocks every frame, and on the IN_GAME → menu transition
emits:

    EVT_MATCH_END result=<win|loss|draw> trainee_stocks=X opp_stocks=Y

The web HUD's footer tallies wins / losses from these. Trainee
"win" = trainee_stocks > opp_stocks at the last in-game frame
before transition.

### Costume diff

`ActorConfig.trainee_costume = 0` (white Fox), `opponent_costume =
3` (green Fox) — the menu helper passes these through. So when
watching the game on screen the user can tell P1 (trainee) from P2
(frozen opponent) at a glance.

## Performance — ref-forward off the critical path

Original per-in-game-frame work during an open episode:

- P1 trainee policy forward (~5 ms) — drives the trainee's controller, critical-path
- P1 reference model forward (~5 ms) — for KL term in PPO, NOT a controller driver
- P2 frozen opponent policy forward (~5 ms) — drives the opp's controller, critical-path
- snapshot_context, FrameRecord append, state machine — ~3 ms

Total ~18 ms vs the 16.67 ms frame budget → the actor drops to ~55fps
during open windows.

The ref-model forward was the only one whose output doesn't drive a
controller — it's just a `logprob_ref` consumed later by PPO's KL
penalty. Since ref-model weights are FROZEN (never updated), and we
already snapshot the full input context per FrameRecord, the ref
forward is **deterministic** whether run live or run later.

Moved: ref forward runs inside `ppo_update`'s minibatch loop,
re-batched. The collect path no longer calls `ref_model` at all.

Net:
- Collect path: ~5 ms / open-episode-frame saved
- PPO step: ~4 s slower (one ref pass per minibatch, batched)
- Wall-clock: PPO step is rare (~1/7 min); the per-frame saving is
  continuous. Net win.
- Correctness: bit-identical PPO loss values, by construction.

`rlvr/online/ppo.py:79` now takes an optional `ref_model` and falls
back to cached `FrameRecord.logprob_ref` when not provided.
`rlvr/online/loop.py:200` passes it.

## Bot-vs-bot setup (replaces CPU-9 opponent)

CPU-9 Fox was the original training opponent. It saturates win-rate
at 100% and doesn't punish reward-hacking. Earlier today's note
already covered this; codified at 2026-05-14 in
`rlvr/online/dolphin_actor.py:ActorConfig.opponent_ckpt`. When set,
the actor loads a second frozen MIMIC checkpoint, builds its own
PlayerRunner / context, drives the cpu_port from that. Menu helper
picks `cpu_level=0` for both ports.

Defaults: P1 trainee in default Fox costume (0), P2 frozen-BC in
green Fox costume (3), both `temperature=1.0`.

## Future work — GRPO via Dolphin savestates

Current loss: vanilla online PPO with batch-relative advantage
normalization (`advantage = reward - batch_mean`, optionally over
batch std).

GRPO (DeepSeek/Qwen style) would be a strict upgrade for our sparse
reward setting: from each "punish window opens" state, sample N
trajectories (different RNG seeds, same policy), use group-relative
advantage `reward_i - mean(N siblings)`. Lower variance, attributable
to actions taken from a known shared starting state.

Required pieces:

- **Savestate API.** The Exi-AI Dolphin build we already use
  (`emulator_ffw/`) supports frame-perfect savestates via the EXI
  device that's wired into Slippi's rollback netcode. libmelee
  doesn't expose this today; we'd add a thin wrapper that issues the
  savestate command and restores from a buffer. ~50 lines.
- **Collect loop refactor.** Currently a single loop:
  `step → maybe_open_window → maybe_close_window → repeat`. GRPO
  needs: `step → window_opens → savestate → for n in range(N):
  load_state, run_to_window_close, record_reward → resume`.
- **Policy context reset.** The trainee's 256-frame input buffer
  needs to be saved at savestate time and restored each rollout.
  Already snapshotted per FrameRecord; just need to wire reset
  during rollout fan-out.

Costs:

- 2-4× slower per "useful episode collected" depending on N.
- Implementation complexity: high (state machine with branching).
- Untested in our framework — Exi-AI savestate semantics need a
  smoke test (does percent / animation / stocks / RNG round-trip?).

Why wait:

1. We don't yet have a CLEAN win-rate gain from vanilla online PPO.
   Baseline first; method upgrade second.
2. Adds debugging surface; better to land on a stable foundation.
3. 2-4× slowdown is real; today's per-update wall time is already
   the dominant cost.

Pick this up after we see a clear positive head-to-head result from
vanilla PPO. If the result is borderline, GRPO's variance reduction
is the natural next move.

## HUD micro-changes after the initial ship

- Combos + updates counters added to the footer
  (`combos X · updates Y`). Earlier version had `windows X` but
  that counts null-closed episodes (single_hit / sub_threshold)
  which is noise; replaced with the PPO update count for a
  meaningful training-progress signal alongside combos.
- Recent-extensions list now shows each combo's **duration in
  seconds** (close_frame - open_frame, /60) instead of seconds-
  since-it-happened. "How long was the combo?" is more useful at
  a glance than "how recently did it close?".

## Files touched today

- `rlvr/online/tasks/combo_extend_online.py` — task rebuild
- `rlvr/tests/test_combo_extend_online_fixtures.py` — 22 tests, full
  slippistats parity, dair-spam guard
- `rlvr/online/dolphin_actor.py` — EVT_EP / EVT_EP_OPEN / EVT_EP_TICK
  / EVT_MATCH_END emission, costume support, ref-forward removal
- `rlvr/online/ppo.py` — optional `ref_model` arg, in-minibatch ref
  forward
- `rlvr/online/loop.py` — wires `ref_model` through to PPO, auto-
  launches the web HUD
- `rlvr/state/gamestate.py`, `rlvr/state/libmelee_adapter.py` —
  `action_frame` field added (needed for move counting)
- `rlvr/eval/training_web/server.py` — new web HUD
- `rlvr/eval/training_hud.py` — pygame HUD kept as manual fallback
