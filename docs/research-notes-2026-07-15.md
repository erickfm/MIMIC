# Research notes — 2026-07-15 (evening) → 07-16

## Savestate harness: Dolphin save/load under Python control — WORKING

Motivation (from the L-cancel postmortem, research-notes-2026-07-14): the
next RLVR objectives need (a) miss-targeted state drilling — batch size
scales ~1/miss-rate without it — and (b) rollout state distributions that
cover deployment states (the on-policy KL leash only constrains behavior
on visited states; vs-CPU rollouts never visit the states where h2h
strength lives). Both need emulator savestates under Python control.

### Recon: no existing control surface

- libmelee: no savestate API at all; its only Dolphin control is the
  controller named-pipe (`PRESS/RELEASE/SET/FLUSH`) + launch-time gecko
  config.
- Neither Dolphin build has a savestate CLI flag; no headless hotkey path.
- The machinery exists inside the ExiAI fork: `SlippiSavestate` (rollback)
  + `SaveState.asm`/`LoadState.asm` gecko, driven via `CEXISlippi` command
  bytes 0xB1/0xB2 — but those are game→emulator (rollback ASM), not
  externally callable. Confirmed against vladfi1/slippi-Ishiiruka source.

### The patch (bundled: `tools/patches/slippi-ishiiruka-savestate.patch`)

~40 lines on vladfi1/slippi-Ishiiruka @ `exi-ai-rebase`
(commit pinned in `tools/patches/slippi-ishiiruka-commit.txt`):

1. **`Pipes.cpp`**: new pipe verbs `SAVESTATE <path>` / `LOADSTATE <path>`
   on the existing bot-controller pipe. Ops are queued as **host jobs** —
   `State::SaveAs/LoadAs` use `Core::PauseAndLock` and deadlock if called
   from the CPU thread that polls the pipe; `MainNoGUI`'s platform loop
   dispatches host jobs.
2. **`MainNoGUI.cpp`**: `-i/--slippi-input` (playback comm file) and
   `--cout` (launcher `[CURRENT_FRAME]` stdout protocol) — both existed
   only in the WX GUI build.

Build recipe: `tools/build_savestate_dolphin.sh` (clone, patch, build
netplay + playback variants, install `emulator_ss/` + `emulator_pb/`,
both gitignored). Gotchas encoded in the script: cmake<4 required;
bundled libusb must build WITHOUT udev
(`-DCMAKE_DISABLE_FIND_PACKAGE_Libudev=TRUE` — its cmake glue mislinks
netlink symbols when libudev-dev is present); the deps are
libudev-dev + libasound2-dev + cargo.

Decision: **bundled in-repo + published as an unmaintained fork**
(`erickfm/slippi-Ishiiruka` branch `mimic-savestates`, commit `b9a0cfa`) —
no upstream PR, no maintenance intent; the fork exists for reproducible
builds and provenance. The .patch file stays in-repo as backup.

### Python-side usage

```python
ctrl._write(f"SAVESTATE {path}\n"); ctrl.flush()
ctrl._write(f"LOADSTATE {path}\n"); ctrl.flush()
console = melee.Console(..., skip_rollback_frames=False)  # REQUIRED
```

`skip_rollback_frames=False` matters: libmelee's default silently drops
any frame ≤ the max seen (`console.py:972`), so a rewind is invisible —
this cost a debugging round; the emulator restore was verified working
(RAM frame counter 601→320) while libmelee showed nothing.

### Validation (all PASS)

| probe | result |
|---|---|
| realtime save→diverge→load | frame-perfect: saved f176 (x 5.79, y 13.06, FALLING) → resumed f177 (x 5.04, y 10.26, FALLING) = exactly one frame of gravity |
| FFW mode | rewind lands exactly on captured state |
| cross-process | fresh Dolphin loads another process's .sav; 300 coherent frames after |
| cross-build | playback-build .sav loads in netplay build (`ok=1`, no marker mismatch) |
| build faithfulness | patched rebuild vs stock `emulator_ffw`, same ckpt: L-cancel 95.2 vs 94.2 (z≈0.9), cstick 40.1 vs 39.5, m_len 6533 vs 6323 — canary-identical |

Operational facts:
- **Save-capture latency**: the host job fires 1–150 game frames after the
  pipe command (worse under FFW — game runs ~4× per wall-second). Fine for
  "harvest a state ~2 s before the event"; not frame-exact capture.
- States are ~90 MB uncompressed in memory, ~9–17 MB on disk.
- **Don't score .slp replays across loads** (duplicate frame spans);
  live-state scoring only — the avoidable-lag metric supports this.
- Debugging trap: concurrent probes fight over spectator port 51441 and
  cross-connect (libmelee silently watches the WRONG dolphin). One
  instance per port; pass distinct `slippi_port`.
- `LoadAs` silently refuses when `IsOnline()` (Slippi netplay) — local
  bot play is fine (no netplay client exists).

### Replay-seeded harvesting: NOT working yet

Goal: play back human .slp in the playback build, savestate at chosen
frames → seed library from deployment-distribution states. Status:
- Playback build compiles, boots, runs the game; `--cout` flag works
  (`shouldOutput=1` traced); comm-file parsing confirmed
  (`mode/replayPath/startFrame/endFrame`).
- **Blocker: the game boots to menus and idles — the playback gecko
  handshake (`CMD_IS_FILE_READY`) never fires**, so the replay never
  starts. The libmelee-cloned User config runs the game but not the
  playback flow; the Slippi Launcher's own playback config
  (`~/.config/SlippiPlayback`) crashes our differently-versioned binary
  at boot. The missing switch is somewhere between those two configs
  (gecko enablement / playback boot flow). To be resumed.

Meanwhile, **miss-harvesting from our own bot rollouts works today** with
`emulator_ss` (the states come from bot-vs-CPU games, port types restore
correctly) — sufficient for the L-cancel-style miss-drilling design.

### Strategic note

2-model RLVR (policy vs frozen BC) needs none of this to *start*: it runs
today, realtime, via `loop.py --opponent-ckpt` (that's the h2h rig).
Savestates buy efficiency (menu skip, mini-episodes, miss drilling) and
distribution coverage. The other unlock now available (we build the fork
from source): investigating the **FFW dual-pad infidelity** directly —
fixing it would give fast self-play, the throughput path for 2-model RLVR.

## 2026-07-16 (day): three parallel agents — all three landed

Owner directive: destroy-and-learn RLVR is acceptable; balance deferred to
multi-RLVR with dual rollouts. That made dual-rollout infrastructure the
priority. Three agents ran concurrently (disjoint build trees + port
ranges):

### 1. FFW dual-pad infidelity: ROOT-CAUSED AND FIXED (fork `241c13f`)

The 2026-07-02 "FFW self-play is broken" finding (~23% L-cancel, 4×-short
matches with 2 EXI pads) was a **frame-identity race in the blocking-pipe
protocol**, not the FFW gecko or EXI overwrite:

- libmelee's `console.step()` writes a bare keep-alive FLUSH to every
  controller pipe every step (console.py:819) besides the bot's real
  commands+FLUSH burst. `PipeDevice::UpdateInput`'s per-frame blocking
  wait ended on the FIRST flush seen — a leftover keep-alive satisfies
  the wait instantly, the 0xD9 pad snapshot serves stale pads, and the
  real burst lands a frame late or is skipped.
- Why only dual-pad FFW: single-pad vs-CPU is saved by device-iteration
  order (blocks on the silent CPU-port pipe whose only traffic is the
  keep-alive written last = accidental serialization); realtime is saved
  by the 16.7 ms frame giving SI polls time to drain the keep-alive. At
  FFW's ~1 ms frames the race fires every frame.
- Fix: per-port player-type tracking from the Slippi GAME_START event +
  a frame-synced combined wait across human-port pipes with a per-pipe
  flush ledger (keep-alive tails never count as the frame barrier).
  Legacy path byte-identical for CPU ports/menus/non-blocking.
- Validation (no-GPU instruments): input-echo probe 57.8%/39.5% on-time →
  **100%/100%** (n=4572/port) at 11–15× realtime; scripted 1-frame
  L-cancel metronome 87% → **100%**. NN self-play validation below.
- The stock `emulator_ffw/` AppImage still has the bug — self-play must
  use `emulator_ss`. Caveat: send SAVESTATE/LOADSTATE verbs in the same
  flush as that frame's inputs, or accept a possible 1-frame skew.

### 2. Miss-drilling harness v1 (MIMIC `4ce353e`)

`rlvr/online/savestate_util.py` + `miss_harvest.py` + `drill_loop.py`
(+ additive `OnlinePPOConfig.use_metadata_advantage`). Harvest: rolling
6-slot savestate ring + live avoidable-lag tracking; on miss, keep the
slot ≥180 frames pre-landing + the policy context window + JSON sidecar.
Drill: LOADSTATE + context restore, N=8 rollouts per state =
matched-context GRPO group, live scoring, PPO update. Smoke-validated:
12 states harvested (7.6% miss vs CPU-9), 96 loads, 3 updates,
non-degenerate gradients, policy plays normally after. **LOADSTATE
latency p50 0.19 s; restored frame == context capture frame exactly.**
Throughput ~1.6 s per known-miss-context episode vs ~25–190 s via full
matches (≈15× enrichment at 6% miss rate, growing as 1/miss-rate).
v2 levers: bigger N (9/12 groups were zero-variance at N=8), library
curation toward mixed-outcome states, mixing with regular rollouts.

### 3. Replay-seeded harvesting: WORKING (MIMIC `0c97823`)

The menu-idle root cause was one JSON key: the playback comm file wants
**`"replay"`, not `"replayPath"`** (`SlippiReplayComm::loadFile` reads
`res.value("replay","")`; an empty path means `CMD_IS_FILE_READY`
answers 0 forever). Config-only fix; gecko injection and the
libmelee-style User dir were fine all along (`enablecheats` is not even
read by this fork; IniFile keys are case-insensitive). Verified
end-to-end: `[CURRENT_FRAME] 1200→16541` headless in <60 s with seek;
frame-5000 savestate harvested from a human master replay; cross-loaded
into `emulator_ss` matching peppi ground truth (x −41.13 vs −41.21,
same stocks/percent/action). Recipe: `docs/playback-harvest-recipe.md`.
Design note: a restored human-game state brings two human-input ports —
the shell match must boot with matching stage/char layout, and port 2
needs a driver (frozen BC) — which the dual-pad fix now makes possible
under FFW.

### NN self-play FFW validation (the original symptom, retested)

`tools/play.py`, AVG_mastfox vs AVG_mastfox, 6 matches, FFW + EXI on the
fixed `emulator_ss` binary: **port 1 = 94.9% (n=117), port 2 = 92.6%
(n=94) L-cancel** — squarely in the healthy band (broken mode: ~23%) —
with normal match lengths (mean 2.08 min; broken mode was 4× short), at
~2–3× realtime for two batch-1 models on one GPU. **FFW self-play is now
faithful. Dual-rollout RLVR (policy vs frozen BC, FFW) is unblocked.**

