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

Decision: **kept local, bundled in-repo** (patch + pinned commit + build
script), not upstreamed — per owner preference.

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
