# Headless Slippi replay playback + savestate harvest (playback build)

2026-07-16. How to play back a human `.slp` headless in our patched playback
Dolphin (`slippi-Ishiiruka` branch `mimic-savestates`, build dir `build-pb/`)
and harvest a savestate at a chosen frame.

## Root cause of the "idles at menus" symptom

The comm file key for the replay path is **`replay`**, not `replayPath`.
`SlippiReplayComm::loadFile()` (Source/Core/Core/Slippi/SlippiReplayComm.cpp)
reads:

```cpp
commFileSettings.replayPath = res.value("replay", "");
```

With `"replayPath"` in the JSON, the path silently parses as empty,
`isNewReplay()` never returns true, the EXI device answers `CMD_IS_FILE_READY`
with 0 forever, and the game sits at the menus. Nothing else was wrong: the
gecko code handler installs fine (log shows `Patching MMIO access at 80001910/
80001e00`), `bootloader.gct` loads, and the game↔EXI handshake machinery all
work with a libmelee-cloned User dir.

Notes on things that were suspected but are NOT problems:

- `enablecheats` in Dolphin.ini is irrelevant: `SConfig` never reads
  `EnableCheats` from Dolphin.ini in this fork; `bEnableCheats` defaults to
  `true` (ConfigManager.h) and `BootManager` only reads the key from the
  *game* ini. Lowercase libmelee-written keys are fine anyway — IniFile keys
  are case-insensitive (`CaseInsensitiveStringCompare`).
- The Slippi EXI device is hardcoded into EXI Slot B
  (`m_EXIDevice[3] = {EXIDEVICE_NONE, EXIDEVICE_SLIPPI, EXIDEVICE_NONE}`);
  `SlotA = 255` in the ini is normal.
- Sys-dir resolution is exe-relative because `build-pb` was configured with
  `-DLINUX_LOCAL_DEV=true`; `Sys/codehandler.bin` and `Sys/bootloader.gct`
  next to the binary are found.
- For Melee NTSC the code handler installs only a bootloader GCT; the real
  gecko list is served by the EXI device (`CMD_GET_GECKO_CODES`,
  `prepareGeckoList()`), which in playback mode comes from
  `Sys/GameSettings/GALE01r2.ini` (+ the replay's own codes). So playback
  gecko injection *requires* the comm file to load a replay — another way the
  bad key manifests as "no codes appear to run".

## Working invocation

```bash
build-pb/Binaries/dolphin-emu-nogui \
  -i /abs/path/comm.json \
  -e /home/erick/projects/MIMIC/melee.iso \
  -u /abs/path/User \
  --cout
```

(`dolphin-emu` in that dir is a byte-identical copy of `dolphin-emu-nogui`.)
Runs headless with `gfxbackend = Null`; no Xvfb needed. Stdout emits the
launcher protocol: `[FILE_PATH]`, `[PLAYBACK_START_FRAME]`,
`[GAME_END_FRAME]`, `[PLAYBACK_END_FRAME]`, then one `[CURRENT_FRAME] N` per
frame starting at `startFrame`. Verified: startFrame 1200 → first line
`[CURRENT_FRAME] 1200`, streamed to game end 16541 in <60 s wall clock.

### comm.json

```json
{
  "mode": "normal",
  "replay": "/abs/path/to/game.slp",
  "startFrame": 1200,
  "commandId": "<any-unique-string>"
}
```

- **`replay`**, not `replayPath` (the whole bug).
- `commandId` must change (or the file's mtime must change) for the same
  replay to be replayed twice — `isNewReplay()` compares both.
- `startFrame` may be as low as -123 (`Slippi::GAME_FIRST_FRAME`).
- Optional: `endFrame`, `mode: "queue"` with a `queue: [{path, startFrame,
  endFrame}, ...]` array for batch harvesting.

### User dir (what actually matters)

A libmelee-cloned User dir works as-is. Required pieces:

- `Config/Dolphin.ini` — libmelee defaults fine; `gfxbackend = Null` for
  headless.
- `Config/GCPadNew.ini` — `[GCPad1] device = Pipe/0/slippibot1` binding, plus
  `Pipes/slippibot1` FIFO (mkfifo) — this is the channel for the
  `SAVESTATE <path>` / `LOADSTATE <path>` pipe verbs (commit `b9a0cfa`).
- `Config/Logger.ini` — optional. Gotchas if you want logs: the `Logs/` dir
  must already exist (FileLogListener won't create it); NoGUI never raises
  log levels above WARNING (NOTICE/ERROR still print); the `[Logs]` key for
  EXPANSIONINTERFACE is its short name `EXI`.

Plus the binary-side `Sys/` (set up by `build-linux.sh playback`):
`Sys/GameSettings/` replaced with `Data/PlaybackGeckoCodes/`,
`Sys/codehandler.bin`, `Sys/bootloader.gct`, and `portable.txt` next to the
binaries.

## Savestate harvest

`harvest_v3.py` (session scratchpad) does: write comm.json (startFrame =
target-300), launch playback, watch `[CURRENT_FRAME]`, send
`SAVESTATE <out.sav>` down the controller pipe at the target frame.

Verified run: target frame 5000 of
`data/raw_slp/fox_master_master/master-master-0005d89cc5b67088cf1f0ce0.slp`
→ `SAVESTATE` sent at frame 5000, `.sav` on disk by frame 5013,
14,779,918 bytes.

## Cross-load into the netplay/savestate build

`test_savestate_xproc.py --dolphin emulator_ss/Binaries/dolphin-emu
--state-file <sav> --slippi-port 522xx` boots a fresh vs-CPU match via
libmelee and sends `LOADSTATE` at in-game frame ~120.

Verified (2026-07-16, frame-5000 harvest above): after `LOADSTATE` the
observed stream jumped from in-game frame 13 to **5013** (exactly the frame
the harvester's save landed on), with Fox at `x=-41.13, y=0.01, 0%,
3 stocks, SHIELD_START` vs replay ground truth at frame 5013 of
`x=-41.21, y=0.01, 0%, 3 stocks, guard` (peppi), and 300 post-load frames
streamed coherently → PASS. Savestates harvested from human replays in the
playback build cross-load into the netplay/savestate build.

Caveat: the restored match inherits the *replay's* entities (2 human-input
ports); the fresh match it was loaded over was P1-human/P2-CPU on FD. The
RL harness should boot its shell match with the same stage/char layout as
the source replay.
