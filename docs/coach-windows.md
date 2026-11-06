# Coach on Windows: what it would take (feasibility, 2026-11-05)

Question: can a Windows player run the coach locally (the native sim + the server), instead of a
hosted server? Findings from reading melee-unlocked at our pinned commit; nothing was built.

## What already exists upstream

- **The game code builds on Windows.** The decomp game library (`sourceport/game`) needs GCC 14
  (`scalar_storage_order` keeps the disc's structs big-endian; no other compiler has it), and
  upstream already supports building it for Windows with MinGW-w64 GCC 14
  (`sourceport/cmake/mingw-w64-x86_64.cmake`); an MSVC host loads it (`port/app/source_host.cpp`,
  target `melee_source`).
- **The runtime is Windows-native.** `port/runtime` is written against Win32 (VirtualAlloc etc.);
  the Linux port wraps it in a compatibility layer (`port/linux/compat/windows.h`). Console RAM at
  0x80000000 is reserved with VirtualAlloc on Windows.

## What we would have to port (ours, not upstream's)

1. **The headless replay/env host.** Our `--replay`/`--env` host with the MIMIC wire protocol
   (pads in, Slippi events out, the lab's place command) is `port/linux/replay_host.cpp`
   (404 lines; POSIX `read`/`write` on inherited pipe fds). On Windows it becomes a console
   executable built with the MSVC host runtime (like `melee_source`) that loads the MinGW-built game
   library, with the same protocol on stdin/stdout in binary mode. The game-side hooks
   (`mu_replay.c`, `mu_lab_place`) are in the game library and carry over.
2. **The Python side of the pipe.** `rlvr/sim/melee_sim.py` uses `os.pipe` + `pass_fds` and
   `select.select` on pipes, which are POSIX-only. On Windows: the child's stdin/stdout and a reader
   thread with a timeout.
3. **A toolchain.** MinGW-w64 GCC **14** for the game library (Ubuntu's `gcc-mingw-w64` is 13.2, so
   not that one; WinLibs/MSYS2 ship 14) and MSVC 2022 for the host, or MinGW for both if the host
   code allows it (untested).
4. **Exactness, re-verified.** The sim is useful only because it replays current-Slippi games
   frame-exactly (465/466 on Linux). The Windows build must pass the same check
   (`tools/sim_replay_check.py`) before branching is trusted there; MSVC vs GCC host differences are
   the main risk.

Everything else is portable already: the server is standard-library Python, torch/peppi/libmelee
have Windows wheels, the page runs in any modern browser, and the ISO is found through Slippi
Launcher's settings on Windows too (`serve.py` already checks `%APPDATA%/Slippi Launcher/Settings`
and the Mac location as well as Linux's).

## A cheaper route: WSL2

Windows 11 runs the Linux build unchanged under WSL2, with the GPU available to CUDA inside it, and
the Windows browser reaches `localhost:8765` in WSL2 directly. It needs `wsl --install` (one reboot)
and running our Linux setup inside it, so it suits technical players, not the average one.

## Mac

No port: the game library requires an x86-64 target, and there is no macOS host. Not feasible
without upstream work.

## Bottom line

A native Windows build is a port of our ~400-line host plus the Python pipe layer, on top of
upstream's existing Windows build of the game, then an exactness check. The hosted server remains
the only route with nothing to install for any OS; WSL2 works today for technical Windows users.
