# Coach: how it is delivered (decision, 2026-11-04)

## Decision

A web app: the player's browser draws everything; a server runs the game simulator and the models.

- **Browser:** library, replay view, timeline, review, lab. Replays and the 3D picture are drawn
  by [melee-3d-viewer](https://github.com/briandle00/melee-3d-viewer) (MIT, a fork of Slippi
  Lab) in an iframe. It draws with the game's own models, read from the player's own disc, and
  already overlays "ghost" replays frame for frame. The player picks their ISO once; the page
  reads the files it needs from it locally. The ISO never leaves their machine.
- **Server:** `tools/coach/serve.py` (Python, standard library HTTP). It holds the models on the
  GPU and runs the native sim to branch from real games (`tools/coach/branch.py`). It receives
  only the one .slp being analysed and returns branch .slp files and numbers.
- **Now:** the server runs on this machine and the app opens at `http://localhost:8765`.
  **Later:** the same server on a hosted GPU box makes it a website with nothing to install. That
  step needs the owner's decision (account, cost, and running the game server-side for others).

## Options compared

| | Browser only (sim in WASM, model in WebGPU) | Desktop app (Electron/Tauri + bundled sim) | **Web app + server (chosen)** |
|---|---|---|---|
| Setup for a player | open a page, pick ISO | download ~1–2 GB installer, pick ISO | open a page, pick ISO |
| Windows / Mac / Linux | all (WebGPU: Chrome/Edge; Safari/Firefox partial) | Win + Linux; **no Mac port** of the sim | all |
| Time for 8 rollouts × 5 s | not measured; model on a laptop GPU via WebGPU, likely tens of s | **~2 min on CPU** (25 ms per decision, measured); fast only with an NVIDIA GPU and a bundled CUDA torch | GPU-batched: measured after the Fox run frees the GPU |
| Exact replay | needs a WASM sim that is exact on current Slippi (melee-sim-light claims it; unverified here) | yes (our sim: 465/466 games exact) | yes (our sim) |
| Legal | ships compiled decomp code to every user | ships compiled decomp code to every user | ships none; game runs only on the server |
| Cost | none | none | a GPU server |
| Build / maintain | port the whole inference pipeline (features, normalization, cond-heads sampling) to JS; sim license unknown | per-OS packaging, CUDA, sim builds | one Python path, already exists |

## Evidence

- **Our sim (melee-unlocked) can't go to WASM as is.** The game code needs GCC's
  `scalar_storage_order` (big-endian structs, 169 uses in headers; Clang, and so Emscripten, has
  no equivalent), and the host maps console RAM at the fixed address 0x80000000.
- **A WASM sim does exist:** kyhavlov/melee-sim-light (decomp-based C, a WASM build used by
  melee-3d-viewer's GX page, ~120k fps per core batched natively). Its repository has **no
  license**, so it can't be redistributed, and its exactness on our replays is unmeasured. It is
  the route to a browser-only version later.
- **Windows:** melee-unlocked ships a win64 build (mingw/MSVC); **Mac:** no port. What our sim would
  need on Windows: see "Windows (feasibility)" below.
- **Model on CPU** (fox M RL policy, 8 threads, while the CPU also ran 8 sims): 24 ms per decision
  (batch 1); batching does not help (batch 16: 27 ms per item). One rollout of 300 frames with
  both sides played by models = 600 decisions ≈ 15 s; 8 rollouts ≈ 2 min.
- **Sim speed:** ~2,000 frames/s per process with rendering, faster without (online RNG). A
  prefix replay to frame 5,000 takes ~2 s; the K rollouts' prefixes run in parallel.
- **Viewers:** Slippi Lab (MIT) draws traced 2D line art from animation zips derived from game
  files. melee-3d-viewer (MIT, active, 2026-10) reads the player's own disc files and draws the
  real models; its embed page takes `load`, `seek`, `play`, `pause`, `range` and `ghosts`
  messages (`src/embed.tsx`). Nothing existing does model-based branching on replays; Slippi
  Lab's Training Mode savestate export is the closest (it hands control to the player in Dolphin).
- **Opponent realism** (`tools/coach/realism_check.py`, 2026-11-05, CPU, k=16): at 60 random real
  decisions per player, the player's real inputs are forced over the decision's stretch and the
  opponent is the master model. A control (both sides' real inputs forced) reproduced the real stretch
  exactly in 120/120 decisions and scores the real outcome through the same sim + V pipeline. If the
  model opponent were indistinguishable from the real one, the real outcome's rank among the 16 tries
  would be uniform (mean 0.5 ± 0.037):

  | player (opponents) | mean rank | rank quintiles | KS (5% crit) | tries − real | real reaction among tries |
  |---|---|---|---|---|---|
  | WAVE#666 Falco (mostly Stuckey's Falco) | 0.566 | 7 10 19 7 17 | 0.150 (0.176) | −1.2 ± 0.6 | 77% of decisions, 50% of tries |
  | STUC#911 Fox (WAVE's Falco, BILL's Luigi) | 0.503 | 9 14 17 10 10 | 0.117 (0.176) | −0.8 ± 1.2 | 85%, 48% |

  Neither rejects uniformity. In the owner's games the model opponent is slightly tougher than the
  real ones (the real outcome sits about 1 point above the tries' mean, ~2 SE; 0.62 mean rank on the 23
  decisions whose tries spread by more than 2 points), so the coach's numbers may lean slightly
  pessimistic there. Nothing in the app was changed on this basis. Single run, 60 decisions each.

## Windows (feasibility, 2026-11-05; read from the code, nothing built)

melee-unlocked is a Windows project first (its release is win64). The parts:

- **The game library** builds on Windows already: upstream compiles it with MinGW-w64 GCC 14
  (`sourceport/cmake/mingw-w64-x86_64.cmake`; GCC because of `scalar_storage_order`). Our two small
  game-side changes (`sourceport/game/shim/mu_replay*.c`, 68 lines) are plain C and come along.
- **The host is the work.** Our sim runs the Linux host (`port/linux`) plus our replay/env host (404
  lines, `replay_host.cpp`). It is Linux-only in a few places: `dlopen`/`dlsym` for the game library,
  `mmap` of console RAM at the fixed address 0x80000000, and reads/writes on pipe file descriptors
  it inherits. Two ways: port those calls (LoadLibrary/GetProcAddress, VirtualAlloc at that address,
  pipe HANDLEs), or move our replay/env mode into upstream's Windows host (`port/app/main.cpp`, MSVC,
  which already has `--headless --fast --frames`).
- **Python side:** `rlvr/sim/melee_sim.py` talks to the sim over `os.pipe` + `pass_fds` + `select`,
  none of which work with pipes on Windows; it would use the child's stdin/stdout with a reader
  thread instead.
- **Exactness must be re-measured:** a different compiler build (MinGW, Windows CRT math) could
  change float results; `tools/sim_replay_check.py` on the same replay set (465/466 exact on Linux)
  decides it.
- **Nothing can be shipped built:** like on Linux, the sim holds decompiled game code, so each player
  builds it on their machine against their own ISO. Upstream's `build.bat` does that with winget
  (Python, CMake, VS 2022 Build Tools; their notes say 5–40 minutes the first time).

**Smallest route: WSL2.** The Linux server and sim run unchanged inside WSL2 (CUDA works there with
NVIDIA's Windows driver), and the Windows browser opens `http://localhost:8765` (WSL2 forwards
localhost). The one code change: from WSL, find Slippi Launcher's settings, replays and ISO under
`/mnt/c/Users/<name>/` (`slippi_launcher_settings`, `default_replay_dir` in `serve.py`) and convert
its Windows paths. Cost for the player: `wsl --install` (a reboot) and our setup script, so
not simple enough for an average player. A native build is the real answer for them; the hosted
server avoids the question entirely.

## Open (owner)

- Hosting: which box, and whether to run the game server-side for other players.
- melee-sim-light's license, if a browser-only version is wanted later.

## Status (2026-11-04, evening)

Runs locally: `tools/coach/coach.sh` (starts `tools/coach/serve.py`, opens http://localhost:8765).
Tested end to end in headless Chrome on the owner's replays (WAVE#666 Falco; STUC#911 Fox).

| Feature | Where | Measured |
|---|---|---|
| Setup: none when Slippi Launcher is installed. The server reads Slippi Launcher's ISO (its Settings `isoPath`) and serves the viewer's game files from it on localhost; replay folder and connect code detected. Fallback without Slippi: pick an ISO in the browser (kept in IndexedDB) | `serve.py` `find_iso`/`/gamefiles/`, `web/iso.js` | fresh browser opens straight to the library |
| Library | `/api/games` | games with you in them; unsupported ones say why |
| Replay view: real 3D game (melee-3d-viewer, MIT, in `web/viewer`, without its GX simulator), timeline, decisions, frame step | `web/app.js` | |
| "What would a master do?": 16 tries from the exact frame (the decision's stretch + 3 s, about 3.4 s; the review's background comparisons run only the stretch), grouped by the master's first move, most likely first, with the opponent's most common reaction; one choice at a time as a ghost of both players, on repeat at ¼ speed (speed button: 1×/½×/¼×); each try saved as .slp (metadata included; parses in slippi-js and peppi) | `branch.py`, `labels.py`, `serve.py` | ~2 s on the GPU (CPU: 450 s for 8 × 5 s, not usable) |
| "Test this move ×32" (on a master option or on what you did): your side presses exactly that try's inputs (or yours) until 10 frames into its first move (as labelled), the master takes over, the opponent is sampled fresh in all 32; ranked by mean expected return (stretch + V), with the middle 80% of outcomes, the opponent's most common responses and their returns, and how often the move came out. "Best" only when the top move beats the next by more than 2 combined standard errors | `branch.py` (`force`), `serve.py` `_test` | ~4 s per move; forced inputs matched their source 2,400/2,400 frames; the move reproduces 27–32/32 in neutral spots, 7–22/32 under pressure |
| "Try it yourself": play the moment live against the bot from the exact state. 1 s of the real game with a countdown, then your controller (Gamepad API, WebUSB GameCube adapter, keyboard) vs the opponent's bot on the CPU (its S RL model; else its default model, slower); the master's try as a blue ghost to follow; scored like the tries; saved as .slp | `live.py`, `/api/live/*`, `web/pad.js`, viewer live append | scripted attempt = the try on 1,912/1,912 frames, same score; S bot 6.8 ms/frame at full speed, input-to-stream p50 16 ms; M bot ~17 ms → ~75% speed. Falco S RL (2026-11-06, while the queue trained): 60.3 frames/s streamed, gap p99 17.3 ms, 1 late tick in 242, bot 7.0 ms p50. GameCube adapter path untested (no hardware here) |
| "Could you have avoided this spot?": the master from your 1–3 decisions in the 3 s before, every line scored at the same end point (realized + V), and how many tries were in the same spot | `serve.py` `_earlier` | 3–4 s |
| Value model V (BC trunk features → MLP, 5-s outcome) | `value.py`, `value_check.py` | held-out R² 0.15 (Falco) / 0.17 (Fox); vs mean of 8 rollouts r = 0.56 / 0.72; cuts between-try spread 21→8; ranks which try did better r ≈ 0.38 |
| Game review: eval graph, biggest swings, you vs the master at each (auto-run) | `/api/review` | review 4.3 s; master comparisons ~55 s per game in the background |
| Lab: drag, flip, set percent; play out 6 ways; continue from any paused frame of a try | sim patch (place command), `/api/lab` | 2–5 s; continuing a try is sim-exact |
| Pattern report: 40 recent games, decisions grouped by situation, master's choices at examples | `/api/patterns` | 150 s for 40 games / 8,068 decisions |

The viewer (`web/viewer`, MIT) is built from melee-3d-viewer `3ec3322` plus `tools/coach/viewer.patch`: live append (the HUD and camera follow a growing replay), and a `speed` message (sets its `fps`) for slow motion, a `ghostAlpha` message (ghost opacity; the app uses 0.45, the viewer's default is 0.9), and a ghost labelled `@` gets no number badge. A try is sent as two one-player ghosts (`?only=<port>`): you in blue, the opponent in red, the timeline's colors.

Limits, measured:
- Without an NVIDIA GPU (server on the CPU, 32-core machine, training running alongside): a 2:10 game's review
  takes 59 s (GPU: 4.3 s) and "What would a master do?" (16 tries) about 80 s (GPU: a few seconds), also while a
  review runs (the review gives way to the player's requests). "Try it yourself" is CPU by design and unaffected.
  Measured 2026-11-05. (An earlier 326 s review was a bug: the review's feature step left the job thread on one
  torch thread, which also made every later CPU job 4-5x slower.)
- A tested move's ± is only the luck of the opponent's responses in the sim. It doesn't include the value
  model's own error or the opponent model being a generic master; two moves within a few points are not
  reliably ordered.
- One choice's score is noisy: ±8 points from luck even with V. The app colors only gaps beyond 8 and
  says "gaps under 8 are within luck"; patterns show ± and warn about many situations being tested.
- Branching needs online games (connect codes): offline games vs CPU diverge in the sim (13 of 14
  divergent games in a 120-game sample). One of the owner's online games diverges at frame 2975.
- Review models exist for Falco and Fox only (`data/coach_rank/value_{falco,fox}.pt`); other
  characters get branching but no review until a V is trained (`value.py extract/train`, ~10 min each).
- In every try the opponent is played by a model of a master of their character, so they react to the new
  choice. It stands in for a strong player, not for this opponent; nothing yet measures how well it
  matches a specific person. The game view ghosts only your character; clicking a try shows the
  opponent's side too (`/api/branch/<job>/<i>.slp?only=<port>` strips the other player).
- The lab places fighters (position, facing, percent) and lets the game settle them; it can't set an
  action state (shield, ledge hang) directly.
- Falco's policy is the 07-20 BC model (no cond-heads). The Fox policy is BC (`fox-M-bc2-mlr01-s1`).

Blocked on the owner: hosting. As built, a player needs this server running; a hosted GPU server
running `serve.py` (with its own ISO for the sim) makes it a website with no install.
