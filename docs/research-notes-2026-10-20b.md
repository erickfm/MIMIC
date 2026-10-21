# 2026-10-20b — Native decomp Melee: replay fidelity check

## TL;DR

A native Linux build of Melee compiled from the decomp (Hero88go/melee-unlocked
0.9.13, headless) replays **current-Slippi 1v1 games physics-exact: 465 of 466
games across all 23 BC characters except Ice Climbers.** About 32k fps across 28
processes, ~18× our best Dolphin harness. **Decided: we are not making old
(Slippi 3.15/3.16) ranked replays resimulate.** See "Not pursuing" below.

## Setup

- Build: `bash tools/build_native_sim.sh` (also run by `setup.sh`, ~30 s from
  scratch). It fetches Hero88go/melee-unlocked at the pinned commit
  (`tools/patches/melee-unlocked-commit.txt`) plus the doldecomp submodule into
  gitignored `sim/`, applies our GPL-3.0 Linux replay host
  (`tools/patches/melee-unlocked-mimic.patch`, which also carries the RL env mode), installs gcc-14 (apt as
  root, otherwise extracted into `sim/toolchain`, no sudo), and builds headless
  against the local `melee.iso` (SHA-1 checked). The replay host plays a
  replay's recorded inputs with resync off, so positions, actions and percent
  are simulated, not restored; the RNG seed is restored per frame, as Dolphin's
  playback does. Never commit or upload anything under `sim/`: it holds
  decompiled game code and binaries built from it.
- Checker: `tools/sim_replay_check.py` compares every native post-frame payload
  byte for byte against the .slp. "Physics-exact" means action state, position,
  facing, percent, shield and stocks match on every match frame (frame ≥ 0),
  with +0/−0 counted equal. Batch mode:
  `--batch <dir> --exe <melee_linux> --iso melee.iso --n N --jobs 28`.

## Results

| set | Slippi | physics-exact |
|---|---|---|
| owner netplay + realtime bot games, 1v1 | 3.19.1 | 40/40 |
| 07-22 h2h rerun (`emulator_ss`), 23 chars, DL/BF/FD | 3.19.0 | 425/462 |
| same, excluding Ice Climbers games | 3.19.0 | 425/426 |
| ranked master Fox (BC training corpus) | 3.15/3.16 | 2/100 |

- **Ice Climbers: 0/36.** The divergence always starts in the ICs' own state
  (probably Nana). Open; don't run ICs RL in the sim until it's fixed.
- One non-ICs miss: a Young Link position drift of at most 256 ulps
  (~1e-4 units) over 223 frames that then re-converges.
- Byte-level differences that don't affect physics: +0.0 vs −0.0 in kb_vel_y,
  and the documented 1-ulp spawn values plus state flags on pre-match frames.
- 4-player games: 2/6 diverge at spawn (positions off ~4 units). Irrelevant for 1v1.
- Speed: ~2,000 fps for one process; ~1,150 fps/process × 28 ≈ 32k fps
  (~540× realtime), still including GX command generation and audio mixing.
- Not covered: free-running RNG (playback restores it each frame); YS/FoD/PS
  were only exercised by the owner's 34 netplay 1v1s (6 characters).

## Not pursuing: resimulating old ranked replays

The ranked dataset (Slippi 3.15/3.16) was recorded with an **older UCF**: it
lacks UCF 0.84's SDI, pad-buffer, shield-SDI, extended-shield-drop and DBOOC
codes, and its dashback/shield-drop/tumble codes have different bytes. The
native game implements only UCF 0.84 (all or nothing, `mu_ucf_enabled`). The
replay-code mapper copied from the Windows host also maps any known UCF address
to 0.84 without checking the bytes. The first ranked divergence we inspected is
an SDI during hitlag that Dolphin didn't apply (frame 330, Dream Land).

The only thing this would unlock is seeding RL from exact states in the old
human corpus. **Owner decision (2026-10-20): drop it.** The sim matches current
Slippi, which is what we deploy on and what RL needs; BC training never touches
the emulator. If replay-state seeding is ever wanted, use current-Slippi
replays, which already resimulate exactly. Don't reopen the old-UCF work.

## RL env, milestone 1 (`rlvr/sim/melee_sim.py`)

One headless game per OS process. One game per process is forced by the game itself: console RAM
lives at the fixed address 0x80000000 and all state is global. The process talks to Python over
two pipes.

```python
from rlvr.sim.melee_sim import MeleeSim, Pad
sim = MeleeSim("some_1v1.slp", seed=1234, rng="offline")   # template: chars, costumes, stage, rules
frame = sim.reset()                                         # frame -123, ~0.04 s, no menus
while not sim.done:
    frame = sim.step([Pad(...), Pad(...), Pad(), Pad()])    # raw pads for the NEXT frame
    frame.post[(port, 0)]                                   # Slippi post-frame payload, as in a .slp
```

- **How it works.** A "live" flag in the replay-start struct (game shim: `mu_replay_input` skips
  injection) enters the match from the template's game info and gameplay-code list, then takes
  pads through `pad_read` like a real controller. Every frame emits frame start, pre-frames and
  post-frames, then the pad alarm reads the pads for the next frame. The host publishes at that
  read, and blocks for the controller.
- **Input timing.** Pads sent after observing frame f land on f+1. This is measured on Dolphin +
  libmelee (realtime, blocking input): presses after frames 60/100/140 were recorded on
  61/101/141. It is also the BC shard alignment.
- **Env behavior.** START never pauses (`disable_pausing` forced). `reset(initial_pads=...)` sets
  the controller state going into the match. A watchdog reports done after 600 pad reads with no
  frame, and the client read times out after 30 s.
- **RNG.** `rng="offline"` is Melee's own chain, seeded once at frame -123. `rng="online"` is
  Slippi netplay's per-frame overwrite `((frame+123) << 16) | (seed & 0xFFFF)`; the counter is
  Slippi's online clock, so it runs ahead across pauses.
- **Speed.** ~2,200 frames/s per process including the Python round trip, the same as the bare sim.

Validation: `tools/sim_env_check.py` plays a recording's raw controller values (pre-frame physical
buttons, raw sticks, physical triggers × 140) through the live path with no restore of anything
after frame -123, and compares every post-frame.

| set | physics-exact |
|---|---|
| realtime offline Dolphin, bot vs bot (Doc/Fox, Samus/Link) | 8/8, RNG in step all game |
| owner netplay, 1v1 (online RNG) | all; the 6 misses of 101 are 4-player games (spawn, teams stock steal unsupported) |
| `emulator_ss` FFW h2h set | 296/462 |

The h2h misses are a property of those recordings. All 128 non-Ice-Climbers failures begin with
an RNG desync, while realtime games of the same characters stay in step. **So FFW (`emulator_ss`,
Fizzi's engine loop) consumes RNG differently from realtime Melee.** The most likely reason is
draws that happen once per rendered frame; that mechanism is inferred, not checked. All 19
"physics first" cases are Ice Climbers games (the known ICs divergence).

Not yet: a feature adapter from post-frames to the policy's input tensor (`inference_utils`), a
multi-env batched runner, savestates/`fork()`, and a template generator (build game info for any
char/stage pair instead of copying a .slp).

## RL env, milestone 2: policies in the loop, and speed

**Policy I/O, exact by construction.**
- `rlvr/sim/gamestate.py` runs libmelee's own Console event parser over sim frames; the template
  .slp supplies Event Payloads and Game Start. Against libmelee's parse of the original .slp:
  10,742 frames × 20 PlayerState fields, 0 mismatches.
- `rlvr/sim/controller.py` converts libmelee controller calls to raw pads with libmelee's own
  fix_analog formulas: sticks `round((v-.5)*160)`, triggers `round(v*140)`, and digital L/R
  forces that trigger fully. Each is measured on realtime Slippi Dolphin with 65-point sweeps.
- `SlpWriter` writes sim games as .slp; they give the same canaries as the original Dolphin games.

**Batched runner** (`rlvr/sim/vec_env.py` + `rlvr/sim/batch.py`). K CPU workers step and parse M
sims each (features into shared memory); the GPU advances each character's policy for all of its
(env, side) slots in one CUDA-graph replay (window shift + `forward_last` + Gumbel-max sampling).
- The fed-back controller feature equals the one-hot of the sampled indices for all 6,993 head
  combinations.
- Batched (argmax, fp32) vs the per-env PlayerState + decode_and_press loop: 4 games × 1,623
  frames, 0 differing bytes.
- Behavior vs realtime Dolphin, Doc vs Fox: sim 48 games vs Dolphin 6. Center-stick 35.2/35.9 vs
  35.9/35.8; L-cancel 68/90 vs 64/84 (Dolphin n = 107/246 landings); Fox wins 43-5 vs 5-1.

**Speed ladder** (one RTX 4090, i9-14900KF):

| change | effect |
|---|---|
| `--no-audio` (AX mixing) | sim 2.95× (exact: audio never touches game state) |
| `--no-render`, used for `rng="online"` | sim 8.5× total, ~16.8k frames per CPU-second. Exact online; shifts the offline RNG (some effects draw random numbers while rendering) |
| `forward_last`, last position only | GPU 1.25× (same math, ≤2e-6) |
| bf16 + torch.compile | GPU ~4× vs fp32 full forward: ~10k env-frames/s per policy at batch 64 |
| CUDA graph + Gumbel-max sampling | step launch overhead removed |
| **end to end**, 64 envs / 16 workers, self-play | **3,581 env-frames/s ≈ 60× realtime**; the Dolphin RL harness managed 5.7× |

The GPU is the bottleneck: every frame recomputes the full 180-frame window (~7 GFLOP per policy).
An exact KV cache needs the model trained with sliding-window attention over longer sequences;
that would cut per-frame inference ~180× and is a model change needing h2h validation. Other
items left: overlapping CPU and GPU by stepping two env groups alternately (hides the ~5 ms of
worker time per step), Ice Climbers, and a template generator.

## Test status (read before relying on any of this)

**Tested, with the evidence:**
- Native sim vs Slippi Dolphin, replay path (inputs injected, RNG restored per frame): current-
  Slippi 1v1, 465/466 games physics-exact across 23 characters except Ice Climbers.
- Live pad path (`--env`), raw recorded pads, no restore after frame -123: realtime offline Dolphin
  8/8 (Doc/Fox, Samus/Link), RNG in step all game; owner's netplay 1v1s all exact (online RNG).
- `--no-audio` exact on those same sets; `--no-render` exact only under online RNG (offline 6/8).
- GameState adapter: **one** replay, 10,742 frames × 20 fields, 0 mismatches.
- SlpWriter: **one** replay; peppi reads it and canaries match the original.
- Controller formulas on realtime official Slippi Dolphin: main/C stick X/Y and the L trigger
  (65-point sweeps); shoulder 0.4/1.0 with and without digital L; digital R; X; Y.
- Batched runner vs the per-env reference: **4 games × 1,623 frames, argmax, fp32, no compile,
  Doc vs Fox only, offline RNG**: 0 differing bytes.
- Behavior vs Dolphin: **one matchup** (Doc vs Fox), 48 sim games (bf16 + compile + CUDA graph +
  Gumbel sampling, offline RNG) against only 6 Dolphin games.
- Throughput numbers: single runs on this box, not repeated.

**Not tested:**
- The fast GPU path's *sampling*: Gumbel-max vs `torch.multinomial` was never checked
  statistically. bf16/compile/graph decisions were never compared to fp32 head by head; the only
  check is the 48-game canary/win-rate comparison above.
- Ditto matchups (same character on both sides, two slots per env), any matchup besides Doc vs
  Fox, and more than two policies in one runner.
- The batched runner under `rng="online"` for fidelity (only benchmarked); templates other than
  the 6 offline Doc/Fox Dolphin games.
- Episode edge cases in the runner: timeouts, sudden death, ties, `max_frames`, the watchdog.
- Failure handling: a crashed sim or worker probably hangs or raises; nothing catches it, and
  `/dev/shm` segments leak on abnormal exit.
- `forward_last` / `BatchedPolicy` on anything but `mimic`-preset 7-class checkpoints
  (fox-master, doc). SwiGLU, pos_emb and 5-class (legacy) models are unsupported or unverified.
- Controller: Z, D-pad, START, analog R sweep.
- GameState adapter and SlpWriter beyond one replay, items/projectiles (the sim emits no item
  events, so `gs.projectiles` is always empty), stage events (FoD platforms, Whispy, Stadium).
- `tools/build_native_sim.sh` with the decomp patch on a fresh machine (tested only as an
  incremental rebuild here).
- Ice Climbers (diverges), 4-player and teams (spawn, stock steal), and pausing (always disabled).
- No RL training has run on the sim yet; PPO is not wired to `VecSim`.
