# 2026-11-09 — where S sim-RL time goes (timing only)

`rl-v2` at S (`fox-S-bc2-hf51k-s1.rl2-s1`, 4090): ~62 s per update = ~30 s rollout + ~32 s update
(4 PPO epochs ~29.5 s); 100 updates ~104 min. Nothing below has been tested for strength.

## Rollout: one lockstep step of 64 envs (profiled outside sim_rl, 1,000 steps after warm-up)

| part | ms/step |
|---|---|
| learner forward + sample (CUDA graph, not compiled) | 4.64 |
| opponent forward + sample (CUDA graph, compiled) | 2.38 |
| sim, 16 worker processes | 4.18 |
| pads to the wire | 0.21 |
| total | 11.43 (5,600 env-frames/s) |

The parts run one after another. In sim_rl the rollout runs ~4,350 env-frames/s (~14.7 ms/step);
the ~3 ms difference is presumably sim_rl's per-slot Python return bookkeeping (not measured directly).
Only 1 frame in 8 is trained on (`--sample-every 8`).

- Compiling the learner like the opponent: 4.64 → 2.54 ms, step 11.43 → 9.35 ms (6,850 env-frames/s).
  The compiled forward follows in-place weight updates (trunk features vs eager after a 1e-2 weight
  perturbation: max diff 0.035 at bf16, outputs moved 0.29). Not wired into sim_rl.
- Untried: overlap the sim (CPU) with the model (GPU) by stepping two env groups alternately.

## Update: `--ppo-amp` at S (3 updates each, seed 1, same args as rl-v2)

| | update_s | epochs_s | rollout_s |
|---|---|---|---|
| fp32 (rl-v2) | 30.8 | 28.3 | 28.4–29.8 |
| bf16 (`--ppo-amp`) | 21.8 | 19.9 | 28.7–29.7 |

Anchor KL and clip fraction in the same range over these 3 updates (says nothing about strength).
At M (2026-11-04) the same flag took the update 83 → 52 s.

Both together would be roughly 59 → 46 s per update (estimated by adding the two measured savings).
Before using either: one 100-update run with them, h2h vs `fox-S-bc2-hf51k-s1.rl2-s1`.
