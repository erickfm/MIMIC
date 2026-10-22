# rlvr — reinforcement learning for MIMIC

RL post-training runs in the native sim: a Linux build of Melee compiled from
the decomp (built by `tools/build_native_sim.sh` into the gitignored `sim/`).

```
rlvr/
├── sim/     melee_sim (one game per process), vec_env (many envs), batch (batched GPU
│            policy, KV-cache streaming for sliding-window models), ppo + trajectory,
│            lcancel (the L-cancel reward), gamestate / controller plumbing
├── state/   canonical GameState + peppi/libmelee adapters (used by the canaries)
└── tests/   pytest rlvr/tests
```

Entry points: `tools/sim_rl.py` (RL), `tools/sim_h2h.py` (head-to-head with
statistics), `tools/sim_play.py`, `tools/sim_replay_check.py` /
`tools/sim_env_check.py` (fidelity vs .slp). Status and limits:
`docs/research-notes-2026-10-20b.md`.

The Dolphin RL loop (`rlvr/online`: actors, savestate drills, VR reward modules,
streaming slippistats ports) and the offline single-frame RLVR pipeline
(`rlvr/train`, `rollout`, `eval`, `sampler`, `tagger`, `tasks`) were removed
2026-10-22; recover them from git history.
