# 2026-10-23 — verifiable rewards in the native sim: first RL policy that beats frozen BC

## TL;DR

- **All 10 VRs we've ever had code for are ported to the sim** (`rlvr/sim/vr/`, commit `9c97300`)
  and validated against independent ground truth on 40 real master-Fox replays and 24 sim games
  (`tools/vr_replay_check.py`): every exact check agrees, except 1 of 1,668 L-cancel landings.
- **Dense-reward sim RL** (`tools/sim_rl.py --vr id:w,...`, commit `85d1fb5`, plus `--critic`).
- **`stock_delta 1 + damage_delta 0.25` with a value baseline and a strong KL anchor beats frozen
  BC 649–503 over 1,152 sim games (56.3%, p ≈ 2×10⁻⁵)**: 119–73 on the 192-game screen, then
  265–215 (55.2%, p = 0.025) on each of two independent 480-game h2hs (seeds 51 and 61; the games
  differ, the identical record is a coincidence — checked). First RL result in the project to
  clear a replicated significance bar.
- Canaries green and no VR it wasn't trained on regresses (self-play vs BC self-play). No obvious
  reward hacking in the stats. Nobody has watched its games yet. Not promoted.
- **Without drift control, every VR made the model worse** (stock_delta 35% and damage_delta 41% vs
  BC after 30 updates). A strong KL anchor holds parity; the critic adds signal; neither outcome
  reward alone survives replication; the mixture does.

## What was ported (`rlvr/sim/vr/`)

`VRModule.observe(state_history) -> float` per frame on the sim's libmelee GameStates (the same
objects the old Dolphin VRs read), `finalize()` at match end; `build("id:w,...", port)`.

| VR | Reward |
|---|---|
| `stock_delta` | +1 earned opponent death (recent hit, or the bot's combo still open), −1 own death, on the DEAD-action edge |
| `damage_delta` | +0.04 × % dealt − 0.04 × % taken |
| `low_percent_kill` | +0.5 earned kill below the opponent character's percent bucket |
| `combo_length` | at combo close: (clip(moves, 2, 8) − 2) / 6 if ≥ 2 moves and ≥ 5% |
| `combo_extend` | at combo close: min(damage, 80) / 80, same gates |
| `neutral_win_loss` | ±1 on clean punish edges; trades within 4 frames net 0 |
| `recovery` | −0.25 knocked off-stage in hitstun then dying |
| `tech` | −0.15 a tech situation that exits into damage |
| `l_cancel` / `l_cancel_miss` | +1 success / 0 miss; penalty-only variant 0 / −1 for mixing |
| `shield_escape` | pressured small shield: +1 escape, 0 hit out, −1 break |

Fixes to the old code found while porting (all validated): hitstun/hitlag from the Slippi bit flags
(libmelee's `hitstun_frames_left` is nonzero outside hitstun: 6,085 vs 2,734 true frames on one
replay); kill credit also when the bot's combo is still open (the old 90-frame gate scored 57 of 252
real deaths — edgeguards — as opponent self-destructs); shield escape opens at the first actionable
frame after shieldstun (the old window could almost never fire, likely May's "never triggered");
combos still open at match end are scored; slippistats move-counting, jab-reset, wall/ceiling tech,
position-based off-stage. The 3 old failing tests were wrong (stocks dropped without the DEAD
animation). Side finding: libmelee keeps the predicted copy of rolled-back netplay frames (~2% of
frames in these replays); the sim has no rollback.

## Sim RL with dense rewards

- VR modules run in the sim workers (`rlvr/sim/vec_env.py`), per side, writing per-frame rewards to
  shared memory; finalize/metadata come back with each game.
- A sampled learner decision (1/8) gets the discounted return (γ 0.99) of the next 300 frames, or
  until the match ends plus finalize, as a one-frame episode, so `ppo_update` is unchanged.
- `--critic`: V(s) = MLP on the frozen BC's final features, fit after use (lagged) on the last 3
  updates; advantage = return − V, z-scored. Explained variance is logged: ~0.2–0.4 for
  stock_delta, ≤ 0 for damage_delta alone.
- Every rollout game is played against frozen BC, but **its record is not a strength signal**: it
  counts games that finish inside a window (short-game selection bias) — damage_delta's was 23–13
  while its h2h was 78–114.
- Run: `tools/vr_rl_run.sh NAME "stock_delta:1,damage_delta:0.25" 30 --kl-beta 0.03 --lr 1e-5
  --global-kl-beta 0.01 --critic` (64 envs, 4,096 sampled decisions/update, 4 PPO epochs; ~40 s/update).

## Results (learner hf_eval/fox-master, opponent frozen BC; sim h2h, bf16, fixed harness)

| Run | Recipe | h2h vs BC | KL (final) | Verdict |
|---|---|---|---|---|
| stock_delta, u10 | lr 3e-5, kl 0.003 | 96–96 (192) | — | parity |
| stock_delta, u30 | lr 3e-5, kl 0.003 | 68–124 (35.4%, p < 0.001) | 0.087 | worse |
| damage_delta, u30 | lr 3e-5, kl 0.003 | 78–114 (40.6%, p = 0.011) | 0.116 | worse |
| stock_delta, u30 | lr 1e-5, kl 0.03, global anchor 0.01 | 98–94 | 0.0085 | parity |
| stock_delta, u30 | + critic | 105–87 (192); 240–240 (480) | 0.0058 | parity |
| stock_delta, u60 | + critic | 98–94 | 0.0093 | parity |
| damage_delta, u30 | + critic | 105–86 (192); 259–221, 249–231 (480 ×2): 53.3% over 1,151 | 0.0079 | small edge, no replication significant — not accepted |
| **stock 1 + damage 0.25, u30** | **+ critic** | **119–73 (192); 265–215, 265–215 (480 ×2): 56.3% over 1,152, p ≈ 2×10⁻⁵** | 0.0072 | **accepted on h2h** |

Finalist checks for the mixture (96 RL and 96 BC self-play games, `tools/vr_finalist_check.py`,
`validate_run.py`): canaries green (center-stick 35%, L-cancel 92%, land/kf 3.54, stick JS 0.001,
state JS 0.005 vs BC); per player-minute RL vs BC — L-cancel 11.27 vs 10.43 successes, −0.94 vs
−0.97 misses; combo_extend 1.28 vs 1.24; combo_length 0.96 vs 0.87; low_percent_kill 0.160 vs
0.146; tech −0.130 vs −0.134; recovery −0.188 vs −0.177 (~6%, within Poisson noise); shield escapes
too rare to judge. Behaviour: no timeouts or draws, games 146 s vs 157 s, off-stage time 17.4% vs
15.9% (edgeguarding), ledge and shield time similar. Replays: `checkpoints/vrsim/finalist_p2/`.

## Lessons

- Drift control is necessary: with lr 3e-5 / KL β 0.003 every outcome VR drifted into a worse
  player within 30 updates (as the July Dolphin runs did). lr 1e-5 + KL β 0.03 + a global anchor
  holds the policy at KL < 0.01 and at parity.
- A 192-game screen produces false positives (54.7% → 50.0%): accept only on 480-game
  replications with new seeds.
- stock_delta alone is too sparse to move the policy; damage_delta alone gives a small edge; the
  combination — consequence-weighted kills plus a dense damage signal — is what wins.

## Not done / open

- Single training seed; one checkpoint (u30) evaluated. Not checked in Dolphin. Nobody has watched
  the finalist's games.
- Phase 1 solo runs of the skill VRs with the working recipe (recovery running at writing; tech,
  neutral_win_loss, combo_extend, low_percent_kill, l_cancel_miss, shield_escape, combo_length held
  in `checkpoints/vrsim/queue_hold_p1d.txt`) and phase-2 mixtures adding skill shaping terms.
- sim_rl doesn't support cond-heads learners yet, so RL starts from the non-cond champion.
