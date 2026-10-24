# 2026-10-23b — guarded reward mixtures, a reward audit, and causal interventions

Follows `research-notes-2026-10-23.md` (stock 1 + damage 0.25 with a critic beats frozen BC 56.3%
over 1,152 games).

## TL;DR

- Lagrangian skill guards (`--guards`) and mixtures with recovery / low_percent_kill as objectives
  screen at 55.7–57.8% vs BC over 192 games. None beat the p2 mixture's 62% screen, and the
  intervals overlap. Not confirmed at 480 games.
- An audit of the skill VRs found recall bugs in recovery and tech (fixed in 67e25e2). It also
  found that on 399 real games only stock_delta, recovery and low_percent_kill predict winning
  beyond damage. neutral_win_loss has the wrong sign once damage is controlled for.
- Causal interventions (new: `rlvr/sim/interventions.py`) rewrite one player's controller by rule
  and play the intervened BC against plain BC. On Fox and Marth, **removing** L-cancels or DI
  loses badly (29–67, 33–63, 34–62, 38–58).
  - **Perfect L-cancel gains nothing:** 240–240 (Fox), 229–251 (Marth), at 480 games each. BC's
    85–91% already captures the value.
  - **Survival DI depends on the character.** Fox: 241–239. Fox lives 9% longer but takes 23%
    more damage. Marth: **311–169 (64.8%, p < 10⁻⁹)**. Marth dies at 156% instead of 134%.
    BC's DI is far from optimal for Marth.

## Guarded mixtures (192-game screens vs frozen BC, same strong-KL + critic recipe, normalized)

| Run | Objective | Guards | h2h192 | KL |
|---|---|---|---|---|
| G1 | stock 1 + damage 1 | all 8 skills (old recovery/tech defs) | 111–81 (57.8%, p = 0.036) | 0.0058 |
| G3b | + recovery (fixed) | 6 skills | 107–85 (55.7%) | 0.0081 |
| G6 | + recovery + low_percent_kill | 5 skills | 107–85 (55.7%) | 0.0053 |

G2 (every VR as an objective) and G4 (neutral as objective) were not run. The audit showed those
rewards don't track winning (held in `checkpoints/vrsim/queue_hold_G.txt`).

## Reward audit (`checkpoints/vrsim/audit/REPORT.md`, gitignored)

- tech counted only a damaged state on the exit frame, which missed grab exits and tech chases
  (recall 45–55%). Fixed: a hit or grab within 30 frames after the exit counts as punished.
- recovery opened only on the transition off-stage while in hitstun, which missed 229 of 660 own
  deaths in sim. Fixed: it opens on any off-stage frame in hitstun, damage state or TUMBLING.
  Earlier (fea5b02): a failure counts only after the player was actionable off-stage.
- neutral_win_loss: 16–25% of its wins are pokes under 5%, and it covers 34–39% of an independent
  "first hit after mutual actionable" definition. Its outcome coefficient after controlling for
  damage is −0.67 ± 0.22. A rework is waiting on the owner's labels (labeling page, 60 blind clips).
- A feature search over real games found almost nothing beyond damage that predicts wins. The
  survivors are kill-linked or Fox-specific: percent at death, edgeguard kill rate, ledgehog with
  opp off-stage, shine out of shield. Correlational only, which is what motivated interventions.

## Causal interventions

`rlvr/sim/interventions.py`: a rule sees the frame's libmelee GameState in the sim worker and
rewrites that player's 8-byte pad before it is sent (`rlvr/sim/vec_env.py`). `tools/sim_h2h.py`
takes it as a third field: `--a iv=hf_eval/fox-master:bf16:lcancel_perfect`. The report adds the
frames per game each rule changed and an engine-flag L-cancel rate that works for any character.
`checkpoints/iv/run_iv.sh CHAR CKPT RULE GAMES SEED` runs one experiment, records replays, and
appends to `checkpoints/iv/results.txt` with percent at death (`checkpoints/iv/death_pct.py`).

Rules:
- `lcancel_perfect`: during an aerial attack (65–69) in the air, the analog L trigger toggles
  0/255 every frame, so a rising edge falls inside any 7-frame window. Analog only, so it can't
  airdodge.
- `lcancel_never`: no L/R or Z during aerial attacks.
- `di_survival`: while being hit (hitlag as the victim, this frame or the previous), the main
  stick points up and toward stage center at 45°.
- `di_none`: main stick neutral during the same frames.
- Also implemented, not yet run: `no_offstage_airdodge`, `no_shield`.

Caveat: the policy's controller-history input holds what it chose, not the rewritten pad. A
"never" rule therefore also puts the policy slightly off-distribution: it pressed L and saw full
landing lag.

Results (intervened BC vs plain BC, sim ditto, bf16, both orientations of 6 templates):

| Char | Rule | Games | Record | Win rate [95%] | p | Rule check | Other |
|---|---|---|---|---|---|---|---|
| Fox | lcancel_never | 96 | 29–67 | 30.2% | 0.0001 | L-cancel 2.4% vs 92.5% | damage dealt −72% ± 11 |
| Fox | lcancel_perfect | 96 | 46–50 | 47.9% | 0.76 | 98.5% vs 91.5% | |
| Fox | lcancel_perfect | 480 | 240–240 | 50.0% [45.5, 54.5] | 1.00 | 98.8% vs 91.3% | damage −0.8% ± 5.6; % at death 113.3 vs 113.0 |
| Marth | lcancel_never | 96 | 33–63 | 34.4% | 0.003 | | |
| Marth | lcancel_perfect | 480 | 229–251 | 47.7% [43.3, 52.2] | 0.34 | flag 97.9% vs 85.5% | damage +6.9% ± 5.4; % at death 126.5 vs 129.3 |
| Fox | di_none | 96 | 34–62 | 35.4% | 0.006 | | % at death 109.2 vs 112.6 |
| Fox | di_survival | 96 | 46–50 | 47.9% | 0.76 | | % at death 126.0 vs 116.5 |
| Fox | di_survival | 480 | 241–239 | 50.2% [45.8, 54.7] | 0.96 | | % at death 122.5 vs 113.4; damage taken +22.9% ± 6.0 |
| Marth | di_none | 96 | 38–58 | 39.6% | 0.052 | | % at death 123.5 vs 130.6 |
| Marth | di_survival | 480 | 311–169 | 64.8% [60.4, 68.9] | < 10⁻⁹ | | % at death 155.9 vs 134.1; damage taken +17.4% ± 5.7 |

Reading:
- Both skills matter: taking either away costs 10–20 points of win rate.
- L-cancel: moving BC to the ceiling is worth nothing measurable (±4.5 points at 480 games), on
  both characters. An L-cancel VR can raise its metric without raising win rate. This matches the
  audit, where l_cancel carried no outcome signal beyond damage.
- DI: one fixed rule (up and toward center at 45°) is a wash for Fox and a large win for Marth.
  Both get hit more (Marth +17%, Fox +23% damage taken). Marth gains 22 points of survival percent
  and Fox only 9. A likely cause, not tested: Fox dittos kill mostly vertically (up-smash, up-air),
  where holding up is the wrong DI, while Marth dittos kill mostly horizontally (forward smash,
  forward air), where up-and-in is right. A DI rule perpendicular to the knockback angle would test
  this.
- Owner decision (2026-10-23): RL objectives stay unconditional outcomes that a player always
  wants more of (stocks, damage delta). Conditional skills are not prioritized as rewards. DI is
  the example: survival DI, combo DI (out and away) and mixed DI are each right in different
  situations, and the fixed-rule result splits by character. Interventions stay useful as
  measurements of where BC is weak.
- Rule for splitting goals: if a behaviour is conditional, it is a behavior-cloning
  goal. If it is better always, or the vast majority of the time, it is a VR we can RL toward.
  BC learns when to do what from human play; RL pushes only on outcomes that are better
  regardless of context.
- Checked: the Marth result holds in both registration orders (156–84, 155–85) and in all 12
  template/side cells. The rule causes no tap-jumps after hitlag (0 vs 1 in 120 games).

## Not done / open

- No replication of any 480-game result with a second h2h seed, including Marth survival DI
  (the seed-61 run was stopped once DI was ruled out as an RL objective).
- DI toward a fixed direction only. Not tested: DI perpendicular to the knockback angle, or combo
  DI at low percent.
- Falco, Sheik, Falcon and Puff not run. `no_offstage_airdodge` and `no_shield` not run.
- Neutral rework waits for labels.

## RL for every model kind (end of day)

`tools/sim_rl.py` used to refuse cond-heads models (the default since 10-22) and sliding-window
models. Now any model trains:
- Each sampled decision stores the exact model input behind it (`BatchedPolicy.context`): the
  180-frame window for full-causal models; for stream models the frames since the episode start,
  up to the receptive field (6 × 179 + 1 = 1,075), kept in a GPU ring buffer during rollouts.
- The update scores decisions with `ppo.head_logits`: features at each decision's position
  (contexts right-padded, exact under causal attention), with the sampled shoulder / c / main
  classes teacher-forced through cond-heads chains, so the summed head log-prob is the exact
  joint log-prob of the played action. The global KL anchor conditions each head on the sampled
  prefix (a one-sample estimate of the joint KL for cond-heads; exact for independent heads).
- Stream learners: the KV cache keeps keys/values from the weights that computed them, and one
  update's rollout (~1,000 frames per learner slot) is shorter than the receptive field, so
  without a fix nearly every rollout decision would mix old and new weights. `BatchedPolicy.refresh`
  recomputes the learner's cache from its stored history after every update.
- `rlvr/tests/test_rl_context.py` (random-init S models, fp32, 1,300 streamed frames, one reset):
  re-scoring matches the sampling logits to 3e-8 to 7e-8 for full/stream × cond/independent; after
  a weight change the stale cache is off by 0.12 in the logits and 2e-7 after `refresh`.
- Smoke runs (2 updates, 16 envs): fox-master, the 10-20 cond-heads Fox, and a stream Fox
  (`p1-C-warmswa720-s1`) all train normally. Stream updates take 14–19 s vs 5–7 s.


## Reward spec review (owner, via the spec review page)

All current definitions approved as written: stock_delta, damage_delta, l_cancel_miss,
low_percent_kill, recovery, tech, combo_length, combo_extend, shield_escape. **neutral_win_loss is
dropped**: damage_delta already carries its signal (the audit found no outcome signal beyond
damage, sign negative after controlling for it), so no rework and no labels needed. damage_delta
is kept alongside stock_delta: no double counting (a kill pays only through stock_delta; respawn
drops are ignored), and it's the dense shaping term that made the sparse stock reward learnable
(stock alone: parity; stock + damage: the one replicated win).
