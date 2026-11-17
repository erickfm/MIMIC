# What we've learned (as of 2026-11-16)

A summary of the project's established results: imitation learning (BC), sim RL, evaluation, and
the recipes that came out of them. Each point links to the dated research note that measured it.
Numbers are head-to-head (h2h) records in the native sim (bf16, 192 games, h2h seed 41) unless
noted. The research notes stay the detailed record; this page is the short version, and like them
it should be checked against the code before being relied on.

## 1. Imitation learning (BC)

### Data

- **Targets are the next frame's input.** libmelee returns post-frame state with pre-frame inputs, so
  frame i's state already shows frame i's button (e.g. KNEE_BEND on the JUMP frame). v2 shards train
  on `target[i] = buttons[i+1]`, which matches inference. Old shards leaked the answer.
- **Filter by the player's own rank, not the game's.** Rank is in each port's netplay name in the
  ranked archives. In the master-tier pool only 61.6% of Fox perspectives are master; mixed-rank pairs
  keep the target character, often the lower-ranked player (`tools/filter_masterfox.py`;
  2026-06-25).
- **C-stick history as nearest-of-9 clusters**, matching the targets and inference (shards from
  2026-07-17). The older 5-class history never showed diagonals.
- **No mirror augmentation** (owner, 2026-07-17). Its one measured gain was made with a facing-flag bug.
- **More data helps.** The full 102k-game master-Fox pool, with the c-stick fix and no mirror, beat
  the previous champion 16–5 over 21 realtime Dolphin matches, the first Fox h2h to clear binomial
  noise (2026-07-18; the three changes aren't separated).
- **Step budget per character** follows the plateau fit `steps ≈ 53.3 · Mframes^1.19` (eff batch 512),
  clipped to 3k–56k (`tools/allchar_queue.py:steps_for`).

### Model

- **Transformer with Shaw relative-position attention, 180-frame context.** RoPE was 1–2% worse on
  val at matched settings (2026-10-21) and was removed with the other position encodings.
- **Full features**: 13 numeric + 5 flags per player, ~1.5–3.5% better val than the old 9-column input.
- **7-class button head** (A, B, Z, Jump, TRIG, A+TRIG, none). TRIG must press the digital button, or
  the bot can't tech, airdodge or wavedash (`tools/inference_utils.py:decode_and_press`).
- **Conditional heads** (`--cond-heads`, default since 2026-10-22): each head is fed the classes of
  the heads before it, so the four heads model the joint distribution of one input. Beat the same
  recipe without them 611–445 (−4.2% val). Chain order: shoulder → c-stick → main stick → buttons;
  `--arch cond_order=XXXX` makes it configurable (2026-11-16); no other order has beaten it (section 5).
- **Muon optimizer** (default since 2026-10-22) on the hidden matrices, AdamW on the rest. At M,
  Muon LR 0.01 beat 0.02 by 57.6% over 3 h2h seeds; 0.005, 0.04 and AdamW were no better. At S,
  0.01 tied 0.02 (2026-10-28/29).
- **Sizes:** S (5.4M params) and M (20M) are the ones in use. XS (1.7M) lost 22% to S at the same
  recipe and is deprecated. L (44M) tied M at 56k steps (L+RL vs M+RL 92–100) at 2.7× the BC cost and
  was dropped (2026-11-04).
- **Promotion uses the tail-SWA average** of the last checkpoints. On a full cosine run it equals the
  raw final checkpoint; it only beats the best-val checkpoint.

### Tried and rejected

- **Sliding-window ("stream") attention** (18 s receptive field, KV-cache inference ~22× faster per
  frame). Lower loss but lost in play: 77–115 vs matched full attention at S 56k. RL didn't close it
  in October, and on 2026-11-16 stream RL measured ~2.1× slower per update than full on the 5090 box
  (122 vs 58 s: the KV-cache speedup is per-frame inference, while full rollouts are already
  CUDA-graphed and compiled, and RL stores and re-scores 1,075-frame contexts). Dropped.
- **LLM-architecture variants** (13, via `--arch`): only gated attention helped, at proxy scale, and
  it overfit and lost 34–62 at S 56k (2026-10-25).
- **Removed:** patching, multi-token prediction, SOAP, weight EMA, focal loss, multi-label heads, the
  MLP backbone.

### Why val loss is a weak guide to strength

Cross-entropy weights each frame's error by how often it happens, not by what it costs. A missed
L-cancel is nearly free; pressing L off-stage loses a stock. Both are rare, so a 0.001 val
difference can mean anything from nothing to a lost game. Val ranks training runs; h2h ranks models.
This is also the main argument for RL: an outcome reward is weighted by consequence by construction.

## 2. Reinforcement learning (native decomp sim)

### Setup

- **Sim:** the decomp Melee built natively (no Dolphin), physics-exact on current Slippi replays
  for 22 of 23 characters (Ice Climbers don't reproduce). Inputs land a frame later, as in Dolphin.
- **Opponent:** the frozen BC parent. CPU-9 doesn't punish reward hacking.
- **Reward:** stock delta + damage delta, unconditional outcomes only. Conditional skill rewards
  (L-cancel, DI, tech) are not pursued.
- **Algorithm:** PPO with a batch-mean baseline (no critic) and one KL term, the anchor (exact KL to
  BC on 512 random learner frames per update, β 0.01). The critic was dropped: its explained variance
  was negative (2026-10-27). A ridge per-state baseline had positive EV but was within noise in play
  (2026-11-09).

### What moves strength

- **More updates beat a bigger batch** (SF7: 16,384 decisions × 100 updates beat fox-master 109–83 in
  the sim and 130–70 in Dolphin). Sim gains carry over to Dolphin.
- **Learning rate and updates trade off.** At Fox S on two training seeds: lr 2e-5 × 200 updates ≈
  lr 1e-5 × 400, both ~1.1 logits above lr 1e-5 × 100. lr 4e-5 is no better than 2e-5. At lr 2e-5,
  u400 adds only ~0.15 logits over u200; at lr 1e-5 gains are still rising at u400 (2026-11-11/12).
- **More RL helps every character tested.** Resuming the coach S models (lr 1e-5, u100) for 100 more
  updates, each vs its own u100 (2026-11-16):

  | character | u200 | u300 | u400 |
  |---|---|---|---|
  | Fox | 241–143 (62.8%, two matches pooled) | 126–66 (65.6%) | 143–49 (74.5%) |
  | Falco | 139–53 (72.4%) | | |
  | Captain Falcon | 112–80 (58.3%) | | |
  | Puff | 123–69 (64.1%) | | |

- **RL beats BC for every character:** BC + 100 updates beat its own BC for all 24 runs of the
  all-character queue, from 108–83 (Bowser) to 181–11 (Falco M) (2026-11-08).
- **Size:** S levels off at about the old fox-master's strength. M at lr 2e-5 × 200 beat that
  fox-master 279–105 (72.7%) and rates ~1.1 logits above every S model: fox-master was short on RL,
  not limited by size (2026-11-12).
- **Speed (rl-v3 flags):** bf16 PPO update + compiled learner forward: same math, ~21% faster
  (96–96 vs plain, Fox S). S ≈ 82 min per 100 updates on the 4090; M lr 2e-5 × 200 ≈ 300 min.

## 3. Evaluation

- **h2h is the strength metric**, played in the sim in bf16. A 192-game match has a binomial sd of
  about 3.6 points; training-seed variance is larger, so one match vs one reference can't rank
  recipes a few points apart (a greedy ablation once flipped on that noise).
- **Matches aren't replayable** at a fixed h2h seed: decoding samples at temperature 1, and the same
  pair at seed 41 went 128–64 and 113–79 on different days. Treat repeat matches as independent samples.
- **Ratings:** a Bradley-Terry fit gives each model one rating r, with P(A beats B) = sigmoid(r_A − r_B),
  like Elo. The 13-model Fox round-robin (78 matches) fit one transitive scale (chi² 71 on 66 df),
  so models that never met can be compared and all games pooled (`tools/rl_ratings.py`).
- **Dolphin faithfulness canaries** catch a mistimed or corrupt live pipeline from replays alone:
  L-cancel rate (healthy ~88–91%), center-stick share (~34%; 100% = dropped inputs), stick and
  action-state distribution JS (`tools/validate_run.py` runs them all). Match length and win rate
  survive symmetric degradation, so they can't diagnose fidelity.
- **FFW Dolphin is faithful** with blocking input (L-cancel 91.5%, normal match length, ~2.9×
  realtime). Its RNG sequence differs from realtime, so FFW replays aren't bit-exact.

## 4. Recipes (`recipes/`, frozen; a change is a new version)

| recipe | what |
|---|---|
| `bc-S-v2` | current S BC (Muon 0.01, cond-heads) |
| `bc-M-v3` | current M BC |
| `rl-v4` | **current RL default** (2026-11-16): rl-v3 at lr 2e-5 × 200 updates; also sim_rl's built-in defaults |
| `rl-v3` | rl-v2 + bf16 update + compiled learner, lr 1e-5 × 100 |
| `rl-v2` | no critic, anchor KL, 16,384 decisions × 100 updates, 4 PPO epochs, lr 1e-5 |
| `rl-v1` | SF7 (with critic); runs only at commit `c2df04b` |
| `bc-L-v1`, `bc-XS-v1` | dropped / deprecated sizes |

Runs are launched by `tools/run.py launch RECIPE` and recorded under `runs/<name>/`; `tools/rl_queue.sh`
runs one RL recipe over a list of BC parents on one GPU.

## 5. Open (2026-11-17)

- **S + rl-v4 for all 23 characters** (the coach's models, on the coach branch). Done: fox 162–30,
  falco 169–23, marth 181–11, sheik 178–14 (each vs its BC parent). Puff and Captain Falcon were
  stopped partway (u100, u65) and 17 characters were never started when the rented box was shut
  down; the coach uses their rl-v2 models until then (`docs/research-notes-2026-11-17.md`).
- **Cond-heads chain order at a real budget.** On the fast proxy the 24 orders span 0.7753–0.7772
  val (current order 0.7766). The four best (cbms, mbsc, bcms, mbcs) played the current order to a
  pooled 383–385; only cbms was above 50% (108–84, p = 0.10). Nothing tested at S 56k; `scmb` stays.

Closed since 2026-11-16: the M lr 2e-5 × 200 model was promoted to fox-master (18–2 vs the previous one
in realtime Dolphin). Win rate vs RL updates for Fox S under rl-v4 rises from 63.5% at u25 to
~84–88% at u125–u200 vs its BC parent (`docs/research-notes-2026-11-17.md`).
