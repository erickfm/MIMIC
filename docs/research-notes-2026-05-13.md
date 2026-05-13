# Research notes — 2026-05-13

## tl;dr

1. **Built a value-function discovery pipeline** in `value/`. The goal: train V(s) on game outcomes, then use it to surface candidate verifiable rewards (VRs) — features the model attends to that could be operationalized as RL training rewards.

2. **The working V(s)** is a small Markov MLP (`checkpoints/v-fox-baseline-20260512_best.pt`, val BCE 0.6005 / 65% acc). Aggregate val_loss looks unimpressive but per-position analysis shows it's actually a sampling artifact (early-game frames are unpredictable from any single frame; the model does real work in mid-late game).

3. **Discovery via matched-pair analysis**, not feature attribution gates. For each frame, bin by `(self_stock_bucket, opp_stock_bucket, self_pct_bucket, opp_pct_bucket)`. Within a bin (matched macro state), compare the frames the model rates high-V to the frames it rates low-V. Features that differ between those groups are what V(s) uses *beyond* stocks-and-percent — candidates.

4. **Engineered features dominate the discovery output**. Ported `slippistats`'s combo state machine to operate on shard tensors (`value/derived_features.py`, 37 features). When derived features compete with raw shard features in matched-pair, the top 11 are all derived: punish-state indicators, combo tracking, frame-since-event timers.

5. **Bucketing audit revealed two bugs** in the initial matched-pair (later fixed):
   - The normalized-percent thresholds I picked (±0.5, ±1.5) collapsed to **2 real buckets** because the actual std of normalized percent was 0.35, not 1.0. Each "bucket" spanned ~43% raw damage.
   - Same normalized percent has different meaning per character (Puff at 65% raw = near-death; Fox at 65% = mid-life). The original bucketing pooled them. Fixed by computing empirical per-character kill percents (`value/char_kill_percents.py`) and bucketing `raw_pct / char_kill_pct` instead.

6. **Per-opp-character audit confirms findings are not pooling artifacts.** Ran matched-pair separately for opp ∈ {Fox, Falco, Marth, Sheik, Puff}. Same feature set on top of all five rankings, slightly different magnitudes. State-machine indicators (`opp_in_hitstun`, `self_in_punish_state`, `combo_on_opp_active`) are most robust; frames-since timers and `speed_y_attack` are real but partially confounded with percent (z dropped from ~0.6 to ~0.5 after tight bucketing).

7. **Candidate VR list, ranked by confidence after the audit:**
   - **High confidence**: `opp_in_hitstun`, `self_in_punish_state` (avoid), `combo_on_opp_damage`, `combo_on_opp_frames`
   - **Medium-high**: `frames_since_self_landed_hit` (low = good), `frames_since_self_took_hit` (high = good)
   - **Medium**: `self_speed_y_attack` (low = good), `opp_off_stage`, `self_jumps_left`
   - **Matchup-specific**: `opp_invulnerable` vs Marth (ledge-stalls), `opp_jumps_left` vs Sheik (recovery-aware edgeguard)

8. **What we did NOT do**: validate any candidate as causally important. The list is correlational on human data. The actual test is to operationalize each as a `rlvr/tasks/` verifier and check whether GRPO training against it raises win-rate via fast-forward Slippi rollouts.

## What V(s) is

V(s) is a model. Input: one frame of game state (positions, actions, percents, stocks, controller). Output: estimated probability that "self" wins the eventual game outcome.

The proposal in `automated_vr_discovery.md` was: train V(s) on game outcomes, then ask "what features did V learn to use?" Those features are candidate things to reward in RL training — they survived a sign-consistency check via the value function's gradient.

We need V(s) to do two distinct things:
- **Predict outcomes**: as a sanity check that the model learned anything real.
- **Surface features**: so we can list candidate VRs.

The two are related but not the same. A V(s) with low BCE that only uses stocks+percent is *useless for discovery*. A V(s) with moderate BCE that uses many different features is *useful for discovery*. We optimize for the second.

## Methodology

### Value function (the model)

The working V(s) is small: a `MimicFlatEncoder` reading the BC feature subset (184 inputs: stage + character embeddings + action embeddings + numerical state + flags + controller one-hot), followed by a 2-layer MLP head with a scalar output. Per-frame, no temporal context. About 0.6M params. Trained with BCE-with-logits against game outcome (1 = self wins, 0 = opp wins), AdamW + cosine LR.

A larger windowed transformer (20M params, 60-frame windows, full WM-schema feature set) was tried and **performed worse**. Per-position breakdown showed the bigger model was worse in mid-late game where the actual signal lives — likely overfitting on the wider feature set combined with the windowed input. The small Markov MLP is the production V(s).

Game-outcome label derivation requires care: shard trajectories truncate *before* the loser's final death lands. Naive `sign(self_stock_last - opp_stock_last)` mislabels ~50% of games as draws. The correct logic (in `value/dataset.py:compute_game_outcomes`) is: if final stocks differ, higher wins; if equal, the player whose last stock-drop is more recent in the trajectory is the loser (they were about to die again); if tied with no drops at all, percent tiebreak. After this fix, class balance is ~50/50 with 0 draws.

### Matched-pair discovery (the main analysis)

The model's prediction is itself a function of stocks+percent (those alone get LR 0.66 BCE — close to the model's 0.60). To extract what the model uses *beyond* the obvious, we control for macro state.

Algorithm:
1. Sample frames from val shards (matching the training distribution).
2. For each frame, compute the macro bin key `(self_stock_bucket, opp_stock_bucket, self_pct_bucket_charadj, opp_pct_bucket_charadj)`.
3. Group samples by bin. Drop bins with fewer than ~80 samples for statistical power.
4. Within each bin, sort by model output. Take top-quantile (Q=0.15) as the "model thinks self winning" group; bottom-Q as "model thinks self losing."
5. For each feature, compute mean(high-V) − mean(low-V) within the bin.
6. Aggregate across bins: weighted average of the per-bin Δ, weighted by bin size.
7. Z-normalize by the feature's global std so all features are comparable on the same scale.

Output: a per-feature `z_diff`. Magnitude tells you how much the model uses that feature for conditional judgments (given matched stocks+percent). Sign tells you direction (positive z = high-V group has higher mean for this feature).

### Bucketing matters a lot (the audit)

Two bucketing issues — both real, both fixed:

**Issue 1: percent thresholds were too coarse.** My initial bucket thresholds at normalized-percent ±0.5, ±1.5 were intended to be ~1 std wide. But the actual std of normalized percent in this data is 0.35, not 1.0. The thresholds were 3× too wide. 99% of frames fell into just 2 of the 5 buckets — each ~43% raw damage wide. "Matched on percent" really meant "matched to ±20% raw damage" — not tight at all.

Fix: tighter thresholds in the *meaningful* range, computed from the actual distribution.

**Issue 2: cross-character pooling.** The percent normalization is global (one min/max across all characters). The same normalized percent has different game-meaning per character. Puff at 65% raw is near-death (Puff dies around 80% in tournament play). Fox at 65% raw is mid-life (Fox dies around 130%). The original bucketing pooled them.

Fix: empirically compute per-character kill percent from the data (see `value/char_kill_percents.py`). Bucket `raw_percent / char_kill_percent` instead of normalized percent. This puts Puff-at-65% in a "near-death" bucket and Fox-at-65% in a "mid-life" bucket, where they belong.

Empirical kill percents from the data (means; ranked play values are higher than tier-list values because of imperfect kill confirms — but the *relative ordering* is correct):

| Character | n deaths | kill_pct_mean |
|---|---:|---:|
| FOX | 5555 | 160 |
| FALCO | 920 | 152 |
| MARTH | 578 | 150 |
| JIGGLYPUFF | 489 | 134 |
| DK | 353 | 185 |
| SHEIK | 328 | 165 |
| PEACH | 296 | 162 |
| CPTFALCON | 284 | 166 |
| SAMUS | 117 | 164 |
| LUIGI | 114 | 160 |
| KIRBY | 21 | 124 |

**Effect of the fixes**: most top features barely moved (state-machine indicators are character-symmetric, so weren't confounded). `frames_since_self_landed_hit` dropped from z=-0.62 to -0.50. `self_speed_y_attack` dropped from -0.54 to -0.44. `opp_percent` dropped out of top 30 entirely (it had been a proxy for "where in the percent-bucket are we" — once the bucket is tight, it has nothing to add).

### Engineered features (slippistats-style)

`slippistats` is the community library for derived Slippi stats. It computes combos / conversions / L-cancels / DI / etc. from raw `.slp` files via a state machine driven by action-state ranges (DAMAGE_START..END = 75..91, CAPTURE_START..END = 223..232, etc., per `slippistats/enums/state.py`).

Operating on raw `.slp` files is incompatible with our shard pipeline (we'd have to re-parse 73K replays and re-link them to shard indices). So I ported the *logic* — the same action-state ranges and the same combo state-machine — to operate directly on shard tensors. That's `value/derived_features.py`. Produces 37 features per frame:

- 18 state indicators: `opp_in_damaged`, `opp_in_grabbed`, `opp_in_hitstun`, `opp_in_punish_state` (the composite), same set mirrored for self.
- 8 combo trackers: `combo_on_opp_active`, `combo_on_opp_hits`, `combo_on_opp_damage`, `combo_on_opp_frames`, and mirror for opp-on-self.
- 6 per-game cumulative: `game_combos_won`, `game_combos_lost`, `game_damage_dealt`, `game_damage_taken`, `game_opp_stocks_taken`, `game_self_stocks_lost`.
- 5 frame-since timers: `frames_since_self_landed_hit`, `frames_since_self_took_hit`, `frames_since_in_neutral`, `frames_since_combo_on_opp_ended`, `frames_since_combo_on_self_ended`.

Runs in ~1.2s per shard (numpy on CPU; sequential per-game state machine).

When derived features compete with raw features in the matched-pair output, the **top 11 features are all derived**. Raw features push to position 12+. This is good — derived features are also *closer to operationalizable VRs* than raw features. "Reward combo damage" is a clearer reward signal than "reward higher `opp_numeric[col 11]` value (which happens to be hitstun_frames_left)."

### Per-opp-character audit (the sanity check)

Even after the bucketing fixes, you can still imagine that the discovered features are artifacts of pooling across opp characters. Test: run matched-pair separately for each top opp character. If the discoveries are real, the rankings should agree across matchups. If they're pooling artifacts, the rankings will diverge.

Did this for opp ∈ {Fox, Falco, Marth, Sheik, Puff}. Result: the same feature set sits on top of all five rankings. Magnitudes vary (Sheik shows strongest z, Puff weakest), but the *set* of top discriminative features is robust. Some matchup-specific signals also appeared (e.g., `opp_invulnerable` against Marth — ledge-stalling is a known Marth pattern). These are correctly matchup-specific, not in the cross-char list.

This is the strongest sanity check we ran. The candidate VR list survived it.

## Findings (the candidate VR list)

After all the audits, the ranked candidate VRs are:

| VR formulation | Confidence | Notes |
|---|---|---|
| `+ time in opp_in_hitstun` / `combo_on_opp_active` | **High** | Largest, most stable across all matchups |
| `- time in self_in_hitstun` / `self_in_punish_state` | **High** | Defensive equivalent |
| `+ combo_on_opp_damage` | **High** | Extract more damage per punish |
| `+ combo_on_opp_frames` | **High** | Extend combos longer |
| `- frames_since_self_landed_hit` | Medium-high | Maintain offensive pressure (less time since hit = better) |
| `+ frames_since_self_took_hit` | Medium-high | Defense: longer since last hit = better |
| `- self_speed_y_attack` (positive value) | Medium | Avoid being launched upward by attacks |
| `+ opp_off_stage` × edge proximity | Medium | Edgeguard pressure |
| `- self_off_stage` | Medium | Stay on stage |
| `+ self_jumps_left` | Medium | Mobility budget — don't waste recovery jumps |
| `- opp_invulnerable` (vs Marth) | Matchup | Punish ledge-stalls |
| `- opp_jumps_left` (vs Sheik) | Matchup | Track opp recovery resources |

Each is a deterministic function of game state. Each can be written as a `rlvr/tasks/` verifier.

## Key gotchas / things that bit me

1. **Stock-drop-recency outcome labeling.** Shard trajectories cut off before the loser dies — the captured last frame has both players at non-zero stocks. Naive `last_stock` comparison labels 50% of games as draws. Use the recency tiebreak. Documented in [[vs-outcome-truncation]].

2. **Percent uses `normalize` (min/max), not `standardize`.** The transform is `2(raw - min)/(max-min) - 1` with min=0, max=343.4 (the max percent observed in fox_all_v2). Inverting: `raw = (normalized + 1) * 171.7`. I got this wrong initially and got kill percents 3× too low.

3. **`speed_y_attack` is knockback velocity from being hit**, not attack motion. Positive y = launched upward. The model uses this strongly because most launches go up.

4. **`GROUND_ATTACK_UP` is the get-up attack from face-up lying**, not a normal up-attack. It sits in the get-up-from-knockdown action cluster (0xb7-0xc6), not the attack cluster. It's a punishable defensive option.

5. **`jumps_left` is jumps remaining**, not jumps used. Verified empirically (on-ground frames have value 2 = full).

6. **`off_stage` triggers on `not on_ground AND (|x| > stage_edge OR y < -6)`** — the recovery danger zone, not just "past the edge."

7. **`opp_DEAD_*` is a perspective-aggregation artifact, not a real signal.** Frames where opp is in DEAD state appear more often in low-V groups because they aggregate across both perspectives of each game, with the eventual loser dying more often → opp-dying frames concentrate in self-eventually-loses games. Don't include `opp_DEAD_*` patterns as VR candidates without re-deriving from one-perspective data.

8. **Bigger model isn't necessarily better for discovery.** 20M params + windowed + full features → val 0.6273 (worse than 0.6M Markov at 0.6005). The aggregate val_loss is dragged up by unpredictable early-game frames regardless of model capacity. What "good V(s)" means is *useful for discovery*, not lowest BCE.

## Code map

```
value/
├── __init__.py
├── build_manifest.py        — wrote data/fox_all_v2/tensor_manifest.json (90/10 split)
├── dataset.py               — FoxValueWindowedDataset + compute_game_outcomes
├── model.py                 — WindowedValueModel, MarkovValueModel
├── encoder.py               — ValueEncoder (full-feature 398-dim input)
├── train.py                 — trainer (position filter, gate, early-stop)
├── analyze.py               — sklearn ceilings + per-position breakdown
├── analyze_ckpt.py          — per-position NN-vs-LR breakdown + calibration
├── matched_pair.py          — within-macro-bin discovery (baseline version)
├── matched_pair_derived.py  — same with derived features competing
├── matched_pair_charadj.py  — same with char-adjusted percent bucketing + per-char filter
├── stratified_robustness.py — stratified by stock-differential
├── derived_features.py      — slippistats-style state machine on shards (37 features)
├── char_kill_percents.py    — empirical per-char kill percent computation
└── char_kill_percents.json  — {char_id: {kill_pct_mean, kill_pct_median, ...}}

checkpoints/
└── v-fox-baseline-20260512_best.pt  — Markov MLP, val 0.6005 — the working V(s)
```

`/tmp/value-*.json` has all the analysis dumps from each run.

## Open work

1. **Validate any candidate VR causally.** Take one candidate (recommend `combo_on_opp_damage` since it has the strongest discovery signal and clear operationalization), write it as a `rlvr/tasks/<name>.py` verifier, run a short GRPO training run against fast-forward Slippi, check whether win-rate vs CPU baseline goes up. If win-rate moves, the discovery worked. If win-rate doesn't move, the candidate was correlational not causal — discovery surfaces co-occurring patterns, but they may not be the load-bearing skill.

2. **Compare derived feature counts to slippistats on the same `.slp` file.** We used the same action-state ranges and the same state machine, so counts should agree within a few percent. Worth a one-time correctness pass before promoting our derived features as canonical.

3. **Add action-specific binary features**. The action-level matched-pair (separate from feature-level) surfaced strong signals on `self_GRAB_PUMMEL`, `self_GROUND_ATTACK_UP`, `opp_TECH_MISS_UP`, `opp_NEUTRAL_GETUP` with log-ratios > 2. These are easy to add to `derived_features.py` as binary indicators and would surface as additional VR candidates.

4. **Level-2 discovery.** Once level-1 features are operationalized, the proposal's recursive decomposition says: take a level-1 candidate (say `combo_on_opp_damage`), predict it over short windows from state, and look at what predicts that. We have all the infrastructure to do this — `value/dataset.py` could yield K-frame deltas of any derived feature as the target instead of game outcome. But we haven't done it yet.

5. **The 0.60 val ceiling is real but not informative.** The Markov baseline plateaus there because half the data is structurally unpredictable. Don't try to push past 0.60 with more model — push past it by restricting to informative game positions if you need a sharper-looking model. Past attempts at position-restricted training improved mid-game but tanked late-game (the gate-trained model lost ~25 points of accuracy in [0.9, 1.0)). Be careful.

## Quick reproduce

To re-run the full discovery from a fresh checkout (assuming `data/fox_all_v2/` exists with the standard shard format):

```bash
# Build manifest (one-time)
python -m value.build_manifest --data-dir data/fox_all_v2

# Train V(s) baseline (if not already in checkpoints/)
python -m value.train --data-dir data/fox_all_v2 \
  --run-name v-fox-baseline-$(date -u +%Y%m%d) \
  --model-type markov \
  --max-steps 30000

# Compute empirical kill percents (one-time)
python -m value.char_kill_percents --max-shards 30

# Run discovery
python -m value.matched_pair_charadj \
  --ckpt checkpoints/v-fox-baseline-*.pt \
  --n-samples 80000 \
  --out /tmp/discovery.json

# Per-opp-char sanity audit
for c in 1 22 18 7 15; do  # Fox/Falco/Marth/Sheik/Puff
  python -m value.matched_pair_charadj \
    --ckpt checkpoints/v-fox-baseline-*.pt \
    --opp-char-filter $c \
    --n-samples 150000 \
    --out /tmp/discovery-char-$c.json
done
```
