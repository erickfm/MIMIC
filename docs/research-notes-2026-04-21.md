# Research notes — 2026-04-21

## tl;dr

1. **Offline RLVR framework stood up end-to-end.** New `rlvr/` package:
   canonical `GameState` dataclass + peppi/libmelee adapters, pluggable
   Task/Verifier registry, corpus tagger to events.parquet, DuckDB
   sampler, single-frame rollout with factored-head logprobs, GRPO
   loss, frozen eval set + diffable JSON reports. 46 unit tests, all
   green.
2. **L-cancel RLVR result on Fox: BC 26.8% → RLVR 98.2%** (post step
   500). Non-overlapping 95% CIs ([0.231, 0.308] vs [0.966, 0.991]).
   +71.4 percentage points in 500 steps, KL climbed steadily (0 → ~6).
   Artifact: `checkpoints/fox-rlvr-lcancel-v1_step000500.pt`.
3. **Shield-escape RLVR result on Fox: BC 47.3% → RLVR 75.7%** at step
   300, non-overlapping CIs. The same framework absorbed a new skill
   in ~1 new file + ~200 LOC; no framework edits needed. Signal
   weaker than L-cancel (BC already at 47%) and pool smaller (1,474
   prompts vs L-cancel's 2.9M) — trained for 300 steps rather than
   2000, with higher `kl_beta=0.05`.
4. **Peppi external-ID bug in the ranked pipeline.**
   `tools/shard_and_upload_ranked.py` builds its character-name
   lookup as `{libmelee.Character.value → .name}` and then indexes
   it with peppi's **external** Melee character ID. Peppi 1 = DK,
   peppi 2 = FOX, but `CHAR_NAME[1]` returns `'FOX'`. Result: the
   FOX_master-master HF tarballs are ~70% **not Fox**. Training
   self-corrects (slp_to_shards re-filters with libmelee), so Fox
   checkpoints were trained on actual Fox data — just with 3× the
   download cost and misleading bucket names. RLVR's peppi adapter
   now remaps external → libmelee via a verified 26-entry lookup
   table.
5. **Rollback-frame duplicates in peppi .slp output.** ~38% of
   frame_ids in master-tier .slp files appear twice in peppi's
   columnar arrays (live-netcode prediction + rollback-corrected
   replay). libmelee's `Console.step()` silently keeps the first
   occurrence per frame_id. The RLVR peppi adapter matches that
   convention so its `GameState` output agrees with MIMIC's training-
   time view.
6. **Online RL infrastructure scaffolded.** New `rlvr/online/`
   package: `DolphinActor` drives a single headless Dolphin at
   realtime, streams libmelee `GameState`s, runs task-defined episode-
   start / episode-end state machines, buffers (obs, action, logprob_θ,
   logprob_ref) per frame. Multi-step PPO loss with group-normalized
   advantages replaces the single-step GRPO. L-cancel online task
   defers reward to a post-match peppi read of `post.l_cancel` at the
   landing frame. Shield-escape online task reads reward directly
   from libmelee state. Infrastructure standup confirmed (Dolphin
   boots, episodes detect, enrichment wires); full training run not
   yet executed.

## Offline RLVR: what got built

### Design

Map LLM RLVR onto Melee: prompt = state context, completion = one
sampled controller action from the factored 4-head policy, verifier =
pure skill checker, GRPO across the N samples per prompt. Single-frame
rollouts only — offline RLVR without a simulator cannot autoregress
past one frame because MIMIC's input at t+1 includes state that is a
physics response to the action at t (see
[feedback_offline_rlvr.md](../../../.claude/projects/-root-MIMIC/memory/feedback_offline_rlvr.md)
for the feedback memory).

Per-frame prompts: the tagger emits one row per frame inside a
skill-opportunity window (7 frames before landing for L-cancel, 3
frames after shield-damage for shield), not one row per opportunity.
Verifier grades one sampled action; reward signal is concentrated.

### Layout

```
rlvr/
├── state/          GameState + peppi / libmelee adapters
├── tasks/          Task + Verifier protocols + registry + per-skill modules
├── tagger/         replays -> events.parquet (peppi-parallel, ~750 replays/sec)
├── sampler/        events.parquet -> Prompt batches (DuckDB + LRU replay cache)
├── rollout/        single-frame batched rollout with factored-head logprobs
├── train/          GRPO loss (group-norm + PPO-clip + Schulman KL)
├── eval/           frozen eval set builder + runner with Wilson CIs
└── tests/          46 unit tests across 5 files
```

### L-cancel results (full run)

Training command:
```bash
python -m rlvr.train.loop \
  --base-ckpt checkpoints/fox-20260420-baseline-33k.pt \
  --data-dir hf_checkpoints/fox \
  --events data/rlvr/events_v0.1.parquet \
  --slp-dir data/fox_ranked_slp \
  --eval-set data/rlvr/eval_v0.1.parquet \
  --task l_cancel_opportunity \
  --run-name fox-rlvr-lcancel-v1 \
  --prompts-per-step 32 --rollouts-per-prompt 8 \
  --lr 1e-6 --kl-beta 0.01 --max-steps 2000 --eval-every 200
```

Eval progression (frozen 500-prompt eval set, 893 held-out replays):

| step | pass rate | 95% CI      |
|------|-----------|-------------|
| BC   | 0.268     | [0.231, 0.308] |
| 200  | 0.720     | —           |
| 400  | 0.954     | —           |
| 500  | 0.982     | [0.966, 0.991] |
| 600  | 0.986     | —           |

Stopped at step 500 once the milestone was decisively met; final
report at step 500 for the canonical artifact. The run would have
continued climbing toward ~1.0 — KL was steady at ~6 without clip_frac
firing.

### Shield-escape results

**Defining the skill** — "over-shielding" is the observed failure mode
(user shield-breaks the bot in live play). Master-tier Fox
shield-breaks are essentially nonexistent in the corpus (2 breaks
across 9,804 replays). Reframed to "escape from pressured shield":
player in SHIELD-family, shield dropped ≥8 strength in last 3 frames
(absorbed hit), current strength <30/60 (half max — visually small,
competitively meaningful). Verifier accepts any exit-from-shield
input: trigger release, grab (Z), jump (X/Y), up-smash OoS (A), roll
(main_x > 0.5), spotdodge (main_y < -0.5).

Training command:
```bash
python -m rlvr.train.loop \
  --base-ckpt checkpoints/fox-20260420-baseline-33k.pt \
  --data-dir hf_checkpoints/fox \
  --events data/rlvr/events_shield.parquet \
  --eval-set data/rlvr/eval_shield_v0.1.parquet \
  --task escape_pressured_shield \
  --run-name fox-rlvr-shield-v1 \
  --prompts-per-step 16 --rollouts-per-prompt 8 \
  --lr 1e-6 --kl-beta 0.05 --max-steps 300 --eval-every 50
```

Eval progression:

| step | pass rate | Δ over BC |
|------|-----------|-----------|
| BC   | 0.473     | —         |
| 50   | 0.473     | +0.000    |
| 100  | 0.500     | +0.027    |
| 150  | 0.540     | +0.067    |
| 200  | 0.597     | +0.124    |
| 250  | 0.677     | +0.204    |
| 300  | 0.757     | +0.284    |

Non-overlapping CIs: BC [0.418, 0.530] vs RLVR [0.705, 0.802]. Signal
accelerates in later steps — likely hasn't saturated. Would continue
improving but the 1,474-prompt pool means each prompt is seen ~3×
already at step 300; overfit risk rises beyond.

### Framework validation

Adding shield-escape as a second task required:
- 1 new file (`rlvr/tasks/escape_pressured_shield.py`, ~120 LOC)
- 1 new test file (`rlvr/tests/test_escape_shield_fixtures.py`, 20 fixtures)
- 1 registration line in `rlvr/tasks/__init__.py`
- 1 generalization in `rlvr/eval/build_set.py` (L-cancel-specific
  stratification → uniform sampling fallback for tasks without an
  aerial-action-state field)

No changes to state / tagger / sampler / rollout / GRPO / runner. The
"adding a new task is a single-file change" success criterion from
the original spec holds.

## Peppi vs libmelee ID conventions

### Character IDs

Peppi returns **external** Melee character IDs (CSS-order). libmelee's
`Character` enum uses a different mapping. Verified empirically against
50 master-master replays via `port-matched` cross-check:

| peppi (external) | libmelee value | name          |
|------------------|----------------|---------------|
| 0                | 2              | CPTFALCON     |
| 1                | 3              | DK            |
| 2                | 1              | FOX           |
| 6                | 6              | LINK (happens to match) |
| 7                | 17             | LUIGI         |
| 9                | 18             | MARTH         |
| 14               | 10             | POPO (Ice Climbers) |
| 15               | 15             | JIGGLYPUFF (matches) |
| 17               | 14             | YOSHI         |
| 19               | 7              | SHEIK         |
| 20               | 22             | FALCO         |
| ...              | ...            | (full table in adapter) |

**Bug:** `tools/shard_and_upload_ranked.py` builds
`CHAR_NAME = {c.value: c.name for c in melee.Character}` and looks up
`CHAR_NAME[peppi.character]`. Because peppi returns external IDs,
`CHAR_NAME[1]` returns `'FOX'` but the actual character is DK. Every
ranked tarball is named after the wrong character. Example sample from
one "FOX_master-master" tar:

| replay | peppi chars | libmelee reports | has Fox? |
|--------|-------------|------------------|----------|
| 01b5757e... | [1, 14]     | DK vs POPO       | no |
| 026b5c0e... | [6, 1]      | LINK vs DK       | no |
| 04a87027... | [1, 2]      | DK vs FOX        | yes |
| 05f28047... | [1, 9]      | DK vs MARTH      | no |

The training pipeline is self-correcting (`slp_to_shards.py --character
1` re-filters with libmelee, keeping only actual Fox frames), so the
models trained on real Fox data. But:
- The HF dataset's per-character tarballs are mis-labeled — anyone
  downloading `FOX_*.tar.gz` gets mostly non-Fox replays.
- Bandwidth is ~3× wasted on the retrain pipeline.
- Anyone reusing these tarballs for non-MIMIC work would get wrong
  data.

Fix is a one-line change in `shard_and_upload_ranked.py`: build
`CHAR_NAME` indexed by external character ID, not libmelee enum value.
RLVR peppi adapter has a verified remap and is unaffected.

### Stage IDs

Same issue. Peppi stage 28 = libmelee 26 = DREAMLAND. Six-entry remap
in the RLVR adapter covers all tournament-legal stages.

### Action state IDs

**Do** match between peppi and libmelee (both read the raw SSBM Action
State ID from the .slp event stream). No remap needed. Verified on
PLATFORM_DROP (244) and the FAIR/BAIR/DAIR landing pairs.

### Rollback duplicates

Peppi exposes every row in the .slp event stream including live-netcode
rollback corrections: for ~38% of frame_ids in master-tier replays,
the frame appears twice (predicted + corrected). libmelee's
`Console.step()` silently keeps the first occurrence per frame_id. The
RLVR peppi adapter reproduces that behavior via a unique-first-index
dedup at load time, so `Replay.frame_ids` has no duplicates and all
columnar arrays are aligned to the same canonical frames as libmelee.

Caught by the roundtrip test: on the raw (un-deduped) arrays, peppi
reported action=24 (KNEE_BEND) while libmelee reported 14 (STANDING)
for the same frame_id; dedup resolved the disagreement.

## Online RL: infrastructure standup

### Why online

Offline RLVR is fundamentally limited to skills definable as "at
frame t in state S, action A is correct" — input-pattern skills
(L-cancel, shield-escape, dashdance, perfect-pivot, tech options).
Skills whose reward is defined by what happens **after** the policy's
action (combos, edgeguards, tech chases, punishes) are state-outcome
skills and need a simulator in the loop. Any reward that reads
"percent dealt," "stocks taken," "did the tech succeed," can't be
computed from a single frame of context.

### Design

Single-actor for Phase 1, headless Dolphin via libmelee, live-detected
mini-episodes (no savestates in Phase 1). Everything else reused from
the offline pipeline: `GameState`, rollout sampling math, factored-
head decode, GRPO math extended to multi-step.

```
rlvr/online/
├── dolphin_actor.py          # Dolphin process + libmelee step loop + episode buffer
├── episode.py                # OnlineTask Protocol + EpisodeOutcome
├── trajectory.py             # Episode + FrameRecord + group-normalize
├── ppo.py                    # multi-step PPO: per-frame advantage from episode return
├── loop.py                   # collect -> ppo_update -> checkpoint
└── tasks/
    ├── l_cancel_online.py    # episode on aerial-enter, end on landing,
    │                         # reward from post.l_cancel via post-match .slp parse
    └── shield_escape_online.py  # episode on pressured shield, end on exit,
                              # reward from live libmelee state (no .slp parse)
```

### Key design decisions

**Reward for online L-cancel uses ground truth from peppi.** libmelee
doesn't expose the engine's `l_cancel` column. Instead of inferring
from landing-state duration (brittle), the actor buffers pending
episodes during the match and calls `task.enrich_with_replay(episodes,
slp_path)` at match end. The task re-parses the .slp Dolphin just
wrote, looks up `post.l_cancel` at the recorded landing frame, and
sets the terminal reward (1 = success, 0 = miss, episode discarded if
the landing wasn't L-cancel-eligible). Same ground-truth signal as
offline, verified against the live game engine.

**Shield-escape has no deferred reward** — every signal the task
needs (shield strength trajectory, SHIELD-family / ESCAPE-family /
BREAK action states) is on libmelee's `PlayerState`. Reward assigned
at `compute_outcome` immediately.

**Per-frame T-context snapshots.** The PPO update needs to re-forward
the policy with gradients on the exact context window each action was
sampled under. The actor stores the full `T×feat` tensor dict per
`FrameRecord` (~100KB per frame, on CPU) so the update can rebuild
the batch without replaying the state stream. Wasteful on memory but
mechanically correct — no subtle "wrong context" bug.

**Dolphin config matches `tools/play_vs_cpu.py` exactly:**
`gfx_backend=""` (empty, inherits Dolphin default), `disable_audio=False`,
`blocking_input=True`. An earlier attempt with `gfx_backend="Vulkan"` +
`disable_audio=True` hung — console.step() never returned. The
empirical rule is: match the known-working inference configs.

### Standup status

- Imports clean; dataclasses instantiate.
- Dolphin boots from the in-tree binary (`emulator/squashfs-root/usr/bin/dolphin-emu`)
  under Xvfb:99.
- Heartbeat logging confirms menu navigation advances through CHARACTER_SELECT
  → STAGE_SELECT → IN_GAME in ~10s.
- Episode detection fires: a 60s Fox-vs-Fox (CPU level 9) match on FD
  produced 17 aerial→landing episodes in live detection.
- Match-end enrichment hook wired; not yet exercised (smoke runs were
  stopped mid-match).
- Full training loop not yet executed — next session's work.

### Open items for next session

- Run first full online L-cancel training pass, compare final pass
  rate vs offline RLVR's 98.2%. Expectation: at least parity, possibly
  better (episode-level credit vs single-frame credit).
- Shield-escape online: same exercise.
- Savestate seeding (Phase 3) — only if realtime throughput is the
  bottleneck. 17 episodes/minute on single actor = ~20 min to collect
  a 300-episode update batch; should be OK for short training runs.
- Parallel actors (Phase 4) — out of scope until single-actor is
  learning.
