# rlvr — MIMIC RLVR suite

Verifiable-reward reinforcement learning post-training for MIMIC.

## Layout

```
rlvr/
├── state/     canonical GameState + peppi/libmelee adapters
├── tasks/     Task + Verifier protocols and per-skill modules
├── tagger/    replays -> events.parquet
├── sampler/   events.parquet -> Prompt batches
├── rollout/   single-frame batched rollout + log-probs
├── train/     GRPO loss + training driver
├── eval/      frozen eval set + runner (JSON report)
└── tests/     unit tests (pytest rlvr/tests)
```

## Data contract

- `GameState` mirrors MIMIC's 13-numeric + 5-flag per-player schema so the
  rollout's frame builder can feed the MIMIC encoder without drift.
- Peppi uses external Melee character/stage IDs; we remap to libmelee's
  enum values at adapter load time (see
  `rlvr/state/peppi_adapter.py:_PEPPI_TO_LIBMELEE_CHAR`).
- Rollback-duplicated frames in peppi are deduped to the first
  occurrence per frame_id (matches libmelee's Console.step() convention).

## Key design decisions

- **Single-frame rollouts only.** Offline RLVR without a simulator
  cannot autoregress past 1 frame because MIMIC's input at frame t+1
  includes state that is a physics response to the action at frame t.
  Multi-frame rollouts would feed the model a state trajectory
  inconsistent with its own actions.
- **Per-frame prompts.** A 7-frame L-cancel window produces 7 prompts,
  not 1. Each is scored independently.
- **Verifier is a pure function of the sampled action.** The tagger
  has already certified via `post.l_cancel != 0` that a frame is inside
  an eligible window, so the verifier doesn't need to look at future
  states.
- **GRPO only**, no value head. Advantages = group-normalized rewards
  within each prompt's N samples.

## Adding a new task

Five-step checklist:

1. Create `rlvr/tasks/<task_id>.py` with two classes:
   - `MyTask`: implements `tag_frames(replay)` — yield a `FrameRow`
     per prompt-relevant frame. Only this method looks ahead.
   - `MyVerifier`: implements `__call__(prompt, sampled_ctrl) -> float`
     in [0, 1]. Pure; no side effects.
2. Both must share a `task_id` string. At module scope, call
   `register_task(MyTask(), MyVerifier())`.
3. Import the module in `rlvr/tasks/__init__.py` so it registers on
   package import.
4. Write a test file `rlvr/tests/test_<task_id>_fixtures.py` covering
   verifier fixtures (positive + negative), tagger fixtures (synthetic
   short-history edge cases), and at least one regression test against
   a real .slp with known expected output.
5. Re-run the tagger corpus scan to generate events.parquet rows for
   the new task. Training/eval pick up the new task_id automatically
   via the registry.

No framework changes needed.

## Running the first milestone

```bash
# Build the prompt dataset (one-shot)
python -m rlvr.tagger.scan \
  --slp-dir data/fox_ranked_slp \
  --out data/rlvr/events_v0.1.parquet \
  --task l_cancel_opportunity \
  --character FOX

# Build the frozen eval set (one-shot)
python -m rlvr.eval.build_set \
  --events data/rlvr/events_v0.1.parquet \
  --out data/rlvr/eval_v0.1.parquet \
  --n 500

# BC baseline report
python -m rlvr.eval.runner \
  --ckpt checkpoints/fox-20260420-baseline-33k.pt \
  --eval-set data/rlvr/eval_v0.1.parquet \
  --data-dir hf_checkpoints/fox \
  --slp-dir data/fox_ranked_slp \
  --out reports/bc_fox_v0.1.json

# Train
python -m rlvr.train.loop \
  --base-ckpt checkpoints/fox-20260420-baseline-33k.pt \
  --data-dir hf_checkpoints/fox \
  --events data/rlvr/events_v0.1.parquet \
  --slp-dir data/fox_ranked_slp \
  --eval-set data/rlvr/eval_v0.1.parquet \
  --task l_cancel_opportunity \
  --run-name fox-rlvr-lcancel-v1 \
  --lr 1e-6 --max-steps 2000

# Post-RLVR report
python -m rlvr.eval.runner \
  --ckpt checkpoints/fox-rlvr-lcancel-v1_final.pt \
  --eval-set data/rlvr/eval_v0.1.parquet \
  --data-dir hf_checkpoints/fox \
  --slp-dir data/fox_ranked_slp \
  --out reports/rlvr_fox_v0.1.json

# Diff
diff reports/bc_fox_v0.1.json reports/rlvr_fox_v0.1.json
```

## Versioning & compatibility

Every persisted artifact stamps `rlvr_schema_version` (gamestate dataclass
version), `rlvr_registry_hash` (hash of task registry entries), and
relevant seed / sample config. Loaders should assert compatibility
rather than silently using mismatched artifacts.
