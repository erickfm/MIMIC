# Research Notes — 2026-03-31 through 2026-04-02

## Context

Continued from [2026-03-21](research-notes-2026-03-21.md). After establishing that MIMIC hits an 88% val btn_f1 ceiling regardless of hyperparameters, we pivoted to reproducing HAL's exact approach — same data, same architecture, same training paradigm — to isolate what makes HAL work in closed-loop while MIMIC doesn't.

---

## Phase 1: Running HAL's Actual Code (2026-03-30 — 2026-03-31)

### GPU status check

| Machine | GPUs | Status |
|---------|------|--------|
| C (194.14.47.19:22874) | 8× RTX 5090 | Running FRAME Falco training (4 jobs) |
| E (66.222.138.178:11335) | 8× RTX 5090 | Idle |
| F (74.2.96.10:18619) | 8× RTX 5090 | Running FRAME Fox training (plateaued) |
| D (50.145.48.107) | 8× H200 | Killed (CUDA zombie issues) — removed from fleet |

FRAME runs on C and F were plateaued at the same 87% val bf1 / 44% mf1 ceiling documented in Finding 9.

### Cloning and running HAL

Cloned Eric Gu's HAL repo (`github.com/ericyuegu/hal`) on Machine E. Required fixes:
- `join=False` → `join=True` in `distributed.py` (mp.spawn bug — parent process didn't wait for children)
- Commented out `clean_stale_shared_memory()` (caused shared memory race condition with DDP)
- Fixed bare imports in `process_replays.py` and `calculate_stats.py`
- Changed `MASTER_PORT` and `REPO_DIR` paths
- Fixed `streaming_dataloader.py` logger lines that crashed when using `--data.data_dir`
- Commented out `AWS_BUCKET` assert in `streams.py`

### Data: Fox replays from HuggingFace

Downloaded 3,230 Fox .slp replay files from `erickfm/slippi-public-dataset-v3.7` (23GB dataset, ~10k replays total, 3.2k with Fox in filename). Processed through HAL's `process_replays.py` → MDS format: 2,830 valid replays, 26.9M frames, 8 train shards (406MB compressed).

Note: the HF dataset only had ~10k of the stated 95k replays (incomplete upload). The full public dataset archive (29GB .7z) has 94,487 replays.

### HAL training results

Trained HAL's `GPTv5Controller-512-6-8-dropout` on 8× RTX 5090 with `mp.spawn`:
- `--n_gpus 8 --local_batch_size 64 --n_samples 16777216 --n_val_samples 4096`
- Config: `baseline_controller_fine_main_analog_shoulder_early_release`
- 37 hand-designed stick clusters, 9 c-stick clusters, 5-class single-label buttons, 3-class combined shoulder

Val loss trajectory (wandb: `erickfm/hal/l8wtjjdk`):

| Checkpoint | Val Loss | Buttons Loss | Main Stick Loss |
|------------|----------|-------------|-----------------|
| 1 (524k) | 0.862 | 0.143 | 0.571 |
| 5 (2.6M) | 0.782 | 0.126 | 0.523 |
| 10 (5.2M) | **0.744** | 0.119 | **0.497** |
| 18 (9.4M) | 0.802 | 0.129 | 0.545 |

Best val loss 0.744 at checkpoint 10 (~5.2M samples). Model began overfitting after that — val loss rose while train loss continued falling. Expected with only 26.9M frames.

### HAL eval results: per-head metrics (checkpoint at 3.1M samples)

| Head | Val Accuracy | Val Loss | Macro F1 |
|------|-------------|----------|----------|
| buttons | 95.9% | 0.121 | 88.0% |
| main_stick | 86.5% | 0.514 | **49.5%** |
| c_stick | 98.9% | 0.039 | 76.4% |
| shoulder | 97.7% | 0.079 | 87.7% |

Main stick macro F1 is 49.5% — same ballpark as MIMIC's 42-44%. Both models struggle with rare stick positions (long tail problem). HAL's 37 hand-designed clusters don't solve this.

### HAL closed-loop eval: 4-stocked level 9 CPU

Ran HAL's `eval.py` on the best checkpoint (5.2M samples) against level 9 Fox CPU on Final Destination:

| | P1 (HAL AI) | P2 (Level 9 CPU) |
|---|---|---|
| Stocks lost | 1 | **4** (all) |
| Damage dealt | 596 | 231 |
| Frames | 10,666 (~3 min) |

**HAL won decisively** despite similar offline metrics to MIMIC. This proves the architecture works — the question is what MIMIC is doing differently.

---

## Phase 2: Making MIMIC Match HAL (2026-03-31 — 2026-04-01)

### Code changes (all toggle-based)

| Flag | What it does | Commit |
|------|-------------|--------|
| `--stick-clusters hal37` | Use HAL's 37 hand-designed stick clusters with runtime re-clustering | `9f3d6bb` |
| `--no-warmup` | Skip LR warmup (HAL uses none) | `9f3d6bb` |
| `--hal-minimal-features` | Drop ECB/speeds/hitlag from numeric (HAL's exact 7 per player) | `b1722df` |
| `--controller-offset` | Shift self-controller input by -1 frame (HAL-style) | `e68e7a4` |
| `--reaction-delay 0` | Predict current frame's action (not next frame's) | `9346d33` |

Inference fixes:
- Pre-fill context window to seq_len on first frame (model never saw T<256 during training)
- Use game readback for controller feedback in HAL mode (not argmax predictions)
- Multinomial sampling for all outputs (HAL uses `torch.multinomial`, not argmax)
- Clamp numeric features to [-10, 10] after normalization (garbage ECB values)
- HAL-mode button/shoulder decoding (auto-detected from output shape)

### Full HAL-exact MIMIC command

```bash
torchrun --nproc_per_node=8 train.py \
  --model hal --hal-mode --self-inputs --encoder flat \
  --seq-len 256 --lr 3e-4 --no-warmup --cosine-min-lr 1e-6 \
  --plain-ce --dropout 0.2 --no-amp --batch-size 64 \
  --stick-clusters hal37 --lean-features --no-compile \
  --reaction-delay 0 --controller-offset \
  --data-dir data/fox_public_shards --seed 42 \
  --run-name mimic-hal-fox-co
```

### Training data: same Fox replays as HAL

Downloaded the same 3,230 Fox .slp files from `erickfm/slippi-public-dataset-v3.7` on Machine E. Tensorized using `tools/slp_to_shards.py` → 19 .pt shards with norm_stats, cat_maps, stick_clusters.

---

## Key Findings

### Finding 10: reaction_delay alignment matters enormously

MIMIC's data has `state[i] == target[i]` — the self-controller buttons at frame i are identical to the target at frame i. This means:

- `reaction_delay=1`: input is frame i, target is frame i+1. Model predicts "what to do next frame." The self-controller at frame i shows what IS being pressed, which is the answer for the current frame but not for the target frame. Reasonable task.
- `reaction_delay=0`: input is frame i, target is frame i. The self-controller shows the exact answer. Trivially easy — model gets 100% by copying input.

**HAL's approach** (`frame_offset=-1` for controller only, target offset=0):
- At position i: gamestate from frame i, controller from frame **i-1**, target is frame **i's** controller
- Model sees "what I was pressing last frame" and predicts "what to press now"
- The answer is NOT in the input (frame i-1's controller ≠ frame i's controller at transitions)

**Our fix**: `--reaction-delay 0 --controller-offset`
- `reaction_delay=0`: target is same frame
- `--controller-offset`: shifts `self_buttons`, `self_analog`, `self_c_dir` by -1 within each window
- At position i: sees controller from frame i-1, predicts frame i's controller

Verified with data inspection:
```
Frame 3048 (target X goes 0→1):
  WITHOUT offset: sees_X=1, target_X=1 → answer in input (BAD)
  WITH offset:    sees_X=0, target_X=1 → must predict new press (CORRECT)
```

### Finding 11: Context window pre-fill is critical for closed-loop

MIMIC's inference started with T=1 frame input and grew to T=256. The model was trained on fixed T=256 windows and had never seen T<256. With T=1, the model produced degenerate output (NONE=99.9%).

Fix: pre-fill the context window with copies of the first frame, so the model always sees T=256 from the start. This matches HAL which initializes its context window to full seq_len.

### Finding 12: Argmax vs multinomial sampling is the difference between "stands still" and "plays"

HAL uses `torch.multinomial(softmax(logits), 1)` for ALL outputs (buttons, sticks, shoulder). MIMIC used `torch.argmax` (via `--deterministic`).

With 73% of training frames being NONE, the model's button distribution is always NONE-heavy (typically NONE=97%, A=1%, B=1%, X=0.5%, Z=0.5%). Argmax always picks NONE → bot stands still. Multinomial occasionally picks A/B/X/Z → bot takes actions → game state changes → model sees new situations → more diverse predictions.

With multinomial sampling: 148 button presses and 1,137 stick movements in a match (vs 0 with argmax).

### Finding 13: Prediction feedback creates a poisoning loop

MIMIC's inference fed back the **argmax-decoded** prediction as the next frame's self-controller input. Since the prediction was always NONE (argmax), the model always saw "I pressed nothing last frame" → predicted "press nothing" → self-reinforcing loop.

HAL doesn't do this — it reads controller state from the **game engine**, which reports what was actually sent to Dolphin. The game readback provides the correct feedback signal.

Fix: use game readback for controller feedback in HAL mode.

### Finding 14: ECB values are garbage in live games

The `self_ecb_top_x` feature returned ~7 trillion from the live game (column 17 of self_numeric). The norm stats have `[mean=0, std=1]` for ECB columns, so this value passes through unchanged. A single 7-trillion input completely corrupts the model's attention and prediction.

Training data has clean ECB values (from parsed replay files). Fix: clamp all numeric features to [-10, 10] after normalization.

---

## Training Results Summary

| Run | reaction_delay | controller_offset | Best Val Loss | bf1 | mf1 |
|-----|---------------|-------------------|---------------|-----|-----|
| HAL (Gu's code) | N/A (offset=-1, target=0) | N/A | 0.744 | ~88% | ~50% |
| MIMIC rd=1 (Falco) | 1 | No | 0.664 | 87.8% | 56% |
| MIMIC rd=1 (Fox) | 1 | No | 0.707 | 87.8% | 54.5% |
| MIMIC rd=0 (Fox) | 0 | No | 0.078 | 100% | 87% |
| MIMIC rd=0+offset (Fox) | 0 | **Yes** | 0.63* | 91.4%* | 53.6%* |

*Still training at time of writing (step 11k/31k).

The rd=0 without offset gives trivially high metrics (answer in input). The rd=0+offset run produces metrics comparable to HAL while correctly hiding the answer.

---

## Closed-Loop Eval Results

| Model | Config | Pressed buttons? | Beat CPU? |
|-------|--------|-----------------|-----------|
| HAL (Gu's code) | HAL eval.py | Yes | **Yes** (4-stocked) |
| MIMIC rd=1 + argmax | Old inference | No (NONE 99.9%) | No |
| MIMIC rd=1 + sampling + prefill + readback | Fixed inference | **Yes** (148 presses) | No (Dolphin disconnected) |
| MIMIC rd=0 (no offset) + sampling | Fixed inference | No (NONE 100%) | No |
| MIMIC rd=0+offset | Not yet tested | — | — |

The rd=0+offset model is currently training. Will test closed-loop when done.

---

## Parallel Work: Dataset Upload to HuggingFace

### Ranked dataset (`erickfm/melee-ranked-replays`)

6 archives of fizzi's ranked anonymized replays (platinum+, ~850k total replays):
- Archives downloaded from Google Drive using Firefox cookie auth (solved rate limiting)
- Machines C, F, and G (38.65.239.14:30709) downloaded and processed in parallel
- Character+rank sharding: parse each .slp for characters, group by (character, rank_pair), compress to tar.gz
- Character normalization: Zelda/Sheik → ZELDA_SHEIK, Popo/Nana → ICE_CLIMBERS
- Each replay appears in both characters' shards (~90% duplication, fits within 4.91TB HF quota)

Status: sharding in progress on all machines. Upload pending.

### Slippi public dataset (`erickfm/slippi-public-dataset-v3.7`)

94,487 .slp files extracted from 29GB .7z archive on Machine G. Upload initiated via `upload_large_folder`. Currently has ~10k files on HF (incomplete from earlier upload).

---

## Architecture Reference: HAL vs MIMIC Exact Diff

| Feature | HAL | MIMIC (with all flags) |
|---------|-----|----------------------|
| d_model / layers | 512 / 6 | 512 / 6 (`--model hal`) |
| Frame encoder | Flat concat → linear | Flat (`--encoder flat`) |
| Position encoding | Relative (Music Transformer skew) | RoPE |
| Buttons | Single-label 5-class softmax | Single-label 5-class (`--hal-mode`) |
| Shoulder | Combined max(L,R), 3-class | Combined 3-class (`--hal-mode`) |
| Stick clusters | 37 hand-designed | 37 HAL (`--stick-clusters hal37`) |
| C-stick | 9 clusters | 5 cardinal (not yet matched) |
| Loss | Plain CE | Plain CE (`--plain-ce`) |
| LR | 3e-4, no warmup, cosine to 1e-6 | 3e-4 (`--lr 3e-4 --no-warmup --cosine-min-lr 1e-6`) |
| Dropout | 0.2 | 0.2 (`--dropout 0.2`) |
| AMP | FP32 | FP32 (`--no-amp`) |
| Self-inputs | Controller from frame i-1 | Controller from frame i-1 (`--self-inputs --controller-offset`) |
| Target | Frame i controller | Frame i (`--reaction-delay 0`) |
| Input features | 9 numeric per player (no ECB/speeds) | 22 numeric per player (includes ECB/speeds) |
| Nana/projectiles | Not included | Dropped (`--lean-features`) |

**Remaining unmatched:**
- Position encoding (RoPE vs relative) — likely minor
- C-stick (5 vs 9 clusters) — minor
- Input feature set (22 vs 9 numeric per player) — `--hal-minimal-features` exists but needs runtime feature masking (not yet implemented)

---

## Machine Status (end of session)

| Machine | Task | Status |
|---------|------|--------|
| E | MIMIC training `mimic-hal-fox-co` (rd=0, offset) | Running, step ~11k/31k |
| C | Sharding ranked archives 3,4,5 | In progress |
| F | Sharding ranked archive 6 | In progress |
| G | Sharding ranked archives 1,2 + slippi public upload | In progress |

## Next Steps

1. Evaluate `mimic-hal-fox-co` checkpoint in closed-loop against level 9 CPU
2. If still not matching HAL, implement runtime feature masking for `--hal-minimal-features` (drop ECB/speeds at the tensor level, not the model level)
3. Match C-stick clusters (9 vs 5) and position encoding (relative vs RoPE) if needed
4. Scale to more data (ranked dataset upload in progress, ~850k replays)
