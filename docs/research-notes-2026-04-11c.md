# Research Notes — 2026-04-11c: Gamestate Leak Fix (Pre-frame Alignment)

## Overview

Found and fixed the root cause of why trained models couldn't initiate actions
during inference. The .slp replay format has two snapshots per frame (pre-frame
and post-frame), and melee-py's `console.step()` returns **post-frame** game
state alongside **pre-frame** controller inputs. This means the game state at
frame `i` already reflects the button press at frame `i` — the model can read
the answer from the action embedding instead of learning when to initiate.

**Impact:** JUMP onset leak rate dropped from 58.2% to 1.9%. The model can no
longer cheat by memorizing action→button mappings.

---

## 1. The Problem: Post-Frame Game State Leaks Button Presses

### What .slp files contain

Each frame in a Slippi replay has two data snapshots:

| Snapshot | Contains | When |
|----------|----------|------|
| **pre-frame** | Controller inputs (buttons, sticks, shoulders) + game state BEFORE engine processes | Before game logic runs |
| **post-frame** | Game state AFTER engine processes inputs | After game logic runs |

Verified by reading raw .slp data with py-slippi:

```
Frame 418 (ground jump from dashing):
  pre.state  = 20  (DASHING)     ← state before jump processed
  pre.buttons = Y                 ← the jump input
  post.state = 24  (KNEE_BEND)   ← state after jump processed
```

### What melee-py returns

`console.step()` returns a `PlayerState` that combines:
- **`ps.action`, `ps.position`, `ps.percent`, `ps.stock`, etc.** — from `__post_frame` (after engine)
- **`ps.controller_state` (buttons, sticks, shoulders)** — from `__pre_frame` (the inputs)

Verified by reading melee-py source:
- `console.py:1149-1153` — `__post_frame` sets `playerstate.action`
- `console.py:1057-1130` — `__pre_frame` sets `playerstate.controller_state`

### What the old shards stored

`slp_to_shards.py` called `_write_player(arrays, wmap, fi, ps, port)` which
wrote both game state AND controller from the same `ps` object. Result:

```
Shard frame i:
  states.self_action  = post-frame action (KNEE_BEND on jump frame)
  states.self_controller = pre-frame buttons (JUMP on jump frame)
  targets.btns_single = pre-frame buttons (JUMP on jump frame)
```

The game state already contains the answer. 58.2% of JUMP onsets had
`self_action=KNEE_BEND` (action 24) on the same frame as `btns_single=JUMP`.

### The inspect_frame.py evidence

Using `tools/inspect_frame.py` to examine frame 534 (a STANDING→JUMP transition):

- **Frame 533**: action=14 (STANDING), target=NONE, prediction=NONE (1.0000)
- **Frame 534**: action=24 (KNEE_BEND), target=JUMP, prediction=JUMP (0.9957)

The ONLY input that changed between frames 533 and 534 was `self_action`
(14→24). Same position, same percent, same flags, same controller input
(still NONE from prev frame). The model keyed entirely off the action
embedding — it learned KNEE_BEND→JUMP, not "I should jump now."

At inference, to GET action=24, Fox already needs to have jumped. But the
model won't predict JUMP until it sees action=24. Chicken-and-egg.

---

## 2. The Fix: Shift Targets Forward by 1 Frame

### Approach

In `slp_to_shards.py`, after building states and targets for each game, shift
targets forward by 1 frame:

```python
shifted_targets = {k: v[1:] for k, v in targets.items()}
shifted_states = {k: v[:-1] for k, v in states.items()}
n_frames_shifted = n_frames - 1
```

This drops the last frame of each game (no next-frame target available).

### New shard alignment

For frame `i` in the new shard:
- **Game state**: post-frame `i` (current situation)
- **Controller in states**: frame `i`'s inputs (what was pressed to get here)
- **Target**: frame `i+1`'s inputs (what to press next)

### Verification

JUMP onset leak rate: **58.2% → 1.9%**

Before fix (old shard):
```
frame 338: action=24 (KNEE_BEND), target=JUMP  ← leaked
frame 361: action=24 (KNEE_BEND), target=JUMP  ← leaked
frame 534: action=24 (KNEE_BEND), target=JUMP  ← leaked
```

After fix (new shard):
```
frame 337: action=350 (LANDING), target=JUMP   ← clean
frame 360: action=20  (DASHING), target=JUMP   ← clean
frame 421: action=56  (ATTACK),  target=JUMP   ← clean
```

The remaining 1.9% are aerial double-jumps where KNEE_BEND legitimately
coincides with a new jump input (different mechanism than ground jumps).

---

## 3. Training Implications

### No controller offset or reaction delay needed

With targets already shifted in the shard:
- `--reaction-delay 0` — targets are already next-frame
- No `--controller-offset` — the controller in states[i] is frame i's input,
  which IS the "previous action" relative to the target (frame i+1)
- No `--self-inputs` needed

The shard alignment matches inference exactly:
- **Inference**: model sees current game state + last frame's inputs → predicts what to press
- **Training**: model sees frame i's game state + frame i's inputs → target is frame i+1's inputs

### Training command

```bash
python3 train.py \
  --model hal --encoder hal_flat \
  --hal-mode --hal-minimal-features --hal-controller-encoding \
  --stick-clusters hal37 --plain-ce \
  --lr 3e-4 --batch-size 512 \
  --max-samples 16777216 \
  --data-dir data/fox_hal_v2 \
  --reaction-delay 0 \
  --run-name hal-7class-v2 \
  --no-warmup --cosine-min-lr 1e-6
```

### Expected metric changes

Val loss will likely be HIGHER than the old rd=0 runs (0.743) because the
model can no longer cheat by reading the action embedding. The old low val
loss was misleading — it reflected the model's ability to copy
action→button, not its ability to play. Higher loss with correct alignment
should translate to better gameplay.

---

## 4. How HAL Handles This

HAL has the **same shard alignment** (post-frame game state + same-frame
buttons). HAL uses `reaction_delay=1` at dataloader time, which shifts the
target forward by 1 frame — achieving the same effect as our shard-level
shift. Both approaches result in the model predicting next-frame inputs
given current game state.

The difference: we bake the shift into the shard so rd=0 "just works" and
no offset flags are needed. HAL does it at dataloader time.

---

## 5. Earlier Findings (Bistable Inference)

Before discovering the gamestate leak, we analyzed 14 saved inference replays
and found the model was **bistable** — sometimes playing well (40%+ button
presses, diverse actions) and sometimes stuck (90%+ idle). This was diagnosed
as a symptom but the root cause was the gamestate leak making the model
unable to initiate actions from idle states.

Key replay analysis findings (using py-slippi, corrected for IntFlag parsing):

| Mode | Replays | Button rate | Idle% | Behavior |
|------|---------|-------------|-------|----------|
| Active | 4 games | 20-42% | 2-8% | Rapid jab, grab, shield, dodge |
| Stuck | 3 games | 0-5% | 64-92% | Standing, crouching |
| Mixed | 2 games | 4-15% | 21% | Partially responsive |

The active-mode games showed the model CAN play when it bootstraps past the
idle attractor. The stuck-mode games showed what happens when it can't.

---

## 6. Tools Added

### tools/inspect_frame.py

Shows exactly what goes into and comes out of the model for any frame:

```bash
python tools/inspect_frame.py \
  --checkpoint checkpoints/hal-7class-relpos_best.pt \
  --data-dir data/fox_hal_full \
  --shard 0 --frame 534 --context 2
```

Displays: all encoder inputs (categoricals, 9 numeric features per player in
HAL order, controller one-hot), targets, and predictions with top-5 button
probs. Critical for debugging train/inference mismatches.

### Gamestate leak check in train.py

At startup with `--hal-mode` and `rd=0`, samples the first shard and reports
what percentage of JUMP onsets already show KNEE_BEND in `self_action`. Warns
if >50%.

---

## 7. Data Directories

| Directory | Alignment | Use |
|-----------|-----------|-----|
| `data/fox_hal_full` | **Old** — same-frame targets (leaked) | Do not use for rd=0 |
| `data/fox_hal_v2` | **New** — next-frame targets (clean) | Use with rd=0, no offset |
| `data/fox_hal_match_shards` | Old — same-frame targets | Do not use for rd=0 |

---

## 8. Common Pitfalls (Updated)

1. **Post-frame vs pre-frame**: melee-py's `console.step()` returns post-frame
   game state (action, position, percent) but pre-frame controller inputs.
   The game state at frame `i` already reflects frame `i`'s button presses.
   Shards built without the target shift have this leak.

2. **Don't use `--controller-offset` with v2 shards**: The v2 shards already
   have the correct alignment. Adding controller offset would double-shift
   and misalign the data.

3. **Don't use `--reaction-delay 1` with v2 shards**: Same reason — the shift
   is already in the shard. rd=1 would shift again, making the model predict
   two frames ahead.

4. **py-slippi IntFlag parsing**: `buttons.physical.A` returns the flag
   constant (256), not a boolean. Use `bool(phys & Buttons.Physical.A)` or
   check the raw `int(phys)` bitmask.

5. **Val loss will be higher with v2 shards**: This is expected and correct.
   The old low val loss (0.743) was inflated by the gamestate leak. The model
   was memorizing action→button mappings, not learning to play.
