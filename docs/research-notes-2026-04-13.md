# Research Notes — 2026-04-13: The Wavedash / TRIG button bug

## Summary

A silent bug in `tools/inference_utils.py:decode_and_press` had been preventing
the 7-class models from ever executing airdodge, wavedash, L-cancel, tech, or
shield-grab — for the entire life of the 7-class inference code.

The TRIG button class (`btn_idx == 4`) appended `"TRIG"` to a label list but
never called `ctrl.press_button(BUTTON_L)`. Only the analog shoulder value was
sent via `press_shoulder(BUTTON_L, value)`. Analog shoulder is enough for
shielding but NOT for any rising-edge event. The game interpreted the inputs as
normal jumps / aerials and the wavedash landing animation (LANDING_FALL_SPECIAL)
never appeared in any replay.

Fix: two lines in `decode_and_press`. TRIG now calls
`ctrl.press_button(BUTTON_L)`; A_TRIG (class 5, shield-grab) now calls both
`press_button(BUTTON_A)` and `press_button(BUTTON_L)`.

**Post-fix ditto results (Final Destination, temperature=1.0):**

| Match | Winner | P1 wavedashes | P2 wavedashes |
|---|---|---|---|
| Falco ditto | P1 2-0 | 34 / 104 jumps (33%) | 17 / 128 (13%) |
| CptFalcon ditto | P2 2-0 | 14 / 87 (16%) | 14 / 88 (16%) |
| Luigi ditto | P1 1-0 | **75 / 90 (83%)** | **74 / 106 (70%)** |

Luigi mains rely on wavedash as their primary movement option in real play, so
70-83% conversion is validating — the model picked up the frequency naturally
from the training replays.

---

## Debug story (chronological)

### 1. The false lead: "model is stuck / bistable"

Yesterday's analysis of the `hal-7class-relpos` model found "bistable
inference" — sometimes the model played actively, sometimes it got stuck in
WAIT or SQUAT_WAIT attractors. We attributed this to an exposure-bias /
idle-state-training-distribution problem. That analysis was correct for the
symptoms but missed the root cause for a specific technique: wavedashes.

### 2. User observation: "the model can't wavedash"

The user noted that even when playing actively, no checkpoint had ever produced
a wavedash. Looking at action-state histograms across all saved replays:
`LANDING_FALL_SPECIAL` (action 43, the wavedash landing animation) appeared
0 times in thousands of frames of bot gameplay. Wavedashes are not rare in
Falco/Luigi training data (6,481 onsets in 5 Falco shards, ~2.6 per 1K frames).

### 3. Synthetic wavedash dataset (`tools/extract_wavedashes.py`)

To separate "model can't learn wavedashes" from "model doesn't get enough
wavedash gradient signal":

- Extracted 2,000 256-frame windows from Falco shards, each ending ~10 frames
  after a `LANDING_FALL_SPECIAL` onset.
- Filter: require a KNEE_BEND (action 24) with a JUMP button press within 30
  frames before the LANDING_FALL_SPECIAL, to ensure the full wavedash setup
  (jump squat → airdodge → land) is in-window. Rejects aerial double-jumps
  that coincidentally end in LANDING_FALL_SPECIAL from other mechanics.
- Saved as `data/falco_wavedash/train_shard_000.pt` + matching val shard,
  metadata copied from `data/falco_v2`.

Training command:
```
python3 train.py --model hal --encoder hal_flat --hal-mode \
  --hal-minimal-features --hal-controller-encoding --stick-clusters hal37 \
  --plain-ce --lr 3e-4 --batch-size 64 --grad-accum-steps 8 \
  --max-samples 8388608 --data-dir data/falco_wavedash \
  --reaction-delay 0 --self-inputs --run-name falco-wavedash-overfit \
  --no-warmup --cosine-min-lr 1e-6
```

Result: val loss 0.0075, btn F1 99.9%, main F1 99.7% at step **2,720**. The
architecture CAN memorize the wavedash input sequence perfectly — no
architectural limitation.

### 4. First inference test of the overfit model

Killed training after step ~2,720 (run was going to 16K). Ran the
`falco-wavedash-overfit_best.pt` checkpoint in a Falco ditto:

```
159 jump attempts (KNEE_BEND onsets)
0 wavedashes (LANDING_FALL_SPECIAL onsets)
```

Zero wavedashes from a model that perfectly memorized wavedashes. Something
between the model's predictions and the game's interpretation was broken.

### 5. Replay inspection

Using `py-slippi` to read `pre.triggers.physical.l` and `pre.buttons.physical`
frame-by-frame during the bot's jumps:

```
KNEE_BEND at frame 178 (3rd jump):
  f+2: KNEE_BEND   X        L=0.00  js=(+1.00, +0.00)
  f+3: KNEE_BEND   X        L=0.00  js=(+1.00, +0.00)
  f+4: KNEE_BEND   X        L=1.00  js=(+0.70, -0.70)   ← shoulder ramps up
  f+5: JUMP_F      NONE     L=1.00  js=(+0.70, -0.70)   ← first airborne, L=1, NO wavedash
  f+6: JUMP_F      NONE     L=1.00  js=(+0.85, -0.50)
```

The bot's controller state IS sending L=1.00 with a down-right diagonal stick
on the first airborne frame. These are correct wavedash inputs. The training
data has almost identical shoulder values at the same frame positions. But the
post-frame action state at f+5 is `JUMP_F`, not `LANDING_FALL_SPECIAL`. The
game did not register the airdodge.

### 6. Model sanity check on training data

Loaded the overfit checkpoint, fed it the exact same training window used for
the above analysis, and compared predictions to targets:

```
pos  action  tgt_btn  pred_btn  tgt_shldr  pred_shldr  tgt_main  pred_main
245   KNEE    TRIG     TRIG      full       full        cl22      cl22 KNEE
246   LAND_SP TRIG     TRIG      full       full        cl22      cl22 LAND_SP
```

Model predictions match targets frame-for-frame, including the `TRIG` button
class on frame 245 (the last KNEE_BEND frame). So the model IS predicting the
right button class. The inputs reaching Dolphin are L=1.00 analog. The game
still doesn't register a wavedash.

### 7. The real bug

Re-read `tools/inference_utils.py:decode_and_press` carefully. The 7-class
button decoder:

```python
if btn_idx == 0:
    ctrl.press_button(melee.enums.Button.BUTTON_A); pressed.append("A")
elif btn_idx == 1:
    ctrl.press_button(melee.enums.Button.BUTTON_B); pressed.append("B")
elif btn_idx == 2:
    ctrl.press_button(melee.enums.Button.BUTTON_Z); pressed.append("Z")
elif btn_idx == 3:
    ctrl.press_button(melee.enums.Button.BUTTON_X); pressed.append("JUMP")
elif btn_idx == 4:
    pressed.append("TRIG")                                       # ← BUG
elif btn_idx == 5:
    ctrl.press_button(melee.enums.Button.BUTTON_A); pressed.append("A+TRIG")  # ← BUG
```

TRIG never calls `ctrl.press_button(BUTTON_L)`. A_TRIG presses only A, not L.
The only shoulder input reaching Dolphin is the analog value from the separate
shoulder head, sent via `ctrl.press_shoulder(BUTTON_L, shldr)`.

**`press_shoulder` only sets the analog component** (confirmed by reading
melee-py source `/usr/local/lib/python3.12/dist-packages/melee/controller.py`).
The docstring explicitly says:

> The 'digital' button press of L or R are handled separately as normal button
> presses. Pressing the shoulder all the way in will not cause the digital
> button to press.

The digital L/R press is a distinct rising-edge event that Melee uses to
trigger airdodge, L-cancel, tech, powershield timing, etc. Shielding works on
analog alone (the game reads the analog value against a threshold every
frame), which is why the bug was never noticed in normal gameplay — bots
shield fine.

### 8. How the bug was introduced

The 2026-04-08 research notes claimed "Shoulder is analog-only. Neither HAL
nor MIMIC triggers the digital L/R click." This was based on observing HAL's
shield behavior and assuming "all shoulder techs use analog." It was a
misinterpretation — shielding uses analog, but airdodge/L-cancel/tech all
require the digital press.

During the 7-class refactor, someone (likely me in a prior session) added a
TRIG class for "shoulder button" and mapped it to "just append TRIG to a
label list" — relying on the (wrong) belief that the analog shoulder head
alone would handle everything.

This silently broke airdodge/wavedash/L-cancel/tech for every 7-class model
we've ever trained. It would never show up in offline metrics because the
training data doesn't know what Dolphin does with the inputs — it only knows
what the model predicted. The model predicted `TRIG` correctly at 99.9%
accuracy. The output just never hit the game.

### 9. The fix

```python
elif btn_idx == 4:
    ctrl.press_button(melee.enums.Button.BUTTON_L); pressed.append("TRIG")
elif btn_idx == 5:
    ctrl.press_button(melee.enums.Button.BUTTON_A)
    ctrl.press_button(melee.enums.Button.BUTTON_L); pressed.append("A+TRIG")
```

Verified on the wavedash-overfit checkpoint: **81 wavedashes for P1, 87 for P2
in a single 175s ditto**, with both sides using the same checkpoint (55%+
conversion rate per jump). The overfit model now executes wavedashes at the
rate it was trained on.

### 10. Audit for similar bugs

Grepped `tools/inference_utils.py` for every `pressed.append(...)`; each now
has a matching `ctrl.press_button(...)` call. Checked `tools/run_hal_model.py`
(HAL 5-class reimplementation) and confirmed it doesn't have a TRIG button
class at all — the 5-class button head is `[A, B, X, Z, NONE]` with no
shoulder slot. HAL therefore structurally cannot airdodge, which matches prior
observations that HAL-lineage bots don't wavedash.

`tools/run_mimic_via_hal_loop.py` and `tools/head_to_head.py` both import
`decode_and_press` from `inference_utils`, so they inherit the fix
automatically.

---

## Post-fix full-character retest

All ditto matches, Final Destination, temperature=1.0, using the best-val
checkpoint from each character's full training run:

**Falco ditto** (`falco-7class-v2-full_best.pt`, val btn F1 88%):
- P1 2-0 P2, 181s, `replays/Game_20260413T035445.slp`
- P1: 34 wavedashes / 104 jumps (33%)
- P2: 17 wavedashes / 128 jumps (13%)

**CptFalcon ditto** (`cptfalcon-7class-v2_best.pt`, val btn F1 89.9%):
- P2 2-0 P1, 179s, `replays/Game_20260413T040411.slp`
- P1: 14 wavedashes / 87 jumps (16%)
- P2: 14 wavedashes / 88 jumps (16%)

**Luigi ditto** (`luigi-7class-v2-long_best.pt`, val btn F1 ~92% early-stop):
- P1 1-0 P2, 203s, `replays/Game_20260413T041156.slp`
- P1: **75 wavedashes / 90 jumps (83%)**
- P2: **74 wavedashes / 106 jumps (70%)**

Luigi's conversion rate is spectacularly high because real Luigi mains rely on
wavedash as their primary movement option (wavedash out of shield, wavedash to
upsmash, waveland platforms). The model picked up this frequency from the
training replays. CptFalcon mains wavedash less (16% conversion), Falco
wavedashes the most of any character in real play but the training data may
have more mixed-playstyle replays (33% for P1, 13% for P2).

---

## Key takeaways

1. **Always verify inference end-to-end against game outcomes, not just model
   predictions.** A model can have 99.9% accuracy on training data and still
   produce 0 wavedashes if the inference-to-game pipeline drops inputs.
2. **"Silent" inference bugs are invisible to training metrics.** No loss
   function will catch a bug where the model predicts correctly but the
   controller layer ignores the prediction.
3. **`press_shoulder` and `press_button(BUTTON_L)` are different inputs.** Any
   future button head class that represents a shoulder technique MUST call the
   digital press. Document this in CLAUDE.md (done).
4. **HAL-lineage bots can't wavedash by design.** Their 5-class button head
   has no shoulder class. This explains years of "why can't BC bots wavedash"
   observations in the community — nothing to do with the model, everything to
   do with the button vocabulary. 7-class with a proper TRIG press is the
   right fix.
5. **Overfit-to-a-subset is the right sanity check for "can the architecture
   do X?"** If a model with 99.9% accuracy on a wavedash-only dataset produces
   0 wavedashes in Dolphin, the problem is not the model.

---

## Files touched

- `tools/inference_utils.py` — 2-line fix in `decode_and_press` (lines 422, 425).
- `tools/extract_wavedashes.py` — new diagnostic tool, kept in repo for future
  sanity tests on other techniques.
- `CLAUDE.md` — updated pitfall #11 to correctly describe the
  analog-vs-digital shoulder distinction.
- `docs/research-notes-2026-04-13.md` — this file.

## Next steps (not in this commit)

- Retest the regular (non-overfit) best checkpoints with the fix applied
  (done — see post-fix results above).
- Consider retraining Luigi with early-stopping at step ~5K to avoid the
  overfitting we saw on the long run; the `_best.pt` from the long run is
  already that early-stop checkpoint so this is moot.
- Discord bot for Slippi Direct Connect matches (separate plan file).
