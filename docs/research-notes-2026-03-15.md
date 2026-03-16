# Research Notes — 2026-03-15

## Context

Continued from [2026-03-14](research-notes-2026-03-14.md). Previous session established that focal loss fixes the class imbalance problem (btn_f1 98%, stick top-1 96% offline). Today's goal: achieve correct closed-loop wavedash behavior in the live emulator.

---

## Phase 1: Canonical Cluster Training

### Problem

Previous training used dataset-specific cluster centers (from the small wavedash dataset). These don't generalize — a model trained on wavedash data can't be evaluated against full-game data without mismatched cluster indices.

### Changes

- **`train.py`**: Added `--clusters-path` argument. Cluster centers are now loaded from a canonical source (`data/full/stick_clusters.json`) by default, regardless of which dataset is being trained on.
- **`dataset.py`**: Updated `_load_cluster_centers()` resolution order: explicit path → data_dir → canonical default.
- **`generate_wavedash_replay.py`**: Added `EDGE_THRESHOLD` and `_safe_direction()` to prevent Falco from wavedashing off-stage. The bot reverses direction when within 15 units of the edge.

### Wavedash v2 Dataset

Regenerated an 8-minute (28,800 frame) wavedash replay with edge-safety, exported to `data/wavedash_v2/`. Preprocessed with canonical 63-cluster stick centers and 4-bin shoulder centers.

### Training Results

Trained `wavedash-canonical` with label_smoothing=0.0 for sharp predictions:
- **Val F1: 98.5%**, total loss: 0.0001 (near-epsilon)
- Converged in ~80 epochs

---

## Phase 2: Closed-Loop Debug Tool

### `closedloop_debug.py`

Built a frame-by-frame tensor comparison tool following [Eric Gu's HAL methodology](https://github.com/ericyuegu/hal): the single most important debugging step is to verify that training and inference data distributions perfectly match.

**Phase 1 (offline)**: Loads a training parquet and processes it through both the training pipeline (`dataset.py`) and the inference pipeline (`inference.py`), then compares every tensor element. Initial run found 57 categorical mismatches caused by `cat_maps.json` using string keys vs. integer keys — fixed by converting keys in `closedloop_debug.py`'s loader. After fix: **perfect alignment, 0 mismatches**.

**Phase 2 (online)**: Added `--diag-log-all` flag to `inference.py` that pickles every raw row dict during live play for post-hoc comparison.

---

## Phase 3: Live Inference — The Pipe Synchronization Bug

### Symptom

Model predictions were correct (high-confidence Y, L, diagonal stick at the right times) but the character wouldn't respond. BUTTON_Y was sent for hundreds of frames before a jump registered. Airdodge never triggered despite correct L + diagonal stick predictions.

### Debugging Timeline

1. **Shoulder conflict hypothesis**: Maybe `press_shoulder()` interferes with `press_button(BUTTON_L)` for airdodge. Fixed to not call analog shoulder when digital L is pressed. → Still broken.
2. **Button debounce hypothesis**: Maybe Melee needs a 0→1 transition to register a new press, and continuous pressing is treated as "held". Implemented alternating press/release. → Still broken.
3. **Controller readback mismatch**: `game_stick=(0.500,0.500)` even when diagonal stick was being sent. Suggested inputs weren't reaching the game.

### Root Cause: `blocking_input=False`

Studied [HAL's eval code](https://github.com/ericyuegu/hal/blob/main/hal/eval/eval.py) and libmelee's `Controller` class source code. Found the critical difference:

**HAL uses `blocking_input=True`** when creating the Console. This sets `BlockingPipes=True` in Dolphin's config, making Dolphin **wait for a pipe FLUSH command before advancing each frame**.

Without `blocking_input=True` (our default), Dolphin runs freely at 60fps. Our inference takes ~65ms per frame (~15fps). By the time our pipe commands arrive, Dolphin has already advanced 3-4 frames. Most inputs are simply never processed — they arrive between frames and are discarded.

This explains every symptom:
- Y pressed for 327 frames before jump: most PRESS commands missed, only occasionally caught
- Airdodge never triggered: L + diagonal stick needed to land on the exact airborne frame, but timing was random
- `game_stick=(0.5, 0.5)` despite sending diagonal: the SET MAIN command arrived after the frame was already processed

### Fix

```python
console = melee.Console(
    path=DOLPHIN_APP,
    slippi_address="127.0.0.1",
    fullscreen=False,
    blocking_input=True,  # Dolphin waits for our flush each frame
)
```

Additional changes adopted from HAL's `send_controller_inputs()`:
- **Explicit press/release for every button every frame** (no `release_all()`) — matches HAL's pattern
- **Always call `press_shoulder()` for L and R** every frame, even when value is 0
- **`flush()` inside `press_output()`** and after menu handling (blocking mode hangs if no flush is sent)
- **`flush()` on `gs.frame < 0` skip** to prevent deadlock

### Result

**Falco wavedashes perfectly.** The model's predictions translate correctly to in-game actions. The wavedash sequence (stand → Y → jumpsquat → L+diagonal → airdodge → slide → repeat) executes flawlessly.

Initial framerate: ~15fps (Dolphin runs in lockstep with our inference speed).

---

## Phase 4: Inference Performance — 15fps → 60fps

### Profiling

Added per-frame timing instrumentation to `run_inference()`:

| Component | Time (ms) | % of Total |
|-----------|-----------|------------|
| `_process_one_row` (pandas) | **57** | **92%** |
| `_cached_state_seq` (torch.cat) | 0.5 | 1% |
| GPU transfer | 0.3 | <1% |
| Model forward | 2.2 | 4% |
| Post-process | 0.1 | <1% |
| **Total** | **~60** | |

The model itself is fast (2.2ms). The entire bottleneck was pandas: creating a `pd.DataFrame` from a single dict, running `preprocess_df()`, `apply_normalization()`, and `df_to_state_tensors()` — all for **one row**. Pandas has massive per-call overhead for metadata management, type checking, and memory allocation that dwarfs the actual computation.

Deeper profiling of the pandas path:

| Pandas Step | Time (ms) |
|------------|-----------|
| `pd.DataFrame([row])` | 2-3 |
| `preprocess_df()` | **25** |
| `apply_normalization()` | **20** |
| `df_to_state_tensors()` | **9** |

### Fix: Pure-Python Preprocessing

Replaced the pandas path with a pure-Python/dict `_process_one_row()` that does the same operations without any DataFrame overhead:

- C-stick direction encoding → scalar `math.hypot` + comparisons
- Categorical mapping → pre-computed dict lookups
- Z-score normalization → `(float(v) - mean) / std`
- Tensor construction → `torch.tensor()` directly from values

Combined with the existing tensor cache (`deque[Dict[str, torch.Tensor]]`) that only processes the new row and `torch.cat`s the sliding window:

| Component | Before (ms) | After (ms) | Speedup |
|-----------|------------|------------|---------|
| Row preprocessing | 57 | **0.3** | **190x** |
| Stack cached tensors | 0.5 | 0.6 | — |
| GPU transfer | 0.3 | 0.3 | — |
| Model forward | 2.2 | 2.2 | — |
| **Total** | **~60** | **~3.5** | **17x** |

3.5ms per frame is well under the 16.7ms budget for 60fps. The game now runs at full speed.

### `torch.compile`

Added `torch.compile(model)` (default mode) for the inference model. The model was already only 2.2ms so the gain is marginal here, but it's free and matches what HAL does. Training already had this.

### Why Training Doesn't Have This Problem

Training uses `MeleeFrameDatasetWithDelay` or `StreamingMeleeDataset`, both of which process entire parquet files (thousands of rows) in bulk at init time or per-file. Pandas overhead is amortized. Only inference processes one row at a time.

---

## Key libmelee Findings

### Pipe Protocol

libmelee communicates with Dolphin via named pipes. Every `press_button()`, `release_button()`, `tilt_analog()`, `press_shoulder()` call immediately writes a command string to the pipe (e.g., `"PRESS Y\n"`, `"SET MAIN 0.85 0.23\n"`). These are buffered by Python's stdio until `flush()` is called, which writes `"FLUSH\n"` and calls `pipe.flush()` to send everything to Dolphin at once.

### `blocking_input` / `BlockingPipes`

When `True`, Dolphin blocks at each frame boundary waiting for a `FLUSH` command on the pipe before advancing. This guarantees every input is processed but caps the game's framerate to the bot's inference speed.

### `fix_analog_inputs`

When `True` (default), `tilt_analog()` and `press_shoulder()` transform input values through `fix_analog_stick()` / `fix_analog_trigger()` to compensate for Melee's internal quantization. This ensures the value you send equals the value you observe in `controller_state`. Our training data was generated with `fix_analog_inputs=False` and cluster centers represent observed values, so using the default `True` at inference correctly maps predicted values to matching observations.

### `press_shoulder` vs `press_button` for L/R

From the libmelee docstring: "The 'digital' button press of L or R are handled separately as normal button presses. Pressing the shoulder all the way in will not cause the digital button to press." Airdodge requires the digital press (`press_button(BUTTON_L)`), not the analog value (`press_shoulder(BUTTON_L, 1.0)`).

---

## HAL Reference Implementation

Key patterns from [ericyuegu/hal](https://github.com/ericyuegu/hal) `emulator_helper.py`:

```python
def send_controller_inputs(controller, inputs):
    controller.tilt_analog(BUTTON_MAIN, inputs["main_stick"][0], inputs["main_stick"][1])
    controller.tilt_analog(BUTTON_C, inputs["c_stick"][0], inputs["c_stick"][1])
    controller.press_shoulder(BUTTON_L, shoulder_value)
    for button_str in ORIGINAL_BUTTONS:
        button = getattr(melee.Button, button_str.upper())
        if button_str in buttons_to_press:
            controller.press_button(button)
        else:
            controller.release_button(button)
    controller.flush()
```

- No `release_all()` — explicit press/release per button per frame
- Always sends shoulder analog value
- `ORIGINAL_BUTTONS` = A, B, X, Y, Z, L, R (7 buttons, no START/D-pad)
- Console created with `blocking_input=True`, `tmp_home_directory=True`
- Uses `torch.compile(model, mode="default")` for GPU inference
- Context window managed as a rolling buffer in shared memory

---

## Files Changed

| File | Change |
|------|--------|
| `inference.py` | `blocking_input=True`; HAL-style press/release; `torch.compile`; pure-Python `_process_one_row()` with tensor caching (0.3ms vs 57ms); flush on menu/skip frames; timing instrumentation |
| `train.py` | `--clusters-path` for canonical cluster loading |
| `dataset.py` | Updated `_load_cluster_centers()` resolution order |
| `generate_wavedash_replay.py` | Edge-safety logic (`EDGE_THRESHOLD`, `_safe_direction()`) |
| `closedloop_debug.py` | New tool for frame-by-frame tensor comparison (HAL methodology) |

---

## Status

**Wavedash works at 60fps in closed loop.** The bot correctly predicts and executes the full wavedash sequence (jump → airdodge → slide → repeat) with the trained model driving Dolphin in real time.

Next steps: train on full Melee dataset and test generalization beyond wavedash.
