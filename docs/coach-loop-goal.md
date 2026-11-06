# Loop goal (owner away 2026-11-05 to ~11-08)

Work on branch `coach`. Commit finished, verified work there; never push, never commit `sim/` or the
ISO. Report each iteration in a few plain lines (Pacific time, 12-hour): what got done, what was
verified, what's next, anything blocked. Measure before claiming; say when a result is noise.

## 1. Training queue (runs by itself)

`tools/allchar_queue.py` trains Falco M, then S for 21 characters: BC, then RL (rl-v2) and a
192-game h2h vs the BC parent. State: `runs/allchar_queue.json` (`python3 tools/allchar_queue.py
--status`), log: `runs/allchar_queue.log`, per-run logs: `runs/<name>/logs/`.

- Check it each iteration. If a job failed, read its log, fix the cause (code, not recipes), and
  restart the queue (`nohup python3 tools/allchar_queue.py > /dev/null 2>&1 &`; it skips finished
  jobs and retries failed ones only if you reset their entry). Don't change recipes or step budgets.
- If the queue process died, restart it the same way. Never run a second copy.
- Record each finished character's h2h result (RL vs its BC parent) in the report.

## 2. "Try it yourself" (build now, on CPU)

From any coach suggestion (a master try, a tested move, or the paused moment), the player plays that
moment themselves against the bot, live, from the exact game state.

- Server: the native sim at 60 fps from the exact frame (rewind ~1 s before the moment), the
  player's pad from the browser, the bot an **S RL model on CPU** (measured: S 6.2 ms per frame on
  4 threads, M 16.2 ms, too slow). Until a character's S RL model exists, use its best available
  policy and say which.
- Browser: controller via the Gamepad API, and the GameCube adapter via WebUSB if feasible; keyboard
  fallback. Live frames into the same 3D viewer (patch it if needed; MIT). Retry instantly; save
  the attempt as .slp; score the attempt the same way as the tries (stretch + V) and show it next
  to the master's.
- **No GPU** for this while the queue runs: the bot, V and all tests on CPU.
- Test end to end without a person: a scripted pad that replays a master try's inputs must
  reproduce that try. Measure frame timing (dropped/late frames) and input-to-screen delay.

## 3. After the queue finishes (GPU free)

- Switch the coach to RL policies (Falco: the new M RL; Fox: `fox-M-bc2-mlr01-s1.rl2-s1`; others:
  their S RL), retrain the value models on those policies (`tools/coach/value.py`), and re-validate
  them (`tools/coach/value_check.py`) before the app uses them. Train value models for the other
  characters too.
- Upload the value models to Hugging Face so a fresh machine gets them, and make the coach download
  them (as it does policies). Confirm before pushing anything to HF.

## Rules

- Never use the GPU while another GPU job runs (the queue counts). Check `nvidia-smi` first.
- Premium and simple: plain words, nothing a player wouldn't understand.
- Stop when 1–3 are done, or when blocked on something only the owner can provide.
