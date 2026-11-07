# Loop goal (owner away 2026-11-05 to ~11-08)

**Duration: keep working until at least Sat 2026-11-07, 6:00 PM Pacific**, and after that until the
backlog below is done or blocked. Finishing parts 1–3 early is not a reason to stop: move on to the
backlog. While only waiting on the GPU, check back every 30–60 minutes; otherwise keep building.

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

### Results (RL vs its BC parent, 192 games, sim, bf16, seed 41)

| run | RL vs BC | stock margin | finished |
|---|---|---|---|
| falco-M-bc3-hf28k-s1.rl2-s1 | 181–11 (94.3%) | +1.91 | Thu 11/5 11:24 PM |
| falco-S-bc2-hf28k-s1.rl2-s1 | 157–35 (81.8%) | +1.32 | Fri 11/6 2:22 AM |
| marth-S-bc2-hf21k-s1.rl2-s1 | 169–23 (88.0%) | +1.48 | Fri 11/6 4:59 AM |
| sheik-S-bc2-hf17k-s1.rl2-s1 | 155–37 (80.7%) | +1.25 | Fri 11/6 7:28 AM |
| puff-S-bc2-hf14k-s1.rl2-s1 | 164–28 (85.4%) | +1.34 | Fri 11/6 9:51 AM |
| cptfalcon-S-bc2-hf12k-s1.rl2-s1 | 160–32 (83.3%) | +1.48 | Fri 11/6 12:08 PM |
| peach-S-bc2-hf11k-s1.rl2-s1 | 129–63 (67.2%) | +0.62 | Fri 11/6 2:27 PM |
| samus-S-bc2-hf9k-s1.rl2-s1 | 135–57 (70.3%) | +0.87 | Fri 11/6 4:40 PM |
| luigi-S-bc2-hf4k-s1.rl2-s1 | 146–46 (76.0%) | +0.92 | Fri 11/6 6:39 PM |
| pikachu-S-bc2-hf4k-s1.rl2-s1 | 137–55 (71.4%) | +1.03 | Fri 11/6 8:37 PM |
| dk-S-bc2-hf4k-s1.rl2-s1 | 153–39 (79.7%) | +1.22 | Fri 11/6 10:36 PM |
| doc-S-bc2-hf3k-s1.rl2-s1 | 160–32 (83.3%) | +1.38 | Sat 11/7 12:37 AM |
| yoshi-S-bc2-hf3k-s1.rl2-s1 | 159–33 (82.8%) | +1.23 | Sat 11/7 2:38 AM |
| ganondorf-S-bc2-hf3k-s1.rl2-s1 | 133–59 (69.3%) | +0.85 | Sat 11/7 4:39 AM |
| link-S-bc2-hf3k-s1.rl2-s1 | 132–60 (68.8%) | +0.80 | Sat 11/7 6:39 AM |
| bowser-S-bc2-hf3k-s1.rl2-s1 | 108–83, 1 draw (56.5%; p 0.08) | +0.20 | Sat 11/7 8:39 AM |
| gameandwatch-S-bc2-hf3k-s1.rl2-s1 | 142–49, 1 draw (74.3%) | +0.94 | Sat 11/7 10:36 AM |
| mario-S-bc2-hf3k-s1.rl2-s1 | 133–59 (69.3%) | +0.81 | Sat 11/7 12:34 PM |

The BC parent is also the opponent RL trained against, so this measures the gain against that one
opponent.

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

- Switch the coach to S RL policies for every character (owner, 2026-11-06: S with RL throughout,
  the same models live play uses; Fox: SF7u100 `fox-S-bc1-s1.rl1-s0`), retrain the value models on those policies (`tools/coach/value.py`), and re-validate
  them (`tools/coach/value_check.py`) before the app uses them. Train value models for the other
  characters too.
- Upload the value models to Hugging Face so a fresh machine gets them, and make the coach download
  them (as it does policies). Confirm before pushing anything to HF.

### Ready to run when the GPU is free (prepared 2026-11-05)

0. Falco S RL vs the July Falco (owner, 2026-11-06: "run the falco h2h after the queue"):
   `tools/run.py register falco-M-master-jul --policy hf_eval/falco --char falco --stage bc --note
   "falco-20260719-master-c9-s99 (HF falco, July)"`, then `tools/run.py h2h falco-S-bc2-hf28k-s1.rl2-s1
   falco-M-master-jul` (192 games, seed 41, bf16). Record the result in the table above.
1. `python3 tools/coach/after_queue.py` (RL policies + value models, item 3 above).
2. Stuckey's Falco model (backlog 4): shards are built (`data/stuckey/falco_train`: 365 games + 40
   for picking the checkpoint; `data/stuckey/falco_test`: 46 games no choice is made on). Then
   `python3 tools/coach/player_model.py finetune --parent falco-S-bc2-hf28k-s1 --data data/stuckey/falco_train
   --name stuckey-falco-S-ft1 --steps 2000` and `player_model.py eval --data data/stuckey/falco_test
   --policies <S BC> <S RL> runs/stuckey-falco-S-ft1/policy` (S, like everything else in the coach). Both paths were smoke-tested on the CPU.
3. `python3 tools/coach/upload_coach.py` lists what would go to `erickfm/MIMIC/coach/` (each character's
   policy + value model, only pairs that belong together); **ask the owner**, then `--push`. The coach
   fetches the value models at start and a policy the first time it needs it (tested: a value model is
   used only with the policy it was trained on, matched by content; the download itself is untested
   until something is uploaded).

## 4. Backlog, in order (after 1–3, or alongside them when they're waiting)

1. Test the whole app as a first-time user on Stuckey's Fox replays (STUC#911) as well as the
   owner's Falco games; fix what breaks. Screenshots, not assumptions.
2. Lab: set a fighter's state too (standing, crouching, shielding, in the air, hanging on the
   ledge), and draw its tries like the game view (one try at a time, blue you / red opponent).
3. Opponent realism check (GPU free only): at real decisions, force the player's real inputs with the
   model opponent and see where the real outcome lands among the tries (should look like one more
   try). Report the numbers; change nothing in the app on its basis without saying why.
4. A model of one specific opponent (GPU free only): fine-tune on Stuckey's replays, measure how
   much better it predicts his real inputs than the generic master model (held-out games). Offer
   it in the coach only if it's clearly better.
5. Polish: response times, loading states, error messages, empty states; read every screen once
   more as a player would.
6. Feasibility only, no big builds: can the sim be built for Windows (melee-unlocked has win64
   support)? Report what it would take.

## Rules

- Never use the GPU while another GPU job runs (the queue counts). Check `nvidia-smi` first.
- Heavy CPU work slows the queue too, even at nice 19: a 12-process value extraction (after_queue
  `--device cpu`, 2026-11-06 1:20 AM) took S RL rollouts from 30 s to 47–56 s per update. Keep CPU jobs
  light (a few threads) while the queue runs.
- Premium and simple: plain words, nothing a player wouldn't understand.
- Don't stop before Sat 2026-11-07, 6:00 PM Pacific. After that, stop when 1–4 are done, or when
  blocked on something only the owner can provide.
