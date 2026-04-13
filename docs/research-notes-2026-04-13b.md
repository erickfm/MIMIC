# Research Notes — 2026-04-13b: Slippi Online Discord bot

Second session of 2026-04-13, building and deploying the Discord front-end
for MIMIC so humans can play the bot over Slippi Online Direct Connect.

## Goal

Let anyone play against the MIMIC checkpoints from anywhere by posting
`!play <character> <connect_code>` in a Discord channel. The bot spawns a
Dolphin instance, joins their direct-connect lobby, plays one match, uploads
the replay as a Discord attachment, and exits cleanly.

## Architecture

```
Discord user: !play falco ERIK#456
      │
      ▼
 discord_bot.py  (long-running, in-memory FIFO queue)
      │ validates, announces queue position
      │ (when it's this user's turn) spawns subprocess:
      ▼
 play_netplay.py
   • Console() launches Dolphin with the bot's Slippi user.json
   • MenuHelper.menu_helper_simple(connect_code=...) auto-navigates
     main menu → Slippi Online → Direct Connect → enters the user's code
   • port_detector (by connectCode field, handles dittos) identifies the
     bot's in-game port
   • Inference loop (shared with run_mimic_via_hal_loop.py, etc.)
   • Breaks on stock-out, kills Dolphin to end netplay session
   • Prints RESULT / SCORE / REPLAY lines on stdout
      │
      ▼
 discord_bot.py parses the lines, announces result with score + replay
```

## New files

- **`tools/play_netplay.py`** (~260 lines) — per-match Slippi Direct Connect
  runner. Based on `run_mimic_via_hal_loop.py` but with one bot controller
  (no CPU), `MenuHelper.menu_helper_simple(connect_code=..., autostart=True)`,
  and a proper result/score reporting tail.
- **`tools/discord_bot.py`** (~370 lines) — Discord front-end using `discord.py`.
  Single-match FIFO queue via `asyncio.Queue`, subprocess spawn per match,
  replay upload on completion.
- **`requirements-discord.txt`** — `discord.py >= 2.3`, `python-dotenv >= 1.0`.
- **`.env.example`** — config template with all options documented.
- **`docs/discord-bot-setup.md`** — full setup guide.

## Commits (chronological)

1. **`7f4b2a7` Discord bot for MIMIC vs human via Slippi Online Direct Connect** —
   initial working version. `play_netplay.py` + `discord_bot.py` +
   `requirements-discord.txt` + `docs/discord-bot-setup.md`.

2. **`b1b6e1d` Portable Discord bot setup + polish play_netplay** —
   made the repo scp-able to any Linux machine:
   - `setup.sh` installs discord.py / python-dotenv, installs + starts Xvfb
     on `:99`, copies `.env.example` → `.env`, creates `./slippi_home/Slippi/`
     skeleton, adds `export DISPLAY=:99` to `~/.bashrc`.
   - `discord_bot.py` resolves paths relative to repo root via
     `_resolve_path()` so `.env` can use `./emulator/...` and `./melee.iso`.
   - `play_netplay.py` tracks last-seen stocks during the match and breaks
     out within 1s of a stock-out (so the bot doesn't hold the opponent
     hostage in the rematch lobby).
   - Added `SCORE:` line to stdout output; Discord announcements now show
     the winner name and final stock/% line.

3. **`08aab46` Synthesize slippi_home/Slippi/user.json from .env vars** —
   so you can upload just `.env` (no `slippi_home/` bundle) and the bot
   regenerates `user.json` on startup. Supports three setup paths in
   priority order: existing file → `SLIPPI_USER_JSON` blob → individual
   `SLIPPI_UID`/`SLIPPI_PLAY_KEY`/`SLIPPI_CONNECT_CODE` fields.

## Bugs found and fixed during live testing

### 1. `DISPLAY` env var missing from subprocess (immediate)

First `!play` attempt: Dolphin launched, then died instantly as a zombie
process with `Unable to initialize GTK+, is DISPLAY set properly?`. The
Discord bot was launched in a shell that didn't have `DISPLAY=:99` exported,
so the subprocess inherited an empty display. Fixed by launching the bot
with `DISPLAY=:99 python3 tools/discord_bot.py` (and adding that export
to `~/.bashrc` via `setup.sh`).

### 2. `port_detector` costume mismatch (character + costume required)

Second `!play`: match ran for 120s with Falco standing still, then ended
with `result=no-opponent`. GPU utilization was 0% — the model was loaded
but the inference loop never executed. Cause: `melee.gamestate.port_detector`
requires character AND costume to match. We hardcoded `costume=0` but Slippi
Online assigned a different costume (maybe palette-swapped due to both sides
picking the same color?), so `port_detector` returned `0` every frame and
we kept resetting `match_started = False` in a busy loop.

Fix: three-tier fallback in `play_netplay.py`:
1. **Match by `connectCode`** — find the player whose `connectCode == MIMIC#01`.
   Unambiguous, handles dittos, palette swaps, anything. Reads the bot's
   own code from `user.json` at startup.
2. **Fall back to character + costume** (the old `port_detector` logic).
3. **Fall back to character-only** scan of `gs.players`.

Also: `PlayerState` in libmelee exposes `connectCode` and `displayName`
fields directly. Any new netplay code should use `connectCode` for
identification rather than `port_detector`.

### 3. Rematch lobby trap

Third `!play`: match played successfully, bot wavedashed, user 3-stocked
the bot. But the bot never disconnected — it was sitting in the post-game
rematch lobby. The user reported "I could keep playing it though?". The
issue: Slippi Online holds you in a "play again?" screen inside the
`IN_GAME` menu state, so my "break when menu_state != IN_GAME" check never
fired.

Fix: detect stock-out directly in the in-game loop. As soon as either side
reaches 0 stocks, sleep 1 second (for the death animation), then `break`.
The `finally` block's `console.stop()` kills Dolphin, ending the netplay
session on both sides.

### 4. `timeout` result code despite clean match end

Same third `!play`: the match ended with the user 3-stocking the bot, but
the Discord announcement came back as `timeout`. Cause: after breaking out
of the match loop, I was reading `gs.players[detected_port].stock` to
determine the result — but by that point `gs.players` had already reset
for the post-game menu, so both sides looked like they had 4 stocks again.

Fix: track `last_seen_me_stock` / `last_seen_opp_stock` inside the in-game
loop. After break, use those values for win/loss determination. Also added
a percent-lead fallback for genuine time-outs (equal stocks → lower percent
wins).

### 5. Stale stock/percent numbers in Discord announcements

While fixing #4, added:
- `SCORE:` machine-readable line in `play_netplay.py` stdout:
  `SCORE: bot=0stk/120% opp=2stk/45%`
- `discord_bot.py` parses the SCORE line and displays it formatted:
  `MIMIC: 0 stk, 120% • whatarewaves: 2 stk, 45%`
- Winner name in the announcement verb (`**MIMIC won**` /
  `**whatarewaves won**`) instead of generic "bot won / bot lost".

### 6. `log` not defined at startup

During the `user.json` synthesis fix, I used `log.info()` / `log.warning()`
inside `_ensure_user_json()` which runs before the logger is created. Python
raised `NameError: name 'log' is not defined`. Fixed by using `print(...,
file=sys.stderr)` in that function since it's pre-logger-init anyway.

## Verification results

Final test after all fixes (same user, Falco vs Falco ditto, `WAVE#666`):

- Match joined in ~10s after spawn
- Bot played actively: button press rate ~20%, stick movement ~54%,
  **20 wavedashes** in ~3 min of gameplay
- User 3-0'd the bot (stock-line: `MIMIC: 0 stk / 120% • whatarewaves: 3 stk / 45%`)
- Bot disconnected cleanly after stock-out
- Discord announcement arrived with winner name, final score, and the
  `.slp` replay as an attachment

## Portability

Three deployment modes, in order of upload weight:

| Mode | Upload | Effort |
|---|---|---|
| **Full**: clone repo + run setup.sh + upload `.env` + upload `slippi_home/` | Repo (or scp) + `.env` + `slippi_home/` | Highest |
| **Mid**: clone repo + run setup.sh + upload `.env` (synthesizes `user.json`) | Repo + `.env` | Typical |
| **Bootstrap**: `git clone && bash setup.sh && vi .env` (hand-enter token + Slippi fields) | Just creds | Lowest |

All three resolve to the same end state: `slippi_home/Slippi/user.json` on
disk with `0600` perms, `DOLPHIN_PATH` and `ISO_PATH` resolved from
`.env.example`'s relative defaults, bot launching at `MIMIC#3609` with
code `MIMIC#01`.

## Known gotchas for future sessions

1. **Every `.env` change requires restarting the bot**. `python-dotenv`
   loads once at startup, no hot-reload.
2. **Xvfb must be running on whatever display the bot process inherits**.
   If `setup.sh` wasn't run (or the Xvfb process died), Dolphin will crash
   with the GTK error. Restart Xvfb: `Xvfb :99 -screen 0 1024x768x24 -ac &`.
3. **The bot's `BOT_SLIPPI_CODE` is display-only** — it's shown in `!info`
   and match-start announcements so users know what to type on their side.
   The actual Slippi login uses `user.json`. Make sure both values match,
   otherwise `!info` will give users the wrong code.
4. **`_ensure_user_json()` runs before the Python logger is configured**.
   Don't use `log.*` inside it; use `print(..., file=sys.stderr)`.
5. **`MenuHelper.menu_helper_simple` with `connect_code` auto-navigates
   everything** — main menu, Slippi Online, Direct Connect, the nametag
   entry screen where each character is entered letter-by-letter, the CSS,
   stage selection, and autostart. You don't have to write any of that.
6. **`port_detector` requires costume match** — don't rely on it. Use
   `connectCode` instead (available on `PlayerState` directly).
7. **Bot's Discord display name (`MIMIC#3609`)** is separate from the Slippi
   connect code (`MIMIC#01`). The `#3609` is Discord's auto-assigned
   discriminator; the `#01` is the Slippi custom tag.
8. **Slippi direct connect uses each player's opponent code**. Both sides
   type in the OTHER player's code. Not a shared lobby code — this confused
   me mid-session.

## Files touched this session

- `tools/play_netplay.py` (new, ~260 lines)
- `tools/discord_bot.py` (new, ~380 lines)
- `requirements-discord.txt` (new)
- `.env.example` (new)
- `docs/discord-bot-setup.md` (new)
- `.gitignore` — added `user.json`, `**/SlippiOnline/**`, `slippi_home/`
- `setup.sh` — added Discord deps, Xvfb install + start, `.env` scaffolding
- `README.md` — rewrote with v2 shards, 7-class, Discord bot, portability
- `CLAUDE.md` — updated tool list, doc index, pitfalls #15-17
