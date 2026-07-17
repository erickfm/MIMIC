# Discord bot setup — MIMIC over Slippi Online Direct Connect

A portable Discord front-end for the MIMIC bot. Users run `!play <character>
<their_connect_code>` (or a `!<character>` shortcut) in a Discord channel;
the bot runs one persistent Dolphin session, joins their Slippi Online
direct-connect lobby, and plays N back-to-back matches. Replays are uploaded
back to the channel as attachments.

## What you need

- A machine with:
  - **NVIDIA GPU** (the bot's inference runs on CUDA; ~3 GB VRAM per active
    match). MIMIC was developed on an RTX 5090; any modern card with bf16
    support should work.
  - **Slippi Dolphin** — `setup.sh` downloads the current release into
    `emulator/` automatically (Slippi's version gate rejects stale builds,
    so don't pin an old one).
  - **Melee 1.02 NTSC ISO** that you legally own. Place the path in `ISO_PATH`.
- A **Slippi account** for the bot. Create one via the Slippi Launcher
  normally — log in with an email and copy the connect code (`TAG#NUMBER`)
  from the Launcher's profile page. The bot's Dolphin will use this
  account's `user.json` automatically.

  **Place `user.json` at `./slippi_home/Slippi/user.json` in the MIMIC repo.**
  This path is gitignored, so you can upload the whole `slippi_home/` dir
  alongside the repo without leaking credentials. To find the file:

  - Linux: `~/.config/SlippiOnline/Slippi/user.json`
  - macOS: `~/Library/Application Support/com.project-slippi.dolphin/netplay/User/Slippi/user.json`
  - Windows: `%APPDATA%\Slippi Launcher\netplay\User\Slippi\user.json`

  Copy it into the repo with:
  ```bash
  mkdir -p slippi_home/Slippi
  cp ~/.config/SlippiOnline/Slippi/user.json slippi_home/Slippi/
  ```
- A **Discord bot application** and token. Create one at
  https://discord.com/developers/applications → New Application →
  Bot → Copy Token. Invite the bot to your server with permissions
  integer `51200` (Send Messages + Embed Links + Attach Files):
  `https://discord.com/api/oauth2/authorize?client_id=<APP_ID>&permissions=51200&scope=bot`.
  **Enable the "Message Content Intent"** under
  Bot → Privileged Gateway Intents — the bot uses prefix commands which
  require reading message content.
- **Checkpoints.** `bash setup.sh --models` pulls
  [erickfm/MIMIC](https://huggingface.co/erickfm/MIMIC) into
  `hf_checkpoints/`. Each character dir there is self-contained
  (`model.pt` + all metadata JSONs) — the bot autodiscovers every dir with
  a `model.pt` + `metadata.json` and registers a `!<dir>` command for it;
  no wiring or renaming needed. Training shard `.pt` files are NOT needed
  for inference.

## Install

The easiest path is to run the repo's `setup.sh` — it installs all Python
deps (including `discord.py` and `python-dotenv`), downloads the current
Slippi Dolphin, downloads the Melee ISO, installs Xvfb for headless display,
and copies `.env.example` to `.env` for you:

```bash
bash setup.sh
```

If you prefer to install the Discord-specific deps manually:

```bash
pip install -r requirements-discord.txt
```

## Configure

After `setup.sh` runs, there's already a `.env` file in the repo root (copied
from `.env.example`). Edit it and fill in:

```env
DISCORD_BOT_TOKEN=your_discord_bot_token_here
BOT_SLIPPI_CODE=MIMIC#01
```

The defaults for `DOLPHIN_PATH`, `ISO_PATH`, and `SLIPPI_HOME` are **relative
paths** (`./emulator/...`, `./melee.iso`, `./slippi_home`) that the bot
resolves against the repo root at runtime. This makes the `.env` file
portable — you can `scp` the entire repo to a different machine and it'll
work without re-tuning paths, as long as `setup.sh` has been run there.

- `DISCORD_BOT_TOKEN`: create a Discord app at
  https://discord.com/developers/applications → Bot → Reset Token. Also
  enable **Message Content Intent** under Privileged Gateway Intents on the
  same page.
- `BOT_SLIPPI_CODE`: the connect code shown in `!info` so users know what
  to type on their side. Must match the `connectCode` field inside
  `slippi_home/Slippi/user.json`.

## Verify `play_netplay.py` standalone first

Before running the Discord bot, sanity-check the per-match script against your
own Slippi Dolphin on a second machine (or a second Slippi account on the
same machine). Pick a connect code (your personal one, e.g. `WAVE#666`):

```bash
python3 tools/play_netplay.py \
  --checkpoint hf_checkpoints/falco/model.pt \
  --dolphin-path ./emulator/squashfs-root/usr/bin/dolphin-emu \
  --iso-path ./melee.iso \
  --data-dir hf_checkpoints/falco \
  --character FALCO \
  --connect-code WAVE#666
```

On your Slippi machine, open Slippi Online → Direct Connect and enter the
bot's connect code. The bot should auto-navigate its own Dolphin to Direct
Connect, enter your code, join the lobby, and start a match. A replay is
saved to `replays/` and the script prints `RESULT: <win|loss|...>` and
`REPLAY: <path>` at the end.

## Run the Discord bot

```bash
python3 tools/discord_bot.py
```

The bot logs to stderr. In your Discord server:

- `!info` — show the bot's connect code and character list
- `!play falco WAVE#666` — queue a Falco match against your lobby
- `!fox-master WAVE#666` — per-character shortcut (one exists for every
  loaded character; the list is discovered from HuggingFace, not hardcoded)
- `!queue` — show queue state
- `!cancel` — remove your queued match
- `!reload` — re-sync characters from HuggingFace without a restart

When your turn comes, the bot posts `▶️ Match starting` and starts its
Dolphin. Enter the bot's connect code (shown in `!info`) on your side
within 2 minutes or the bot will give up with `no-opponent`.

## Operational notes

- **One match at a time.** The queue is strictly sequential. Parallel matches
  are out of scope for this MVP — a single Dolphin + model per GPU.
- **In-memory state.** The queue is lost on bot restart. No persistent
  storage. If the bot crashes, users have to re-queue.
- **Match timeout.** Hard-capped at `MATCH_TIMEOUT_SEC` (default 15 min). The
  per-match script also has its own internal timeout. If a match goes over,
  the subprocess is killed and `⏱️ timeout` is announced.
- **Replay uploads.** Discord's free-tier attachment limit is 25 MB. Typical
  MIMIC replays are 3–6 MB. If a replay is too large, the bot announces the
  result without the file and logs the local path.
- **TOS reminder.** libmelee's README explicitly warns against playing bots
  on Slippi Unranked/Ranked. This bot only joins Direct Connect lobbies that
  a human explicitly opted into by entering the bot's code. This is fine.
  Do NOT try to adapt it for matchmaking.

## Troubleshooting

**"Could not find user.json / Slippi won't let me connect"**
  → Launch Slippi Launcher at least once and log in with the bot's account.
    libmelee copies your home directory to the tmp Dolphin user via
    `copy_home_directory=True`. If that doesn't pick it up, set
    `dolphin_home_path` explicitly in `play_netplay.py`.

**"required update available" when going online**
  → The Dolphin build is older than Slippi's server-side minimum version.
    Delete `emulator/` and re-run `setup.sh` (it always fetches the current
    release). Do not switch to mainline Dolphin to dodge this — it crashes
    under libmelee's EXI protocol or is itself gate-rejected
    (CLAUDE.md pitfalls #19/#20).

**Bot's Dolphin launches but stays at the main menu**
  → First suspect a stale Dolphin build (see the item above — an outdated
    build can also stall menu navigation). If the build is current: menu
    navigation in libmelee relies on specific `SubMenu` IDs, and a new
    Slippi menu layout may need a libmelee update.

**Match starts but bot controls the opponent's character (inverted
perspective)**
  → `port_detector` is returning the wrong port, or the bot's character
    selection didn't stick. Confirm the `--character` arg matches what
    MenuHelper selected (it should, since both use the same `BOT_CHAR`).

**"RESULT: failed"**
  → Check stderr tail in the Discord message or the bot's log. Common causes:
    missing checkpoint file, invalid data dir (missing JSON metadata),
    Dolphin binary not found.
