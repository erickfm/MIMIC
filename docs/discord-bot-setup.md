# Discord bot setup — MIMIC over Slippi Online Direct Connect

A portable Discord front-end for the MIMIC bot. Users run `!play <character>
<their_connect_code>` in a Discord channel; the bot spawns a Dolphin instance,
joins their Slippi Online direct-connect lobby, and plays a match. Replays are
uploaded back to the channel as attachments.

## What you need

- A machine with:
  - **NVIDIA GPU** (the bot's inference runs on CUDA; ~3 GB VRAM per active
    match). MIMIC was developed on an RTX 5090; any modern card with bf16
    support should work.
  - **Slippi Dolphin** installed (download from https://slippi.gg/netplay).
  - **Melee 1.02 NTSC ISO** that you legally own. Place the path in `ISO_PATH`.
- A **Slippi account** for the bot. Create one via the Slippi Launcher
  normally — log in with an email and copy the connect code (`TAG#NUMBER`)
  from the Launcher's profile page. The bot's Dolphin will use this
  account's `user.json` automatically (libmelee copies your Slippi home dir
  into a tmp user directory when `copy_home_directory=True`).
- A **Discord bot application** and token. Create one at
  https://discord.com/developers/applications → New Application →
  Bot → Copy Token. Invite the bot to your server with at least these
  permissions: Send Messages, Attach Files, Read Message History,
  Read Messages/View Channels. **Enable the "Message Content Intent"** under
  Bot → Privileged Gateway Intents — the bot uses prefix commands which
  require reading message content.
- A **checkpoint per character** you want to serve. The easiest way is to
  grab `mimic_best_checkpoints.zip` (built in the last session — not in git)
  and unzip into `checkpoints/`.
- **Data dir per character**. Each needs a `tensor_manifest.json`,
  `hal_norm.json`, `controller_combos.json`, `cat_maps.json`,
  `stick_clusters.json`, `norm_stats.json`. The .pt shard files are NOT
  needed for inference — only these metadata JSONs. If you trained on this
  machine the dirs are already there at `data/fox_hal_v2`, `data/falco_v2`,
  `data/cptfalcon_v2`, `data/luigi_v2`.

## Install

```bash
# from the MIMIC repo root
pip install -r requirements.txt           # existing MIMIC deps (torch, melee, etc.)
pip install -r requirements-discord.txt   # discord.py, python-dotenv
```

## Configure

Create a `.env` file in the repo root (gitignored via `.env` pattern — if
not, add it manually):

```env
DISCORD_BOT_TOKEN=your_discord_bot_token_here
BOT_SLIPPI_CODE=MIMIC#001
DOLPHIN_PATH=/path/to/dolphin-emu
ISO_PATH=/path/to/Melee.iso
# Optional:
MIMIC_REPO=/path/to/MIMIC
MATCH_TIMEOUT_SEC=900
```

- `BOT_SLIPPI_CODE` is displayed in `!info` so users know what to enter on
  their side. It does NOT have to match the actual account the bot's Dolphin
  logs into — libmelee figures that out from `user.json`.
- `DOLPHIN_PATH` must be the `dolphin-emu` executable (not the Slippi
  Launcher wrapper). On Linux this is typically inside the extracted
  `squashfs-root/usr/bin/dolphin-emu`.

## Verify `play_netplay.py` standalone first

Before running the Discord bot, sanity-check the per-match script against your
own Slippi Dolphin on a second machine (or a second Slippi account on the
same machine). Pick a connect code (your personal one, e.g. `ERIK#456`):

```bash
python3 tools/play_netplay.py \
  --checkpoint checkpoints/falco-7class-v2-full_best.pt \
  --dolphin-path /path/to/dolphin-emu \
  --iso-path /path/to/melee.iso \
  --data-dir data/falco_v2 \
  --character FALCO \
  --connect-code ERIK#456
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
- `!play falco ERIK#456` — queue a Falco match against your lobby
- `!queue` — show queue state
- `!cancel` — remove your queued match

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

**Bot's Dolphin launches but stays at the main menu**
  → Menu navigation in libmelee relies on specific `SubMenu` IDs. If Slippi
    updates add a new main menu layout, `MenuHelper.menu_helper_simple` may
    need to be updated in libmelee itself.

**Match starts but bot controls the opponent's character (inverted
perspective)**
  → `port_detector` is returning the wrong port, or the bot's character
    selection didn't stick. Confirm the `--character` arg matches what
    MenuHelper selected (it should, since both use the same `BOT_CHAR`).

**"RESULT: failed"**
  → Check stderr tail in the Discord message or the bot's log. Common causes:
    missing checkpoint file, invalid data dir (missing JSON metadata),
    Dolphin binary not found.
