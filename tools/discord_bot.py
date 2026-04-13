#!/usr/bin/env python3
"""MIMIC Discord bot for Slippi Online Direct Connect matches.

Users request a match by posting `!play <character> <connect_code>` in any
channel the bot can see. The bot queues the request, and when it's the user's
turn spawns `tools/play_netplay.py` to join the user's direct-connect lobby.
When the match ends the bot uploads the saved replay as a Discord attachment.

Commands (prefix `!`):
    !play <character> <code>   Queue a match. Character ∈ {FOX,FALCO,CPTFALCON,LUIGI}.
    !queue                     Show current queue and who's playing.
    !cancel                    Remove your queued match (only if not running).
    !info                      Show bot's own connect code, supported characters, usage.

Required env vars (see .env or environment):
    DISCORD_BOT_TOKEN   Discord bot token
    BOT_SLIPPI_CODE     Bot's own Slippi code (e.g. MIMIC#001), shown in !info
    DOLPHIN_PATH        Path to dolphin-emu executable
    ISO_PATH            Path to Melee 1.02 NTSC ISO
Optional env vars:
    MIMIC_REPO          Path to the MIMIC repo (default: parent dir of this script)
    MATCH_TIMEOUT_SEC   Hard timeout per match in seconds (default 900)
"""

import os
import re
import sys
import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import discord
    from discord.ext import commands
except ImportError:
    sys.stderr.write(
        "\nERROR: discord.py not installed. Run:\n"
        "    pip install -r requirements-discord.txt\n\n"
    )
    sys.exit(1)

# Resolve repo root as the parent dir of tools/
_REPO_ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv
    # Load .env from the repo root explicitly so it works from any cwd
    load_dotenv(_REPO_ROOT / ".env")
except ImportError:
    pass  # python-dotenv optional; env vars can be set directly


def _resolve_path(env_value: str, default: str = "") -> str:
    """Resolve a path from .env — handles relative paths against repo root."""
    val = env_value or default
    if not val:
        return ""
    p = Path(val).expanduser()
    if not p.is_absolute():
        p = _REPO_ROOT / p
    return str(p.resolve())


# ---- Config ----

DISCORD_BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN", "").strip()
BOT_SLIPPI_CODE = os.environ.get("BOT_SLIPPI_CODE", "MIMIC#000").strip()
DOLPHIN_PATH = _resolve_path(
    os.environ.get("DOLPHIN_PATH", ""),
    "emulator/squashfs-root/usr/bin/dolphin-emu",
)
ISO_PATH = _resolve_path(
    os.environ.get("ISO_PATH", ""),
    "melee.iso",
)
# Slippi home dir containing Slippi/user.json. Required for Slippi Online login.
# Defaults to ./slippi_home in the repo root (gitignored).
SLIPPI_HOME = _resolve_path(
    os.environ.get("SLIPPI_HOME", ""),
    "slippi_home",
)
MIMIC_REPO = os.environ.get("MIMIC_REPO", str(_REPO_ROOT))
MATCH_TIMEOUT_SEC = float(os.environ.get("MATCH_TIMEOUT_SEC", "900"))

if not DISCORD_BOT_TOKEN:
    sys.stderr.write(
        "ERROR: DISCORD_BOT_TOKEN not set.\n"
        "       Copy .env.example to .env and fill in the token.\n"
    )
    sys.exit(1)
if not os.path.exists(DOLPHIN_PATH):
    sys.stderr.write(
        f"ERROR: Dolphin not found at {DOLPHIN_PATH}\n"
        f"       Run `bash setup.sh` or set DOLPHIN_PATH in .env.\n"
    )
    sys.exit(1)
if not os.path.exists(ISO_PATH):
    sys.stderr.write(
        f"ERROR: Melee ISO not found at {ISO_PATH}\n"
        f"       Run `bash setup.sh` or set ISO_PATH in .env.\n"
    )
    sys.exit(1)
if not os.path.exists(os.path.join(SLIPPI_HOME, "Slippi", "user.json")):
    sys.stderr.write(
        f"ERROR: Slippi user.json not found at {SLIPPI_HOME}/Slippi/user.json\n"
        f"       Create a Slippi account via Slippi Launcher, then copy the\n"
        f"       user.json to {SLIPPI_HOME}/Slippi/user.json\n"
        f"       (or set SLIPPI_HOME in .env to a different path).\n"
    )
    sys.exit(1)

# Character -> (best checkpoint, data dir, melee name)
CHARACTERS = {
    "FOX":       ("checkpoints/hal-7class-v2-long_best.pt",   "data/fox_hal_v2",   "FOX"),
    "FALCO":     ("checkpoints/falco-7class-v2-full_best.pt", "data/falco_v2",     "FALCO"),
    "CPTFALCON": ("checkpoints/cptfalcon-7class-v2_best.pt",  "data/cptfalcon_v2", "CPTFALCON"),
    "LUIGI":     ("checkpoints/luigi-7class-v2-long_best.pt", "data/luigi_v2",     "LUIGI"),
}

CHAR_ALIASES = {
    "fox": "FOX",
    "falco": "FALCO",
    "cptfalcon": "CPTFALCON",
    "falcon": "CPTFALCON",
    "cpt": "CPTFALCON",
    "cf": "CPTFALCON",
    "luigi": "LUIGI",
}

CONNECT_CODE_RE = re.compile(r"^[A-Z]{1,8}#\d+$")

# ---- Logging ----

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
)
log = logging.getLogger("discord_bot")

# ---- Queue model ----

@dataclass
class MatchRequest:
    user_id: int
    user_name: str
    channel_id: int
    character: str       # canonical (FOX/FALCO/...)
    connect_code: str    # opponent's code (e.g. ERIK#456)

# In-memory queue and current-match state
queue: asyncio.Queue = None  # created in on_ready
current_match: Optional[MatchRequest] = None
pending_list: list[MatchRequest] = []  # mirror of queue for display / cancel

# ---- Discord client ----

intents = discord.Intents.default()
intents.message_content = True  # needed for prefix commands
bot = commands.Bot(command_prefix="!", intents=intents, help_command=None)


def parse_character(token: str) -> Optional[str]:
    key = token.strip().lower()
    if key in CHAR_ALIASES:
        return CHAR_ALIASES[key]
    upper = token.strip().upper()
    if upper in CHARACTERS:
        return upper
    return None


def normalize_code(code: str) -> Optional[str]:
    code = code.strip().upper()
    if CONNECT_CODE_RE.match(code):
        return code
    return None


@bot.event
async def on_ready():
    global queue
    if queue is None:
        queue = asyncio.Queue()
        bot.loop.create_task(match_worker())
    log.info("Discord bot logged in as %s (id=%s)", bot.user, bot.user.id)
    log.info("Serving bot code: %s", BOT_SLIPPI_CODE)


@bot.command(name="info")
async def cmd_info(ctx: commands.Context):
    chars = ", ".join(sorted(CHARACTERS.keys()))
    msg = (
        f"**MIMIC Melee bot**\n"
        f"Bot's Slippi connect code: `{BOT_SLIPPI_CODE}` — enter this on YOUR side.\n"
        f"Characters: {chars}\n\n"
        f"Usage: `!play <character> <your_connect_code>`\n"
        f"Example: `!play falco ERIK#456`\n\n"
        f"The bot joins your Slippi Direct Connect lobby. Launch Slippi Online → "
        f"Direct Connect, enter `{BOT_SLIPPI_CODE}`, and wait — the bot will join you.\n\n"
        f"`!queue` shows current queue. `!cancel` removes your pending match."
    )
    await ctx.reply(msg)


@bot.command(name="queue")
async def cmd_queue(ctx: commands.Context):
    lines = []
    if current_match is not None:
        lines.append(
            f"▶️ **Playing now**: {current_match.user_name} — "
            f"{current_match.character} vs {current_match.connect_code}"
        )
    else:
        lines.append("▶️ **Playing now**: (idle)")
    if pending_list:
        lines.append(f"**Queue** ({len(pending_list)}):")
        for i, req in enumerate(pending_list, start=1):
            lines.append(f"  {i}. {req.user_name} — {req.character} vs {req.connect_code}")
    else:
        lines.append("**Queue**: empty")
    await ctx.reply("\n".join(lines))


@bot.command(name="cancel")
async def cmd_cancel(ctx: commands.Context):
    uid = ctx.author.id
    removed = [r for r in pending_list if r.user_id == uid]
    if not removed:
        await ctx.reply("❌ You don't have a pending match in the queue.")
        return
    for r in removed:
        pending_list.remove(r)
    await ctx.reply(f"✅ Cancelled your queued match ({removed[0].character} vs {removed[0].connect_code}).")


@bot.command(name="play")
async def cmd_play(ctx: commands.Context, character: str = None, connect_code: str = None):
    if character is None or connect_code is None:
        await ctx.reply(
            "Usage: `!play <character> <your_connect_code>`\n"
            f"Example: `!play falco ERIK#456`\n"
            f"Characters: {', '.join(sorted(CHARACTERS.keys()))}"
        )
        return

    char = parse_character(character)
    if char is None:
        await ctx.reply(
            f"❌ Unknown character `{character}`. "
            f"Valid: {', '.join(sorted(CHARACTERS.keys()))} (aliases: falcon, cf, cpt)."
        )
        return

    code = normalize_code(connect_code)
    if code is None:
        await ctx.reply(
            f"❌ Invalid connect code `{connect_code}`. Format is `TAG#NUMBER` "
            f"(1-8 letters, then `#`, then digits). Example: `ERIK#456`."
        )
        return

    # Prevent the same user from queueing twice
    if any(r.user_id == ctx.author.id for r in pending_list):
        await ctx.reply("❌ You already have a match in the queue. Use `!cancel` first.")
        return
    if current_match is not None and current_match.user_id == ctx.author.id:
        await ctx.reply("❌ Your match is already running.")
        return

    req = MatchRequest(
        user_id=ctx.author.id,
        user_name=ctx.author.display_name,
        channel_id=ctx.channel.id,
        character=char,
        connect_code=code,
    )
    pending_list.append(req)
    await queue.put(req)

    pos = len(pending_list)
    if current_match is None and pos == 1:
        eta_msg = "starting soon"
    else:
        eta_msg = f"position **{pos}** in queue"
    await ctx.reply(
        f"✅ Queued: {req.character} vs `{req.connect_code}` — {eta_msg}.\n"
        f"Enter `{BOT_SLIPPI_CODE}` in your Slippi Online → Direct Connect now."
    )


async def match_worker():
    """Single background task: pops queue, runs matches one at a time."""
    global current_match
    while True:
        req: MatchRequest = await queue.get()
        # Remove from pending_list mirror (in case user cancelled)
        if req in pending_list:
            pending_list.remove(req)
        else:
            log.info("Skipping cancelled request from %s", req.user_name)
            continue

        current_match = req
        log.info("Starting match: %s -> %s (%s)", req.user_name, req.character, req.connect_code)

        channel = bot.get_channel(req.channel_id)
        if channel is not None:
            try:
                await channel.send(
                    f"▶️ **Match starting**: {req.user_name} vs MIMIC ({req.character})\n"
                    f"Enter `{BOT_SLIPPI_CODE}` in Slippi Online → Direct Connect now."
                )
            except Exception:
                log.exception("Failed to send match-starting message")

        result, replay_path, err_tail, score = await run_match(req)
        log.info("Match finished: result=%s score=%s replay=%s", result, score, replay_path)

        if channel is not None:
            try:
                await announce_result(channel, req, result, replay_path, err_tail, score)
            except Exception:
                log.exception("Failed to send result message")

        current_match = None


async def run_match(req: MatchRequest) -> tuple[str, str, str]:
    """Spawn play_netplay.py for one match. Returns (result, replay_path, err_tail)."""
    ckpt, data_dir, melee_char = CHARACTERS[req.character]
    ckpt_abs = os.path.join(MIMIC_REPO, ckpt)
    data_dir_abs = os.path.join(MIMIC_REPO, data_dir)
    script = os.path.join(MIMIC_REPO, "tools", "play_netplay.py")

    cmd = [
        sys.executable, script,
        "--checkpoint", ckpt_abs,
        "--dolphin-path", DOLPHIN_PATH,
        "--iso-path", ISO_PATH,
        "--data-dir", data_dir_abs,
        "--character", melee_char,
        "--connect-code", req.connect_code,
        "--match-timeout", str(MATCH_TIMEOUT_SEC),
        "--slippi-home", SLIPPI_HOME,
    ]
    log.info("Spawn: %s", " ".join(cmd))

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=MIMIC_REPO,
        )
    except Exception as e:
        log.exception("Failed to spawn play_netplay.py")
        return ("failed", "", str(e))

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(),
            timeout=MATCH_TIMEOUT_SEC + 120,  # extra buffer over the script's own timeout
        )
    except asyncio.TimeoutError:
        log.warning("play_netplay.py exceeded hard wall-clock limit; killing")
        try:
            proc.kill()
        except Exception:
            pass
        await proc.communicate()
        return ("timeout", "", "hard wall-clock timeout")

    stdout = stdout_bytes.decode(errors="replace")
    stderr = stderr_bytes.decode(errors="replace")

    result = "failed"
    replay_path = ""
    score = ""
    for line in stdout.splitlines():
        if line.startswith("RESULT:"):
            result = line.split(":", 1)[1].strip()
        elif line.startswith("REPLAY:"):
            replay_path = line.split(":", 1)[1].strip()
        elif line.startswith("SCORE:"):
            score = line.split(":", 1)[1].strip()

    err_tail = stderr[-500:] if stderr else ""
    if proc.returncode != 0 and result == "failed":
        log.warning("play_netplay.py exit=%d, stderr tail:\n%s", proc.returncode, err_tail)
    return (result, replay_path, err_tail, score)


async def announce_result(channel, req: MatchRequest, result: str, replay_path: str,
                           err_tail: str, score: str = ""):
    emoji_map = {
        "win":         "🏆",
        "loss":        "💀",
        "draw":        "🤝",
        "disconnect":  "🔌",
        "no-opponent": "⌛",
        "timeout":     "⏱️",
        "failed":      "❌",
    }
    winner_map = {
        "win":         "**MIMIC won**",
        "loss":        f"**{req.user_name} won**",
        "draw":        "**Draw**",
        "disconnect":  "Opponent disconnected",
        "no-opponent": "Opponent never joined",
        "timeout":     "Match timed out (inference hung)",
        "failed":      "Match failed to run",
    }
    emoji = emoji_map.get(result, "❓")
    verb = winner_map.get(result, f"Result: {result}")

    # score looks like "bot=2stk/45% opp=0stk/120%" — rewrite using user's name
    score_line = ""
    if score:
        score_fmt = (
            score
            .replace("bot=", f"MIMIC: ")
            .replace("opp=", f"{req.user_name}: ")
            .replace("stk/", " stk, ")
            .replace("%", "%")
        )
        score_line = f"\nFinal: {score_fmt}"

    msg = (
        f"{emoji} {verb} — {req.character} ditto\n"
        f"Opponent code: `{req.connect_code}`"
        f"{score_line}"
    )

    files = []
    if replay_path and os.path.exists(replay_path):
        size = os.path.getsize(replay_path)
        if size < 25 * 1024 * 1024:  # Discord free-tier attachment limit
            files.append(discord.File(replay_path, filename=os.path.basename(replay_path)))
        else:
            msg += f"\n(replay file is {size / 1e6:.1f} MB — too large to upload, saved at `{replay_path}`)"

    if result == "failed" and err_tail:
        # Include a tiny error snippet for debugging
        msg += f"\n```\n{err_tail[-400:]}\n```"

    await channel.send(msg, files=files)


if __name__ == "__main__":
    bot.run(DISCORD_BOT_TOKEN)
