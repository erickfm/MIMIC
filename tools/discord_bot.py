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
import signal
import subprocess
import sys
import time
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
def _ensure_user_json(slippi_home: str) -> bool:
    """Ensure <slippi_home>/Slippi/user.json exists.

    Priority:
      1. Existing file — leave it alone.
      2. SLIPPI_USER_JSON env var containing the JSON blob verbatim.
      3. Individual fields: SLIPPI_UID, SLIPPI_CONNECT_CODE, SLIPPI_PLAY_KEY,
         SLIPPI_DISPLAY_NAME, SLIPPI_LATEST_VERSION.

    Returns True if user.json is now present, False otherwise.
    """
    import json
    user_json_path = os.path.join(slippi_home, "Slippi", "user.json")
    if os.path.exists(user_json_path):
        return True

    data = None

    # Option 2: full JSON blob
    blob = os.environ.get("SLIPPI_USER_JSON", "").strip()
    if blob:
        try:
            data = json.loads(blob)
        except Exception as e:
            print(f"WARNING: SLIPPI_USER_JSON set but failed to parse: {e}",
                  file=sys.stderr)

    # Option 3: individual fields
    if data is None:
        uid = os.environ.get("SLIPPI_UID", "").strip()
        code = os.environ.get("SLIPPI_CONNECT_CODE", "").strip() or BOT_SLIPPI_CODE
        play_key = os.environ.get("SLIPPI_PLAY_KEY", "").strip()
        display = os.environ.get("SLIPPI_DISPLAY_NAME", "").strip() or "MIMIC"
        latest = os.environ.get("SLIPPI_LATEST_VERSION", "").strip() or "3.5.2"
        if uid and play_key:
            data = {
                "uid": uid,
                "connectCode": code,
                "playKey": play_key,
                "displayName": display,
                "latestVersion": latest,
            }

    if data is None:
        return False

    os.makedirs(os.path.dirname(user_json_path), exist_ok=True)
    with open(user_json_path, "w") as f:
        json.dump(data, f, indent=2)
    # Make it readable only to the owner — it contains the playKey
    try:
        os.chmod(user_json_path, 0o600)
    except Exception:
        pass
    print(f"[discord_bot] wrote {user_json_path} from env vars "
          f"(code={data.get('connectCode', '?')})", file=sys.stderr)
    return True


if not _ensure_user_json(SLIPPI_HOME):
    sys.stderr.write(
        f"ERROR: Slippi user.json not found at {SLIPPI_HOME}/Slippi/user.json\n"
        f"       Three options:\n"
        f"         A) Copy user.json to {SLIPPI_HOME}/Slippi/user.json\n"
        f"         B) Set SLIPPI_USER_JSON in .env to the full JSON blob\n"
        f"         C) Set SLIPPI_UID, SLIPPI_CONNECT_CODE, SLIPPI_PLAY_KEY,\n"
        f"            SLIPPI_DISPLAY_NAME in .env (bot will synthesize user.json)\n"
        f"       Create the Slippi account once via Slippi Launcher to get these\n"
        f"       values (uid + playKey live in the account's user.json).\n"
    )
    sys.exit(1)

import json as _json

HF_REPO_ID = os.environ.get("MIMIC_HF_REPO", "erickfm/MIMIC").strip()

# Aliases we hardcode for short-forms that can't be auto-derived from the
# character's HF dir name. Per-character lowercase aliases are added
# automatically by _build_aliases().
_HARDCODED_ALIASES = {
    "falcon":    "CPTFALCON",
    "cpt":       "CPTFALCON",
    "cf":        "CPTFALCON",
}

# Non-character top-level dirs on HF that we shouldn't treat as playable.
_HF_NON_CHAR_DIRS = {"checkpoints"}


def _scan_local_hf_checkpoints() -> tuple[dict, dict]:
    """Build CHARACTERS + per-char metadata from hf_checkpoints/*/model.pt
    on disk. Returns (characters, metadata)."""
    hf_root = _REPO_ROOT / "hf_checkpoints"
    chars: dict = {}
    meta: dict = {}
    if not hf_root.exists():
        return chars, meta
    for d in sorted(hf_root.iterdir()):
        if not d.is_dir() or d.name in _HF_NON_CHAR_DIRS:
            continue
        model_path = d / "model.pt"
        if not model_path.exists():
            continue
        key = d.name.upper()
        chars[key] = (
            str(model_path.relative_to(_REPO_ROOT)),
            str(d.relative_to(_REPO_ROOT)),
            key,
        )
        meta_path = d / "metadata.json"
        if meta_path.exists():
            try:
                meta[key] = _json.loads(meta_path.read_text())
            except Exception:
                meta[key] = {}
        else:
            meta[key] = {}
    return chars, meta


def _sync_hf_to_local(repo_id: str = None) -> tuple[dict, dict]:
    """Pull any character dirs from HF into hf_checkpoints/ (cached — only
    changed files redownload), then scan. Falls back to a local-only scan
    if HF is unreachable so the bot still starts with the last-known set."""
    repo_id = repo_id or HF_REPO_ID
    try:
        from huggingface_hub import HfApi, snapshot_download
        api = HfApi()
        files = api.list_repo_files(repo_id, repo_type="model")
        char_dirs = set()
        for f in files:
            parts = f.split("/")
            if len(parts) >= 2 and parts[1] == "model.pt":
                char_dirs.add(parts[0])
        char_dirs = {d for d in char_dirs if d not in _HF_NON_CHAR_DIRS}
        if char_dirs:
            patterns = [f"{d}/*" for d in sorted(char_dirs)]
            snapshot_download(
                repo_id,
                local_dir=str(_REPO_ROOT / "hf_checkpoints"),
                allow_patterns=patterns,
            )
        log.info("HF sync (%s): characters on HF = %s", repo_id, sorted(char_dirs))
    except Exception as e:
        log.warning("HF sync failed (%s) — falling back to local cache", e)
    return _scan_local_hf_checkpoints()


def _build_aliases(chars: dict) -> dict:
    aliases = dict(_HARDCODED_ALIASES)
    for key in chars:
        aliases[key.lower()] = key
    return aliases


# Character -> (best checkpoint, data dir, melee name).
# Populated from HF at startup via _load_character_catalog() and refreshable
# at runtime via the !reload command.
CHARACTERS: dict = {}
CHAR_META: dict = {}
CHAR_ALIASES: dict = {}

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
    connect_code: str    # opponent's code (e.g. WAVE#666)

# In-memory queue and current-match state
queue: asyncio.Queue = None  # created in on_ready
current_match: Optional[MatchRequest] = None
pending_list: list[MatchRequest] = []  # mirror of queue for display / cancel

# Live play_netplay.py subprocess for the in-flight session (if any).
# Used so cmd_play / cmd_cancel can write "STOP\n" to its stdin to end
# the back-to-back match chain cleanly after the current match.
current_proc: Optional[asyncio.subprocess.Process] = None

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
    char_lines = []
    for name in sorted(CHARACTERS.keys()):
        meta = CHAR_META.get(name) or {}
        # Prefer the run_name + step count from metadata; fall back to
        # the filename basename if metadata is missing.
        run = meta.get("run_name", "").strip()
        ckpt_label = run if run and run != "?" else os.path.basename(CHARACTERS[name][0])
        extras = []
        if meta.get("val_loss"):
            extras.append(f"val {meta['val_loss']}")
        if meta.get("global_step") and meta["global_step"] != "?":
            try:
                extras.append(f"{int(meta['global_step']) // 1000}k steps")
            except (TypeError, ValueError):
                pass
        extra_str = f" ({', '.join(extras)})" if extras else ""
        char_lines.append(f"  • **{name}** — `{ckpt_label}`{extra_str}")
    char_block = "\n".join(char_lines) if char_lines else "  (none loaded — HF sync failed?)"
    msg = (
        f"**MIMIC Melee bot**\n"
        f"Bot's Slippi connect code: `{BOT_SLIPPI_CODE}` — enter this on YOUR side.\n\n"
        f"**Characters + checkpoints** (from `{HF_REPO_ID}`):\n{char_block}\n\n"
        f"Usage: `!play <character> <your_connect_code>`\n"
        f"Example: `!play falco WAVE#666`\n\n"
        f"The bot joins your Slippi Direct Connect lobby. Launch Slippi Online → "
        f"Direct Connect, enter `{BOT_SLIPPI_CODE}`, and wait — the bot will join you.\n\n"
        f"`!queue` shows current queue. `!cancel` removes your pending match. "
        f"`!reload` re-syncs the character list from HuggingFace.\n\n"
        f"**Back-to-back matches:** the bot stays in your lobby after each "
        f"match. Pick your next character and press Start on your side "
        f"within 30s to queue another round — otherwise the bot disconnects "
        f"and the chain ends. `!cancel` stops it early; another user's "
        f"`!play` also ends your chain after the current match."
    )
    await ctx.reply(msg)


@bot.command(name="reload")
async def cmd_reload(ctx: commands.Context):
    """Re-sync CHARACTERS from HF without a bot restart."""
    global CHARACTERS, CHAR_META, CHAR_ALIASES
    await ctx.reply("⏳ Re-syncing character list from HuggingFace…")
    try:
        new_chars, new_meta = await asyncio.to_thread(_sync_hf_to_local)
    except Exception as e:
        log.exception("HF reload failed")
        await ctx.reply(f"❌ Reload failed: {e}")
        return
    added = sorted(set(new_chars) - set(CHARACTERS))
    removed = sorted(set(CHARACTERS) - set(new_chars))
    changed = sorted(
        k for k in (set(new_chars) & set(CHARACTERS))
        if new_chars[k] != CHARACTERS[k]
    )
    CHARACTERS = new_chars
    CHAR_META = new_meta
    CHAR_ALIASES = _build_aliases(new_chars)
    lines = [f"✅ Reloaded: {len(CHARACTERS)} characters"]
    if added:
        lines.append(f"  added: {', '.join(added)}")
    if removed:
        lines.append(f"  removed: {', '.join(removed)}")
    if changed:
        lines.append(f"  updated: {', '.join(changed)}")
    if not (added or removed or changed):
        lines.append("  no changes")
    await ctx.reply("\n".join(lines))


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


async def _send_stop_to_current_session() -> bool:
    """Write 'STOP\\n' to the running play_netplay subprocess's stdin so it
    ends the chain cleanly after the current match. No-op if no session is
    running or stdin is already closed. Returns True if STOP was sent."""
    p = current_proc
    if p is None or p.returncode is not None or p.stdin is None:
        return False
    try:
        p.stdin.write(b"STOP\n")
        await asyncio.wait_for(p.stdin.drain(), timeout=2.0)
        log.info("Sent STOP to running session")
        return True
    except (BrokenPipeError, ConnectionResetError):
        return False
    except asyncio.TimeoutError:
        log.warning("stdin.drain() timed out; subprocess may be hung")
        return False


@bot.command(name="cancel")
async def cmd_cancel(ctx: commands.Context):
    uid = ctx.author.id
    # Case 1: cancelling a pending (not-yet-running) queued match.
    removed = [r for r in pending_list if r.user_id == uid]
    if removed:
        for r in removed:
            pending_list.remove(r)
        await ctx.reply(f"✅ Cancelled your queued match ({removed[0].character} vs {removed[0].connect_code}).")
        return
    # Case 2: cancelling mid-chain — user is the current player. Ask the
    # running session to stop after the current match.
    if current_match is not None and current_match.user_id == uid:
        ok = await _send_stop_to_current_session()
        if ok:
            await ctx.reply("✅ Chain will end after the current match.")
        else:
            await ctx.reply("⚠️ Couldn't reach the running session — it may already be exiting.")
        return
    await ctx.reply("❌ You don't have a pending or running match.")


@bot.command(name="play")
async def cmd_play(ctx: commands.Context, character: str = None, connect_code: str = None):
    if character is None or connect_code is None:
        await ctx.reply(
            "Usage: `!play <character> <your_connect_code>`\n"
            f"Example: `!play falco WAVE#666`\n"
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
            f"(1-8 letters, then `#`, then digits). Example: `WAVE#666`."
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

    # If someone else is currently playing a chain, ask their session to
    # wrap up after the current match so this user gets served.
    if current_match is not None and current_match.user_id != ctx.author.id:
        await _send_stop_to_current_session()

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
    """Single background task: pops queue, runs one persistent-Dolphin
    session at a time. Each session plays back-to-back matches until the
    opponent idles, disconnects, another user queues (writes STOP), or
    the user !cancels."""
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
        ckpt_name = os.path.basename(CHARACTERS[req.character][0])
        log.info("Starting session: %s -> %s (%s) checkpoint=%s",
                 req.user_name, req.character, req.connect_code, ckpt_name)

        channel = bot.get_channel(req.channel_id)
        if channel is not None:
            try:
                await channel.send(
                    f"🎯 **Session starting**: {req.user_name} vs MIMIC ({req.character})\n"
                    f"Checkpoint: `{ckpt_name}`\n"
                    f"Enter `{BOT_SLIPPI_CODE}` in Slippi Online → Direct Connect now. "
                    f"After each match, pick your next character and press Start within 30s "
                    f"to keep going. `!cancel` ends the chain."
                )
            except Exception:
                log.exception("Failed to send session-starting message")

        try:
            await run_session(req, channel, ckpt_name)
        except Exception:
            log.exception("run_session raised")

        current_match = None


async def run_session(req: MatchRequest, channel, ckpt_name: str) -> None:
    """Spawn play_netplay.py in persistent-session mode and stream
    per-match announcements in real time. Returns when the subprocess
    emits SESSION_END and exits."""
    global current_proc

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
        "--max-matches", "-1",
        "--rematch-timeout", "30",
        "--check-stdin",
    ]
    log.info("Spawn: %s", " ".join(cmd))

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=MIMIC_REPO,
        )
    except Exception as e:
        log.exception("Failed to spawn play_netplay.py")
        if channel is not None:
            try:
                await channel.send(f"❌ Failed to start session: {e}")
            except Exception:
                pass
        return

    current_proc = proc
    pending: dict = {}          # accumulates one match's RESULT/SCORE/REPLAY
    session_end_reason: Optional[str] = None

    async def _flush_pending():
        """Announce the accumulated match result, if any."""
        nonlocal pending
        if not pending.get("result"):
            pending = {}
            return
        try:
            await announce_result(
                channel, req,
                pending["result"],
                pending.get("replay", ""),
                "",
                pending.get("score", ""),
                ckpt_name,
            )
        except Exception:
            log.exception("Failed to announce match result")
        log.info("Match %s finished: result=%s score=%s replay=%s",
                 pending.get("idx", "?"),
                 pending["result"],
                 pending.get("score", ""),
                 pending.get("replay", ""))
        pending = {}

    try:
        # Hard upper bound on a single session: 30 matches × match-timeout.
        # Well beyond what anyone will play in one sitting; guards against a
        # stuck subprocess that never emits SESSION_END.
        session_deadline = MATCH_TIMEOUT_SEC * 30
        try:
            async with asyncio.timeout(session_deadline):
                while True:
                    raw = await proc.stdout.readline()
                    if not raw:
                        break
                    text = raw.decode(errors="replace").strip()
                    if not text:
                        continue
                    if text.startswith("MATCH_START:"):
                        await _flush_pending()
                        idx = text.split(":", 1)[1].strip()
                        pending = {"idx": idx}
                        if channel is not None:
                            try:
                                await channel.send(
                                    f"▶️ **Match {idx} starting** — "
                                    f"{req.user_name} vs MIMIC ({req.character})"
                                )
                            except Exception:
                                log.exception("Failed to send match-starting message")
                    elif text.startswith("RESULT:"):
                        pending["result"] = text.split(":", 1)[1].strip()
                    elif text.startswith("SCORE:"):
                        pending["score"] = text.split(":", 1)[1].strip()
                    elif text.startswith("REPLAY:"):
                        pending["replay"] = text.split(":", 1)[1].strip()
                    elif text.startswith("SESSION_END:"):
                        session_end_reason = text.split(":", 1)[1].strip()
                        await _flush_pending()
                        break
                    # Any other line on stdout is ignored.
        except asyncio.TimeoutError:
            log.warning("Session exceeded hard wall-clock limit; killing")
            try:
                proc.kill()
            except Exception:
                pass
            session_end_reason = "hard-timeout"
            await _flush_pending()
    finally:
        current_proc = None

    # Drain remaining stderr + wait for exit.
    try:
        await asyncio.wait_for(proc.wait(), timeout=30.0)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        await proc.wait()

    stderr_tail = ""
    try:
        stderr_bytes = await proc.stderr.read()
        stderr_tail = stderr_bytes.decode(errors="replace")[-500:]
    except Exception:
        pass

    # Session-end announcements.
    if channel is not None:
        try:
            if session_end_reason == "opponent-timeout":
                await channel.send(
                    "⌛ Opponent idled on the rematch screen — chain ended. "
                    "`!play` again to come back."
                )
            elif session_end_reason == "stopped":
                # Another user queued and triggered a graceful handoff.
                # Worker will pick up the next request immediately.
                pass
            elif session_end_reason in ("opponent-gone", "max-matches", "signal"):
                # Already covered by the final match's RESULT announcement.
                pass
            elif session_end_reason == "hard-timeout":
                await channel.send("⏱️ Session watchdog fired — subprocess killed.")
            elif session_end_reason == "error" or (proc.returncode not in (0, None)):
                msg = f"❌ Session crashed (exit={proc.returncode}, reason={session_end_reason})"
                if stderr_tail:
                    msg += f"\n```\n{stderr_tail[-400:]}\n```"
                await channel.send(msg)
        except Exception:
            log.exception("Failed to send session-end message")

    if proc.returncode not in (0, None):
        log.warning("play_netplay.py exit=%d, stderr tail:\n%s",
                    proc.returncode, stderr_tail)


async def announce_result(channel, req: MatchRequest, result: str, replay_path: str,
                           err_tail: str, score: str = "", ckpt_name: str = ""):
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

    msg_lines = [f"{emoji} {verb} — MIMIC as **{req.character}** vs {req.user_name}"]
    msg_lines.append(f"Opponent code: `{req.connect_code}`")
    if ckpt_name:
        msg_lines.append(f"Checkpoint: `{ckpt_name}`")
    if score_line:
        msg_lines.append(score_line.lstrip("\n"))
    msg = "\n".join(msg_lines)

    # Don't publish a replay for results where no actual match took place —
    # "no-opponent" and "failed" would otherwise upload a stale .slp left
    # over from a previous run in the replays/ dir.
    skip_replay = result in ("no-opponent", "failed")

    files = []
    if not skip_replay and replay_path and os.path.exists(replay_path):
        size = os.path.getsize(replay_path)
        if size < 25 * 1024 * 1024:  # Discord free-tier attachment limit
            files.append(discord.File(replay_path, filename=os.path.basename(replay_path)))
        else:
            msg += f"\n(replay file is {size / 1e6:.1f} MB — too large to upload, saved at `{replay_path}`)"

    if result == "failed" and err_tail:
        # Include a tiny error snippet for debugging
        msg += f"\n```\n{err_tail[-400:]}\n```"

    await channel.send(msg, files=files)


def _cleanup_orphan_processes() -> None:
    """Kill leftover play_netplay.py + dolphin-emu processes from a prior
    bot run. These get reparented to init when the parent bot exits and
    otherwise burn CPU/GPU forever (we've seen a 10+ hour Dolphin eating
    85% CPU with no replay activity). Called once at startup BEFORE the
    match worker starts consuming the queue — at this point no legitimate
    children exist yet, so anything matching our spawn pattern is junk."""
    my_pid = os.getpid()

    def _pgrep(pattern: str) -> list[int]:
        try:
            out = subprocess.run(
                ["pgrep", "-f", pattern],
                capture_output=True, text=True, timeout=5,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return []
        return [int(p) for p in out.stdout.split() if p.strip() and int(p) != my_pid]

    def _kill(pids: list[int], sig: int) -> None:
        for pid in pids:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
            except PermissionError:
                log.warning("Can't signal pid %d (permission denied)", pid)

    play_pids = _pgrep("tools/play_netplay.py")
    if play_pids:
        log.warning("Found orphan play_netplay.py processes: %s — terminating", play_pids)
        _kill(play_pids, signal.SIGTERM)
        time.sleep(3.0)
        survivors = [p for p in play_pids if _pgrep_alive(p)]
        if survivors:
            log.warning("play_netplay.py survivors after SIGTERM: %s — killing", survivors)
            _kill(survivors, signal.SIGKILL)

    # Sweep any Dolphin instances from our emulator path. play_netplay's
    # shutdown_handler should cascade into Dolphin via console.stop(), but
    # we've seen cases where Dolphin keeps running after the parent dies.
    dolphin_pat = str(_REPO_ROOT / "emulator/squashfs-root/usr/bin/dolphin-emu")
    dolphin_pids = _pgrep(dolphin_pat)
    if dolphin_pids:
        log.warning("Found orphan dolphin-emu processes: %s — killing", dolphin_pids)
        _kill(dolphin_pids, signal.SIGKILL)


def _pgrep_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _load_character_catalog() -> None:
    """Sync HF → local cache and populate the module-level CHARACTERS /
    CHAR_META / CHAR_ALIASES dicts. Called once at startup."""
    global CHARACTERS, CHAR_META, CHAR_ALIASES
    CHARACTERS, CHAR_META = _sync_hf_to_local()
    CHAR_ALIASES = _build_aliases(CHARACTERS)
    if CHARACTERS:
        log.info("Loaded %d characters: %s",
                 len(CHARACTERS), ", ".join(sorted(CHARACTERS.keys())))
    else:
        log.warning("No characters loaded! !play will reject everything. "
                    "Check HF connectivity and hf_checkpoints/ contents.")


if __name__ == "__main__":
    _cleanup_orphan_processes()
    _load_character_catalog()
    bot.run(DISCORD_BOT_TOKEN)
