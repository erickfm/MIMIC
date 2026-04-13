#!/usr/bin/env bash
set -euo pipefail

# MIMIC - Setup for a fresh machine with an NVIDIA GPU.
#
# Usage:
#   bash setup.sh                       # install deps, emulator, ISO, download data
#   bash setup.sh --run                 # also starts training after setup
#   bash setup.sh --rsync               # pull pre-built shards from Machine A
#   bash setup.sh --run --model small   # extra args forwarded to train.py

DATA_DIR="${DATA_DIR:-data/fox_hal_full}"
EMULATOR_DIR="emulator"
ISO_PATH="melee.iso"
RUN_AFTER=false
RSYNC_DATA=false
EXTRA_TRAIN_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run)       RUN_AFTER=true; shift ;;
        --rsync)     RSYNC_DATA=true; shift ;;
        --data-dir)  DATA_DIR="$2"; shift 2 ;;
        *)           EXTRA_TRAIN_ARGS+=("$1"); shift ;;
    esac
done

echo "=== MIMIC setup ==="

# ── 0. Git LFS ─────────────────────────────────────────────────────────────
echo ""
echo "── Ensuring Git LFS files are pulled ──"
if ! command -v git-lfs &>/dev/null && ! git lfs version &>/dev/null 2>&1; then
    echo "  Installing git-lfs ..."
    (curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
     apt-get install -y -qq git-lfs) 2>&1 | tail -1
fi
git lfs install --skip-smudge 2>/dev/null || true
git lfs pull 2>/dev/null || true
echo "  LFS files ready."

# ── 1. Python deps ──────────────────────────────────────────────────────────
echo ""
echo "── Installing Python dependencies ──"
PYDEPS="torch numpy pandas pyarrow wandb tensordict huggingface_hub melee==0.45.1 discord.py python-dotenv py-slippi"
pip install $PYDEPS --quiet 2>/dev/null \
  || pip install $PYDEPS --quiet --break-system-packages

# ── 2. Dolphin emulator ────────────────────────────────────────────────────
echo ""
echo "── Setting up Dolphin emulator ──"
if [[ -x "$EMULATOR_DIR/squashfs-root/usr/bin/dolphin-emu" ]]; then
    echo "  Dolphin already extracted."
else
    if [[ ! -f emulator.tar.gz ]]; then
        echo "  ERROR: emulator.tar.gz not found. Run: git lfs pull"
        exit 1
    fi
    echo "  Extracting emulator.tar.gz ..."
    mkdir -p "$EMULATOR_DIR"
    tar xzf emulator.tar.gz -C "$EMULATOR_DIR" --strip-components=0
    echo "  Dolphin ready at $EMULATOR_DIR/squashfs-root/usr/bin/dolphin-emu"
fi

# ── 3. Melee ISO ───────────────────────────────────────────────────────────
echo ""
echo "── Setting up Melee ISO ──"
if [[ -f "$ISO_PATH" ]]; then
    echo "  ISO found at $ISO_PATH"
else
    echo "  Downloading ISO ..."
    curl -L -o melee.zip "https://melee.today/download/melee.zip"
    unzip -o melee.zip -d _iso_tmp
    # Find the .iso inside the zip and move it to repo root
    ISO_FILE=$(find _iso_tmp -name "*.iso" -o -name "*.ISO" | head -1)
    if [[ -z "$ISO_FILE" ]]; then
        echo "  ERROR: No .iso found in melee.zip"
        rm -rf _iso_tmp melee.zip
        exit 1
    fi
    mv "$ISO_FILE" "$ISO_PATH"
    rm -rf _iso_tmp melee.zip
    echo "  ISO saved to $ISO_PATH"
fi

# ── 4. Data ─────────────────────────────────────────────────────────────────
echo ""
echo "── Setting up training data ──"
if [[ -d "$DATA_DIR" ]] && ls "$DATA_DIR"/train_shard_*.pt &>/dev/null; then
    SHARD_COUNT=$(ls "$DATA_DIR"/train_shard_*.pt | wc -l)
    echo "  Found $SHARD_COUNT shards in $DATA_DIR"
elif $RSYNC_DATA; then
    echo "  Pulling pre-built shards from Machine A ..."
    mkdir -p "$DATA_DIR"
    rsync -avz --progress -e "ssh -p 22877" \
        root@194.14.47.19:/root/MIMIC/data/fox_hal_full/ "$DATA_DIR/"
    echo "  Data synced to $DATA_DIR"
else
    echo "  No shards found in $DATA_DIR."
    echo "  Options:"
    echo "    1. bash setup.sh --rsync          # pull shards from Machine A"
    echo "    2. python tools/slp_to_shards.py  # build from raw .slp replays"
    echo ""
    echo "  Skipping data setup."
fi

# ── 5. GitHub CLI ──────────────────────────────────────────────────────────
echo ""
echo "── Installing GitHub CLI ──"
if command -v gh &>/dev/null; then
    echo "  gh already installed."
else
    echo "  Installing gh ..."
    (curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg 2>/dev/null && \
     echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
        | tee /etc/apt/sources.list.d/github-cli.list >/dev/null && \
     apt-get update -qq && apt-get install -y -qq gh) 2>&1 | tail -1
    echo "  gh installed."
fi

# ── 6. Claude Code ──────────────────────────────────────────────────────────
echo ""
echo "── Installing Claude Code ──"
if command -v claude &>/dev/null; then
    echo "  Claude Code already installed."
else
    curl -fsSL https://claude.ai/install.sh | sh
    echo "  Claude Code installed."
fi

# ── 7. Shell alias ─────────────────────────────────────────────────────────
echo ""
echo "── Setting up 'ai' alias ──"
ALIAS_LINE='alias ai="claude --dangerously-skip-permissions"'
if grep -qF "$ALIAS_LINE" ~/.bashrc 2>/dev/null; then
    echo "  Alias already in ~/.bashrc"
else
    echo "" >> ~/.bashrc
    echo "$ALIAS_LINE" >> ~/.bashrc
    echo "  Added alias: ai -> claude --dangerously-skip-permissions"
fi

# ── 8. GPU check ────────────────────────────────────────────────────────────
echo ""
echo "── GPU check ──"
python3 -c "
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f'  GPU {i}: {name} ({mem:.0f} GB)')
else:
    print('  WARNING: No CUDA GPU detected. Training will be very slow.')
"

# ── 9. Headless display (Xvfb) ──────────────────────────────────────────────
echo ""
echo "── Headless display (Xvfb) ──"
if command -v Xvfb &>/dev/null; then
    echo "  Xvfb already installed."
else
    echo "  Installing xvfb (needed for Dolphin on headless machines) ..."
    apt-get install -y -qq xvfb 2>&1 | tail -1
fi
if pgrep -x Xvfb >/dev/null; then
    echo "  Xvfb already running on display :99"
else
    Xvfb :99 -screen 0 1024x768x24 -ac >/dev/null 2>&1 &
    disown 2>/dev/null || true
    sleep 1
    echo "  Started Xvfb on :99"
fi
# Add DISPLAY=:99 to bashrc so new shells automatically see it
if grep -qF 'export DISPLAY=:99' ~/.bashrc 2>/dev/null; then
    echo "  DISPLAY=:99 already exported in ~/.bashrc"
else
    echo 'export DISPLAY=:99' >> ~/.bashrc
    echo "  Added 'export DISPLAY=:99' to ~/.bashrc"
fi
export DISPLAY=:99

# ── 10. Discord bot config (.env) ───────────────────────────────────────────
echo ""
echo "── Discord bot config (.env) ──"
if [[ -f .env ]]; then
    echo "  .env already exists — leaving it alone."
else
    if [[ -f .env.example ]]; then
        cp .env.example .env
        echo "  Created .env from .env.example."
        echo ""
        echo "  ⚠ You still need to fill in:"
        echo "      DISCORD_BOT_TOKEN  (from https://discord.com/developers/applications)"
        echo "      BOT_SLIPPI_CODE    (your bot's Slippi direct-connect code, e.g. MIMIC#01)"
        echo ""
        echo "  Then place the bot's Slippi user.json at:"
        echo "      ./slippi_home/Slippi/user.json"
        echo "  (create the account once via Slippi Launcher → log in → copy user.json here)"
    else
        echo "  .env.example missing — skipping."
    fi
fi

# Ensure slippi_home dir exists
mkdir -p slippi_home/Slippi
if [[ ! -f slippi_home/Slippi/user.json ]]; then
    echo "  ⚠ slippi_home/Slippi/user.json is missing — Slippi Online login will fail"
    echo "     until you place it there."
fi

echo ""
echo "=== Setup complete ==="
echo "  Dolphin:  $EMULATOR_DIR/squashfs-root/usr/bin/dolphin-emu"
echo "  ISO:      $ISO_PATH"
echo "  Data:     $DATA_DIR"
echo "  Display:  \$DISPLAY=$DISPLAY"
echo ""
echo "To run the Discord bot:"
echo "  1. Edit .env and set DISCORD_BOT_TOKEN + BOT_SLIPPI_CODE"
echo "  2. Place Slippi user.json at ~/.config/SlippiOnline/Slippi/user.json"
echo "  3. python3 tools/discord_bot.py"

# ── 9. Optionally start training ────────────────────────────────────────────
if $RUN_AFTER; then
    echo ""
    echo "── Starting training ──"
    python3 train.py --data-dir "$DATA_DIR" "${EXTRA_TRAIN_ARGS[@]}"
fi
