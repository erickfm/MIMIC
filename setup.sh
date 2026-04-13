#!/usr/bin/env bash
set -euo pipefail

# MIMIC - Setup for a fresh machine with an NVIDIA GPU.
#
# Usage:
#   bash setup.sh                       # install deps, emulator, ISO, download data
#   bash setup.sh --run                 # also starts training after setup
#   bash setup.sh --rsync               # pull pre-built shards from Machine A
#   bash setup.sh --models              # also pull released checkpoints from HuggingFace
#   bash setup.sh --run --model small   # extra args forwarded to train.py

DATA_DIR="${DATA_DIR:-data/fox_hal_v2}"
EMULATOR_DIR="emulator"
ISO_PATH="melee.iso"
RUN_AFTER=false
RSYNC_DATA=false
PULL_MODELS=false
EXTRA_TRAIN_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run)       RUN_AFTER=true; shift ;;
        --rsync)     RSYNC_DATA=true; shift ;;
        --models)    PULL_MODELS=true; shift ;;
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
    if ! command -v unzip &>/dev/null; then
        echo "  Installing unzip ..."
        apt-get update -qq 2>&1 | tail -1
        apt-get install -y -qq unzip 2>&1 | tail -1
    fi
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
        root@194.14.47.19:/root/MIMIC/data/fox_hal_v2/ "$DATA_DIR/"
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
    curl -fsSL https://claude.ai/install.sh | bash
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
    apt-get update -qq 2>&1 | tail -1
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
        echo "  Created .env from .env.example (all fields empty)."
        echo ""
        echo "  Fastest path: scp an existing .env from another machine."
        echo "  Otherwise, fill in these fields in .env:"
        echo "      DISCORD_BOT_TOKEN              (discord.com/developers/applications)"
        echo "      BOT_SLIPPI_CODE                (e.g. MIMIC#01)"
        echo "      SLIPPI_UID, SLIPPI_PLAY_KEY,   (from a Slippi user.json)"
        echo "      SLIPPI_CONNECT_CODE,"
        echo "      SLIPPI_DISPLAY_NAME,"
        echo "      SLIPPI_LATEST_VERSION"
        echo "      HF_TOKEN, WANDB_API_KEY        (optional; for training + uploads)"
    else
        echo "  .env.example missing — skipping."
    fi
fi

# Ensure slippi_home dir exists. user.json itself is either:
#   (a) already present (you placed it there yourself), or
#   (b) synthesized at runtime by discord_bot.py / play_netplay.py from the
#       SLIPPI_UID / SLIPPI_PLAY_KEY / SLIPPI_CONNECT_CODE / ... env vars in .env.
# So we only warn if BOTH the file is missing AND .env has no creds.
mkdir -p slippi_home/Slippi
if [[ ! -f slippi_home/Slippi/user.json ]]; then
    if [[ -f .env ]] && grep -qE '^SLIPPI_UID=.+' .env && grep -qE '^SLIPPI_PLAY_KEY=.+' .env; then
        echo "  slippi_home/Slippi/user.json will be synthesized from .env creds on first run."
    else
        echo "  ⚠ slippi_home/Slippi/user.json is missing AND .env has no SLIPPI_UID/SLIPPI_PLAY_KEY."
        echo "     Fill those in .env, or drop an existing user.json at slippi_home/Slippi/user.json."
    fi
fi

# ── 11. Released MIMIC models from HuggingFace (optional) ──────────────────
if $PULL_MODELS; then
    echo ""
    echo "── Pulling released models from huggingface.co/erickfm/MIMIC ──"
    # The repo is public — no HF_TOKEN required to read. We still export
    # it if it's set in .env so rate limits are lifted for the download.
    if [[ -f .env ]]; then
        # shellcheck disable=SC2046
        export $(grep -E '^(HF_TOKEN|HUGGING_FACE_HUB_TOKEN)=' .env | xargs -d '\n' -r) 2>/dev/null || true
    fi
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('erickfm/MIMIC', local_dir='hf_checkpoints')
" || { echo "  ❌ HF download failed — check your network connection"; exit 1; }

    mkdir -p checkpoints
    # character → (checkpoint filename, data dir)
    declare -A CP_NAMES=(
      [fox]="fox-20260413-rope-32k.pt"
      [falco]="falco-20260412-relpos-28k.pt"
      [cptfalcon]="cptfalcon-20260412-relpos-27k.pt"
      [luigi]="luigi-20260412-relpos-5k.pt"
    )
    declare -A DATA_DIRS=(
      [fox]="data/fox_hal_v2"
      [falco]="data/falco_v2"
      [cptfalcon]="data/cptfalcon_v2"
      [luigi]="data/luigi_v2"
    )
    for char in fox falco cptfalcon luigi; do
        src="hf_checkpoints/$char"
        cp_name="${CP_NAMES[$char]}"
        data_dir="${DATA_DIRS[$char]}"
        if [[ -f "$src/model.pt" ]]; then
            ln -sf "../$src/model.pt" "checkpoints/$cp_name"
            mkdir -p "$data_dir"
            for f in hal_norm.json controller_combos.json cat_maps.json stick_clusters.json norm_stats.json; do
                if [[ -f "$src/$f" ]]; then
                    ln -sf "../../$src/$f" "$data_dir/$f"
                fi
            done
            echo "  $char → checkpoints/$cp_name + $data_dir/"
        else
            echo "  ⚠ $src/model.pt missing, skipping $char"
        fi
    done
    echo "  Models ready — Discord bot and play_netplay.py can find them via the symlinks."
fi

echo ""
echo "=== Setup complete ==="
echo "  Dolphin:  $EMULATOR_DIR/squashfs-root/usr/bin/dolphin-emu"
echo "  ISO:      $ISO_PATH"
echo "  Data:     $DATA_DIR"
echo "  Display:  \$DISPLAY=$DISPLAY"
if $PULL_MODELS; then
    echo "  Models:  hf_checkpoints/ (symlinked into checkpoints/ + data/*_v2/)"
fi
echo ""
echo "Next steps:"
if ! $PULL_MODELS; then
    echo "  • Pull trained checkpoints:    bash setup.sh --models"
fi
echo "  • Make sure .env is filled in  (scp from another machine is fastest)"
echo "  • Run the Discord bot:         python3 tools/discord_bot.py"
echo "  • Or play vs a CPU locally:    python3 tools/run_mimic_via_hal_loop.py --help"

# ── 12. Optionally start training ───────────────────────────────────────────
if $RUN_AFTER; then
    echo ""
    echo "── Starting training ──"
    python3 train.py --data-dir "$DATA_DIR" "${EXTRA_TRAIN_ARGS[@]}"
fi
