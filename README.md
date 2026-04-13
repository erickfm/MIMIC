# MIMIC: Melee Imitation Model for Input Cloning

> **For agent/developer orientation, see [CLAUDE.md](CLAUDE.md).** It covers
> naming gotchas, shard alignment pitfalls, data directories, and common
> mistakes. This README is for humans.

MIMIC is a behavior-cloning bot for Super Smash Bros. Melee. It watches human
Slippi replays and learns to predict controller inputs from game state. At
inference it drives a virtual GameCube controller through Dolphin via libmelee
at 60 fps. The same model can play a CPU opponent locally or join a human
opponent over **Slippi Online Direct Connect**, mediated by an included
[Discord bot](#play-against-the-bot-on-discord).

The reference implementation is [HAL](https://github.com/ericyuegu/hal) by
Eric Gu, which MIMIC reproduces and extends with a 7-class button head, v2
shard alignment, and fixes for several silent inference bugs that plagued
earlier BC bots (including the missing digital L press that prevented
wavedashes — see research notes 2026-04-13).

---

## Current state (2026-04-13)

Trained character-specific models for Fox, Falco, Captain Falcon, and Luigi.
All play actively in Dolphin (no stuck modes, diverse actions) and execute
advanced techniques including **wavedashes**, shield grabs, dash dances, and
rapid jabs. In bot-vs-bot Luigi dittos the models convert 70–83% of jumps
into wavedashes, matching real high-level Luigi play.

| Character | Games trained | Val btn F1 | Sample replay |
|---|---|---|---|
| Fox | 17,319 | — (10K steps only, legacy) | — |
| Falco | 9,110 | 88.2% | `replays/Game_20260413T035445.slp` |
| Captain Falcon | 9,404 | 89.9% | `replays/Game_20260413T040411.slp` |
| Luigi | 1,951 (early-stop) | ~91% | `replays/Game_20260413T041156.slp` |

Best-val checkpoint per character is bundled as `mimic_best_checkpoints.zip`
(828 MB, gitignored).

---

## Architecture

Using `--model hal`, MIMIC matches HAL's GPTv5Controller (~19.95M params):

```
Slippi Frame ──► HALFlatEncoder (Linear 166→512) ──► 512-d per-frame vector
                                                          │
256-frame window ──► + Relative Position Encoding ────────┘
                         │
                    6× Pre-Norm Causal Transformer Blocks (512-d, 8 heads)
                         │
                    Autoregressive Output Heads (with detach)
                         │
              ┌──────────┼──────────┬───────────┐
           shoulder(3) c_stick(9) main_stick(37) buttons(7)
```

### Button head (7-class)

MIMIC uses a 7-class button vocabulary derived from Melee's input resolution
rules, distinct from HAL's 5-class (`A, B, Jump, Z, None`):

| Class | Meaning |
|---|---|
| 0 | A |
| 1 | B |
| 2 | Z |
| 3 | JUMP (X or Y) |
| 4 | TRIG (digital L or R) |
| 5 | A_TRIG (shield grab) |
| 6 | NONE |

**The TRIG class must actually press the digital L button** (`ctrl.press_button(BUTTON_L)`),
not just send the analog shoulder value. Analog shoulder triggers shielding
but NOT airdodge, L-cancel, tech, or wavedash — those need the digital press.
HAL's 5-class head has no TRIG class and cannot airdodge by design, which is
why HAL-lineage bots never demonstrated wavedashes. This was silently broken
in MIMIC until 2026-04-13 — see `docs/research-notes-2026-04-13.md`.

### v2 shard alignment

The 2026-04-11c shift fixed a training-time bug where the game state at frame
`i` already reflected the button press at frame `i` (because melee-py returns
post-frame state alongside pre-frame controller inputs). v2 shards in
`data/*_v2/` have targets shifted forward by 1 frame — `target[i] = buttons[i+1]`
— so the model predicts what to press *next* given the current state, which
matches inference semantics. **Train with `--reaction-delay 0` and NO
`--controller-offset` on v2 shards.** See `docs/research-notes-2026-04-11c.md`.

### Input features

9 numeric features per player (ego + opponent = 18 total):
`percent, stock, facing, invulnerable, jumps_left, on_ground, shield_strength, position_x, position_y`

Plus categorical embeddings: stage(4d), 2× character(12d), 2× action(32d).
Plus controller state from previous frame as a 56-dim one-hot
(37 stick + 9 c-stick + 7 button + 3 shoulder). Total input per frame: 166 →
projected to 512.

---

## Setup

Fresh machine with an NVIDIA GPU:

```bash
git clone https://github.com/erickfm/MIMIC.git
cd MIMIC
bash setup.sh
```

`setup.sh` handles:
- Git LFS pull (Dolphin AppImage, sample shards)
- Python dependencies (torch, melee, `discord.py`, `python-dotenv`, py-slippi)
- Slippi-capable Dolphin extracted into `./emulator/`
- Melee 1.02 NTSC ISO downloaded to `./melee.iso`
- Training data into `./data/fox_hal_full/` (or `--rsync` from another machine)
- GitHub CLI + Claude Code
- Headless display (Xvfb on `:99`, auto-started and added to `.bashrc`)
- `.env` copied from `.env.example` for the Discord bot
- `./slippi_home/Slippi/` skeleton created (you drop `user.json` there)

After `setup.sh`, verify your GPU:

```bash
python3 -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

## Training

v2 shards, relpos, `--self-inputs`, 7-class button head:

```bash
# Single GPU, effective batch 512 via grad accum
python3 train.py \
  --model hal --encoder hal_flat \
  --hal-mode --hal-minimal-features --hal-controller-encoding \
  --stick-clusters hal37 --plain-ce \
  --lr 3e-4 --batch-size 512 \
  --max-samples 16777216 \
  --data-dir data/fox_hal_v2 \
  --reaction-delay 0 --self-inputs \
  --run-name my-run \
  --no-warmup --cosine-min-lr 1e-6
```

Key settings: AdamW, no warmup, CosineAnnealingLR (eta_min=1e-6), dropout
0.2, weight decay 0.01, gradient clip 1.0, sequence length 256, BF16 AMP
with FP32 upcast for relpos attention. 32K steps is typical (~2h on an
RTX 5090).

Build v2 shards from .slp replays:

```bash
python3 tools/slp_to_shards.py \
  --slp-dir data/falco_all_slp --meta-dir data/fox_hal_full \
  --repo dummy --no-upload --staging-dir data/falco_v2 --keep-staging \
  --hal-norm data/fox_hal_full/hal_norm.json \
  --character 22 --shard-gb 0.8 --val-frac 0.1 --workers 8
```

---

## Inference

### Local: bot vs CPU level 9

```bash
python3 tools/run_mimic_via_hal_loop.py \
  --checkpoint checkpoints/falco-20260412-relpos-28k.pt \
  --dolphin-path ./emulator/squashfs-root/usr/bin/dolphin-emu \
  --iso-path ./melee.iso \
  --data-dir data/falco_v2 \
  --character FALCO --cpu-character FALCO --cpu-level 9 \
  --stage FINAL_DESTINATION
```

### Local: bot vs bot (watchable ditto)

```bash
python3 tools/head_to_head.py \
  --p1-checkpoint checkpoints/falco-20260412-relpos-28k.pt \
  --p2-checkpoint checkpoints/falco-20260412-relpos-28k.pt \
  --dolphin-path ./emulator/squashfs-root/usr/bin/dolphin-emu \
  --iso-path ./melee.iso \
  --data-dir data/falco_v2 \
  --p1-character FALCO --p2-character FALCO \
  --stage FINAL_DESTINATION
```

### Online: bot joins a Slippi Direct Connect lobby

```bash
python3 tools/play_netplay.py \
  --checkpoint checkpoints/falco-20260412-relpos-28k.pt \
  --dolphin-path ./emulator/squashfs-root/usr/bin/dolphin-emu \
  --iso-path ./melee.iso \
  --data-dir data/falco_v2 \
  --character FALCO \
  --connect-code YOUR_CODE#123
```

You enter the bot's code (from `slippi_home/Slippi/user.json`) on your side,
the bot enters yours on its side, and Slippi rollback netplay pairs you up.

**Don't run inference while training on the same GPU.** Frame drops from GPU
contention will make gameplay look broken when the model is actually fine.

---

## Play against the bot on Discord

MIMIC ships with a Discord bot (`tools/discord_bot.py`) that lets anyone
request a match via prefix command. The bot queues requests, spawns a
Dolphin instance per match, joins the user's Slippi Direct Connect lobby,
plays the game, and uploads the replay file back to the channel.

**One-time setup:**

1. Run `bash setup.sh` on a machine with a GPU, Slippi Dolphin, the ISO,
   and the character checkpoints.
2. Create a Slippi account for the bot via Slippi Launcher. Log in once
   to generate `user.json`. Copy it to `./slippi_home/Slippi/user.json`
   in the repo.
3. Create a Discord application at https://discord.com/developers/applications
   → Bot → Reset Token. Enable **Message Content Intent**. Invite the bot
   to your server with this OAuth URL (replacing the client id):
   ```
   https://discord.com/oauth2/authorize?client_id=YOUR_APP_ID&scope=bot&permissions=51200
   ```
4. Edit `.env` (auto-copied from `.env.example` by setup.sh):
   ```env
   DISCORD_BOT_TOKEN=your_bot_token
   BOT_SLIPPI_CODE=MIMIC#01
   SLIPPI_HOME=./slippi_home
   DOLPHIN_PATH=./emulator/squashfs-root/usr/bin/dolphin-emu
   ISO_PATH=./melee.iso
   ```
5. Run the bot: `python3 tools/discord_bot.py`

**Commands (prefix `!`):**

- `!play <character> <your_code>` — queue a match. E.g. `!play falco ERIK#456`.
  Characters: `FOX`, `FALCO`, `CPTFALCON` (aliases: `falcon`, `cf`), `LUIGI`.
- `!queue` — show what's playing and what's queued.
- `!cancel` — remove your pending match.
- `!info` — show the bot's own connect code, character list, and usage.

Full setup guide with troubleshooting: [`docs/discord-bot-setup.md`](docs/discord-bot-setup.md).

---

## Portability

The repo is designed to be `rsync`/`scp`-able to any Linux box with an
NVIDIA GPU:

- `.env` uses **relative paths** (`./emulator/...`, `./melee.iso`,
  `./slippi_home`). The Discord bot resolves them against the repo root
  at runtime.
- `slippi_home/` is gitignored and bundled with the repo, so uploading the
  directory to a new machine carries the Slippi credentials with it.
- `setup.sh` is idempotent — re-running it on a new machine installs the
  same dependencies, extracts Dolphin, and downloads the ISO without
  touching your `.env` or `slippi_home/`.
- All paths referenced in CLI scripts default to the repo-relative versions
  described in `.env.example`.

---

## Project structure

```
.
├── CLAUDE.md               # Agent/developer orientation (read first!)
├── README.md               # This file
├── .env.example            # Template for Discord bot config
├── setup.sh                # One-shot machine setup
├── train.py                # Training loop (v2 shards, relpos, BF16 AMP)
│
├── mimic/                  # Core library
│   ├── model.py            # FramePredictor, HALPredictionHeads, relpos attention
│   ├── frame_encoder.py    # HALFlatEncoder
│   ├── features.py         # 7-class button collapse, cluster centers, norm
│   └── dataset.py          # StreamingMeleeDataset
│
├── tools/                  # Inference, data pipeline, diagnostics
│   ├── play_netplay.py     # Bot vs human over Slippi Direct Connect
│   ├── discord_bot.py      # Discord frontend (prefix commands + queue)
│   ├── run_mimic_via_hal_loop.py  # Bot vs CPU locally
│   ├── head_to_head.py     # Bot vs bot (watchable ditto)
│   ├── inference_utils.py  # Shared inference (build_frame, decode_and_press)
│   ├── slp_to_shards.py    # .slp replays → v2 shards
│   ├── extract_wavedashes.py  # Build wavedash-only diagnostic dataset
│   ├── inspect_frame.py    # Per-frame model input/output inspector
│   ├── validate_checkpoint.py
│   └── verify_hal_pipeline.py
│
├── docs/                   # Research notes (2026-04-07+)
│   ├── discord-bot-setup.md
│   ├── research-notes-2026-04-13.md   # TRIG button L-press fix
│   ├── research-notes-2026-04-12.md   # Multi-character v2 training
│   ├── research-notes-2026-04-11c.md  # v2 shard target shift
│   └── archive/                        # 2026-03-14 through 2026-04-06
│
├── checkpoints/            # Best-val per character (v2 run)
├── data/                   # Shards (fox/falco/cptfalcon/luigi _v2)
├── emulator/               # Extracted Slippi Dolphin (from setup.sh)
├── slippi_home/            # Bot's Slippi user.json (gitignored!)
└── replays/                # Saved .slp replays from inference runs
```

---

## Research notes & known pitfalls

The `docs/` directory is a chronological journal. Notable recent entries:

- **2026-04-13** — The TRIG button L-press fix. Every 7-class model trained
  before this date was incapable of airdodge/wavedash/L-cancel/tech because
  `decode_and_press` never called `press_button(BUTTON_L)`. Post-fix Luigi
  dittos produce wavedashes on 70-83% of jumps.
- **2026-04-12** — Multi-character v2 training (Falco, CptFalcon, Luigi)
  and bot-vs-bot ditto validation.
- **2026-04-11c** — The v2 shard target shift. Post-frame game state leaks
  button presses in old shards, making the model memorize action→button
  rather than learn to initiate actions. Fixed by shifting targets forward
  by 1 frame in `tools/slp_to_shards.py`.

Historical notes in `docs/archive/` (2026-03-14 through 2026-04-06) contain
claims that were later disproven (e.g. "HAL doesn't overfit", "26.3M params",
"shoulder is analog-only"). Don't treat specific numbers in the archive as
ground truth — verify against code.

---

## Dependencies

Core:
```bash
pip install torch numpy pandas pyarrow wandb tensordict huggingface_hub melee==0.45.1 py-slippi
```

Discord bot (optional):
```bash
pip install -r requirements-discord.txt   # discord.py, python-dotenv
```

Or just run `bash setup.sh` which installs everything.

---

## License

See [LICENSE](LICENSE).
