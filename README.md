# MIMIC

**Behavior-cloned Super Smash Bros. Melee bots, trained from human Slippi replays.**

MIMIC learns to map game state to controller inputs by watching thousands of
human matches. At inference it drives a virtual GameCube controller through
Dolphin via [libmelee](https://github.com/altf4/libmelee) at 60 fps. Each
trained bot can play a CPU opponent locally, run bot-vs-bot dittos, or join
a human opponent over **Slippi Online Direct Connect** — optionally mediated
by a Discord bot that queues matches.

Four character models are released: **Fox, Falco, Captain Falcon, Luigi**.
The models wavedash — bot-vs-bot Luigi dittos convert 70–83% of jumps
into wavedashes, which matches high-level human Luigi play.

Weights on HuggingFace: **[erickfm/MIMIC](https://huggingface.co/erickfm/MIMIC)**

---

## Results

| Character | Training games | Val btn F1 | Val main-stick F1 | Val loss |
|---|---|---|---|---|
| Fox            | 17,319 | 87.7% | ~55% | 0.77 |
| Falco          |  9,110 | 88.2% | 58.5% | 0.68 |
| Captain Falcon |  9,404 | 89.9% | 52.2% | 0.71 |
| Luigi          |  1,951 | ~91%  | ~60% | ~1.0 |

Wavedash conversion rates in bot-vs-bot dittos on Final Destination:
Luigi 70–83%, Falco 13–33%, CptFalcon 16%. See `replays/` for sample `.slp`
files playable in Slippi Playback.

---

## Install

Fresh Linux box with an NVIDIA GPU:

```bash
git clone https://github.com/erickfm/MIMIC.git
cd MIMIC
bash setup.sh
```

`setup.sh` installs Python deps, pulls the Dolphin AppImage via Git LFS,
downloads the Melee 1.02 NTSC ISO, starts Xvfb (for headless Dolphin), and
copies `.env.example` to `.env`. Add `--models` to also pull the released
checkpoints from HuggingFace and wire them into `checkpoints/` + `data/*_v2/`
so `tools/discord_bot.py` and `tools/play_netplay.py` can find them
without manual renaming:

```bash
bash setup.sh --models
```

Verify the GPU afterward:

```bash
python3 -c "import torch; print(torch.cuda.get_device_name(0))"
```

Each character directory under `hf_checkpoints/` contains `model.pt` plus
all metadata JSONs needed for inference — no extra data downloads required.
`setup.sh --models` symlinks these into the paths the inference tools
expect.

### Tokens (optional, for training and model uploads)

`train.py` and `tools/upload_models_to_hf.py` both load `.env` at startup,
so any of the following placed in `.env` are picked up automatically:

```env
WANDB_API_KEY=...             # from https://wandb.ai/authorize
HF_TOKEN=...                  # from https://huggingface.co/settings/tokens
```

This means on a fresh machine you can just `scp .env` over and training
runs log to wandb and model uploads push to HuggingFace without any
separate `wandb login` / `huggingface-cli login` step.

---

## Play

### Against a CPU locally

```bash
python3 tools/play_vs_cpu.py \
  --checkpoint checkpoints/falco-20260412-relpos-28k.pt \
  --dolphin-path ./emulator/squashfs-root/usr/bin/dolphin-emu \
  --iso-path ./melee.iso \
  --data-dir data/falco_v2 \
  --character FALCO --cpu-character FALCO --cpu-level 9 \
  --stage FINAL_DESTINATION
```

### Bot vs bot (watchable ditto)

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

### Against a human over Slippi netplay

```bash
python3 tools/play_netplay.py \
  --checkpoint checkpoints/falco-20260412-relpos-28k.pt \
  --dolphin-path ./emulator/squashfs-root/usr/bin/dolphin-emu \
  --iso-path ./melee.iso \
  --data-dir data/falco_v2 \
  --character FALCO \
  --connect-code YOUR_CODE#123
```

You enter the bot's connect code on your side; the bot enters yours.
Slippi rollback netplay pairs you up.

---

## Discord bot

`tools/discord_bot.py` is a Discord front-end that lets anyone queue a match
against the bot with a prefix command. It spawns a Dolphin instance per
match, joins the user's Slippi Direct Connect lobby, plays the game, and
uploads the saved `.slp` replay back to the channel.

```
!play <character> <your_code>   # queue a match (e.g. !play falco WAVE#666)
!queue                          # show what's playing + queued
!cancel                         # remove your pending match
!info                           # bot's connect code, character list, usage
```

Supported characters: `FOX`, `FALCO`, `CPTFALCON` (aliases `falcon`, `cf`),
`LUIGI`.

Setup (one-time per machine):

1. Create a Discord application at <https://discord.com/developers/applications>,
   reset the bot token, enable **Message Content Intent**, and invite the bot
   to your server with `permissions=51200`.
2. Create a Slippi account via Slippi Launcher, log in once to generate
   `user.json`.
3. Fill in `.env` — see `.env.example` for the full list. Minimally:
   ```env
   DISCORD_BOT_TOKEN=...
   BOT_SLIPPI_CODE=MIMIC#01
   SLIPPI_UID=...
   SLIPPI_PLAY_KEY=...
   SLIPPI_CONNECT_CODE=MIMIC#01
   ```
   The bot synthesizes `slippi_home/Slippi/user.json` from these on startup.
4. Run: `python3 tools/discord_bot.py`

Full troubleshooting guide: [`docs/discord-bot-setup.md`](docs/discord-bot-setup.md).

---

## Train your own

```bash
python3 train.py \
  --model hal-rope --encoder hal_flat \
  --hal-mode --hal-minimal-features --hal-controller-encoding \
  --stick-clusters hal37 --plain-ce \
  --lr 3e-4 --batch-size 512 \
  --max-samples 16777216 \
  --data-dir data/fox_v2 \
  --reaction-delay 0 --self-inputs \
  --run-name fox-$(date +%Y%m%d)-rope \
  --no-warmup --cosine-min-lr 1e-6
```

A 32K-step run at batch 512 is typical — about 30 minutes on an RTX 5090
with RoPE, or ~60 minutes with relpos attention. Training logs to
[Weights & Biases](https://wandb.ai/) (set `WANDB_API_KEY` in `.env`).

To build fresh v2 shards from `.slp` replays:

```bash
python3 tools/slp_to_shards.py \
  --slp-dir data/falco_all_slp --meta-dir data/fox_v2 \
  --staging-dir data/falco_v2 --mimic-norm data/fox_v2/mimic_norm.json \
  --character 22 --shard-gb 0.8 --val-frac 0.1 --workers 8
```

---

## Architecture

~20M-parameter causal transformer, matching HAL's GPTv5Controller config:

- **Encoder**: `Linear(166 → 512)` over a 166-dim frame vector:
  stage (4) + 2× character (12) + 2× action (32) + gamestate (18) +
  controller (56 one-hot: 37 stick + 9 c-stick + 7 button + 3 shoulder)
- **Transformer**: 6 layers, 8 heads, d=512, dropout 0.1 or 0.2,
  256-frame context (~4.3s), RoPE *or* Shaw relative position attention
- **Heads (autoregressive with detach)**: shoulder(3) → c_stick(9) →
  main_stick(37 k-means clusters) → buttons(7)

The 7-class button vocabulary extends HAL's 5-class `{A, B, Jump, Z, None}`
with `TRIG` (L/R digital press) and `A_TRIG` (shield grab). Melee splits
shoulder events by analog vs digital: shield and L-cancel read the analog
threshold, but tech, airdodge, and **wavedash** require the digital press.
HAL's 5-class head has no way to emit a shoulder press at all, so
HAL-lineage bots are structurally incapable of teching, airdodging, or
wavedashing.

v2 shards shift button targets forward by one frame
(`target[i] = buttons[i+1]`) so the model learns to predict the *next*
input given the current state, rather than cheat via post-frame action
state encoding the answer. Train with `--reaction-delay 0` on v2 shards.

---

## Project layout

```
.
├── train.py                          # Training entry point
├── mimic/                            # Core library
│   ├── model.py                      # FramePredictor, attention variants, heads
│   ├── frame_encoder.py              # Frame → 512-d encoder
│   ├── features.py                   # Feature encoding, 7-class collapse
│   └── dataset.py                    # Shard streaming
├── tools/
│   ├── play_vs_cpu.py     # Bot vs CPU (local)
│   ├── head_to_head.py               # Bot vs bot (local)
│   ├── play_netplay.py               # Bot vs human (Slippi netplay)
│   ├── discord_bot.py                # Discord queue/frontend
│   ├── inference_utils.py            # Shared decode + frame building
│   ├── slp_to_shards.py              # .slp → v2 shards
│   └── upload_models_to_hf.py        # Package + push to HuggingFace
├── docs/
│   ├── discord-bot-setup.md          # Full Discord bot setup guide
│   └── research-notes-*.md           # Dev journal (chronological)
├── checkpoints/                      # Best-val per character
├── data/                             # Training shards (fox/falco/... _v2)
├── emulator/                         # Dolphin AppImage (via Git LFS)
├── slippi_home/                      # Bot's Slippi credentials (gitignored)
└── replays/                          # Saved .slp from inference runs
```

---

## Contributing

See [`CLAUDE.md`](CLAUDE.md) for a contributor's orientation — naming
conventions, shard alignment pitfalls, training gotchas, and the reasoning
behind architectural choices. Research notes in `docs/` document how we
got here.

## License

See [`LICENSE`](LICENSE).

## Credits

- Architecture and early data pipeline built on [HAL](https://github.com/ericyuegu/hal) (Eric Gu).
- Slippi and [libmelee](https://github.com/altf4/libmelee) by the [Project Slippi](https://slippi.gg/) team.
