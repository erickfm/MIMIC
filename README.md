# MIMIC

**Behavior-cloned Super Smash Bros. Melee bots, trained from human Slippi replays.**

MIMIC learns to map game state to controller inputs by watching thousands of
human matches. At inference it drives a virtual GameCube controller through
Dolphin via [libmelee](https://github.com/altf4/libmelee) at 60 fps. Each
trained bot can play a CPU opponent locally, run bot-vs-bot dittos, or join
a human opponent over **Slippi Online Direct Connect** — optionally mediated
by a Discord bot that queues matches.

On top of the behavior-cloned base, an **RLVR** (reinforcement learning from
verifiable rewards) loop fine-tunes individual skills against
engine-confirmed metrics — one skill at a time.

Weights on HuggingFace: **[erickfm/MIMIC](https://huggingface.co/erickfm/MIMIC)**

---

## Results

Current production bots (all ~20M-param transformers, one per character or
rank tier; full list on HF):

| Bot | Training data | Val loss |
|---|---|---|
| `fox-master` | ~77k master-rank Fox perspectives (ranked Slippi) | 0.68 |
| `fox-diamond` | diamond-rank Fox | 0.73 |
| `fox-platinum` | platinum-rank Fox | 0.75 |
| `puff` | master-tier Jigglypuff | 0.66 |
| `falco` | 9,110 games | 0.74 |
| `cptfalcon` | 9,404 games | 0.71 |
| `luigi` | 1,951 games | ~1.0 |

Val losses are only comparable within a row's own dataset — and val loss is
a weak proxy for playing strength in general (cross-entropy weights errors
by frequency, not by consequence), so promotion decisions are made by
head-to-head matches, not val.

**Benchmark:** the pure-BC `fox-master` wins **29% (4–10)** against a hosted
[Phillip](https://github.com/vladfi1/slippi-ai) (PHAI#591) over Slippi
netplay — to our knowledge the first third-party head-to-head eval against
Phillip.

**RLVR (in progress):** the first objective, L-cancelling, went from a BC
baseline of 94.2% to **99.2%** success vs CPU (engine-confirmed
avoidable-lag metric, n=2,230) with a tuned PPO recipe. The open problem is
preserving overall playing strength while drilling a skill; current
approach is dense checkpointing + head-to-head selection.

---

## Install

Fresh Linux box with an NVIDIA GPU:

```bash
git clone https://github.com/erickfm/MIMIC.git
cd MIMIC
bash setup.sh
```

`setup.sh` installs Python deps, downloads the **current** Slippi Online
Dolphin (Slippi's version gate rejects stale builds), downloads the Melee
1.02 NTSC ISO, starts Xvfb (for headless Dolphin), and copies `.env.example`
to `.env`. Add `--models` to also pull the released checkpoints from
HuggingFace:

```bash
bash setup.sh --models
```

Each character directory under `hf_checkpoints/` is self-contained —
`model.pt` plus every norm/metadata JSON needed for inference — so it works
directly as both the checkpoint source and the `--data-dir`.

Verify the GPU afterward:

```bash
python3 -c "import torch; print(torch.cuda.get_device_name(0))"
```

### Tokens (optional, for training and model uploads)

`train.py` and the upload tools load `.env` at startup, so placing these in
`.env` is all that's needed — no separate `wandb login` /
`huggingface-cli login`:

```env
WANDB_API_KEY=...             # from https://wandb.ai/authorize
HF_TOKEN=...                  # from https://huggingface.co/settings/tokens
```

---

## Play

### Against a CPU locally

```bash
python3 tools/play.py \
  --ckpt hf_checkpoints/falco/model.pt \
  --opponent cpu:9 \
  --dolphin-path ./emulator/squashfs-root/usr/bin/dolphin-emu \
  --iso-path ./melee.iso \
  --data-dir hf_checkpoints/falco \
  --character FALCO --opponent-character FALCO \
  --stage FINAL_DESTINATION
```

### Bot vs bot (watchable ditto)

Same command, but point `--opponent` at a second checkpoint instead of
`cpu:9`.

### Against a human over Slippi netplay

```bash
python3 tools/play_netplay.py \
  --checkpoint hf_checkpoints/falco/model.pt \
  --dolphin-path ./emulator/squashfs-root/usr/bin/dolphin-emu \
  --iso-path ./melee.iso \
  --data-dir hf_checkpoints/falco \
  --character FALCO \
  --connect-code YOUR_CODE#123
```

You enter the bot's connect code on your side; the bot enters yours.
Slippi rollback netplay pairs you up. The bot plays N back-to-back matches
in one persistent Dolphin session.

---

## Discord bot

`tools/discord_bot.py` is a Discord front-end that lets anyone queue a match
against the bot with a prefix command. It runs one persistent Dolphin
session per queued match series, joins the user's Slippi Direct Connect
lobby, plays, and uploads the saved `.slp` replay back to the channel.

```
!play <character> <your_code>   # queue a match (e.g. !play falco WAVE#666)
!<character> <your_code>        # shortcut (e.g. !fox-master WAVE#666)
!queue                          # show what's playing + queued
!cancel                         # remove your pending match
!info                           # bot's connect code, character list, usage
!reload                         # re-sync characters from HuggingFace
```

The character list is not hardcoded: the bot discovers every
`erickfm/MIMIC/{dir}/` on HuggingFace that has a `model.pt` +
`metadata.json` and registers a `!<dir>` shortcut for each (e.g.
`!fox-master`, `!fox-diamond`, `!puff`) — newly uploaded characters appear
after `!reload`.

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

Two GPUs:

```bash
torchrun --nproc_per_node=2 train.py \
  --model mimic --encoder mimic_flat \
  --mimic-mode --mimic-controller-encoding \
  --stick-clusters hal37 --plain-ce \
  --lr 3e-4 --batch-size 256 --grad-accum-steps 1 \
  --max-samples 16777216 \
  --data-dir data/fox_v2 \
  --self-inputs --reaction-delay 0 \
  --run-name fox-$(date +%Y%m%d)-relpos \
  --no-warmup --cosine-min-lr 1e-6
```

Single GPU: swap `torchrun --nproc_per_node=2` for `python3` and use
`--batch-size 64 --grad-accum-steps 8` (keeps the effective batch at 512).
A typical character trains in ~1.5–2.5 hr on 2×RTX 5090. Training logs to
[Weights & Biases](https://wandb.ai/) (set `WANDB_API_KEY` in `.env`).

Use `--model mimic` (Shaw relative-position attention) — the RoPE presets
underperform it and are deprecated. Full features are the default; only
pass `--mimic-minimal-features` to reproduce legacy baselines.

To build fresh v2 shards from `.slp` replays:

```bash
python3 tools/slp_to_shards.py \
  --slp-dir data/falco_all_slp --meta-dir data/fox_v2 \
  --staging-dir data/falco_v2 --mimic-norm data/fox_v2/mimic_norm.json \
  --character 22 --shard-gb 0.8 --val-frac 0.1 --workers 8
```

---

## Architecture

~20M-parameter causal transformer:

- **Encoder**: `Linear(184 → 512)` over a per-frame feature vector:
  stage (4) + 2× character (12) + 2× action (32) + 2× 18-dim gamestate
  (13 numeric + 5 flags per player) + controller history (56 one-hot:
  37 stick + 9 c-stick + 7 button + 3 shoulder)
- **Transformer**: 6 layers, 8 heads, d=512, dropout 0.2,
  180-frame context (~3 s), Shaw relative-position attention
- **Heads (autoregressive with detach)**: shoulder(3) → c_stick(9) →
  main_stick(37 k-means clusters) → buttons(7)

The 7-class button vocabulary extends the usual `{A, B, Jump, Z, None}`
with `TRIG` (L/R digital press) and `A_TRIG` (shield grab). Melee splits
shoulder events by analog vs digital: shield and L-cancel read the analog
threshold, but tech, airdodge, and **wavedash** require the digital press —
without a TRIG class a bot is structurally incapable of all three.

v2 shards shift button targets forward by one frame
(`target[i] = buttons[i+1]`) so the model learns to predict the *next*
input given the current state, rather than cheat via post-frame action
state encoding the answer. Train with `--reaction-delay 0` on v2 shards.

## RLVR

`rlvr/online/` holds the reinforcement-learning loop: parallel Dolphin
actors (fast-forward emulation, ~5.7× realtime for model-vs-model on one
GPU), PPO with group-normalized advantages, per-skill verifiable rewards
scored against the engine (post-match replay parsing, not proxies), and a
savestate harness (a small patch to the Slippi Dolphin fork adds
`SAVESTATE`/`LOADSTATE` pipe verbs — `tools/build_savestate_dolphin.sh`)
for drilling harvested miss states. Skills are tackled one at a time;
L-cancel is the first and is in progress.

---

## Project layout

```
.
├── train.py                          # BC training entry point
├── mimic/                            # Core library
│   ├── model.py                      # FramePredictor, attention variants, heads
│   ├── frame_encoder.py              # Frame → 512-d encoder
│   ├── features.py                   # Feature encoding, 7-class collapse
│   └── dataset.py                    # Shard streaming
├── rlvr/                             # RL from verifiable rewards
│   └── online/                       # Live-Dolphin actors, PPO, savestate drill
├── tools/                            # See tools/README.md for the full index
│   ├── play.py                       # Bot vs CPU / bot vs bot (local)
│   ├── play_netplay.py               # Bot vs human (Slippi netplay)
│   ├── discord_bot.py                # Discord queue/frontend
│   ├── inference_utils.py            # Shared decode + frame building
│   ├── slp_to_shards.py              # .slp → v2 shards
│   └── build_savestate_dolphin.sh    # Patched-fork emulator build (RL)
├── docs/                             # Dev journal + setup guides
├── setup.sh                          # Fresh-machine bootstrap
├── checkpoints/                      # Local checkpoints (gitignored)
├── hf_checkpoints/                   # Released models pulled from HF
├── data/                             # Training shards (gitignored)
├── emulator*/                        # Dolphin builds (gitignored; see setup.sh)
└── slippi_home/                      # Bot's Slippi credentials (gitignored)
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
- Savestate/FFW emulator patches build on [vladfi1's slippi-Ishiiruka fork](https://github.com/vladfi1/slippi-Ishiiruka).
