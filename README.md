# MIMIC

**Super Smash Bros. Melee bots: small decoder-only transformers trained by next-token prediction
on human Slippi replays, then fine-tuned with RL from verifiable rewards.**

Each game frame is one token (60 tokens/s). The model reads a context of recent frames and
predicts the controller input for the next frame: behavior cloning as next-token CE. At inference
it drives a virtual GameCube controller through Dolphin via
[libmelee](https://github.com/altf4/libmelee), sampling one action per frame in realtime. A bot
can play a CPU locally, play itself, take on a human with a GameCube adapter, or join a human
over **Slippi Online Direct Connect**, optionally queued through a Discord bot.

RL runs in a native build of Melee compiled from the decomp (no emulator): PPO with a KL penalty
to the frozen BC reference, against that same reference as the opponent, with rewards computed
from exact engine state.

Weights on HuggingFace: **[erickfm/MIMIC](https://huggingface.co/erickfm/MIMIC)**

---

## Results

Current production bots (all `M`, ~20M params, one per character or rank tier; full list on HF):

| Bot | Training data | Val loss |
|---|---|---|
| `fox-master` | 97k master-rank Fox perspectives (ranked Slippi) | 0.68 |
| `fox-diamond` | diamond-rank Fox | 0.73 |
| `fox-platinum` | platinum-rank Fox | 0.75 |
| `puff` | master-tier Jigglypuff | 0.66 |
| `falco` | 9,110 games | 0.74 |
| `cptfalcon` | 9,404 games | 0.71 |
| `luigi` | 1,951 games | ~1.0 |

Val loss is comparable only within a row's own dataset, and it's a weak proxy for strength:
CE weights an error by how often it happens, not by what it costs (a slightly-off stick angle
is common and free; an airdodge off-stage is rare and loses a stock). So models are promoted
on head-to-head win rate, not val.

**Benchmark:** the pure-BC `fox-master` wins **29% (4–10)** against a hosted
[Phillip](https://github.com/vladfi1/slippi-ai) (PHAI#591) over Slippi netplay.

**RL:** PPO from `fox-master` with a stock-delta + damage-delta reward beats the frozen BC
reference **56.3% over 1,152 games** in the native sim, replicated on two evaluation seeds. Not
yet promoted. Single-skill rewards (L-cancel rate, combo length, ...) raise their own metric
without raising win rate; see `docs/research-notes-2026-10-23*.md`.

RL on a small model (5.4M params, a quarter of `fox-master`'s size, 56k BC steps) for 100 updates
beats `fox-master` itself: **130–70 (65%) over 200 Dolphin matches** (109–83 in the sim), and
154–46 against its own BC start. Not promoted; see `docs/research-notes-2026-10-26.md`.

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

### Against you, locally (GameCube adapter)

```bash
python3 tools/play_human.py \
  --ckpt hf_checkpoints/fox-master/model.pt --data-dir hf_checkpoints/fox-master \
  --dolphin-path ./emulator/squashfs-root/usr/bin/dolphin-emu --iso-path ./melee.iso
```

Your adapter (Wii U mode) is port 1 and the bot is port 2. You pick your character and the stage.

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

```bash
torchrun --nproc_per_node=2 train.py --data-dir data/fox_v2 --run-name fox-$(date +%Y%m%d)-base \
  --batch-size 384 --max-samples 215040000
```

The defaults are the production recipe: model `M`, Muon on the hidden matrices (AdamW on
embeddings, heads and norms), lr 3e-4 with cosine decay to 1e-6 and no warmup, bf16 autocast,
`torch.compile`. `--max-samples` is the total over all GPUs; the example is the Fox champion's
budget (280k steps at an effective batch of 768). Single GPU: `python3 train.py ...
--batch-size 384 --grad-accum-steps 2`. Logs go to [Weights & Biases](https://wandb.ai/) (set
`WANDB_API_KEY` in `.env`).

For research, `tools/fast_proxy.sh NAME S 1900 SEED [flags]` trains the 5.4M `S` model for 1,900
steps (~5 min on a 4090) and scores it on a fixed held-out set.

To build fresh v2 shards from `.slp` replays:

```bash
python3 tools/slp_to_shards.py \
  --slp-dir data/falco_all_slp --meta-dir data/fox_v2 \
  --staging-dir data/falco_v2 --mimic-norm data/fox_v2/mimic_norm.json \
  --character 22 --shard-gb 0.8 --val-frac 0.1 --workers 8
```

---

## Model

A decoder-only transformer over frame tokens:

- **Sizes:** `M` (d=512, 6 layers, 8 heads, ~20.0M params; production) and `S` (d=256, 6
  layers, 4 heads, ~5.4M; research proxy). Dropout 0.2.
- **Token embedding:** each frame is a 184-dim feature vector projected to d by one Linear: stage
  embedding (4) + both characters (2×12) + both action states (2×32) + 18 numeric/flag features
  per player (positions, velocities, percent, hitstun, shield, ...) + the player's own previous
  controller input as a 56-dim one-hot.
- **Position:** Shaw-style learned relative-position biases (RoPE underperformed and was removed).
- **Output:** the next frame's controller input, factorized into four categorical heads decoded
  autoregressively inside each token: shoulder (3) → c-stick (9) → main stick (37 k-means
  clusters) → buttons (7). Each head conditions on the earlier heads' classes (teacher-forced in
  training, sampled at inference), so together they model the joint distribution.
- **Targets:** shifted one frame (`target[i] = input[i+1]`). Replays log the frame's state after
  that frame's input was applied, so an unshifted target can be read off the state.

**Context: full vs stream.**
- **Full** (default, all production models): trained on 180-token sequences with full causal
  attention, run at inference on a rolling 180-token (3 s) context. Dropping the oldest token
  invalidates every cached K/V above layer 1, so each frame recomputes the whole forward.
- **Stream** (`--attn-window 180` plus a longer `--seq-len`): sliding-window attention at every
  layer. The receptive field grows to 6 × 179 + 1 = 1,075 tokens (~18 s), and decoding is exact
  with a per-layer ring-buffer KV cache (`mimic/kv_cache.py`): one step per frame, ~22× faster at
  batch 64. Every inference path streams these models automatically. Not the default yet: trained
  on sequences longer than its receptive field (so training matches streaming inference), it
  scores worse than full.

**Button vocabulary.** Seven classes: A, B, Z, Jump, None, plus `TRIG` (digital L/R) and `A_TRIG`
(shield grab). Melee reads shield and L-cancel from the analog trigger, but tech, airdodge and
wavedash need the digital press, so without `TRIG` a bot can't do any of the three.

## RL

`rlvr/sim/` wraps a native Linux build of Melee compiled from the decomp: no emulator, inputs land
one frame later as in Dolphin, and current-Slippi 1v1 replays re-simulate physics-exact. Many sim
processes run in parallel and the policy runs batched on the GPU.

- **Trainer** (`tools/sim_rl.py`): PPO initialized from the BC checkpoint, a per-token KL penalty
  to the frozen reference, and a value head on the reference's features. The opponent is the
  frozen reference itself, never the in-game CPU, which doesn't punish reward hacking.
- **Rewards** (`rlvr/sim/vr/`): verifiable, computed from exact engine state each frame, e.g.
  stock delta, damage delta, kills at low percent, recovery, tech, L-cancel. Outcome rewards
  (stocks, damage) are the objective. Conditional skills, whose right execution depends on the
  situation (DI direction, for one), are left to BC.
- **Eval** (`tools/sim_h2h.py`): head-to-head against the reference, both port orientations of
  every stage template, a fixed number of games per env (no short-game bias), Wilson interval and
  exact binomial p.
- **Causal probes** (`rlvr/sim/interventions.py`): rewrite one player's controller by rule (e.g.
  never L-cancel, always DI toward stage) and measure the win-rate change against the unmodified
  model.

---

## Project layout

```
.
├── train.py                          # BC training entry point
├── mimic/                            # Core library
│   ├── model.py                      # FramePredictor, relpos attention, heads
│   ├── kv_cache.py                   # Streaming KV cache for sliding-window models
│   ├── frame_encoder.py              # Frame → 512-d encoder
│   ├── features.py                   # Feature encoding, 7-class collapse
│   └── dataset.py                    # Shard streaming
├── rlvr/                             # RL
│   ├── sim/                          # Native-sim env, batched policy, PPO, interventions
│   │   └── vr/                       # Verifiable rewards
│   └── state/                        # GameState + peppi/libmelee adapters (canaries)
├── tools/                            # See tools/README.md for the full index
│   ├── play.py                       # Bot vs CPU / bot vs bot (local)
│   ├── play_netplay.py               # Bot vs human (Slippi netplay)
│   ├── play_human.py                 # Bot vs you (GameCube adapter, local)
│   ├── discord_bot.py                # Discord queue/frontend
│   ├── inference_utils.py            # Shared decode + frame building
│   ├── slp_to_shards.py              # .slp → v2 shards
│   ├── sim_rl.py / sim_h2h.py        # RL and head-to-head in the native sim
│   └── build_savestate_dolphin.sh    # Patched-fork emulator build (dual-pad FFW)
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

- The transformer backbone, relative-position attention, and early data pipeline started from
  [HAL](https://github.com/ericyuegu/hal) by Eric Gu.
- Slippi and [libmelee](https://github.com/altf4/libmelee) by the [Project Slippi](https://slippi.gg/) team.
- Savestate/FFW emulator patches build on [vladfi1's slippi-Ishiiruka fork](https://github.com/vladfi1/slippi-Ishiiruka).
