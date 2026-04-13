#!/usr/bin/env python3
"""Upload MIMIC character checkpoints to HuggingFace Hub.

Packages the best checkpoint per character (Fox, Falco, CptFalcon, Luigi)
into a single HF model repo with one directory per character. Each dir
contains:

    model.pt                 # the raw PyTorch checkpoint
    config.json              # ModelConfig from ckpt (for reading without torch.load)
    metadata.json            # provenance: step, run name, games, val metrics, etc.
    mimic_norm.json          # normalization stats (required by inference)
    controller_combos.json   # 7-class button combo config
    cat_maps.json
    stick_clusters.json
    norm_stats.json

Plus a top-level README.md (model card).

Usage:
    python3 tools/upload_models_to_hf.py                  # dry run — just build staging dir
    python3 tools/upload_models_to_hf.py --push           # actually push to HF
    python3 tools/upload_models_to_hf.py --repo erickfm/MIMIC --push

Requires huggingface_hub and a valid HF token (via `huggingface-cli login`
or HUGGING_FACE_HUB_TOKEN env var).
"""

import argparse
import json
import shutil
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
STAGING_DIR = REPO_ROOT / "_hf_staging"

# Load .env for HF_TOKEN / HUGGING_FACE_HUB_TOKEN so `--push` just works.
try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass


@dataclass
class CharacterEntry:
    name: str                 # directory name in the HF repo ("fox", "falco", ...)
    display: str              # display name in the README ("Fox", "Falco", ...)
    melee_enum: str           # melee.Character enum name ("FOX", "FALCO", ...)
    checkpoint: Path          # local path to .pt
    data_dir: Path            # local data dir with metadata JSONs
    games_trained: int        # number of filtered training games
    val_btn_f1: str           # val button F1 (or "n/a" for legacy runs)
    val_main_f1: str          # val main stick F1
    val_loss: str             # val total loss
    training_notes: str       # free-text context

    def to_metadata_dict(self) -> dict:
        d = asdict(self)
        d["checkpoint"] = str(self.checkpoint.relative_to(REPO_ROOT))
        d["data_dir"] = str(self.data_dir.relative_to(REPO_ROOT))
        return d


# Ground-truth provenance for each character. Metrics pulled from research
# notes 2026-04-12 and 2026-04-13.
CHARACTERS: list[CharacterEntry] = [
    CharacterEntry(
        name="fox",
        display="Fox",
        melee_enum="FOX",
        checkpoint=REPO_ROOT / "checkpoints/fox-20260413-rope-32k.pt",
        data_dir=REPO_ROOT / "data/fox_v2",
        games_trained=17319,
        val_btn_f1="87.7%",
        val_main_f1="~55%",
        val_loss="0.77",
        training_notes=(
            "Trained 2026-04-13, fox-rope-v2 run. v2 shards, RoPE position "
            "encoding (hal-rope preset), --self-inputs, dropout 0.1, 32K "
            "steps at batch 512. Replaces the legacy hal-7class-v2-long "
            "checkpoint which was trained without --self-inputs and stuck "
            "at val loss 2.27. The self-inputs flag was the real fix — "
            "RoPE vs relpos is secondary. Metrics land alongside "
            "Falco/CptFalcon/Luigi."
        ),
    ),
    CharacterEntry(
        name="falco",
        display="Falco",
        melee_enum="FALCO",
        checkpoint=REPO_ROOT / "checkpoints/falco-20260412-relpos-28k.pt",
        data_dir=REPO_ROOT / "data/falco_v2",
        games_trained=9110,
        val_btn_f1="88.2%",
        val_main_f1="58.5%",
        val_loss="0.68",
        training_notes=(
            "Trained 2026-04-12 on the full Falco HuggingFace set (10K files "
            "→ 9110 games after quality filters). v2 shards, relpos, "
            "--self-inputs, 32K steps at batch 512. In bot-vs-bot dittos, "
            "wavedashes on ~25-50% of jumps, rapid-jabs, dash-dances, "
            "and forward-aerials actively."
        ),
    ),
    CharacterEntry(
        name="cptfalcon",
        display="Captain Falcon",
        melee_enum="CPTFALCON",
        checkpoint=REPO_ROOT / "checkpoints/cptfalcon-20260412-relpos-27k.pt",
        data_dir=REPO_ROOT / "data/cptfalcon_v2",
        games_trained=9404,
        val_btn_f1="89.9%",
        val_main_f1="52.2%",
        val_loss="0.71",
        training_notes=(
            "Trained 2026-04-12, same recipe as Falco but on Captain Falcon "
            "replays (10K files → 9404 games). In bot-vs-bot dittos, "
            "wavedashes ~16%, actively dashes, grabs, and does aerials."
        ),
    ),
    CharacterEntry(
        name="luigi",
        display="Luigi",
        melee_enum="LUIGI",
        checkpoint=REPO_ROOT / "checkpoints/luigi-20260412-relpos-5k.pt",
        data_dir=REPO_ROOT / "data/luigi_v2",
        games_trained=1951,
        val_btn_f1="~91%",
        val_main_f1="~60%",
        val_loss="~1.0",
        training_notes=(
            "Trained 2026-04-12 on only 1951 Luigi games (all that was "
            "available on the HuggingFace dataset). The training run was set "
            "to 262K steps but val loss started climbing around step 5K due "
            "to overfitting on the small dataset — the auto-saved _best.pt "
            "is from step 5242, the early-stop sweet spot. Despite the "
            "small dataset, this checkpoint wavedashes at 70-83% conversion "
            "rate in bot-vs-bot dittos, matching real high-level Luigi play."
        ),
    ),
]


MODEL_CARD_TEMPLATE = """---
license: mit
tags:
- behavior-cloning
- imitation-learning
- super-smash-bros-melee
- reinforcement-learning
- gaming
library_name: pytorch
---

# MIMIC: Melee Imitation Model for Input Cloning

Behavior-cloned Super Smash Bros. Melee bots trained on human Slippi replays.
Four character-specific models (Fox, Falco, Captain Falcon, Luigi), each a
~20M-parameter transformer that takes a 256-frame window of game state and
outputs controller inputs (main stick, c-stick, shoulder, buttons) at 60 Hz.

- **Repo**: https://github.com/erickfm/MIMIC
- **Base architecture**: HAL's GPTv5Controller (Eric Gu,
  https://github.com/ericyuegu/hal) — 6-layer causal transformer,
  512 d_model, 8 heads, 256-frame context, relative position encoding
  (Shaw et al.)
- **MIMIC-specific changes**: 7-class button head (distinct TRIG class for
  airdodge/wavedash, which HAL's 5-class head cannot represent); v2 shard
  alignment that fixes a subtle gamestate leak in the training targets
  (see research notes 2026-04-11c); fix for the digital L press bug that
  prevented all 7-class BC bots from wavedashing until 2026-04-13.
- **Training data**: filtered from
  [erickfm/slippi-public-dataset-v3.7](https://huggingface.co/datasets/erickfm/slippi-public-dataset-v3.7)
  (~95K Slippi replays).

## Per-character checkpoints

{metrics_table}

## Repo layout

```
MIMIC/
├── README.md                      # this file
├── fox/
│   ├── model.pt                   # raw PyTorch checkpoint
│   ├── config.json                # ModelConfig (copied from ckpt["config"])
│   ├── metadata.json              # provenance (step, val metrics, notes)
│   ├── mimic_norm.json            # normalization stats
│   ├── controller_combos.json     # 7-class button combo spec
│   ├── cat_maps.json
│   ├── stick_clusters.json
│   └── norm_stats.json
├── falco/      (same layout)
├── cptfalcon/  (same layout)
└── luigi/      (same layout)
```

Each character directory is self-contained — the JSONs are the exact
metadata used during training, copied verbatim from the MIMIC data dir so
any inference script can load them without touching the MIMIC repo.

## Usage

Clone the MIMIC repo and pull this model:

```bash
git clone https://github.com/erickfm/MIMIC.git
cd MIMIC
bash setup.sh  # installs Dolphin, deps, ISO

# Download all four characters
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('erickfm/MIMIC', local_dir='./hf_checkpoints')
"
```

Run a character against a level-9 CPU:

```bash
python3 tools/play_vs_cpu.py \\
  --checkpoint hf_checkpoints/falco/model.pt \\
  --dolphin-path ./emulator/squashfs-root/usr/bin/dolphin-emu \\
  --iso-path ./melee.iso \\
  --data-dir hf_checkpoints/falco \\
  --character FALCO --cpu-character FALCO --cpu-level 9 \\
  --stage FINAL_DESTINATION
```

Or play the bot over Slippi Online Direct Connect:

```bash
python3 tools/play_netplay.py \\
  --checkpoint hf_checkpoints/falco/model.pt \\
  --dolphin-path ./emulator/squashfs-root/usr/bin/dolphin-emu \\
  --iso-path ./melee.iso \\
  --data-dir hf_checkpoints/falco \\
  --character FALCO \\
  --connect-code YOUR#123
```

The MIMIC repo also includes a Discord bot frontend
(`tools/discord_bot.py`) that queues direct-connect matches per user.
See [docs/discord-bot-setup.md](https://github.com/erickfm/MIMIC/blob/main/docs/discord-bot-setup.md).

## Architecture

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

### 7-class button head

| Class | Meaning |
|---|---|
| 0 | A |
| 1 | B |
| 2 | Z |
| 3 | JUMP (X or Y) |
| 4 | TRIG (digital L or R) |
| 5 | A_TRIG (shield grab) |
| 6 | NONE |

HAL's original 5-class head (`A, B, Jump, Z, None`) has no TRIG class and
structurally cannot execute airdodge, which means HAL-lineage bots cannot
wavedash. MIMIC's 7-class encoding plus a fix for `decode_and_press`
(which was silently dropping the digital L press until 2026-04-13) is
what enables the wavedashing you'll see in the replays.

### Input features

9 numeric features per player (ego + opponent = 18 total):
`percent, stock, facing, invulnerable, jumps_left, on_ground,
shield_strength, position_x, position_y`

Plus categorical embeddings: stage(4d), 2× character(12d), 2× action(32d).
Plus controller state from the previous frame as a 56-dim one-hot
(37 stick + 9 c-stick + 7 button + 3 shoulder).

Total input per frame: 166 dimensions → projected to 512.

## Training

- Optimizer: AdamW, LR 3e-4, weight decay 0.01, no warmup
- LR schedule: CosineAnnealingLR, eta_min 1e-6
- Gradient clip: 1.0
- Dropout: 0.2
- Sequence length: 256 frames (~4.3 seconds)
- Mixed precision: BF16 AMP with FP32 upcast for relpos attention
  (prevents BF16 overflow in the manual Q@K^T + Srel computation)
- Batch size: 512 (typically single-GPU on an RTX 5090)
- Steps: ~32K for well-represented characters, early-stopped for Luigi
- Reaction delay: 0 (v2 shards have target[i] = buttons[i+1], so the
  default rd=0 matches inference — do NOT use `--reaction-delay 1` or
  `--controller-offset` with v2 shards)

## Known limitations

1. **Character-locked**: each model only plays the character it was trained
   on. No matchup generalization. Training a multi-character model with a
   character embedding is a natural next step but not done yet.
2. **Fox model is legacy**: the Fox checkpoint is from an earlier run that
   predates the `--self-inputs` fix. Its val metrics are much lower than
   the others and it plays slightly worse.
3. **Small-dataset overfitting**: Luigi only has 1951 training games after
   filtering. The `_best.pt` checkpoint is early-stopped at step 5242 to
   avoid the val-loss climb. Plays surprisingly well for the data volume.
4. **Edge guarding and recovery weaknesses**: the bot doesn't consistently
   go for off-stage edge guards or execute high-skill recovery mixups.
5. **No matchmaking / Ranked**: the Discord bot only joins explicit Direct
   Connect lobbies. Do NOT adapt it for Slippi Online Unranked or Ranked —
   the libmelee README explicitly forbids bots on those ladders, and
   Slippi has not yet opened a "bot account" opt-in system.

## Acknowledgments

- **Eric Gu** for HAL, the reference implementation MIMIC is based on.
  HAL's architecture, tokenization, and training pipeline are the
  foundation. https://github.com/ericyuegu/hal
- **Vlad Firoiu and collaborators** for libmelee, the Python interface
  to Dolphin + Slippi. https://github.com/altf4/libmelee
- **Project Slippi** for the Slippi Dolphin fork, replay format, and
  Direct Connect rollback netplay. https://slippi.gg

## License

MIT — see the MIMIC repo's LICENSE file.
"""


def build_metrics_table(characters: list[CharacterEntry]) -> str:
    header = "| Character | Games | Val btn F1 | Val main F1 | Val loss | Step |"
    sep    = "|---|---|---|---|---|---|"
    rows = []
    for c in characters:
        ckpt = torch.load(c.checkpoint, map_location="cpu", weights_only=False)
        step = ckpt.get("global_step", "?")
        rows.append(
            f"| **{c.display}** | {c.games_trained:,} | "
            f"{c.val_btn_f1} | {c.val_main_f1} | {c.val_loss} | {step:,} |"
        )
    return "\n".join([header, sep, *rows])


def stage_character(entry: CharacterEntry, out_dir: Path) -> None:
    """Assemble one character directory in the staging area."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy the checkpoint verbatim
    dest_ckpt = out_dir / "model.pt"
    print(f"  copying {entry.checkpoint.name} → {dest_ckpt.relative_to(REPO_ROOT)}")
    shutil.copy2(entry.checkpoint, dest_ckpt)

    # 2. Write config.json from ckpt["config"]
    ckpt = torch.load(entry.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    # Ensure everything is JSON-serializable
    def _jsonable(v):
        if isinstance(v, (str, int, float, bool, type(None))):
            return v
        if isinstance(v, (list, tuple)):
            return [_jsonable(x) for x in v]
        if isinstance(v, dict):
            return {k: _jsonable(x) for k, x in v.items()}
        return str(v)
    config_clean = {k: _jsonable(v) for k, v in config.items()}
    (out_dir / "config.json").write_text(json.dumps(config_clean, indent=2))

    # 3. Copy metadata JSONs from the training data dir
    for fname in ("mimic_norm.json", "controller_combos.json", "cat_maps.json",
                  "stick_clusters.json", "norm_stats.json"):
        src = entry.data_dir / fname
        if src.exists():
            shutil.copy2(src, out_dir / fname)
        else:
            print(f"  ⚠ missing {src} — skipping")

    # 4. Write metadata.json with provenance
    meta = {
        "character": entry.display,
        "melee_enum": entry.melee_enum,
        "run_name": config.get("run_name", "?"),
        "global_step": ckpt.get("global_step", "?"),
        "games_trained": entry.games_trained,
        "val_btn_f1": entry.val_btn_f1,
        "val_main_f1": entry.val_main_f1,
        "val_loss": entry.val_loss,
        "n_params": sum(v.numel() for v in ckpt["model_state_dict"].values()),
        "n_controller_combos": config.get("n_controller_combos", 7),
        "model_preset": config.get("model_preset", "hal"),
        "pos_enc": config.get("pos_enc", "relpos"),
        "no_self_inputs": config.get("no_self_inputs", False),
        "training_notes": entry.training_notes,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"  step={meta['global_step']} val_btn_f1={entry.val_btn_f1}")


def build_staging(characters: list[CharacterEntry]) -> Path:
    if STAGING_DIR.exists():
        print(f"Removing existing staging dir {STAGING_DIR}")
        shutil.rmtree(STAGING_DIR)
    STAGING_DIR.mkdir()

    # Build model card
    print("Building README.md...")
    metrics_table = build_metrics_table(characters)
    readme = MODEL_CARD_TEMPLATE.format(metrics_table=metrics_table)
    (STAGING_DIR / "README.md").write_text(readme)

    # Stage each character
    for entry in characters:
        print(f"\nStaging {entry.display}...")
        if not entry.checkpoint.exists():
            print(f"  ⚠ checkpoint missing at {entry.checkpoint} — skipping")
            continue
        if not entry.data_dir.exists():
            print(f"  ⚠ data dir missing at {entry.data_dir} — skipping")
            continue
        stage_character(entry, STAGING_DIR / entry.name)

    # Report total size
    total_size = sum(f.stat().st_size for f in STAGING_DIR.rglob("*") if f.is_file())
    print(f"\nStaging complete: {STAGING_DIR} ({total_size / 1e9:.2f} GB)")
    return STAGING_DIR


def push_to_hf(staging: Path, repo_id: str) -> None:
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("ERROR: huggingface_hub not installed. Install with:")
        print("  pip install huggingface_hub")
        sys.exit(1)

    api = HfApi()
    print(f"\nCreating/updating HF repo: {repo_id}")
    create_repo(repo_id, repo_type="model", exist_ok=True)

    print(f"Uploading {staging} → {repo_id} ...")
    api.upload_folder(
        folder_path=str(staging),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload MIMIC character checkpoints + model card",
    )
    print(f"\n✅ Pushed to https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload MIMIC models to HuggingFace Hub",
        epilog="By default this script is a dry run and never overwrites HF. "
               "Use --only <char>[,<char>...] to push specific characters. "
               "Use --all to rebuild and push every character (destructive — "
               "overwrites existing model.pt + metadata for untouched chars).")
    parser.add_argument("--push", action="store_true",
                        help="Actually push to HF (default: build staging dir only)")
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated character names to stage/push "
                             "(e.g. 'fox' or 'fox,falco'). Other characters "
                             "are untouched on HF.")
    parser.add_argument("--all", action="store_true",
                        help="Stage and push ALL characters. Required to "
                             "overwrite characters that already exist on HF.")
    parser.add_argument("--repo", default="erickfm/MIMIC",
                        help="HuggingFace repo id (default: erickfm/MIMIC)")
    parser.add_argument("--keep-staging", action="store_true",
                        help="Don't delete _hf_staging/ after push")
    args = parser.parse_args()

    # Resolve which characters to stage
    if args.only:
        wanted = {c.strip().lower() for c in args.only.split(",")}
        entries = [c for c in CHARACTERS if c.name in wanted]
        missing = wanted - {c.name for c in entries}
        if missing:
            print(f"ERROR: unknown character name(s) in --only: {sorted(missing)}")
            print(f"  available: {[c.name for c in CHARACTERS]}")
            sys.exit(1)
    elif args.all:
        entries = CHARACTERS
    else:
        print("ERROR: must pass --only <char>[,...] or --all")
        print(f"  available characters: {[c.name for c in CHARACTERS]}")
        print("  --only falco             # stage/push just Falco")
        print("  --only fox,falco         # stage/push Fox + Falco")
        print("  --all                    # stage/push every character (destructive)")
        sys.exit(2)

    print(f"Staging characters: {[c.name for c in entries]}")
    staging = build_staging(entries)

    if args.push:
        # Safety net: refuse to --all without a second confirmation flag if
        # anyone forgets just how destructive that is.
        if args.all and not args.only:
            print("\n⚠ --all will OVERWRITE every character on "
                  f"https://huggingface.co/{args.repo}")
            print("   Press Ctrl-C in the next 3 seconds to abort ...")
            import time
            try:
                time.sleep(3)
            except KeyboardInterrupt:
                print("Aborted.")
                sys.exit(0)
        push_to_hf(staging, args.repo)
        if not args.keep_staging:
            shutil.rmtree(staging)
            print(f"Removed staging dir {staging}")
    else:
        print(f"\nDry run complete. Inspect {staging} then re-run with --push to upload.")


if __name__ == "__main__":
    main()
