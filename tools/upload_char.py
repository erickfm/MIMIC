#!/usr/bin/env python3
"""Upload ONE character's checkpoint to erickfm/MIMIC/<char>/ on HuggingFace.

Pulls the bestloss checkpoint + metadata JSONs, writes config.json + metadata.json
from the checkpoint + training log, and uploads the single-character subfolder
via api.upload_folder (other characters on the repo are untouched).

Usage:
    python3 tools/upload_char.py --char marth \\
        --checkpoint checkpoints/marth-20260417-relpos_bestloss.pt \\
        --data-dir data/marth_v2 \\
        --log checkpoints/marth-20260417-relpos.log
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_ID = "erickfm/MIMIC"

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass


CHAR_DISPLAY = {
    "fox": ("Fox", "FOX"),
    "falco": ("Falco", "FALCO"),
    "marth": ("Marth", "MARTH"),
    "sheik": ("Sheik", "SHEIK"),
    "cptfalcon": ("Captain Falcon", "CPTFALCON"),
    "puff": ("Jigglypuff", "JIGGLYPUFF"),
    "luigi": ("Luigi", "LUIGI"),
    "peach": ("Peach", "PEACH"),
    "ice_climbers": ("Ice Climbers", "POPO"),
}


def _best_val_metrics(log_path: Path) -> dict:
    """Pull val metrics at the best-val-loss step from the training log."""
    if not log_path.exists():
        return {}
    text = log_path.read_text(errors="ignore")
    m = re.search(r"Best val_loss=([0-9.]+) @ step (\d+)", text)
    best_loss = float(m.group(1)) if m else None
    best_step = int(m.group(2)) if m else None

    best_val_line = None
    lowest = float("inf")
    for line in re.findall(r"-- val total=[^\n]+", text):
        vm = re.search(r"val total=([0-9.]+)", line)
        if vm:
            v = float(vm.group(1))
            if v < lowest:
                lowest = v
                best_val_line = line
    metrics = {"val_loss": f"{best_loss:.4f}" if best_loss else "?",
               "best_step": best_step}
    if best_val_line:
        for key, pat in (("val_btn_f1", r"bf1=([0-9.]+%)"),
                         ("val_main_f1", r"mf1=([0-9.]+%)"),
                         ("val_shldr_f1", r"sf1=([0-9.]+%)"),
                         ("val_cdir_f1", r"cf1=([0-9.]+%)")):
            m2 = re.search(pat, best_val_line)
            if m2:
                metrics[key] = m2.group(1)
    return metrics


def stage(char: str, checkpoint: Path, data_dir: Path, log_path: Path, out_dir: Path):
    display, melee_enum = CHAR_DISPLAY[char]
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  copy {checkpoint.name} → {out_dir.name}/model.pt")
    shutil.copy2(checkpoint, out_dir / "model.pt")

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {}) or {}

    def _jsonable(v):
        if isinstance(v, (str, int, float, bool, type(None))):
            return v
        if isinstance(v, (list, tuple)):
            return [_jsonable(x) for x in v]
        if isinstance(v, dict):
            return {k: _jsonable(x) for k, x in v.items()}
        return str(v)
    (out_dir / "config.json").write_text(
        json.dumps({k: _jsonable(v) for k, v in cfg.items()}, indent=2))

    for fname in ("mimic_norm.json", "controller_combos.json", "cat_maps.json",
                  "stick_clusters.json", "norm_stats.json"):
        src = data_dir / fname
        if src.exists():
            shutil.copy2(src, out_dir / fname)
        else:
            print(f"  ⚠ missing {src}")

    val = _best_val_metrics(log_path)
    meta = {
        "character": display,
        "melee_enum": melee_enum,
        "run_name": cfg.get("run_name", checkpoint.stem),
        "global_step": ckpt.get("global_step", "?"),
        "n_params": sum(v.numel() for v in ckpt["model_state_dict"].values()),
        "n_controller_combos": cfg.get("n_controller_combos", 7),
        "model_preset": cfg.get("model_preset", "mimic"),
        "pos_enc": cfg.get("pos_enc", "relpos"),
        **val,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"  step={meta['global_step']} val_loss={meta.get('val_loss')}"
          f" btn_f1={meta.get('val_btn_f1', '?')}")


def push(staging: Path, char: str):
    from huggingface_hub import HfApi, create_repo
    api = HfApi()
    create_repo(REPO_ID, repo_type="model", exist_ok=True)
    print(f"  uploading → huggingface.co/{REPO_ID}/{char}/")
    api.upload_folder(
        folder_path=str(staging),
        repo_id=REPO_ID,
        repo_type="model",
        path_in_repo=f"{char}/",
        commit_message=f"Update {char} checkpoint + metadata",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--char", required=True, choices=list(CHAR_DISPLAY.keys()))
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()

    if not args.checkpoint.exists():
        print(f"ERROR: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)

    staging = REPO_ROOT / "_hf_staging_single" / args.char
    if staging.exists():
        shutil.rmtree(staging)
    stage(args.char, args.checkpoint, args.data_dir, args.log, staging)

    if args.no_push:
        print(f"  dry run: staged at {staging}")
        return
    push(staging, args.char)
    shutil.rmtree(staging.parent, ignore_errors=True)
    print(f"  done uploading {args.char}")


if __name__ == "__main__":
    main()
