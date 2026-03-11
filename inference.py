#!/usr/bin/env python3
# inference.py  –  FRAME bot
#
# Converts live Slippi frames → model tensors via features.py,
# runs FramePredictor on a rolling window,
# converts model output back to Dolphin controller actions.
# ---------------------------------------------------------------------------

import argparse
import logging
import math
import os
import signal
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, List

import melee
import numpy as np
import pandas as pd
import torch

import features as F
from model import FramePredictor, ModelConfig

# ── CLI / logging ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Realtime FRAME bot")
parser.add_argument("--debug", action="store_true", help="Verbose sanity checks")
parser.add_argument("--dolphin-path", type=str,
                    default=os.getenv("DOLPHIN_PATH", ""),
                    help="Path to Slippi Dolphin app")
parser.add_argument("--iso-path", type=str,
                    default=os.getenv("ISO_PATH", ""),
                    help="Path to Melee ISO")
args = parser.parse_args()
DEBUG = args.debug or bool(os.getenv("DEBUG", ""))

logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s"
)
log = logging.getLogger(__name__)
torch.set_printoptions(sci_mode=False, precision=4)

# ── Device + checkpoint ───────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info("Device: %s", DEVICE)

ckpt_patterns = ["step_*.pt", "epoch_*.pt"]
ckpts: list[Path] = []
for pat in ckpt_patterns:
    ckpts.extend(Path("./checkpoints").glob(pat))
if not ckpts:
    log.error("No checkpoints found."); sys.exit(1)
ckpt_path = max(ckpts, key=lambda p: p.stat().st_mtime)
ckpt      = torch.load(ckpt_path, map_location=DEVICE)
cfg       = ModelConfig(**ckpt["config"])

model = FramePredictor(cfg).to(DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
log.info("Loaded checkpoint %s", ckpt_path)

# ── Normalization stats + categorical maps ───────────────────────────────
import json

_SEARCH_DIRS = [
    Path("./data"), Path("./data/subset"), Path("./data/full"),
]

norm_stats: Dict[str, tuple] = ckpt.get("norm_stats", {})
if not norm_stats:
    for p in _SEARCH_DIRS:
        ns = p / "norm_stats.json"
        if ns.exists():
            with open(ns) as fh:
                norm_stats = json.load(fh)
            break
if norm_stats:
    log.info("Loaded normalization stats for %d columns", len(norm_stats))
else:
    log.warning("No normalization stats found")

cat_maps: Dict[str, Dict[int, int]] = {}
for p in _SEARCH_DIRS:
    cm = p / "cat_maps.json"
    if cm.exists():
        with open(cm) as fh:
            raw = json.load(fh)
            cat_maps = {col: {int(k): v for k, v in m.items()} for col, m in raw.items()}
        log.info("Loaded categorical maps for %d columns", len(cat_maps))
        break
if not cat_maps:
    log.warning("No cat_maps.json found -- dynamic categoricals will map to 0")

ROLL_WIN = cfg.max_seq_len

# ── Feature spec ─────────────────────────────────────────────────────────
_fg = F.build_feature_groups()
_categorical_cols = F.get_categorical_cols(_fg)

# ── Debug helpers ────────────────────────────────────────────────────────
def check_tensor_dict(tdict: Dict[str, torch.Tensor], where: str) -> None:
    if not DEBUG:
        return
    for k, v in tdict.items():
        if torch.isnan(v).any() or torch.isinf(v).any():
            log.warning("NaN/Inf detected in %s -> %s", where, k)

# ── DataFrame conversion ────────────────────────────────────────────────
def rows_to_state_seq(rows: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Convert rolling list of row dicts -> tensor batch (B=1)."""
    df = pd.DataFrame(rows)

    df = F.preprocess_df(df, _categorical_cols, cat_maps)

    F.apply_normalization(df, norm_stats)

    missing_cats = [c for c in _categorical_cols if c not in df.columns]
    if missing_cats:
        df = pd.concat([df, pd.DataFrame({c: 0 for c in missing_cats},
                                         index=df.index)], axis=1)

    numeric_missing = {}
    for _, meta in F.walk_groups(_fg, return_meta=True):
        if meta["ftype"] != "categorical":
            for col in meta["cols"]:
                if col not in df.columns:
                    numeric_missing[col] = 0.0
    if numeric_missing:
        df = pd.concat([df, pd.DataFrame(numeric_missing, index=df.index)], axis=1)

    if DEBUG and df.isna().any().any():
        bad = df.columns[df.isna().any()].tolist()
        log.warning("DataFrame still has NaNs in cols: %s", bad)

    state_seq = F.df_to_state_tensors(df, _fg)
    state_seq = {k: v.unsqueeze(0) for k, v in state_seq.items()}

    if DEBUG:
        check_tensor_dict(state_seq, "state_seq")
    return state_seq

# ── Inference wrapper ────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(win_rows: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    batch = rows_to_state_seq(win_rows)
    for k, v in batch.items():
        batch[k] = v.to(DEVICE, non_blocking=True)

    check_tensor_dict(batch, "batch_before_model")
    preds = model(batch)
    preds = {k: v[:, -1] for k, v in preds.items()}
    check_tensor_dict(preds, "model_output")

    return {k: v.cpu().squeeze(0) for k, v in preds.items()}

# ── Controller output ────────────────────────────────────────────────────
C_DIR_TO_FLOAT = {
    0: (0.5, 0.5),
    1: (0.5, 1.0),
    2: (0.5, 0.0),
    3: (0.0, 0.5),
    4: (1.0, 0.5),
}

IDX_TO_BUTTON = [
    melee.enums.Button.BUTTON_A, melee.enums.Button.BUTTON_B,
    melee.enums.Button.BUTTON_X, melee.enums.Button.BUTTON_Y,
    melee.enums.Button.BUTTON_Z, melee.enums.Button.BUTTON_L,
    melee.enums.Button.BUTTON_R, melee.enums.Button.BUTTON_START,
    melee.enums.Button.BUTTON_D_UP, melee.enums.Button.BUTTON_D_DOWN,
    melee.enums.Button.BUTTON_D_LEFT, melee.enums.Button.BUTTON_D_RIGHT,
]

def _safe(val: float, default: float = 0.5) -> float:
    if not math.isfinite(val):
        return default
    return min(max(val, 0.0), 1.0)


def press_output(ctrl: melee.Controller,
                 pred: Dict[str, torch.Tensor],
                 thresh: float = 0.5):
    clamped_main = torch.clamp(pred["main_xy"], 0.0, 1.0)
    mx, my = map(float, clamped_main.tolist())

    dir_idx = int(torch.argmax(pred["c_dir_logits"]))
    cx, cy  = C_DIR_TO_FLOAT.get(dir_idx, (0.5, 0.5))

    l_val = _safe(torch.clamp(pred["L_val"], 0.0, 1.0).item(), 0.0)
    r_val = _safe(torch.clamp(pred["R_val"], 0.0, 1.0).item(), 0.0)

    ctrl.tilt_analog(melee.enums.Button.BUTTON_MAIN, mx, my)
    ctrl.tilt_analog(melee.enums.Button.BUTTON_C,    cx, cy)
    ctrl.press_shoulder(melee.enums.Button.BUTTON_L, l_val)
    ctrl.press_shoulder(melee.enums.Button.BUTTON_R, r_val)

    btn_probs = torch.sigmoid(pred["btn_logits"])
    pressed = []
    for prob, btn in zip(btn_probs, IDX_TO_BUTTON):
        if prob.item() > thresh:
            ctrl.press_button(btn)
            pressed.append(btn.name)
        else:
            ctrl.release_button(btn)

    log.info(
        "MAIN=(%.2f,%.2f) C=(%.2f,%.2f) L=%.2f R=%.2f BUTTONS=%s",
        mx, my, cx, cy, l_val, r_val, pressed
    )

# ── Dolphin loop ─────────────────────────────────────────────────────────
def signal_handler(sig, _):
    for c in controllers.values():
        c.disconnect()
    console.stop()
    log.info("Shutting down...")
    sys.exit(0)

if __name__ == "__main__":
    DOLPHIN_APP = args.dolphin_path
    ISO_PATH    = args.iso_path

    if not DOLPHIN_APP or not ISO_PATH:
        log.error("Must provide --dolphin-path and --iso-path (or set DOLPHIN_PATH / ISO_PATH env vars)")
        sys.exit(1)

    console = melee.Console(path=DOLPHIN_APP, slippi_address="127.0.0.1", fullscreen=False)
    ports = [1, 2]
    controllers = {p: melee.Controller(console, p) for p in ports}

    signal.signal(signal.SIGINT, signal_handler)
    console.run(iso_path=ISO_PATH)

    if not console.connect():
        log.error("Console connect failed"); sys.exit(1)
    for c in controllers.values():
        if not c.connect():
            log.error("Controller connect failed"); sys.exit(1)
    log.info("Console + controllers connected.")

    rows: deque[Dict[str, Any]] = deque(maxlen=ROLL_WIN)
    while True:
        gs = console.step()
        if gs is None:
            continue

        if gs.menu_state not in (melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH):
            c0, c1 = controllers[ports[0]], controllers[ports[1]]
            melee.MenuHelper().menu_helper_simple(
                gs, c0, melee.Character.FALCO, melee.Stage.FINAL_DESTINATION,
                cpu_level=0, autostart=0
            )
            melee.MenuHelper().menu_helper_simple(
                gs, c1, melee.Character.FALCO, melee.Stage.FINAL_DESTINATION,
                cpu_level=0, autostart=1
            )
            continue

        # -- build row (matches extract.py schema) --
        row: Dict[str, Any] = {}
        row["stage"]    = gs.stage.value if gs.stage else -1
        row["frame"]    = gs.frame
        row["distance"] = gs.distance
        row["startAt"]  = gs.startAt

        for idx, (port, ps) in enumerate(gs.players.items()):
            pref = "self_" if idx == 0 else "opp_"
            row[f"{pref}port"]          = port
            row[f"{pref}character"]     = ps.character.value
            row[f"{pref}action"]        = ps.action.value
            row[f"{pref}action_frame"]  = ps.action_frame
            row[f"{pref}costume"]       = ps.costume

            for btn, st in ps.controller_state.button.items():
                row[f"{pref}btn_{btn.name}"] = int(st)

            row[f"{pref}main_x"], row[f"{pref}main_y"] = ps.controller_state.main_stick
            row[f"{pref}c_x"],   row[f"{pref}c_y"]    = ps.controller_state.c_stick
            row[f"{pref}l_shldr"] = ps.controller_state.l_shoulder
            row[f"{pref}r_shldr"] = ps.controller_state.r_shoulder

            row[f"{pref}percent"]              = float(ps.percent)
            row[f"{pref}pos_x"]                = float(ps.position.x)
            row[f"{pref}pos_y"]                = float(ps.position.y)
            row[f"{pref}stock"]                = ps.stock
            row[f"{pref}facing"]               = ps.facing
            row[f"{pref}on_ground"]            = ps.on_ground
            row[f"{pref}off_stage"]            = ps.off_stage
            row[f"{pref}invulnerable"]         = ps.invulnerable
            row[f"{pref}moonwalkwarning"]      = ps.moonwalkwarning
            row[f"{pref}shield_strength"]      = float(ps.shield_strength)
            row[f"{pref}jumps_left"]           = ps.jumps_left
            row[f"{pref}hitlag_left"]          = ps.hitlag_left
            row[f"{pref}hitstun_left"]         = ps.hitstun_frames_left
            row[f"{pref}invuln_left"]          = ps.invulnerability_left
            row[f"{pref}speed_air_x_self"]     = float(ps.speed_air_x_self)
            row[f"{pref}speed_ground_x_self"]  = float(ps.speed_ground_x_self)
            row[f"{pref}speed_x_attack"]       = float(ps.speed_x_attack)
            row[f"{pref}speed_y_attack"]       = float(ps.speed_y_attack)
            row[f"{pref}speed_y_self"]         = float(ps.speed_y_self)
            row[f"{pref}ecb_bottom_x"]         = float(ps.ecb_bottom[0])
            row[f"{pref}ecb_bottom_y"]         = float(ps.ecb_bottom[1])
            row[f"{pref}ecb_left_x"]           = float(ps.ecb_left[0])
            row[f"{pref}ecb_left_y"]           = float(ps.ecb_left[1])
            row[f"{pref}ecb_right_x"]          = float(ps.ecb_right[0])
            row[f"{pref}ecb_right_y"]          = float(ps.ecb_right[1])
            row[f"{pref}ecb_top_x"]            = float(ps.ecb_top[0])
            row[f"{pref}ecb_top_y"]            = float(ps.ecb_top[1])

            nana = ps.nana
            npref = f"{pref}nana_"
            if nana:
                row[f"{npref}character"]           = nana.character.value
                row[f"{npref}action"]              = nana.action.value
                row[f"{npref}action_frame"]        = nana.action_frame
                for btn, st in nana.controller_state.button.items():
                    row[f"{npref}btn_{btn.name}"] = int(st)
                row[f"{npref}main_x"], row[f"{npref}main_y"] = nana.controller_state.main_stick
                row[f"{npref}c_x"],    row[f"{npref}c_y"]    = nana.controller_state.c_stick
                row[f"{npref}l_shldr"] = nana.controller_state.l_shoulder
                row[f"{npref}r_shldr"] = nana.controller_state.r_shoulder
                row[f"{npref}percent"]             = float(nana.percent)
                row[f"{npref}pos_x"]               = float(nana.position.x)
                row[f"{npref}pos_y"]               = float(nana.position.y)
                row[f"{npref}stock"]               = nana.stock
                row[f"{npref}facing"]              = nana.facing
                row[f"{npref}on_ground"]           = nana.on_ground
                row[f"{npref}off_stage"]           = nana.off_stage
                row[f"{npref}invulnerable"]        = nana.invulnerable
                row[f"{npref}moonwalkwarning"]     = nana.moonwalkwarning
                row[f"{npref}shield_strength"]     = float(nana.shield_strength)
                row[f"{npref}jumps_left"]          = nana.jumps_left
                row[f"{npref}hitlag_left"]         = nana.hitlag_left
                row[f"{npref}hitstun_left"]        = nana.hitstun_frames_left
                row[f"{npref}invuln_left"]         = nana.invulnerability_left
                row[f"{npref}speed_air_x_self"]    = float(nana.speed_air_x_self)
                row[f"{npref}speed_ground_x_self"] = float(nana.speed_ground_x_self)
                row[f"{npref}speed_x_attack"]      = float(nana.speed_x_attack)
                row[f"{npref}speed_y_attack"]      = float(nana.speed_y_attack)
                row[f"{npref}speed_y_self"]        = float(nana.speed_y_self)
                row[f"{npref}ecb_bottom_x"]        = float(nana.ecb_bottom[0])
                row[f"{npref}ecb_bottom_y"]        = float(nana.ecb_bottom[1])
                row[f"{npref}ecb_left_x"]          = float(nana.ecb_left[0])
                row[f"{npref}ecb_left_y"]          = float(nana.ecb_left[1])
                row[f"{npref}ecb_right_x"]         = float(nana.ecb_right[0])
                row[f"{npref}ecb_right_y"]         = float(nana.ecb_right[1])
                row[f"{npref}ecb_top_x"]           = float(nana.ecb_top[0])
                row[f"{npref}ecb_top_y"]           = float(nana.ecb_top[1])
            else:
                row[f"{npref}character"]    = -1
                row[f"{npref}action"]       = -1
                row[f"{npref}action_frame"] = -1
                for b in F.BTN:
                    row[f"{npref}btn_{b}"] = 0
                row[f"{npref}main_x"] = 0.5
                row[f"{npref}main_y"] = 0.5
                row[f"{npref}c_x"]    = 0.5
                row[f"{npref}c_y"]    = 0.5
                row[f"{npref}l_shldr"] = 0.0
                row[f"{npref}r_shldr"] = 0.0
                for field in ("percent", "pos_x", "pos_y", "stock",
                              "facing", "on_ground", "off_stage",
                              "invulnerable", "moonwalkwarning",
                              "shield_strength", "jumps_left",
                              "hitlag_left", "hitstun_left", "invuln_left",
                              "speed_air_x_self", "speed_ground_x_self",
                              "speed_x_attack", "speed_y_attack",
                              "speed_y_self"):
                    row[f"{npref}{field}"] = 0.0
                for part in ("bottom", "left", "right", "top"):
                    for axis in ("x", "y"):
                        row[f"{npref}ecb_{part}_{axis}"] = 0.0

        for j in range(F.PROJ_SLOTS):
            pp = f"proj{j}_"
            if j < len(gs.projectiles):
                p = gs.projectiles[j]
                row[f"{pp}owner"]   = p.owner
                row[f"{pp}type"]    = p.type.value
                row[f"{pp}subtype"] = p.subtype
                row[f"{pp}pos_x"]   = float(p.position.x)
                row[f"{pp}pos_y"]   = float(p.position.y)
                row[f"{pp}speed_x"] = float(p.speed.x)
                row[f"{pp}speed_y"] = float(p.speed.y)
                row[f"{pp}frame"]   = p.frame
            else:
                row[f"{pp}owner"]   = -1
                row[f"{pp}type"]    = -1
                row[f"{pp}subtype"] = -1
                row[f"{pp}pos_x"]   = np.nan
                row[f"{pp}pos_y"]   = np.nan
                row[f"{pp}speed_x"] = np.nan
                row[f"{pp}speed_y"] = np.nan
                row[f"{pp}frame"]   = -1

        rows.append(row)
        if len(rows) == ROLL_WIN:
            pred = run_inference(list(rows))
            ctrl = controllers[ports[0]]
            ctrl.release_all()
            press_output(ctrl, pred)
            ctrl.flush()
