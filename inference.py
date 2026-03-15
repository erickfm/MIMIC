#!/usr/bin/env python3
# inference.py  –  MIMIC bot
#
# Converts live Slippi frames → model tensors via features.py,
# runs FramePredictor on a rolling window,
# converts model output back to Dolphin controller actions.
# ---------------------------------------------------------------------------

import argparse
import csv
import logging
import math
import os
import signal
import sys
import time
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
parser = argparse.ArgumentParser(description="Realtime MIMIC bot")
parser.add_argument("--debug", action="store_true", help="Verbose sanity checks")
parser.add_argument("--dolphin-path", type=str,
                    default=os.getenv("DOLPHIN_PATH", ""),
                    help="Path to Slippi Dolphin app")
parser.add_argument("--iso-path", type=str,
                    default=os.getenv("ISO_PATH", ""),
                    help="Path to Melee ISO")
parser.add_argument("--cpu-level", type=int, default=7,
                    help="CPU level for opponent (0 = human/bot, 1-9 = CPU)")
parser.add_argument("--character", type=str, default="FALCO",
                    help="Bot character (e.g. FALCO, FOX, MARTH)")
parser.add_argument("--cpu-character", type=str, default="FALCO",
                    help="CPU opponent character")
parser.add_argument("--stage", type=str, default="FINAL_DESTINATION",
                    help="Stage name (e.g. FINAL_DESTINATION, BATTLEFIELD)")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Path to a specific checkpoint file (overrides auto-discovery)")
parser.add_argument("--log-dir", type=str, default=None,
                    help="Directory for inference CSV logs (default: logs/)")
args = parser.parse_args()

BOT_CHARACTER = melee.Character[args.character.upper()]
CPU_CHARACTER = melee.Character[args.cpu_character.upper()]
STAGE = melee.Stage[args.stage.upper()]
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

if args.checkpoint:
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        log.error("Checkpoint not found: %s", ckpt_path); sys.exit(1)
else:
    ckpts: list[Path] = list(Path("./checkpoints").glob("*.pt"))
    if not ckpts:
        log.error("No checkpoints found in ./checkpoints/"); sys.exit(1)
    ckpt_path = max(ckpts, key=lambda p: p.stat().st_mtime)
    log.info("Auto-discovered %d checkpoints, using newest: %s", len(ckpts), ckpt_path.name)
ckpt      = torch.load(ckpt_path, map_location=DEVICE)
cfg       = ModelConfig(**ckpt["config"])

model = FramePredictor(cfg).to(DEVICE)
state_dict = ckpt["model_state_dict"]
state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.eval()
log.info("Loaded checkpoint %s", ckpt_path)

# ── Cluster centers (for discrete stick_loss="clusters" mode) ────────────
_stick_centers: np.ndarray | None = None
_shoulder_centers: np.ndarray | None = None

if cfg.stick_loss == "clusters":
    if "stick_centers" in ckpt:
        _stick_centers = np.array(ckpt["stick_centers"], dtype=np.float32)
        _shoulder_centers = np.array(ckpt["shoulder_centers"], dtype=np.float32)
        log.info("Loaded cluster centers from checkpoint: %d stick, %d shoulder",
                 len(_stick_centers), len(_shoulder_centers))
    else:
        import json as _json
        for _sd in [Path("./data/full"), Path("./data"), Path("./data/subset")]:
            _sc_path = _sd / "stick_clusters.json"
            if _sc_path.exists():
                with open(_sc_path) as _fh:
                    _sc_raw = _json.load(_fh)
                _stick_centers = np.array(_sc_raw["stick_centers"], dtype=np.float32)
                _shoulder_centers = np.array(_sc_raw["shoulder_centers"], dtype=np.float32)
                log.info("Loaded cluster centers from %s: %d stick, %d shoulder",
                         _sc_path, len(_stick_centers), len(_shoulder_centers))
                break
        if _stick_centers is None:
            log.error("stick_loss='clusters' but no cluster centers in checkpoint or data dir")
            sys.exit(1)

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
_fg = F.build_feature_groups(no_opp_inputs=cfg.no_opp_inputs)
_categorical_cols = F.get_categorical_cols(_fg)

# ── Prediction feedback state ─────────────────────────────────────────────
from typing import Optional
_prev_pred: Optional[Dict[str, torch.Tensor]] = None
_prev_btns_fired: Optional[List[bool]] = None

# ── Inference logger ──────────────────────────────────────────────────────
class InferenceLogger:
    """Per-frame CSV logger capturing game state, raw model outputs, and
    the controller commands actually sent to Dolphin."""

    _HEADER = [
        "frame", "wall_ms",
        # game state
        "self_pos_x", "self_pos_y", "self_action", "self_action_frame",
        "self_percent", "self_stock", "self_facing", "self_on_ground", "self_off_stage",
        "opp_pos_x", "opp_pos_y", "opp_action", "opp_percent", "opp_stock",
        "distance",
        # raw model predictions
        "pred_main_x", "pred_main_y",
        "pred_c_dir", "pred_c_dir_probs",
        "pred_L", "pred_R",
        "pred_btn_probs",
        # actual controller output
        "sent_main_x", "sent_main_y",
        "sent_c_dir",
        "sent_L", "sent_R",
        "sent_btns",
    ]

    def __init__(self, log_dir: str | Path | None = None):
        log_dir = Path(log_dir or "logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self._path = log_dir / f"inference_{stamp}.csv"
        self._fh = open(self._path, "w", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=self._HEADER,
                                      extrasaction="ignore")
        self._writer.writeheader()
        self._t0 = time.monotonic()
        log.info("Inference log → %s", self._path)

    def log_frame(
        self,
        row: Dict[str, Any],
        pred: Dict[str, torch.Tensor] | None,
        sent_main: tuple[float, float] | None,
        sent_c_dir: int | None,
        sent_L: float | None,
        sent_R: float | None,
        sent_btns: List[bool] | None,
    ) -> None:
        rec: Dict[str, Any] = {
            "frame": row.get("frame", -1),
            "wall_ms": f"{(time.monotonic() - self._t0) * 1000:.1f}",
            "self_pos_x":      f"{row.get('self_pos_x', 0):.2f}",
            "self_pos_y":      f"{row.get('self_pos_y', 0):.2f}",
            "self_action":     row.get("self_action", ""),
            "self_action_frame": row.get("self_action_frame", ""),
            "self_percent":    row.get("self_percent", ""),
            "self_stock":      row.get("self_stock", ""),
            "self_facing":     row.get("self_facing", ""),
            "self_on_ground":  row.get("self_on_ground", ""),
            "self_off_stage":  row.get("self_off_stage", ""),
            "opp_pos_x":      f"{row.get('opp_pos_x', 0):.2f}",
            "opp_pos_y":      f"{row.get('opp_pos_y', 0):.2f}",
            "opp_action":     row.get("opp_action", ""),
            "opp_percent":    row.get("opp_percent", ""),
            "opp_stock":      row.get("opp_stock", ""),
            "distance":       f"{row.get('distance', 0):.2f}",
        }
        if pred is not None:
            if cfg.stick_loss == "clusters":
                _lg_mx, _lg_my, _lg_l, _lg_r = _decode_clusters(pred)
                rec["pred_main_x"] = f"{_lg_mx:.4f}"
                rec["pred_main_y"] = f"{_lg_my:.4f}"
                rec["pred_L"] = f"{_lg_l:.4f}"
                rec["pred_R"] = f"{_lg_r:.4f}"
            else:
                main = torch.clamp(pred["main_xy"], 0, 1)
                rec["pred_main_x"] = f"{main[0].item():.4f}"
                rec["pred_main_y"] = f"{main[1].item():.4f}"
                rec["pred_L"] = f"{torch.clamp(pred['L_val'], 0, 1).item():.4f}"
                rec["pred_R"] = f"{torch.clamp(pred['R_val'], 0, 1).item():.4f}"
            c_probs = torch.softmax(pred["c_dir_logits"], dim=-1)
            rec["pred_c_dir"] = int(torch.argmax(c_probs))
            rec["pred_c_dir_probs"] = " ".join(f"{p:.3f}" for p in c_probs.tolist())
            btn_p = torch.sigmoid(pred["btn_logits"])
            rec["pred_btn_probs"] = " ".join(
                f"{IDX_TO_BUTTON[i].name}={p:.3f}"
                for i, p in enumerate(btn_p.tolist())
            )
        if sent_main is not None:
            rec["sent_main_x"] = f"{sent_main[0]:.4f}"
            rec["sent_main_y"] = f"{sent_main[1]:.4f}"
        if sent_c_dir is not None:
            rec["sent_c_dir"] = sent_c_dir
        if sent_L is not None:
            rec["sent_L"] = f"{sent_L:.4f}"
        if sent_R is not None:
            rec["sent_R"] = f"{sent_R:.4f}"
        if sent_btns is not None:
            rec["sent_btns"] = " ".join(
                IDX_TO_BUTTON[i].name
                for i, fired in enumerate(sent_btns) if fired
            )
        self._writer.writerow(rec)

    def close(self) -> None:
        self._fh.close()
        log.info("Inference log closed (%s)", self._path)

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
_inf_call_count = 0

@torch.no_grad()
def run_inference(win_rows: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    global _inf_call_count
    _inf_call_count += 1

    batch = rows_to_state_seq(win_rows)

    if _inf_call_count <= 3 or _inf_call_count in (300, 600, 1200):
        torch.save({k: v.cpu() for k, v in batch.items()},
                   f"/tmp/frame_inf_batch_{_inf_call_count}.pt")
        import pickle
        with open(f"/tmp/frame_inf_rows_{_inf_call_count}.pkl", "wb") as fh:
            pickle.dump(win_rows, fh)
        log.info("Saved inference batch %d to /tmp/", _inf_call_count)

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


def _decode_clusters(pred: Dict[str, torch.Tensor]):
    """For clusters mode: argmax logits -> continuous (mx, my, l, r) via cluster lookup."""
    main_idx = int(torch.argmax(pred["main_xy"]))
    mx, my = float(_stick_centers[main_idx][0]), float(_stick_centers[main_idx][1])
    l_idx = int(torch.argmax(pred["L_val"]))
    r_idx = int(torch.argmax(pred["R_val"]))
    l_val = float(_shoulder_centers[l_idx])
    r_val = float(_shoulder_centers[r_idx])
    return mx, my, l_val, r_val


def press_output(ctrl: melee.Controller,
                 pred: Dict[str, torch.Tensor],
                 sample: bool = True) -> tuple[List[bool], tuple[float, float], int, float, float]:
    """Send model predictions to the controller.
    Returns (fired_buttons, (mx,my), c_dir_idx, l_val, r_val)."""
    import random

    if cfg.stick_loss == "clusters":
        mx, my, l_val, r_val = _decode_clusters(pred)
    else:
        clamped_main = torch.clamp(pred["main_xy"], 0.0, 1.0)
        mx, my = map(float, clamped_main.tolist())
        l_val = _safe(torch.clamp(pred["L_val"], 0.0, 1.0).item(), 0.0)
        r_val = _safe(torch.clamp(pred["R_val"], 0.0, 1.0).item(), 0.0)

    dir_idx = int(torch.argmax(pred["c_dir_logits"]))
    cx, cy  = C_DIR_TO_FLOAT.get(dir_idx, (0.5, 0.5))

    ctrl.tilt_analog(melee.enums.Button.BUTTON_MAIN, mx, my)
    ctrl.tilt_analog(melee.enums.Button.BUTTON_C,    cx, cy)
    ctrl.press_shoulder(melee.enums.Button.BUTTON_L, l_val)
    ctrl.press_shoulder(melee.enums.Button.BUTTON_R, r_val)

    btn_probs = torch.sigmoid(pred["btn_logits"])
    pressed = []
    fired: List[bool] = []
    for prob, btn in zip(btn_probs, IDX_TO_BUTTON):
        p = prob.item()
        fire = (random.random() < p) if sample else (p > 0.5)
        fired.append(fire)
        if fire:
            ctrl.press_button(btn)
            pressed.append(btn.name)
        else:
            ctrl.release_button(btn)

    top3 = btn_probs.topk(3)
    top3_str = " ".join(f"{IDX_TO_BUTTON[i].name}={v:.3f}"
                        for v, i in zip(top3.values.tolist(), top3.indices.tolist()))
    log.info(
        "MAIN=(%.2f,%.2f) C=%d L=%.2f R=%.2f BTN=%s  top3=[%s]",
        mx, my, dir_idx, l_val, r_val, pressed, top3_str
    )
    return fired, (mx, my), dir_idx, l_val, r_val

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
    ports = [1, 4]
    controllers = {p: melee.Controller(console, p) for p in ports}

    signal.signal(signal.SIGINT, signal_handler)
    console.run(iso_path=ISO_PATH)

    if not console.connect():
        log.error("Console connect failed"); sys.exit(1)
    for c in controllers.values():
        if not c.connect():
            log.error("Controller connect failed"); sys.exit(1)
    log.info("Console + controllers connected.")

    menu_helper_bot = melee.MenuHelper()
    menu_helper_cpu = melee.MenuHelper()

    inf_logger = InferenceLogger(args.log_dir)

    rows: deque[Dict[str, Any]] = deque(maxlen=ROLL_WIN)
    _step_ct = 0
    _was_in_game = False
    while True:
        gs = console.step()
        _step_ct += 1
        if _step_ct % 300 == 1:
            log.info("step %d  gs=%s  menu=%s", _step_ct,
                     "None" if gs is None else "ok",
                     getattr(gs, "menu_state", "N/A") if gs else "N/A")
        if gs is None:
            continue

        if gs.menu_state not in (melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH):
            if _was_in_game:
                log.info("Game ended (menu_state=%s). Shutting down.", gs.menu_state)
                break
            log.debug("Menu state: %s", gs.menu_state)
            c0, c1 = controllers[ports[0]], controllers[ports[1]]
            menu_helper_bot.menu_helper_simple(
                gs, c0, BOT_CHARACTER, STAGE,
                cpu_level=0, autostart=False,
            )
            menu_helper_cpu.menu_helper_simple(
                gs, c1, CPU_CHARACTER, STAGE,
                cpu_level=args.cpu_level, autostart=True,
            )
            continue

        _was_in_game = True

        # -- build row (matches extract.py schema) --
        row: Dict[str, Any] = {}
        row["stage"]    = gs.stage.value if gs.stage else -1
        row["frame"]    = gs.frame
        row["distance"] = 0.0
        row["startAt"]  = gs.startAt

        # Stage geometry (critical — model trained with these)
        stg = gs.stage
        bz = melee.stages.BLASTZONES.get(stg, (0, 0, 0, 0))
        row["blastzone_left"]  = bz[0]
        row["blastzone_right"] = bz[1]
        row["blastzone_top"]   = bz[2]
        row["blastzone_bottom"] = bz[3]
        edge = melee.stages.EDGE_POSITION.get(stg, 0)
        row["stage_edge_left"]  = -edge
        row["stage_edge_right"] = edge
        lp = melee.stages.left_platform_position(stg)
        row["left_platform_height"] = lp[0] if lp[0] is not None else float("nan")
        row["left_platform_left"]   = lp[1] if lp[1] is not None else float("nan")
        row["left_platform_right"]  = lp[2] if lp[2] is not None else float("nan")
        rp = melee.stages.right_platform_position(stg)
        row["right_platform_height"] = rp[0] if rp[0] is not None else float("nan")
        row["right_platform_left"]   = rp[1] if rp[1] is not None else float("nan")
        row["right_platform_right"]  = rp[2] if rp[2] is not None else float("nan")
        tp = melee.stages.top_platform_position(stg)
        row["top_platform_height"] = tp[0] if tp[0] is not None else float("nan")
        row["top_platform_left"]   = tp[1] if tp[1] is not None else float("nan")
        row["top_platform_right"]  = tp[2] if tp[2] is not None else float("nan")
        try:
            rnd = melee.stages.randall_position(stg, gs.frame)
            row["randall_height"] = rnd[0] if rnd[0] is not None else float("nan")
            row["randall_left"]   = rnd[1] if rnd[1] is not None else float("nan")
            row["randall_right"]  = rnd[2] if rnd[2] is not None else float("nan")
        except Exception:
            row["randall_height"] = float("nan")
            row["randall_left"]   = float("nan")
            row["randall_right"]  = float("nan")

        for idx, (port, ps) in enumerate(gs.players.items()):
            pref = "self_" if idx == 0 else "opp_"
            row[f"{pref}port"]          = -1  # training data always uses -1
            row[f"{pref}character"]     = ps.character.value
            row[f"{pref}action"]        = ps.action.value
            row[f"{pref}action_frame"]  = ps.action_frame
            row[f"{pref}costume"]       = ps.costume

            if pref == "self_" and _prev_pred is not None:
                if cfg.stick_loss == "clusters":
                    _fb_mx, _fb_my, _fb_l, _fb_r = _decode_clusters(_prev_pred)
                    row[f"{pref}main_x"] = _fb_mx
                    row[f"{pref}main_y"] = _fb_my
                    row[f"{pref}l_shldr"] = _fb_l
                    row[f"{pref}r_shldr"] = _fb_r
                else:
                    _fb_main = torch.clamp(_prev_pred["main_xy"], 0.0, 1.0)
                    row[f"{pref}main_x"] = _fb_main[0].item()
                    row[f"{pref}main_y"] = _fb_main[1].item()
                    row[f"{pref}l_shldr"] = torch.clamp(_prev_pred["L_val"], 0.0, 1.0).item()
                    row[f"{pref}r_shldr"] = torch.clamp(_prev_pred["R_val"], 0.0, 1.0).item()
                _fb_cdir = int(torch.argmax(_prev_pred["c_dir_logits"]))
                row[f"{pref}c_x"], row[f"{pref}c_y"] = C_DIR_TO_FLOAT.get(_fb_cdir, (0.5, 0.5))
                if _prev_btns_fired is not None:
                    for _fired, _btn in zip(_prev_btns_fired, IDX_TO_BUTTON):
                        row[f"{pref}btn_{_btn.name}"] = int(_fired)
                else:
                    _fb_btn_probs = torch.sigmoid(_prev_pred["btn_logits"])
                    for _bp, _btn in zip(_fb_btn_probs, IDX_TO_BUTTON):
                        row[f"{pref}btn_{_btn.name}"] = int(_bp.item() > 0.5)
            else:
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
            row[f"{pref}hitlag_left"]          = 0
            row[f"{pref}hitstun_left"]         = ps.hitstun_frames_left
            row[f"{pref}invuln_left"]          = 0
            row[f"{pref}speed_air_x_self"]     = 0.0
            row[f"{pref}speed_ground_x_self"]  = 0.0
            row[f"{pref}speed_x_attack"]       = 0.0
            row[f"{pref}speed_y_attack"]       = 0.0
            row[f"{pref}speed_y_self"]         = 0.0

            for ecb_axis in ("ecb_bottom_x", "ecb_bottom_y", "ecb_left_x", "ecb_left_y",
                             "ecb_right_x", "ecb_right_y", "ecb_top_x", "ecb_top_y"):
                row[f"{pref}{ecb_axis}"] = 0.0

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
                row[f"{npref}hitlag_left"]         = 0
                row[f"{npref}hitstun_left"]        = nana.hitstun_frames_left
                row[f"{npref}invuln_left"]         = 0
                row[f"{npref}speed_air_x_self"]    = 0.0
                row[f"{npref}speed_ground_x_self"] = 0.0
                row[f"{npref}speed_x_attack"]      = 0.0
                row[f"{npref}speed_y_attack"]      = 0.0
                row[f"{npref}speed_y_self"]        = 0.0
                for ecb_axis in ("ecb_bottom_x", "ecb_bottom_y", "ecb_left_x", "ecb_left_y",
                                 "ecb_right_x", "ecb_right_y", "ecb_top_x", "ecb_top_y"):
                    row[f"{npref}{ecb_axis}"] = 0.0
            else:
                row[f"{npref}character"]    = -1
                row[f"{npref}action"]       = -1
                row[f"{npref}action_frame"] = -1
                for b in F.BTN:
                    row[f"{npref}btn_{b}"] = 0
                row[f"{npref}main_x"] = np.nan
                row[f"{npref}main_y"] = np.nan
                row[f"{npref}c_x"]    = np.nan
                row[f"{npref}c_y"]    = np.nan
                row[f"{npref}l_shldr"] = np.nan
                row[f"{npref}r_shldr"] = np.nan
                for field in ("percent", "pos_x", "pos_y",
                              "shield_strength",
                              "speed_air_x_self", "speed_ground_x_self",
                              "speed_x_attack", "speed_y_attack",
                              "speed_y_self"):
                    row[f"{npref}{field}"] = np.nan
                for field in ("stock", "jumps_left",
                              "hitlag_left", "hitstun_left", "invuln_left"):
                    row[f"{npref}{field}"] = -1
                for field in ("facing", "on_ground", "off_stage",
                              "invulnerable", "moonwalkwarning"):
                    row[f"{npref}{field}"] = 0
                for part in ("bottom", "left", "right", "top"):
                    for axis in ("x", "y"):
                        row[f"{npref}ecb_{part}_{axis}"] = np.nan

        for j in range(F.PROJ_SLOTS):
            pp = f"proj{j}_"
            row[f"{pp}owner"]   = -1
            row[f"{pp}type"]    = -1
            row[f"{pp}subtype"] = -1
            row[f"{pp}pos_x"]   = 0.0
            row[f"{pp}pos_y"]   = 0.0
            row[f"{pp}speed_x"] = 0.0
            row[f"{pp}speed_y"] = 0.0
            row[f"{pp}frame"]   = -1

        if gs.frame < 0:
            continue
        rows.append(row)
        if len(rows) == ROLL_WIN:
            pred = run_inference(list(rows))
            _prev_pred = pred

            bot_ps = gs.players.get(ports[0])
            if bot_ps and _step_ct % 60 == 0:
                gm = bot_ps.controller_state.main_stick
                if cfg.stick_loss == "clusters":
                    _dbg_mx, _dbg_my, _dbg_l, _dbg_r = _decode_clusters(pred)
                    pm = [_dbg_mx, _dbg_my]
                    pl, pr = _dbg_l, _dbg_r
                else:
                    pm = torch.clamp(pred["main_xy"], 0, 1).tolist()
                    pl = torch.clamp(pred["L_val"], 0, 1).item()
                    pr = torch.clamp(pred["R_val"], 0, 1).item()
                gl = bot_ps.controller_state.l_shoulder
                gr = bot_ps.controller_state.r_shoulder
                log.info(
                    "FEEDBACK: game_stick=(%.3f,%.3f) pred_stick=(%.3f,%.3f) "
                    "game_shldr=(%.3f,%.3f) pred_shldr=(%.3f,%.3f)",
                    gm[0], gm[1], pm[0], pm[1], gl, gr, pl, pr,
                )

            ctrl = controllers[ports[0]]
            ctrl.release_all()
            _prev_btns_fired, sent_main, sent_cdir, sent_L, sent_R = press_output(ctrl, pred)
            ctrl.flush()

            inf_logger.log_frame(
                row, pred,
                sent_main=sent_main, sent_c_dir=sent_cdir,
                sent_L=sent_L, sent_R=sent_R,
                sent_btns=_prev_btns_fired,
            )
        else:
            inf_logger.log_frame(
                row, pred=None,
                sent_main=None, sent_c_dir=None,
                sent_L=None, sent_R=None, sent_btns=None,
            )

    log.info("Cleaning up...")
    inf_logger.close()
    for c in controllers.values():
        c.disconnect()
    console.stop()
    log.info("Done.")
