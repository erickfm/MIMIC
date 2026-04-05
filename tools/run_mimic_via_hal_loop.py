#!/usr/bin/env python3
"""Run MIMIC's FramePredictor through run_hal_model.py's preprocessing loop.

Diagnostic: if MIMIC model plays well here but not in inference.py,
the issue is in inference.py. If it plays badly here too, the issue
is the model itself.

Usage:
    python tools/run_mimic_via_hal_loop.py \
      --checkpoint checkpoints/hal-correct-v2-si_best.pt \
      --dolphin-path /path/to/dolphin \
      --iso-path /path/to/melee.iso
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse, json, logging, math, signal, time
from collections import deque

import melee
import numpy as np
import torch
import torch.nn.functional as Fn

from mimic.model import FramePredictor, ModelConfig
from mimic.features import (
    HAL_STICK_CLUSTERS_37, HAL_CSTICK_CLUSTERS_9, HAL_SHOULDER_CLUSTERS_3,
    encode_controller_onehot_single,
)

log = logging.getLogger("mimic_hal_loop")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  [%(levelname)s]  %(message)s")

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--dolphin-path", required=True)
parser.add_argument("--iso-path", required=True)
parser.add_argument("--character", default="FOX")
parser.add_argument("--cpu-character", default="FOX")
parser.add_argument("--cpu-level", type=int, default=9)
parser.add_argument("--stage", default="BATTLEFIELD")
parser.add_argument("--temperature", type=float, default=1.0)
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load MIMIC model ────────────────────────────────────────────────────────
ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
cfg = ModelConfig(**ckpt["config"])
model = FramePredictor(cfg)
sd = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model_state_dict"].items()}
model.load_state_dict(sd)
model.eval()
model.to(DEVICE)
log.info("Loaded MIMIC model: %d params, encoder=%s", sum(p.numel() for p in model.parameters()), cfg.encoder_type)

# ── HAL stats ───────────────────────────────────────────────────────────────
HAL_STATS_PATH = Path("/home/erick/projects/hal/hal/data/stats.json")
with open(HAL_STATS_PATH) as f:
    _raw_stats = json.load(f)
p1s = {k.removeprefix("p1_"): _raw_stats[k] for k in _raw_stats if k.startswith("p1_")}
p2s = {k.removeprefix("p2_"): _raw_stats[k] for k in _raw_stats if k.startswith("p2_")}
for d in (p1s, p2s):
    d["pos_x"] = d["position_x"]
    d["pos_y"] = d["position_y"]
    d["invuln_left"] = d.get("invulnerability_left", {"min": 0, "max": 0, "mean": 0, "std": 0})

def _norm(val, s): return 2*(val-s["min"])/(s["max"]-s["min"])-1 if s["max"]!=s["min"] else 0.0
def _inv(val, s): return 2*(s["max"]-val)/(s["max"]-s["min"])-1 if s["max"]!=s["min"] else 0.0
def _std(val, s): return (val-s["mean"])/s["std"] if s["std"]!=0 else 0.0

XFORM = {"percent": _norm, "stock": _norm, "facing": _norm, "invulnerable": _norm,
         "jumps_left": _norm, "on_ground": _norm, "shield_strength": _inv,
         "pos_x": _std, "pos_y": _std, "invuln_left": _norm}

# ── Enum maps (same as inference.py) ────────────────────────────────────────
from mimic.features import get_enum_map
STAGE_MAP = get_enum_map("stage", {})
CHAR_MAP = get_enum_map("self_character", {})
ACTION_MAP = get_enum_map("self_action", {})

# ── Combo map ───────────────────────────────────────────────────────────────
COMBO_MAP = {(1,0,0,0,0):0,(0,1,0,0,0):1,(0,0,1,0,0):2,(0,0,0,1,0):3,
             (0,0,0,0,0):4,(0,0,0,0,1):4,
             (1,0,0,0,1):0,(0,1,0,0,1):1,(0,0,1,0,1):2,(0,0,0,1,1):3}

# ── Build frame ────────────────────────────────────────────────────────────
SEQ_LEN = cfg.max_seq_len
_frame_cache = deque(maxlen=SEQ_LEN)
_prev_sent = None

# MIMIC numeric order: pos_x, pos_y, percent, stock, jumps_left, invuln_left, shield_strength
MIMIC_NUM = ["pos_x", "pos_y", "percent", "stock", "jumps_left", "invuln_left", "shield_strength"]
MIMIC_FLAGS = ["on_ground", "facing", "invulnerable"]  # HAL flag indices [0, 2, 3]

def build_frame(gs, prev_sent):
    """Build MIMIC-format frame dict from gamestate."""
    players = list(gs.players.items())
    if len(players) < 2:
        return None
    _, ps1 = players[0]
    _, ps2 = players[1]

    stage_idx = STAGE_MAP.get(gs.stage.value, 0)

    def player_feats(ps, is_ego):
        stats = p1s if is_ego else p2s
        char_idx = CHAR_MAP.get(ps.character.value, 0)
        action_idx = ACTION_MAP.get(ps.action.value, 0)

        raw_num = {"pos_x": float(ps.position.x), "pos_y": float(ps.position.y),
                   "percent": float(ps.percent), "stock": float(ps.stock),
                   "jumps_left": float(ps.jumps_left), "invuln_left": float(ps.invulnerability_left),
                   "shield_strength": float(ps.shield_strength)}
        raw_flags = {"on_ground": float(ps.on_ground), "facing": float(ps.facing),
                     "invulnerable": float(ps.invulnerable)}

        nums = [XFORM[f](raw_num[f], stats[f]) for f in MIMIC_NUM]
        flags_norm = [XFORM[f](raw_flags[f], stats[f]) for f in MIMIC_FLAGS]
        # Full 5 flags: [on_ground, off_stage, facing, invulnerable, moonwalkwarning]
        all_flags = [flags_norm[0], 0.0, flags_norm[1], flags_norm[2], 0.0]

        return char_idx, action_idx, nums, all_flags

    ego_char, ego_action, ego_nums, ego_flags = player_feats(ps1, True)
    opp_char, opp_action, opp_nums, opp_flags = player_feats(ps2, False)

    # Controller from prev_sent
    if prev_sent is not None:
        mx, my = prev_sent["main_x"], prev_sent["main_y"]
        cx, cy = prev_sent["c_x"], prev_sent["c_y"]
        ls, rs = prev_sent["l_shldr"], prev_sent["r_shldr"]
        btns = {b: prev_sent.get(f"btn_{b}", 0) for b in
                ["BUTTON_A","BUTTON_B","BUTTON_X","BUTTON_Y","BUTTON_Z","BUTTON_L","BUTTON_R"]}
    else:
        mx, my, cx, cy, ls, rs = 0.5, 0.5, 0.5, 0.5, 0.0, 0.0
        btns = {}

    controller = encode_controller_onehot_single(mx, my, cx, cy, ls, rs, btns, COMBO_MAP, 5)

    return {
        "stage": torch.tensor([stage_idx], dtype=torch.long),
        "self_character": torch.tensor([ego_char], dtype=torch.long),
        "opp_character": torch.tensor([opp_char], dtype=torch.long),
        "self_action": torch.tensor([ego_action], dtype=torch.long),
        "opp_action": torch.tensor([opp_action], dtype=torch.long),
        "self_numeric": torch.tensor([ego_nums], dtype=torch.float32),
        "opp_numeric": torch.tensor([opp_nums], dtype=torch.float32),
        "self_flags": torch.tensor([ego_flags], dtype=torch.float32),
        "opp_flags": torch.tensor([opp_flags], dtype=torch.float32),
        "self_controller": torch.from_numpy(controller).unsqueeze(0),
        # Stubs for keys the model might expect
        "self_costume": torch.tensor([0], dtype=torch.long),
        "opp_costume": torch.tensor([0], dtype=torch.long),
        "self_port": torch.tensor([0], dtype=torch.long),
        "opp_port": torch.tensor([0], dtype=torch.long),
        "self_c_dir": torch.tensor([0], dtype=torch.long),
        "opp_c_dir": torch.tensor([0], dtype=torch.long),
    }


def stack_frames():
    frames = list(_frame_cache)
    batch = {}
    for k in frames[0]:
        batch[k] = torch.cat([f[k] for f in frames], dim=0).unsqueeze(0).to(DEVICE)
    return batch


# ── Decode + press ──────────────────────────────────────────────────────────
BUTTONS_NO_SHOULDER = [
    melee.enums.Button.BUTTON_A, melee.enums.Button.BUTTON_B,
    melee.enums.Button.BUTTON_X, melee.enums.Button.BUTTON_Z,
]

def decode_and_press(ctrl, preds, gs=None, temperature=1.0):
    global _prev_sent

    main_probs = Fn.softmax(preds["main_xy"][0, -1].float() / temperature, dim=-1)
    main_idx = int(torch.multinomial(main_probs, 1))
    mx, my = float(HAL_STICK_CLUSTERS_37[main_idx][0]), float(HAL_STICK_CLUSTERS_37[main_idx][1])

    shldr_probs = Fn.softmax(preds["shoulder_val"][0, -1].float() / temperature, dim=-1)
    shldr_idx = int(torch.multinomial(shldr_probs, 1))
    shldr = [0.0, 0.4, 1.0][shldr_idx]

    n_cdir = preds["c_dir_logits"].size(-1)
    if n_cdir == 9:
        c_probs = Fn.softmax(preds["c_dir_logits"][0, -1].float() / temperature, dim=-1)
        c_idx = int(torch.multinomial(c_probs, 1))
        cx, cy = float(HAL_CSTICK_CLUSTERS_9[c_idx][0]), float(HAL_CSTICK_CLUSTERS_9[c_idx][1])
    else:
        dir_idx = int(torch.argmax(preds["c_dir_logits"][0, -1]))
        C_DIR = {0:(0.5,0.5),1:(0.5,1.0),2:(0.5,0.0),3:(0.0,0.5),4:(1.0,0.5)}
        cx, cy = C_DIR.get(dir_idx, (0.5, 0.5))

    btn_probs = Fn.softmax(preds["btn_logits"][0, -1].float() / temperature, dim=-1)
    btn_idx = int(torch.multinomial(btn_probs, 1))

    ctrl.tilt_analog(melee.enums.Button.BUTTON_MAIN, mx, my)
    ctrl.tilt_analog(melee.enums.Button.BUTTON_C, cx, cy)
    ctrl.press_shoulder(melee.enums.Button.BUTTON_L, shldr)

    pressed = []
    for btn in BUTTONS_NO_SHOULDER:
        ctrl.release_button(btn)
    if btn_idx < 4:
        btn = BUTTONS_NO_SHOULDER[btn_idx]
        ctrl.press_button(btn)
        pressed.append(btn.name)
    ctrl.flush()

    _prev_sent = {"main_x": mx, "main_y": my, "c_x": cx, "c_y": cy,
                  "l_shldr": shldr, "r_shldr": 0.0}
    for b in ["BUTTON_A","BUTTON_B","BUTTON_X","BUTTON_Y","BUTTON_Z","BUTTON_L","BUTTON_R"]:
        _prev_sent[f"btn_{b}"] = 0
    if btn_idx < 4:
        _prev_sent[f"btn_{BUTTONS_NO_SHOULDER[btn_idx].name}"] = 1

    NAMES = ["A", "B", "Jump", "Z", "NONE"]
    top3 = btn_probs.topk(min(3, len(btn_probs)))
    top3_str = " ".join(f"{NAMES[i]}={v:.3f}" for v, i in zip(top3.values.tolist(), top3.indices.tolist()))
    gs_str = ""
    if gs is not None:
        players = list(gs.players.items())
        if len(players) >= 2:
            ps1, ps2 = players[0][1], players[1][1]
            gs_str = f"  S={ps1.stock}({ps1.percent:.0f}%) O={ps2.stock}({ps2.percent:.0f}%)"
    log.info("MAIN=(%.2f,%.2f) C=(%.2f,%.2f) L=%.2f BTN=%s  top3=[%s]%s",
             mx, my, cx, cy, shldr, pressed, top3_str, gs_str)


# ── Dolphin loop ────────────────────────────────────────────────────────────
console = melee.Console(path=args.dolphin_path, is_dolphin=True, tmp_home_directory=True,
    copy_home_directory=False, blocking_input=False, online_delay=0, setup_gecko_codes=True,
    fullscreen=False, gfx_backend="", disable_audio=False, use_exi_inputs=False, enable_ffw=False)
ego_ctrl = melee.Controller(console=console, port=1, type=melee.ControllerType.STANDARD)
cpu_ctrl = melee.Controller(console=console, port=2, type=melee.ControllerType.STANDARD)
console.run(iso_path=args.iso_path)
console.connect(); ego_ctrl.connect(); cpu_ctrl.connect()
log.info("Connected")

menu_bot = melee.MenuHelper()
menu_cpu = melee.MenuHelper()
BOT_CHAR = melee.Character[args.character]
CPU_CHAR = melee.Character[args.cpu_character]
STAGE = melee.Stage[args.stage]

def shutdown(*a):
    console.stop(); sys.exit(0)
signal.signal(signal.SIGINT, shutdown)

_was_in_game = False
while True:
    gs = console.step()
    if gs is None: continue
    if gs.menu_state not in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
        if _was_in_game:
            log.info("Game ended."); shutdown()
        menu_bot.menu_helper_simple(gs, ego_ctrl, BOT_CHAR, STAGE, cpu_level=0, autostart=False)
        menu_cpu.menu_helper_simple(gs, cpu_ctrl, CPU_CHAR, STAGE, cpu_level=args.cpu_level, autostart=True)
        ego_ctrl.flush(); cpu_ctrl.flush()
        continue

    _was_in_game = True
    frame = build_frame(gs, _prev_sent)
    if frame is None: continue

    if len(_frame_cache) == 0:
        for _ in range(SEQ_LEN - 1):
            _frame_cache.append({k: v.clone() for k, v in frame.items()})
    _frame_cache.append(frame)

    batch = stack_frames()
    with torch.no_grad():
        preds = model(batch)

    decode_and_press(ego_ctrl, preds, gs=gs, temperature=args.temperature)
