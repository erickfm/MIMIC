#!/usr/bin/env python3
"""Run HAL's trained GPTv5Controller model through MIMIC's Dolphin inference loop.

Reimplements HAL's architecture minimally, loads HAL's checkpoint,
preprocesses with HAL's exact normalization, and plays via Dolphin.

Usage:
    python tools/run_hal_model.py \
      --checkpoint checkpoints/hal_original.pt \
      --dolphin-path /path/to/dolphin-emu \
      --iso-path /path/to/melee.iso \
      --character FOX
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import logging
import math
import random
import signal
import time
from collections import deque

import melee
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

import mimic.features as F
from mimic.features import (
    HAL_STICK_CLUSTERS_37, HAL_CSTICK_CLUSTERS_9, HAL_SHOULDER_CLUSTERS_3,
    hal_normalize, load_hal_norm,
    encode_controller_onehot_single,
)

log = logging.getLogger("hal_inf")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  [%(levelname)s]  %(message)s")

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Run HAL model via MIMIC inference")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--dolphin-path", required=True)
parser.add_argument("--iso-path", required=True)
parser.add_argument("--character", default="FOX")
parser.add_argument("--cpu-character", default="FOX")
parser.add_argument("--cpu-level", type=int, default=9)
parser.add_argument("--stage", default="FINAL_DESTINATION")
parser.add_argument("--data-dir", default="data/fox_public_shards")
parser.add_argument("--temperature", type=float, default=1.0)
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── HAL Model (minimal reimplementation) ─────────────────────────────────────

def skew(QEr):
    padded = Fn.pad(QEr, (1, 0))
    B, nh, nr, nc = padded.shape
    reshaped = padded.reshape(B, nh, nc, nr)
    return reshaped[:, :, 1:, :]


class CausalSelfAttentionRelPos(nn.Module):
    def __init__(self, n_embd=512, n_head=8, block_size=1024, dropout=0.2):
        super().__init__()
        self.n_head = n_head
        self.hs = n_embd // n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.Er = nn.Parameter(torch.randn(block_size, self.hs))
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.register_buffer("bias",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))
        self.block_size = block_size

    def forward(self, x):
        B, L, D = x.size()
        q, k, v = self.c_attn(x).split(D, dim=2)
        k = k.view(B, L, self.n_head, self.hs).transpose(1, 2)
        q = q.view(B, L, self.n_head, self.hs).transpose(1, 2)
        v = v.view(B, L, self.n_head, self.hs).transpose(1, 2)

        start = self.block_size - L
        Er_t = self.Er[start:, :].transpose(0, 1)
        QEr = q @ Er_t
        Srel = skew(QEr)

        QK_t = q @ k.transpose(-2, -1)
        scale = 1.0 / math.sqrt(k.size(-1))
        att = (QK_t + Srel) * scale
        att = att.masked_fill(self.bias[:, :, :L, :L] == 0, float("-inf"))
        att = Fn.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, L, D)
        return self.resid_dropout(self.c_proj(y))


class Block(nn.Module):
    def __init__(self, n_embd=512, n_head=8, block_size=1024, dropout=0.2):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttentionRelPos(n_embd, n_head, block_size, dropout)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc=nn.Linear(n_embd, 4 * n_embd),
            c_proj=nn.Linear(4 * n_embd, n_embd),
        ))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        h = self.mlp.c_fc(self.ln_2(x))
        h = Fn.gelu(h)
        h = self.dropout(self.mlp.c_proj(h))
        return x + h


class HALModel(nn.Module):
    """Minimal GPTv5Controller reimplementation."""
    def __init__(self, n_embd=512, n_head=8, n_layer=6, block_size=1024, dropout=0.2):
        super().__init__()
        self.stage_emb = nn.Embedding(6, 4)
        self.character_emb = nn.Embedding(27, 12)
        self.action_emb = nn.Embedding(396, 32)

        self.transformer = nn.ModuleDict(dict(
            proj_down=nn.Linear(164, n_embd),
            drop=nn.Dropout(dropout),
            h=nn.ModuleList([Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]),
            ln_f=nn.LayerNorm(n_embd),
        ))

        def _head(in_dim, out_dim):
            return nn.Sequential(
                nn.LayerNorm(in_dim), nn.Linear(in_dim, in_dim // 2),
                nn.GELU(), nn.Linear(in_dim // 2, out_dim))

        self.shoulder_head = _head(n_embd, 3)
        self.c_stick_head = _head(n_embd + 3, 9)
        self.main_stick_head = _head(n_embd + 3 + 9, 37)
        self.button_head = _head(n_embd + 3 + 9 + 37, 5)

    def forward(self, stage, ego_char, opp_char, ego_action, opp_action, gamestate, controller):
        combined = torch.cat([
            self.stage_emb(stage),
            self.character_emb(ego_char),
            self.character_emb(opp_char),
            self.action_emb(ego_action),
            self.action_emb(opp_action),
            gamestate,
            controller,
        ], dim=-1)

        x = self.transformer.proj_down(combined)
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        shoulder = self.shoulder_head(x)
        c_stick = self.c_stick_head(torch.cat([x, shoulder.detach()], dim=-1))
        main_stick = self.main_stick_head(torch.cat([x, shoulder.detach(), c_stick.detach()], dim=-1))
        buttons = self.button_head(torch.cat([x, shoulder.detach(), c_stick.detach(), main_stick.detach()], dim=-1))

        return {"shoulder": shoulder, "c_stick": c_stick, "main_stick": main_stick, "buttons": buttons}


# ── Load model ───────────────────────────────────────────────────────────────

log.info("Loading checkpoint: %s", args.checkpoint)
raw = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)

# Detect checkpoint format: HAL (bare state_dict) vs MIMIC (wrapped with config)
_is_mimic = isinstance(raw, dict) and "model_state_dict" in raw


class _InferenceModel(nn.Module):
    """Universal inference wrapper for any FramePredictor checkpoint.

    Bypasses the encoder's shard-format column selection and feeds pre-assembled
    (embeddings + gamestate + controller) directly into projection → backbone → heads.
    Works with any architecture (LayerNorm/RMSNorm, GELU/SwiGLU, MHA/GQA, RoPE/relpos).
    """
    def __init__(self, fp):
        super().__init__()
        self.fp = fp

    def forward(self, stage, ego_char, opp_char, ego_action, opp_action, gamestate, controller):
        enc = self.fp.encoder
        combined = torch.cat([
            enc.stage_emb((stage - 1).clamp(min=0) if enc.stage_emb.num_embeddings == 6 else stage),
            enc.char_emb(ego_char),
            enc.char_emb(opp_char),
            enc.action_emb(ego_action),
            enc.action_emb(opp_action),
            gamestate,
            controller,
        ], dim=-1)
        x = enc.drop(enc.proj(combined))
        for blk in self.fp.blocks:
            x = blk(x)
        x = self.fp.final_norm(x)
        preds = self.fp.heads(x)
        return {
            "shoulder": preds.get("shoulder_val", preds.get("shoulder")),
            "c_stick": preds.get("c_dir_logits", preds.get("c_stick")),
            "main_stick": preds.get("main_xy", preds.get("main_stick")),
            "buttons": preds.get("btn_logits", preds.get("buttons")),
        }


if _is_mimic:
    from mimic.model import FramePredictor, ModelConfig
    import dataclasses
    # Reconstruct model from saved config — works for any architecture
    ckpt_cfg = raw.get("config", {})
    valid_fields = {f.name for f in dataclasses.fields(ModelConfig)}
    mimic_cfg = ModelConfig(**{k: v for k, v in ckpt_cfg.items() if k in valid_fields})
    fp = FramePredictor(mimic_cfg).to(DEVICE)
    fp.load_state_dict(raw["model_state_dict"])
    fp.eval()
    model = _InferenceModel(fp)
    preset = ckpt_cfg.get("model_preset", "unknown")
    log.info("MIMIC model loaded (preset=%s): %d params", preset,
             sum(p.numel() for p in fp.parameters()))
else:
    # HAL's bare state_dict — use the hardcoded HALModel
    state_dict = {k.removeprefix("module."): v for k, v in raw.items()}
    hal = HALModel().to(DEVICE)
    hal.load_state_dict(state_dict)
    hal.eval()
    model = hal
    log.info("HAL model loaded: %d params", sum(p.numel() for p in hal.parameters()))

# ── Load HAL's actual normalization stats ─────────────────────────────────────

# Load checkpoint stats — despite play.py appearing to override to hal/data/stats.json,
# the Preprocessor actually loads checkpoints/stats.json (Fox subset, 27M frames).
# Verified: pp.stats["p1_percent"].max == 236.0, not 362.0.
HAL_STATS_PATH = Path("/home/erick/projects/hal/checkpoints/stats.json")
with open(HAL_STATS_PATH) as _f:
    _raw_stats = json.load(_f)

class _Stats:
    """Mimics HAL's FeatureStats(mean, std, min, max)."""
    def __init__(self, d):
        self.mean = d["mean"]
        self.std = d["std"]
        self.min = d["min"]
        self.max = d["max"]

# Player-specific stats — HAL uses p1 stats for ego, p2 stats for opponent
HAL_P1_STATS = {k.removeprefix("p1_"): _Stats(_raw_stats[k])
                for k in _raw_stats if k.startswith("p1_")}
HAL_P2_STATS = {k.removeprefix("p2_"): _Stats(_raw_stats[k])
                for k in _raw_stats if k.startswith("p2_")}

# HAL's exact transforms per feature (from hal/preprocess/input_configs.py baseline())
def _hal_normalize(val, stats):
    """normalize: [-1, 1]"""
    return 2.0 * (val - stats.min) / (stats.max - stats.min) - 1.0

def _hal_invert_normalize(val, stats):
    """invert_and_normalize: [-1, 1] inverted"""
    return 2.0 * (stats.max - val) / (stats.max - stats.min) - 1.0

def _hal_standardize(val, stats):
    """standardize: zero mean, unit variance"""
    return (val - stats.mean) / stats.std

# Which transform each feature uses (from HAL's input_configs.py baseline())
HAL_TRANSFORM = {
    "percent": _hal_normalize,
    "stock": _hal_normalize,
    "facing": _hal_normalize,
    "invulnerable": _hal_normalize,
    "jumps_left": _hal_normalize,
    "on_ground": _hal_normalize,
    "shield_strength": _hal_invert_normalize,
    "position_x": _hal_standardize,
    "position_y": _hal_standardize,
}

log.info("Loaded HAL stats from %s (%d p1 features, %d p2 features)",
         HAL_STATS_PATH, len(HAL_P1_STATS), len(HAL_P2_STATS))

# ── HAL categorical mappings ─────────────────────────────────────────────────

from melee.enums import Character, Action, Stage

HAL_CHARACTERS = [
    "MARIO", "FOX", "CPTFALCON", "DK", "KIRBY", "BOWSER", "LINK", "SHEIK",
    "NESS", "PEACH", "POPO", "NANA", "PIKACHU", "SAMUS", "YOSHI",
    "JIGGLYPUFF", "MEWTWO", "LUIGI", "MARTH", "ZELDA", "YLINK", "DOC",
    "FALCO", "PICHU", "GAMEANDWATCH", "GANONDORF", "ROY",
]
HAL_CHAR_MAP = {char: i for i, char in enumerate(
    c for c in Character if c.name in HAL_CHARACTERS)}

HAL_STAGES = ["FINAL_DESTINATION", "BATTLEFIELD", "POKEMON_STADIUM",
              "DREAMLAND", "FOUNTAIN_OF_DREAMS", "YOSHIS_STORY"]
HAL_STAGE_MAP = {stage: i for i, stage in enumerate(
    s for s in Stage if s.name in HAL_STAGES)}

# Actions: HAL uses all actions (melee.Action enum → dense index)
HAL_ACTION_MAP = {a: i for i, a in enumerate(Action)}

# ── Preprocessing ────────────────────────────────────────────────────────────

# HAL's button class ordering: A=0, B=1, Jump=2, Z=3, NONE=4
# (from encode_buttons_one_hot_no_shoulder: stacked = [a, b, jump, z, no_button])
# The 5th combo element is shoulder — HAL treats shoulder-only as NONE for buttons.
COMBO_MAP = {
    (1, 0, 0, 0, 0): 0,  # A
    (0, 1, 0, 0, 0): 1,  # B
    (0, 0, 1, 0, 0): 2,  # Jump
    (0, 0, 0, 1, 0): 3,  # Z
    (0, 0, 0, 0, 0): 4,  # NONE
    # Shoulder-only combos → NONE for button section (shoulder is separate)
    (0, 0, 0, 0, 1): 4,  # shoulder only → NONE
    # Combined button + shoulder → use the button class
    (1, 0, 0, 0, 1): 0,  # A + shoulder → A
    (0, 1, 0, 0, 1): 1,  # B + shoulder → B
    (0, 0, 1, 0, 1): 2,  # Jump + shoulder → Jump
    (0, 0, 0, 1, 1): 3,  # Z + shoulder → Z
}
N_COMBOS = 5

SEQ_LEN = 256
_frame_cache = deque(maxlen=SEQ_LEN)
_prev_sent = None

BTN_ENUMS = [melee.enums.Button[name] for name in F.BTN]


def build_frame(gs, prev_sent):
    """Build a single frame dict for HAL's model."""
    players = sorted(gs.players.items())
    if len(players) < 2:
        return None

    port1, ps1 = players[0]
    port2, ps2 = players[1]

    # Stage
    stage_idx = HAL_STAGE_MAP.get(gs.stage, 0)

    # Per player: normalize with HAL's exact transforms and player-specific stats
    def player_features(ps, is_ego):
        char_idx = HAL_CHAR_MAP.get(ps.character, 0)
        action_idx = HAL_ACTION_MAP.get(ps.action, 0)
        stats = HAL_P1_STATS if is_ego else HAL_P2_STATS

        # 9 numeric features in HAL's exact order (from input_configs.py baseline())
        # Order: percent, stock, facing, invulnerable, jumps_left, on_ground,
        #        shield_strength, position_x, position_y
        raw = {
            "percent": float(ps.percent),
            "stock": float(ps.stock),
            "facing": float(ps.facing),
            "invulnerable": float(ps.invulnerable),
            "jumps_left": float(ps.jumps_left),
            "on_ground": float(ps.on_ground),
            "shield_strength": float(ps.shield_strength),
            "position_x": float(ps.position.x),
            "position_y": float(ps.position.y),
        }
        normed = []
        for feat in ["percent", "stock", "facing", "invulnerable", "jumps_left",
                      "on_ground", "shield_strength", "position_x", "position_y"]:
            transform = HAL_TRANSFORM[feat]
            normed.append(transform(raw[feat], stats[feat]))

        return char_idx, action_idx, normed

    ego_char, ego_action, ego_nums = player_features(ps1, is_ego=True)
    opp_char, opp_action, opp_nums = player_features(ps2, is_ego=False)

    # Controller: read from game engine (matches HAL's extract_eval_gamestate_as_tensordict).
    # HAL's play.py reads controller_state directly from the gamestate — the game engine
    # naturally provides the previous frame's controller state, matching the -1 offset.
    cs1 = ps1.controller_state
    mx = float(cs1.main_stick[0])
    my = float(cs1.main_stick[1])
    cx = float(cs1.c_stick[0])
    cy = float(cs1.c_stick[1])
    ls = float(cs1.l_shoulder)
    rs = float(cs1.r_shoulder)
    btns = {}
    for bname in ["BUTTON_A", "BUTTON_B", "BUTTON_X", "BUTTON_Y", "BUTTON_Z",
                   "BUTTON_L", "BUTTON_R"]:
        btns[bname] = int(cs1.button.get(melee.enums.Button[bname], False))

    controller = encode_controller_onehot_single(
        mx, my, cx, cy, ls, rs, btns, COMBO_MAP, N_COMBOS)

    # Gamestate: concat ego + opp numeric (9 + 9 = 18)
    gamestate = np.array(ego_nums + opp_nums, dtype=np.float32)

    return {
        "stage": stage_idx,
        "ego_char": ego_char,
        "opp_char": opp_char,
        "ego_action": ego_action,
        "opp_action": opp_action,
        "gamestate": gamestate,
        "controller": controller,
    }


def stack_frames():
    """Stack cached frames into batched tensors."""
    frames = list(_frame_cache)
    T = len(frames)
    return {
        "stage": torch.tensor([[f["stage"] for f in frames]], dtype=torch.long, device=DEVICE),
        "ego_char": torch.tensor([[f["ego_char"] for f in frames]], dtype=torch.long, device=DEVICE),
        "opp_char": torch.tensor([[f["opp_char"] for f in frames]], dtype=torch.long, device=DEVICE),
        "ego_action": torch.tensor([[f["ego_action"] for f in frames]], dtype=torch.long, device=DEVICE),
        "opp_action": torch.tensor([[f["opp_action"] for f in frames]], dtype=torch.long, device=DEVICE),
        "gamestate": torch.tensor([[f["gamestate"] for f in frames]], dtype=torch.float32, device=DEVICE),
        "controller": torch.tensor([[f["controller"] for f in frames]], dtype=torch.float32, device=DEVICE),
    }


# ── Dolphin loop ─────────────────────────────────────────────────────────────

HAL_STICK_37 = HAL_STICK_CLUSTERS_37
HAL_CSTICK_9 = HAL_CSTICK_CLUSTERS_9
HAL_SHOULDER_3 = HAL_SHOULDER_CLUSTERS_3
INCLUDED_BUTTONS_NO_SHOULDER = [
    melee.enums.Button.BUTTON_A, melee.enums.Button.BUTTON_B,
    melee.enums.Button.BUTTON_X, melee.enums.Button.BUTTON_Z,
]
# HAL's send_controller_inputs releases ALL 7 original buttons every frame
ALL_BUTTONS = [
    melee.enums.Button.BUTTON_A, melee.enums.Button.BUTTON_B,
    melee.enums.Button.BUTTON_X, melee.enums.Button.BUTTON_Y,
    melee.enums.Button.BUTTON_Z, melee.enums.Button.BUTTON_L,
    melee.enums.Button.BUTTON_R,
]


def decode_and_press(ctrl, preds, gs=None, temperature=1.0):
    """Decode model predictions and press controller buttons."""
    global _prev_sent

    # Main stick
    main_probs = Fn.softmax(preds["main_stick"][0, -1].float() / temperature, dim=-1)
    main_idx = int(torch.multinomial(main_probs, 1))
    mx, my = float(HAL_STICK_37[main_idx][0]), float(HAL_STICK_37[main_idx][1])

    # C-stick
    c_probs = Fn.softmax(preds["c_stick"][0, -1].float() / temperature, dim=-1)
    c_idx = int(torch.multinomial(c_probs, 1))
    cx, cy = float(HAL_CSTICK_9[c_idx][0]), float(HAL_CSTICK_9[c_idx][1])

    # Shoulder
    s_probs = Fn.softmax(preds["shoulder"][0, -1].float() / temperature, dim=-1)
    s_idx = int(torch.multinomial(s_probs, 1))
    shldr = float(HAL_SHOULDER_3[s_idx])

    # Buttons
    btn_probs = Fn.softmax(preds["buttons"][0, -1].float() / temperature, dim=-1)
    btn_idx = int(torch.multinomial(btn_probs, 1))

    ctrl.tilt_analog(melee.enums.Button.BUTTON_MAIN, mx, my)
    ctrl.tilt_analog(melee.enums.Button.BUTTON_C, cx, cy)
    ctrl.press_shoulder(melee.enums.Button.BUTTON_L, shldr)

    # Release ALL 7 buttons (matching HAL's send_controller_inputs which iterates
    # ORIGINAL_BUTTONS). Without this, Y/L/R can get stuck pressed.
    for btn in ALL_BUTTONS:
        ctrl.release_button(btn)
    pressed = []
    if btn_idx < 4:
        btn = INCLUDED_BUTTONS_NO_SHOULDER[btn_idx]
        ctrl.press_button(btn)
        pressed.append(btn.name)

    ctrl.flush()

    # Track what we sent
    _prev_sent = {
        "main_x": mx, "main_y": my,
        "c_x": cx, "c_y": cy,
        "l_shldr": shldr, "r_shldr": 0.0,
    }
    for b in ["BUTTON_A", "BUTTON_B", "BUTTON_X", "BUTTON_Y", "BUTTON_Z",
              "BUTTON_L", "BUTTON_R"]:
        _prev_sent[f"btn_{b}"] = 0
    if btn_idx < 4:
        btn = INCLUDED_BUTTONS_NO_SHOULDER[btn_idx]
        _prev_sent[f"btn_{btn.name}"] = 1

    # Track game state for summary and stock-change events
    global _game_start_stocks, _game_max_damage
    if gs is not None:
        players = sorted(gs.players.items())
        if len(players) >= 2:
            ps1, ps2 = players[0][1], players[1][1]
            cur_stocks = (ps1.stock, ps2.stock)
            if _game_start_stocks is None:
                _game_start_stocks = cur_stocks
            # Log stock changes
            _game_max_damage[0] = max(_game_max_damage[0], ps1.percent)
            _game_max_damage[1] = max(_game_max_damage[1], ps2.percent)

    # Per-frame logging: only every 60 frames (~1 second)
    if game_frame % 60 == 0:
        top3 = btn_probs.topk(min(3, len(btn_probs)))
        NAMES = ["A", "B", "Jump", "Z", "NONE"]
        top3_str = " ".join(f"{NAMES[i]}={v:.3f}" for v, i in zip(top3.values.tolist(), top3.indices.tolist()))
        gs_str = ""
        if gs is not None and len(players) >= 2:
            gs_str = f"  S={ps1.stock}({ps1.percent:.0f}%) O={ps2.stock}({ps2.percent:.0f}%)"
        log.info("[f%d] MAIN=(%.2f,%.2f) C=(%.2f,%.2f) L=%.2f BTN=%s  top3=[%s]%s",
                 game_frame, mx, my, cx, cy, shldr, pressed, top3_str, gs_str)


# ── Main ─────────────────────────────────────────────────────────────────────

console = melee.Console(
    path=args.dolphin_path,
    is_dolphin=True,
    tmp_home_directory=True,
    copy_home_directory=False,
    blocking_input=False,
    online_delay=0,
    setup_gecko_codes=True,
    fullscreen=False,
    gfx_backend="",
    disable_audio=False,
    use_exi_inputs=False,
    enable_ffw=False,
)
ego_ctrl = melee.Controller(console=console, port=1, type=melee.ControllerType.STANDARD)
cpu_ctrl = melee.Controller(console=console, port=2, type=melee.ControllerType.STANDARD)
console.run(iso_path=args.iso_path)
console.connect()
ego_ctrl.connect()
cpu_ctrl.connect()
log.info("Connected to Dolphin")

menu_bot = melee.MenuHelper()
menu_cpu = melee.MenuHelper()
BOT_CHAR = melee.Character[args.character]
CPU_CHAR = melee.Character[args.cpu_character]
STAGE = melee.Stage[args.stage]

def _shutdown(*a):
    ego_ctrl.disconnect()
    cpu_ctrl.disconnect()
    console.stop()
    log.info("Done.")
    sys.exit(0)

signal.signal(signal.SIGINT, _shutdown)

step = 0
game_frame = 0
_was_in_game = False
_game_start_stocks = None
_game_max_damage = [0.0, 0.0]  # track max percent seen per player

# Pre-allocate context window matching HAL's play.py:
# HAL fills mock with torch.ones for all features, then preprocesses them.
# We compute the preprocessed mock values directly.

# Mock gamestate: normalize raw value 1.0 through each feature transform
_mock_ego = []
_mock_opp = []
for feat in ["percent", "stock", "facing", "invulnerable", "jumps_left",
             "on_ground", "shield_strength", "position_x", "position_y"]:
    _mock_ego.append(HAL_TRANSFORM[feat](1.0, HAL_P1_STATS[feat]))
    _mock_opp.append(HAL_TRANSFORM[feat](1.0, HAL_P2_STATS[feat]))
_mock_gs_vec = torch.tensor(_mock_ego + _mock_opp, dtype=torch.float32, device=DEVICE)

# Mock controller: HAL's mock fills torch.ones for all features, so
# main_stick=(1.0,1.0), c_stick=(1.0,1.0), all buttons=1, shoulder=1.0
# These get encoded through the target config to produce one-hot vectors.
_mock_btns = {"BUTTON_A": 1, "BUTTON_B": 1, "BUTTON_X": 1, "BUTTON_Y": 1,
              "BUTTON_Z": 1, "BUTTON_L": 1, "BUTTON_R": 1}
_mock_ctrl_vec = torch.from_numpy(
    encode_controller_onehot_single(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, _mock_btns, COMBO_MAP, N_COMBOS)
).to(dtype=torch.float32, device=DEVICE)

log.info("Mock gamestate (18d): %s", [f"{v:.4f}" for v in _mock_gs_vec.tolist()])
log.info("Mock controller nonzero: %s", _mock_ctrl_vec.nonzero().squeeze().tolist())

_ctx_stage = torch.ones(1, SEQ_LEN, dtype=torch.long, device=DEVICE)
_ctx_ego_char = torch.ones(1, SEQ_LEN, dtype=torch.long, device=DEVICE)
_ctx_opp_char = torch.ones(1, SEQ_LEN, dtype=torch.long, device=DEVICE)
_ctx_ego_act = torch.ones(1, SEQ_LEN, dtype=torch.long, device=DEVICE)
_ctx_opp_act = torch.ones(1, SEQ_LEN, dtype=torch.long, device=DEVICE)
_ctx_gs = _mock_gs_vec.unsqueeze(0).unsqueeze(0).expand(1, SEQ_LEN, -1).clone()
_ctx_ctrl = _mock_ctrl_vec.unsqueeze(0).unsqueeze(0).expand(1, SEQ_LEN, -1).clone()

while True:
    gs = console.step()
    if gs is None:
        continue
    if gs.menu_state not in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
        if _was_in_game:
            # Game ended — print summary
            players = sorted(gs.players.items()) if gs.players else []
            if len(players) >= 2:
                ps1, ps2 = players[0][1], players[1][1]
                ego_stocks = ps1.stock
                opp_stocks = ps2.stock
            else:
                ego_stocks = opp_stocks = "?"
            duration_s = game_frame / 60.0
            log.info("=" * 60)
            log.info("GAME OVER — %d frames (%.1fs)", game_frame, duration_s)
            log.info("  Bot:  %s stocks remaining", ego_stocks)
            log.info("  CPU:  %s stocks remaining", opp_stocks)
            if isinstance(ego_stocks, int) and isinstance(opp_stocks, int):
                if ego_stocks > opp_stocks:
                    log.info("  Result: BOT WINS")
                elif opp_stocks > ego_stocks:
                    log.info("  Result: CPU WINS")
                else:
                    log.info("  Result: DRAW")
            log.info("=" * 60)
            _shutdown()
        menu_bot.menu_helper_simple(gs, ego_ctrl, BOT_CHAR, STAGE,
                                     cpu_level=0, autostart=False)
        menu_cpu.menu_helper_simple(gs, cpu_ctrl, CPU_CHAR, STAGE,
                                     cpu_level=args.cpu_level, autostart=True)
        ego_ctrl.flush()
        cpu_ctrl.flush()
        step += 1
        continue

    _was_in_game = True

    frame = build_frame(gs, _prev_sent)
    if frame is None:
        continue

    # Fill context window matching HAL's play.py exactly:
    # Fill from left for first SEQ_LEN frames, then shift left + append right
    if game_frame < SEQ_LEN:
        idx = game_frame
        _ctx_stage[0, idx] = frame["stage"]
        _ctx_ego_char[0, idx] = frame["ego_char"]
        _ctx_opp_char[0, idx] = frame["opp_char"]
        _ctx_ego_act[0, idx] = frame["ego_action"]
        _ctx_opp_act[0, idx] = frame["opp_action"]
        _ctx_gs[0, idx] = torch.from_numpy(frame["gamestate"])
        _ctx_ctrl[0, idx] = torch.from_numpy(frame["controller"])
    else:
        _ctx_stage[0, :-1] = _ctx_stage[0, 1:].clone()
        _ctx_ego_char[0, :-1] = _ctx_ego_char[0, 1:].clone()
        _ctx_opp_char[0, :-1] = _ctx_opp_char[0, 1:].clone()
        _ctx_ego_act[0, :-1] = _ctx_ego_act[0, 1:].clone()
        _ctx_opp_act[0, :-1] = _ctx_opp_act[0, 1:].clone()
        _ctx_gs[0, :-1] = _ctx_gs[0, 1:].clone()
        _ctx_ctrl[0, :-1] = _ctx_ctrl[0, 1:].clone()
        _ctx_stage[0, -1] = frame["stage"]
        _ctx_ego_char[0, -1] = frame["ego_char"]
        _ctx_opp_char[0, -1] = frame["opp_char"]
        _ctx_ego_act[0, -1] = frame["ego_action"]
        _ctx_opp_act[0, -1] = frame["opp_action"]
        _ctx_gs[0, -1] = torch.from_numpy(frame["gamestate"])
        _ctx_ctrl[0, -1] = torch.from_numpy(frame["controller"])

    seq_idx = min(SEQ_LEN - 1, game_frame)
    with torch.no_grad():
        preds = model(_ctx_stage, _ctx_ego_char, _ctx_opp_char,
                       _ctx_ego_act, _ctx_opp_act, _ctx_gs, _ctx_ctrl)
    # Index the correct position (HAL uses seq_idx, not always -1)
    single_preds = {k: v[:, seq_idx:seq_idx+1] for k, v in preds.items()}

    decode_and_press(ego_ctrl, single_preds, gs=gs, temperature=args.temperature)
    game_frame += 1
    step += 1
