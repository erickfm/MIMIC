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

log.info("Loading HAL checkpoint: %s", args.checkpoint)
state_dict = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
# Strip "module." prefix from DDP
state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}

model = HALModel().to(DEVICE)
model.load_state_dict(state_dict)
model.eval()
log.info("Model loaded: %d params", sum(p.numel() for p in model.parameters()))

# ── Load HAL normalization + action mapping ──────────────────────────────────

hal_norm = load_hal_norm(Path(args.data_dir))
log.info("HAL normalization: %d features", len(hal_norm))

# HAL action mapping: same as MIMIC (melee.Action enum → dense index)
from mimic.cat_maps import ACTION_MAP, CHARACTER_MAP, STAGE_MAP

# HAL uses its own character/stage mappings — but they should be compatible
# HAL: IDX_BY_CHARACTER = {char: i for i, char in enumerate(char for char in Character if char.name in INCLUDED_CHARACTERS)}
# MIMIC: CHARACTER_MAP = {c.value: i for i, c in enumerate(Character)}
# These differ! HAL only includes 27 characters, MIMIC includes all.
# We need HAL's mapping.

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

# Actions: HAL uses all actions like MIMIC
HAL_ACTION_MAP = {a: i for i, a in enumerate(Action)}

# ── Preprocessing ────────────────────────────────────────────────────────────

COMBO_MAP = {
    (0,0,0,0,0): 0,  # NONE
    (1,0,0,0,0): 1,  # A
    (0,1,0,0,0): 2,  # B
    (0,0,1,0,0): 3,  # Jump
    (0,0,0,1,0): 4,  # Z
}
N_COMBOS = 5

SEQ_LEN = 256
_frame_cache = deque(maxlen=SEQ_LEN)
_prev_sent = None

BTN_ENUMS = [melee.enums.Button[name] for name in F.BTN]


def build_frame(gs, prev_sent):
    """Build a single frame dict for HAL's model."""
    players = list(gs.players.items())
    if len(players) < 2:
        return None

    port1, ps1 = players[0]
    port2, ps2 = players[1]

    # Stage
    stage_idx = HAL_STAGE_MAP.get(gs.stage, 0)

    # Per player: normalize with HAL transforms
    def player_features(ps, prefix, is_ego):
        char_idx = HAL_CHAR_MAP.get(ps.character, 0)
        action_idx = HAL_ACTION_MAP.get(ps.action, 0)

        # 9 numeric features (HAL's gamestate per player)
        raw = {
            "percent": float(ps.percent),
            "stock": ps.stock,
            "facing": float(ps.facing),
            "invulnerable": float(ps.invulnerable),
            "jumps_left": ps.jumps_left,
            "on_ground": float(ps.on_ground),
            "shield_strength": float(ps.shield_strength),
            "pos_x": float(ps.position.x),
            "pos_y": float(ps.position.y),
        }
        normed = []
        for feat in ["percent", "stock", "facing", "invulnerable", "jumps_left",
                      "on_ground", "shield_strength"]:
            if feat in hal_norm:
                normed.append(hal_normalize(raw[feat], hal_norm[feat]))
            else:
                normed.append(raw[feat])
        for feat in ["pos_x", "pos_y"]:
            if feat in hal_norm:
                normed.append(hal_normalize(raw[feat], hal_norm[feat]))
            else:
                normed.append(raw[feat])

        return char_idx, action_idx, normed

    ego_char, ego_action, ego_nums = player_features(ps1, "self", True)
    opp_char, opp_action, opp_nums = player_features(ps2, "opp", False)

    # Controller: from prev_sent (HAL's frame_offset=-1)
    if prev_sent is not None:
        mx = prev_sent["main_x"]
        my = prev_sent["main_y"]
        cx = prev_sent["c_x"]
        cy = prev_sent["c_y"]
        ls = prev_sent["l_shldr"]
        rs = prev_sent["r_shldr"]
        btns = {b: prev_sent.get(f"btn_{b}", 0) for b in
                ["BUTTON_A", "BUTTON_B", "BUTTON_X", "BUTTON_Y", "BUTTON_Z",
                 "BUTTON_L", "BUTTON_R"]}
    else:
        mx, my, cx, cy, ls, rs = 0.5, 0.5, 0.5, 0.5, 0.0, 0.0
        btns = {}

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


def decode_and_press(ctrl, preds, temperature=1.0):
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

    for btn in INCLUDED_BUTTONS_NO_SHOULDER:
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

    top3 = btn_probs.topk(min(3, len(btn_probs)))
    NAMES = ["A", "B", "Jump", "Z", "NONE"]
    top3_str = " ".join(f"{NAMES[i]}={v:.3f}" for v, i in zip(top3.values.tolist(), top3.indices.tolist()))
    log.info("MAIN=(%.2f,%.2f) C=(%.2f,%.2f) L=%.2f BTN=%s  top3=[%s]",
             mx, my, cx, cy, shldr, pressed, top3_str)


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
_was_in_game = False
while True:
    gs = console.step()
    if gs is None:
        continue
    if gs.menu_state not in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
        if _was_in_game:
            log.info("Game ended.")
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

    if len(_frame_cache) == 0:
        for _ in range(SEQ_LEN - 1):
            _frame_cache.append(frame)
    _frame_cache.append(frame)

    batch = stack_frames()
    with torch.no_grad():
        preds = model(
            batch["stage"], batch["ego_char"], batch["opp_char"],
            batch["ego_action"], batch["opp_action"],
            batch["gamestate"], batch["controller"],
        )

    decode_and_press(ego_ctrl, preds, temperature=args.temperature)
    step += 1
