#!/usr/bin/env python3
"""Run two HAL/MIMIC checkpoints head-to-head in Dolphin.

P1 (port 1) and P2 (port 2) are each driven by a model checkpoint.
Supports both HAL bare state_dicts and MIMIC wrapped checkpoints.

Usage:
    python tools/head_to_head.py \
      --p1-checkpoint /path/to/hal_original.pt \
      --p2-checkpoint checkpoints/hal-local_best.pt \
      --dolphin-path /path/to/dolphin-emu \
      --iso-path /path/to/melee.iso
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import logging
import math
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
    encode_controller_onehot_single,
)

log = logging.getLogger("h2h")
_h = logging.StreamHandler(sys.stderr)
_h.setFormatter(logging.Formatter("%(asctime)s  [%(levelname)s]  %(message)s"))
log.addHandler(_h)
log.setLevel(logging.INFO)

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Head-to-head: two models fight")
parser.add_argument("--p1-checkpoint", required=True)
parser.add_argument("--p2-checkpoint", required=True)
parser.add_argument("--p1-character", default="FOX")
parser.add_argument("--p2-character", default="FOX")
parser.add_argument("--stage", default="FINAL_DESTINATION")
parser.add_argument("--dolphin-path", required=True)
parser.add_argument("--iso-path", required=True)
parser.add_argument("--temperature", type=float, default=1.0)
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 256


# ── HAL Model (minimal reimplementation for HAL bare checkpoints) ───────────

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
            self.stage_emb(stage), self.character_emb(ego_char),
            self.character_emb(opp_char), self.action_emb(ego_action),
            self.action_emb(opp_action), gamestate, controller,
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


class _InferenceModel(nn.Module):
    """Wrapper for MIMIC FramePredictor checkpoints."""
    def __init__(self, fp):
        super().__init__()
        self.fp = fp

    def forward(self, stage, ego_char, opp_char, ego_action, opp_action, gamestate, controller):
        enc = self.fp.encoder
        combined = torch.cat([
            enc.stage_emb(stage),  # inference uses 0-based stage indices
            enc.char_emb(ego_char), enc.char_emb(opp_char),
            enc.action_emb(ego_action), enc.action_emb(opp_action),
            gamestate, controller,
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


# ── Model loader ─────────────────────────────────────────────────────────────

def load_model(path, device):
    raw = torch.load(path, map_location=device, weights_only=False)
    is_mimic = isinstance(raw, dict) and "model_state_dict" in raw
    if is_mimic:
        from mimic.model import FramePredictor, ModelConfig
        import dataclasses
        cfg = raw.get("config", {})
        valid_fields = {f.name for f in dataclasses.fields(ModelConfig)}
        mc = ModelConfig(**{k: v for k, v in cfg.items() if k in valid_fields})
        fp = FramePredictor(mc).to(device)
        sd = {k.removeprefix("_orig_mod."): v for k, v in raw["model_state_dict"].items()}
        fp.load_state_dict(sd)
        fp.eval()
        model = _InferenceModel(fp)
        desc = f"MIMIC({cfg.get('model_preset', '?')})"
        n_params = sum(p.numel() for p in fp.parameters())
    else:
        sd = {k.removeprefix("module."): v for k, v in raw.items()}
        model = HALModel().to(device)
        model.load_state_dict(sd)
        model.eval()
        desc = "HAL"
        n_params = sum(p.numel() for p in model.parameters())
    return model, desc, n_params


log.info("Loading P1: %s", args.p1_checkpoint)
model_p1, desc_p1, npar_p1 = load_model(args.p1_checkpoint, DEVICE)
log.info("  P1 = %s (%d params)", desc_p1, npar_p1)

log.info("Loading P2: %s", args.p2_checkpoint)
model_p2, desc_p2, npar_p2 = load_model(args.p2_checkpoint, DEVICE)
log.info("  P2 = %s (%d params)", desc_p2, npar_p2)


# ── HAL normalization stats ─────────────────────────────────────────────────

HAL_STATS_PATH = Path("/home/erick/projects/hal/checkpoints/stats.json")
with open(HAL_STATS_PATH) as _f:
    _raw_stats = json.load(_f)

class _Stats:
    def __init__(self, d):
        self.mean, self.std, self.min, self.max = d["mean"], d["std"], d["min"], d["max"]

HAL_P1_STATS = {k.removeprefix("p1_"): _Stats(_raw_stats[k])
                for k in _raw_stats if k.startswith("p1_")}
HAL_P2_STATS = {k.removeprefix("p2_"): _Stats(_raw_stats[k])
                for k in _raw_stats if k.startswith("p2_")}

def _hal_normalize(val, stats):
    return 2.0 * (val - stats.min) / (stats.max - stats.min) - 1.0

def _hal_invert_normalize(val, stats):
    return 2.0 * (stats.max - val) / (stats.max - stats.min) - 1.0

def _hal_standardize(val, stats):
    return (val - stats.mean) / stats.std

HAL_TRANSFORM = {
    "percent": _hal_normalize, "stock": _hal_normalize,
    "facing": _hal_normalize, "invulnerable": _hal_normalize,
    "jumps_left": _hal_normalize, "on_ground": _hal_normalize,
    "shield_strength": _hal_invert_normalize,
    "position_x": _hal_standardize, "position_y": _hal_standardize,
}

FEAT_ORDER = ["percent", "stock", "facing", "invulnerable", "jumps_left",
              "on_ground", "shield_strength", "position_x", "position_y"]

# ── Categorical mappings ────────────────────────────────────────────────────

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
HAL_ACTION_MAP = {a: i for i, a in enumerate(Action)}

COMBO_MAP = {
    (1, 0, 0, 0, 0): 0, (0, 1, 0, 0, 0): 1, (0, 0, 1, 0, 0): 2,
    (0, 0, 0, 1, 0): 3, (0, 0, 0, 0, 0): 4, (0, 0, 0, 0, 1): 4,
    (1, 0, 0, 0, 1): 0, (0, 1, 0, 0, 1): 1, (0, 0, 1, 0, 1): 2,
    (0, 0, 0, 1, 1): 3,
}
N_COMBOS = 5

HAL_STICK_37 = HAL_STICK_CLUSTERS_37
HAL_CSTICK_9 = HAL_CSTICK_CLUSTERS_9
HAL_SHOULDER_3 = HAL_SHOULDER_CLUSTERS_3
BTN_LIST = [melee.enums.Button.BUTTON_A, melee.enums.Button.BUTTON_B,
            melee.enums.Button.BUTTON_X, melee.enums.Button.BUTTON_Z]
ALL_BUTTONS = [
    melee.enums.Button.BUTTON_A, melee.enums.Button.BUTTON_B,
    melee.enums.Button.BUTTON_X, melee.enums.Button.BUTTON_Y,
    melee.enums.Button.BUTTON_Z, melee.enums.Button.BUTTON_L,
    melee.enums.Button.BUTTON_R,
]


# ── Per-player state ────────────────────────────────────────────────────────

class PlayerInference:
    """Holds model + context window for one player."""

    def __init__(self, model, name, port):
        self.model = model
        self.name = name
        self.port = port

        # Pre-compute mock fill values (HAL fills context with preprocessed ones)
        mock_ego = [HAL_TRANSFORM[f](1.0, HAL_P1_STATS[f]) for f in FEAT_ORDER]
        mock_opp = [HAL_TRANSFORM[f](1.0, HAL_P2_STATS[f]) for f in FEAT_ORDER]
        self.mock_gs = torch.tensor(mock_ego + mock_opp, dtype=torch.float32, device=DEVICE)

        mock_btns = {"BUTTON_A": 1, "BUTTON_B": 1, "BUTTON_X": 1, "BUTTON_Y": 1,
                     "BUTTON_Z": 1, "BUTTON_L": 1, "BUTTON_R": 1}
        self.mock_ctrl = torch.from_numpy(
            encode_controller_onehot_single(1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                            mock_btns, COMBO_MAP, N_COMBOS)
        ).to(dtype=torch.float32, device=DEVICE)

        # Context window
        self.ctx_stage = torch.ones(1, SEQ_LEN, dtype=torch.long, device=DEVICE)
        self.ctx_ego_char = torch.ones(1, SEQ_LEN, dtype=torch.long, device=DEVICE)
        self.ctx_opp_char = torch.ones(1, SEQ_LEN, dtype=torch.long, device=DEVICE)
        self.ctx_ego_act = torch.ones(1, SEQ_LEN, dtype=torch.long, device=DEVICE)
        self.ctx_opp_act = torch.ones(1, SEQ_LEN, dtype=torch.long, device=DEVICE)
        self.ctx_gs = self.mock_gs.unsqueeze(0).unsqueeze(0).expand(1, SEQ_LEN, -1).clone()
        self.ctx_ctrl = self.mock_ctrl.unsqueeze(0).unsqueeze(0).expand(1, SEQ_LEN, -1).clone()

    def build_frame(self, gs, ego_ps, opp_ps):
        """Build a frame from this player's perspective."""
        stage_idx = HAL_STAGE_MAP.get(gs.stage, 0)

        def feats(ps, is_ego):
            char_idx = HAL_CHAR_MAP.get(ps.character, 0)
            action_idx = HAL_ACTION_MAP.get(ps.action, 0)
            stats = HAL_P1_STATS if is_ego else HAL_P2_STATS
            normed = [HAL_TRANSFORM[f](getattr(ps, f) if f not in ("position_x", "position_y")
                       else (ps.position.x if f == "position_x" else ps.position.y),
                       stats[f]) for f in FEAT_ORDER]
            return char_idx, action_idx, normed

        ego_char, ego_action, ego_nums = feats(ego_ps, True)
        opp_char, opp_action, opp_nums = feats(opp_ps, False)

        # Controller from game engine readback (ego's controller_state)
        cs = ego_ps.controller_state
        btns = {f"BUTTON_{b}": int(cs.button.get(melee.enums.Button[f"BUTTON_{b}"], False))
                for b in ["A", "B", "X", "Y", "Z", "L", "R"]}
        controller = encode_controller_onehot_single(
            float(cs.main_stick[0]), float(cs.main_stick[1]),
            float(cs.c_stick[0]), float(cs.c_stick[1]),
            float(cs.l_shoulder), float(cs.r_shoulder),
            btns, COMBO_MAP, N_COMBOS)

        return {
            "stage": stage_idx,
            "ego_char": ego_char, "opp_char": opp_char,
            "ego_action": ego_action, "opp_action": opp_action,
            "gamestate": np.array(ego_nums + opp_nums, dtype=np.float32),
            "controller": controller,
        }

    def update_context(self, frame, game_frame):
        """Push a frame into the context window."""
        if game_frame < SEQ_LEN:
            idx = game_frame
            self.ctx_stage[0, idx] = frame["stage"]
            self.ctx_ego_char[0, idx] = frame["ego_char"]
            self.ctx_opp_char[0, idx] = frame["opp_char"]
            self.ctx_ego_act[0, idx] = frame["ego_action"]
            self.ctx_opp_act[0, idx] = frame["opp_action"]
            self.ctx_gs[0, idx] = torch.from_numpy(frame["gamestate"])
            self.ctx_ctrl[0, idx] = torch.from_numpy(frame["controller"])
        else:
            for t in (self.ctx_stage, self.ctx_ego_char, self.ctx_opp_char,
                      self.ctx_ego_act, self.ctx_opp_act):
                t[0, :-1] = t[0, 1:].clone()
            self.ctx_gs[0, :-1] = self.ctx_gs[0, 1:].clone()
            self.ctx_ctrl[0, :-1] = self.ctx_ctrl[0, 1:].clone()
            self.ctx_stage[0, -1] = frame["stage"]
            self.ctx_ego_char[0, -1] = frame["ego_char"]
            self.ctx_opp_char[0, -1] = frame["opp_char"]
            self.ctx_ego_act[0, -1] = frame["ego_action"]
            self.ctx_opp_act[0, -1] = frame["opp_action"]
            self.ctx_gs[0, -1] = torch.from_numpy(frame["gamestate"])
            self.ctx_ctrl[0, -1] = torch.from_numpy(frame["controller"])

    def predict(self, game_frame, temperature=1.0):
        """Run model and return decoded actions."""
        seq_idx = min(SEQ_LEN - 1, game_frame)
        with torch.no_grad():
            preds = self.model(self.ctx_stage, self.ctx_ego_char, self.ctx_opp_char,
                               self.ctx_ego_act, self.ctx_opp_act, self.ctx_gs, self.ctx_ctrl)
        return {k: v[:, seq_idx:seq_idx+1] for k, v in preds.items()}


def decode_and_press(ctrl, preds, temperature=1.0):
    """Decode model predictions and press controller."""
    main_probs = Fn.softmax(preds["main_stick"][0, -1].float() / temperature, dim=-1)
    main_idx = int(torch.multinomial(main_probs, 1))
    mx, my = float(HAL_STICK_37[main_idx][0]), float(HAL_STICK_37[main_idx][1])

    c_probs = Fn.softmax(preds["c_stick"][0, -1].float() / temperature, dim=-1)
    c_idx = int(torch.multinomial(c_probs, 1))
    cx, cy = float(HAL_CSTICK_9[c_idx][0]), float(HAL_CSTICK_9[c_idx][1])

    s_probs = Fn.softmax(preds["shoulder"][0, -1].float() / temperature, dim=-1)
    s_idx = int(torch.multinomial(s_probs, 1))
    shldr = float(HAL_SHOULDER_3[s_idx])

    btn_probs = Fn.softmax(preds["buttons"][0, -1].float() / temperature, dim=-1)
    btn_idx = int(torch.multinomial(btn_probs, 1))

    ctrl.tilt_analog(melee.enums.Button.BUTTON_MAIN, mx, my)
    ctrl.tilt_analog(melee.enums.Button.BUTTON_C, cx, cy)
    ctrl.press_shoulder(melee.enums.Button.BUTTON_L, shldr)

    for btn in ALL_BUTTONS:
        ctrl.release_button(btn)
    if btn_idx < 4:
        ctrl.press_button(BTN_LIST[btn_idx])

    ctrl.flush()


# ── Dolphin setup ───────────────────────────────────────────────────────────

console = melee.Console(
    path=args.dolphin_path,
    is_dolphin=True,
    tmp_home_directory=True,
    copy_home_directory=False,
    blocking_input=True,
    online_delay=0,
    setup_gecko_codes=True,
    fullscreen=False,
    gfx_backend="",
    disable_audio=False,
    use_exi_inputs=False,
    enable_ffw=False,
)
ctrl_p1 = melee.Controller(console=console, port=1, type=melee.ControllerType.STANDARD)
ctrl_p2 = melee.Controller(console=console, port=2, type=melee.ControllerType.STANDARD)
console.run(iso_path=args.iso_path)
console.connect()
ctrl_p1.connect()
ctrl_p2.connect()
log.info("Connected to Dolphin")

P1_CHAR = melee.Character[args.p1_character]
P2_CHAR = melee.Character[args.p2_character]
STAGE = melee.Stage[args.stage]

menu_p1 = melee.MenuHelper()
menu_p2 = melee.MenuHelper()

player1 = PlayerInference(model_p1, desc_p1, 1)
player2 = PlayerInference(model_p2, desc_p2, 2)

log.info("P1 (port 1): %s — %s", desc_p1, args.p1_checkpoint)
log.info("P2 (port 2): %s — %s", desc_p2, args.p2_checkpoint)

# ── Main loop ───────────────────────────────────────────────────────────────

game_frame = 0
_was_in_game = False
_last_stocks = None

def _print_summary():
    if _last_stocks:
        s1, s2 = _last_stocks
        log.info("=" * 60)
        log.info("GAME OVER — %d frames (%.1fs)", game_frame, game_frame / 60.0)
        log.info("  P1 (%s): %d stocks", desc_p1, s1)
        log.info("  P2 (%s): %d stocks", desc_p2, s2)
        if s1 > s2: log.info("  Result: P1 (%s) WINS", desc_p1)
        elif s2 > s1: log.info("  Result: P2 (%s) WINS", desc_p2)
        else: log.info("  Result: DRAW")
        log.info("=" * 60)

def _shutdown(*a):
    _print_summary()
    ctrl_p1.disconnect()
    ctrl_p2.disconnect()
    console.stop()
    sys.exit(0)

import atexit
atexit.register(_print_summary)
signal.signal(signal.SIGINT, _shutdown)

while True:
    gs = console.step()
    if gs is None:
        continue

    if gs.menu_state not in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
        if _was_in_game:
            _shutdown()
        menu_p1.menu_helper_simple(gs, ctrl_p1, P1_CHAR, STAGE,
                                    cpu_level=0, autostart=False)
        menu_p2.menu_helper_simple(gs, ctrl_p2, P2_CHAR, STAGE,
                                    cpu_level=0, autostart=True)
        ctrl_p1.flush()
        ctrl_p2.flush()
        continue

    _was_in_game = True
    players = sorted(gs.players.items())
    if len(players) < 2:
        continue

    port1, ps1 = players[0]
    port2, ps2 = players[1]

    # Build frames from each perspective
    frame_p1 = player1.build_frame(gs, ego_ps=ps1, opp_ps=ps2)
    frame_p2 = player2.build_frame(gs, ego_ps=ps2, opp_ps=ps1)

    player1.update_context(frame_p1, game_frame)
    player2.update_context(frame_p2, game_frame)

    preds_p1 = player1.predict(game_frame, args.temperature)
    preds_p2 = player2.predict(game_frame, args.temperature)

    decode_and_press(ctrl_p1, preds_p1, args.temperature)
    decode_and_press(ctrl_p2, preds_p2, args.temperature)

    _last_stocks = (ps1.stock, ps2.stock)

    if game_frame % 60 == 0:
        log.info("[f%d]  P1(%s) %dstk %.0f%%  |  P2(%s) %dstk %.0f%%",
                 game_frame, desc_p1, ps1.stock, ps1.percent,
                 desc_p2, ps2.stock, ps2.percent)

    game_frame += 1
