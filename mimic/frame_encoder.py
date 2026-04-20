# frame_encoder.py – intra-frame encoder variants for MIMIC
# -----------------------------------------------------------------------------
# Converts structured Melee frame dictionaries -> (B, T, d_model) tensor.
#
# Encoder variants (selected via --encoder CLI flag):
#   "default"     – N individual tokens + intra-frame self-attention (original)
#   "flat"        – concat all tokens, 2-layer MLP to d_model (no attention)
#   "composite8"  – 7-8 semantic group tokens + intra-frame self-attention
#   "hybrid16"    – 15-16 entity-level tokens + intra-frame self-attention
# -----------------------------------------------------------------------------

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mlp(in_dim: int, out_dim: int, dropout: float = 0.0) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, out_dim),
        nn.GELU(),
        nn.Dropout(dropout),
    )


class _GroupAttention(nn.Module):
    """Self-attention across G feature-group tokens inside one frame."""

    def __init__(self, d_intra: int = 256, nhead: int = 4, nlayers: int = 2,
                 k_query: int = 1, dropout: float = 0.0):
        super().__init__()
        if nlayers == 0:
            self.encoder = None
        else:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_intra, nhead=nhead, dim_feedforward=4 * d_intra,
                dropout=dropout, batch_first=True, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, nlayers, norm=nn.LayerNorm(d_intra))

        self.k_query = k_query
        self.queries = nn.Parameter(torch.randn(k_query, d_intra) * 0.02)

    def forward(self, group_tokens: torch.Tensor) -> torch.Tensor:
        """(B*T, G, d_intra) -> (B*T, d_intra)"""
        BxT = group_tokens.size(0)
        k = self.k_query
        q = self.queries.unsqueeze(0).expand(BxT, -1, -1)
        x = torch.cat([q, group_tokens], dim=1)
        if self.encoder is not None:
            x = self.encoder(x)
        if k == 1:
            return x[:, 0]
        return x[:, :k].mean(dim=1)


# ---------------------------------------------------------------------------
# Base class: shared embedding tables and numeric encoders
# ---------------------------------------------------------------------------

class _BaseFrameEncoder(nn.Module):
    GLOBAL_NUM   = 20
    PLAYER_NUM   = 22
    PLAYER_NUM_MINIMAL = 7  # minimal mode: only 7 numeric per player
    PLAYER_NUM_HAL_MINIMAL = PLAYER_NUM_MINIMAL  # legacy alias
    NANA_NUM     = 27
    ANALOG_DIM   = 4
    BTN_DIM      = 12
    FLAGS_DIM    = 5
    NANA_FLAGS   = 6
    PROJ_NUM_PER = 5
    PROJ_SLOTS   = 8

    def __init__(
        self,
        *,
        num_stages: int,
        num_ports: int,
        num_characters: int,
        num_actions: int,
        num_costumes: int,
        num_proj_types: int,
        num_proj_subtypes: int,
        num_c_dirs: int = 5,
        d_intra: int = 256,
        dropout: float = 0.0,
        scaled_emb: bool = False,
        no_opp_inputs: bool = False,
        no_self_inputs: bool = False,
        lean_features: bool = False,
        mimic_minimal_features: bool = False,
        mimic_controller_encoding: bool = False,
        n_controller_combos: int = 5,
        # Legacy kwarg aliases
        hal_minimal_features: bool = None,
        hal_controller_encoding: bool = None,
    ) -> None:
        super().__init__()
        # Accept legacy kwargs
        if hal_minimal_features is not None:
            mimic_minimal_features = hal_minimal_features
        if hal_controller_encoding is not None:
            mimic_controller_encoding = hal_controller_encoding
        self._d_intra = d_intra
        self._dropout = dropout
        self._lean = lean_features
        self._minimal = mimic_minimal_features
        self._no_opp_inputs = no_opp_inputs
        self._no_self_inputs = no_self_inputs
        self._ctrl_enc = mimic_controller_encoding
        # Legacy aliases (read-only by the old callers)
        self._hal_minimal = self._minimal
        self._hal_ctrl_enc = self._ctrl_enc
        self.si_drop_prob: float = 0.0

        def _emb_dim(n_vocab: int) -> int:
            if scaled_emb:
                return max(16, int(n_vocab ** 0.25 * 16))
            return d_intra

        def cat_block(n_vocab: int) -> nn.Module:
            edim = _emb_dim(n_vocab)
            layers: list[nn.Module] = [nn.Embedding(n_vocab, edim)]
            if edim != d_intra:
                layers.append(nn.Linear(edim, d_intra))
            layers.append(nn.Dropout(dropout))
            return nn.Sequential(*layers)

        self.stage_emb = cat_block(num_stages)
        self.port_emb  = cat_block(num_ports)
        self.char_emb  = cat_block(num_characters)
        self.act_emb   = cat_block(num_actions)
        self.cost_emb  = cat_block(num_costumes)
        self.cdir_emb  = cat_block(num_c_dirs)
        self.ptype_emb = cat_block(num_proj_types)
        self.psub_emb  = cat_block(num_proj_subtypes)

        _player_num = self.PLAYER_NUM_MINIMAL if mimic_minimal_features else self.PLAYER_NUM
        # minimal: 7 numeric + 3 flags (on_ground, facing, invulnerable) = 10
        # normal: 22 numeric + 1 action_elapsed = 23
        _player_enc_in = _player_num + 3 if mimic_minimal_features else _player_num + 1
        self.glob_enc      = _mlp(self.GLOBAL_NUM, d_intra, dropout)
        self.player_enc    = _mlp(_player_enc_in, d_intra, dropout)
        self.nana_enc      = _mlp(self.NANA_NUM + 1, d_intra, dropout)
        self.analog_enc    = _mlp(self.ANALOG_DIM, d_intra, dropout)
        self.proj_num_enc  = _mlp(self.PROJ_NUM_PER * self.PROJ_SLOTS, d_intra, dropout)

        has_self_btn = not no_self_inputs
        has_opp_btn = not no_opp_inputs
        if hal_controller_encoding and has_self_btn:
            # HAL-style: single MLP for concatenated one-hot controller vector
            controller_dim = 37 + 9 + n_controller_combos + 3  # stick + cstick + combos + shoulder
            self.controller_enc = _mlp(controller_dim, d_intra, dropout)
            # With no_opp_inputs=True (standard), no separate btn_enc needed.
            # With no_opp_inputs=False, opp buttons still need the old encoder.
            if has_opp_btn:
                self.btn_enc = _mlp(self.BTN_DIM, d_intra, dropout)
            btn_parts = 0  # for nana logic below
        else:
            btn_parts = int(has_self_btn) + int(has_opp_btn)
            if btn_parts > 0:
                self.btn_enc = _mlp(self.BTN_DIM * btn_parts, d_intra, dropout)
        nana_btn_parts = int(has_self_btn) + int(has_opp_btn)
        if nana_btn_parts > 0:
            self.nana_btn_enc = _mlp(self.BTN_DIM * nana_btn_parts, d_intra, dropout)
        self.flag_enc      = _mlp(self.FLAGS_DIM * 2, d_intra, dropout)
        self.nana_flag_enc = _mlp(self.NANA_FLAGS * 2, d_intra, dropout)

    @property
    def n_raw_tokens(self) -> int:
        # base=55 (all inputs), -4 if no_opp_inputs, -5 if no_self_inputs
        n = 55
        if self._no_opp_inputs:
            n -= 4  # opp_c_dir, opp_analog, opp_nana_c_dir, opp_nana_analog
        if self._no_self_inputs:
            n -= 5  # self_c_dir, self_analog, self_nana_c_dir, self_nana_analog, (buttons+nana_buttons counted below)
        # buttons/nana_buttons exist only if at least one side has inputs
        has_self_btn = not self._no_self_inputs
        has_opp_btn = not self._no_opp_inputs
        if not has_self_btn and not has_opp_btn:
            n -= 2  # buttons, nana_buttons tokens gone entirely
        if self._hal_minimal:
            n -= 1  # drop global_num token
            n -= 4  # drop self_port, opp_port, self_costume, opp_costume
            n -= 1  # drop flags token (folded into player_num)
        if self._lean:
            # Drop: nana chars/actions (4), nana c_dirs (up to 2), proj categoricals (24),
            # nana numerics (2), nana analogs (up to 2), proj_num (1), nana_buttons (1), nana_flags (1)
            n -= 4 + 24 + 2 + 1 + 1 + 1  # = 33 always dropped
            if not self._no_self_inputs:
                n -= 1  # self_nana_c_dir
                n -= 1  # self_nana_analog
            if not self._no_opp_inputs:
                n -= 1  # opp_nana_c_dir
                n -= 1  # opp_nana_analog
            # nana_buttons: only counted if buttons exist
            if has_self_btn or has_opp_btn:
                pass  # already subtracted nana_buttons above (-1 in the 33)
        if self._hal_ctrl_enc and not self._no_self_inputs:
            # Replaces 3 tokens (self_analog, self_c_dir, buttons) with 1 (self_controller)
            # Net: -2 tokens.  If no_opp_inputs=True, buttons token was self-only.
            # If no_opp_inputs=False, buttons token remains for opp, so only -2 (analog + c_dir - controller).
            n -= 2
        return n

    def _build_raw_tokens(self, seq: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Build named feature-group tokens, each (B,T,d_intra)."""
        t: Dict[str, torch.Tensor] = {}
        noi = self._no_opp_inputs
        nsi = self._no_self_inputs

        # -- Curriculum: stochastic self-input masking --
        if not nsi and self.si_drop_prob > 0:
            seq = dict(seq)  # shallow copy to avoid mutating caller's dict
            _ref = seq.get("self_controller") if self._hal_ctrl_enc else seq.get("self_analog")
            if _ref is not None:
                B = _ref.shape[0]
                device = _ref.device
            else:
                B = seq["self_numeric"].shape[0]
                device = seq["self_numeric"].device
            if self.training and self.si_drop_prob < 1.0:
                keep = (torch.rand(B, 1, 1, device=device) >= self.si_drop_prob).float()
            else:
                keep = torch.zeros(B, 1, 1, device=device)
            keep_idx = keep.squeeze(-1).long()  # (B, 1) for index tensors
            if self._hal_ctrl_enc and "self_controller" in seq:
                seq["self_controller"] = seq["self_controller"] * keep
            else:
                seq["self_analog"] = seq["self_analog"] * keep
                seq["self_c_dir"] = seq["self_c_dir"] * keep_idx
                seq["self_buttons"] = seq["self_buttons"] * keep_idx.unsqueeze(-1)
            if "self_nana_analog" in seq:
                seq["self_nana_analog"] = seq["self_nana_analog"] * keep
            if "self_nana_c_dir" in seq:
                seq["self_nana_c_dir"] = seq["self_nana_c_dir"] * keep_idx
            if "self_nana_buttons" in seq:
                seq["self_nana_buttons"] = seq["self_nana_buttons"] * keep_idx.unsqueeze(-1)

        t["stage"]        = self.stage_emb(seq["stage"])
        if not self._hal_minimal:
            t["self_port"]    = self.port_emb(seq["self_port"])
            t["opp_port"]     = self.port_emb(seq["opp_port"])
            t["self_costume"] = self.cost_emb(seq["self_costume"])
            t["opp_costume"]  = self.cost_emb(seq["opp_costume"])
        t["self_char"]    = self.char_emb(seq["self_character"])
        t["opp_char"]     = self.char_emb(seq["opp_character"])
        t["self_action"]  = self.act_emb(seq["self_action"])
        t["opp_action"]   = self.act_emb(seq["opp_action"])
        if not nsi and not self._hal_ctrl_enc:
            t["self_c_dir"] = self.cdir_emb(seq["self_c_dir"])
        if not noi:
            t["opp_c_dir"] = self.cdir_emb(seq["opp_c_dir"])

        if not self._lean:
            # Full feature set: nana, projectiles, all numerics
            t["self_nana_char"]   = self.char_emb(seq["self_nana_character"])
            t["opp_nana_char"]    = self.char_emb(seq["opp_nana_character"])
            t["self_nana_action"] = self.act_emb(seq["self_nana_action"])
            t["opp_nana_action"]  = self.act_emb(seq["opp_nana_action"])
            if not nsi:
                t["self_nana_c_dir"] = self.cdir_emb(seq["self_nana_c_dir"])
            if not noi:
                t["opp_nana_c_dir"] = self.cdir_emb(seq["opp_nana_c_dir"])

            for j in range(self.PROJ_SLOTS):
                t[f"proj{j}_owner"]   = self.port_emb(seq[f"proj{j}_owner"])
                t[f"proj{j}_type"]    = self.ptype_emb(seq[f"proj{j}_type"])
                t[f"proj{j}_subtype"] = self.psub_emb(seq[f"proj{j}_subtype"])

        if self._hal_minimal:
            # HAL uses 7 numeric + 3 flags = 10 per player
            _HAL_IDX = [0, 1, 2, 3, 4, 12, 13]  # pos_x, pos_y, pct, stock, jumps, invuln, shield
            _HAL_FLAG_IDX = [0, 2, 3]  # on_ground, facing, invulnerable
            # Select from 22-col data (training) or use as-is if already 7-col (inference)
            sn = seq["self_numeric"]
            on = seq["opp_numeric"]
            if sn.shape[-1] > 7:
                sn = sn[..., _HAL_IDX]
                on = on[..., _HAL_IDX]
            self_num = torch.cat([
                sn,
                seq["self_flags"][..., _HAL_FLAG_IDX].float()
            ], dim=-1)  # (B, T, 10)
            opp_num = torch.cat([
                on,
                seq["opp_flags"][..., _HAL_FLAG_IDX].float()
            ], dim=-1)  # (B, T, 10)
            t["self_player_num"] = self.player_enc(self_num)
            t["opp_player_num"]  = self.player_enc(opp_num)
        else:
            t["global_num"]      = self.glob_enc(seq["numeric"])
            t["self_player_num"] = self.player_enc(torch.cat([seq["self_numeric"], seq["self_action_elapsed"].unsqueeze(-1).float()], dim=-1))
            t["opp_player_num"]  = self.player_enc(torch.cat([seq["opp_numeric"], seq["opp_action_elapsed"].unsqueeze(-1).float()], dim=-1))

        if not self._lean:
            t["self_nana_num"]   = self.nana_enc(torch.cat([seq["self_nana_numeric"], seq["self_nana_action_elapsed"].unsqueeze(-1).float()], dim=-1))
            t["opp_nana_num"]    = self.nana_enc(torch.cat([seq["opp_nana_numeric"], seq["opp_nana_action_elapsed"].unsqueeze(-1).float()], dim=-1))

        if not nsi:
            if self._hal_ctrl_enc:
                t["self_controller"] = self.controller_enc(seq["self_controller"])
            else:
                t["self_analog"] = self.analog_enc(seq["self_analog"])
            if not self._lean:
                t["self_nana_analog"] = self.analog_enc(seq["self_nana_analog"])
        if not noi:
            t["opp_analog"]      = self.analog_enc(seq["opp_analog"])
            if not self._lean:
                t["opp_nana_analog"] = self.analog_enc(seq["opp_nana_analog"])

        if not self._lean:
            t["proj_num"] = self.proj_num_enc(torch.cat([seq[f"{k}_numeric"] for k in map(str, range(self.PROJ_SLOTS))], dim=-1))

        # Buttons: combine whichever sides are present
        has_self_btn = not nsi
        has_opp_btn = not noi
        if self._hal_ctrl_enc and has_self_btn:
            # Self buttons are inside self_controller; only opp needs separate encoding
            if has_opp_btn:
                t["buttons"] = self.btn_enc(seq["opp_buttons"].float())
            # else: no separate buttons token (self is in controller, opp is excluded)
            if not self._lean:
                t["nana_buttons"] = self.nana_btn_enc(seq["self_nana_buttons"].float())
        elif has_self_btn and has_opp_btn:
            t["buttons"]      = self.btn_enc(torch.cat([seq["self_buttons"].float(), seq["opp_buttons"].float()], dim=-1))
            if not self._lean:
                t["nana_buttons"] = self.nana_btn_enc(torch.cat([seq["self_nana_buttons"].float(), seq["opp_nana_buttons"].float()], dim=-1))
        elif has_self_btn:
            t["buttons"]      = self.btn_enc(seq["self_buttons"].float())
            if not self._lean:
                t["nana_buttons"] = self.nana_btn_enc(seq["self_nana_buttons"].float())
        elif has_opp_btn:
            t["buttons"]      = self.btn_enc(seq["opp_buttons"].float())
            if not self._lean:
                t["nana_buttons"] = self.nana_btn_enc(seq["opp_nana_buttons"].float())
        # else: no button tokens at all

        if not self._hal_minimal:
            t["flags"]      = self.flag_enc(torch.cat([seq["self_flags"].float(), seq["opp_flags"].float()], dim=-1))
        if not self._lean:
            t["nana_flags"] = self.nana_flag_enc(torch.cat([seq["self_nana_flags"].float(), seq["opp_nana_flags"].float()], dim=-1))

        return t

    @staticmethod
    def _collapse(x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape[:2]
        return x.reshape(B * T, *x.shape[2:])


# ---------------------------------------------------------------------------
# Original encoder: N tokens + intra-frame attention
# ---------------------------------------------------------------------------

class FrameEncoder(_BaseFrameEncoder):
    """Variable-count token intra-frame attention encoder."""

    def __init__(
        self,
        *,
        d_model: int = 1024,
        d_intra: int = 256,
        dropout: float = 0.0,
        nlayers: int = 2,
        k_query: int = 1,
        scaled_emb: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(d_intra=d_intra, dropout=dropout, scaled_emb=scaled_emb, **kwargs)
        nhead = max(1, d_intra // 64)
        self.set_attn = _GroupAttention(d_intra=d_intra, nhead=nhead, nlayers=nlayers,
                                        k_query=k_query, dropout=dropout)
        self.to_model = nn.Sequential(
            nn.LayerNorm(d_intra),
            nn.Linear(d_intra, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self._d_model = d_model

    def forward(self, seq: Dict[str, torch.Tensor]) -> torch.Tensor:
        B, T = seq["stage"].shape
        raw = self._build_raw_tokens(seq)
        tokens = list(raw.values())
        group_stack = torch.stack(tokens, dim=2)
        group_flat = self._collapse(group_stack)
        pooled_flat = self.set_attn(group_flat)
        pooled = pooled_flat.view(B, T, self._d_intra)
        return self.to_model(pooled)


# ---------------------------------------------------------------------------
# Flat encoder: concat all tokens, 2-layer MLP (no intra-frame attention)
# ---------------------------------------------------------------------------

class FlatFrameEncoder(_BaseFrameEncoder):
    """Concatenate all token representations and project via MLP. No attention."""

    def __init__(
        self,
        *,
        d_model: int = 1024,
        d_intra: int = 256,
        dropout: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(d_intra=d_intra, dropout=dropout, **kwargs)
        cat_dim = self.n_raw_tokens * d_intra
        self.proj = nn.Sequential(
            nn.LayerNorm(cat_dim),
            nn.Linear(cat_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, seq: Dict[str, torch.Tensor]) -> torch.Tensor:
        raw = self._build_raw_tokens(seq)
        tokens = list(raw.values())
        cat = torch.cat(tokens, dim=-1)
        return self.proj(cat)


# ---------------------------------------------------------------------------
# Composite encoder: 7-8 semantic group tokens + intra-frame attention
# ---------------------------------------------------------------------------

class _CompositeTokenMixer(nn.Module):
    """Mix N sub-tokens into 1 composite token: sum + LN + Linear."""
    def __init__(self, d_intra: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_intra)
        self.proj = nn.Sequential(
            nn.Linear(d_intra, d_intra),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: List[torch.Tensor]) -> torch.Tensor:
        x = torch.stack(tokens, dim=0).sum(dim=0)
        return self.proj(self.norm(x))


class CompositeFrameEncoder(_BaseFrameEncoder):
    """7-8 semantic composite tokens + intra-frame attention.

    When no_opp_inputs=False (default, 8 tokens):
      0: GAME_STATE  (stage, global numerics)
      1: SELF_INPUT  (self analog, buttons)
      2: OPP_INPUT   (opp analog, buttons)
      3: SELF_STATE  (self port/char/act/cost/cdir, self numeric, flags)
      4: OPP_STATE   (opp port/char/act/cost/cdir, opp numeric, flags)
      5: NANA_SELF   (all self nana features)
      6: NANA_OPP    (all opp nana features)
      7: PROJECTILES (all proj cats + proj numerics)

    When no_opp_inputs=True (7 tokens): OPP_INPUT removed, OPP_STATE/NANA_OPP
    lose opp controller input tokens.

    When no_self_inputs=True: SELF_INPUT also removed, SELF_STATE/NANA_SELF
    lose self controller input tokens.
    """

    def __init__(
        self,
        *,
        d_model: int = 1024,
        d_intra: int = 256,
        dropout: float = 0.0,
        nlayers: int = 2,
        k_query: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(d_intra=d_intra, dropout=dropout, **kwargs)
        n_tokens = 8
        if self._no_opp_inputs:
            n_tokens -= 1
        if self._no_self_inputs:
            n_tokens -= 1
        self.mixers = nn.ModuleList([_CompositeTokenMixer(d_intra, dropout) for _ in range(n_tokens)])
        nhead = max(1, d_intra // 64)
        self.set_attn = _GroupAttention(d_intra=d_intra, nhead=nhead, nlayers=nlayers,
                                        k_query=k_query, dropout=dropout)
        self.to_model = nn.Sequential(
            nn.LayerNorm(d_intra),
            nn.Linear(d_intra, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self._d_model = d_model

    def forward(self, seq: Dict[str, torch.Tensor]) -> torch.Tensor:
        B, T = seq["stage"].shape
        t = self._build_raw_tokens(seq)
        noi = self._no_opp_inputs
        nsi = self._no_self_inputs

        proj_cats = []
        for j in range(self.PROJ_SLOTS):
            proj_cats.extend([t[f"proj{j}_owner"], t[f"proj{j}_type"], t[f"proj{j}_subtype"]])

        opp_state = [t["opp_port"], t["opp_char"], t["opp_action"], t["opp_costume"], t["opp_player_num"], t["flags"]]
        if not noi:
            opp_state.insert(4, t["opp_c_dir"])

        has_btn_token = "buttons" in t
        nana_opp = [t["opp_nana_char"], t["opp_nana_action"], t["opp_nana_num"]]
        if has_btn_token:
            nana_opp.append(t["nana_buttons"])
        nana_opp.append(t["nana_flags"])
        if not noi:
            nana_opp.insert(2, t["opp_nana_c_dir"])
            if "opp_nana_analog" in t:
                nana_opp.insert(-1, t["opp_nana_analog"])

        self_state = [t["self_port"], t["self_char"], t["self_action"], t["self_costume"], t["self_player_num"], t["flags"]]
        if not nsi:
            self_state.insert(4, t["self_c_dir"])

        nana_self = [t["self_nana_char"], t["self_nana_action"], t["self_nana_num"]]
        if not nsi:
            nana_self.insert(2, t["self_nana_c_dir"])
            nana_self.append(t["self_nana_analog"])
        if has_btn_token:
            nana_self.append(t["nana_buttons"])
        nana_self.append(t["nana_flags"])

        groups = [
            [t["stage"], t["global_num"]],                                       # GAME_STATE
        ]
        if not nsi and has_btn_token:
            groups.append([t["self_analog"], t["buttons"]])                      # SELF_INPUT
        if not noi and has_btn_token:
            groups.append([t["opp_analog"], t["buttons"]])                       # OPP_INPUT
        groups.extend([
            self_state,                                                          # SELF_STATE
            opp_state,                                                           # OPP_STATE
            nana_self,                                                           # NANA_SELF
            nana_opp,                                                            # NANA_OPP
            proj_cats + [t["proj_num"]],                                         # PROJECTILES
        ])

        composite = [self.mixers[i](g) for i, g in enumerate(groups)]
        stack = torch.stack(composite, dim=2)
        flat = self._collapse(stack)
        pooled = self.set_attn(flat)
        pooled = pooled.view(B, T, self._d_intra)
        return self.to_model(pooled)


# ---------------------------------------------------------------------------
# Hybrid encoder: 15-16 entity-level tokens + intra-frame attention
# ---------------------------------------------------------------------------

class HybridFrameEncoder(_BaseFrameEncoder):
    """15-16 tokens (entity x feature-type) + intra-frame attention.

    When no_opp_inputs=False (default, 16 tokens):
      0: GAME         (stage, global numerics)
      1: SELF_IDENT   (self port, char, costume)
      2: SELF_ACTION  (self action, cdir)
      3: SELF_STATE   (self numeric, flags)
      4: SELF_INPUT   (self analog, buttons)
      5: OPP_IDENT    (opp port, char, costume)
      6: OPP_ACTION   (opp action, cdir)
      7: OPP_STATE    (opp numeric, flags)
      8: OPP_INPUT    (opp analog, buttons)
      9: NANA_SELF_ID  (self nana char, action, cdir)
      10: NANA_SELF_ST (self nana numeric, analog, buttons, flags)
      11: NANA_OPP_ID  (opp nana char, action, cdir)
      12: NANA_OPP_ST  (opp nana numeric, analog, buttons, flags)
      13-15: PROJ_A/B/C (projectile slots split into 3 groups)

    When no_opp_inputs=True (15 tokens): OPP_INPUT removed, OPP_ACTION/
    NANA_OPP_ID/NANA_OPP_ST lose opp controller input tokens.

    When no_self_inputs=True: SELF_INPUT also removed, SELF_ACTION/
    NANA_SELF_ID/NANA_SELF_ST lose self controller input tokens.
    """

    def __init__(
        self,
        *,
        d_model: int = 1024,
        d_intra: int = 256,
        dropout: float = 0.0,
        nlayers: int = 2,
        k_query: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(d_intra=d_intra, dropout=dropout, **kwargs)
        n_tokens = 16
        if self._no_opp_inputs:
            n_tokens -= 1
        if self._no_self_inputs:
            n_tokens -= 1
        self.mixers = nn.ModuleList([_CompositeTokenMixer(d_intra, dropout) for _ in range(n_tokens)])
        nhead = max(1, d_intra // 64)
        self.set_attn = _GroupAttention(d_intra=d_intra, nhead=nhead, nlayers=nlayers,
                                        k_query=k_query, dropout=dropout)
        self.to_model = nn.Sequential(
            nn.LayerNorm(d_intra),
            nn.Linear(d_intra, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self._d_model = d_model

    def forward(self, seq: Dict[str, torch.Tensor]) -> torch.Tensor:
        B, T = seq["stage"].shape
        t = self._build_raw_tokens(seq)
        noi = self._no_opp_inputs
        nsi = self._no_self_inputs
        has_btn_token = "buttons" in t

        opp_action_group = [t["opp_action"]] if noi else [t["opp_action"], t["opp_c_dir"]]

        nana_opp_id = [t["opp_nana_char"], t["opp_nana_action"]]
        if not noi:
            nana_opp_id.append(t["opp_nana_c_dir"])

        nana_opp_st = [t["opp_nana_num"]]
        if not noi and "opp_nana_analog" in t:
            nana_opp_st.append(t["opp_nana_analog"])
        if has_btn_token:
            nana_opp_st.append(t["nana_buttons"])
        nana_opp_st.append(t["nana_flags"])

        self_action_group = [t["self_action"]] if nsi else [t["self_action"], t["self_c_dir"]]

        nana_self_id = [t["self_nana_char"], t["self_nana_action"]]
        if not nsi:
            nana_self_id.append(t["self_nana_c_dir"])

        nana_self_st = [t["self_nana_num"]]
        if not nsi and "self_nana_analog" in t:
            nana_self_st.append(t["self_nana_analog"])
        if has_btn_token:
            nana_self_st.append(t["nana_buttons"])
        nana_self_st.append(t["nana_flags"])

        proj_a = [t[f"proj{j}_{k}"] for j in range(3) for k in ("owner", "type", "subtype")] + [t["proj_num"]]
        proj_b = [t[f"proj{j}_{k}"] for j in range(3, 6) for k in ("owner", "type", "subtype")]
        proj_c = [t[f"proj{j}_{k}"] for j in range(6, 8) for k in ("owner", "type", "subtype")]

        groups = [
            [t["stage"], t["global_num"]],                                       # GAME
            [t["self_port"], t["self_char"], t["self_costume"]],                  # SELF_IDENT
            self_action_group,                                                   # SELF_ACTION
            [t["self_player_num"], t["flags"]],                                  # SELF_STATE
        ]
        if not nsi and has_btn_token:
            groups.append([t["self_analog"], t["buttons"]])                      # SELF_INPUT
        groups.extend([
            [t["opp_port"], t["opp_char"], t["opp_costume"]],                    # OPP_IDENT
            opp_action_group,                                                    # OPP_ACTION
            [t["opp_player_num"], t["flags"]],                                   # OPP_STATE
        ])
        if not noi and has_btn_token:
            groups.append([t["opp_analog"], t["buttons"]])                       # OPP_INPUT
        groups.extend([
            nana_self_id,                                                        # NANA_SELF_ID
            nana_self_st,                                                        # NANA_SELF_ST
            nana_opp_id,                                                         # NANA_OPP_ID
            nana_opp_st,                                                         # NANA_OPP_ST
            proj_a,                                                              # PROJ_A
            proj_b,                                                              # PROJ_B
            proj_c,                                                              # PROJ_C
        ])

        composite = [self.mixers[i](g) for i, g in enumerate(groups)]
        stack = torch.stack(composite, dim=2)
        flat = self._collapse(stack)
        pooled = self.set_attn(flat)
        pooled = pooled.view(B, T, self._d_intra)
        return self.to_model(pooled)


# ---------------------------------------------------------------------------
# Flat encoder: concat all raw features → single Linear (no per-group MLPs).
# MIMIC's canonical encoder, bootstrapped from HAL's input projection.
# ---------------------------------------------------------------------------

class MimicFlatEncoder(nn.Module):
    """Flat concat + Linear input projection.

    Per frame:
      concat([stage_emb(4), char_emb(12)*2, action_emb(32)*2,
              gamestate(numeric), controller(one-hot)]) → Linear → d_model

    No per-group MLPs. Categorical embeddings are small.
    Requires --mimic-minimal-features and --mimic-controller-encoding.
    """

    STAGE_EMB_DIM = 4
    CHAR_EMB_DIM = 12
    ACTION_EMB_DIM = 32

    def __init__(
        self,
        *,
        d_model: int = 512,
        dropout: float = 0.0,
        num_stages: int,
        num_characters: int,
        num_actions: int,
        mimic_minimal_features: bool = True,
        mimic_controller_encoding: bool = True,
        n_controller_combos: int = 5,
        no_self_inputs: bool = False,
        use_input_gate: bool = False,
        # Legacy aliases
        hal_minimal_features: bool = None,
        hal_controller_encoding: bool = None,
        **kwargs,  # absorb unused params from build_encoder
    ) -> None:
        super().__init__()
        if hal_minimal_features is not None:
            mimic_minimal_features = hal_minimal_features
        if hal_controller_encoding is not None:
            mimic_controller_encoding = hal_controller_encoding
        self._no_self_inputs = no_self_inputs
        self._ctrl_enc = mimic_controller_encoding
        # Legacy alias
        self._hal_ctrl_enc = self._ctrl_enc
        self._minimal = mimic_minimal_features
        self.si_drop_prob: float = 0.0

        # Categorical embeddings (HAL sizes)
        self.stage_emb = nn.Embedding(num_stages, self.STAGE_EMB_DIM)
        self.char_emb = nn.Embedding(num_characters, self.CHAR_EMB_DIM)
        self.action_emb = nn.Embedding(num_actions, self.ACTION_EMB_DIM)

        # Compute total input dimension
        emb_dim = self.STAGE_EMB_DIM + 2 * self.CHAR_EMB_DIM + 2 * self.ACTION_EMB_DIM
        # = 4 + 24 + 64 = 92

        # Gamestate dim per player × 2:
        #   minimal: 6 numeric (pos_x/y, percent, stock, jumps_left, shield) + 3 flags
        #            (on_ground, facing, invulnerable) = 9
        #   full:    22 numeric (adds speeds/hitlag/hitstun/invuln_left/ECB) + 5 flags
        #            (adds off_stage, moonwalkwarning) = 27
        per_player = 9 if mimic_minimal_features else 27
        numeric_dim = per_player * 2

        # Controller: one-hot vector (37 stick + 9 cstick + N combos + 3 shoulder)
        ctrl_dim = 0
        if mimic_controller_encoding and not no_self_inputs:
            ctrl_dim = 37 + 9 + n_controller_combos + 3

        self._input_dim = emb_dim + numeric_dim + ctrl_dim
        self._n_controller_combos = n_controller_combos

        # Single projection (matching HAL's proj_down)
        self.proj = nn.Linear(self._input_dim, d_model)
        self.drop = nn.Dropout(dropout)

        # Optional per-input-column sigmoid gate for feature importance.
        # Init logits so sigmoid ≈ 0.88 (start near fully-on, let L1 decay).
        self._use_input_gate = use_input_gate
        if use_input_gate:
            self.input_gate_logits = nn.Parameter(
                torch.full((self._input_dim,), 2.0)
            )
        self._feature_names: List[str] = self._build_feature_names()
        assert len(self._feature_names) == self._input_dim, (
            f"feature_names len {len(self._feature_names)} != input_dim {self._input_dim}"
        )

    def forward(self, seq: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Stochastic self-input masking
        if not self._no_self_inputs and self.si_drop_prob > 0 and self._hal_ctrl_enc:
            if "self_controller" in seq:
                seq = dict(seq)
                B = seq["self_controller"].shape[0]
                device = seq["self_controller"].device
                if self.training and self.si_drop_prob < 1.0:
                    keep = (torch.rand(B, 1, 1, device=device) >= self.si_drop_prob).float()
                else:
                    keep = torch.zeros(B, 1, 1, device=device)
                seq["self_controller"] = seq["self_controller"] * keep

        # Categorical embeddings
        # Stage remap: MIMIC shards use NO_STAGE=0, so tournament stages are 1-6.
        # HAL expects 0-5. Subtract 1 and clamp to handle the offset.
        stage_idx = seq["stage"]
        if self.stage_emb.num_embeddings == 6:
            stage_idx = (stage_idx - 1).clamp(min=0)
        stage = self.stage_emb(stage_idx)                      # (B, T, 4)
        self_char = self.char_emb(seq["self_character"])        # (B, T, 12)
        opp_char = self.char_emb(seq["opp_character"])          # (B, T, 12)
        self_action = self.action_emb(seq["self_action"])       # (B, T, 32)
        opp_action = self.action_emb(seq["opp_action"])         # (B, T, 32)

        # Numeric gamestate. Shard stores 22 numeric + 5 flag cols per player
        # (see mimic/features.py:numeric_state + flags for the full column list).
        # HAL normalizes boolean flags: 0→-1, 1→+1 (normalize transform with min=0,max=1)
        sn = seq["self_numeric"]
        on = seq["opp_numeric"]
        sf_all = seq["self_flags"].float() * 2.0 - 1.0
        of_all = seq["opp_flags"].float() * 2.0 - 1.0

        if self._minimal:
            # HAL minimal: 6 numeric + 3 flags per player, reordered into HAL's
            # exact concat order so minimal-trained checkpoints load unchanged.
            if sn.shape[-1] > 7:
                _IDX = [0, 1, 2, 3, 4, 13]  # pos_x, pos_y, percent, stock, jumps_left, shield
                sn = sn[..., _IDX]
                on = on[..., _IDX]
            elif sn.shape[-1] == 7:
                _IDX = [0, 1, 2, 3, 4, 6]   # legacy 7-col minimal shards (drop invuln_left)
                sn = sn[..., _IDX]
                on = on[..., _IDX]
            _FLAG_IDX = [0, 2, 3]  # on_ground, facing, invulnerable
            sf = sf_all[..., _FLAG_IDX]
            of = of_all[..., _FLAG_IDX]
            # Reorder to HAL: percent, stock, facing, invulnerable, jumps_left,
            #                  on_ground, shield, pos_x, pos_y
            _HAL_ORDER = [2, 3, 7, 8, 4, 6, 5, 0, 1]
            self_num = torch.cat([sn, sf], dim=-1)[..., _HAL_ORDER]
            opp_num = torch.cat([on, of], dim=-1)[..., _HAL_ORDER]
        else:
            # Full: all 22 numeric + 5 flags per player, no HAL reorder (a single
            # learned Linear downstream makes the ordering functionally irrelevant).
            if sn.shape[-1] != 22:
                raise ValueError(
                    f"MimicFlatEncoder(mimic_minimal_features=False) requires 22-col "
                    f"self_numeric shards; got shape[-1]={sn.shape[-1]}. Either rebuild "
                    f"shards with the full numeric state or pass --mimic-minimal-features."
                )
            self_num = torch.cat([sn, sf_all], dim=-1)  # (B, T, 27)
            opp_num = torch.cat([on, of_all], dim=-1)

        # (parts built below — gate applied after concat)

        parts = [stage, self_char, opp_char, self_action, opp_action,
                 self_num, opp_num]

        # Controller one-hot
        if self._hal_ctrl_enc and not self._no_self_inputs and "self_controller" in seq:
            parts.append(seq["self_controller"])

        # Concat everything → single projection
        combined = torch.cat(parts, dim=-1)     # (B, T, input_dim)
        if self._use_input_gate:
            gate = torch.sigmoid(self.input_gate_logits)  # (input_dim,)
            combined = combined * gate
        return self.drop(self.proj(combined))    # (B, T, d_model)

    # ------------------------------------------------------------------
    # Input-gate helpers (no-ops when use_input_gate=False)
    # ------------------------------------------------------------------
    def gate_l1_penalty(self) -> torch.Tensor:
        """Mean of sigmoid(gate_logits). Add to loss via  loss += lam * gate_l1_penalty().

        Returns a 0-d tensor on the same device as gate_logits. Returns 0 when
        the gate is disabled.
        """
        if not self._use_input_gate:
            # Return a zero tensor that's safe to .backward() through.
            return self.proj.weight.new_zeros(())
        return torch.sigmoid(self.input_gate_logits).mean()

    @torch.no_grad()
    def gate_values(self) -> Optional[torch.Tensor]:
        """Detached gate values for inspection, or None if gate disabled."""
        if not self._use_input_gate:
            return None
        return torch.sigmoid(self.input_gate_logits).detach().cpu()

    def feature_names(self) -> List[str]:
        return list(self._feature_names)

    def gate_report(self) -> List[Tuple[str, float]]:
        """Ranked (name, gate_value) list, ascending — smallest values first
        are the features the model has pruned most. Returns [] if gate disabled.
        """
        if not self._use_input_gate:
            return []
        vals = self.gate_values().tolist()
        return sorted(zip(self._feature_names, vals), key=lambda x: x[1])

    def _build_feature_names(self) -> List[str]:
        """Names aligned with the concat order in forward(): stage_emb, self/opp
        char_emb, self/opp action_emb, self/opp numeric+flags, self_controller.
        """
        names: List[str] = []
        names.extend(f"stage_emb[{i}]" for i in range(self.STAGE_EMB_DIM))
        names.extend(f"self_char_emb[{i}]" for i in range(self.CHAR_EMB_DIM))
        names.extend(f"opp_char_emb[{i}]" for i in range(self.CHAR_EMB_DIM))
        names.extend(f"self_action_emb[{i}]" for i in range(self.ACTION_EMB_DIM))
        names.extend(f"opp_action_emb[{i}]" for i in range(self.ACTION_EMB_DIM))

        if self._minimal:
            # Names match the reordered 9-slot HAL layout from the minimal path
            # (percent, stock, facing, invulnerable, jumps_left, on_ground, shield,
            #  pos_x, pos_y) — see _HAL_ORDER in forward().
            mini_ordered = ["percent", "stock", "facing", "invulnerable",
                            "jumps_left", "on_ground", "shield_strength",
                            "pos_x", "pos_y"]
            names.extend(f"self_{n}" for n in mini_ordered)
            names.extend(f"opp_{n}" for n in mini_ordered)
        else:
            # Full: all 22 numeric + 5 flags per player in native order.
            num_cols = [
                "pos_x", "pos_y", "percent", "stock", "jumps_left",
                "speed_air_x_self", "speed_ground_x_self",
                "speed_x_attack", "speed_y_attack", "speed_y_self",
                "hitlag_left", "hitstun_left", "invuln_left",
                "shield_strength",
                "ecb_bottom_x", "ecb_bottom_y", "ecb_left_x", "ecb_left_y",
                "ecb_right_x", "ecb_right_y", "ecb_top_x", "ecb_top_y",
            ]
            flag_cols = ["on_ground", "off_stage", "facing",
                         "invulnerable", "moonwalkwarning"]
            full = num_cols + flag_cols
            names.extend(f"self_{n}" for n in full)
            names.extend(f"opp_{n}" for n in full)

        # Controller one-hot slots (when enabled and self is visible).
        if self._hal_ctrl_enc and not self._no_self_inputs:
            names.extend(f"ctrl_main[{i}]" for i in range(37))
            names.extend(f"ctrl_cstick[{i}]" for i in range(9))
            names.extend(f"ctrl_combo[{i}]" for i in range(self._n_controller_combos))
            names.extend(f"ctrl_shoulder[{i}]" for i in range(3))
        return names


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

ENCODER_REGISTRY = {
    "default": FrameEncoder,
    "flat": FlatFrameEncoder,
    "composite8": CompositeFrameEncoder,
    "hybrid16": HybridFrameEncoder,
    "mimic_flat": MimicFlatEncoder,
    # Legacy alias
    "hal_flat": MimicFlatEncoder,
}

# Legacy class alias
HALFlatEncoder = MimicFlatEncoder


def build_encoder(
    encoder_type: str = "default",
    *,
    d_model: int = 1024,
    d_intra: int = 256,
    dropout: float = 0.0,
    nlayers: int = 2,
    k_query: int = 1,
    scaled_emb: bool = False,
    no_opp_inputs: bool = False,
    no_self_inputs: bool = False,
    lean_features: bool = False,
    mimic_minimal_features: bool = False,
    mimic_controller_encoding: bool = False,
    n_controller_combos: int = 5,
    use_input_gate: bool = False,
    # Legacy aliases
    hal_minimal_features: bool = None,
    hal_controller_encoding: bool = None,
    num_stages: int,
    num_ports: int,
    num_characters: int,
    num_actions: int,
    num_costumes: int,
    num_proj_types: int,
    num_proj_subtypes: int,
    num_c_dirs: int = 5,
) -> _BaseFrameEncoder:
    # Accept legacy kwargs
    if hal_minimal_features is not None:
        mimic_minimal_features = hal_minimal_features
    if hal_controller_encoding is not None:
        mimic_controller_encoding = hal_controller_encoding

    cls = ENCODER_REGISTRY.get(encoder_type)
    if cls is None:
        raise ValueError(f"Unknown encoder type: {encoder_type!r}. "
                         f"Available: {list(ENCODER_REGISTRY.keys())}")

    vocab_kwargs = dict(
        num_stages=num_stages, num_ports=num_ports, num_characters=num_characters,
        num_actions=num_actions, num_costumes=num_costumes,
        num_proj_types=num_proj_types, num_proj_subtypes=num_proj_subtypes,
        num_c_dirs=num_c_dirs, no_opp_inputs=no_opp_inputs,
        no_self_inputs=no_self_inputs, lean_features=lean_features,
        mimic_minimal_features=mimic_minimal_features,
        mimic_controller_encoding=mimic_controller_encoding,
        n_controller_combos=n_controller_combos,
    )

    common = dict(d_model=d_model, d_intra=d_intra, dropout=dropout, scaled_emb=scaled_emb)

    if encoder_type in ("mimic_flat", "hal_flat"):
        return cls(d_model=d_model, dropout=dropout,
                   use_input_gate=use_input_gate, **vocab_kwargs)
    elif encoder_type == "flat":
        return cls(**common, **vocab_kwargs)
    else:
        return cls(**common, nlayers=nlayers, k_query=k_query, **vocab_kwargs)
