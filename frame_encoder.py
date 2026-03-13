# frame_encoder.py – intra-frame encoder variants for FRAME
# -----------------------------------------------------------------------------
# Converts structured Melee frame dictionaries -> (B, T, d_model) tensor.
#
# Encoder variants (selected via --encoder CLI flag):
#   "default"     – 55 individual tokens + intra-frame self-attention (original)
#   "flat"        – concat all tokens, 2-layer MLP to d_model (no attention)
#   "composite8"  – 8 semantic group tokens + intra-frame self-attention
#   "hybrid16"    – 16 entity-level tokens + intra-frame self-attention
# -----------------------------------------------------------------------------

from __future__ import annotations

import math
from typing import Dict, List, Optional

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
    NANA_NUM     = 27
    ANALOG_DIM   = 4
    BTN_DIM      = 12
    FLAGS_DIM    = 5
    NANA_FLAGS   = 6
    PROJ_NUM_PER = 5
    PROJ_SLOTS   = 8
    N_RAW_TOKENS = 55

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
    ) -> None:
        super().__init__()
        self._d_intra = d_intra
        self._dropout = dropout

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

        self.glob_enc      = _mlp(self.GLOBAL_NUM, d_intra, dropout)
        self.player_enc    = _mlp(self.PLAYER_NUM + 1, d_intra, dropout)
        self.nana_enc      = _mlp(self.NANA_NUM + 1, d_intra, dropout)
        self.analog_enc    = _mlp(self.ANALOG_DIM, d_intra, dropout)
        self.proj_num_enc  = _mlp(self.PROJ_NUM_PER * self.PROJ_SLOTS, d_intra, dropout)
        self.btn_enc       = _mlp(self.BTN_DIM * 2, d_intra, dropout)
        self.flag_enc      = _mlp(self.FLAGS_DIM * 2, d_intra, dropout)
        self.nana_btn_enc  = _mlp(self.BTN_DIM * 2, d_intra, dropout)
        self.nana_flag_enc = _mlp(self.NANA_FLAGS * 2, d_intra, dropout)

    def _build_raw_tokens(self, seq: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """Build all ~55 individual feature-group tokens, each (B,T,d_intra)."""
        tok: List[torch.Tensor] = []

        tok.append(self.stage_emb(seq["stage"]))
        tok.append(self.port_emb(seq["self_port"]))
        tok.append(self.port_emb(seq["opp_port"]))
        tok.append(self.char_emb(seq["self_character"]))
        tok.append(self.char_emb(seq["opp_character"]))
        tok.append(self.act_emb(seq["self_action"]))
        tok.append(self.act_emb(seq["opp_action"]))
        tok.append(self.cost_emb(seq["self_costume"]))
        tok.append(self.cost_emb(seq["opp_costume"]))
        tok.append(self.cdir_emb(seq["self_c_dir"]))
        tok.append(self.cdir_emb(seq["opp_c_dir"]))

        tok.append(self.char_emb(seq["self_nana_character"]))
        tok.append(self.char_emb(seq["opp_nana_character"]))
        tok.append(self.act_emb(seq["self_nana_action"]))
        tok.append(self.act_emb(seq["opp_nana_action"]))
        tok.append(self.cdir_emb(seq["self_nana_c_dir"]))
        tok.append(self.cdir_emb(seq["opp_nana_c_dir"]))

        for j in range(self.PROJ_SLOTS):
            tok.append(self.port_emb(seq[f"proj{j}_owner"]))
            tok.append(self.ptype_emb(seq[f"proj{j}_type"]))
            tok.append(self.psub_emb(seq[f"proj{j}_subtype"]))

        tok.append(self.glob_enc(seq["numeric"]))
        tok.append(self.player_enc(torch.cat([seq["self_numeric"], seq["self_action_elapsed"].unsqueeze(-1).float()], dim=-1)))
        tok.append(self.player_enc(torch.cat([seq["opp_numeric"], seq["opp_action_elapsed"].unsqueeze(-1).float()], dim=-1)))
        tok.append(self.nana_enc(torch.cat([seq["self_nana_numeric"], seq["self_nana_action_elapsed"].unsqueeze(-1).float()], dim=-1)))
        tok.append(self.nana_enc(torch.cat([seq["opp_nana_numeric"], seq["opp_nana_action_elapsed"].unsqueeze(-1).float()], dim=-1)))
        tok.append(self.analog_enc(seq["self_analog"]))
        tok.append(self.analog_enc(seq["opp_analog"]))
        tok.append(self.analog_enc(seq["self_nana_analog"]))
        tok.append(self.analog_enc(seq["opp_nana_analog"]))
        tok.append(self.proj_num_enc(torch.cat([seq[f"{k}_numeric"] for k in map(str, range(self.PROJ_SLOTS))], dim=-1)))
        tok.append(self.btn_enc(torch.cat([seq["self_buttons"].float(), seq["opp_buttons"].float()], dim=-1)))
        tok.append(self.flag_enc(torch.cat([seq["self_flags"].float(), seq["opp_flags"].float()], dim=-1)))
        tok.append(self.nana_btn_enc(torch.cat([seq["self_nana_buttons"].float(), seq["opp_nana_buttons"].float()], dim=-1)))
        tok.append(self.nana_flag_enc(torch.cat([seq["self_nana_flags"].float(), seq["opp_nana_flags"].float()], dim=-1)))

        return tok

    @staticmethod
    def _collapse(x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape[:2]
        return x.reshape(B * T, *x.shape[2:])


# ---------------------------------------------------------------------------
# Original encoder: 55 tokens + intra-frame attention
# ---------------------------------------------------------------------------

class FrameEncoder(_BaseFrameEncoder):
    """55-token intra-frame attention encoder (original architecture)."""

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
        tokens = self._build_raw_tokens(seq)
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
        cat_dim = self.N_RAW_TOKENS * d_intra
        self.proj = nn.Sequential(
            nn.LayerNorm(cat_dim),
            nn.Linear(cat_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, seq: Dict[str, torch.Tensor]) -> torch.Tensor:
        tokens = self._build_raw_tokens(seq)
        cat = torch.cat(tokens, dim=-1)
        return self.proj(cat)


# ---------------------------------------------------------------------------
# Composite encoder: 8 semantic group tokens + intra-frame attention
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
    """8 semantic composite tokens + intra-frame attention.

    Groups:
      0: GAME_STATE  (stage, global numerics)
      1: SELF_INPUT  (self analog, self buttons)
      2: OPP_INPUT   (opp analog, opp buttons)
      3: SELF_STATE  (self port/char/act/cost/cdir, self numeric, self flags)
      4: OPP_STATE   (opp port/char/act/cost/cdir, opp numeric, opp flags)
      5: NANA_SELF   (all self nana features)
      6: NANA_OPP    (all opp nana features)
      7: PROJECTILES (all proj cats + proj numerics)
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
        self.mixers = nn.ModuleList([_CompositeTokenMixer(d_intra, dropout) for _ in range(8)])
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

        # Token indices from _build_raw_tokens:
        # 0: stage
        # 1-2: self/opp port
        # 3-4: self/opp char
        # 5-6: self/opp action
        # 7-8: self/opp costume
        # 9-10: self/opp cdir
        # 11-16: nana cats (self_char, opp_char, self_act, opp_act, self_cdir, opp_cdir)
        # 17-40: proj cats (8 slots × 3)
        # 41: global numeric
        # 42-43: self/opp player numeric
        # 44-45: self/opp nana numeric
        # 46-47: self/opp analog
        # 48-49: self/opp nana analog
        # 50: proj numerics
        # 51: buttons
        # 52: flags
        # 53: nana buttons
        # 54: nana flags

        groups = [
            [raw[0], raw[41]],                                                  # GAME_STATE
            [raw[46], raw[51]],                                                 # SELF_INPUT
            [raw[47], raw[51]],                                                 # OPP_INPUT
            [raw[1], raw[3], raw[5], raw[7], raw[9], raw[42], raw[52]],         # SELF_STATE
            [raw[2], raw[4], raw[6], raw[8], raw[10], raw[43], raw[52]],        # OPP_STATE
            [raw[11], raw[13], raw[15], raw[44], raw[48], raw[53], raw[54]],    # NANA_SELF
            [raw[12], raw[14], raw[16], raw[45], raw[49], raw[53], raw[54]],    # NANA_OPP
            raw[17:41] + [raw[50]],                                             # PROJECTILES
        ]

        composite = [self.mixers[i](g) for i, g in enumerate(groups)]
        stack = torch.stack(composite, dim=2)           # (B,T,8,d_intra)
        flat = self._collapse(stack)                    # (B*T,8,d_intra)
        pooled = self.set_attn(flat)                    # (B*T,d_intra)
        pooled = pooled.view(B, T, self._d_intra)
        return self.to_model(pooled)


# ---------------------------------------------------------------------------
# Hybrid encoder: 16 entity-level tokens + intra-frame attention
# ---------------------------------------------------------------------------

class HybridFrameEncoder(_BaseFrameEncoder):
    """16 tokens (entity x feature-type) + intra-frame attention.

    Tokens:
      0: GAME         (stage, global numerics)
      1: SELF_IDENT   (self port, char, costume)
      2: SELF_ACTION   (self action, cdir)
      3: SELF_STATE   (self numeric, flags)
      4: SELF_INPUT   (self analog, buttons)
      5: OPP_IDENT    (opp port, char, costume)
      6: OPP_ACTION    (opp action, cdir)
      7: OPP_STATE    (opp numeric, flags)
      8: OPP_INPUT    (opp analog, buttons)
      9: NANA_SELF_ID  (self nana char, action, cdir)
      10: NANA_SELF_ST (self nana numeric, analog, buttons, flags)
      11: NANA_OPP_ID  (opp nana char, action, cdir)
      12: NANA_OPP_ST  (opp nana numeric, analog, buttons, flags)
      13-15: PROJ_A/B/C (projectile slots split into 3 groups)
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
        raw = self._build_raw_tokens(seq)

        groups = [
            [raw[0], raw[41]],                      # 0: GAME
            [raw[1], raw[3], raw[7]],                # 1: SELF_IDENT
            [raw[5], raw[9]],                        # 2: SELF_ACTION
            [raw[42], raw[52]],                      # 3: SELF_STATE
            [raw[46], raw[51]],                      # 4: SELF_INPUT
            [raw[2], raw[4], raw[8]],                # 5: OPP_IDENT
            [raw[6], raw[10]],                       # 6: OPP_ACTION
            [raw[43], raw[52]],                      # 7: OPP_STATE
            [raw[47], raw[51]],                      # 8: OPP_INPUT
            [raw[11], raw[13], raw[15]],             # 9: NANA_SELF_ID
            [raw[44], raw[48], raw[53], raw[54]],    # 10: NANA_SELF_ST
            [raw[12], raw[14], raw[16]],             # 11: NANA_OPP_ID
            [raw[45], raw[49], raw[53], raw[54]],    # 12: NANA_OPP_ST
            raw[17:25] + [raw[50]],                  # 13: PROJ_A (slots 0-2)
            raw[25:33],                              # 14: PROJ_B (slots 3-5)
            raw[33:41],                              # 15: PROJ_C (slots 6-7)
        ]

        composite = [self.mixers[i](g) for i, g in enumerate(groups)]
        stack = torch.stack(composite, dim=2)
        flat = self._collapse(stack)
        pooled = self.set_attn(flat)
        pooled = pooled.view(B, T, self._d_intra)
        return self.to_model(pooled)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

ENCODER_REGISTRY = {
    "default": FrameEncoder,
    "flat": FlatFrameEncoder,
    "composite8": CompositeFrameEncoder,
    "hybrid16": HybridFrameEncoder,
}


def build_encoder(
    encoder_type: str = "default",
    *,
    d_model: int = 1024,
    d_intra: int = 256,
    dropout: float = 0.0,
    nlayers: int = 2,
    k_query: int = 1,
    scaled_emb: bool = False,
    num_stages: int,
    num_ports: int,
    num_characters: int,
    num_actions: int,
    num_costumes: int,
    num_proj_types: int,
    num_proj_subtypes: int,
    num_c_dirs: int = 5,
) -> _BaseFrameEncoder:
    cls = ENCODER_REGISTRY.get(encoder_type)
    if cls is None:
        raise ValueError(f"Unknown encoder type: {encoder_type!r}. "
                         f"Available: {list(ENCODER_REGISTRY.keys())}")

    vocab_kwargs = dict(
        num_stages=num_stages, num_ports=num_ports, num_characters=num_characters,
        num_actions=num_actions, num_costumes=num_costumes,
        num_proj_types=num_proj_types, num_proj_subtypes=num_proj_subtypes,
        num_c_dirs=num_c_dirs,
    )

    common = dict(d_model=d_model, d_intra=d_intra, dropout=dropout, scaled_emb=scaled_emb)

    if encoder_type == "flat":
        return cls(**common, **vocab_kwargs)
    else:
        return cls(**common, nlayers=nlayers, k_query=k_query, **vocab_kwargs)
