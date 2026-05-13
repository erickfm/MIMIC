"""ValueEncoder — feature-complete per-frame encoder for V(s) discovery.

Unlike MimicFlatEncoder (which BC uses with a strict subset of shard keys),
ValueEncoder reads the full WM-schema feature set:

  Categorical:  stage, self/opp character, self/opp action,
                projectile owner/type/subtype (×8 slots)
  Numeric:      self/opp numeric+flags, self/opp action_elapsed,
                projectile numeric (×8), stage geometry (18)
  One-hot:      self_controller, opp_controller

This is the input space V(s) gets to discover from. The BC encoder
deliberately omits opp_controller, action_elapsed, projectiles, nana, and
stage geometry — including them here lets V(s) attend to features like
"opp is committed to a punishable action with N frames left" or "fox in
recovery + needle in flight on stage edge."

Nana state is omitted: fox_all_v2 is fox-only and Nana columns are zeros.
Add a `use_nana=True` flag if/when training on Ice Climbers.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class ValueEncoder(nn.Module):
    """Flat concat + Linear projection → (B, T, d_model).

    Matches MimicFlatEncoder's structural pattern (categorical embeddings +
    numeric concat + single Linear) but over the full feature set.
    """

    # Embedding dims (mirrors MimicFlatEncoder where applicable)
    STAGE_EMB_DIM = 4
    CHAR_EMB_DIM = 12
    ACTION_EMB_DIM = 32
    PROJ_OWNER_EMB_DIM = 4
    PROJ_TYPE_EMB_DIM = 4
    PROJ_SUBTYPE_EMB_DIM = 4

    PROJ_SLOTS = 8
    PROJ_NUM_PER_SLOT = 5    # per projectile numeric columns
    # Top-level "numeric" key: [distance, frame, *18 stage_geom_cols]
    # See mimic/features.py:182 — "numeric": ["distance", "frame", *STAGE_GEOM_COLS]
    GLOBAL_NUM_DIM = 20
    PER_PLAYER_NUMERIC = 13   # post-2026-04-20 schema
    PER_PLAYER_FLAGS = 5
    CTRL_DIM = 56             # 37 stick + 9 c_stick + 7 buttons + 3 shoulder
    NUM_CONTROLLER_COMBOS = 7  # fox_all_v2 uses 7-combo buttons

    def __init__(
        self,
        *,
        d_model: int = 512,
        dropout: float = 0.1,
        num_stages: int = 6,
        num_characters: int = 27,
        num_actions: int = 396,
        num_proj_types: int = 103,
        num_proj_subtypes: int = 40,
        num_proj_owners: int = 4,
        n_controller_combos: int = 7,
        use_input_gate: bool = False,
    ):
        super().__init__()
        self._n_combos = n_controller_combos
        # Sanity: shard ctrl one-hot is built for n_combos=7
        assert n_controller_combos == 7, (
            f"fox_all_v2 shards use 7-combo controller; got n={n_controller_combos}"
        )

        # Categorical embedding tables
        self.stage_emb = nn.Embedding(num_stages, self.STAGE_EMB_DIM)
        self.char_emb = nn.Embedding(num_characters, self.CHAR_EMB_DIM)
        self.action_emb = nn.Embedding(num_actions, self.ACTION_EMB_DIM)
        self.proj_owner_emb = nn.Embedding(num_proj_owners, self.PROJ_OWNER_EMB_DIM)
        self.proj_type_emb = nn.Embedding(num_proj_types, self.PROJ_TYPE_EMB_DIM)
        self.proj_subtype_emb = nn.Embedding(num_proj_subtypes, self.PROJ_SUBTYPE_EMB_DIM)

        # Compute input_dim by group (each group's contribution to the concat)
        groups: List[Tuple[str, int]] = []
        groups.append(("stage_emb", self.STAGE_EMB_DIM))
        groups.append(("self_char_emb", self.CHAR_EMB_DIM))
        groups.append(("opp_char_emb", self.CHAR_EMB_DIM))
        groups.append(("self_action_emb", self.ACTION_EMB_DIM))
        groups.append(("opp_action_emb", self.ACTION_EMB_DIM))
        groups.append(("self_numeric+flags",
                       self.PER_PLAYER_NUMERIC + self.PER_PLAYER_FLAGS))
        groups.append(("opp_numeric+flags",
                       self.PER_PLAYER_NUMERIC + self.PER_PLAYER_FLAGS))
        groups.append(("self_action_elapsed", 1))
        groups.append(("opp_action_elapsed", 1))
        # Per projectile slot: owner_emb + type_emb + subtype_emb + 5 numeric
        per_proj = (self.PROJ_OWNER_EMB_DIM + self.PROJ_TYPE_EMB_DIM
                    + self.PROJ_SUBTYPE_EMB_DIM + self.PROJ_NUM_PER_SLOT)
        for i in range(self.PROJ_SLOTS):
            groups.append((f"proj{i}", per_proj))
        groups.append(("global_numeric", self.GLOBAL_NUM_DIM))
        groups.append(("self_controller", self.CTRL_DIM))
        groups.append(("opp_controller", self.CTRL_DIM))

        self._groups = groups
        self._input_dim = sum(d for _, d in groups)

        # Single projection, matching MimicFlatEncoder's structural pattern
        self.proj = nn.Linear(self._input_dim, d_model)
        self.drop = nn.Dropout(dropout)

        # Per-input-column sigmoid gate for feature attribution
        self._use_input_gate = use_input_gate
        if use_input_gate:
            self.input_gate_logits = nn.Parameter(
                torch.full((self._input_dim,), 2.0)  # sigmoid(2) ≈ 0.88
            )

        self._feature_names: List[str] = self._build_feature_names()
        assert len(self._feature_names) == self._input_dim, (
            f"feature_names {len(self._feature_names)} != input_dim {self._input_dim}"
        )

    @property
    def input_dim(self) -> int:
        return self._input_dim

    def feature_groups(self) -> List[Tuple[str, int]]:
        """Returns the (name, width) decomposition of the concat."""
        return list(self._groups)

    def feature_names(self) -> List[str]:
        return list(self._feature_names)

    def _build_feature_names(self) -> List[str]:
        names: List[str] = []
        names.extend(f"stage_emb[{i}]" for i in range(self.STAGE_EMB_DIM))
        names.extend(f"self_char_emb[{i}]" for i in range(self.CHAR_EMB_DIM))
        names.extend(f"opp_char_emb[{i}]" for i in range(self.CHAR_EMB_DIM))
        names.extend(f"self_action_emb[{i}]" for i in range(self.ACTION_EMB_DIM))
        names.extend(f"opp_action_emb[{i}]" for i in range(self.ACTION_EMB_DIM))
        # 13 numeric + 5 flags per player, native shard order
        num_cols = [
            "pos_x", "pos_y", "percent", "stock", "jumps_left",
            "speed_air_x_self", "speed_ground_x_self", "speed_x_attack",
            "speed_y_attack", "speed_y_self", "hitlag_left", "hitstun_left",
            "shield_strength",
        ]
        flag_cols = ["on_ground", "off_stage", "facing", "invulnerable",
                     "moonwalkwarning"]
        for side in ("self", "opp"):
            names.extend(f"{side}_{c}" for c in num_cols)
            names.extend(f"{side}_{c}" for c in flag_cols)
        names.append("self_action_elapsed")
        names.append("opp_action_elapsed")
        # Projectile slots
        for i in range(self.PROJ_SLOTS):
            names.extend(f"proj{i}_owner_emb[{j}]"
                         for j in range(self.PROJ_OWNER_EMB_DIM))
            names.extend(f"proj{i}_type_emb[{j}]"
                         for j in range(self.PROJ_TYPE_EMB_DIM))
            names.extend(f"proj{i}_subtype_emb[{j}]"
                         for j in range(self.PROJ_SUBTYPE_EMB_DIM))
            names.extend(f"proj{i}_numeric[{j}]"
                         for j in range(self.PROJ_NUM_PER_SLOT))
        # Global numeric: distance, frame, 18 stage geometry cols
        # (matches mimic/features.py:182 "numeric" column order)
        global_cols = [
            "distance", "frame",
            "blastzone_left", "blastzone_right", "blastzone_top",
            "blastzone_bottom", "stage_edge_left", "stage_edge_right",
            "left_platform_height", "left_platform_left", "left_platform_right",
            "right_platform_height", "right_platform_left", "right_platform_right",
            "top_platform_height", "top_platform_left", "top_platform_right",
            "randall_height", "randall_left", "randall_right",
        ]
        assert len(global_cols) == self.GLOBAL_NUM_DIM
        names.extend(f"global_{c}" for c in global_cols)
        # Controller one-hots: 37 stick + 9 c_stick + N combos + 3 shoulder
        for side in ("self", "opp"):
            names.extend(f"{side}_ctrl_stick[{i}]" for i in range(37))
            names.extend(f"{side}_ctrl_cstick[{i}]" for i in range(9))
            names.extend(f"{side}_ctrl_btn[{i}]" for i in range(self._n_combos))
            names.extend(f"{side}_ctrl_shoulder[{i}]" for i in range(3))
        return names

    def forward(self, seq: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Stage remap (NO_STAGE=0 → tournament stages 1-6 → emb idx 0-5)
        stage_idx = seq["stage"]
        if self.stage_emb.num_embeddings == 6:
            stage_idx = (stage_idx - 1).clamp(min=0)
        stage = self.stage_emb(stage_idx)
        self_char = self.char_emb(seq["self_character"])
        opp_char = self.char_emb(seq["opp_character"])
        self_action = self.action_emb(seq["self_action"])
        opp_action = self.action_emb(seq["opp_action"])

        # Numeric+flags per player (full 13+5 schema)
        sn, on = seq["self_numeric"], seq["opp_numeric"]
        if sn.shape[-1] != 13:
            raise ValueError(
                f"ValueEncoder requires 13-col self_numeric (post 2026-04-20 "
                f"schema); got width {sn.shape[-1]}"
            )
        sf = seq["self_flags"].float() * 2.0 - 1.0
        of = seq["opp_flags"].float() * 2.0 - 1.0
        self_pn = torch.cat([sn, sf], dim=-1)   # (B, T, 18)
        opp_pn = torch.cat([on, of], dim=-1)

        # action_elapsed (already normalized in shards)
        self_ae = seq["self_action_elapsed"].unsqueeze(-1)
        opp_ae = seq["opp_action_elapsed"].unsqueeze(-1)

        # Projectiles
        proj_parts = []
        for i in range(self.PROJ_SLOTS):
            owner = self.proj_owner_emb(seq[f"proj{i}_owner"])
            type_ = self.proj_type_emb(seq[f"proj{i}_type"])
            subtype = self.proj_subtype_emb(seq[f"proj{i}_subtype"])
            numeric = seq[f"{i}_numeric"]  # note: shard key is "{i}_numeric"
            proj_parts.append(torch.cat([owner, type_, subtype, numeric], dim=-1))

        # Top-level "numeric" key = [distance, frame, 18 stage_geom]
        global_num = seq["numeric"]
        if global_num.shape[-1] != self.GLOBAL_NUM_DIM:
            raise ValueError(
                f"ValueEncoder expects {self.GLOBAL_NUM_DIM} global numeric cols; "
                f"got width {global_num.shape[-1]}"
            )

        # Controllers (both)
        self_ctrl = seq["self_controller"]
        opp_ctrl = seq["opp_controller"]

        parts = [
            stage, self_char, opp_char, self_action, opp_action,
            self_pn, opp_pn,
            self_ae, opp_ae,
            *proj_parts,
            global_num,
            self_ctrl, opp_ctrl,
        ]
        combined = torch.cat(parts, dim=-1)
        if self._use_input_gate:
            gate = torch.sigmoid(self.input_gate_logits)
            combined = combined * gate
        return self.drop(self.proj(combined))

    def gate_l1_penalty(self) -> torch.Tensor:
        if not self._use_input_gate:
            return self.proj.weight.new_zeros(())
        return torch.sigmoid(self.input_gate_logits).mean()

    @torch.no_grad()
    def gate_values(self) -> Optional[torch.Tensor]:
        if not self._use_input_gate:
            return None
        return torch.sigmoid(self.input_gate_logits).detach().cpu()

    def gate_report(self) -> List[Tuple[str, float]]:
        if not self._use_input_gate:
            return []
        vals = self.gate_values().tolist()
        return sorted(zip(self._feature_names, vals), key=lambda x: x[1])
