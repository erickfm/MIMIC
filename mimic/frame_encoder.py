# frame_encoder.py – the MIMIC frame encoder
# -----------------------------------------------------------------------------
# Converts structured Melee frame dictionaries -> (B, T, d_model) tensor:
# categorical embeddings + per-player gamestate + controller one-hot, concatenated
# and projected by one Linear (MimicFlatEncoder). The older token/attention
# encoders ("default", "flat", "composite8", "hybrid16") were removed 2026-10-22.
# -----------------------------------------------------------------------------

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class MimicFlatEncoder(nn.Module):
    """Flat concat + Linear input projection.

    Per frame:
      concat([stage_emb(4), char_emb(12)*2, action_emb(32)*2,
              gamestate(numeric), controller(one-hot)]) → Linear → d_model

    No per-group MLPs. Categorical embeddings are small. Gamestate is the full
    13 numeric + 5 flag columns per player (the minimal 9-column path was removed
    2026-10-22).
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
        mimic_controller_encoding: bool = True,
        n_controller_combos: int = 5,
        no_self_inputs: bool = False,
        use_input_gate: bool = False,
        # Legacy aliases
        hal_controller_encoding: bool = None,
        **kwargs,  # absorb unused params from build_encoder
    ) -> None:
        super().__init__()
        if hal_controller_encoding is not None:
            mimic_controller_encoding = hal_controller_encoding
        self._no_self_inputs = no_self_inputs
        self._ctrl_enc = mimic_controller_encoding
        # Legacy alias
        self._hal_ctrl_enc = self._ctrl_enc

        # Categorical embeddings (HAL sizes)
        self.stage_emb = nn.Embedding(num_stages, self.STAGE_EMB_DIM)
        self.char_emb = nn.Embedding(num_characters, self.CHAR_EMB_DIM)
        self.action_emb = nn.Embedding(num_actions, self.ACTION_EMB_DIM)

        # Compute total input dimension
        emb_dim = self.STAGE_EMB_DIM + 2 * self.CHAR_EMB_DIM + 2 * self.ACTION_EMB_DIM
        # = 4 + 24 + 64 = 92

        # Gamestate per player (schema as of 2026-04-20, after dropping invuln_left
        # + 8 ECB corners): 13 numeric + 5 flags = 18, × 2 players.
        numeric_dim = 18 * 2

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

        # Numeric gamestate. Shard stores 13 numeric + 5 flag cols per player
        # (see mimic/features.py:numeric_state + flags for the full column list).
        # HAL normalizes boolean flags: 0→-1, 1→+1 (normalize transform with min=0,max=1)
        sn = seq["self_numeric"]
        on = seq["opp_numeric"]
        sf_all = seq["self_flags"].float() * 2.0 - 1.0
        of_all = seq["opp_flags"].float() * 2.0 - 1.0

        # Full: 13 numeric + 5 flags per player. No HAL reorder (a single
        # learned Linear downstream makes ordering functionally irrelevant).
        if sn.shape[-1] != 13:
            raise ValueError(
                f"MimicFlatEncoder(fullfeat): requires 13-col self_numeric "
                f"shards (post 2026-04-20 schema drop); got shape[-1]={sn.shape[-1]}. "
                f"Legacy 22-col fullfeat shards / checkpoints need reshard or retrain."
            )
        self_num = torch.cat([sn, sf_all], dim=-1)  # (B, T, 18)
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

        # Full: 13 numeric + 5 flags per player in native order
        # (schema as of 2026-04-20 — see numeric_state in features.py).
        num_cols = [
            "pos_x", "pos_y", "percent", "stock", "jumps_left",
            "speed_air_x_self", "speed_ground_x_self",
            "speed_x_attack", "speed_y_attack", "speed_y_self",
            "hitlag_left", "hitstun_left",
            "shield_strength",
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
    "mimic_flat": MimicFlatEncoder,
    "hal_flat": MimicFlatEncoder,   # pre-rename name, still in older checkpoints' configs
}


def build_encoder(
    encoder_type: str = "mimic_flat",
    *,
    d_model: int = 512,
    dropout: float = 0.0,
    no_self_inputs: bool = False,
    mimic_controller_encoding: bool = True,
    n_controller_combos: int = 5,
    use_input_gate: bool = False,
    num_stages: int,
    num_characters: int,
    num_actions: int,
) -> MimicFlatEncoder:
    cls = ENCODER_REGISTRY.get(encoder_type)
    if cls is None:
        raise ValueError(f"Unknown encoder type: {encoder_type!r} (only mimic_flat exists since "
                         f"2026-10-22; available: {list(ENCODER_REGISTRY)})")
    return cls(d_model=d_model, dropout=dropout, num_stages=num_stages,
               num_characters=num_characters, num_actions=num_actions,
               mimic_controller_encoding=mimic_controller_encoding,
               n_controller_combos=n_controller_combos, no_self_inputs=no_self_inputs,
               use_input_gate=use_input_gate)
