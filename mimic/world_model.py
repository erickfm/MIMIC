"""World-model heads + `WorldModel` predictor.

Predicts `state[t+1]` given `state[t]` (encoded by the standard
`MimicFlatEncoder`) plus both players' controllers at `t+1` (injected as
conditioning via the encoder's `next_ctrl_dim` hook).

Scope is deliberately narrow: we predict only the state fields the BC
encoder consumes — action + numeric + flags for each player. Projectiles,
action_elapsed, Nana state, and static fields (stage/character/port) are
not modeled.

Design:
- Reuses `FramePredictor`'s encoder + transformer + final LayerNorm.
- Swaps `MimicPredictionHeads` for `WorldModelHeads` (6 independent MLPs,
  no autoregressive chain — the targets are independent given the shared
  latent `h[t]`).

See `docs/research-notes-*.md` and the plan file for the broader rationale.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class WorldModelHeads(nn.Module):
    """Six per-column heads predicting state[t+1].

    All heads take the shared transformer output `h` of shape (B, T, d_model)
    and return per-frame predictions. No autoregressive chain — predictions
    are conditionally independent given `h`.
    """

    def __init__(
        self,
        d_model: int,
        num_actions: int,
        n_numeric: int = 13,
        n_flags: int = 5,
    ) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.n_numeric = n_numeric
        self.n_flags = n_flags

        def _head(in_dim: int, out_dim: int) -> nn.Sequential:
            h = in_dim // 2
            return nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, h),
                nn.GELU(),
                nn.Linear(h, out_dim),
            )

        # Categorical (396-way softmax per player)
        self.self_action_head = _head(d_model, num_actions)
        self.opp_action_head = _head(d_model, num_actions)

        # Numeric regression (13 per player, normalized space)
        self.self_numeric_head = _head(d_model, n_numeric)
        self.opp_numeric_head = _head(d_model, n_numeric)

        # Binary flags (5 per player, BCE with logits)
        self.self_flags_head = _head(d_model, n_flags)
        self.opp_flags_head = _head(d_model, n_flags)

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        return dict(
            self_action_logits=self.self_action_head(h),
            opp_action_logits=self.opp_action_head(h),
            self_numeric_pred=self.self_numeric_head(h),
            opp_numeric_pred=self.opp_numeric_head(h),
            self_flags_logits=self.self_flags_head(h),
            opp_flags_logits=self.opp_flags_head(h),
        )


class WorldModel(nn.Module):
    """Encoder → transformer → WorldModelHeads.

    Constructed by `mimic.model.get_model` when `cfg.wm_mode=True`. Reuses
    `FramePredictor`'s backbone verbatim, only swapping heads.
    """

    def __init__(self, cfg, encoder: Optional[nn.Module] = None) -> None:
        super().__init__()
        # Lazy import: avoid circular dep with mimic.model.
        from .model import (
            FramePredictor,
            MimicTransformerBlock,
            TransformerBlock,
            RMSNorm,
            _sinusoidal_embeddings,
        )
        from .frame_encoder import build_encoder

        self.cfg = cfg

        # Build encoder with world-model conditioning hook.
        # next_ctrl_dim: self_controller(37+9+combos+3) + opp_buttons(12) +
        #                opp_analog(4) + opp_c_dir(num_c_dirs one-hot)
        ctrl_dim = 37 + 9 + cfg.n_controller_combos + 3
        next_ctrl_dim = ctrl_dim + 12 + 4 + cfg.num_c_dirs

        self.encoder = encoder or build_encoder(
            encoder_type=cfg.encoder_type,
            d_model=cfg.d_model,
            d_intra=cfg.d_intra,
            dropout=cfg.dropout,
            nlayers=cfg.encoder_nlayers,
            k_query=cfg.k_query,
            scaled_emb=cfg.scaled_emb,
            num_stages=cfg.num_stages,
            num_ports=cfg.num_ports,
            num_characters=cfg.num_characters,
            num_actions=cfg.num_actions,
            num_costumes=cfg.num_costumes,
            num_proj_types=cfg.num_proj_types,
            num_proj_subtypes=cfg.num_proj_subtypes,
            num_c_dirs=cfg.num_c_dirs,
            no_opp_inputs=cfg.no_opp_inputs,
            no_self_inputs=cfg.no_self_inputs,
            lean_features=cfg.lean_features,
            mimic_minimal_features=cfg.mimic_minimal_features,
            mimic_controller_encoding=cfg.mimic_controller_encoding,
            n_controller_combos=cfg.n_controller_combos,
            use_input_gate=cfg.use_input_gate,
            next_ctrl_dim=next_ctrl_dim,
        )

        if cfg.pos_enc == "learned":
            self.pos_emb = nn.Parameter(
                torch.randn(1, cfg.max_seq_len, cfg.d_model) * 0.02
            )
        elif cfg.pos_enc == "sinusoidal":
            self.register_buffer(
                "pos_emb", _sinusoidal_embeddings(cfg.max_seq_len, cfg.d_model)
            )
        else:
            self.pos_emb = None

        use_relpos_block = cfg.pos_enc == "relpos"
        if use_relpos_block:
            self.blocks = nn.ModuleList(
                [MimicTransformerBlock(cfg) for _ in range(cfg.num_layers)]
            )
        else:
            self.blocks = nn.ModuleList(
                [TransformerBlock(cfg) for _ in range(cfg.num_layers)]
            )

        FinalNorm = RMSNorm if getattr(cfg, "use_rmsnorm", False) else nn.LayerNorm
        self.final_norm = FinalNorm(cfg.d_model)

        self.heads = WorldModelHeads(
            d_model=cfg.d_model,
            num_actions=cfg.num_actions,
            n_numeric=13 if not cfg.mimic_minimal_features else 6,
            n_flags=5 if not cfg.mimic_minimal_features else 3,
        )

        # Reuse FramePredictor's weight init (attention + FFN residual scaling).
        self.apply(FramePredictor._init_weights)
        import math as _math

        residual_std = 0.02 / _math.sqrt(2 * cfg.num_layers)
        for blk in self.blocks:
            if use_relpos_block:
                nn.init.normal_(blk.self_attn.c_proj.weight, std=residual_std)
                try:
                    nn.init.normal_(blk.mlp.c_proj.weight, std=residual_std)
                except AttributeError:
                    # SwiGLU path
                    nn.init.normal_(blk.mlp.w_down.weight, std=residual_std)
            else:
                nn.init.normal_(blk.self_attn.out_proj.weight, std=residual_std)
                try:
                    nn.init.normal_(blk.ff[-1].weight, std=residual_std)
                except (AttributeError, TypeError):
                    nn.init.normal_(blk.ff.w_down.weight, std=residual_std)

    def forward(
        self,
        frames: Dict[str, torch.Tensor],
        btn_targets: Optional[torch.Tensor] = None,  # unused; kept for API parity
    ) -> Dict[str, torch.Tensor]:
        x = self.encoder(frames)
        if self.pos_emb is not None:
            T = x.size(1)
            x = x + self.pos_emb[:, :T]
        for blk in self.blocks:
            x = blk(x)
        x = self.final_norm(x)
        return self.heads(x)
