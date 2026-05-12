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
    """Per-column heads predicting state[t+1].

    All heads take the shared transformer output `h` of shape (B, T, d_model)
    and return per-frame predictions. No autoregressive chain — predictions
    are conditionally independent given `h`.

    Heads:
      - action (396-way CE)            × 2 players
      - numeric (n_numeric, MSE/Huber) × 2 players
      - flags (n_flags, BCEwithLogits) × 2 players
      - action_elapsed (scalar, MSE)   × 2 players, enabled via
        predict_action_elapsed=True. The shard stores this as a
        frames-since-action-change counter; we predict its value at t+1.
        Action CE alone has nothing to learn on intra-animation frames
        (most frames), so the elapsed head gives a softer continuous
        signal for "how close are we to the next transition?".
    """

    # Column indices inside the 13-col `self_numeric` that become CE heads
    # when discretize_counters=True. The rest stay on Huber regression.
    DISC_NUMERIC_COLS = (2, 3, 4, 10, 11)  # percent, stock, jumps, hitlag, hitstun
    CONTINUOUS_NUMERIC_COLS = (0, 1, 5, 6, 7, 8, 9, 12)  # pos_x, pos_y, 5 speeds, shield
    DISC_BIN_SIZES = {  # must match wm_dataset.DISC_BINS
        "percent": 237, "stock": 5, "jumps": 7,
        "hitlag": 21, "hitstun": 62, "elapsed": 62,
    }

    def __init__(
        self,
        d_model: int,
        num_actions: int,
        n_numeric: int = 13,
        n_flags: int = 5,
        predict_action_elapsed: bool = True,
        action_elapsed_scale: float = 30.0,
        discretize_counters: bool = False,
    ) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.n_numeric = n_numeric
        self.n_flags = n_flags
        self.predict_action_elapsed = predict_action_elapsed
        self.discretize_counters = discretize_counters
        # Scale the raw counter so regression target sits in Huber range.
        # Unused when discretize_counters=True (elapsed moves to CE).
        self.action_elapsed_scale = action_elapsed_scale

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

        # Numeric regression (normalized space). When discretize_counters=True
        # the integer-counter columns move to dedicated CE heads (below), so
        # the numeric head shrinks to just the continuous cols.
        if discretize_counters:
            assert n_numeric == 13, (
                "discretize_counters assumes the 13-col schema "
                "(percent/stock/jumps/hitlag/hitstun at fixed indices)."
            )
            numeric_out = len(self.CONTINUOUS_NUMERIC_COLS)  # 8
        else:
            numeric_out = n_numeric
        self.numeric_out = numeric_out
        self.self_numeric_head = _head(d_model, numeric_out)
        self.opp_numeric_head = _head(d_model, numeric_out)

        # Binary flags (BCE with logits)
        self.self_flags_head = _head(d_model, n_flags)
        self.opp_flags_head = _head(d_model, n_flags)

        if predict_action_elapsed and not discretize_counters:
            # Scalar Huber regression (old behavior).
            self.self_action_elapsed_head = _head(d_model, 1)
            self.opp_action_elapsed_head = _head(d_model, 1)

        if discretize_counters:
            # CE heads for each discretized counter column + action_elapsed.
            bins = self.DISC_BIN_SIZES
            for side in ("self", "opp"):
                self.add_module(f"{side}_percent_head", _head(d_model, bins["percent"]))
                self.add_module(f"{side}_stock_head",   _head(d_model, bins["stock"]))
                self.add_module(f"{side}_jumps_head",   _head(d_model, bins["jumps"]))
                self.add_module(f"{side}_hitlag_head",  _head(d_model, bins["hitlag"]))
                self.add_module(f"{side}_hitstun_head", _head(d_model, bins["hitstun"]))
                self.add_module(f"{side}_elapsed_head", _head(d_model, bins["elapsed"]))

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = dict(
            self_action_logits=self.self_action_head(h),
            opp_action_logits=self.opp_action_head(h),
            self_numeric_pred=self.self_numeric_head(h),
            opp_numeric_pred=self.opp_numeric_head(h),
            self_flags_logits=self.self_flags_head(h),
            opp_flags_logits=self.opp_flags_head(h),
        )
        if self.predict_action_elapsed and not self.discretize_counters:
            out["self_action_elapsed_pred"] = (
                self.self_action_elapsed_head(h).squeeze(-1)
            )
            out["opp_action_elapsed_pred"] = (
                self.opp_action_elapsed_head(h).squeeze(-1)
            )
        if self.discretize_counters:
            for side in ("self", "opp"):
                for name in ("percent", "stock", "jumps", "hitlag", "hitstun", "elapsed"):
                    out[f"{side}_{name}_logits"] = getattr(self, f"{side}_{name}_head")(h)
        return out


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

        # Opponent current-frame controller: symmetric with self_controller
        # (56-dim baked one-hot). Gated on cfg.no_opp_inputs — when False, the
        # WM sees both players' current-frame controllers, matching what
        # Melee's engine has access to when it computes state[t+1]. Requires
        # shards with `opp_controller` baked in (add_opp_controller_to_shards).
        include_opp_controller = not cfg.no_opp_inputs

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
            include_opp_controller=include_opp_controller,
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
            discretize_counters=getattr(cfg, "discretize_counters", False),
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
