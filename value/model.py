"""WindowedValueModel — windowed transformer V(s) over fox_all_v2.

Replaces the earlier Markov MLP. Architecture mirrors the MIMIC BC backbone
(Shaw relative-position attention, d_model=512, 6 layers, 8 heads) so
capacity isn't the bottleneck. The BC controller heads are stripped and
replaced with a scalar BCE-with-logits head applied to the last-position
hidden state.

The transformer body is lifted from `mimic/model.py:MimicTransformerBlock`
(relpos, pre-norm, GELU FFN) unchanged. Only the encoder (ValueEncoder
instead of MimicFlatEncoder) and the head (scalar instead of controller)
differ from the BC FramePredictor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from mimic.frame_encoder import MimicFlatEncoder
from mimic.model import MimicTransformerBlock, SwiGLU
from value.encoder import ValueEncoder


@dataclass
class _BlockCfg:
    """Minimal config object for MimicTransformerBlock; mirrors the
    fields it reads off of `mimic.model.ModelConfig`."""
    d_model: int
    nhead: int
    dim_feedforward: int
    dropout: float
    use_rmsnorm: bool = False
    use_swiglu: bool = False


class WindowedValueModel(nn.Module):
    """V(s) over a 60-frame (or arbitrary T) window of game state.

    Forward returns a (B,) logit tensor per window — the last-position
    transformer output projected to a scalar via a small MLP head.
    """

    def __init__(
        self,
        *,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        head_hidden: int = 512,
        head_layers: int = 2,
        use_input_gate: bool = False,
        # Encoder/categorical sizes (defaults match fox_all_v2 schema)
        num_stages: int = 6,
        num_characters: int = 27,
        num_actions: int = 396,
        num_proj_types: int = 103,
        num_proj_subtypes: int = 40,
        num_proj_owners: int = 4,
        n_controller_combos: int = 7,
    ):
        super().__init__()

        self.encoder = ValueEncoder(
            d_model=d_model,
            dropout=dropout,
            num_stages=num_stages,
            num_characters=num_characters,
            num_actions=num_actions,
            num_proj_types=num_proj_types,
            num_proj_subtypes=num_proj_subtypes,
            num_proj_owners=num_proj_owners,
            n_controller_combos=n_controller_combos,
            use_input_gate=use_input_gate,
        )

        block_cfg = _BlockCfg(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
        )
        self.blocks = nn.ModuleList([
            MimicTransformerBlock(block_cfg) for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # Scalar head over the last-position hidden state
        head: list = [nn.LayerNorm(d_model)]
        in_dim = d_model
        for _ in range(head_layers):
            head += [nn.Linear(in_dim, head_hidden),
                     nn.GELU(),
                     nn.Dropout(dropout)]
            in_dim = head_hidden
        head.append(nn.Linear(in_dim, 1))
        self.scalar_head = nn.Sequential(*head)

        # Init (mirrors FramePredictor). Skip residual-scaling when there
        # are no transformer blocks (degenerate Markov-on-full case).
        self.apply(self._init_weights)
        if num_layers > 0:
            residual_std = 0.02 / math.sqrt(2 * num_layers)
            for blk in self.blocks:
                nn.init.normal_(blk.self_attn.c_proj.weight, std=residual_std)
                if isinstance(blk.mlp, SwiGLU):
                    nn.init.normal_(blk.mlp.w_down.weight, std=residual_std)
                else:
                    nn.init.normal_(blk.mlp.c_proj.weight, std=residual_std)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, state: dict) -> torch.Tensor:
        """state: dict of (B, T, ...) tensors. Returns (B,) logits."""
        x = self.encoder(state)            # (B, T, d_model)
        for blk in self.blocks:
            x = blk(x)
        x = self.final_norm(x)             # (B, T, d_model)
        # Take last-position hidden; collapse T → scalar logit per window
        last = x[:, -1, :]                 # (B, d_model)
        logit = self.scalar_head(last).squeeze(-1)  # (B,)
        return logit


class MarkovValueModel(nn.Module):
    """Per-frame Markov V(s) over the BC feature subset.

    Uses MimicFlatEncoder (same encoder BC uses) → MLP head → scalar logit.
    Reproduces the architecture of the 0.6005-val baseline.

    Input shape contract: state dict of (B, T=1, ...) tensors (consistent
    with the windowed dataset emitting W=1 slices). Output is (B,).
    """

    def __init__(
        self,
        *,
        d_model: int = 512,
        dropout: float = 0.1,
        head_hidden: int = 512,
        head_layers: int = 2,
        use_input_gate: bool = False,
        num_stages: int = 6,
        num_characters: int = 27,
        num_actions: int = 396,
        n_controller_combos: int = 7,
    ):
        super().__init__()
        self.encoder = MimicFlatEncoder(
            d_model=d_model,
            dropout=dropout,
            num_stages=num_stages,
            num_characters=num_characters,
            num_actions=num_actions,
            mimic_minimal_features=False,
            mimic_controller_encoding=True,
            n_controller_combos=n_controller_combos,
            use_input_gate=use_input_gate,
        )
        layers = [nn.LayerNorm(d_model)]
        in_dim = d_model
        for _ in range(head_layers):
            layers += [nn.Linear(in_dim, head_hidden),
                       nn.GELU(),
                       nn.Dropout(dropout)]
            in_dim = head_hidden
        layers.append(nn.Linear(in_dim, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, state: dict) -> torch.Tensor:
        h = self.encoder(state)             # (B, T, d_model), T=1
        logits = self.head(h).squeeze(-1)   # (B, T)
        return logits[:, -1]                # (B,) — return last-position scalar
