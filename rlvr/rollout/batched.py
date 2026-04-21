"""Single-frame batched rollout for RLVR.

For each Prompt:
  1. Materialize the T-frame state_context into a tensor batch.
  2. Replicate N times along the batch axis (same context, N samples).
  3. Forward the policy model AND the frozen reference model.
  4. Sample from each head's logits in the within-frame autoregressive
     order (shoulder -> c_stick -> main_stick -> buttons).
  5. Compute log-prob of the sampled index under each model, sum across
     the 4 heads per rollout -> (B*N,) logprob vectors.
  6. Decode sampled class indices back to ControllerInput objects for
     the verifier.

No K-frame autoregression. The rollout is one frame per prompt (see
plan: offline RLVR requires single-frame rollouts).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from mimic.features import (
    HAL_CSTICK_CLUSTERS_9,
    HAL_SHOULDER_CLUSTERS_3,
    HAL_STICK_CLUSTERS_37,
)
from rlvr.rollout.frame_builder import build_context_batch
from rlvr.state.gamestate import ControllerInput
from rlvr.tasks.base import Prompt


@dataclass
class RolloutBatch:
    prompts: List[Prompt]                    # length B
    sampled_ctrls: List[ControllerInput]     # length B*N, row-major by prompt
    sampled_indices: torch.Tensor            # (B*N, 4) — main, shldr, cdir, btn
    logprobs_theta: torch.Tensor             # (B*N,) sum over 4 heads
    logprobs_ref: torch.Tensor               # (B*N,) under frozen ref
    n_per_prompt: int


def _sample_head(
    logits: torch.Tensor,                  # (B, V)
    temperature: float,
    generator: Optional[torch.Generator],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample class indices and return (sampled_idx, logprob_of_sample).

    If temperature is <= 0 we return argmax (greedy). Guards against
    NaN/Inf logits (undertrained-model safety, matching
    inference_utils._safe_sample)."""
    logits = logits.float()
    if temperature <= 0:
        idx = logits.argmax(dim=-1)  # (B,)
        log_probs = F.log_softmax(logits, dim=-1)
        lp = log_probs.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
        return idx, lp

    safe = torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)
    probs = F.softmax(safe / temperature, dim=-1)
    # If any row is all-zero after softmax, fallback to argmax for that row.
    row_ok = probs.sum(dim=-1) > 0
    if not bool(row_ok.all()):
        idx = torch.where(
            row_ok,
            torch.multinomial(probs.clamp_min(1e-30), 1, generator=generator).squeeze(-1)
                if generator is not None
                else torch.multinomial(probs.clamp_min(1e-30), 1).squeeze(-1),
            safe.argmax(dim=-1),
        )
    else:
        if generator is not None:
            idx = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
        else:
            idx = torch.multinomial(probs, 1).squeeze(-1)
    # logprob under the un-temperature-scaled distribution (= policy log-prob)
    log_probs = F.log_softmax(logits, dim=-1)
    lp = log_probs.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
    return idx, lp


def _logprob_of_idx(
    logits: torch.Tensor,   # (B, V)
    idx: torch.Tensor,       # (B,)
) -> torch.Tensor:
    logits = logits.float()
    safe = torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)
    log_probs = F.log_softmax(safe, dim=-1)
    return log_probs.gather(-1, idx.unsqueeze(-1)).squeeze(-1)


def _decode_controller(
    main_idx: int, shldr_idx: int, cdir_idx: int, btn_idx: int,
    n_btn: int,
) -> ControllerInput:
    """Map sampled class indices -> ControllerInput.

    Mirrors the decode logic in tools/inference_utils.decode_and_press
    but produces a RLVR dataclass instead of pressing a libmelee ctrl.
    """
    mx_01 = float(HAL_STICK_CLUSTERS_37[main_idx][0])   # [0, 1]
    my_01 = float(HAL_STICK_CLUSTERS_37[main_idx][1])
    cx_01 = float(HAL_CSTICK_CLUSTERS_9[cdir_idx][0])
    cy_01 = float(HAL_CSTICK_CLUSTERS_9[cdir_idx][1])
    shldr = float(HAL_SHOULDER_CLUSTERS_3[shldr_idx])

    # Convert [0, 1] libmelee convention -> [-1, 1] peppi/ControllerInput.
    def _01_to_pm1(v: float) -> float:
        return v * 2.0 - 1.0

    # Start from neutral then set buttons per the 7-class decode.
    kwargs = dict(
        main_x=_01_to_pm1(mx_01), main_y=_01_to_pm1(my_01),
        c_x=_01_to_pm1(cx_01), c_y=_01_to_pm1(cy_01),
        shoulder_l=shldr, shoulder_r=0.0,
        a_button=False, b_button=False, x_button=False, y_button=False,
        z_button=False, l_button=False, r_button=False, start_button=False,
        d_up=False, d_down=False, d_left=False, d_right=False,
    )
    if n_btn == 7:
        if btn_idx == 0: kwargs["a_button"] = True
        elif btn_idx == 1: kwargs["b_button"] = True
        elif btn_idx == 2: kwargs["z_button"] = True
        elif btn_idx == 3: kwargs["x_button"] = True  # JUMP
        elif btn_idx == 4: kwargs["l_button"] = True  # TRIG
        elif btn_idx == 5:
            kwargs["a_button"] = True
            kwargs["l_button"] = True
    else:
        if btn_idx == 0: kwargs["a_button"] = True
        elif btn_idx == 1: kwargs["b_button"] = True
        elif btn_idx == 2: kwargs["x_button"] = True  # JUMP
        elif btn_idx == 3: kwargs["z_button"] = True

    return ControllerInput(**kwargs)


def rollout(
    model,
    ref_model,
    prompts: List[Prompt],
    n_per_prompt: int,
    ctx: dict,
    temperature: float = 1.0,
    seed: Optional[int] = None,
    device: str = "cuda",
) -> RolloutBatch:
    """Run a single-frame batched rollout across prompts.

    Args:
        model: trainable policy (FramePredictor in train mode or eval —
            gradients flow from this call's outputs).
        ref_model: frozen reference model (wrapped in no_grad internally).
        prompts: list of B prompts.
        n_per_prompt: N samples per prompt (group size for GRPO).
        ctx: inference context from load_inference_context(data_dir).
        temperature: sampling temperature.
        seed: optional RNG seed for reproducibility.
        device: 'cuda' or 'cpu'.

    Returns:
        RolloutBatch with one row per (prompt, sample).
    """
    B = len(prompts)
    N = n_per_prompt
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    # Build (T, feat) tensor batch for each prompt, stack to (B, T, feat).
    per_prompt_batches = []
    for p in prompts:
        stk = build_context_batch(p.state_context, p.player_port, ctx)
        per_prompt_batches.append(stk)

    # Stack into (B, T, ...). Each value: stk[k] is (T, D) or (T,). We
    # unsqueeze at dim=0 and concat.
    batch: Dict[str, torch.Tensor] = {}
    for k in per_prompt_batches[0]:
        tensors = [pb[k].unsqueeze(0) for pb in per_prompt_batches]  # each (1, T, ...)
        batch[k] = torch.cat(tensors, dim=0).to(device)  # (B, T, ...)

    # Replicate along batch axis to (B*N, T, ...).
    if N > 1:
        batch = {k: v.repeat_interleave(N, dim=0) for k, v in batch.items()}

    # Forward both models.
    theta_logits = model(batch)
    with torch.no_grad():
        ref_logits = ref_model(batch)

    # Extract logits at the last time position.
    def _last(t: torch.Tensor) -> torch.Tensor:
        # t shape (B*N, T, V) or (B*N, V)
        if t.dim() == 3:
            return t[:, -1, :]
        return t

    # Sample from policy. Autoregressive order (shoulder -> cdir ->
    # main -> btn) doesn't require re-running forward: the model's
    # internal detached-chain already conditions each head on its
    # predecessors via the latent in `h`. We sample each head
    # independently from the emitted logits.
    shldr_l = _last(theta_logits["shoulder_val"])
    cdir_l = _last(theta_logits["c_dir_logits"])
    main_l = _last(theta_logits["main_xy"])
    btn_l = _last(theta_logits["btn_logits"])

    shldr_idx, lp_sh = _sample_head(shldr_l, temperature, generator)
    cdir_idx, lp_cd = _sample_head(cdir_l, temperature, generator)
    main_idx, lp_mn = _sample_head(main_l, temperature, generator)
    btn_idx, lp_bt = _sample_head(btn_l, temperature, generator)

    logprobs_theta = lp_sh + lp_cd + lp_mn + lp_bt  # (B*N,)

    # Compute ref log-probs at the sampled indices.
    lp_ref_sh = _logprob_of_idx(_last(ref_logits["shoulder_val"]), shldr_idx)
    lp_ref_cd = _logprob_of_idx(_last(ref_logits["c_dir_logits"]), cdir_idx)
    lp_ref_mn = _logprob_of_idx(_last(ref_logits["main_xy"]), main_idx)
    lp_ref_bt = _logprob_of_idx(_last(ref_logits["btn_logits"]), btn_idx)
    logprobs_ref = (lp_ref_sh + lp_ref_cd + lp_ref_mn + lp_ref_bt).detach()

    # Decode to ControllerInput.
    n_btn = int(btn_l.shape[-1])
    sampled_indices = torch.stack(
        [main_idx, shldr_idx, cdir_idx, btn_idx], dim=-1
    )  # (B*N, 4)
    sampled_ctrls: List[ControllerInput] = []
    idx_np = sampled_indices.detach().cpu().numpy()
    for m, s, c, b in idx_np:
        sampled_ctrls.append(
            _decode_controller(int(m), int(s), int(c), int(b), n_btn)
        )

    return RolloutBatch(
        prompts=prompts,
        sampled_ctrls=sampled_ctrls,
        sampled_indices=sampled_indices,
        logprobs_theta=logprobs_theta,
        logprobs_ref=logprobs_ref,
        n_per_prompt=N,
    )
