"""Rollout acceptance test.

Gate: for a greedy (temperature -> 0) rollout, `logprobs_theta` must
equal the sum of `log_softmax(logits)[argmax]` across the 4 heads,
within float tolerance.

Runs on CPU for CI reproducibility. Uses a 2-prompt batch with N=2
samples per prompt to exercise the repeat_interleave path.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from rlvr.rollout import rollout
from rlvr.sampler import sample_states
from rlvr.tasks import list_tasks
from tools.inference_utils import load_inference_context, load_mimic_model


_EVENTS_PARQUET = Path("/tmp/rlvr_test_cache/events_smoke.parquet")
_SLP_DIR = Path("/tmp/rlvr_test_cache/slp")
_CKPT = Path("/root/MIMIC/checkpoints/fox-20260420-baseline-33k.pt")
_DATA_DIR = Path("/root/MIMIC/hf_checkpoints/fox")


@pytest.fixture(scope="module")
def loaded_model():
    if not _CKPT.exists():
        pytest.skip(f"checkpoint missing: {_CKPT}")
    if not _EVENTS_PARQUET.exists():
        pytest.skip(f"events parquet missing: {_EVENTS_PARQUET} — run tagger smoke test")
    device = "cpu"
    model, cfg = load_mimic_model(str(_CKPT), device)
    ref_model, _ = load_mimic_model(str(_CKPT), device)
    for p in ref_model.parameters():
        p.requires_grad_(False)
    ctx = load_inference_context(_DATA_DIR)
    return {"model": model, "ref_model": ref_model, "cfg": cfg, "ctx": ctx, "device": device}


def test_greedy_logprobs_match_manual_argmax(loaded_model):
    prompts = sample_states(
        events_path=_EVENTS_PARQUET,
        slp_dir=_SLP_DIR,
        task_id="l_cancel_opportunity",
        n=2,
        seed=42,
    )
    assert len(prompts) >= 2

    model = loaded_model["model"]
    ref_model = loaded_model["ref_model"]
    ctx = loaded_model["ctx"]
    device = loaded_model["device"]

    model.eval()
    rb = rollout(
        model, ref_model, prompts,
        n_per_prompt=2,
        ctx=ctx,
        temperature=0.0,  # greedy
        device=device,
    )

    # Sanity: shape (B*N=4,)
    assert rb.logprobs_theta.shape == (4,)
    assert rb.logprobs_ref.shape == (4,)
    assert rb.sampled_indices.shape == (4, 4)

    # With greedy sampling and model == ref_model, theta and ref should
    # produce identical values (up to float precision).
    diff = (rb.logprobs_theta - rb.logprobs_ref).abs().max().item()
    assert diff < 1e-4, f"theta vs ref diff too large: {diff}"

    # Values should be non-positive (log of a probability in [0, 1]).
    assert (rb.logprobs_theta <= 1e-6).all()

    # Verify: greedy logprobs_theta == sum of log_softmax[argmax] over
    # the 4 heads. We recompute by re-running the forward pass.
    from rlvr.rollout.frame_builder import build_context_batch
    # Build batch the same way rollout does (for the same prompts +
    # replication).
    per_prompt = []
    for p in prompts:
        stk = build_context_batch(p.state_context, p.player_port, ctx)
        per_prompt.append(stk)
    batch = {}
    for k in per_prompt[0]:
        tensors = [pb[k].unsqueeze(0) for pb in per_prompt]
        batch[k] = torch.cat(tensors, dim=0).to(device)
    batch = {k: v.repeat_interleave(2, dim=0) for k, v in batch.items()}

    with torch.no_grad():
        logits = model(batch)

    def _sum_argmax_lp(head_logits: torch.Tensor) -> torch.Tensor:
        if head_logits.dim() == 3:
            head_logits = head_logits[:, -1, :]
        head_logits = torch.nan_to_num(head_logits.float())
        log_probs = F.log_softmax(head_logits, dim=-1)
        return log_probs.max(dim=-1).values

    expected = (
        _sum_argmax_lp(logits["shoulder_val"])
        + _sum_argmax_lp(logits["c_dir_logits"])
        + _sum_argmax_lp(logits["main_xy"])
        + _sum_argmax_lp(logits["btn_logits"])
    )

    # Compare greedy rollout logprobs to manual argmax-sum.
    max_err = (rb.logprobs_theta.detach() - expected.detach()).abs().max().item()
    assert max_err < 1e-4, f"greedy logprobs mismatch: max_err={max_err}"


def test_stochastic_rollout_runs_and_shapes_are_right(loaded_model):
    prompts = sample_states(
        events_path=_EVENTS_PARQUET,
        slp_dir=_SLP_DIR,
        task_id="l_cancel_opportunity",
        n=3,
        seed=0,
    )
    model = loaded_model["model"]
    ref_model = loaded_model["ref_model"]
    ctx = loaded_model["ctx"]
    device = loaded_model["device"]

    model.eval()
    rb = rollout(
        model, ref_model, prompts,
        n_per_prompt=4,
        ctx=ctx,
        temperature=1.0,
        seed=1234,
        device=device,
    )
    assert len(rb.sampled_ctrls) == len(prompts) * 4
    assert rb.logprobs_theta.shape == (len(prompts) * 4,)
    # All rollouts produce valid controller dataclasses.
    for c in rb.sampled_ctrls:
        # Trigger values in sensible range.
        assert 0.0 <= c.shoulder_l <= 1.0
        assert 0.0 <= c.shoulder_r <= 1.0
