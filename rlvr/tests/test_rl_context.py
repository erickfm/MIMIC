"""RL re-scoring is exact for every model kind (2026-10-23).

For full-causal and sliding-window models, with and without --cond-heads: stream frames through
rlvr.sim.batch.BatchedPolicy (fp32, one env reset mid-stream), then re-score each decision from
BatchedPolicy.context() with rlvr.sim.ppo.head_logits, as the PPO update does. The logits must
equal the ones the action was sampled from. For sliding-window models, also: after a weight
change, BatchedPolicy.refresh() makes the next streamed step equal a fresh forward with the new
weights (and without it, it doesn't).

  python3 rlvr/tests/test_rl_context.py
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from inference_utils import build_mock_frame, load_inference_context  # noqa: E402
from mimic.model import MODEL_PRESETS, FramePredictor, ModelConfig  # noqa: E402
from rlvr.sim.batch import HEADS, BatchedPolicy  # noqa: E402
from rlvr.sim.ppo import collate, head_logits  # noqa: E402

N_ENVS, STEPS, RESET_AT, CHECK_AT = 4, 1300, 700, (5, 400, 1100, 1299)


def make(window, cond):
    o = dict(MODEL_PRESETS["S"])
    seq = o.pop("max_seq_len")
    cfg = ModelConfig(max_seq_len=seq, mimic_mode=True, no_opp_inputs=True, no_self_inputs=False,
                      hal_controller_encoding=True, n_controller_combos=7, n_stick_clusters=37,
                      cond_heads=cond, attn_window=window, **o)
    return FramePredictor(cfg).cuda().eval(), cfg


def frame_feats(mock, g):
    f = {}
    for k, v in mock.items():
        if k == "self_controller":
            continue
        x = v[0].clone().repeat(N_ENVS, *[1] * (v.dim() - 1))
        if k in ("self_action", "opp_action"):
            x = torch.randint(0, 396, x.shape, generator=g)
        elif k.endswith("numeric"):
            x = torch.randn(x.shape, generator=g)
        elif k.endswith("flags"):
            x = (torch.rand(x.shape, generator=g) < 0.5).float()
        f[k] = x.numpy()
    return f


def ppo_order(i):   # BatchedPolicy (main, c, shoulder, button) -> PPO (main, shoulder, c, button)
    return i[:, [0, 2, 1, 3]]


def rescore_err(model, pol, stash):
    ctx = pol.context(list(range(N_ENVS)))
    batch, last = collate(ctx)
    with torch.no_grad():
        out = head_logits(model, batch, last, ppo_order(pol.last_idx))
    return max(float((out[h] - stash[h]).abs().max()) for h in HEADS)


def run(window, cond, mock):
    torch.manual_seed(0)
    model, cfg = make(window, cond)
    pol = BatchedPolicy(model, cfg.max_seq_len, {k: v.cuda() for k, v in mock.items()}, N_ENVS,
                        amp=False, compile=False)
    pol.use_graph = False
    stash = {}
    orig = pol._sample

    def sample(preds):
        stash.update({h: preds[h][:, -1].float().clone() for h in HEADS})
        orig(preds)
    pol._sample = sample
    g = torch.Generator().manual_seed(1)
    pol.reset(range(N_ENVS))
    errs = []
    for t in range(STEPS):
        if t == RESET_AT:
            pol.reset([2])
        pol.step(frame_feats(mock, g))
        if t in CHECK_AT:
            errs.append(rescore_err(model, pol, stash))
    res = dict(rescore=max(errs))
    if window:
        with torch.no_grad():   # an "optimizer step"
            for p in model.parameters():
                p.add_(0.02 * torch.randn_like(p))
        pol.step(frame_feats(mock, g))
        res["stale_after_update"] = rescore_err(model, pol, stash)
        pol.refresh()
        pol.step(frame_feats(mock, g))
        res["after_refresh"] = rescore_err(model, pol, stash)
    return res


def main():
    mock = build_mock_frame(load_inference_context(str(ROOT / "hf_eval" / "fox-master")))
    ok = True
    for window in (0, 180):
        for cond in (False, True):
            r = run(window, cond, mock)
            good = r["rescore"] < 1e-3 and (not window or (r["after_refresh"] < 1e-3 and r["stale_after_update"] > 1e-2))
            ok &= good
            print(f"{'stream' if window else 'full':6s} cond={cond!s:5s} "
                  + "  ".join(f"{k} {v:.2e}" for k, v in r.items()) + ("" if good else "  FAIL"))
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
