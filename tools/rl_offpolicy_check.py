#!/usr/bin/env python3
"""How far is sim RL's rollout policy from the policy PPO scores? (2026-10-24)

sim_rl samples actions from the learner in bf16 inside a CUDA graph (stream models through a bf16
KV cache), then recomputes the PPO "old" log-probs in fp32 from BatchedPolicy.context(). If the two
differ, the PPO ratio is off-policy by that much. rlvr/tests/test_rl_context.py checks the logic in
fp32 on random inputs; this checks the production rollout path on real sim states.

It runs the learner against frozen BC in the sim exactly as sim_rl does (VecSim, amp, CUDA graph),
stashes the logits each sampled action was drawn from (inside the graph), and re-scores a random
sample of decisions with ppo.head_logits in fp32. --swap loads other weights into the live
learner mid-rollout and calls refresh(), as sim_rl does after every update, then measures again.

  python3 tools/rl_offpolicy_check.py --policy-dir checkpoints/smallfox/base_stream \
      [--weights RL_CKPT] [--swap RL_CKPT] [--fp32] --out out.json
"""
import argparse
import json
import random
import tempfile
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-dir", required=True, help="learner start (unless --weights), and the opponent")
    ap.add_argument("--weights", help="RL checkpoint to load into the learner before the rollout")
    ap.add_argument("--swap", help="after --warm frames, load these weights in place + refresh(), then measure")
    ap.add_argument("--fp32", action="store_true", help="rollout without bf16 autocast (control)")
    ap.add_argument("--templates", default="replays_doc_rt")
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--warm", type=int, default=1500, help="frames before measuring (with --swap: before the swap)")
    ap.add_argument("--frames", type=int, default=3000, help="frames measured")
    ap.add_argument("--pick", type=float, default=1 / 16, help="probability a learner decision is re-scored")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()

    from inference_utils import load_mimic_model
    from rlvr.sim.batch import HEADS
    from rlvr.sim.ppo import collate, head_logits
    from rlvr.sim.vec_env import VecSim
    from sim_h2h import make_ditto

    torch.manual_seed(a.seed)
    learner, cfg = load_mimic_model(str(Path(a.policy_dir) / "model.pt"), "cuda")
    if a.weights:
        learner.load_state_dict(torch.load(a.weights, map_location="cpu", weights_only=False)["model_state_dict"])
    tdir = Path(tempfile.mkdtemp(prefix="rl_offpolicy_"))
    templates = []
    for src in sorted(Path(a.templates).glob("*.slp")):
        make_ditto(src, "FOX", tdir / src.name)
        templates.append(tdir / src.name)
    assign = lambda e, side, ch: ("LEARN", "OPP")[(e + side) % 2]
    v = VecSim(templates, {"LEARN": dict(dir=a.policy_dir, model=learner, cfg=cfg, amp=not a.fp32, compile=False),
                           "OPP": dict(dir=a.policy_dir, amp=True, compile=True)},
               a.envs, a.workers, seed=a.seed, assign=assign)
    pol = v.policies["LEARN"]

    stash = {}
    orig = pol._sample

    def sample(preds):   # graph-capturable: buffers are allocated on the warm-up call, outside capture
        for h in HEADS:
            x = preds[h][:, -1].float()
            if h not in stash:
                stash[h] = torch.empty_like(x)
            stash[h].copy_(x)
        orig(preds)
    pol._sample = sample

    rng = random.Random(a.seed)
    rec = []          # (context, sampled idx in PPO order, {head: rollout logits})
    measuring = [False]

    def on_decide(name, idx, p):
        if name != "LEARN" or not measuring[0]:
            return
        slots = [s for s in range(p.n) if rng.random() < a.pick]
        if not slots:
            return
        ctx = p.context(slots)
        for j, s in enumerate(slots):
            m, c, sh, b = (int(x) for x in idx[s])
            rec.append((ctx[j], torch.tensor([m, sh, c, b]), {h: stash[h][s].cpu() for h in HEADS}))
    v.on_decide = on_decide

    for _ in range(a.warm):
        v.step()
    if a.swap:
        learner.load_state_dict(torch.load(a.swap, map_location="cpu", weights_only=False)["model_state_dict"])
        pol.refresh()
    measuring[0] = True
    for _ in range(a.frames):
        v.step()
    v.close()

    # fp32 re-score, as sim_rl computes the PPO "old" log-probs
    learner.eval()
    kl, dlp, roll_lp = [], [], []
    per_head = {h: [] for h in HEADS}
    with torch.no_grad():
        for i in range(0, len(rec), 32):
            chunk = rec[i:i + 32]
            batch, last = collate([r[0] for r in chunk])
            idx = torch.stack([r[1] for r in chunk]).cuda()
            f = head_logits(learner, batch, last, idx)
            # PPO order (main, shoulder, c, button) -> head names
            col = {"main_xy": 0, "shoulder_val": 1, "c_dir_logits": 2, "btn_logits": 3}
            k_sum = torch.zeros(len(chunk), device="cuda")
            d_sum = torch.zeros(len(chunk), device="cuda")
            r_sum = torch.zeros(len(chunk), device="cuda")
            for h in HEADS:
                lf = torch.log_softmax(f[h].float(), -1)
                lr = torch.log_softmax(torch.stack([r[2][h] for r in chunk]).cuda(), -1)
                k = (lr.exp() * (lr - lf)).sum(-1)            # KL(rollout || fp32 re-score)
                per_head[h] += k.tolist()
                k_sum += k
                a_idx = idx[:, col[h]:col[h] + 1]
                d_sum += (lf.gather(1, a_idx) - lr.gather(1, a_idx))[:, 0]
                r_sum += lr.gather(1, a_idx)[:, 0]
            kl += k_sum.tolist()
            dlp += d_sum.tolist()
            roll_lp += r_sum.tolist()
    kl, dlp = np.array(kl), np.array(dlp)
    ratio = np.exp(dlp)
    res = dict(policy_dir=a.policy_dir, weights=a.weights, swap=a.swap, rollout="fp32" if a.fp32 else "bf16",
               stream=bool(getattr(cfg, "attn_window", 0)), decisions=len(kl),
               kl_mean=float(kl.mean()), kl_median=float(np.median(kl)), kl_p99=float(np.quantile(kl, 0.99)),
               kl_max=float(kl.max()),
               kl_per_head={h: float(np.mean(x)) for h, x in per_head.items()},
               dlogp_mean=float(dlp.mean()), dlogp_abs_mean=float(np.abs(dlp).mean()),
               ratio_outside_clip=float(np.mean(np.abs(ratio - 1) > 0.2)),
               ratio_p01=float(np.quantile(ratio, 0.01)), ratio_p99=float(np.quantile(ratio, 0.99)),
               rollout_entropy_proxy=float(-np.mean(roll_lp)))
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
