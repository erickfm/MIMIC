#!/usr/bin/env python3
"""L-cancel RL in the native sim: rollouts from rlvr/sim/vec_env.py, update from rlvr/online/ppo.py.

The learner plays Fox dittos against a frozen copy of the starting policy (frozen BC, not CPU-9:
see memory feedback_rlvr_training_opponent), ports alternated across envs. Episodes and reward
are rlvr/sim/lcancel.py (the l_cancel_online definitions, resolved live from the sim's exact
post-frames). PPO is the unchanged ppo_update with a frozen reference for the KL term; the
tuned Dolphin recipe's defaults (docs/research-notes-2026-07-14.md, loop.py docstring).

Rollouts run the learner in bf16 inside a CUDA graph that reads its live parameters, so optimizer
steps reach it in place. The PPO "old" log-probs are recomputed in fp32 at update time (the
update's own numerics), so the rollout precision cannot bias the ratio.

  python3 tools/sim_rl.py --policy-dir hf_eval/fox-master --templates replays_doc_rt \
      --out-dir checkpoints/simrl_lcancel --updates 60
"""
import argparse, copy, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
# No cuBLAS workspace. With one, the PPO update's fp32 SGEMMs intermittently launched with a bad
# address (Xid 13 misaligned / out-of-range in cutlass_80_simt_sgemm_128x256_8x4_nt_align1, from a
# GPU core dump) after the CUDA-graph rollouts; 0 crashes in 34 updates without it vs one every
# ~5-10 (2026-10-21). Must be set before torch initializes cuBLAS.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":0:0")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy-dir", required=True, help="hf-style dir: model.pt + norm JSONs")
    ap.add_argument("--templates", required=True, help="1v1 .slp templates (made into Fox dittos)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--updates", type=int, default=60)
    ap.add_argument("--episodes-per-update", type=int, default=192,
                    help="minimum scored episodes per update")
    ap.add_argument("--misses-per-update", type=int, default=0,
                    help="also keep collecting until the batch holds this many misses (batch grows "
                         "~1/miss-rate as success saturates: the tuned recipe's batch scaling)")
    ap.add_argument("--max-episodes-per-update", type=int, default=2000)
    ap.add_argument("--global-kl-beta", type=float, default=0.0,
                    help="anchor the policy to BC on a random sample of ALL its frames (exact KL over "
                         "the 4 heads), not just the L-cancel episode frames PPO's KL term sees")
    ap.add_argument("--anchor-frames", type=int, default=512, help="anchor sample size per update")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--kl-beta", type=float, default=0.003)
    ap.add_argument("--ppo-epochs", type=int, default=4)
    ap.add_argument("--checkpoint-every", type=int, default=5)
    ap.add_argument("--rng", default="online", choices=["online", "offline"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", help="rl_uNNNN.pt to continue from (its opt_uNNNN.pt restores AdamW state)")
    a = ap.parse_args()

    import torch
    from inference_utils import load_mimic_model
    from rlvr.online.ppo import OnlinePPOConfig, _logprob_at_indices, ppo_update
    from rlvr.online.trajectory import FrameRecord
    from rlvr.sim.lcancel import LCancelTracker
    from rlvr.sim.vec_env import VecSim
    from sim_h2h import make_ditto

    torch.manual_seed(a.seed)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = str(Path(a.policy_dir) / "model.pt")
    learner, cfg = load_mimic_model(ckpt_path, "cuda")
    if getattr(learner.heads, "cond", False):
        raise SystemExit("ppo_update scores heads independently; --cond-heads policies are not supported yet")
    ref = copy.deepcopy(learner).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    base_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    start = 1
    if a.resume:   # the KL reference stays the BC policy; only the learner resumes
        rck = torch.load(a.resume, map_location="cpu", weights_only=False)
        learner.load_state_dict(rck["model_state_dict"])
        start = rck["rl"]["update"] + 1

    tdir = out / "templates"
    tdir.mkdir(exist_ok=True)
    templates = []
    for src in sorted(Path(a.templates).glob("*.slp")):
        make_ditto(src, "FOX", tdir / src.name)
        templates.append(tdir / src.name)
    assign = lambda e, side, ch: ("LEARN", "OPP")[(e + side) % 2]
    v = VecSim(templates, {"LEARN": dict(dir=a.policy_dir, model=learner, cfg=cfg, amp=True, compile=False),
                           "OPP": dict(dir=a.policy_dir, amp=True, compile=True)},
               a.envs, a.workers, rng=a.rng, seed=a.seed, assign=assign)

    # One tracker per learner slot; where that slot's player sits in the env.
    slots = []
    for e, env in enumerate(v.envs):
        for side, (port, name) in enumerate(env["players"]):
            if name == "LEARN":
                slots.append(dict(env=e, side=side, slot=env["slots"][side], tracker=LCancelTracker()))
    keys = None
    import random as _random
    anchor_rng = _random.Random(a.seed)
    anchor = []           # reservoir sample of the learner's frame windows, refilled per update
    anchor_seen = 0

    def on_decide(name, idx, pol):
        nonlocal keys, anchor_seen
        if name != "LEARN":
            return
        keys = keys or list(pol.buf)
        if a.global_kl_beta > 0:
            for s in slots:   # reservoir sampling over every learner frame of this update's rollouts
                anchor_seen += 1
                j = anchor_rng.randrange(anchor_seen) if len(anchor) >= a.anchor_frames else len(anchor)
                if j < a.anchor_frames:
                    w = {k: pol.buf[k][s["slot"]].detach().to("cpu", copy=True) for k in keys}
                    if j < len(anchor):
                        anchor[j] = w
                    else:
                        anchor.append(w)
        for s in slots:
            t = s["tracker"]
            if t.recording():
                obs = {k: pol.buf[k][s["slot"]].detach().to("cpu", copy=True) for k in keys}
                m, c, sh, b = (int(x) for x in idx[s["slot"]])
                t.active.frames.append(FrameRecord(
                    obs=obs, sampled_indices=torch.tensor([m, sh, c, b], dtype=torch.long),   # PPO order
                    logprob_old=torch.zeros(()), logprob_ref=torch.zeros(()),
                    game_frame_id=int(v.info.a[s["env"], s["side"], 0])))

    v.on_decide = on_decide
    opt = torch.optim.AdamW([p for p in learner.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    opt_path = a.resume and Path(a.resume).with_name(Path(a.resume).name.replace("rl_u", "opt_u"))
    if opt_path and opt_path.exists():
        opt.load_state_dict(torch.load(opt_path, map_location="cuda", weights_only=False))
    pcfg = OnlinePPOConfig(kl_beta=a.kl_beta)
    log = open(out / "metrics.jsonl", "a")
    t_start = time.time()
    for upd in range(start, a.updates + 1):
        t0, f0 = time.time(), v.frames
        episodes = []
        n_miss = 0
        while (len(episodes) < a.episodes_per_update or n_miss < a.misses_per_update) \
                and len(episodes) < a.max_episodes_per_update:
            for s in slots:   # the policy's current frame, before it decides on it
                if s["env"] in v.last_resets:
                    s["tracker"].reset()
                fr, act, ground, hitstun, valid = v.info.a[s["env"], s["side"]]
                if valid:
                    s["tracker"].observe(fr, act, ground, hitstun)
                episodes += s["tracker"].finished
                n_miss += sum(ep.terminal_reward < 0.5 for ep in s["tracker"].finished)
                s["tracker"].finished = []
            v.step()
        episodes = [ep for ep in episodes if ep.frames]
        n_succ = sum(ep.terminal_reward > 0.5 for ep in episodes)
        t_roll = time.time() - t0
        # PPO "old" log-probs, in the update's own (fp32) numerics, before any epoch.
        frames = [fr for ep in episodes for fr in ep.frames]
        learner.eval()
        with torch.no_grad():
            for i in range(0, len(frames), pcfg.minibatch_frames):
                chunk = frames[i:i + pcfg.minibatch_frames]
                batch = {k: torch.stack([fr.obs[k] for fr in chunk]).cuda() for k in chunk[0].obs}
                idx = torch.stack([fr.sampled_indices for fr in chunk]).cuda()
                lp = _logprob_at_indices(learner(batch), idx).cpu()
                for fr, x in zip(chunk, lp):
                    fr.logprob_old = x
        extra = None
        if a.global_kl_beta > 0 and anchor:
            abatch = {k: torch.stack([w[k] for w in anchor]).cuda() for k in anchor[0]}
            with torch.no_grad():
                ref_logp = {h: torch.log_softmax(v[:, -1].float(), -1) for h, v in ref(abatch).items()
                            if h in ("main_xy", "shoulder_val", "c_dir_logits", "btn_logits")}

            def extra():   # exact KL(policy || BC) over the 4 heads, mean over anchor frames
                n, total = len(anchor), 0.0
                for i in range(0, n, pcfg.minibatch_frames):   # chunked like PPO's own forward
                    sl = slice(i, i + pcfg.minibatch_frames)
                    out = learner({k: v[sl] for k, v in abatch.items()})
                    kl = 0.0
                    for h, lr_ in ref_logp.items():
                        lp = torch.log_softmax(out[h][:, -1].float(), -1)
                        kl = kl + (lp.exp() * (lp - lr_[sl])).sum(-1).sum()
                    (a.global_kl_beta * kl / n).backward()
                    total += float(kl)
                return torch.tensor(a.global_kl_beta * total / n)
        for _ in range(a.ppo_epochs):
            m = ppo_update(learner, episodes, opt, pcfg, device="cuda", ref_model=ref, extra_loss=extra)
        anchor_kl = (m.get("extra_loss") / a.global_kl_beta) if (a.global_kl_beta > 0 and m.get("extra_loss") is not None) else None
        anchor.clear()
        anchor_seen = 0
        rec = dict(update=upd, episodes=len(episodes), misses=len(episodes) - n_succ, anchor_kl=anchor_kl,
                   lcancel_rate=n_succ / max(1, len(episodes)),
                   frames_recorded=len(frames), rollout_s=round(t_roll, 1),
                   env_fps=round((v.frames - f0) / max(t_roll, 1e-9)),
                   update_s=round(time.time() - t0 - t_roll, 1), kl=m.get("kl"), clip_frac=m.get("clip_frac"),
                   grad_norm=m.get("grad_norm"), elapsed_min=round((time.time() - t_start) / 60, 1))
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if upd % a.checkpoint_every == 0 or upd == a.updates:
            ck = dict(base_ckpt)
            ck["model_state_dict"] = {k: t.detach().cpu() for k, t in learner.state_dict().items()}
            ck["rl"] = dict(task="l_cancel_sim", update=upd, args=vars(a))
            torch.save(ck, out / f"rl_u{upd:04d}.pt")
            torch.save(opt.state_dict(), out / f"opt_u{upd:04d}.pt")
    v.close()


if __name__ == "__main__":
    main()
