#!/usr/bin/env python3
"""L-cancel RL in the native sim: rollouts from rlvr/sim/vec_env.py, update from rlvr/sim/ppo.py.

The learner plays Fox dittos against a frozen copy of the starting policy (frozen BC, not CPU-9:
see memory feedback_rlvr_training_opponent), ports alternated across envs. Episodes and reward
are rlvr/sim/lcancel.py (the l_cancel_online definitions, resolved live from the sim's exact
post-frames). PPO is the unchanged ppo_update with a frozen reference for the KL term; the
tuned Dolphin recipe's defaults (docs/research-notes-2026-07-14.md, loop.py docstring).

Rollouts run the learner in bf16 inside a CUDA graph that reads its live parameters, so optimizer
steps reach it in place. The PPO "old" log-probs are recomputed in fp32 at update time (the
update's own numerics), so the rollout precision cannot bias the ratio.

Any model trains: full-causal or sliding-window (stream), with or without --cond-heads. Each
sampled decision keeps the exact input behind it (BatchedPolicy.context: the 180-frame window,
or a stream model's last receptive-field frames since the episode start), and the update scores
it with ppo.head_logits, which teacher-forces the sampled classes through a cond-heads chain. A
stream learner's KV cache is recomputed from its history after every update
(BatchedPolicy.refresh), so rollouts never mix old and new weights. Check:
rlvr/tests/test_rl_context.py (re-scoring matches the sampling logits to ~1e-7, fp32).

  python3 tools/sim_rl.py --policy-dir hf_eval/fox-master --templates replays_doc_rt \
      --out-dir checkpoints/simrl_lcancel --updates 60

Dense verifiable rewards (--vr "stock_delta:1,damage_delta:0.01"): rlvr/sim/vr modules run in the
sim workers on every frame; their weighted sum is the reward stream. A sampled learner decision
(probability 1/--sample-every) gets the discounted return of the next --horizon frames (or until
the match ends, plus each module's end-of-match term) and becomes a one-frame episode, so
ppo_update's batch normalization turns returns into advantages (the batch mean is the baseline;
no critic). The policy is held near BC by an exact KL on a random sample of all its frames
(--global-kl-beta), not by PPO's per-decision term (--kl-beta, 0 in dense mode). Every finished
rollout game is also a game against frozen BC: the log carries the learner's record.
"""
import argparse, copy, json, os, random as _random_mod, sys, time
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
    ap.add_argument("--updates", type=int, default=None, help="default: 100 dense, 60 L-cancel")
    ap.add_argument("--episodes-per-update", type=int, default=192,
                    help="minimum scored episodes per update")
    ap.add_argument("--misses-per-update", type=int, default=0,
                    help="also keep collecting until the batch holds this many misses (batch grows "
                         "~1/miss-rate as success saturates: the tuned recipe's batch scaling)")
    ap.add_argument("--max-episodes-per-update", type=int, default=2000)
    ap.add_argument("--global-kl-beta", type=float, default=None,
                    help="anchor the policy to BC on a random sample of ALL its frames (exact KL over "
                         "the 4 heads), not just the L-cancel episode frames PPO's KL term sees "
                         "(default: 0.01 dense, 0 L-cancel)")
    ap.add_argument("--anchor-frames", type=int, default=512, help="anchor sample size per update")
    ap.add_argument("--lr", type=float, default=None, help="default: 1e-5 dense, 3e-5 L-cancel")
    ap.add_argument("--kl-beta", type=float, default=None,
                    help="per-decision KL to BC on the sampled actions (default: 0 dense, 0.003 L-cancel)")
    ap.add_argument("--ppo-epochs", type=int, default=4)
    ap.add_argument("--ppo-minibatch", type=int, default=0,
                    help="decisions per update forward (memory only; default: 64 at 180 frames, fewer for "
                         "longer contexts)")
    ap.add_argument("--checkpoint-every", type=int, default=5)
    ap.add_argument("--rng", default="online", choices=["online", "offline"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", help="rl_uNNNN.pt to continue from (its opt_uNNNN.pt restores AdamW state)")
    ap.add_argument("--vr", default=None,
                    help="dense VR mode: 'id:weight,...' over rlvr/sim/vr ids (replaces the L-cancel episodes)")
    ap.add_argument("--gamma", type=float, default=0.99, help="dense mode: per-frame discount")
    ap.add_argument("--horizon", type=int, default=300, help="dense mode: return horizon in frames")
    ap.add_argument("--sample-every", type=int, default=8,
                    help="dense mode: store each learner decision with probability 1/N")
    ap.add_argument("--frames-per-update", type=int, default=16384,
                    help="dense mode: sampled decisions (with finished returns) per update. 16,384 since "
                         "2026-10-25: at 4,096 RL on a strong base (S-full 56k) barely moved the training "
                         "reward; 16,384 climbed and won 63%% vs the base (docs/research-notes-2026-10-25.md)")
    ap.add_argument("--packed-update", action=argparse.BooleanOptionalAction, default=True,
                    help="sliding-window models, dense mode: stitch each slot's overlapping decision "
                         "contexts into one sequence and score all its decisions in one forward "
                         "(exact; ppo.pack_streams). --no-packed-update scores each decision alone")
    ap.add_argument("--guards", default=None,
                    help="dense mode: 'id[:margin],...' VRs kept at least at BC's level (+margin, relative) "
                         "in the same rollout games, by adaptive (Lagrangian) weights")
    ap.add_argument("--guard-lr", type=float, default=0.2, help="guard weight step per unit relative gap")
    ap.add_argument("--guard-init", type=float, default=0.1)
    ap.add_argument("--guard-max", type=float, default=2.0)
    ap.add_argument("--normalize", action="store_true",
                    help="dense mode: scale each VR's returns by its running std, so weights compare across VRs")
    a = ap.parse_args()
    # Dense defaults: 16,384 decisions x 100 updates, 4 PPO epochs, lr 1e-5, the global KL anchor
    # (0.01) and nothing else: no critic (batch-mean baseline) and no per-decision KL term. Both
    # were removed 2026-10-27 without losing to SF7, which had them (one training seed,
    # docs/research-notes-2026-10-27.md; commit c2df04b reproduces SF7). Gains stop at ~100
    # updates (u150 and u200 tied u100). L-cancel keeps the July recipe.
    dense_defaults = a.vr is not None
    for k, dense_v, lc_v in (("updates", 100, 60), ("lr", 1e-5, 3e-5), ("kl_beta", 0.0, 0.003),
                             ("global_kl_beta", 0.01, 0.0)):
        if getattr(a, k) is None:
            setattr(a, k, dense_v if dense_defaults else lc_v)

    import torch
    from inference_utils import load_mimic_model
    from rlvr.sim.ppo import (OnlinePPOConfig, _logprob_at_indices, collate, head_logits, last_features,
                              pack_streams, packed_features, ppo_update)
    from rlvr.sim.trajectory import FrameRecord
    from rlvr.sim.lcancel import LCancelTracker
    from rlvr.sim.vec_env import VecSim
    from sim_h2h import make_ditto

    torch.manual_seed(a.seed)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = str(Path(a.policy_dir) / "model.pt")
    learner, cfg = load_mimic_model(ckpt_path, "cuda")
    # Any model: --cond-heads chains are scored with the sampled classes teacher-forced
    # (ppo.head_logits); sliding-window models record their receptive-field history
    # (BatchedPolicy.context) and are re-scored by a window-attention forward over it.
    window = getattr(cfg, "attn_window", 0) or 0
    ctx_len = cfg.num_layers * (window - 1) + 1 if window else cfg.max_seq_len
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
    vr_ids, vr_w = [], []
    if a.vr:
        for part in a.vr.split(","):
            i, _, w = part.partition(":")
            vr_ids.append(i.strip())
            vr_w.append(float(w) if w else 1.0)
    n_obj = len(vr_ids)
    guard_margin = []   # guards follow the objectives in vr_ids; their weights adapt each update
    if a.guards:
        for part in a.guards.split(","):
            i, _, m = part.partition(":")
            vr_ids.append(i.strip())
            vr_w.append(a.guard_init)
            guard_margin.append(float(m) if m else 0.0)
    import numpy as np
    vr_w = np.asarray(vr_w, np.float64)
    vr_scale = np.ones(len(vr_ids))          # running std of each VR's returns (--normalize)
    rate_ema = None                           # EMA of (learner, BC) per-frame rates, for the guards
    scale_seen = np.zeros(len(vr_ids), bool)
    v = VecSim(templates, {"LEARN": dict(dir=a.policy_dir, model=learner, cfg=cfg, amp=True, compile=False),
                           "OPP": dict(dir=a.policy_dir, amp=True, compile=True)},
               a.envs, a.workers, rng=a.rng, seed=a.seed, assign=assign, vr_ids=vr_ids)

    # One tracker per learner slot; where that slot's player sits in the env.
    slots = []
    for e, env in enumerate(v.envs):
        for side, (port, name) in enumerate(env["players"]):
            if name == "LEARN":
                slots.append(dict(env=e, side=side, slot=env["slots"][side], tracker=LCancelTracker(),
                                  pending=[], pend=dict(G=np.zeros((0, len(vr_ids))), disc=np.zeros(0),
                                                        age=np.zeros(0, np.int64))))
    dense = bool(vr_ids)
    packed = bool(window and a.packed_update and dense)
    sample_rng = _random_mod.Random(a.seed + 1)
    import random as _random
    anchor_rng = _random.Random(a.seed)
    anchor = []           # reservoir sample of the learner's frame windows, refilled per update
    anchor_seen = 0

    def ppo_idx(row):   # BatchedPolicy order (main, c, shoulder, button) -> PPO order
        m, c, sh, b = (int(x) for x in row)
        return torch.tensor([m, sh, c, b], dtype=torch.long)

    def on_decide(name, idx, pol):
        nonlocal anchor_seen
        if name != "LEARN":
            return
        if a.global_kl_beta > 0:
            picks = []
            for s in slots:   # reservoir sampling over every learner frame of this update's rollouts
                anchor_seen += 1
                j = anchor_rng.randrange(anchor_seen) if len(anchor) >= a.anchor_frames else len(anchor)
                if j < a.anchor_frames:
                    picks.append((j, s))
            if picks:
                for (j, s), obs in zip(picks, pol.context([s["slot"] for _, s in picks])):
                    w = (obs, ppo_idx(idx[s["slot"]]))
                    if j < len(anchor):
                        anchor[j] = w
                    else:
                        anchor.append(w)
        if dense:   # sample decisions; their returns accumulate from the next frame on
            chosen = [s for s in slots if sample_rng.random() * a.sample_every < 1.0]
            step = int(pol.hist_g) if packed else -1
            for s, obs in zip(chosen, pol.context([s["slot"] for s in chosen]) if chosen else []):
                fr = FrameRecord(obs=obs, sampled_indices=ppo_idx(idx[s["slot"]]),
                                 logprob_old=torch.zeros(()), logprob_ref=torch.zeros(()),
                                 game_frame_id=int(v.info.a[s["env"], s["side"], 0]),
                                 stream=s["slot"], step=step)
                s["pending"].append(fr)   # its returns, discount and age live in s["pend"]
                P = s["pend"]
                P["G"] = np.vstack([P["G"], np.zeros((1, len(vr_ids)))])
                P["disc"] = np.append(P["disc"], 1.0)
                P["age"] = np.append(P["age"], 0)
            return
        chosen = [s for s in slots if s["tracker"].recording()]
        for s, obs in zip(chosen, pol.context([s["slot"] for s in chosen]) if chosen else []):
            s["tracker"].active.frames.append(FrameRecord(
                obs=obs, sampled_indices=ppo_idx(idx[s["slot"]]),
                logprob_old=torch.zeros(()), logprob_ref=torch.zeros(()),
                game_frame_id=int(v.info.a[s["env"], s["side"], 0])))

    v.on_decide = on_decide
    opt = torch.optim.AdamW([p for p in learner.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    opt_path = a.resume and Path(a.resume).with_name(Path(a.resume).name.replace("rl_u", "opt_u"))
    if opt_path and opt_path.exists():
        opt.load_state_dict(torch.load(opt_path, map_location="cuda", weights_only=False))
    # Update minibatches only bound memory (gradients accumulate over them, one optimizer step per
    # epoch): ~64 x 180 frames whatever the context length. Bigger is not faster: 256 for S was
    # ~8% slower per update than 64 (2026-10-23).
    mb = a.ppo_minibatch or max(4, 64 * 180 // ctx_len)
    pcfg = OnlinePPOConfig(kl_beta=a.kl_beta, minibatch_frames=mb)
    log = open(out / "metrics.jsonl", "a")
    t_start = time.time()
    from rlvr.sim.trajectory import Episode
    res_seen = [0]

    def new_pend():
        return dict(G=np.zeros((0, len(vr_ids))), disc=np.zeros(0), age=np.zeros(0, np.int64))

    def close_pending(s, bonus, out):   # bonus: per-VR end-of-match terms
        P = s["pend"]
        for fr, G, disc in zip(s["pending"], P["G"], P["disc"]):
            out.append(Episode(task_id="vr_sim", frames=[fr], metadata={"G": G + disc * bonus}))
        s["pending"] = []
        s["pend"] = new_pend()

    def collect_dense():
        """Step until frames_per_update sampled decisions have finished returns."""
        out = []
        st = dict(win=0, loss=0, draw=0, vr_learn=np.zeros(len(vr_ids)), vr_opp=np.zeros(len(vr_ids)),
                  learner_frames=0)
        while len(out) < a.frames_per_update:
            new = v.results[res_seen[0]:]
            res_seen[0] = len(v.results)
            fin = {r["env"]: r for r in new}
            for r in new:   # rollout games are games against frozen BC
                ls, os_ = r["final"]["LEARN"][0], r["final"]["OPP"][0]
                st["win" if ls > os_ else "loss" if ls < os_ else "draw"] += 1
            for s in slots:
                e, side = s["env"], s["side"]
                if e in v.last_resets:   # this frame starts a new match: close the old one's returns
                    r = fin.get(e)
                    bonus = np.asarray(r["vr_final"][side]) if r and "vr_final" in r else np.zeros(len(vr_ids))
                    close_pending(s, bonus, out)
                    continue
                st["learner_frames"] += 1
                st["vr_learn"] += v.vr.a[e, side]
                st["vr_opp"] += v.vr.a[e, 1 - side]
                if s["pending"]:   # decisions before this frame earn its reward (vectorized per slot)
                    P = s["pend"]
                    rew = v.vr.a[e, side][:len(vr_ids)].astype(np.float64)
                    P["G"] += P["disc"][:, None] * rew
                    P["disc"] *= a.gamma
                    P["age"] += 1
                    done = P["age"] >= a.horizon
                    if done.any():
                        for j in np.flatnonzero(done):
                            out.append(Episode(task_id="vr_sim", frames=[s["pending"][j]],
                                               metadata={"G": P["G"][j].copy()}))
                        keep = ~done
                        s["pending"] = [fr for fr, k in zip(s["pending"], keep) if k]
                        for key in P:
                            P[key] = P[key][keep]
            v.step()
        return out, st

    for upd in range(start, a.updates + 1):
        t0, f0 = time.time(), v.frames
        episodes = []
        n_miss = 0
        dstats = None
        guard_stats = {}
        if dense:
            episodes, dstats = collect_dense()
            if packed:   # pack order: each pack's decisions are contiguous (one PPO minibatch)
                order, packs = pack_streams([ep.frames[0] for ep in episodes])
                episodes = [episodes[i] for i in order]
            Gm = np.stack([ep.metadata["G"] for ep in episodes])            # (N, n_vr) raw returns
            if a.normalize:   # running std per VR (EMA over updates); a VR keeps scale 1 until it varies
                sd = Gm.std(0)
                for i, x in enumerate(sd):
                    if x > 1e-9:
                        vr_scale[i] = x if not scale_seen[i] else np.sqrt(0.9 * vr_scale[i] ** 2 + 0.1 * x ** 2)
                        scale_seen[i] = True
                vr_scale = np.maximum(vr_scale, 1e-3)
            if guard_margin:   # Lagrangian guards: raise a guard's weight while the learner is below BC
                n = max(1, dstats["learner_frames"])
                rates = np.stack([dstats["vr_learn"], dstats["vr_opp"]])[:, :len(vr_ids)] / n
                rate_ema = rates if rate_ema is None else 0.9 * rate_ema + 0.1 * rates   # rare events: slow
                for j, m in enumerate(guard_margin):
                    i = n_obj + j
                    lrn, bc = rate_ema[0, i], rate_ema[1, i]
                    gap = float(np.clip((bc + m * abs(bc) - lrn) / (abs(bc) + 1e-6), -1, 1))   # > 0: below target
                    vr_w[i] = float(np.clip(vr_w[i] + a.guard_lr * gap, 0.0, a.guard_max))
                    guard_stats[vr_ids[i]] = dict(w=round(vr_w[i], 4), gap=round(float(gap), 4))
            for ep, g in zip(episodes, Gm):   # scalar return under the current weights
                ep.terminal_reward = float((vr_w * g / vr_scale).sum())
        while not dense and (len(episodes) < a.episodes_per_update or n_miss < a.misses_per_update) \
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
        if not packed:
            packs = None
        # minibatches: (start, end, pack or None); a pack is scored by one forward over its sequences
        spans = [(o, o + pk["n"], pk) for o, pk in zip(np.cumsum([0] + [pk["n"] for pk in packs[:-1]]), packs)] \
            if packs else [(i, min(i + pcfg.minibatch_frames, len(frames)), None)
                           for i in range(0, len(frames), pcfg.minibatch_frames)]

        def feats_of(model, i, j, pk):
            if pk is not None:
                return packed_features(model, pk)
            batch, last = collate([fr.obs for fr in frames[i:j]])
            return last_features(model, batch, last)
        learner.eval()
        with torch.no_grad():
            for i, j, pk in spans:
                chunk = frames[i:j]
                idx = torch.stack([fr.sampled_indices for fr in chunk]).cuda()
                lp = _logprob_at_indices(head_logits(learner, None, None, idx, h=feats_of(learner, i, j, pk)), idx).cpu()
                lr_ = lp if a.kl_beta == 0 else \
                    _logprob_at_indices(head_logits(ref, None, None, idx, h=feats_of(ref, i, j, pk)), idx).cpu()
                for fr, x, y in zip(chunk, lp, lr_):   # the reference never changes: score it once
                    fr.logprob_old, fr.logprob_ref = x, y
        extra = None
        if a.global_kl_beta > 0 and anchor:
            # KL(policy || BC) summed over the 4 heads at each anchor frame: exact for independent
            # heads; for --cond-heads each head's KL is conditioned on the rollout's sampled
            # classes for the earlier heads (a one-sample estimate of the joint KL's chain rule).
            achunks = []
            with torch.no_grad():
                for i in range(0, len(anchor), pcfg.minibatch_frames):
                    part = anchor[i:i + pcfg.minibatch_frames]
                    ab, al = collate([o for o, _ in part])
                    ai = torch.stack([x for _, x in part]).cuda()
                    ref_logp = {h: torch.log_softmax(x.float(), -1) for h, x in head_logits(ref, ab, al, ai).items()}
                    achunks.append((ab, al, ai, ref_logp))

            def extra():
                n, total = len(anchor), 0.0
                for ab, al, ai, ref_logp in achunks:   # chunked like PPO's own forward
                    out = head_logits(learner, ab, al, ai)
                    kl = 0.0
                    for h, lr_ in ref_logp.items():
                        lp = torch.log_softmax(out[h].float(), -1)
                        kl = kl + (lp.exp() * (lp - lr_)).sum(-1).sum()
                    (a.global_kl_beta * kl / n).backward()
                    total += float(kl)
                return torch.tensor(a.global_kl_beta * total / n)
        for _ in range(a.ppo_epochs):
            m = ppo_update(learner, episodes, opt, pcfg, device="cuda", ref_model=None, extra_loss=extra,
                           packs=packs)
        v.policies["LEARN"].refresh()   # sliding-window learner: re-key its cache with the new weights
        anchor_kl = (m.get("extra_loss") / a.global_kl_beta) if (a.global_kl_beta > 0 and m.get("extra_loss") is not None) else None
        anchor.clear()
        anchor_seen = 0
        rec = dict(update=upd, episodes=len(episodes), misses=len(episodes) - n_succ, anchor_kl=anchor_kl,
                   lcancel_rate=n_succ / max(1, len(episodes)),
                   frames_recorded=len(frames), rollout_s=round(t_roll, 1),
                   env_fps=round((v.frames - f0) / max(t_roll, 1e-9)),
                   update_s=round(time.time() - t0 - t_roll, 1), kl=m.get("kl") if a.kl_beta > 0 else None, clip_frac=m.get("clip_frac"),
                   grad_norm=m.get("grad_norm"), elapsed_min=round((time.time() - t_start) / 60, 1))
        if dstats is not None:   # dense mode: per-frame VR means (learner / opponent) and the record
            n = max(1, dstats["learner_frames"])
            rec.update(lcancel_rate=None, misses=None,
                       vr={i: dict(learn=round(float(x) / n * 3600, 4), opp=round(float(y) / n * 3600, 4))
                           for i, x, y in zip(vr_ids, dstats["vr_learn"], dstats["vr_opp"])},
                       games=dict(win=dstats["win"], loss=dstats["loss"], draw=dstats["draw"]),
                       return_mean=float(np.mean([ep.terminal_reward for ep in episodes])) if episodes else None,
                       guards=guard_stats or None, vr_scale=[round(float(x), 4) for x in vr_scale])
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if upd % a.checkpoint_every == 0 or upd == a.updates:
            ck = dict(base_ckpt)
            ck["model_state_dict"] = {k: t.detach().cpu() for k, t in learner.state_dict().items()}
            ck["rl"] = dict(task="vr_sim" if dense else "l_cancel_sim", update=upd, args=vars(a))
            torch.save(ck, out / f"rl_u{upd:04d}.pt")
            torch.save(opt.state_dict(), out / f"opt_u{upd:04d}.pt")
    v.close()


if __name__ == "__main__":
    main()
