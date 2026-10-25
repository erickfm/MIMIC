"""Multi-step PPO update for online RL.

Given a batch of Episodes (each with stored T-frame context snapshots
per frame + old logprobs + ref logprobs), this:

  1. Computes discounted MC returns per episode.
  2. Group-normalizes across all frames in the batch -> advantages.
  3. For each frame, re-forwards the policy on the stored context (any
     length: full-causal windows or sliding-window histories) and computes
     the current log-prob of the sampled action under the 4 heads, with the
     sampled classes teacher-forced through --cond-heads chains.
  4. PPO-clipped surrogate + Schulman KL estimator to the frozen ref.

Memory: episode frames are stored CPU-side. For the update we build
chunks of M frames, move to GPU, forward, discard — controls peak
memory.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List

import torch
import torch.nn.functional as F

from rlvr.sim.trajectory import Episode, group_normalize_across_episodes


@dataclass
class OnlinePPOConfig:
    clip_eps: float = 0.2
    kl_beta: float = 0.01
    gamma: float = 1.0
    advantage_eps: float = 1e-8
    minibatch_frames: int = 64        # chunk size for GPU forward
    normalize_by_std: bool = True
    # Off by default (normal path unchanged). When True, each episode
    # must carry a precomputed scalar advantage in
    # ep.metadata["advantage"]; every frame of that episode gets it and
    # the global return z-scoring above is bypassed (e.g. matched-context
    # group means, reward - group mean, rather than batch-wide z-scores;
    # the removed savestate drill loop used it).
    use_metadata_advantage: bool = False


def collate(obs_list, device="cuda"):
    """Stack per-decision contexts {key: (L_i, ...)} into a (B, T, ...) batch, right-padded to
    the longest; returns (batch, last) with last[i] = L_i - 1, the decision's position. Right
    padding is exact under causal attention: no real position attends to a later one."""
    lens = [next(iter(o.values())).shape[0] for o in obs_list]
    T = max(lens)
    batch = {}
    for k in obs_list[0]:
        rows = []
        for o, L in zip(obs_list, lens):
            x = o[k]
            if L < T:
                x = torch.cat([x, x[-1:].expand(T - L, *x.shape[1:])], 0)
            rows.append(x)
        batch[k] = torch.stack(rows).to(device)
    return batch, torch.tensor(lens, device=device) - 1


def pack_streams(frames, max_frames=32768):
    """Exact packing for sliding-window models. One slot's consecutive decisions in the same match
    have overlapping contexts (each is the last <= receptive-field frames before it), so they are
    stitched into one sequence, and a single causal forward over it gives every decision's output
    (the streamed output at a position depends only on its receptive field). Two decisions are
    stitched only when the later context's overlapping frames equal the sequence's (checked on
    the first and last overlapping frame, every key), else a new sequence starts.
    frames: FrameRecords with .stream / .step set. Returns (order, packs): order lists the frame
    indices pack by pack; each pack is dict(n, batch {key: (B, T, ...)} on the CPU, row, pos)."""
    seqs = []                                  # [pieces, length, [(frame index, position)], last step]
    open_seq = {}
    for i, fr in enumerate(frames):
        c = fr.obs
        L = next(iter(c.values())).shape[0]
        s = open_seq.get(fr.stream)
        d = fr.step - s[3] if s is not None else 0
        if s is not None and 0 < d < L and s[1] >= L - d:
            n_ov = L - d                       # overlap: c[:n_ov] == sequence[-n_ov:]
            first = _seq_frame(s[0], s[1] - n_ov, s[1])
            last = _seq_frame(s[0], s[1] - 1, s[1])
            ok = all(torch.equal(c[k][0], first[k]) and torch.equal(c[k][n_ov - 1], last[k]) for k in c)
        else:
            ok = False
        if ok:
            s[0].append({k: v[n_ov:] for k, v in c.items()})
            s[1] += d
            s[2].append((i, s[1] - 1))
            s[3] = fr.step
        else:
            s = [[c], L, [(i, L - 1)], fr.step]
            seqs.append(s)
            open_seq[fr.stream] = s
    seqs.sort(key=lambda s: -s[1])
    order, packs, cur = [], [], []

    def flush():
        T = cur[0][1]
        batch = {}
        for k in cur[0][0][0]:
            rows = []
            for s in cur:
                x = torch.cat([p[k] for p in s[0]], 0)
                if s[1] < T:   # right padding is exact under causal attention
                    x = torch.cat([x, x[-1:].expand(T - s[1], *x.shape[1:])], 0)
                rows.append(x)
            batch[k] = torch.stack(rows).pin_memory()
        row = [r for r, s in enumerate(cur) for _ in s[2]]
        pos = [p for s in cur for _, p in s[2]]
        order.extend(i for s in cur for i, _ in s[2])
        packs.append(dict(n=len(pos), batch=batch, row=torch.tensor(row), pos=torch.tensor(pos)))
        cur.clear()
    for s in seqs:   # longest first, so each pack's padded size is set by its first sequence
        if cur and cur[0][1] * (len(cur) + 1) > max_frames:
            flush()
        cur.append(s)
    if cur:
        flush()
    return order, packs


def _seq_frame(pieces, j, total):
    """Frame j of a sequence stored as pieces (total frames), searching from the end."""
    j -= total
    for p in reversed(pieces):
        n = next(iter(p.values())).shape[0]
        if -j <= n:
            return {k: v[n + j] for k, v in p.items()}
        j += n
    raise IndexError(j)


def packed_features(model, pack, device="cuda"):
    """Final-norm features at each decision of a pack_streams() pack: (n, d)."""
    raw = getattr(model, "_orig_mod", model)
    batch = {k: v.to(device, non_blocking=True) for k, v in pack["batch"].items()}
    h = raw.features(batch)
    return h[pack["row"].to(device), pack["pos"].to(device)]


def last_features(model, batch, last):
    """Final-norm features at each row's decision position: (B, d), for a collate() batch.
    Full-causal models: the plain forward. Sliding-window models compute only what the decision
    depends on: one query in the top layer, W - 1 more positions in each layer below (exact; over a
    1,075-frame stream context the full forward computes ~2.4x the positions and ~3x the attention
    scores). Each row is first right-aligned (its decision at the last position, the frames before
    its episode start filled with copies of its first frame and masked as keys)."""
    raw = getattr(model, "_orig_mod", model)
    B = last.shape[0]
    W = getattr(raw.cfg, "attn_window", 0) or 0
    if not W:
        h = raw.features(batch)
        return h[torch.arange(B, device=h.device), last]
    T = next(iter(batch.values())).shape[1]
    pad = T - 1 - last                                     # row i's padding: positions < pad[i]
    src = (torch.arange(T, device=last.device)[None] - pad[:, None]).clamp_(min=0)   # (B, T)

    def align(x):
        return x.gather(1, src.view(B, T, *[1] * (x.dim() - 2)).expand(B, T, *x.shape[2:]))
    x = raw.encoder({k: align(v) for k, v in batch.items()})
    keep = [1]   # trailing positions needed out of each layer, top down; keep[0]: the encoder's
    for _ in raw.blocks:
        keep.append(min(T, keep[-1] + W - 1))
    keep = keep[::-1]
    x = x[:, T - keep[0]:]
    for blk, n_q in zip(raw.blocks, keep[1:]):
        x = _block_tail(blk, x, n_q, T, pad, W)
    return raw.final_norm(x[:, -1])


def _block_tail(blk, x, n_q, T, pad, W):
    """MimicTransformerBlock.forward (sliding-window, fp32 attention path) at only the last n_q of
    x's positions; x holds absolute positions T - x.shape[1] .. T - 1, padding below pad[row]."""
    a = blk.self_attn
    B, Lk, D = x.shape
    q, k, v = a.c_attn(blk.ln_1(x)).split(D, dim=2)
    q = q[:, Lk - n_q:].reshape(B, n_q, a.n_head, a.hs).transpose(1, 2)
    k = k.reshape(B, Lk, a.n_head, a.hs).transpose(1, 2)
    v = v.reshape(B, Lk, a.n_head, a.hs).transpose(1, 2)
    with torch.autocast("cuda", enabled=False):
        q, k = q.float(), k.float()
        QEr = q @ a.Er.flip(0)[:W].float().transpose(0, 1)               # (B, H, n_q, W)
        kp = torch.arange(Lk, device=x.device)
        d = (kp[Lk - n_q:, None] - kp[None, :])                           # query - key distance
        Srel = QEr.gather(-1, d.clamp(0, W - 1).expand(B, a.n_head, n_q, Lk))
        att = (q @ k.transpose(-2, -1) + Srel) * (1.0 / math.sqrt(k.size(-1)))
        ok = ((d >= 0) & (d < W))[None] & ((kp + T - Lk)[None, None] >= pad[:, None, None])
        ok = ok | (d == 0)[None]   # a padding query sees itself (finite); no real query sees it
        att = F.softmax(att.masked_fill(~ok[:, None], float("-inf")), dim=-1)
    y = (a.attn_dropout(att) @ v).transpose(1, 2).contiguous().view(B, n_q, D)
    x = x[:, Lk - n_q:] + a.resid_dropout(a.c_proj(y))
    if blk._use_swiglu:
        return x + blk.mlp(blk.ln_2(x))
    return x + blk.dropout(blk.mlp.c_proj(F.gelu(blk.mlp.c_fc(blk.ln_2(x)))))


def head_logits(model, batch, last, sampled_indices=None, h=None):
    """Head logits at each decision's position: {head: (B, V)}. For --cond-heads models the
    sampled (shoulder, c, main) classes are teacher-forced through the chain, so the logits are
    the ones the action was sampled from and the summed log-prob is its exact joint log-prob.
    sampled_indices: (B, 4) in PPO order (main, shoulder, c, button). h: precomputed (B, d)
    features (e.g. packed_features), in place of batch / last."""
    raw = getattr(model, "_orig_mod", model)
    h = (last_features(raw, batch, last) if h is None else h).unsqueeze(1)   # (B, 1, d)
    if getattr(raw.heads, "cond", False):
        if sampled_indices is None:
            raise ValueError("cond-heads logits need the sampled classes to condition on")
        si = sampled_indices.unsqueeze(1)
        out = raw.heads(h, cond_classes=(si[..., 1], si[..., 2], si[..., 0]))
    else:
        out = raw.heads(h)
    return {k: v[:, 0] for k, v in out.items() if k in ("main_xy", "shoulder_val", "c_dir_logits", "btn_logits")}


def _last_logits(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 3:
        return t[:, -1, :]
    return t


def _logprob_at_indices(logits_dict, sampled_indices):
    """Sum log-prob across the 4 heads at the sampled indices.
    logits_dict values: (B, T, V) or (B, V).
    sampled_indices: (B, 4) long tensor."""
    def _lp(head, idx):
        lg = _last_logits(head).float()
        lg = torch.nan_to_num(lg, nan=-1e9, posinf=1e9, neginf=-1e9)
        logp = F.log_softmax(lg, dim=-1)
        return logp.gather(-1, idx.unsqueeze(-1)).squeeze(-1)

    m_idx = sampled_indices[:, 0]
    s_idx = sampled_indices[:, 1]
    c_idx = sampled_indices[:, 2]
    b_idx = sampled_indices[:, 3]
    return (
        _lp(logits_dict["main_xy"], m_idx)
        + _lp(logits_dict["shoulder_val"], s_idx)
        + _lp(logits_dict["c_dir_logits"], c_idx)
        + _lp(logits_dict["btn_logits"], b_idx)
    )


def _compute_advantages(episodes: List[Episode], gamma: float, normalize_by_std: bool,
                        eps: float) -> torch.Tensor:
    """Flatten per-frame returns across all episodes + z-score."""
    returns_per_ep = [ep.returns(gamma=gamma) for ep in episodes]
    if not returns_per_ep:
        return torch.empty(0)
    flat = torch.cat(returns_per_ep, dim=0).float()
    mean = flat.mean()
    if normalize_by_std:
        std = flat.std(unbiased=False)
        return (flat - mean) / (std + eps)
    return flat - mean


def ppo_update(
    model,
    episodes: List[Episode],
    optimizer,
    cfg: OnlinePPOConfig = OnlinePPOConfig(),
    device: str = "cuda",
    ref_model=None,
    extra_loss=None,
    packs=None,
) -> Dict[str, float]:
    """Run one PPO epoch over the collected episodes.

    packs: pack_streams() packs over the episodes' frames, in that order (sliding-window models);
    each pack is one minibatch, forwarded once for all its decisions.

    Returns a metrics dict averaged over the batch (mean reward, KL,
    clip frac, loss, etc.).
    """
    if not episodes:
        return {"n_frames": 0, "n_episodes": 0}

    # --- flatten frames across episodes and stack tensors ---
    all_frames = []
    ep_boundaries = []
    for ep in episodes:
        ep_boundaries.append(len(all_frames))
        all_frames.extend(ep.frames)
    N = len(all_frames)
    if N == 0:
        return {"n_frames": 0, "n_episodes": len(episodes)}

    if cfg.use_metadata_advantage:
        advantages = torch.cat([
            torch.full((len(ep.frames),), float(ep.metadata["advantage"]),
                       dtype=torch.float32)
            for ep in episodes
        ]).to(device).detach()
    else:
        advantages = _compute_advantages(
            episodes, cfg.gamma, cfg.normalize_by_std, cfg.advantage_eps
        ).to(device).detach()

    sampled_indices = torch.stack(
        [fr.sampled_indices for fr in all_frames], dim=0
    ).to(device)                                                 # (N, 4)
    logprobs_old = torch.stack(
        [fr.logprob_old for fr in all_frames], dim=0
    ).to(device).float().detach()                                # (N,)
    # logprob_ref is computed below per-minibatch from ref_model if
    # provided (cheaper than running ref_model live during collection,
    # which would slow each in-game frame). Fall back to the cached
    # value stored in FrameRecord if no ref_model was passed (legacy
    # path).
    if ref_model is None:
        logprobs_ref_cached = torch.stack(
            [fr.logprob_ref for fr in all_frames], dim=0
        ).to(device).float().detach()                            # (N,)
    else:
        logprobs_ref_cached = None
    rewards_end = torch.tensor(
        [fr.reward + (ep.terminal_reward if i == len(ep.frames) - 1 else 0.0)
         for ep in episodes
         for i, fr in enumerate(ep.frames)],
        dtype=torch.float32,
    ).to(device)

    # --- re-forward in minibatches with gradient ---
    all_logprobs_theta = []
    # eval(), not train(): the re-forward must be deterministic so the
    # policy logprobs are comparable to the eval-mode reference and to
    # the rollout-time logprobs. eval() does not block gradients.
    model.eval()
    optimizer.zero_grad()
    minibatch = cfg.minibatch_frames
    total_loss = 0.0
    total_kl = 0.0
    total_clip = 0.0
    total_n = 0

    if packs is not None:
        bounds = [0]
        for pk in packs:
            bounds.append(bounds[-1] + pk["n"])
        assert bounds[-1] == N, (bounds[-1], N)
        spans = list(zip(bounds[:-1], bounds[1:], packs))
    else:
        spans = [(s, min(s + minibatch, N), None) for s in range(0, N, minibatch)]
    for start, end, pk in spans:
        idx = sampled_indices[start:end]
        if pk is not None:
            batch = last = None
            lp_theta = _logprob_at_indices(head_logits(model, None, None, idx, h=packed_features(model, pk, device)), idx)
        else:
            batch, last = collate([fr.obs for fr in all_frames[start:end]], device)
            lp_theta = _logprob_at_indices(head_logits(model, batch, last, idx), idx)
        all_logprobs_theta.append(lp_theta.detach())

        # Per-chunk PPO objective (bptt cheap since per-chunk).
        adv_chunk = advantages[start:end]
        lp_old_chunk = logprobs_old[start:end]
        if ref_model is not None:
            with torch.no_grad():
                ref_logits = head_logits(ref_model, batch, last, idx,
                                         h=packed_features(ref_model, pk, device) if pk is not None else None)
            lp_ref_chunk = _logprob_at_indices(ref_logits, idx).detach()
        else:
            lp_ref_chunk = logprobs_ref_cached[start:end]

        log_ratio = lp_theta - lp_old_chunk
        ratio = torch.exp(log_ratio)
        unclipped = ratio * adv_chunk
        clipped = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_chunk
        pg_obj = torch.minimum(unclipped, clipped)
        pg_loss = -pg_obj.mean()

        # Schulman low-bias KL
        lr_diff = lp_ref_chunk - lp_theta
        kl = (lr_diff.exp() - lr_diff - 1.0).mean()

        loss = pg_loss + cfg.kl_beta * kl
        (loss * (end - start) / N).backward()   # accumulate, scaled by chunk frac

        total_loss += float(loss.item()) * (end - start)
        total_kl += float(kl.item()) * (end - start)
        total_clip += float(
            ((ratio < 1 - cfg.clip_eps) | (ratio > 1 + cfg.clip_eps)).float().mean().item()
        ) * (end - start)
        total_n += (end - start)

    extra = None
    if extra_loss is not None:
        # Optional extra objective (e.g. the sim RL's global KL anchor), added to the same step.
        extra = extra_loss()
        if extra.requires_grad:   # a hook may backpropagate itself (e.g. in chunks) and return the value
            extra.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return {
        "loss": total_loss / max(total_n, 1),
        "kl": total_kl / max(total_n, 1),
        "clip_frac": total_clip / max(total_n, 1),
        "reward_mean": float(rewards_end.mean().item()),
        "advantage_mean": float(advantages.mean().item()),
        "advantage_std": float(advantages.std(unbiased=False).item()),
        "grad_norm": float(grad_norm),
        "n_frames": total_n,
        "n_episodes": len(episodes),
        "extra_loss": float(extra.item()) if extra is not None else None,
    }
