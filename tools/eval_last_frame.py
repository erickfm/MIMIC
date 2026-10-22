#!/usr/bin/env python3
"""Inference-faithful val loss: score each checkpoint only on the newest frame, with the
context it actually gets at inference, on one fixed set of held-out target frames.

train.py's val loss averages over every position of a window, so position 0 (no history)
counts as much as position 179 (full history). At inference the bot always predicts the
newest frame with a full window. This scores exactly that:

  - full-causal models (attn_window == 0): the last cfg.max_seq_len frames, loss at the last;
  - sliding-window models: their whole receptive field (num_layers * (W-1) + 1 frames), which
    equals streaming with mimic/kv_cache.py.

Every model sees the same targets, so differences are paired.

    python3 tools/eval_last_frame.py --data-dir data/mastfox_c9_half \
        base=checkpoints/proxy/p1-C-base-s1.avg.pt swa=checkpoints/proxy/swa.avg.pt
"""
import argparse, dataclasses, json, math, random, sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import train as T
from mimic import model as M

KEYS = ["stage", "self_character", "opp_character", "self_action", "opp_action", "self_numeric",
        "opp_numeric", "self_flags", "opp_flags", "self_controller", "self_port", "opp_port"]


def load(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    fields = {f.name for f in dataclasses.fields(M.ModelConfig)}
    cfg = M.ModelConfig(**{k: v for k, v in ck["config"].items() if k in fields})
    m = M.FramePredictor(cfg).to(device)
    m.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck["model_state_dict"].items()})
    m.eval()
    w = getattr(cfg, "attn_window", 0)
    ctx = cfg.num_layers * (w - 1) + 1 if w > 0 else cfg.max_seq_len
    return m, cfg, ctx, torch.tensor(ck["stick_centers"], dtype=torch.float32, device=device), \
        torch.tensor(ck["shoulder_centers"], dtype=torch.float32, device=device)


def targets(data_dir, n, min_hist, seed):
    """(shard path, absolute frame index) pairs: each target has >= min_hist frames of history."""
    man = json.load(open(Path(data_dir) / "tensor_manifest.json"))
    rng = random.Random(seed)
    games = []
    for name in man["val_shards"]:
        sh = torch.load(Path(data_dir) / name, weights_only=True, mmap=True)
        off = sh["offsets"]
        for g in range(int(sh["n_games"])):
            a, b = int(off[g]), int(off[g + 1])
            if b - a > min_hist:
                games.append((name, a, b))
    per = max(1, math.ceil(n / len(games)))
    out = [(name, rng.randint(a + min_hist - 1, b - 1)) for name, a, b in games for _ in range(per)]
    rng.shuffle(out)
    return sorted(out[:n])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="name=checkpoint.pt (first one is the paired reference)")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda"
    T.DEVICE = dev
    models = {}
    for spec in args.models:
        name, path = spec.split("=", 1)
        models[name] = load(path, dev)
    min_hist = max(ctx for _, _, ctx, _, _ in models.values())
    tg = targets(args.data_dir, args.n, min_hist, args.seed)
    shards = {}
    per = {name: [] for name in models}            # per-target total loss
    for i in range(0, len(tg), args.batch):
        chunk = tg[i:i + args.batch]
        for name, (m, cfg, ctx, stick, shoulder) in models.items():
            st, tt = {k: [] for k in KEYS}, None
            tl = []
            for sname, t in chunk:
                if sname not in shards:
                    shards[sname] = torch.load(Path(args.data_dir) / sname, weights_only=True, mmap=True)
                sh = shards[sname]
                for k in KEYS:
                    st[k].append(sh["states"][k][t - ctx + 1: t + 1])
                tl.append({k: v[t: t + 1] for k, v in sh["targets"].items()})
            st = {k: torch.stack(v).to(dev) for k, v in st.items()}
            tt = {k: torch.stack([d[k] for d in tl]).to(dev) for k in tl[0]}
            xy = torch.stack([tt["main_x"], tt["main_y"]], -1)
            tt["main_cluster"] = torch.cdist(xy.reshape(-1, 2), stick).argmin(-1).reshape(xy.shape[:-1])
            with torch.no_grad():   # backbone over the context, heads on the newest frame only
                h = m.features(st)[:, -1:]
                if getattr(cfg, "cond_heads", False):
                    p = m.heads(h, cond_classes=T._cond_head_classes(tt, shoulder, cfg.num_c_dirs))
                else:
                    p = m.heads(h)
            for b in range(len(chunk)):    # per-target losses for paired statistics
                _, losses = T.compute_loss({k: v[b:b + 1] for k, v in p.items()},
                                           {k: v[b:b + 1] for k, v in tt.items()}, plain_ce=True,
                                           hal_mode=True, shoulder_centers=shoulder,
                                           n_cdir=cfg.num_c_dirs, lite=True)
                per[name].append(float(sum(l.item() for l in losses)))
    ref = next(iter(models))
    print(f"targets: {len(tg)} val frames, each with >= {min_hist} frames of history")
    for name, (m, cfg, ctx, _, _) in models.items():
        x = torch.tensor(per[name])
        line = f"{name:24s} ctx={ctx:5d}  last-frame loss {x.mean():.4f} ± {x.std() / math.sqrt(len(x)):.4f}"
        if name != ref:
            d = x - torch.tensor(per[ref])
            line += f"   vs {ref}: {d.mean():+.4f} ± {d.std() / math.sqrt(len(d)):.4f} (paired)"
        print(line)


if __name__ == "__main__":
    main()
