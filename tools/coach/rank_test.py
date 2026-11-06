#!/usr/bin/env python3
"""Does a BC model's surprise at a player's inputs separate ranks?

For each set of shards (e.g. master val, diamond, platinum) it samples target frames, gives the
model the context it gets at inference (the last K frames, newest frame scored), and records
per head the NLL of the human's input and the entropy of the model's distribution, both
conditioned on the human's earlier-head classes (cond-heads teacher forcing). NLL - entropy
("excess") is 0 in expectation when the input is a sample from the model, so it removes how
open a situation is.

Reports per set: mean NLL, entropy and excess per head and summed, with standard errors
clustered by game, and game-level AUROC between sets (per-game mean excess, and per-game mean
NLL for comparison). Run with several --ctx values: if the model adapts to the player's own
history, the rank gap shrinks as K grows.

  python3 tools/coach/rank_test.py --ckpt runs/fox-M-bc2-mlr01-s1/policy/model.pt \
      --set master=data/fox_full_hf/fox:val --set diamond=data/coach_rank/shards_diamond:all \
      --set platinum=data/coach_rank/shards_platinum:all --ctx 180 30 --out results.json
"""
import argparse, json, random, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import train as T                                        # noqa: E402
from eval_last_frame import KEYS, load                    # noqa: E402

HEADS = ["shoulder", "c_dir", "main", "buttons"]


def games_of(spec, min_len):
    d, _, split = spec.partition(":")
    man = json.load(open(Path(d) / "tensor_manifest.json"))
    names = {"val": man["val_shards"], "train": man["train_shards"],
             "all": man["train_shards"] + man["val_shards"]}[split or "all"]
    out = []
    for name in names:
        sh = torch.load(Path(d) / name, weights_only=True, mmap=True)
        off = sh["offsets"]
        for g in range(int(sh["n_games"])):
            a, b = int(off[g]), int(off[g + 1])
            if b - a > min_len:
                out.append((str(Path(d) / name), a, b))
    return out


def score(model, cfg, stick, shoulder, shards, targets, ctx, batch, dev):
    """Per target: (NLL[4], H[4]) for heads shoulder, c_dir, main, buttons."""
    nll, ent = [], []
    for i in range(0, len(targets), batch):
        chunk = targets[i:i + batch]
        st = {k: torch.stack([shards[s]["states"][k][t - ctx + 1:t + 1] for s, t in chunk]).to(dev)
              for k in KEYS}
        tt = {k: torch.stack([shards[s]["targets"][k][t:t + 1] for s, t in chunk]).to(dev)
              for k in shards[chunk[0][0]]["targets"]}
        xy = torch.stack([tt["main_x"], tt["main_y"]], -1)
        tt["main_cluster"] = torch.cdist(xy.reshape(-1, 2), stick).argmin(-1).reshape(xy.shape[:-1])
        cls = T._cond_head_classes(tt, shoulder, cfg.num_c_dirs)
        with torch.no_grad():
            h = model.features(st)[:, -1:]
            p = model.heads(h, cond_classes=cls) if getattr(cfg, "cond_heads", False) else model.heads(h)
        logits = [p["shoulder_val"], p["c_dir_logits"], p["main_xy"], p["btn_logits"]]
        tgt = [cls[0], cls[1], cls[2], tt["btns_single"].long()]
        n_b, e_b = [], []
        for lg, y in zip(logits, tgt):
            lp = F.log_softmax(lg.float()[:, -1], -1)
            n_b.append(-lp.gather(-1, y[:, -1:]).squeeze(-1))
            e_b.append(-(lp.exp() * lp).sum(-1))
        nll.append(torch.stack(n_b, -1).cpu())
        ent.append(torch.stack(e_b, -1).cpu())
    return torch.cat(nll).numpy(), torch.cat(ent).numpy()


def clustered_se(x, g):
    r = x - x.mean()
    s = np.bincount(g, weights=r)
    return float(np.sqrt((s ** 2).sum()) / len(x))


def auroc(a, b):
    """P(score of a random game in b > one in a): 0.5 = no separation."""
    a, b = np.asarray(a), np.asarray(b)
    ranks = np.argsort(np.argsort(np.concatenate([a, b]))) + 1
    rb = ranks[len(a):].sum()
    return float((rb - len(b) * (len(b) + 1) / 2) / (len(a) * len(b)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--set", action="append", required=True, help="name=shard_dir[:val|train|all]")
    ap.add_argument("--ctx", type=int, nargs="+", default=[180])
    ap.add_argument("--games", type=int, default=1000, help="max games per set")
    ap.add_argument("--frames-per-game", type=int, default=64)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = "cuda"
    T.DEVICE = dev
    model, cfg, full_ctx, stick, shoulder = load(a.ckpt, dev)
    max_ctx = max(a.ctx + [full_ctx])
    rng = random.Random(a.seed)
    sets, shards = {}, {}
    for spec in a.set:
        name, path = spec.split("=", 1)
        gs = games_of(path, max_ctx)
        rng.shuffle(gs)
        gs = gs[:a.games]
        tg = []
        for gi, (s, lo, hi) in enumerate(gs):
            pool = range(lo + max_ctx - 1, hi - 1)
            tg += [(s, t, gi) for t in rng.sample(pool, min(a.frames_per_game, len(pool)))]
            if s not in shards:
                shards[s] = torch.load(s, weights_only=True, mmap=True)
        tg.sort()
        sets[name] = tg
        print(f"{name}: {len(gs)} games, {len(tg)} target frames", flush=True)
    res = {}
    for k in a.ctx:
        res[k] = {}
        for name, tg in sets.items():
            nll, ent = score(model, cfg, stick, shoulder, shards, [(s, t) for s, t, _ in tg], k, a.batch, dev)
            res[k][name] = dict(nll=nll, ent=ent, game=np.array([g for _, _, g in tg]))
            print(f"  scored ctx={k} {name}", flush=True)
    report = {}
    for k, by in res.items():
        print(f"\n=== context {k} frames ===")
        print(f"{'set':10s} {'NLL':>15s} {'entropy':>15s} {'excess':>15s}   excess by head ("
              + ", ".join(HEADS) + ")")
        report[k] = {}
        for name, r in by.items():
            tot_n, tot_e = r["nll"].sum(1), r["ent"].sum(1)
            ex = tot_n - tot_e
            heads = (r["nll"] - r["ent"]).mean(0)
            f = lambda x: f"{x.mean():.4f}±{clustered_se(x, r['game']):.4f}"
            print(f"{name:10s} {f(tot_n):>15s} {f(tot_e):>15s} {f(ex):>15s}   "
                  + "  ".join(f"{v:+.4f}" for v in heads))
            per_game_ex = np.bincount(r["game"], weights=ex) / np.bincount(r["game"])
            per_game_n = np.bincount(r["game"], weights=tot_n) / np.bincount(r["game"])
            report[k][name] = dict(nll=float(tot_n.mean()), ent=float(tot_e.mean()), excess=float(ex.mean()),
                                   excess_se=clustered_se(ex, r["game"]), excess_by_head=heads.tolist(),
                                   game_excess=per_game_ex.tolist(), game_nll=per_game_n.tolist())
        names = list(by)
        print("game-level AUROC (higher score in the second set; 0.5 = no separation):")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                x, y = report[k][names[i]], report[k][names[j]]
                print(f"  {names[i]} vs {names[j]}:  excess {auroc(x['game_excess'], y['game_excess']):.3f}"
                      f"   NLL {auroc(x['game_nll'], y['game_nll']):.3f}")
    if a.out:
        json.dump(dict(ckpt=a.ckpt, sets={n: len(t) for n, t in sets.items()}, report=report),
                  open(a.out, "w"))


if __name__ == "__main__":
    main()
