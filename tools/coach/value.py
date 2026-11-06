#!/usr/bin/env python3
"""Value model V: from a game state, the expected outcome over the next 5 seconds (300 frames)
for one player: damage dealt, damage taken, stocks taken, stocks lost.

Input: the character's BC model's own reading of the situation, i.e. its trunk output at the
last frame of the usual 180-frame window (model.features), built exactly as at inference
(inference_utils.build_frame from libmelee GameStates). A small MLP head maps it to the four
outcomes. The trunk is frozen.

  extract: replays (any player of CHAR, current or old Slippi) -> features + labels (.npz)
  train:   fit the head, report held-out-game R^2
  python3 tools/coach/value.py extract --char FALCO --replays ~/Slippi --out data/coach_rank/vfeat_falco.npz
  python3 tools/coach/value.py train --data data/coach_rank/vfeat_falco.npz --out data/coach_rank/value_falco.pt
"""
import argparse, glob, hashlib, os, struct, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "tools", ROOT / "tools" / "coach"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

H = 300
STOCK = 40.0     # a stock counts as 40% in the single score


def score(d):
    """The single number the app shows: dealt - taken + 40 per stock difference."""
    return d[..., 0] - d[..., 1] + STOCK * (d[..., 2] - d[..., 3])


def policy_for(char):
    from serve import POLICY
    return str(ROOT / POLICY[char])


def game_frames(slp):
    """The replay's frames (last copy of each rolled-back frame), in order, as sim Frames."""
    from rlvr.sim.melee_sim import to_frames
    from tools.sim_replay_check import slp_events
    evs = list(slp_events(slp))
    fr = {}
    for f in to_frames(evs):
        fr[f.index] = f
    return [fr[k] for k in sorted(fr)]


_CTX = None


def _per_game(job):
    """CPU side: (slp, port, policy_dir, stride) -> stacked features of every frame + labels.
    One torch thread while it runs (it's run many at a time in extract's pool), then the caller's
    count again: torch's thread count is per thread, and the coach's job worker runs its CPU
    rollouts after this on the same thread (left at 1, a CPU branch took 4-5x longer)."""
    import torch
    n0 = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return _per_game_1(job)
    finally:
        torch.set_num_threads(n0)


def _per_game_1(job):
    import torch
    from inference_utils import build_frame, build_frame_p2, load_inference_context, load_mimic_model
    from play import _make_ctx
    from rlvr.sim.gamestate import SlippiParser
    from copy_test import prev_from_controller
    slp, port, pdir, stride = job
    global _CTX
    if _CTX is None or _CTX[0] != pdir:
        _, cfg = load_mimic_model(str(Path(pdir) / "model.pt"), "cpu")
        _CTX = (pdir, _make_ctx(load_inference_context(pdir), cfg))
    ctx = _CTX[1]
    try:
        frames = game_frames(slp)
        if len(frames) < 1200:      # replays without metadata (the ranked dataset) are checked here
            return None
        parser = SlippiParser(slp)
        ports = sorted(p for (p, _) in frames[-1].post)
        if len(ports) != 2 or port not in ports:
            return None
        other = [p for p in ports if p != port][0]
        build = build_frame if port == ports[0] else build_frame_p2
        feats, pct, opct, stk, ostk, fidx = {}, [], [], [], [], []
        for fr in frames:
            gs = parser.parse(fr)
            if fr.index < -60:
                continue
            me, op = gs.players[port + 1], gs.players[other + 1]
            x = build(gs, prev_from_controller(me.controller_state), ctx)
            for k, v in x.items():
                feats.setdefault(k, []).append(v[0].numpy())
            pct.append(float(me.percent)); opct.append(float(op.percent))
            stk.append(int(me.stock)); ostk.append(int(op.stock)); fidx.append(fr.index)
    except Exception as e:
        return dict(error=f"{slp}: {e}"[:300])
    feats = {k: np.stack(v) for k, v in feats.items()}
    pct, opct, stk, ostk = map(np.asarray, (pct, opct, stk, ostk))
    n = len(pct)
    ts = np.arange(180, n - H - 1, stride)
    lab = []
    for t in ts:
        d = lambda v: np.clip(np.diff(v[t:t + H + 1]), 0, None).sum()
        s = lambda v: np.clip(-np.diff(v[t:t + H + 1]), 0, None).sum()
        lab.append((d(opct), d(pct), s(ostk), s(stk)))
    return dict(slp=slp, port=port, feats=feats, ts=ts, frames=np.asarray(fidx)[ts], labels=np.asarray(lab, np.float32))


def candidates(replays, char):
    from peppi_py import read_slippi
    from sim_play import EXT_TO_MELEE
    out = []
    for p in sorted(glob.glob(os.path.join(replays, "**", "*.slp"), recursive=True)):
        try:
            g = read_slippi(p, skip_frames=True)
        except Exception:
            continue
        pls = [(i, pl) for i, pl in enumerate(g.start.players) if pl]
        if len(pls) != 2 or g.start.slippi.version[:2] < (3, 0):
            continue
        n = (g.metadata or {}).get("lastFrame")
        if n is not None and n < 1200:   # no metadata (ranked replays): the length is checked per game
            continue
        for i, pl in pls:
            if EXT_TO_MELEE.get(pl.character) == char:
                out.append((p, i))
    return out


def file_sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def is_val(slp):
    return int(hashlib.sha1(os.path.basename(slp).encode()).hexdigest(), 16) % 10 == 0


def extract(a):
    import torch
    from inference_utils import load_mimic_model
    pdir = str(Path(a.policy).resolve()) if a.policy else policy_for(a.char)
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_mimic_model(str(Path(pdir) / "model.pt"), dev)
    model.eval()
    cands = candidates(os.path.expanduser(a.replays), a.char)
    if a.max_games:
        import random
        random.Random(0).shuffle(cands)
        cands = cands[:a.max_games]
    print(f"{len(cands)} {a.char} perspectives", flush=True)
    out = dict(x=[], y=[], frame=[], game=[], val=[])
    games = []
    t0 = time.time()
    import multiprocessing as mp
    with ProcessPoolExecutor(a.jobs, mp_context=mp.get_context("spawn")) as ex:
        for gi, r in enumerate(ex.map(_per_game, [(s, p, pdir, a.stride) for s, p in cands], chunksize=1)):
            if r is None or "error" in r:
                if r:
                    print("  skip:", r["error"][:150], flush=True)
                continue
            f, ts = r["feats"], r["ts"]
            if len(ts) == 0:
                continue
            gid = len(games)
            games.append(f"{r['slp']}#{r['port']}")
            for b in range(0, len(ts), a.batch):
                tb = ts[b:b + a.batch]
                idx = tb[:, None] + np.arange(-179, 1)[None]          # (B, 180)
                batch = {k: torch.from_numpy(v[idx]).to(dev) for k, v in f.items()}
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                    h = model.features(batch)[:, -1].float()
                out["x"].append(h.cpu().numpy().astype(np.float16))
            out["y"].append(r["labels"]); out["frame"].append(r["frames"])
            out["game"].append(np.full(len(ts), gid)); out["val"].append(np.full(len(ts), is_val(r["slp"])))
            if (gi + 1) % 25 == 0:
                print(f"  {gi + 1}/{len(cands)} games, {sum(len(y) for y in out['y'])} samples, {time.time() - t0:.0f}s", flush=True)
    np.savez(a.out, **{k: np.concatenate(v) for k, v in out.items()}, games=np.array(games), policy=np.array(pdir),
             policy_sha1=np.array(file_sha1(Path(pdir) / "model.pt")))
    print("saved", a.out, flush=True)


def make_head(d):
    import torch.nn as nn
    return nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 256), nn.GELU(), nn.Dropout(0.1),
                         nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 4))


def train(a):
    import torch
    z = np.load(a.data)
    x, y, val = z["x"].astype(np.float32), z["y"].astype(np.float32), z["val"].astype(bool)
    print(f"{len(x)} samples, {val.mean():.1%} held out (by game); games {len(z['games'])}")
    if not val.any() or val.all():
        sys.exit("need both held-out and training games (value.is_val holds out ~10% of games by file name)")
    scale = np.array([30, 30, 1, 1], np.float32)
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    X, Y = torch.from_numpy(x).to(dev), torch.from_numpy(y / scale).to(dev)
    tr, va = torch.from_numpy(~val).to(dev), torch.from_numpy(val).to(dev)
    head = make_head(x.shape[1]).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=0.01)
    Xt, Yt = X[tr], Y[tr]
    n = len(Xt)
    best, best_state = -1e9, None
    for ep in range(a.epochs):
        head.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, 1024):
            j = perm[i:i + 1024]
            loss = torch.nn.functional.smooth_l1_loss(head(Xt[j]), Yt[j])
            opt.zero_grad(); loss.backward(); opt.step()
        r2 = evaluate(head, X[va], y[val], scale)
        if r2["score"] > best:
            best, best_state = r2["score"], {k: v.detach().clone() for k, v in head.state_dict().items()}
        print(f"ep {ep}: held-out R^2 score {r2['score']:.3f}  dealt {r2['dealt']:.3f}  taken {r2['taken']:.3f}  "
              f"stocks+ {r2['stk_t']:.3f}  stocks- {r2['stk_l']:.3f}", flush=True)
    head.load_state_dict(best_state)
    # the policy whose trunk made the features: V only means anything on that model's features
    pol = str(z["policy"]) if "policy" in z.files else None
    sha = str(z["policy_sha1"]) if "policy_sha1" in z.files else None   # the policy's identity on any machine
    torch.save(dict(state=best_state, d=x.shape[1], scale=scale.tolist(), heldout_r2=best, policy=pol,
                    policy_sha1=sha), a.out)
    print("saved", a.out, "best held-out R^2", round(best, 3))


def evaluate(head, X, y, scale):
    import torch
    head.eval()
    with torch.no_grad():
        p = head(X).cpu().numpy() * scale
    r2 = lambda t, q: 1 - ((t - q) ** 2).mean() / t.var()
    return dict(score=r2(score(y), score(p)), dealt=r2(y[:, 0], p[:, 0]), taken=r2(y[:, 1], p[:, 1]),
                stk_t=r2(y[:, 2], p[:, 2]), stk_l=r2(y[:, 3], p[:, 3]))


class Value:
    """V for one character at serving time: value(feature vectors) -> (N, 4) outcomes."""

    def __init__(self, path, device):
        import torch
        ck = torch.load(path, map_location=device, weights_only=False)
        self.head = make_head(ck["d"]).to(device)
        self.head.load_state_dict(ck["state"])
        self.head.eval()
        self.scale = np.asarray(ck["scale"], np.float32)
        self.r2 = ck.get("heldout_r2")
        self.policy = ck.get("policy")   # the policy dir whose features it was trained on (None: before this was kept)
        self.policy_sha1 = ck.get("policy_sha1")   # that policy's model.pt, by content (None: before this was kept)

    def __call__(self, h):
        import torch
        with torch.no_grad():
            return self.head(h.float()).cpu().numpy() * self.scale


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--char", required=True)
    e.add_argument("--replays", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--stride", type=int, default=30)
    e.add_argument("--batch", type=int, default=128)
    e.add_argument("--jobs", type=int, default=12)
    e.add_argument("--max-games", type=int, default=0)
    e.add_argument("--device", help="cuda or cpu (default: cuda if available)")
    e.add_argument("--policy", help="policy dir whose trunk gives the features (default: the coach's policy for --char)")
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--device", help="cuda or cpu (default: cuda if available)")
    t.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()
    extract(a) if a.cmd == "extract" else train(a)


if __name__ == "__main__":
    main()


def game_value(slp, port, pdir, model, head, device="cuda", stride=2, batch=256, between=None):
    """V along a whole game for one player: (frames, (N, 4) predicted outcomes), every `stride` frames
    from the first full 180-frame window on. model: the char's BC model (features), head: Value.
    between(fraction done): called after each batch (progress, and letting other work run)."""
    import torch
    r = _per_game((slp, port, pdir, 10 ** 9))
    if r is None or "error" in r:
        raise RuntimeError(r["error"] if r else "not a 1v1 game")
    f = r["feats"]
    n = len(f["stage"])
    ts = np.arange(179, n, stride)
    frames_all = _frame_index(slp)
    outs = []
    for b in range(0, len(ts), batch):
        tb = ts[b:b + batch]
        idx = tb[:, None] + np.arange(-179, 1)[None]
        x = {k: torch.from_numpy(v[idx]).to(device) for k, v in f.items()}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            h = model.features(x)[:, -1].float()
        outs.append(head(h))
        if between:
            between(min(1.0, (b + batch) / len(ts)))
    return frames_all[ts], np.concatenate(outs)


def _frame_index(slp):
    """Frame index of each row of _per_game's feature arrays (frames >= -60, rollbacks deduped)."""
    return np.asarray([fr.index for fr in game_frames(slp) if fr.index >= -60])
