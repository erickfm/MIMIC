#!/usr/bin/env python3
"""Branches from a real game: replay a .slp exactly to frame F in the native sim, then let the
models play K rollouts from that exact state, all K in lockstep with one batched forward per side.

Each rollout runs in its own sim process (one game per process). The K sims replay the recorded
pads up to F at full speed; one of them also parses the last 180 frames to build the models'
history, which every rollout shares (the prefix is identical in all of them). From F on each
side's policy decides every frame (GPU: rlvr.sim.batch.BatchedPolicy, CUDA graph; CPU: the same
computation without the graph). The sim state at F is checked against the recording.

Every rollout can be written as a .slp: the real game up to F, then the branch.

  python3 tools/coach/branch.py GAME.slp --frame 4200 --k 8 --horizon 300 --out-dir /tmp/b
"""
import argparse, json, os, struct, sys, threading, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "tools", ROOT / "tools" / "coach"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

CTRL_SEGMENTS = ((0, 37), (37, 46), (53, 56), (46, 53))
# the lab's fighter states: the action states that count as being in them (libmelee values)
SETTLE = {"stand": {14}, "crouch": {40}, "shield": {179}, "ledge": {253}}
SETTLE_MAX = 45


def settle_wire(fr, settle):
    """The lab's settle step: neutral pads, except what gets each fighter into its state (hold down
    to crouch; R to shield, once grounded: R in the air is an air dodge; the ledge is caught by
    falling past it facing the stage). fr: the frame now (None: the placement frame)."""
    from rlvr.sim.melee_sim import Pad
    out = []
    for q in range(4):
        p = Pad()
        st = settle.get(q)
        post = fr.post.get((q, 0)) if fr is not None else None
        grounded = post is not None and len(post) > 0x2E and post[0x2E] == 0
        if st == "crouch":
            p.stick_y = -80
        elif st == "shield" and grounded:
            p.trigger_r, p.button = 255, 0x0020
        out.append(p)
    return b"".join(p.pack() for p in out)


def is_settled(fr, settle):
    return all((q, 0) in fr.post and struct.unpack_from(">H", fr.post[(q, 0)], 7)[0] in SETTLE[st]
               for q, st in settle.items() if st in SETTLE)   # main, c, shoulder, button -> last_idx order


def ctrl_indices(onehot):
    """The 56-dim controller one-hot of build_frame -> (main, c, shoulder, button) head indices."""
    v = np.asarray(onehot).reshape(-1)
    return [int(np.argmax(v[a:b])) for a, b in CTRL_SEGMENTS]


class CPUPolicy:
    """BatchedPolicy's interface and computation on the CPU (no CUDA graph, fp32)."""

    def __init__(self, model, seq_len, mock_frame, n_envs):
        import torch
        self.torch = torch
        self.model = model.eval()
        self.n, self.T = n_envs, seq_len
        self.mock = {k: v[0] for k, v in mock_frame.items()}
        self.buf = {k: v.unsqueeze(0).unsqueeze(0).repeat(n_envs, seq_len, *[1] * v.dim()).clone()
                    for k, v in self.mock.items()}
        self.last_idx = torch.zeros(n_envs, 4, dtype=torch.long)
        self.last_idx[:, 3] = 6

    def reset(self, env_ids):
        for k, v in self.mock.items():
            self.buf[k][list(env_ids)] = v
        self.last_idx[list(env_ids)] = self.torch.tensor([0, 0, 0, 6])

    def _controller(self):
        F, i = self.torch.nn.functional, self.last_idx
        return self.torch.cat([F.one_hot(i[:, 0], 37), F.one_hot(i[:, 1], 9),
                               F.one_hot(i[:, 3], 7), F.one_hot(i[:, 2], 3)], dim=1).float()

    def step(self, feats, temperature=1.0):
        torch = self.torch
        with torch.no_grad():
            for k, v in feats.items():
                b = self.buf[k]
                b[:, :-1] = b[:, 1:].clone()
                b[:, -1] = torch.as_tensor(v).to(b.dtype)
            b = self.buf["self_controller"]
            b[:, :-1] = b[:, 1:].clone()
            b[:, -1] = self._controller()
            self.model.heads.sample_temperature = float(temperature)
            preds = self.model(self.buf)
            if "cond_sample" in preds:
                self.last_idx = preds["cond_sample"][:, -1].clone()
            else:
                from rlvr.sim.batch import HEADS, _gumbel_sample
                t = torch.tensor(1.0 / temperature if temperature > 0 else 1.0)
                g = torch.tensor(temperature <= 0)
                self.last_idx = torch.stack([_gumbel_sample(preds[h][:, -1].float(), t, g) for h in HEADS], 1)
        return self.last_idx.numpy()


class Brancher:
    """Holds the models (loaded once) and runs branch requests."""

    def __init__(self, device=None, iso=None):
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.iso = iso   # the Melee ISO the sim reads (default: melee.iso in the repo)
        self._models, self._policies = {}, {}
        self.lock = threading.Lock()

    def model(self, policy_dir):
        from inference_utils import load_inference_context, load_mimic_model, set_cond_temperature
        from play import _make_ctx
        d = str(Path(policy_dir).resolve())
        if d not in self._models:
            m, cfg = load_mimic_model(str(Path(d) / "model.pt"), self.device)
            set_cond_temperature(m, 1.0)
            self._models[d] = (m, cfg, _make_ctx(load_inference_context(d), cfg))
        return self._models[d]

    def policy(self, policy_dir, k, slot):
        """A batched policy over k envs; cached per (model, k, side), since a CUDA graph is
        captured for a fixed batch."""
        from inference_utils import build_mock_frame
        key = (str(Path(policy_dir).resolve()), k, slot)
        if key not in self._policies:
            m, cfg, ctx = self.model(policy_dir)
            mock = build_mock_frame(ctx)
            if self.device == "cuda":
                from rlvr.sim.batch import BatchedPolicy
                pol = BatchedPolicy(m, cfg.max_seq_len, mock, k, device="cuda", compile=False)
            else:
                pol = CPUPolicy(m, cfg.max_seq_len, mock, k)
            self._policies[key] = pol
        return self._policies[key]

    def run(self, slp, frame, k, horizon, policies, temperature=1.0, out_dir=None, progress=None, value_at=None,
            place=None, pads_slp=None, force=None):
        """policies: {port (0-based): policy dir} for each side the models play from `frame`.
        Returns dict(exact, frames=[...], rollouts=[...]) with per-frame states of both players."""
        with self.lock:
            return self._run(str(slp), int(frame), int(k), int(horizon), policies, temperature,
                             out_dir, progress, value_at, place, pads_slp, force)

    def _run(self, slp, F, k, horizon, policies, temperature, out_dir, progress, value_at=None, place=None,
             pads_slp=None, force=None):
        """value_at: (port, [steps]): after each listed step, keep that side's model features at its
        newest frame (the input to tools/coach/value.Value), per rollout.
        place: (PF, {port: (x, y, facing, percent)}[, {port: state}]): the lab. After frame PF the fighters
        are put there (MeleeSim.place); the models' history starts over after it. Needs F > PF. With
        states ("stand", "crouch", "shield", "ledge"; "air" = as placed) the fighters then press what
        gets them there (SETTLE) until all are in it, at most 45 frames, and the run starts from that
        frame (F is then that frame, not the one asked for).
        pads_slp: take the inputs up to F from this .slp instead (a branch written earlier: the real
        game, the placement, then that try), so a branch can continue from any frame of a try.
        force: (port, slp, n), or a list of them (both sides): for the first n live frames, that side presses exactly what it pressed
        in `slp` (a try, or the real game) from frame F on; its model then takes over, with the forced
        inputs in its controller history. Used to test one option against many opponent responses."""
        import torch
        from inference_utils import build_frame, build_frame_p2
        from rlvr.sim.batch import pads_from_indices
        from rlvr.sim.gamestate import SlippiParser, SlpWriter
        from rlvr.sim.melee_sim import MeleeSim, Pad
        from sim_play import template_players
        from tools.sim_env_check import recorded
        from tools.sim_replay_check import physics_diff, slp_events
        from copy_test import prev_from_controller
        from pads import pad_from_pre

        t0 = time.time()
        players = template_players(slp)
        if len(players) != 2:
            raise ValueError("branches need a 1v1 game")
        builds = dict(zip([p for p, _ in players], (build_frame, build_frame_p2)))
        pre, post = recorded(pads_slp or slp)
        rec_seed = {struct.unpack(">i", p[:4])[0]: struct.unpack(">I", p[4:8])[0]
                    for c, p in slp_events(slp) if c == 0x3A}
        last_rec = max(f for f, _, _ in pre)
        if not (-123 < F <= last_rec):
            raise ValueError(f"frame {F} is outside the game (-122..{last_rec})")
        rec_wire = {}

        def rec_pads(f):
            if f not in rec_wire:
                rec_wire[f] = b"".join((pad_from_pre(pre[(f, q, 0)]) if (f, q, 0) in pre else Pad()).pack()
                                       for q in range(4))
            return rec_wire[f]

        sides = []
        for port, d in sorted(policies.items()):
            m, cfg, ctx = self.model(d)
            sides.append(dict(port=port, dir=d, ctx=ctx, build=builds[port], pol=self.policy(d, k, len(sides))))
        T = max(s["pol"].T for s in sides)

        # 1) prefix: K sims replay the recording to F; sim 0 also parses its last T frames.
        sims = [MeleeSim(slp, seed=rec_seed[min(rec_seed)], rng="online", **({"iso": self.iso} if self.iso else {}))
                for _ in range(k)]
        hist = {s["port"]: [] for s in sides}
        writer = SlpWriter(slp) if out_dir else None
        cur = [None] * k
        parser0 = SlippiParser(slp)

        settle = {int(q): st for q, st in (place[2] if place and len(place) > 2 and not pads_slp else {}).items()
                  if st in SETTLE}

        unsettled = []      # fighters the settle couldn't get into their states
        settle_pads = lambda fr: settle_wire(fr, settle)
        settled = lambda fr: is_settled(fr, settle)

        def keep(i, fr):
            if i == 0:
                if writer:
                    writer.add(fr)
                if fr.index > (place[0] if place else F - T - 1):
                    gs = parser0.parse(fr)
                    for s in sides:
                        prev = prev_from_controller(gs.players[s["port"] + 1].controller_state)
                        hist[s["port"]].append(s["build"](gs, prev, s["ctx"]))

        def prefix(i):
            fr = sims[i].reset(initial_pads=[pad_from_pre(pre[(-123, q, 0)]) if (-123, q, 0) in pre else Pad()
                                             for q in range(4)])
            while fr is not None and fr.index < F:
                keep(i, fr)
                if place and fr.index == place[0]:
                    fr = sims[i].place(place[1], settle_pads(None) if settle else rec_pads(fr.index + 1))
                    if settle:
                        # into the asked-for states; the run starts once all are in them
                        n = 0
                        while fr is not None and not settled(fr) and n < SETTLE_MAX:
                            keep(i, fr)
                            fr = sims[i].step(settle_pads(fr))
                            n += 1
                        break
                else:
                    fr = sims[i].step(rec_pads(fr.index + 1))
            cur[i] = fr

        ths = [threading.Thread(target=prefix, args=(i,)) for i in range(k)]
        [t.start() for t in ths]
        [t.join() for t in ths]
        if settle and all(c is not None for c in cur):
            if len({c.index for c in cur}) != 1:
                raise RuntimeError("the sims settled on different frames")
            F = cur[0].index
            unsettled = [q for q, st in settle.items() if not is_settled(cur[0], {q: st})]
        if any(c is None or c.index != F for c in cur):
            raise RuntimeError("The simulator lost sync with this replay before this moment, so it can't play it out from here.")
        exact = (all(not physics_diff(post[(F, s, 0)], c.post[(s, 0)]) for c in cur for s, _ in players)
                 if (F, players[0][0], 0) in post and not (place and not pads_slp) else None)
        t_prefix = time.time() - t0
        writers = None
        if writer:
            base = bytes(writer.raw)
            writers = []
            for _ in range(k):
                w = SlpWriter.__new__(SlpWriter)
                w.payloads, w.game_start, w.sizes, w.raw = writer.payloads, writer.game_start, writer.sizes, bytearray(base)
                writers.append(w)

        # 2) prime each side's window with the shared history; the first live step pushes frame F.
        parsers = [parser0] + [SlippiParser(slp) for _ in range(k - 1)]
        gs_F = [parsers[i].parse(cur[i]) for i in range(k)]
        for s in sides:
            pol, h = s["pol"], hist[s["port"]]
            pol.reset(range(k))
            n = min(len(h), pol.T)
            for key in (pol.buf if n else []):
                v = torch.cat([f[key] for f in h[-n:]], 0).to(pol.buf[key].device, pol.buf[key].dtype)
                pol.buf[key][:, -n:].copy_(v.unsqueeze(0).expand(k, *v.shape))
            ctrl = s["build"](gs_F[0], prev_from_controller(gs_F[0].players[s["port"] + 1].controller_state),
                              s["ctx"])["self_controller"][0].numpy()
            pol.last_idx.copy_(torch.tensor([ctrl_indices(ctrl)] * k, device=pol.last_idx.device))

        # 3) lockstep rollouts
        def snap(gs):
            out = {}
            for p, _ in players:
                ps = gs.players[p + 1]
                out[p] = [round(float(ps.position.x), 3), round(float(ps.position.y), 3), int(bool(ps.facing)),
                          int(ps.action.value), int(ps.action_frame), round(float(ps.percent), 2), int(ps.stock),
                          int(bool(ps.on_ground))]
            return out

        roll = [dict(frames=[snap(gs_F[i])], dealt={}, done=False) for i in range(k)]
        # forced sides: {port: {step: wire}}; `force` is one (port, slp, n) or a list of them
        fwires = {}
        for fport, fslp, fn in ([force] if force and isinstance(force[0], int) else (force or [])):
            fpre, _ = recorded(fslp)
            fw = fwires.setdefault(fport, {})
            for st_ in range(int(fn)):
                key = (F + st_ + 1, fport, 0)
                if key not in fpre:
                    break
                fw[st_] = pad_from_pre(fpre[key]).pack()
        feats_v = {}
        gss = gs_F
        for step in range(horizon):
            if writers:
                for i in range(k):
                    writers[i].add(cur[i])
            pads = [bytearray(b"".join(Pad().pack() for _ in range(4))) for _ in range(k)]
            for s in sides:
                feats = {}
                for i in range(k):
                    fr = s["build"](gss[i], None, s["ctx"])
                    for key, v in fr.items():
                        if key != "self_controller":
                            feats.setdefault(key, []).append(v[0].numpy())
                idx = s["pol"].step({key: np.stack(v) for key, v in feats.items()}, temperature)
                if value_at and s["port"] == value_at[0] and step in value_at[1]:
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device == "cuda"):
                        buf = {kk: vv.clone() for kk, vv in s["pol"].buf.items()}
                        feats_v[step] = self.model(s["dir"])[0].features(buf)[:, -1].float()
                wire = pads_from_indices(idx)
                for i in range(k):
                    fw = fwires.get(s["port"], {})
                    pads[i][8 * s["port"]:8 * s["port"] + 8] = fw[step] if step in fw else wire[i].tobytes()
            live = [i for i in range(k) if not sims[i].done]
            for i in live:
                sims[i].send(bytes(pads[i]))
            for i in live:
                fr = sims[i].recv()
                if fr is None:
                    continue
                cur[i] = fr
                gss[i] = parsers[i].parse(fr)
                roll[i]["frames"].append(snap(gss[i]))
                for fside in sides:   # a forced side's model sees what was really pressed
                    if step not in fwires.get(fside["port"], {}):
                        continue
                    cs = gss[i].players[fside["port"] + 1].controller_state
                    ctrl = fside["build"](gss[i], prev_from_controller(cs), fside["ctx"])["self_controller"][0].numpy()
                    li = fside["pol"].last_idx
                    li[i].copy_(torch.tensor(ctrl_indices(ctrl), device=li.device, dtype=li.dtype))
            if progress:
                progress((step + 1) / horizon)
            if not live:
                break
        for sm in sims:
            sm.close()

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        for i, r in enumerate(roll):
            if writers:
                p = Path(out_dir) / f"branch_{F}_{i}.slp"
                import datetime as _dt
                writers[i].write(str(p), metadata=dict(
                    startAt=_dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    lastFrame=int(cur[i].index) - 1 if cur[i] is not None else F, playedOn="mimic-coach"))
                r["slp"] = str(p)
            first, last = r["frames"][0], r["frames"][-1]
            r["outcome"] = {}
            for p, _ in players:
                pct = [f[p][5] for f in r["frames"]]
                st = [f[p][6] for f in r["frames"]]
                r["outcome"][p] = dict(damage_taken=round(float(sum(max(0.0, b - a) for a, b in zip(pct, pct[1:]))), 2),
                                       stocks_lost=int(sum(max(0, a - b) for a, b in zip(st, st[1:]))))
        return dict(slp=slp, frame=F, exact=exact, unsettled=unsettled, value_feats=feats_v, players=[[p, c] for p, c in players],
                    policies={str(p): d for p, d in policies.items()}, rollouts=roll,
                    seconds=dict(prefix=round(t_prefix, 2), total=round(time.time() - t0, 2)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("slp")
    ap.add_argument("--frame", type=int, required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=300)
    ap.add_argument("--policy", action="append", required=True, metavar="PORT=DIR",
                    help="0-based port the model plays from the branch frame, e.g. 0=runs/x/policy")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    if a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)
    pol = {int(p): d for p, d in (x.split("=", 1) for x in a.policy)}
    r = Brancher(a.device).run(a.slp, a.frame, a.k, a.horizon, pol, out_dir=a.out_dir)
    print(json.dumps(dict(exact=r["exact"], seconds=r["seconds"],
                          outcomes=[x["outcome"] for x in r["rollouts"]],
                          slps=[x.get("slp") for x in r["rollouts"]])))
    if a.json:
        json.dump(r, open(a.json, "w"))


if __name__ == "__main__":
    main()
