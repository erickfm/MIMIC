"""Many native-sim matches stepped in lockstep, policies batched on the GPU.

  CPU: K worker processes, each driving M sims (rlvr/sim/melee_sim.py, one game per process).
       A worker sends every sim its pads before collecting any frame, so its games advance in
       parallel; it parses each frame with libmelee (rlvr/sim/gamestate.py) and builds each
       side's model features (inference_utils.build_frame / build_frame_p2) into shared memory.
  GPU: the calling process holds one BatchedPolicy per character and advances all of that
       character's (env, side) slots in one forward + one vectorized sample per step.

Templates are fixed per env (env i uses templates[i % len]); each episode gets a fresh seed.
Batched decisions equal the per-env PlayerState + decode_and_press loop exactly (argmax, fp32;
checked 2026-10-20, 4 envs x 1,623 frames, 0 differing bytes).
"""
import multiprocessing as mp
import os
import random
import time
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np

EXT_TO_MELEE = {0: "CPTFALCON", 1: "DK", 2: "FOX", 3: "GAMEANDWATCH", 4: "KIRBY", 5: "BOWSER",
                6: "LINK", 7: "LUIGI", 8: "MARIO", 9: "MARTH", 10: "MEWTWO", 11: "NESS",
                12: "PEACH", 13: "PIKACHU", 14: "POPO", 15: "JIGGLYPUFF", 16: "SAMUS",
                17: "YOSHI", 18: "ZELDA", 19: "SHEIK", 20: "FALCO", 21: "YLINK", 22: "DOC",
                23: "ROY", 24: "PICHU", 25: "GANONDORF"}


def template_players(slp):
    """[(port 0-3, CHARACTER name)] from a 1v1 template's Game Start, in port order."""
    from tools.sim_replay_check import slp_events
    for c, p in slp_events(slp):
        if c == 0x36:
            pl = [(i, EXT_TO_MELEE[p[0x64 + 0x24 * i]]) for i in range(4) if p[0x65 + 0x24 * i] != 3]
            if len(pl) != 2:
                raise ValueError(f"{slp}: needs a 1v1 template, has {len(pl)} players")
            return pl
    raise ValueError(f"{slp}: no Game Start")


class _Shm:
    """A named shared numpy array (created by the parent, attached by workers)."""

    def __init__(self, shape, dtype, name=None):
        self.shape, self.dtype = tuple(shape), np.dtype(dtype)
        size = max(1, int(np.prod(self.shape)) * self.dtype.itemsize)
        self.shm = shared_memory.SharedMemory(create=name is None, size=size, name=name)
        self.a = np.ndarray(self.shape, self.dtype, buffer=self.shm.buf)

    def spec(self):
        return (self.shape, self.dtype.str, self.shm.name)


def _worker(wid, env_ids, envs, spec, conn, opts):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
    from inference_utils import build_frame, build_frame_p2, load_inference_context
    from play import _make_ctx
    from rlvr.sim.gamestate import SlippiParser, SlpWriter
    from rlvr.sim.melee_sim import MeleeSim

    feats = {(ch, k): _Shm(*s[:2], name=s[2]) for (ch, k), s in spec["feats"].items()}
    pads = _Shm(*spec["pads"][:2], name=spec["pads"][2])
    ctxs = {ch: _make_ctx(load_inference_context(d), _Cfg(n)) for ch, (d, n) in spec["ctx"].items()}
    rng = random.Random(opts["seed"] * 1000 + wid)
    builds = (build_frame, build_frame_p2)
    S = {}
    for e in env_ids:
        t = envs[e]["template"]
        S[e] = dict(sim=MeleeSim(t, rng=opts["rng"]), parser=None, writer=None, episodes=0,
                    gs=None, t=t)

    def start(e):
        s = S[e]
        s["sim"].seed = rng.getrandbits(32)
        s["parser"] = SlippiParser(s["t"])
        s["writer"] = SlpWriter(s["t"]) if opts["record_dir"] else None
        return s["sim"].reset()

    def emit(e, frame):
        s = S[e]
        if s["writer"] is not None:
            s["writer"].add(frame)
        gs = s["parser"].parse(frame)
        s["gs"] = gs
        for side, (port, ch) in enumerate(envs[e]["players"]):
            fr = builds[side](gs, None, ctxs[ch])
            row = envs[e]["slots"][side]
            for k, v in fr.items():
                if (ch, k) in feats:
                    feats[(ch, k)].a[row] = v[0].numpy()

    def finish(e):
        s = S[e]
        gs = s["gs"]
        res = dict(env=e, episode=s["episodes"], seed=s["sim"].seed, frames=int(gs.frame) if gs else 0,
                   final={ch: (int(gs.players[p + 1].stock), float(gs.players[p + 1].percent))
                          for p, ch in envs[e]["players"]} if gs else {})
        if s["writer"] is not None:
            path = os.path.join(opts["record_dir"], f"env{e:03d}_ep{s['episodes']:04d}.slp")
            s["writer"].write(path)
            res["slp"] = path
        s["episodes"] += 1
        return res

    resets = []
    for e in env_ids:
        emit(e, start(e))
        resets.append(e)
    conn.send(("ready", resets, []))
    while True:
        cmd = conn.recv()
        if cmd == "stop":
            break
        for e in env_ids:
            S[e]["sim"].send(pads.a[e].tobytes())
        resets, done = [], []
        for e in env_ids:
            f = S[e]["sim"].recv()
            if S[e]["sim"].done or f is None or (opts["max_frames"] and f.index >= opts["max_frames"]):
                done.append(finish(e))
                f = start(e)
                resets.append(e)
            emit(e, f)
        conn.send(("stepped", resets, done))
    for s in S.values():
        s["sim"].close()


class _Cfg:   # what _make_ctx reads from a model config
    def __init__(self, n_controller_combos):
        self.n_controller_combos = n_controller_combos


class VecSim:
    def __init__(self, templates, policies, n_envs, n_workers, rng="online", seed=0, temperature=1.0,
                 record_dir=None, max_frames=0, amp=True, compile=True, device="cuda"):
        """policies: {CHARACTER name: checkpoint dir}. templates: 1v1 .slp paths (cycled over envs)."""
        import torch
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
        from inference_utils import build_mock_frame, load_inference_context, load_mimic_model
        from play import _make_ctx
        from rlvr.sim.batch import BatchedPolicy

        self.temperature = temperature
        self.envs = []
        n_slots = {ch: 0 for ch in policies}
        for e in range(n_envs):
            t = str(templates[e % len(templates)])
            players = template_players(t)
            slots = []
            for port, ch in players:
                if ch not in policies:
                    raise ValueError(f"{t}: no policy for {ch}")
                slots.append(n_slots[ch])
                n_slots[ch] += 1
            self.envs.append(dict(template=t, players=players, slots=slots))
        self.slot_env = {ch: np.zeros(n, np.int64) for ch, n in n_slots.items()}
        self.slot_port = {ch: np.zeros(n, np.int64) for ch, n in n_slots.items()}
        for e, env in enumerate(self.envs):
            for (port, ch), s in zip(env["players"], env["slots"]):
                self.slot_env[ch][s] = e
                self.slot_port[ch][s] = port
        self.policies, self.feats, ctx_spec = {}, {}, {}
        for ch, d in policies.items():
            if n_slots[ch] == 0:
                continue
            model, cfg = load_mimic_model(str(Path(d) / "model.pt"), device)
            ctx = _make_ctx(load_inference_context(str(d)), cfg)
            mock = build_mock_frame(ctx)
            self.policies[ch] = BatchedPolicy(model, cfg.max_seq_len, mock, n_slots[ch], device=device,
                                              amp=amp, compile=compile)
            for k, v in mock.items():
                if k != "self_controller":
                    arr = v[0].numpy()
                    self.feats[(ch, k)] = _Shm((n_slots[ch],) + arr.shape, arr.dtype)
            ctx_spec[ch] = (str(d), cfg.n_controller_combos)
        self.pads = _Shm((n_envs, 32), np.uint8)
        spec = dict(feats={k: v.spec() for k, v in self.feats.items()}, pads=self.pads.spec(), ctx=ctx_spec)
        opts = dict(rng=rng, seed=seed, record_dir=record_dir, max_frames=max_frames)
        if record_dir:
            os.makedirs(record_dir, exist_ok=True)
        ctxm = mp.get_context("spawn")
        self.conns, self.procs = [], []
        chunks = np.array_split(np.arange(n_envs), n_workers)
        for w, ids in enumerate(chunks):
            a, b = ctxm.Pipe()
            p = ctxm.Process(target=_worker, args=(w, ids.tolist(), self.envs, spec, b, opts), daemon=True)
            p.start()
            self.conns.append(a)
            self.procs.append(p)
        self.results = []
        self.frames = 0
        self._collect()

    def _collect(self):
        resets = []
        for c in self.conns:
            _, r, done = c.recv()
            resets += r
            self.results += done
        rs = set(resets)
        for ch, pol in self.policies.items():
            pol.reset([s for s, e in enumerate(self.slot_env[ch]) if e in rs])

    def step(self):
        """One frame for every env: decide on the current frames, send the pads, collect."""
        wire = np.zeros_like(self.pads.a)
        for ch, pol in self.policies.items():
            idx = pol.step({k: self.feats[(c, k)].a for (c, k) in self.feats if c == ch},
                           temperature=self.temperature)
            from rlvr.sim.batch import pads_from_indices
            p = pads_from_indices(idx)
            for s in range(len(p)):
                e, port = self.slot_env[ch][s], self.slot_port[ch][s]
                wire[e, 8 * port:8 * port + 8] = p[s]
        self.pads.a[:] = wire
        for c in self.conns:
            c.send("step")
        self._collect()
        self.frames += len(self.envs)

    def close(self):
        for c in self.conns:
            try:
                c.send("stop")
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=10)
            if p.is_alive():
                p.kill()
        for s in list(self.feats.values()) + [self.pads]:
            s.shm.close()
            s.shm.unlink()
