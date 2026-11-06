"""Try it yourself: play a moment of a real game live, against the bot, from the exact game state.

The native sim replays the recording to a second before the moment (the lead-in: what really
happened, so the player sees the run-up), then hands one side to the player's controller and the
other to the bot (a policy on the CPU: no GPU while training runs). One game frame per tick at 60
per second (or slower, for practice). Each frame's Slippi events go to the page, which appends
them to the replay its viewer shows. At the end the attempt is scored like the master's tries
(what happened over the decision's stretch, plus the value model after it) and saved as a .slp.
"""
import os, queue, struct, threading, time
from pathlib import Path

import numpy as np

# GameCube PADStatus button bits (rlvr/sim/melee_sim.Pad.button)
BUTTONS = {"a": 0x0100, "b": 0x0200, "x": 0x0400, "y": 0x0800, "start": 0x1000, "z": 0x0010,
           "r": 0x0020, "l": 0x0040, "dleft": 0x0001, "dright": 0x0002, "ddown": 0x0004, "dup": 0x0008}


def pad_from_json(d):
    """The page's controller state -> a raw pad. d: {stick: [x, y], c: [x, y] (each -1..1),
    l, r (analog 0..1), buttons: [names]}. Sticks are raw = 80 * value (the game's full tilt),
    triggers raw = 140 * value (pads.pad_from_pre's scale), so recorded inputs round-trip exactly."""
    from rlvr.sim.melee_sim import Pad
    s8 = lambda v: max(-128, min(127, int(round(float(v) * 80))))
    tr = lambda v: max(0, min(255, int(round(float(v) * 140))))
    st, c = d.get("stick") or (0, 0), d.get("c") or (0, 0)
    bits = 0
    for b in d.get("buttons") or ():
        bits |= BUTTONS.get(b, 0)
    return Pad(button=bits, stick_x=s8(st[0]), stick_y=s8(st[1]), sub_x=s8(c[0]), sub_y=s8(c[1]),
               trigger_l=tr(d.get("l", 0)), trigger_r=tr(d.get("r", 0)))


def json_from_pad(p):
    """The inverse of pad_from_json (scripted tests feed recorded inputs through the page's path)."""
    return dict(stick=[p.stick_x / 80, p.stick_y / 80], c=[p.sub_x / 80, p.sub_y / 80],
                l=p.trigger_l / 140, r=p.trigger_r / 140,
                buttons=[n for n, b in BUTTONS.items() if p.button & b])


class LiveSession:
    """One attempt. prepare() replays to the lead-in's start; run() plays it live."""

    def __init__(self, cpu, slp, me, opp, moment, stretch, bot_dir, me_dir, value, iso=None, lead=60,
                 after=180, out_dir=None, ghost=None, script=None, place=None):
        """cpu: a branch.Brancher on the CPU (models). me/opp: (port, libmelee char). moment: the game
        frame the player takes over at; the attempt runs `after` frames past the stretch.
        value: tools/coach/value.Value or None. script: {port: slp} feed these recorded inputs
        instead of the player/bot (tests). place: (at, {port: (x, y, facing, percent)}, {port: state})
        the lab: after frame `at` the fighters are put there and settle into their states
        (branch.settle_wire); the attempt starts there, with no run-up."""
        self.cpu, self.slp, self.iso = cpu, str(slp), iso
        self.me, self.opp = me, opp
        self.moment, self.stretch, self.lead, self.after = int(moment), int(stretch), int(lead), int(after)
        self.start = self.moment - self.lead
        self.end = self.moment + self.stretch + self.after
        self.bot_dir, self.me_dir, self.value = bot_dir, me_dir, value
        self.out_dir = Path(out_dir) if out_dir else None
        self.ghost, self.script, self.place = ghost, script or {}, place
        if place:
            self.start = int(place[0])
        self.state, self.error, self.result = "preparing", None, None
        self.pad = None              # the player's latest pad (pad_from_json), None = neutral
        self.latch = 0               # buttons pressed since the last frame took the pad (a quick tap still counts)
        self.speed = 1.0
        self.listeners = []          # queues of ("f", bytes) / ("s", dict) for the page's streams
        self.timing = dict(late=0, ticks=0, max_ms=0.0, bot_ms=[])
        self._stop = threading.Event()
        self._go = threading.Event()
        self.prefix = None           # the replay up to the lead-in's start, for the page to load

    # ── page side ──
    def listen(self):
        q = queue.Queue()
        self.listeners.append(q)
        q.put(("s", self.status()))
        return q

    def _emit(self, kind, data):
        for q in list(self.listeners):
            q.put((kind, data))

    def status(self):
        return dict(state=self.state, error=self.error, result=self.result, start=self.start,
                    moment=self.moment, end=self.end, speed=self.speed, sizes=getattr(self, "sizes", None),
                    bot=Path(self.bot_dir).name if self.bot_dir else None, bot_policy=str(self.bot_dir),
                    max_speed=getattr(self, "max_speed", 1.0), unsettled=getattr(self, "unsettled", []))

    def _set(self, state, **kw):
        self.state = state
        for k, v in kw.items():
            setattr(self, k, v)
        self._emit("s", self.status())

    def set_pad(self, p):
        self.pad = p
        self.latch |= p.button

    def go(self, speed=1.0):
        self.speed = max(0.1, min(float(speed), getattr(self, "max_speed", 1.0)))
        self._go.set()

    def stop(self):
        self._stop.set()
        self._go.set()

    # ── game side ──
    def prepare(self):
        import torch
        from inference_utils import build_frame, build_frame_p2
        from copy_test import prev_from_controller
        from pads import pad_from_pre
        from rlvr.sim.gamestate import SlippiParser, SlpWriter
        from rlvr.sim.melee_sim import MeleeSim, Pad
        from sim_play import template_players
        from tools.sim_env_check import recorded
        from tools.sim_replay_check import slp_events
        try:
            players = template_players(self.slp)
            builds = dict(zip([p for p, _ in players], (build_frame, build_frame_p2)))
            self.pre, _ = recorded(self.slp)
            seed = {struct.unpack(">i", p[:4])[0]: struct.unpack(">I", p[4:8])[0]
                    for c, p in slp_events(self.slp) if c == 0x3A}
            self.scripts = {port: recorded(s)[0] for port, s in self.script.items()}
            self.Pad, self.pad_from_pre, self.prev_from = Pad, pad_from_pre, prev_from_controller
            # the two sides' models: the bot decides; the player's own model only keeps a window of
            # frames, for the value model's read at the end of the stretch
            self.sides = {}
            for port, d in ((self.opp[0], self.bot_dir), (self.me[0], self.me_dir)):
                m, cfg, ctx = self.cpu.model(d)
                pol = self.cpu.policy(d, 1, 0 if port == self.opp[0] else 1)
                pol.reset([0])
                self.sides[port] = dict(model=m, ctx=ctx, pol=pol, build=builds[port])
            # how fast the bot decides here: the game can't run faster than that
            bot = self.sides[self.opp[0]]
            mock = {k: v[0, -1].numpy()[None] for k, v in bot["pol"].buf.items() if k != "self_controller"}
            ts = []
            for _ in range(12):
                t = time.perf_counter()
                bot["pol"].step(mock)
                ts.append(time.perf_counter() - t)
            bot["pol"].reset([0])
            p90 = sorted(ts[2:])[int(0.9 * (len(ts) - 2))]
            self.bot_ms = round(1000 * p90, 1)
            self.max_speed = 1.0 if p90 < 0.0125 else round(max(0.25, 0.0125 / p90), 2)
            self.sim = MeleeSim(self.slp, seed=seed[min(seed)], rng="online", **({"iso": self.iso} if self.iso else {}))
            self.parser = SlippiParser(self.slp)
            self.writer = SlpWriter(self.slp)
            self.sizes = {int(k): int(v) for k, v in self.writer.sizes.items()}
            rec = lambda f: b"".join((pad_from_pre(self.pre[(f, q, 0)]) if (f, q, 0) in self.pre else Pad()).pack()
                                     for q in range(4))
            self.rec = rec
            T = max(s["pol"].T for s in self.sides.values())
            fr = self.sim.reset(initial_pads=[pad_from_pre(self.pre[(-123, q, 0)]) if (-123, q, 0) in self.pre else Pad()
                                              for q in range(4)])
            while fr is not None and fr.index < self.start:
                self.writer.add(fr)
                if fr.index > self.start - T - 1 and not self.place:
                    self._push(self.parser.parse(fr))
                fr = self.sim.step(rec(fr.index + 1))
            if fr is None or fr.index != self.start:
                raise RuntimeError("The simulator lost sync with this replay before this moment.")
            if self.place:
                # the lab: put the fighters there; the models' history starts over after it
                from branch import SETTLE_MAX, is_settled, settle_wire
                at, places, states = self.place
                settle = {int(q): st for q, st in (states or {}).items()}
                for sd in self.sides.values():
                    sd["pol"].reset([0])
                self.writer.add(fr)
                fr = self.sim.place({int(q): v for q, v in places.items()}, settle_wire(None, settle))
                n = 0
                while fr is not None and not is_settled(fr, settle) and n < SETTLE_MAX:
                    self.writer.add(fr)
                    self._push(self.parser.parse(fr))
                    fr = self.sim.step(settle_wire(fr, settle))
                    n += 1
                if fr is None:
                    raise RuntimeError("The simulator stopped while setting up the moment.")
                self.unsettled = [q for q, st in settle.items() if not is_settled(fr, {q: st})]
                self.start = self.moment = fr.index
                self.end = self.moment + self.stretch + self.after
            self.cur = fr
            self.prefix = self._slp_bytes()
            self._set("ready")
        except Exception as e:
            self._set("error", error=str(e)[:300])
            raise

    def _push(self, gs):
        """Both sides' windows take this frame as it was (features + the pad pressed on it)."""
        import torch
        for port, s in self.sides.items():
            fr = s["build"](gs, self.prev_from(gs.players[port + 1].controller_state), s["ctx"])
            for k, v in fr.items():
                b = s["pol"].buf[k]
                b[:, :-1] = b[:, 1:].clone()
                b[:, -1] = torch.as_tensor(v[0]).to(b.dtype)

    def _slp_bytes(self):
        """The replay so far, without a Game End (it goes on live), with the metadata the viewer reads."""
        raw = bytes(self.writer.raw)
        md = b""
        for k, v in (("startAt", "2026-01-01T00:00:00Z"), ("lastFrame", int(self.cur.index) - 1), ("playedOn", "mimic-coach")):
            kb = k.encode()
            md += b"U" + bytes([len(kb)]) + kb + (b"l" + struct.pack(">i", v) if isinstance(v, int)
                                                  else b"SU" + bytes([len(v)]) + v.encode())
        return b"{U\x03raw[$U#l" + struct.pack(">I", len(raw)) + raw + b"U\x08metadata{" + md + b"}}"

    def run(self):
        """Wait for go(), then play from the lead-in's start to the end, one frame per tick."""
        import torch
        from branch import ctrl_indices
        from rlvr.sim.batch import pads_from_indices
        try:
            self._go.wait()
            if self._stop.is_set():
                self._set("stopped")
                return
            self._set("live")
            bot, mine = self.sides[self.opp[0]], self.sides[self.me[0]]
            mp, op = self.me[0], self.opp[0]
            snaps, gs = [], self.parser.parse(self.cur)
            period = 1.0 / (60.0 * self.speed)
            t_next = time.perf_counter()
            vfeat = None
            while self.cur.index < self.end and not self._stop.is_set():
                f = self.cur.index
                # this frame's events, as the .slp has them, to the page
                n0 = len(self.writer.raw)
                self.writer.add(self.cur)
                self._emit("f", bytes(self.writer.raw[n0:]))
                if f < self.moment:
                    # the lead-in: what really happened
                    self._push(gs)
                    wire = bytearray(self.rec(f + 1))
                else:
                    if f == self.moment:
                        # the bot's previous pad: the one really pressed on this frame
                        cs = gs.players[op + 1].controller_state
                        ctrl = bot["build"](gs, self.prev_from(cs), bot["ctx"])["self_controller"][0].numpy()
                        bot["pol"].last_idx[0] = torch.tensor(ctrl_indices(ctrl))
                    snaps.append(self._snap(gs))
                    # my side's window (no decision: the player decides)
                    fr = mine["build"](gs, self.prev_from(gs.players[mp + 1].controller_state), mine["ctx"])
                    for k, v in fr.items():
                        b = mine["pol"].buf[k]
                        b[:, :-1] = b[:, 1:].clone()
                        b[:, -1] = torch.as_tensor(v[0]).to(b.dtype)
                    if f == self.moment + self.stretch and self.value is not None:
                        with torch.no_grad():
                            vfeat = mine["model"].features(mine["pol"].buf)[:, -1].float()
                    wire = bytearray(b"".join(self.Pad().pack() for _ in range(4)))
                    if op in self.scripts:
                        wire[8 * op:8 * op + 8] = self.pad_from_pre(self.scripts[op][(f + 1, op, 0)]).pack()
                    else:
                        t = time.perf_counter()
                        feats = {k: v[0].numpy()[None] for k, v in bot["build"](gs, None, bot["ctx"]).items()
                                 if k != "self_controller"}
                        idx = bot["pol"].step(feats)
                        self.timing["bot_ms"].append(1000 * (time.perf_counter() - t))
                        wire[8 * op:8 * op + 8] = pads_from_indices(idx)[0].tobytes()
                    if mp in self.scripts:
                        p = pad_from_json(json_from_pad(self.pad_from_pre(self.scripts[mp][(f + 1, mp, 0)])))
                    else:
                        p = self.pad or self.Pad()
                        if self.latch:
                            import dataclasses
                            p = dataclasses.replace(p, button=p.button | self.latch)
                            self.latch = 0
                    wire[8 * mp:8 * mp + 8] = p.pack()
                fr = self.sim.step(bytes(wire))
                if fr is None:
                    break
                self.cur = fr
                gs = self.parser.parse(fr)
                st = gs.players
                if self.cur.index >= self.moment and (st[mp + 1].stock < snaps[0][mp][1] if snaps else False):
                    break   # a stock lost ends it
                # the clock: one frame per period, never bunching up after a slow tick
                self.timing["ticks"] += 1
                t_next += 1.0 / (60.0 * self.speed)
                now = time.perf_counter()
                if now > t_next:
                    late = now - t_next
                    self.timing["late"] += late > 0.004
                    self.timing["max_ms"] = max(self.timing["max_ms"], 1000 * late)
                    t_next = now
                else:
                    # sleep overshoots by milliseconds on a busy machine: sleep most of the wait,
                    # then spin the last 2 ms
                    if t_next - now > 0.002:
                        time.sleep(t_next - now - 0.002)
                    while time.perf_counter() < t_next:
                        pass
            n0 = len(self.writer.raw)
            self.writer.add(self.cur)
            self._emit("f", bytes(self.writer.raw[n0:]))
            snaps.append(self._snap(gs))
            self.sim.close()
            self._finish(snaps, vfeat)
        except Exception as e:
            self._set("error", error=str(e)[:300])
            raise

    def _snap(self, gs):
        return {p: (float(gs.players[p + 1].percent), int(gs.players[p + 1].stock))
                for p in (self.me[0], self.opp[0])}

    def _finish(self, snaps, vfeat):
        mp, op = self.me[0], self.opp[0]
        seg = snaps[:self.stretch + 1]
        dmg = lambda p: float(sum(max(0.0, b[p][0] - a[p][0]) for a, b in zip(seg, seg[1:])))
        stk = lambda p: float(sum(max(0, a[p][1] - b[p][1]) for a, b in zip(seg, seg[1:])))
        realized = dmg(op) - dmg(mp) + 40 * (stk(op) - stk(mp))
        expected = None
        if vfeat is not None:
            from value import score
            expected = round(realized + float(score(self.value(vfeat))[0]), 1)
        whole = lambda p: float(sum(max(0.0, b[p][0] - a[p][0]) for a, b in zip(snaps, snaps[1:])))
        path = None
        if self.out_dir:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            path = self.out_dir / "attempt.slp"
            import datetime as _dt
            self.writer.write(str(path), metadata=dict(
                startAt=_dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                lastFrame=int(self.cur.index), playedOn="mimic-coach"))
        bot = sorted(self.timing["bot_ms"])
        self.result = dict(expected=expected, realized=round(realized, 1), frames=len(snaps),
                           dealt=round(whole(op), 1), taken=round(whole(mp), 1),
                           complete=len(seg) > self.stretch, slp=str(path) if path else None,
                           timing=dict(ticks=self.timing["ticks"], late=self.timing["late"],
                                       max_late_ms=round(self.timing["max_ms"], 1),
                                       bot_ms_p50=round(bot[len(bot) // 2], 1) if bot else None,
                                       bot_ms_p90=round(bot[int(0.9 * len(bot))], 1) if bot else None))
        self._set("done")
