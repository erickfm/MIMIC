"""Native (decomp) Melee as an RL environment: one headless game per OS process.

The game (built by tools/build_native_sim.sh) runs as a separate process in `--env` mode: it enters
a live match straight from a template .slp's game info (characters, costumes, stage, rules) and
gameplay-code list, with no menus, and takes every pad from us. After each engine frame it sends
that frame's Slippi events (frame start, pre-frame and post-frame per player), the same records a
.slp holds, then blocks for the next pads. Pads sent after observing frame f land on frame f+1,
the same as Dolphin + libmelee with blocking input (measured) and the BC shard alignment
(target[i] = buttons[i+1]).

One game per process is a hard constraint, not a choice: the game maps console RAM at the fixed
address 0x80000000 and keeps all state in globals.

Pads are raw controller values, exactly what PADRead returns on hardware (and what Slippi records
as the pre-frame "raw" fields): the game does its own clamping and deadzones.

    sim = MeleeSim("replays/some_1v1.slp")
    frame = sim.reset()
    while not sim.done:
        frame = sim.step([Pad(), Pad(), Pad(), Pad()])
"""
import os
import select
import struct
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_EXE = REPO / "sim/melee-unlocked/build-linux/port/linux/melee_linux"
DEFAULT_ISO = REPO / "melee.iso"

FRAME_START, PRE_FRAME, POST_FRAME, FRAME_BOOKEND = 0x3A, 0x37, 0x38, 0x3C
KIND_FRAME, KIND_DONE = 1, 2

# PADStatus button bits
BTN = {"DL": 0x0001, "DR": 0x0002, "DD": 0x0004, "DU": 0x0008, "Z": 0x0010, "R": 0x0020,
       "L": 0x0040, "A": 0x0100, "B": 0x0200, "X": 0x0400, "Y": 0x0800, "START": 0x1000}


@dataclass
class Pad:
    """One raw controller state. Sticks are signed (-128..127, 0 = neutral), triggers 0..255."""
    button: int = 0
    stick_x: int = 0
    stick_y: int = 0
    sub_x: int = 0
    sub_y: int = 0
    trigger_l: int = 0
    trigger_r: int = 0

    def pack(self):
        return struct.pack("<Hbbbbbb", self.button, self.stick_x, self.stick_y, self.sub_x,
                           self.sub_y, self.trigger_l - 256 if self.trigger_l > 127 else self.trigger_l,
                           self.trigger_r - 256 if self.trigger_r > 127 else self.trigger_r)


@dataclass
class Frame:
    """One engine frame's Slippi events. Payloads exclude the command byte (offsets = spec - 1)."""
    index: int
    pre: dict = field(default_factory=dict)    # (port, follower) -> pre-frame payload
    post: dict = field(default_factory=dict)   # (port, follower) -> post-frame payload
    events: list = field(default_factory=list)  # [(command, payload)] in emission order


def parse_events(blob):
    out, i = [], 0
    while i + 5 <= len(blob):
        c = blob[i]
        n = struct.unpack_from("<I", blob, i + 1)[0]
        out.append((c, blob[i + 5:i + 5 + n]))
        i += 5 + n
    return out


def to_frames(events):
    frames, cur = [], None
    for c, p in events:
        if c == FRAME_START:
            cur = Frame(struct.unpack_from(">i", p)[0])
        if cur is None:
            continue
        cur.events.append((c, p))
        if c == PRE_FRAME:
            cur.pre[(p[4], p[5])] = p
        elif c == POST_FRAME:
            cur.post[(p[4], p[5])] = p
        elif c == FRAME_BOOKEND:   # the previous frame's; it trails into the next message
            frames.append(cur)
            cur = None
    if cur is not None:
        frames.append(cur)
    return frames


class MeleeSim:
    def __init__(self, template_slp, exe=DEFAULT_EXE, iso=DEFAULT_ISO, seed=None, rng="offline",
                 audio=False, render=None, log_path=None, timeout=30.0):
        self.template = str(Path(template_slp).resolve())
        self.exe = Path(exe).resolve()
        self.iso = str(Path(iso).resolve())
        # rng="offline": Melee's own RNG chain, set to `seed` at frame -123 (None: the setup's).
        # rng="online": Slippi netplay's per-frame seed ((frame+123) << 16 | seed & 0xFFFF), as
        # in the online games the bot plays.
        if rng not in ("offline", "online"):
            raise ValueError(f"rng must be 'offline' or 'online', not {rng!r}")
        self.seed = seed
        self.rng = rng
        # Speed: audio mixing never touches game state (exact without it, ~3x faster). Skipping the
        # render pass is exact under online RNG (overwritten every frame) but shifts the offline
        # RNG stream (some effects draw random numbers while rendering), so by default it is
        # skipped only for rng="online" (~8.5x faster with audio off). Measured 2026-10-20.
        self.audio = audio
        self.render = (rng != "online") if render is None else render
        self.log_path = log_path
        self.timeout = timeout   # seconds a single read may wait before the sim counts as hung
        self.proc = None
        self.done = True
        self._tmp = None
        self.extra_frames = 0   # engine frames that ran without a pad read of their own

    def reset(self, initial_pads=None):
        """Start a fresh match; returns its first frame (index -123). `initial_pads` is the
        controller state going into the match (neutral by default): frame -123 runs on it."""
        self.close()
        self._tmp = tempfile.TemporaryDirectory(prefix="meleesim_")
        to_host_r, to_host_w = os.pipe()
        from_host_r, from_host_w = os.pipe()
        cmd = [str(self.exe), "--iso", self.iso, "--headless", "--fast",
               "--card-dir", os.path.join(self._tmp.name, "card"),
               "--env-template", self.template, "--env-fds", f"{to_host_r},{from_host_w}"]
        if self.seed is not None:
            cmd += ["--env-seed", str(self.seed)]
        if self.rng == "online":
            cmd += ["--env-online-rng"]
        if not self.audio:
            cmd += ["--no-audio"]
        if not self.render:
            cmd += ["--no-render"]
        if self.log_path:
            cmd += ["--log-file", str(self.log_path)]
        # cwd: the checkout root, where the host finds port/slippi_sys_playback.
        self.proc = subprocess.Popen(cmd, cwd=self.exe.parents[3], pass_fds=(to_host_r, from_host_w),
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.close(to_host_r)
        os.close(from_host_w)
        self._w = os.fdopen(to_host_w, "wb", buffering=0)
        self._r = os.fdopen(from_host_r, "rb", buffering=0)
        self.done = False
        self._w.write(b"".join(p.pack() for p in (initial_pads or [Pad()] * 4)))
        return self._recv()

    def step(self, pads):
        """Send the pads for the next frame (4 Pads, ports 1-4); returns that frame."""
        self.send(pads)
        return self.recv()

    # step() split in two, so one controller can send to many sims before collecting any: the
    # games then advance in parallel.
    def send(self, pads):
        if self.done:
            raise RuntimeError("step() after the match ended; call reset()")
        self._w.write(pads if isinstance(pads, (bytes, bytearray)) else b"".join(p.pack() for p in pads))

    def recv(self):
        return self._recv()

    def _read(self, n):
        buf = bytearray()
        while len(buf) < n:
            if not select.select([self._r], [], [], self.timeout)[0]:
                raise TimeoutError(f"sim sent nothing for {self.timeout}s (pid {self.proc.pid})")
            chunk = self._r.read(n - len(buf))
            if not chunk:
                raise EOFError(f"sim process exited (rc={self.proc.poll()})")
            buf += chunk
        return bytes(buf)

    def _recv(self):
        """The newest finished frame. Normally one per message; when the game ran more than one
        engine frame between pad reads, the earlier ones are kept in self.last_frames and counted
        in self.extra_frames (those frames reused the previous pads)."""
        kind, size = struct.unpack("<II", self._read(8))
        frames = to_frames(parse_events(self._read(size)))
        if kind == KIND_DONE:
            self.done = True
        elif not frames:
            raise RuntimeError("message without a finished frame")
        self.extra_frames += max(0, len(frames) - 1)
        self.last_frames = frames
        return frames[-1] if frames else None

    def close(self):
        if self.proc is not None:
            for f in (getattr(self, "_w", None), getattr(self, "_r", None)):
                try:
                    f and f.close()
                except OSError:
                    pass
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
            self.proc = None
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None
        self.done = True

    def __del__(self):
        self.close()
