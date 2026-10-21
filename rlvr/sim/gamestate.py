"""libmelee GameStates from the native sim's frames, parsed by libmelee itself.

The sim emits the same Slippi events Dolphin streams to libmelee, so the inference pipeline sees
exactly what it sees on Dolphin: SlippiParser runs libmelee's own Console event parser (its
pre/post-frame decoding, derived fields like off_stage / invulnerable, frame-index and IASA
fixups) over each sim frame. The Event Payloads and Game Start events, which the sim does not
re-emit in env mode, come from the template .slp the match was entered from.

SlpWriter writes the same events back out as a .slp, so sim games go through the existing replay
tools (canaries, validate_run.py, peppi).
"""
import struct

import melee
from melee.gamestate import GameState

from rlvr.sim.melee_sim import FRAME_BOOKEND

PAYLOADS, GAME_START, GAME_END = 0x35, 0x36, 0x39


def template_header(template_slp):
    """(Event Payloads event, Game Start event) of a .slp, each as command byte + payload."""
    from tools.sim_replay_check import slp_events   # noqa: WPS433 (tools is not a package dep elsewhere)
    d = open(template_slp, "rb").read()
    i = d.find(b"raw[$U#l") + 12
    n = d[i + 1]
    payloads = d[i:i + 1 + n]
    for c, p in slp_events(template_slp):
        if c == GAME_START:
            return payloads, bytes([c]) + p
    raise ValueError(f"{template_slp}: no Game Start event")


def event_sizes(payloads_event):
    """command -> payload size, from an Event Payloads event (command byte included)."""
    n = payloads_event[1]
    return {payloads_event[2 + 3 * k]: struct.unpack_from(">H", payloads_event, 3 + 3 * k)[0]
            for k in range((n - 1) // 3)}


def fit(payload, size):
    """A payload at the size the stream declares (zero-padded or cut)."""
    return payload[:size] if len(payload) >= size else payload + bytes(size - len(payload))


class SlippiParser:
    def __init__(self, template_slp):
        self.payloads, self.game_start = template_header(template_slp)
        self.sizes = event_sizes(self.payloads)
        # A Console that never connects: is_dolphin=False opens nothing until connect().
        self.console = melee.Console(is_dolphin=False, path=str(template_slp), blocking_input=False)
        self.console._events_this_frame = []   # normally set up by Console.step()
        gs = GameState()
        self.console._Console__handle_slippstream_events(self.payloads + self.game_start, gs)

    def parse(self, frame):
        """A melee.GameState for one sim Frame (rlvr.sim.melee_sim.Frame)."""
        c = self.console
        buf = bytearray()
        for cmd, p in frame.events:
            if cmd in self.sizes and cmd != FRAME_BOOKEND:
                buf.append(cmd)
                buf += fit(p, self.sizes[cmd])
        # The frame's own bookend arrives with the next message; libmelee returns a GameState at a
        # bookend, so give it one now (frame, latest finalized frame = frame).
        buf.append(FRAME_BOOKEND)
        buf += fit(struct.pack(">ii", frame.index, frame.index), self.sizes[FRAME_BOOKEND])
        if c._temp_gamestate is None:
            c._temp_gamestate = GameState()
            c._events_this_frame = []
        gs = c._temp_gamestate
        if not c._Console__handle_slippstream_events(bytes(buf), gs):
            raise RuntimeError(f"libmelee did not finish frame {frame.index}")
        c._temp_gamestate = None
        c._Console__fixframeindexing(gs)
        c._Console__fixiasa(gs)
        return gs


class SlpWriter:
    """Collects a sim match's events and writes them as a .slp."""

    def __init__(self, template_slp):
        self.payloads, self.game_start = template_header(template_slp)
        self.sizes = event_sizes(self.payloads)
        self.raw = bytearray(self.payloads + self.game_start)

    def add(self, frame):
        for cmd, p in frame.events:
            if cmd in self.sizes and cmd != FRAME_BOOKEND:
                self.raw.append(cmd)
                self.raw += fit(p, self.sizes[cmd])
        # The frame's own bookend arrives with the next sim message: write it here.
        self.raw.append(FRAME_BOOKEND)
        self.raw += fit(struct.pack(">ii", frame.index, frame.index), self.sizes[FRAME_BOOKEND])

    def write(self, path, end_method=2):
        """end_method 2 = game end (stocks/time); the sim does not emit Game End in env mode."""
        if GAME_END in self.sizes:
            self.raw.append(GAME_END)
            self.raw += fit(bytes([end_method, 0xFF]), self.sizes[GAME_END])
        out = (b"{U\x03raw[$U#l" + struct.pack(">I", len(self.raw)) + bytes(self.raw)
               + b"U\x08metadata{}" + b"}")
        with open(path, "wb") as f:
            f.write(out)
