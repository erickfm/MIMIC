"""Raw controller state from a pre-frame payload, for any Slippi version.

Older replays (Slippi < 3.15, e.g. the ranked dataset's 3.14/3.15 games) lack the raw stick-y
and raw C-stick bytes that tools/sim_env_check.pad_from_pre reads. Those fall back to the
processed stick values times 80 (the game's raw-to-processed scale; inside the deadzone the
processed value is 0 and so is the fallback). This is approximate: the game's processing is not
exactly invertible, so a replay fed this way can drift from the recording.
"""
import struct

from rlvr.sim.melee_sim import Pad


def _s8(p, o):
    return struct.unpack_from("b", p, o)[0]


def _f(p, o):
    return struct.unpack_from(">f", p, o)[0]


def pad_from_pre(p):
    """Payload offsets = Slippi spec offset - 1 (no command byte)."""
    def raw_or(o_raw, o_proc):
        if len(p) > o_raw:
            return _s8(p, o_raw)
        return max(-128, min(127, round(_f(p, o_proc) * 80)))
    trig = lambda o: min(255, round(_f(p, o) * 140))
    return Pad(button=struct.unpack_from(">H", p, 48)[0],
               stick_x=raw_or(58, 24), stick_y=raw_or(63, 28),
               sub_x=raw_or(64, 32), sub_y=raw_or(65, 36),
               trigger_l=trig(50), trigger_r=trig(54))
