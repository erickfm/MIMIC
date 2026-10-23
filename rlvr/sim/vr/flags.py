"""Slippi state bit flags on libmelee PlayerStates.

slippistats decides hitstun and hitlag from the post-frame state bit flags (Field4.HITSTUN,
Field2.HITLAG). libmelee doesn't expose those: its `hitstun_frames_left` is the post-frame "misc"
float (0x2B), which only means hitstun-remaining while the character is in hitstun — other action
states reuse it (e.g. 31 during TECH_MISS_UP), so `> 0` is not a hitstun test. `install()` wraps
libmelee's post-frame parser to also set, on each PlayerState (and Nana):

  hitstun_flag   bool — state bit flags 4 (0x29) & 0x02   (slippistats Field4.HITSTUN)
  hitlag_flag    bool — state bit flags 2 (0x27) & 0x20   (slippistats Field2.HITLAG)

Offsets are the Slippi post-frame layout libmelee itself uses (it reads byte 0x29 for
is_powershield). Replays too old to carry the flags get None, and the predicates in stream.py
fall back to libmelee's fields. The patch is on the class, so it covers every Console, including
rlvr/sim/gamestate.SlippiParser's; import rlvr.sim.vr (which calls install()) before parsing.
"""
import melee
import numpy as np
from melee.gamestate import PlayerState

_ORIG = None


class FlagPlayerState(PlayerState):
    """libmelee's PlayerState (slotted) plus the two Slippi flag slots."""
    __slots__ = ("hitstun_flag", "hitlag_flag")


_SLOTS = [s for c in PlayerState.__mro__ for s in getattr(c, "__slots__", ())]


def _with_flags(ps, event_bytes):
    out = FlagPlayerState.__new__(FlagPlayerState)
    for s in _SLOTS:
        if hasattr(ps, s):
            setattr(out, s, getattr(ps, s))
    if len(event_bytes) > 0x29:
        out.hitlag_flag = bool(event_bytes[0x27] & 0x20)
        out.hitstun_flag = bool(event_bytes[0x29] & 0x02)
    else:
        out.hitlag_flag = out.hitstun_flag = None
    return out


def install():
    """Wrap libmelee's post-frame parser: each parsed PlayerState (and Nana) is replaced by a
    FlagPlayerState copy carrying hitstun_flag / hitlag_flag. libmelee builds fresh player objects
    every frame and nothing writes to them after the post-frame event, so the swap is safe."""
    global _ORIG
    if _ORIG is not None:
        return
    _ORIG = melee.Console._Console__post_frame

    def post_frame(self, gamestate, event_bytes):
        _ORIG(self, gamestate, event_bytes)
        port = int(np.ndarray((1,), ">B", event_bytes, 0x5)[0]) + 1
        ps = gamestate.players.get(port)
        if ps is None:
            return
        if int(np.ndarray((1,), ">B", event_bytes, 0x6)[0]) == 1:
            if ps.nana is not None:
                ps.nana = _with_flags(ps.nana, event_bytes)
        else:
            gamestate.players[port] = _with_flags(ps, event_bytes)

    melee.Console._Console__post_frame = post_frame
