"""Controller interventions for causal experiments in the native sim.

An intervention rewrites one player's controller input under a rule (force or forbid one
behaviour) and leaves everything else to the policy. Playing an intervened copy of a model against
the unmodified model measures the causal effect of that behaviour on winning, given otherwise
identical play (docs/research-notes-2026-10-23.md).

Rules run in the sim workers (rlvr/sim/vec_env.py) on the same frame the policy decided on: they
see that frame's libmelee GameState and mutate the 8-byte pad (rlvr/sim/melee_sim.Pad: buttons
u16 LE, main x/y, C x/y as int8, L/R analog triggers). The policy's own controller-history input
still holds what it chose, not the rewritten pad.
"""
from rlvr.sim.vr import stream as S

L_BIT, R_BIT, Z_BIT, A_BIT = 0x0040, 0x0020, 0x0010, 0x0100
AERIAL_ATTACKS = {65, 66, 67, 68, 69}          # NAIR, FAIR, BAIR, UAIR, DAIR
STICK_MAX = 80                                  # rlvr/sim/batch.py: (v - 0.5) * 160


def _bits(p):
    return p[0] | (p[1] << 8)


def _set_bits(p, b):
    p[0], p[1] = b & 0xFF, (b >> 8) & 0xFF


def _no_shoulders(p):
    _set_bits(p, _bits(p) & ~(L_BIT | R_BIT))
    p[6] = p[7] = 0


def _i8(v):
    return int(v) & 0xFF


class Intervention:
    id = ""

    def __init__(self, port: int):
        self.port = port            # libmelee 1-based port
        self.frames = 0             # frames this rule changed the pad
        self.reset()

    def reset(self):
        pass

    def apply(self, history, pad: bytearray) -> None:
        ps = S.get_player(history[-1], self.port)
        if ps is not None and self.rule(history, ps, pad):
            self.frames += 1

    def rule(self, history, ps, pad) -> bool:     # True if it changed the pad
        raise NotImplementedError


class LCancelPerfect(Intervention):
    """During an aerial attack in the air, the analog trigger toggles every frame, so a rising
    edge lands inside any 7-frame L-cancel window. Analog only: it cannot airdodge."""
    id = "lcancel_perfect"

    def reset(self):
        self._on = False

    def rule(self, history, ps, pad):
        if S.act(ps) in AERIAL_ATTACKS and not ps.on_ground:
            self._on = not self._on
            pad[6] = 255 if self._on else 0
            return True
        self._on = False
        return False


class LCancelNever(Intervention):
    """No shoulder or Z input during aerial attacks (Z also L-cancels)."""
    id = "lcancel_never"

    def rule(self, history, ps, pad):
        if S.act(ps) in AERIAL_ATTACKS and not ps.on_ground:
            _no_shoulders(pad)
            _set_bits(pad, _bits(pad) & ~Z_BIT)
            return True
        return False


def _victim_hitlag(history, port):
    """Being hit, this frame or last: the input that lands on the frame after hitlag ends is the
    one that sets DI, and this pad lands one frame after the state it was decided on."""
    for gs in list(history)[-2:]:
        ps = S.get_player(gs, port)
        if ps is not None and S.in_hitlag(ps) and (S.in_hitstun(ps) or S.is_damaged(S.act(ps))):
            return True
    return False


class DISurvival(Intervention):
    """While being hit (in hitlag as the victim), the stick points up and toward stage center."""
    id = "di_survival"

    def rule(self, history, ps, pad):
        if _victim_hitlag(history, self.port):
            x = -1 if float(ps.position.x) > 0 else 1
            pad[2], pad[3] = _i8(x * STICK_MAX * 0.707), _i8(STICK_MAX * 0.707)
            return True
        return False


class DINone(Intervention):
    """While being hit, the stick is neutral: no DI."""
    id = "di_none"

    def rule(self, history, ps, pad):
        if _victim_hitlag(history, self.port):
            pad[2] = pad[3] = 0
            return True
        return False


class NoOffstageAirdodge(Intervention):
    """No shoulder input while airborne off-stage (where an airdodge is the classic SD)."""
    id = "no_offstage_airdodge"

    def rule(self, history, ps, pad):
        if not ps.on_ground and S.is_offstage(ps, S.stage_name(history[-1])):
            if _bits(pad) & (L_BIT | R_BIT) or pad[6] or pad[7]:
                _no_shoulders(pad)
                return True
        return False


class NoShield(Intervention):
    """No shoulder input on the ground: never shields (or shield-grabs, rolls, spotdodges)."""
    id = "no_shield"

    def rule(self, history, ps, pad):
        if ps.on_ground and (_bits(pad) & (L_BIT | R_BIT) or pad[6] or pad[7]):
            _no_shoulders(pad)
            return True
        return False


RULES = {c.id: c for c in (LCancelPerfect, LCancelNever, DISurvival, DINone, NoOffstageAirdodge, NoShield)}
