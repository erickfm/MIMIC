"""A libmelee-Controller stand-in for the native sim: same calls, raw pad out.

tools/inference_utils.decode_and_press drives a melee.Controller (tilt_analog, press_shoulder,
press_button, release_button, flush). SimController takes the same calls and converts the
state to the raw pad Melee would receive from official Slippi Dolphin through libmelee's pipe:

  sticks   raw = round((v - 0.5) * 160)   (libmelee's fix_analog_stick: Python round, half-even)
  triggers raw = round(v * 140)           (libmelee's fix_analog_trigger)
  digital L/R also drives that trigger fully

All three measured on realtime Slippi Dolphin 2026-10-20: 65-point sweeps of every stick axis and
the L trigger, plus shoulder/digital combinations, read back from the replay's raw pre-frame fields
(docs/research-notes-2026-10-20b.md).
"""
from melee import enums

from rlvr.sim.melee_sim import BTN, Pad

_BUTTON_BITS = {
    enums.Button.BUTTON_A: BTN["A"], enums.Button.BUTTON_B: BTN["B"],
    enums.Button.BUTTON_X: BTN["X"], enums.Button.BUTTON_Y: BTN["Y"],
    enums.Button.BUTTON_Z: BTN["Z"], enums.Button.BUTTON_L: BTN["L"],
    enums.Button.BUTTON_R: BTN["R"], enums.Button.BUTTON_START: BTN["START"],
    enums.Button.BUTTON_D_UP: BTN["DU"], enums.Button.BUTTON_D_DOWN: BTN["DD"],
    enums.Button.BUTTON_D_LEFT: BTN["DL"], enums.Button.BUTTON_D_RIGHT: BTN["DR"],
}


def _clamp01(v):
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


def stick_raw(v):
    return round((_clamp01(v) - 0.5) * 160)


def trigger_raw(v):
    return round(_clamp01(v) * 140)


class SimController:
    def __init__(self):
        self.release_all()

    def release_all(self):
        self.buttons = set()
        self.main = (0.5, 0.5)
        self.c = (0.5, 0.5)
        self.l = 0.0
        self.r = 0.0

    def press_button(self, button):
        self.buttons.add(button)

    def release_button(self, button):
        self.buttons.discard(button)

    def tilt_analog(self, button, x, y):
        if button == enums.Button.BUTTON_MAIN:
            self.main = (x, y)
        elif button == enums.Button.BUTTON_C:
            self.c = (x, y)
        else:
            raise ValueError(f"Invalid button type {button} for tilt_analog.")

    def press_shoulder(self, button, amount):
        if button == enums.Button.BUTTON_L:
            self.l = amount
        elif button == enums.Button.BUTTON_R:
            self.r = amount

    def flush(self):
        pass   # the sim reads pad() once per step

    def pad(self):
        bits = 0
        for b in self.buttons:
            bits |= _BUTTON_BITS[b]
        return Pad(button=bits, stick_x=stick_raw(self.main[0]), stick_y=stick_raw(self.main[1]),
                   sub_x=stick_raw(self.c[0]), sub_y=stick_raw(self.c[1]),
                   trigger_l=255 if enums.Button.BUTTON_L in self.buttons else trigger_raw(self.l),
                   trigger_r=255 if enums.Button.BUTTON_R in self.buttons else trigger_raw(self.r))
