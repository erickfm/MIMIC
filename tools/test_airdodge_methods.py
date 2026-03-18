#!/usr/bin/env python3
"""
test_airdodge_methods.py -- Try every L/R press method to find what triggers airdodge.

For each jump cycle, tries a different method of pressing L/R on the first
airborne frame. Reports which method (if any) produces AIRDODGE action state.

Also tests whether L/R can trigger shield while grounded.
"""

import csv
import sys
import time
from pathlib import Path

import melee

DOLPHIN_PATH = "/home/erick/.config/Slippi Launcher/netplay/Slippi_Online-x86_64.AppImage"
ISO_PATH = "/home/erick/Downloads/Super Smash Bros. Melee (USA) (En,Ja) (Rev 2).iso"

FALCO_JUMPSQUAT = 5
WD_RIGHT = (0.85, 0.23)

METHODS = [
    "digital_L",
    "analog_L_1.0",
    "analog_L_0.8",
    "both_L",
    "digital_R",
    "analog_R_1.0",
    "both_R",
    "both_L_no_release",
    "analog_L_raw_pipe",
    "shield_ground_L",
    "shield_ground_R",
    "shield_ground_analog_L",
]


def apply_method(ctrl, method, skip_release=False):
    """Apply a specific L/R press method. Returns description."""
    sx, sy = WD_RIGHT

    if not skip_release and "no_release" not in method:
        ctrl.release_all()

    ctrl.tilt_analog(melee.enums.Button.BUTTON_MAIN, sx, sy)

    if method == "digital_L":
        ctrl.press_button(melee.enums.Button.BUTTON_L)
        return "press_button(L)"

    elif method == "analog_L_1.0":
        ctrl.press_shoulder(melee.enums.Button.BUTTON_L, 1.0)
        return "press_shoulder(L, 1.0)"

    elif method == "analog_L_0.8":
        ctrl.press_shoulder(melee.enums.Button.BUTTON_L, 0.8)
        return "press_shoulder(L, 0.8)"

    elif method == "both_L":
        ctrl.press_button(melee.enums.Button.BUTTON_L)
        ctrl.press_shoulder(melee.enums.Button.BUTTON_L, 1.0)
        return "press_button(L) + press_shoulder(L, 1.0)"

    elif method == "digital_R":
        ctrl.press_button(melee.enums.Button.BUTTON_R)
        return "press_button(R)"

    elif method == "analog_R_1.0":
        ctrl.press_shoulder(melee.enums.Button.BUTTON_R, 1.0)
        return "press_shoulder(R, 1.0)"

    elif method == "both_R":
        ctrl.press_button(melee.enums.Button.BUTTON_R)
        ctrl.press_shoulder(melee.enums.Button.BUTTON_R, 1.0)
        return "press_button(R) + press_shoulder(R, 1.0)"

    elif method == "both_L_no_release":
        ctrl.press_button(melee.enums.Button.BUTTON_L)
        ctrl.press_shoulder(melee.enums.Button.BUTTON_L, 1.0)
        ctrl.tilt_analog(melee.enums.Button.BUTTON_MAIN, sx, sy)
        return "NO release_all + press_button(L) + press_shoulder(L, 1.0)"

    elif method == "analog_L_raw_pipe":
        if hasattr(ctrl, 'pipe') and ctrl.pipe:
            ctrl.pipe.write(f"SET MAIN {sx} {sy}\n")
            ctrl.pipe.write("PRESS L\n")
            ctrl.pipe.write("SET L 1.0\n")
        return "raw pipe: PRESS L + SET L 1.0"

    elif method == "shield_ground_L":
        ctrl.tilt_analog(melee.enums.Button.BUTTON_MAIN, 0.5, 0.5)
        ctrl.press_button(melee.enums.Button.BUTTON_L)
        ctrl.press_shoulder(melee.enums.Button.BUTTON_L, 1.0)
        return "GROUND shield: press_button(L) + press_shoulder(L, 1.0)"

    elif method == "shield_ground_R":
        ctrl.tilt_analog(melee.enums.Button.BUTTON_MAIN, 0.5, 0.5)
        ctrl.press_button(melee.enums.Button.BUTTON_R)
        ctrl.press_shoulder(melee.enums.Button.BUTTON_R, 1.0)
        return "GROUND shield: press_button(R) + press_shoulder(R, 1.0)"

    elif method == "shield_ground_analog_L":
        ctrl.tilt_analog(melee.enums.Button.BUTTON_MAIN, 0.5, 0.5)
        ctrl.press_shoulder(melee.enums.Button.BUTTON_L, 1.0)
        return "GROUND shield: press_shoulder(L, 1.0) only"

    return f"unknown: {method}"


def log(msg):
    print(msg, flush=True)


def main():
    console = melee.Console(
        path=DOLPHIN_PATH,
        slippi_address="127.0.0.1",
        fullscreen=False,
    )
    ctrl_p1 = melee.Controller(console, 1, fix_analog_inputs=False)
    ctrl_p2 = melee.Controller(console, 4)

    log("Starting Dolphin...")
    console.run(iso_path=ISO_PATH)

    log("Connecting...")
    if not console.connect():
        log("ERROR: Console connect failed"); sys.exit(1)
    if not ctrl_p1.connect():
        log("ERROR: P1 connect failed"); sys.exit(1)
    if not ctrl_p2.connect():
        log("ERROR: P2 connect failed"); sys.exit(1)
    log("All connected.")

    menu_p1 = melee.MenuHelper()
    menu_p2 = melee.MenuHelper()

    method_idx = 0
    phase = "PRESS_JUMP"
    frames_in_phase = 0
    was_in_game = False
    game_frames = 0
    saw_airdodge = False
    saw_landing_special = False
    current_method = METHODS[0]
    results = []

    # Ground shield tests first
    ground_tests = ["shield_ground_L", "shield_ground_R", "shield_ground_analog_L"]
    air_tests = [m for m in METHODS if m not in ground_tests]
    all_tests = ground_tests + air_tests
    test_idx = 0

    while test_idx < len(all_tests):
        gs = console.step()
        if gs is None:
            continue

        if gs.menu_state not in (melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH):
            if was_in_game:
                break
            menu_p1.menu_helper_simple(
                gs, ctrl_p1, melee.Character.FALCO, melee.Stage.FINAL_DESTINATION,
                cpu_level=0, autostart=False,
            )
            menu_p2.menu_helper_simple(
                gs, ctrl_p2, melee.Character.FALCO, melee.Stage.FINAL_DESTINATION,
                cpu_level=0, autostart=True,
            )
            continue

        was_in_game = True
        if gs.frame < 0:
            ctrl_p1.release_all(); ctrl_p1.flush()
            ctrl_p2.release_all(); ctrl_p2.flush()
            continue

        game_frames += 1

        ps1 = None
        for port, ps in gs.players.items():
            if port == 1:
                ps1 = ps
                break
        if ps1 is None:
            for port, ps in gs.players.items():
                ps1 = ps
                break

        action = ps1.action
        af = ps1.action_frame
        current_test = all_tests[test_idx]
        is_ground_test = current_test.startswith("shield_ground")

        # ── Ground shield tests ──
        if is_ground_test:
            if phase == "PRESS_JUMP":
                # Actually test shield, not jump
                ctrl_p1.release_all()
                if action in (melee.Action.STANDING, melee.Action.TURNING):
                    desc = apply_method(ctrl_p1, current_test)
                    phase = "WAIT_SHIELD"
                    frames_in_phase = 0
                    log(f"\n[Test {test_idx}] {current_test}: {desc}")
                ctrl_p1.flush()
                ctrl_p2.release_all(); ctrl_p2.flush()
                continue

            if phase == "WAIT_SHIELD":
                frames_in_phase += 1
                shield_actions = {
                    melee.Action.SHIELD, melee.Action.SHIELD_START,
                    melee.Action.SHIELD_REFLECT, melee.Action.SHIELD_STUN,
                }
                got_shield = action in shield_actions or "SHIELD" in action.name

                if frames_in_phase <= 5:
                    apply_method(ctrl_p1, current_test)
                    ctrl_p1.flush()
                    ctrl_p2.release_all(); ctrl_p2.flush()
                    log(f"  frame {frames_in_phase}: {action.name} (shield={'YES' if got_shield else 'no'})")
                    continue

                result = "SHIELD" if got_shield else "NO_SHIELD"
                log(f"  → Result: {result} (last action: {action.name})")
                results.append((current_test, result))

                ctrl_p1.release_all(); ctrl_p1.flush()
                # Wait a few frames to recover
                for _ in range(30):
                    gs2 = console.step()
                    if gs2: 
                        ctrl_p1.release_all(); ctrl_p1.flush()
                        ctrl_p2.release_all(); ctrl_p2.flush()

                test_idx += 1
                phase = "PRESS_JUMP"
                continue

        # ── Air airdodge tests ──
        else:
            if phase == "PRESS_JUMP":
                ctrl_p1.release_all()
                if action in (melee.Action.STANDING, melee.Action.TURNING,
                              melee.Action.CROUCHING, melee.Action.CROUCH_START):
                    ctrl_p1.press_button(melee.enums.Button.BUTTON_Y)
                    phase = "RELEASE_Y"
                ctrl_p1.flush()
                ctrl_p2.release_all(); ctrl_p2.flush()
                continue

            if phase == "RELEASE_Y":
                ctrl_p1.release_all()
                ctrl_p1.flush()
                ctrl_p2.release_all(); ctrl_p2.flush()
                if action == melee.Action.KNEE_BEND:
                    phase = "JUMPSQUAT"
                    log(f"\n[Test {test_idx}] {current_test}: in jumpsquat")
                continue

            if phase == "JUMPSQUAT":
                ctrl_p1.release_all()
                ctrl_p1.flush()
                ctrl_p2.release_all(); ctrl_p2.flush()
                if action != melee.Action.KNEE_BEND:
                    phase = "AIRBORNE"
                    frames_in_phase = 0
                    saw_airdodge = False
                    saw_landing_special = False
                    log(f"  Airborne! action={action.name} → applying {current_test}")
                continue

            if phase == "AIRBORNE":
                frames_in_phase += 1

                if action == melee.Action.AIRDODGE:
                    saw_airdodge = True
                if action == melee.Action.LANDING_SPECIAL:
                    saw_landing_special = True

                if frames_in_phase <= 5:
                    desc = apply_method(ctrl_p1, current_test)
                    ctrl_p1.flush()
                    ctrl_p2.release_all(); ctrl_p2.flush()
                    log(f"  f+{frames_in_phase}: {action.name} af={af} (applied: {desc})")
                    continue

                # After 5 frames of input, just wait for landing
                ctrl_p1.release_all()
                ctrl_p1.flush()
                ctrl_p2.release_all(); ctrl_p2.flush()

                actionable = {melee.Action.STANDING, melee.Action.TURNING,
                              melee.Action.CROUCHING, melee.Action.LANDING,
                              melee.Action.LANDING_SPECIAL}
                if frames_in_phase > 60 or (action in actionable and frames_in_phase > 10):
                    if saw_airdodge:
                        result = "AIRDODGE"
                    elif saw_landing_special:
                        result = "LANDING_SPECIAL"
                    else:
                        result = "NO_AIRDODGE"
                    log(f"  → Result: {result} (airdodge={saw_airdodge}, land_special={saw_landing_special})")
                    results.append((current_test, result))

                    # Wait for standing
                    for _ in range(60):
                        gs2 = console.step()
                        if gs2:
                            ctrl_p1.release_all(); ctrl_p1.flush()
                            ctrl_p2.release_all(); ctrl_p2.flush()

                    test_idx += 1
                    phase = "PRESS_JUMP"
                    continue

                if frames_in_phase % 10 == 0:
                    log(f"  f+{frames_in_phase}: {action.name} pos=({ps1.position.x:.1f},{ps1.position.y:.1f})")
                continue

    # Print summary
    log("\n" + "=" * 60)
    log("RESULTS SUMMARY")
    log("=" * 60)
    for method, result in results:
        status = "✓" if result in ("AIRDODGE", "LANDING_SPECIAL", "SHIELD") else "✗"
        log(f"  {status} {method:30s} → {result}")

    ctrl_p1.disconnect()
    ctrl_p2.disconnect()
    console.stop()


if __name__ == "__main__":
    main()
