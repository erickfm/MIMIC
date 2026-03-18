#!/usr/bin/env python3
"""
generate_wavedash_replay.py -- Programmatic wavedash replay generator.

Launches Dolphin via libmelee with two Falcos on FD.
P1 wavedashes back and forth continuously.
P2 stands still (no inputs).
Slippi auto-saves the replay. A per-frame CSV log is written for human review.

Usage:
    python3 generate_wavedash_replay.py [--duration 7200] [--log-dir logs/]
"""

import argparse
import csv
import os
import random
import signal
import sys
import time
from pathlib import Path

import melee
import pandas as pd

sys.path.insert(0, "/home/erick/projects/slippi-frame-extractor")
from extract import (extract_player, extract_projectiles, extract_stage_static,
                     preseed_nana, perspective)

DOLPHIN_PATH = "/home/erick/.config/Slippi Launcher/netplay/Slippi_Online-x86_64.AppImage"
ISO_PATH = "/home/erick/Downloads/Super Smash Bros. Melee (USA) (En,Ja) (Rev 2).iso"

FALCO_JUMPSQUAT = 5

# Stick positions for wavedash angles (nearly horizontal for max distance)
# x: 0.15 = far left, 0.85 = far right; y: 0.25 = down
WD_RIGHT = (0.85, 0.23)
WD_LEFT  = (0.15, 0.23)

ACTIONABLE_STATES = frozenset({
    melee.Action.STANDING,        # 14
    melee.Action.TURNING,         # 18
    melee.Action.DASHING,         # 20
    melee.Action.RUNNING,         # 21
    melee.Action.RUN_BRAKE,       # 23
    melee.Action.CROUCH_START,    # 39
    melee.Action.CROUCHING,       # 40
    # LANDING (42) and LANDING_SPECIAL (43) intentionally excluded:
    # can't act during landing lag, pressing Y here is wasted.
})

# FD stage boundaries: ledges at ~±85.6, keep a safe margin
EDGE_THRESHOLD = 60.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=int, default=28800,
                   help="Number of in-game frames to run (default 28800 = ~8 min)")
    p.add_argument("--log-dir", type=str, default="logs/")
    p.add_argument("--dolphin-path", type=str, default=DOLPHIN_PATH)
    p.add_argument("--iso-path", type=str, default=ISO_PATH)
    p.add_argument("--parquet-dir", type=str, default="data/wavedash",
                   help="Output dir for parquet files (same schema as extract.py)")
    return p.parse_args()


class WavedashBot:
    """State machine that wavedashes back and forth with variable idle periods.

    Phases:
        IDLE       – stand still for a random number of frames
        RELEASE    – 1 frame with no buttons so next Y press is a fresh button-down
        PRESS_JUMP – press Y for 1 frame to initiate jump
        JUMPSQUAT  – in KNEE_BEND, wait (no buttons at all)
        AIRDODGE   – first frame after KNEE_BEND ends: fresh L press + stick
        WAIT_LAND  – airborne / sliding, wait for actionable state
    """

    def __init__(self):
        self.direction = "right"
        self.wd_count = 0
        self.phase = "IDLE"
        self.idle_frames_left = random.randint(30, 120)
        self.wd_since_idle = 0
        self.next_idle_at = random.randint(3, 7)

    def _safe_direction(self, pos_x: float) -> str:
        """Force direction away from the nearest edge if too close."""
        if pos_x > EDGE_THRESHOLD:
            return "left"
        if pos_x < -EDGE_THRESHOLD:
            return "right"
        return self.direction

    def act(self, ctrl: melee.Controller, ps) -> str:
        action = ps.action
        af = ps.action_frame
        pos_x = float(ps.position.x)

        ctrl.release_all()

        # ── IDLE: stand still for variable duration ──
        if self.phase == "IDLE":
            self.idle_frames_left -= 1
            if self.idle_frames_left <= 0:
                self.direction = self._safe_direction(pos_x)
                self.phase = "PRESS_JUMP"
                return "idle done → press_jump"
            return f"idle ({self.idle_frames_left} left)"

        # ── RELEASE: 1 empty frame so the next Y press is a new button-down ──
        if self.phase == "RELEASE":
            self.direction = self._safe_direction(pos_x)
            self.phase = "PRESS_JUMP"
            return "release (gap before Y)"

        # ── PRESS_JUMP: send Y on exactly one frame ──
        if self.phase == "PRESS_JUMP":
            if action in ACTIONABLE_STATES:
                ctrl.press_button(melee.enums.Button.BUTTON_Y)
                self.phase = "WAIT_JUMPSQUAT"
                return f"press_Y (wd #{self.wd_count}, dir={self.direction})"
            return f"wait_actionable ({action.name} f={af})"

        # ── WAIT_JUMPSQUAT: Y was just pressed, waiting for KNEE_BEND ──
        if self.phase == "WAIT_JUMPSQUAT":
            if action == melee.Action.KNEE_BEND:
                self.phase = "JUMPSQUAT"
                return f"jumpsquat f={af}"
            if action in ACTIONABLE_STATES:
                return f"wait_for_jumpsquat ({action.name} f={af})"
            return f"transition ({action.name} f={af})"

        # ── JUMPSQUAT: in KNEE_BEND, do NOTHING (no L press!) ──
        if self.phase == "JUMPSQUAT":
            if action == melee.Action.KNEE_BEND:
                return f"jumpsquat f={af}"
            sx, sy = WD_RIGHT if self.direction == "right" else WD_LEFT
            ctrl.tilt_analog(melee.enums.Button.BUTTON_MAIN, sx, sy)
            ctrl.press_button(melee.enums.Button.BUTTON_L)
            self.phase = "AIRDODGE_SENT"
            self.airdodge_frames = 0
            return f"airdodge_{self.direction} stick=({sx},{sy})"

        # ── AIRDODGE_SENT: keep pressing L+stick for a few frames ──
        if self.phase == "AIRDODGE_SENT":
            self.airdodge_frames += 1
            if action == melee.Action.AIRDODGE:
                self.phase = "WAIT_LAND"
                return f"airdodge_active f={af}"
            if action == melee.Action.LANDING_SPECIAL:
                self.phase = "WAIT_LAND"
                return f"wavedash_slide f={af}"
            if self.airdodge_frames < 4:
                sx, sy = WD_RIGHT if self.direction == "right" else WD_LEFT
                ctrl.tilt_analog(melee.enums.Button.BUTTON_MAIN, sx, sy)
                ctrl.press_button(melee.enums.Button.BUTTON_L)
                return f"airdodge_hold f={self.airdodge_frames} ({action.name})"
            self.phase = "WAIT_LAND"
            return f"airdodge_timeout ({action.name} f={af})"

        # ── WAIT_LAND: airborne or sliding, wait until actionable again ──
        if self.phase == "WAIT_LAND":
            if action in ACTIONABLE_STATES:
                self.direction = "left" if self.direction == "right" else "right"
                self.direction = self._safe_direction(pos_x)
                self.wd_count += 1
                self.wd_since_idle += 1
                if self.wd_since_idle >= self.next_idle_at:
                    self.phase = "IDLE"
                    self.idle_frames_left = random.randint(20, 80)
                    self.wd_since_idle = 0
                    self.next_idle_at = random.randint(3, 7)
                    return f"wd #{self.wd_count} done → idle ({self.idle_frames_left}f)"
                self.phase = "RELEASE"
                return f"wd #{self.wd_count} done → {self.direction}"
            return f"airborne ({action.name} f={af})"

        return f"unknown phase={self.phase} ({action.name} f={af})"


def log(msg):
    print(msg, flush=True)


def main():
    args = parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = log_dir / f"wavedash_gen_{stamp}.csv"

    log(f"Wavedash Replay Generator")
    log(f"  Duration: {args.duration} frames (~{args.duration/60:.0f}s)")
    log(f"  Log: {csv_path}")
    log(f"  Dolphin: {args.dolphin_path}")

    console = melee.Console(
        path=args.dolphin_path,
        slippi_address="127.0.0.1",
        fullscreen=False,
    )
    ctrl_p1 = melee.Controller(console, 1, fix_analog_inputs=False)
    ctrl_p2 = melee.Controller(console, 4)

    log("Starting Dolphin...")
    console.run(iso_path=args.iso_path)

    log("Waiting for console connection...")
    if not console.connect():
        log("ERROR: Console connect failed"); sys.exit(1)
    log("Console connected. Connecting P1...")
    if not ctrl_p1.connect():
        log("ERROR: P1 controller connect failed"); sys.exit(1)
    log("P1 connected. Connecting P2...")
    if not ctrl_p2.connect():
        log("ERROR: P2 controller connect failed"); sys.exit(1)
    log("All connected.")

    menu_p1 = melee.MenuHelper()
    menu_p2 = melee.MenuHelper()

    bot = WavedashBot()

    csv_fh = open(csv_path, "w", newline="")
    writer = csv.writer(csv_fh)
    writer.writerow([
        "frame", "action", "action_value", "action_frame",
        "pos_x", "pos_y", "on_ground", "facing",
        "bot_description", "direction", "wd_count",
        "stick_x", "stick_y", "buttons",
    ])

    parquet_dir = Path(args.parquet_dir)
    parquet_dir.mkdir(parents=True, exist_ok=True)
    parquet_rows = []
    stage_static = None

    game_frames = 0
    was_in_game = False

    def save_parquet():
        if not parquet_rows:
            return
        df = pd.DataFrame(parquet_rows)
        for col in df.select_dtypes(include=["float64"]).columns:
            df[col] = df[col].astype("float32")
        df_p1 = perspective(df, "p1_", "p2_")
        out = parquet_dir / f"wavedash_gen_{stamp}-p1.parquet"
        df_p1.to_parquet(out, index=False)
        log(f"Saved {len(df)} frames → {out}")

    def shutdown():
        csv_fh.close()
        save_parquet()
        ctrl_p1.disconnect()
        ctrl_p2.disconnect()
        console.stop()
        log(f"\nDone. {game_frames} game frames, {bot.wd_count} wavedashes.")
        log(f"Log: {csv_path}")

    signal.signal(signal.SIGINT, lambda *_: (shutdown(), sys.exit(0)))

    while True:
        gs = console.step()
        if gs is None:
            continue

        if gs.menu_state not in (melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH):
            if was_in_game:
                log(f"Game ended at frame {game_frames}.")
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

        desc = bot.act(ctrl_p1, ps1)
        ctrl_p1.flush()

        ctrl_p2.release_all()
        ctrl_p2.flush()

        # ── Parquet extraction (same schema as slippi-frame-extractor) ──
        if stage_static is None:
            stage_static = extract_stage_static(gs.stage)
        pq_row = {
            "frame": gs.frame,
            "distance": gs.distance,
            "stage": gs.stage.value,
            **stage_static,
            "randall_height": float("nan"),
            "randall_left": float("nan"),
            "randall_right": float("nan"),
        }
        for idx, (port, ps) in enumerate(gs.players.items()):
            pref = f"p{idx+1}_"
            pq_row[f"{pref}port"] = port
            extract_player(pq_row, pref, ps)
        extract_projectiles(pq_row, gs.projectiles)
        preseed_nana(pq_row)
        parquet_rows.append(pq_row)

        stick = ctrl_p1.current.main_stick if hasattr(ctrl_p1, 'current') else (0.5, 0.5)
        btns = []
        if hasattr(ctrl_p1, 'current') and hasattr(ctrl_p1.current, 'button'):
            for btn, pressed in ctrl_p1.current.button.items():
                if pressed:
                    btns.append(btn.name)

        writer.writerow([
            gs.frame,
            ps1.action.name, ps1.action.value, ps1.action_frame,
            f"{ps1.position.x:.2f}", f"{ps1.position.y:.2f}",
            ps1.on_ground, ps1.facing,
            desc, bot.direction, bot.wd_count,
            f"{stick[0]:.3f}" if isinstance(stick, tuple) else "",
            f"{stick[1]:.3f}" if isinstance(stick, tuple) else "",
            " ".join(btns),
        ])

        if game_frames % 300 == 0:
            log(f"  frame {game_frames}: pos=({ps1.position.x:.1f},{ps1.position.y:.1f}) "
                f"action={ps1.action.name} wd_count={bot.wd_count} dir={bot.direction}")

        if game_frames >= args.duration:
            log(f"Reached {args.duration} frames. Stopping.")
            break

    shutdown()


if __name__ == "__main__":
    main()
