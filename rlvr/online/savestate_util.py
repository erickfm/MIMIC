"""Shared helpers for the miss-targeted savestate drilling harness.

Used by rlvr/online/miss_harvest.py (harvest savestates ~3 s before
detected L-cancel misses) and rlvr/online/drill_loop.py (LOADSTATE each
harvested state N times, score the first aerial landing live, PPO-update
on matched-context groups).

Savestate mechanism (validated 2026-07-15, docs/research-notes-2026-07-15.md):
  - `SAVESTATE <path>` / `LOADSTATE <path>` are pipe verbs on the bot
    controller pipe, patched into the emulator_ss/ Dolphin build.
  - Ops are queued as host jobs: they fire 1-150 game frames AFTER the
    pipe command (worse under FFW). Not frame-exact.
  - `melee.Console(skip_rollback_frames=False)` is REQUIRED to observe a
    LOADSTATE rewind — the default silently drops any frame <= max seen.
  - Never score .slp replays written across loads; live-state scoring only.
  - One Dolphin instance per slippi_port. This harness uses 52100-52199.

Live L-cancel scoring reimplements the avoidable-lag rule of
rlvr/online/tasks/l_cancel_online.py (whose constants we import) on
streaming gamestates instead of post-match .slp parses: count the
LANDING_AIR_* run length from consecutive frames; when the run exits
into a non-damage/non-dead state, avoidable = max(0, len - cancelled_min).
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional

import melee
import torch

from rlvr.online.dolphin_actor import ALL_ACTION_BUTTONS
from rlvr.online.tasks.l_cancel_online import (
    CANCELLED_MIN,
    DAMAGE_STATES,
    DEAD_STATES,
    LANDING_AIR_STATES,
)

log = logging.getLogger("rlvr.online.savestate")

MOVE_NAMES = {70: "NAIR", 71: "FAIR", 72: "BAIR", 73: "UAIR", 74: "DAIR"}

SLIPPI_PORT_MIN = 52100
SLIPPI_PORT_MAX = 52199


# -- pipe verbs ---------------------------------------------------------------

def send_savestate(ctrl: melee.Controller, path: str) -> None:
    ctrl._write(f"SAVESTATE {path}\n")
    ctrl.flush()


def send_loadstate(ctrl: melee.Controller, path: str) -> None:
    ctrl._write(f"LOADSTATE {path}\n")
    ctrl.flush()


# -- neutral controller -------------------------------------------------------

def press_neutral(ctrl: melee.Controller) -> None:
    ctrl.tilt_analog(melee.enums.Button.BUTTON_MAIN, 0.5, 0.5)
    ctrl.tilt_analog(melee.enums.Button.BUTTON_C, 0.5, 0.5)
    for b in ALL_ACTION_BUTTONS:
        ctrl.release_button(b)
    ctrl.press_shoulder(melee.enums.Button.BUTTON_L, 0.0)
    ctrl.flush()


def neutral_prev_sent() -> dict:
    """prev_sent dict for tools.inference_utils.build_frame matching a
    neutral controller (same shape _press_controller returns)."""
    prev = {"main_x": 0.5, "main_y": 0.5, "c_x": 0.5, "c_y": 0.5,
            "l_shldr": 0.0, "r_shldr": 0.0}
    for b in ["BUTTON_A", "BUTTON_B", "BUTTON_X", "BUTTON_Y",
              "BUTTON_Z", "BUTTON_L", "BUTTON_R"]:
        prev[f"btn_{b}"] = 0
    return prev


# -- live avoidable-lag tracker -----------------------------------------------

@dataclass
class LandingRun:
    """A completed LANDING_AIR_* state run observed live."""
    landing_state: int
    start_frame: int
    length: int
    exit_state: int

    @property
    def scoreable(self) -> bool:
        # Damage/dead exits: the opponent cut the landing short; realized
        # lag is opponent-determined, no input-timing signal. Same rule
        # as l_cancel_online.enrich_with_replay.
        return (self.exit_state not in DEAD_STATES
                and self.exit_state not in DAMAGE_STATES)

    @property
    def avoidable_lag(self) -> int:
        return max(0, self.length - CANCELLED_MIN[self.landing_state])

    @property
    def reward(self) -> float:
        return 1.0 if self.avoidable_lag == 0 else 0.0

    @property
    def move(self) -> str:
        return MOVE_NAMES.get(self.landing_state, str(self.landing_state))


class LandingRunTracker:
    """Streaming landing-run state machine. Feed (frame_id, action_state)
    once per game frame; returns a LandingRun when one completes.

    Must be reset() whenever the frame counter becomes discontinuous
    (match transition, savestate load)."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._state: Optional[int] = None
        self._start_frame = 0
        self._len = 0
        self._last_frame: Optional[int] = None

    @property
    def open(self) -> bool:
        return self._state is not None

    def push(self, frame_id: int, action: int) -> Optional[LandingRun]:
        if self._last_frame is not None and frame_id == self._last_frame:
            return None  # duplicate frame delivery
        self._last_frame = frame_id
        if self._state is not None:
            if action == self._state:
                self._len += 1
                return None
            run = LandingRun(self._state, self._start_frame, self._len, action)
            self._state = None
            if action in LANDING_AIR_STATES:
                # direct landing->landing chain (rare)
                self._state, self._start_frame, self._len = action, frame_id, 1
            return run
        if action in LANDING_AIR_STATES:
            self._state, self._start_frame, self._len = action, frame_id, 1
        return None


# -- policy context snapshot / restore ---------------------------------------

def snapshot_policy(policy) -> dict:
    """Snapshot a _PolicyRunner's context window + prev controller state
    (CPU, detached). Restorable via restore_policy — this is what lets a
    drill rollout resume the policy exactly as it was at harvest time."""
    if policy._cpu_window is None:
        raise RuntimeError("policy has no context window yet (no frames pushed)")
    return {
        "window": {k: v.clone() for k, v in policy._cpu_window.items()},
        "prev_sent": dict(policy.prev_sent) if policy.prev_sent else None,
    }


def restore_policy(policy, snap: dict, device) -> None:
    """Load a snapshot_policy() dict back into a _PolicyRunner. After
    this, push_frame takes the incremental (slide-left) path, appending
    live frames to the restored window."""
    policy._cpu_window = {k: v.clone() for k, v in snap["window"].items()}
    policy._gpu_window = {k: v.to(device) for k, v in snap["window"].items()}
    policy.prev_sent = (dict(snap["prev_sent"])
                        if snap.get("prev_sent") else None)


# -- console session ----------------------------------------------------------

@dataclass
class SessionConfig:
    dolphin_path: str
    iso_path: str
    slippi_port: int
    character: str = "FOX"
    cpu_character: str = "FOX"
    cpu_level: int = 9
    stage: str = "FINAL_DESTINATION"
    enable_ffw: bool = True
    gfx_backend: str = "Null"


class SavestateSession:
    """One patched-Dolphin (emulator_ss) bot-vs-CPU session with the
    savestate pipe verbs available on `ego_ctrl`. Menu navigation,
    keepalive across PPO pauses, and clean shutdown (console.stop() only
    — never a broad pkill; other agents run Dolphins concurrently)."""

    def __init__(self, cfg: SessionConfig):
        if not (SLIPPI_PORT_MIN <= cfg.slippi_port <= SLIPPI_PORT_MAX):
            raise ValueError(
                f"slippi_port {cfg.slippi_port} outside reserved range "
                f"{SLIPPI_PORT_MIN}-{SLIPPI_PORT_MAX}")
        self.cfg = cfg
        self.console: Optional[melee.Console] = None
        self.ego_ctrl: Optional[melee.Controller] = None
        self.cpu_ctrl: Optional[melee.Controller] = None
        self._menu_ego = melee.MenuHelper()
        self._menu_cpu = melee.MenuHelper()
        self._keepalive_thread: Optional[threading.Thread] = None
        self._keepalive_stop = threading.Event()

    def start(self) -> None:
        self.console = melee.Console(
            path=self.cfg.dolphin_path, is_dolphin=True,
            tmp_home_directory=True, copy_home_directory=False,
            blocking_input=True, online_delay=0,
            slippi_port=self.cfg.slippi_port,
            setup_gecko_codes=True, fullscreen=False,
            gfx_backend=self.cfg.gfx_backend,
            disable_audio=True,
            use_exi_inputs=True, enable_ffw=self.cfg.enable_ffw,
            save_replays=False,
            # REQUIRED to observe LOADSTATE rewinds: the default drops
            # any frame <= the max frame already seen.
            skip_rollback_frames=False,
        )
        self.ego_ctrl = melee.Controller(
            console=self.console, port=1, type=melee.ControllerType.STANDARD)
        self.cpu_ctrl = melee.Controller(
            console=self.console, port=2, type=melee.ControllerType.STANDARD)
        self.console.run(iso_path=self.cfg.iso_path)
        if not self.console.connect():
            raise RuntimeError("console.connect() failed")
        if not self.ego_ctrl.connect() or not self.cpu_ctrl.connect():
            raise RuntimeError("controller connect failed")
        log.info("session up: dolphin=%s slippi_port=%d ffw=%s",
                 self.cfg.dolphin_path, self.cfg.slippi_port,
                 self.cfg.enable_ffw)

    def stop(self) -> None:
        self.stop_keepalive()
        if self.console is not None:
            try:
                self.console.stop()
            except Exception:
                pass
            self.console = None

    def in_game(self, gs) -> bool:
        return gs.menu_state in (melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH)

    def menu_frame(self, gs) -> None:
        """Drive one menu frame toward a bot(P1) vs CPU(P2) match."""
        self._menu_ego.menu_helper_simple(
            gs, self.ego_ctrl, melee.Character[self.cfg.character],
            melee.Stage[self.cfg.stage], cpu_level=0, autostart=False,
            costume=0)
        self._menu_cpu.menu_helper_simple(
            gs, self.cpu_ctrl, melee.Character[self.cfg.cpu_character],
            melee.Stage[self.cfg.stage], cpu_level=self.cfg.cpu_level,
            autostart=True, costume=1)
        self.ego_ctrl.flush()
        self.cpu_ctrl.flush()

    # -- keepalive (same rationale as DolphinActor.start_keepalive):
    # nothing pumps console.step() during a PPO update, enet drops the
    # slippstream peer after ~20 s of silence. Idle-step every 10 s.
    def start_keepalive(self) -> None:
        if self.console is None:
            return
        if self._keepalive_thread is not None and self._keepalive_thread.is_alive():
            return
        self._keepalive_stop.clear()
        self._keepalive_thread = threading.Thread(
            target=self._keepalive_loop, name="ss-keepalive", daemon=True)
        self._keepalive_thread.start()

    def stop_keepalive(self) -> None:
        if self._keepalive_thread is None:
            return
        self._keepalive_stop.set()
        self._keepalive_thread.join(timeout=10.0)
        self._keepalive_thread = None

    def _keepalive_loop(self) -> None:
        KEEPALIVE_PERIOD = 10.0  # < ~20 s enet timeout
        first = True
        while not self._keepalive_stop.is_set():
            if not first:
                if self._keepalive_stop.wait(timeout=KEEPALIVE_PERIOD):
                    return
            first = False
            try:
                self.console.step()
                for c in (self.ego_ctrl, self.cpu_ctrl):
                    if c is not None:
                        press_neutral(c)
            except Exception as e:
                log.warning("keepalive step failed (%s); stopping", e)
                break
