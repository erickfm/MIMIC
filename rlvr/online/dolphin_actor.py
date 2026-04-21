"""Dolphin actor: drives one headless Dolphin and collects online episodes.

Architecture per step:
    1. `console.step()` blocks until the next game frame arrives
       (Dolphin `blocking_input=True`).
    2. If in menu, drive menu navigation to start a match.
    3. If in-game, build the MIMIC input tensor dict, forward the policy
       + frozen reference, sample the 4 factored heads, record the log-
       probs under each, press the sampled controller, advance.
    4. Push the GameState into the task's state history. Ask the task
       whether an episode just started / ended. When one ends,
       `compute_outcome()` produces the reward, and the buffered frames
       inside the episode become one `Episode`.
    5. When enough episodes are buffered, yield them to the caller.

The actor does NOT compute losses or step optimizers — it only collects
episodes. Training happens in rlvr/online/loop.py.

Runs fully headless under Xvfb with Vulkan (see CLAUDE.md pitfalls
16-17 for why).
"""
from __future__ import annotations

import logging
import os
import signal
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional

import melee
import numpy as np
import torch
import torch.nn.functional as F

from mimic.features import (
    HAL_CSTICK_CLUSTERS_9,
    HAL_SHOULDER_CLUSTERS_3,
    HAL_STICK_CLUSTERS_37,
)
from rlvr.online.episode import EpisodeOutcome, OnlineTask
from rlvr.online.trajectory import Episode, FrameRecord
from rlvr.state.libmelee_adapter import _ps_from_libmelee
from rlvr.state.gamestate import SCHEMA_VERSION, ControllerInput, GameState, PlayerState
from tools.inference_utils import (
    build_frame,
    load_inference_context,
    load_mimic_model,
)


log = logging.getLogger("rlvr.online.actor")

ALL_ACTION_BUTTONS = [
    melee.enums.Button.BUTTON_A, melee.enums.Button.BUTTON_B,
    melee.enums.Button.BUTTON_X, melee.enums.Button.BUTTON_Y,
    melee.enums.Button.BUTTON_Z, melee.enums.Button.BUTTON_L,
    melee.enums.Button.BUTTON_R,
]


@dataclass
class ActorConfig:
    dolphin_path: str
    iso_path: str
    character: str = "FOX"
    cpu_character: str = "FOX"
    cpu_level: int = 9
    stage: str = "FINAL_DESTINATION"
    temperature: float = 1.0
    gfx_backend: str = ""               # "" inherits Dolphin default (works headless here)
    disable_audio: bool = False         # match tools/play_vs_cpu.py
    replay_dir: Optional[str] = None
    state_history_len: int = 256       # long enough for any task's episode
    max_episode_frames: int = 600      # safety: kill runaway episodes


class _PolicyRunner:
    """Wraps the streaming context deque + per-frame model call for one
    player. Mirrors tools/inference_utils.PlayerState but returns the
    logits we need for online RL (not just the sampled action)."""

    def __init__(self, model, seq_len: int, device, ctx: dict):
        self.model = model
        self.seq_len = seq_len
        self.device = device
        self.ctx = ctx
        self._cache: Deque[Dict[str, torch.Tensor]] = deque(maxlen=seq_len)
        self.prev_sent: Optional[dict] = None

    def push_frame(self, frame: Dict[str, torch.Tensor]) -> None:
        if len(self._cache) == 0:
            from tools.inference_utils import build_mock_frame
            mock = build_mock_frame(self.ctx)
            for _ in range(self.seq_len - 1):
                self._cache.append({k: v.clone() for k, v in mock.items()})
        self._cache.append(frame)

    def forward_latest(self, model=None) -> Dict[str, torch.Tensor]:
        """Run the model on the current context window; return the
        logits at the final position (B=1)."""
        m = model if model is not None else self.model
        frames = list(self._cache)
        batch = {}
        for k in frames[0]:
            batch[k] = torch.cat([f[k] for f in frames], dim=0).unsqueeze(0).to(self.device)
        return m(batch)


def _sample_four_heads(logits: Dict[str, torch.Tensor], temperature: float):
    """Sample each factored head and return (indices[4], logprobs_sum).
    Indices order: main, shldr, cdir, btn."""
    def _last(t: torch.Tensor) -> torch.Tensor:
        return t[0, -1] if t.dim() == 3 else t[0]

    shldr_l = _last(logits["shoulder_val"]).float()
    cdir_l = _last(logits["c_dir_logits"]).float()
    main_l = _last(logits["main_xy"]).float()
    btn_l = _last(logits["btn_logits"]).float()

    def _samp(lg, T):
        safe = torch.nan_to_num(lg, nan=-1e9, posinf=1e9, neginf=-1e9)
        if T <= 0:
            idx = int(torch.argmax(safe))
        else:
            probs = F.softmax(safe / T, dim=-1)
            if not torch.isfinite(probs).all() or float(probs.sum()) <= 0:
                idx = int(torch.argmax(safe))
            else:
                idx = int(torch.multinomial(probs, 1))
        log_probs = F.log_softmax(safe, dim=-1)
        return idx, float(log_probs[idx])

    m_i, m_lp = _samp(main_l, temperature)
    s_i, s_lp = _samp(shldr_l, temperature)
    c_i, c_lp = _samp(cdir_l, temperature)
    b_i, b_lp = _samp(btn_l, temperature)

    return (m_i, s_i, c_i, b_i), (m_lp + s_lp + c_lp + b_lp)


def _logprob_of_indices(logits: Dict[str, torch.Tensor], indices):
    m_i, s_i, c_i, b_i = indices
    def _lp(t, idx):
        lg = (t[0, -1] if t.dim() == 3 else t[0]).float()
        safe = torch.nan_to_num(lg, nan=-1e9, posinf=1e9, neginf=-1e9)
        return float(F.log_softmax(safe, dim=-1)[idx])
    return (
        _lp(logits["main_xy"], m_i)
        + _lp(logits["shoulder_val"], s_i)
        + _lp(logits["c_dir_logits"], c_i)
        + _lp(logits["btn_logits"], b_i)
    )


def _press_controller(ctrl, main_idx, shldr_idx, cdir_idx, btn_idx, n_btn: int) -> dict:
    """Map sampled indices -> controller presses (in place on `ctrl`) +
    return the prev_sent dict for the next frame's encoder input."""
    mx = float(HAL_STICK_CLUSTERS_37[main_idx][0])
    my = float(HAL_STICK_CLUSTERS_37[main_idx][1])
    cx = float(HAL_CSTICK_CLUSTERS_9[cdir_idx][0])
    cy = float(HAL_CSTICK_CLUSTERS_9[cdir_idx][1])
    shldr = float(HAL_SHOULDER_CLUSTERS_3[shldr_idx])

    ctrl.tilt_analog(melee.enums.Button.BUTTON_MAIN, mx, my)
    ctrl.tilt_analog(melee.enums.Button.BUTTON_C, cx, cy)
    for b in ALL_ACTION_BUTTONS:
        ctrl.release_button(b)
    ctrl.press_shoulder(melee.enums.Button.BUTTON_L, shldr)

    pressed = []
    if n_btn == 7:
        if btn_idx == 0: ctrl.press_button(melee.enums.Button.BUTTON_A); pressed.append("A")
        elif btn_idx == 1: ctrl.press_button(melee.enums.Button.BUTTON_B); pressed.append("B")
        elif btn_idx == 2: ctrl.press_button(melee.enums.Button.BUTTON_Z); pressed.append("Z")
        elif btn_idx == 3: ctrl.press_button(melee.enums.Button.BUTTON_X); pressed.append("JUMP")
        elif btn_idx == 4: ctrl.press_button(melee.enums.Button.BUTTON_L); pressed.append("TRIG")
        elif btn_idx == 5:
            ctrl.press_button(melee.enums.Button.BUTTON_A)
            ctrl.press_button(melee.enums.Button.BUTTON_L); pressed.append("A+TRIG")
    else:
        names = {0: "A", 1: "B", 2: "X", 3: "Z"}
        if btn_idx in names:
            b = {"A": melee.enums.Button.BUTTON_A, "B": melee.enums.Button.BUTTON_B,
                 "X": melee.enums.Button.BUTTON_X, "Z": melee.enums.Button.BUTTON_Z}[names[btn_idx]]
            ctrl.press_button(b); pressed.append(names[btn_idx])

    ctrl.flush()

    prev = {"main_x": mx, "main_y": my, "c_x": cx, "c_y": cy,
            "l_shldr": shldr, "r_shldr": 0.0}
    for b in ["BUTTON_A", "BUTTON_B", "BUTTON_X", "BUTTON_Y",
              "BUTTON_Z", "BUTTON_L", "BUTTON_R"]:
        prev[f"btn_{b}"] = 0
    for p in pressed:
        if p == "A": prev["btn_BUTTON_A"] = 1
        elif p == "B": prev["btn_BUTTON_B"] = 1
        elif p == "Z": prev["btn_BUTTON_Z"] = 1
        elif p == "JUMP" or p == "X": prev["btn_BUTTON_X"] = 1
        elif p == "TRIG": prev["btn_BUTTON_L"] = 1
        elif p == "A+TRIG":
            prev["btn_BUTTON_A"] = 1
            prev["btn_BUTTON_L"] = 1
    return prev


class DolphinActor:
    """Streams a single Dolphin session + collects online episodes.

    Usage:
        actor = DolphinActor(cfg, task, model, ref_model, ctx, device)
        actor.start()
        for episode in actor.collect(n_episodes=64):
            ...
        actor.stop()
    """

    def __init__(
        self,
        cfg: ActorConfig,
        task: OnlineTask,
        model,
        ref_model,
        ctx: dict,
        device: str = "cuda",
        model_seq_len: int = 256,
        self_port: int = 1,
    ):
        self.cfg = cfg
        self.task = task
        self.model = model
        self.ref_model = ref_model
        self.ctx = ctx
        self.device = device
        self.self_port = self_port
        self.policy = _PolicyRunner(model, model_seq_len, device, ctx)

        # Streaming state history (libmelee GameStates as RLVR PlayerState
        # objects via the libmelee_adapter shim — enough for task logic).
        self._state_history: Deque[GameState] = deque(maxlen=cfg.state_history_len)

        # Pending frames inside the current episode (if one is open).
        self._episode_open_idx: Optional[int] = None
        self._pending: List[FrameRecord] = []

        self.console: Optional[melee.Console] = None
        self.ego_ctrl: Optional[melee.Controller] = None
        self.cpu_ctrl: Optional[melee.Controller] = None
        self._menu_ego = melee.MenuHelper()
        self._menu_cpu = melee.MenuHelper()

        self._bot_char = melee.Character[cfg.character]
        self._cpu_char = melee.Character[cfg.cpu_character]
        self._stage = melee.Stage[cfg.stage]

        self._in_game = False
        self._call_count = 0
        self.step_count = 0
        self.episode_count = 0
        # Per-match episode buffer. Flushed on menu re-entry (= match end)
        # so enrichment can run on the freshly-written .slp.
        self._match_episodes: List[Episode] = []
        # Path of the most recently-closed replay (set by libmelee's
        # Console when it writes the .slp).
        self._last_replay_path: Optional[Path] = None

    def start(self):
        replay_dir = self.cfg.replay_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "replays_online",
        )
        os.makedirs(replay_dir, exist_ok=True)
        self.console = melee.Console(
            path=self.cfg.dolphin_path, is_dolphin=True,
            tmp_home_directory=True, copy_home_directory=False,
            blocking_input=True, online_delay=0,
            setup_gecko_codes=True, fullscreen=False,
            gfx_backend=self.cfg.gfx_backend,
            disable_audio=self.cfg.disable_audio, use_exi_inputs=False, enable_ffw=False,
            save_replays=True, replay_dir=replay_dir,
        )
        self.ego_ctrl = melee.Controller(
            console=self.console, port=self.self_port,
            type=melee.ControllerType.STANDARD,
        )
        cpu_port = 2 if self.self_port == 1 else 1
        self.cpu_ctrl = melee.Controller(
            console=self.console, port=cpu_port,
            type=melee.ControllerType.STANDARD,
        )
        self.console.run(iso_path=self.cfg.iso_path)
        self.console.connect()
        self.ego_ctrl.connect()
        self.cpu_ctrl.connect()
        log.info("actor connected (self_port=%d, cpu_port=%d)", self.self_port, cpu_port)

    def stop(self):
        if self.console is not None:
            try:
                self.console.stop()
            except Exception:
                pass
            self.console = None

    def _snapshot_context(self) -> Dict[str, torch.Tensor]:
        """Stack the policy cache into a (T, ...) tensor dict (on CPU,
        detached). Each value is the concatenation of the per-frame
        leading-dim-1 tensors stored in the deque."""
        frames = list(self.policy._cache)
        out: Dict[str, torch.Tensor] = {}
        for k in frames[0]:
            out[k] = torch.cat(
                [f[k].detach().cpu() for f in frames], dim=0
            )
        return out

    def _rlvr_gamestate(self, gs) -> GameState:
        """Convert a libmelee GameState -> RLVR GameState (for task logic)."""
        players = sorted(gs.players.items())
        return GameState(
            schema_version=SCHEMA_VERSION,
            frame_idx=int(gs.frame),
            stage=int(gs.stage.value),
            players=tuple(_ps_from_libmelee(ps, port=int(port))
                          for port, ps in players),
        )

    def _step_one_frame(self):
        """Advance Dolphin one frame; if in-game, sample an action,
        record a FrameRecord inside any open episode, and press the
        controller."""
        gs = self.console.step()
        if gs is None:
            return

        self._call_count += 1
        if self._call_count % 180 == 0:
            log.info("call=%d step=%d menu=%s in_game=%s ep_open=%s pending=%d match_eps=%d",
                     self._call_count, self.step_count,
                     getattr(gs.menu_state, "name", gs.menu_state),
                     self._in_game, self._episode_open_idx is not None,
                     len(self._pending), len(self._match_episodes))

        if gs.menu_state not in (melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH):
            if self._in_game:
                log.info("match ended, returning to menu")
                self._in_game = False
                self._close_open_episode_abortive()
                self._find_latest_replay()
            self._menu_ego.menu_helper_simple(
                gs, self.ego_ctrl, self._bot_char, self._stage,
                cpu_level=0, autostart=False)
            self._menu_cpu.menu_helper_simple(
                gs, self.cpu_ctrl, self._cpu_char, self._stage,
                cpu_level=self.cfg.cpu_level, autostart=True)
            self.ego_ctrl.flush()
            self.cpu_ctrl.flush()
            return

        self._in_game = True
        self.step_count += 1

        # Build input frame + forward
        frame = build_frame(gs, self.policy.prev_sent, self.ctx)
        if frame is None:
            return
        self.policy.push_frame(frame)

        # Two forwards: policy (for sampling + old logprob) and ref (no grad)
        with torch.no_grad():
            theta_logits = self.policy.forward_latest(self.model)
            ref_logits = self.policy.forward_latest(self.ref_model)

        (m_i, s_i, c_i, b_i), lp_old = _sample_four_heads(
            theta_logits, self.cfg.temperature
        )
        lp_ref = _logprob_of_indices(ref_logits, (m_i, s_i, c_i, b_i))

        # Track task state machine FIRST so should_start sees the latest frame.
        rlvr_gs = self._rlvr_gamestate(gs)
        self._state_history.append(rlvr_gs)

        if self._episode_open_idx is None:
            if self.task.should_start(self._state_history):
                self._episode_open_idx = len(self._state_history) - 1
                self._pending = []

        # Snapshot the full T-frame context at this moment — required so
        # PPO can recompute logprobs with gradient on the exact input
        # the policy saw at sampling time. Stored on CPU to keep GPU
        # free for the live inference stream.
        if self._episode_open_idx is not None:
            cache_snapshot = self._snapshot_context()
            rec = FrameRecord(
                obs=cache_snapshot,
                sampled_indices=torch.tensor(
                    [m_i, s_i, c_i, b_i], dtype=torch.long
                ),
                logprob_old=torch.tensor(lp_old, dtype=torch.float32),
                logprob_ref=torch.tensor(lp_ref, dtype=torch.float32),
                reward=0.0,
                game_frame_id=int(gs.frame),
            )
            self._pending.append(rec)

        # Press controller. n_btn from logits shape.
        n_btn = int(theta_logits["btn_logits"].shape[-1])
        self.policy.prev_sent = _press_controller(
            self.ego_ctrl, m_i, s_i, c_i, b_i, n_btn
        )

    def _close_open_episode_abortive(self) -> None:
        """Discard any in-progress episode (menu-return, abort)."""
        self._episode_open_idx = None
        self._pending = []

    def _find_latest_replay(self) -> None:
        """Locate the .slp libmelee just saved for the finished match."""
        replay_dir = self.cfg.replay_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "replays_online",
        )
        p = Path(replay_dir)
        if not p.exists():
            return
        slps = sorted(p.rglob("*.slp"), key=lambda f: f.stat().st_mtime)
        if slps:
            self._last_replay_path = slps[-1]
            log.info("match replay: %s", self._last_replay_path)

    def _finalize_match_episodes(self) -> List[Episode]:
        """Call task.enrich_with_replay on the buffered match episodes
        using the just-written .slp. Returns the possibly-filtered list
        and clears the per-match buffer."""
        if not self._match_episodes:
            return []
        episodes = self._match_episodes
        self._match_episodes = []
        if self._last_replay_path is None or not self._last_replay_path.exists():
            log.warning("no replay path found; skipping enrichment for %d eps",
                        len(episodes))
            # Drop pending-reward eps we can't score.
            import math as _m
            return [ep for ep in episodes
                    if not _m.isnan(ep.terminal_reward)]
        enrich = getattr(self.task, "enrich_with_replay", None)
        if enrich is None:
            return episodes
        return enrich(episodes, self._last_replay_path, self.self_port)

    def collect(self, n_episodes: int) -> List[Episode]:
        """Run Dolphin until `n_episodes` episodes have been finished
        and enriched. Matches boundaries are detected automatically —
        episodes are buffered per-match and enriched (task.enrich_with_
        replay) on match end, then surfaced to the caller.

        Returns the list once the quota is met, possibly slightly over
        the quota because enrichment only happens at match granularity.
        """
        collected: List[Episode] = []
        max_steps = n_episodes * self.cfg.max_episode_frames + 60 * 60 * 60  # broad safety
        steps_this_call = 0
        while len(collected) < n_episodes and steps_this_call < max_steps:
            was_in_game = self._in_game
            steps_this_call += 1
            self._step_one_frame()

            # Match just ended? Finalize buffered episodes.
            if was_in_game and not self._in_game:
                finalized = self._finalize_match_episodes()
                collected.extend(finalized)

            # Episode boundary check (only if open).
            if self._episode_open_idx is not None and self._in_game:
                if self.task.should_end(self._state_history, self._episode_open_idx):
                    outcome = self.task.compute_outcome(
                        self._state_history, self._episode_open_idx
                    )
                    if outcome.per_frame_reward is not None:
                        for i, r in enumerate(outcome.per_frame_reward[:len(self._pending)]):
                            self._pending[i].reward = float(r)
                    metadata = dict(outcome.metadata or {})
                    ep = Episode(
                        task_id=self.task.id,
                        frames=list(self._pending),
                        terminal_reward=float(outcome.terminal_reward),
                        start_game_frame=self._pending[0].game_frame_id if self._pending else 0,
                        end_game_frame=self._pending[-1].game_frame_id if self._pending else 0,
                        metadata=metadata,
                    )
                    self._match_episodes.append(ep)
                    self.episode_count += 1
                    self._pending = []
                    self._episode_open_idx = None
                    continue
                if len(self._pending) >= self.cfg.max_episode_frames:
                    log.warning("episode exceeded max_episode_frames=%d; aborting",
                                self.cfg.max_episode_frames)
                    self._close_open_episode_abortive()

        if len(collected) < n_episodes:
            log.warning("collected %d/%d episodes before step cap (%d steps)",
                        len(collected), n_episodes, steps_this_call)
        return collected
