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

import json
import logging
import os
import signal
import sys
import threading
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
    build_frame_p2,
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
    disable_audio: bool = False         # match tools/play.py
    # FFW-only: enable via the ExiAI Ishiiruka Dolphin fork at
    # emulator_ffw/. Realtime mode (default) uses the regular online
    # emulator at emulator/. Both can't be combined: ffw needs
    # use_exi_inputs=True which requires the Exi-AI build.
    use_exi_inputs: bool = False
    enable_ffw: bool = False
    # Pacing knobs. With FFW the dominant bottleneck on our pipeline
    # becomes the Python step loop (~17 ms/frame) rather than Dolphin
    # itself. Setting blocking_input=False unblocks Dolphin from
    # waiting on our input each frame, and polling_mode=True lets
    # console.step() return immediately when no new gamestate is ready.
    # Together they let Dolphin emulate at its own pace; we read
    # whatever's freshest. Comes at the cost of input lag — fine for
    # episodic outcomes (combo extension, edgeguard) but bad for
    # frame-perfect tasks (L-cancel).
    blocking_input: bool = True
    polling_mode: bool = False
    replay_dir: Optional[str] = None
    state_history_len: int = 256       # long enough for any task's episode
    max_episode_frames: int = 600      # safety: kill runaway episodes
    # Whole-match episode mode (CompositeVRTask / the VR suite): one
    # episode == one whole match. Disables the max_episode_frames cap;
    # the match-end transition scores (not drops) the open episode.
    whole_match_episode: bool = False
    # Bot-vs-bot training. When opponent_ckpt is None, the cpu_port is
    # driven by cpu_level (legacy CPU-9 mode). When set, that .pt is
    # loaded as a frozen second policy and pressed every frame. cpu_level
    # is ignored in this case (the menu helper picks 0 for both ports).
    # opponent_data_dir defaults to the trainee's data_dir if None.
    opponent_ckpt: Optional[str] = None
    opponent_data_dir: Optional[str] = None
    opponent_temperature: float = 1.0   # same sampling as trainee by default
    # Costume indices (Fox: 0=default, 1=red, 2=black, 3=green). Used
    # to visually distinguish trainee from frozen opponent when both
    # play the same character. Default trainee=0 (white), opp=3 (green).
    trainee_costume: int = 0
    opponent_costume: int = 3


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

        # Optional frozen opponent policy. When configured, the cpu_port
        # is driven by this model on every frame instead of by CPU-9.
        self.opp_model = None
        self.opp_ctx: Optional[dict] = None
        self.opp_policy: Optional[_PolicyRunner] = None
        self.opp_n_btn: Optional[int] = None
        # CUDA side stream for the opponent's forward. None when there
        # is no opponent or when running CPU-only. Allocated below.
        self._opp_stream: Optional["torch.cuda.Stream"] = None
        if cfg.opponent_ckpt:
            opp_data_dir = cfg.opponent_data_dir or ""
            log.info("loading frozen opponent: %s (data_dir=%s, T=%.2f)",
                     cfg.opponent_ckpt, opp_data_dir or "<trainee's>",
                     cfg.opponent_temperature)
            self.opp_model, opp_cfg = load_mimic_model(
                cfg.opponent_ckpt, device)
            for p in self.opp_model.parameters():
                p.requires_grad_(False)
            self.opp_model.eval()
            # Per-policy context with matching n_combos.
            opp_ctx_base = (load_inference_context(opp_data_dir)
                            if opp_data_dir else ctx)
            from mimic.features import BTN7_N_CLASSES
            opp_ctx = dict(opp_ctx_base)
            n = opp_cfg.n_controller_combos
            if n == BTN7_N_CLASSES:
                opp_ctx["combo_map"] = {}
                opp_ctx["n_combos"] = n
            elif n == 5:
                opp_ctx["combo_map"] = {
                    (1, 0, 0, 0, 0): 0, (0, 1, 0, 0, 0): 1, (0, 0, 1, 0, 0): 2,
                    (0, 0, 0, 1, 0): 3, (0, 0, 0, 0, 0): 4, (0, 0, 0, 0, 1): 4,
                    (1, 0, 0, 0, 1): 0, (0, 1, 0, 0, 1): 1, (0, 0, 1, 0, 1): 2,
                    (0, 0, 0, 1, 1): 3,
                }
                opp_ctx["n_combos"] = 5
            self.opp_ctx = opp_ctx
            self.opp_policy = _PolicyRunner(
                self.opp_model, opp_cfg.max_seq_len, device, opp_ctx)
            self.opp_n_btn = n
            # Side stream lets the opp forward overlap the trainee's on
            # the GPU. Frame budget is 16.67 ms and each ~20M-param
            # forward is ~5-10 ms; without overlap the two serialize
            # and Dolphin runs <60 fps (blocking_input waits on Python).
            if "cuda" in str(device):
                self._opp_stream = torch.cuda.Stream(device=device)

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

        # enet keepalive (see start_keepalive): a background thread that
        # keeps Dolphin stepping during the inter-update PPO pause.
        self._keepalive_thread: Optional[threading.Thread] = None
        self._keepalive_stop = threading.Event()

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
        # Last in-game stocks (for emitting win/loss on match-end). The
        # menu-transition frame doesn't have player stocks reliably, so
        # we snapshot them every in-game frame and read them on close.
        self._last_trainee_stocks: int = 0
        self._last_opp_stocks: int = 0

    def start(self):
        replay_dir = self.cfg.replay_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "replays_online",
        )
        os.makedirs(replay_dir, exist_ok=True)
        # Always-headless: route Dolphin to the Xvfb buffer (:99) when no
        # display is inherited. setdefault, so an explicit DISPLAY (e.g. a
        # dev watching on :0) still wins. :99 is reached only here and in
        # tools/play_netplay.py — never via a global export (see setup.sh).
        os.environ.setdefault("DISPLAY", ":99")
        self.console = melee.Console(
            path=self.cfg.dolphin_path, is_dolphin=True,
            tmp_home_directory=True, copy_home_directory=False,
            blocking_input=self.cfg.blocking_input,
            polling_mode=self.cfg.polling_mode,
            online_delay=0,
            setup_gecko_codes=True, fullscreen=False,
            gfx_backend=self.cfg.gfx_backend,
            disable_audio=self.cfg.disable_audio,
            use_exi_inputs=self.cfg.use_exi_inputs,
            enable_ffw=self.cfg.enable_ffw,
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

    # -- enet keepalive ------------------------------------------------------
    # Between collect() calls the main thread is busy in ppo_update (~90 s of
    # GPU work) and nothing pumps console.step(). enet is only serviced inside
    # step(), so Dolphin's slippstream peer stops seeing traffic and drops the
    # connection (melee.slippstream.EnetDisconnected) after ~20 s — fatal for
    # FFW runs (this is why RL runs were forced to realtime). The keepalive
    # thread idle-steps Dolphin (neutral input, between matches) so the
    # connection stays warm across the PPO pause.
    def start_keepalive(self) -> None:
        """Begin idle-stepping Dolphin on a background thread. Call right
        after collect() returns; pair with stop_keepalive() before the next
        collect() — the two must never drive the console concurrently."""
        if self.console is None:
            return
        if self._keepalive_thread is not None and self._keepalive_thread.is_alive():
            return
        self._keepalive_stop.clear()
        self._keepalive_thread = threading.Thread(
            target=self._keepalive_loop, name="dolphin-keepalive", daemon=True)
        self._keepalive_thread.start()

    def stop_keepalive(self) -> None:
        """Stop the keepalive thread and wait for it to exit, so the main
        thread regains exclusive ownership of the console before collect()."""
        if self._keepalive_thread is None:
            return
        self._keepalive_stop.set()
        self._keepalive_thread.join(timeout=10.0)
        self._keepalive_thread = None

    def _keepalive_loop(self) -> None:
        """Step Dolphin with neutral input until stopped. Discards game
        state — its only job is to keep enet serviced. Touches only the
        console + controllers (never episode/model state), so it is safe
        to run concurrently with ppo_update on the main thread."""
        while not self._keepalive_stop.is_set():
            try:
                self.console.step()
                for c in (self.ego_ctrl, self.cpu_ctrl):
                    if c is None:
                        continue
                    c.tilt_analog(melee.enums.Button.BUTTON_MAIN, 0.5, 0.5)
                    c.tilt_analog(melee.enums.Button.BUTTON_C, 0.5, 0.5)
                    for b in ALL_ACTION_BUTTONS:
                        c.release_button(b)
                    c.press_shoulder(melee.enums.Button.BUTTON_L, 0.0)
                    c.flush()
            except Exception as e:
                log.warning("keepalive step failed (%s) — stopping keepalive", e)
                break

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
                # Emit win/loss based on the last in-game stocks we saw.
                ts, os_ = self._last_trainee_stocks, self._last_opp_stocks
                if ts > os_:
                    result = "win"
                elif os_ > ts:
                    result = "loss"
                else:
                    result = "draw"
                log.info("EVT_MATCH_END result=%s trainee_stocks=%d opp_stocks=%d",
                         result, ts, os_)
                self._in_game = False
                # Whole-match episode: this match IS the episode, so
                # score it. Legacy scenario tasks abort an unfinished
                # scenario at match end, as before.
                if (self.cfg.whole_match_episode
                        and self._episode_open_idx is not None):
                    self._score_and_close_open_episode()
                else:
                    self._close_open_episode_abortive()
                self._find_latest_replay()
            # If opponent is a bot, both ports are bot-driven (cpu_level=0).
            opp_cpu_level = 0 if self.opp_policy is not None else self.cfg.cpu_level
            self._menu_ego.menu_helper_simple(
                gs, self.ego_ctrl, self._bot_char, self._stage,
                cpu_level=0, autostart=False,
                costume=self.cfg.trainee_costume)
            self._menu_cpu.menu_helper_simple(
                gs, self.cpu_ctrl, self._cpu_char, self._stage,
                cpu_level=opp_cpu_level, autostart=True,
                costume=self.cfg.opponent_costume)
            self.ego_ctrl.flush()
            self.cpu_ctrl.flush()
            return

        self._in_game = True
        self.step_count += 1

        # Snapshot current stocks for the eventual EVT_MATCH_END emission.
        try:
            cpu_port_for_stocks = 2 if self.self_port == 1 else 1
            _t_ps = gs.players.get(self.self_port)
            _o_ps = gs.players.get(cpu_port_for_stocks)
            if _t_ps is not None:
                self._last_trainee_stocks = int(_t_ps.stock)
            if _o_ps is not None:
                self._last_opp_stocks = int(_o_ps.stock)
        except Exception:
            pass

        # Build input frame(s). Trainee always; opponent if configured.
        # Both frames are built eagerly so we can launch the two GPU
        # forwards concurrently — see "concurrent dispatch" below.
        frame = build_frame(gs, self.policy.prev_sent, self.ctx)
        if frame is None:
            return
        self.policy.push_frame(frame)

        opp_frame = None
        if self.opp_policy is not None:
            if self.self_port == 1:
                opp_frame = build_frame_p2(gs, self.opp_policy.prev_sent,
                                           self.opp_ctx)
            else:
                opp_frame = build_frame(gs, self.opp_policy.prev_sent,
                                        self.opp_ctx)
            if opp_frame is not None:
                self.opp_policy.push_frame(opp_frame)

        # Concurrent dispatch: launch BOTH forwards before draining
        # either. The opp forward goes on a side CUDA stream so it
        # overlaps the trainee's default-stream forward on the GPU. The
        # trainee sample below implicitly syncs the default stream; the
        # opp stream is waited on later, after the CPU-bound state
        # machine / HUD / snapshot work runs alongside the opp kernel.
        opp_logits = None
        with torch.no_grad():
            if opp_frame is not None and self._opp_stream is not None:
                self._opp_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self._opp_stream):
                    opp_logits = self.opp_policy.forward_latest(self.opp_model)
            theta_logits = self.policy.forward_latest(self.model)
            # CPU / no-side-stream fallback: sequential.
            if opp_frame is not None and self._opp_stream is None:
                opp_logits = self.opp_policy.forward_latest(self.opp_model)

        (m_i, s_i, c_i, b_i), lp_old = _sample_four_heads(
            theta_logits, self.cfg.temperature
        )

        # Track task state machine FIRST so should_start sees the latest frame.
        rlvr_gs = self._rlvr_gamestate(gs)
        self._state_history.append(rlvr_gs)

        if self._episode_open_idx is None:
            if self.task.should_start(self._state_history):
                self._episode_open_idx = len(self._state_history) - 1
                self._pending = []
                # Per-episode "open" event for the live HUD.
                # Schema: EVT_EP_OPEN frame=<g> start_pct=<f>
                start_pct = ""
                start_state = getattr(self.task,
                                       "_episode_start_opp_state", None)
                if start_state is not None:
                    start_pct = f"{start_state[0]:.1f}"
                log.info("EVT_EP_OPEN frame=%d start_pct=%s",
                         int(gs.frame), start_pct)

        # Live opp-percent tick for the HUD's live plot — every 6 game
        # frames while a window is open. Schema:
        #   EVT_EP_TICK frame=<g> opp_pct=<f>
        if self._episode_open_idx is not None and (int(gs.frame) % 6) == 0:
            opp = None
            for port, ps in gs.players.items():
                if int(port) != self.self_port:
                    opp = ps
                    break
            if opp is not None:
                log.info("EVT_EP_TICK frame=%d opp_pct=%.1f",
                         int(gs.frame), float(opp.percent))

        # Live per-VR pulse for the HUD — every 30 game frames (~2 Hz at
        # 60 fps) emit the composite task's in-progress per-VR state, so
        # the HUD can show each card's reward climbing during the match
        # instead of going silent for ~3 min between EVT_EP_VR events.
        if (self._episode_open_idx is not None
                and (int(gs.frame) % 30) == 0
                and hasattr(self.task, "live_state")):
            log.info("EVT_VR_TICK %s",
                     json.dumps(self.task.live_state(), sort_keys=True))

        # Ref-model forward is now done at PPO-update time on the
        # cached obs (frozen weights → deterministic logprob, doesn't
        # need to be computed live). This saves ~5ms/frame during
        # open windows where the actor would otherwise burn budget
        # forwarding the ref. The FrameRecord still gets a placeholder
        # for back-compat with old PPO code paths that read it.
        lp_ref = lp_old

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
            # Per-frame VR hook: whole-match composite tasks accumulate
            # their per-frame reward here. Optional — scenario tasks omit
            # it. Called in lockstep with _pending so the two stay aligned
            # (one observe per FrameRecord).
            _observe = getattr(self.task, "observe", None)
            if _observe is not None:
                _observe(self._state_history)

        # Press controller. n_btn from logits shape.
        n_btn = int(theta_logits["btn_logits"].shape[-1])
        self.policy.prev_sent = _press_controller(
            self.ego_ctrl, m_i, s_i, c_i, b_i, n_btn
        )

        # Opponent step (when configured): the forward was already
        # launched concurrently above; just wait, sample, and press.
        if opp_logits is not None:
            if self._opp_stream is not None:
                torch.cuda.current_stream().wait_stream(self._opp_stream)
            (om_i, os_i, oc_i, ob_i), _ = _sample_four_heads(
                opp_logits, self.cfg.opponent_temperature
            )
            self.opp_policy.prev_sent = _press_controller(
                self.cpu_ctrl, om_i, os_i, oc_i, ob_i, self.opp_n_btn
            )

    def _close_open_episode_abortive(self) -> None:
        """Discard any in-progress episode (menu-return, abort)."""
        self._episode_open_idx = None
        self._pending = []

    def _score_and_close_open_episode(self) -> None:
        """Score the open episode via task.compute_outcome, build the
        Episode, buffer it for the match, and reset episode state.

        Used both for scenario-task closes (should_end / cap) and for
        whole-match composite episodes at the match-end transition."""
        if self._episode_open_idx is None:
            return
        outcome = self.task.compute_outcome(
            self._state_history, self._episode_open_idx
        )
        if outcome.per_frame_reward is not None:
            for i, r in enumerate(outcome.per_frame_reward[:len(self._pending)]):
                self._pending[i].reward = float(r)
        metadata = dict(outcome.metadata or {})
        # Composite-VR per-episode event for the live HUD: one JSON blob,
        # parsed with json.loads (no brittle regex). `vrs` maps each VR id
        # -> {weight, reward, ...diagnostic counts}.
        log.info("EVT_EP_VR %s", json.dumps({
            "n_frames": metadata.get("n_frames"),
            "reward_sum": round(float(metadata.get("reward_sum", 0.0)), 4),
            "terminal": round(float(outcome.terminal_reward), 4),
            "result": metadata.get("result", "?"),
            "vrs": {k: v for k, v in metadata.items() if isinstance(v, dict)},
        }, sort_keys=True))
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
        # Per-episode log line for the live HUD watcher.
        _r = metadata.get("result", "?")
        _dmg = metadata.get("damage", "")
        _sp = metadata.get("start_percent", "")
        _ep_ = metadata.get("end_percent", "")
        _st = metadata.get("stocks_taken", "")
        log.info("EVT_EP frame=%d result=%s reward=%.3f "
                 "damage=%s start_pct=%s end_pct=%s stocks_taken=%s",
                 ep.end_game_frame, _r, float(outcome.terminal_reward),
                 _dmg, _sp, _ep_, _st)
        self._pending = []
        self._episode_open_idx = None

    def _find_latest_replay(self) -> None:
        """Locate the .slp libmelee just saved for the finished match.

        libmelee closes the .slp on the same tick as the menu transition,
        but the OS may not have fully flushed it to disk by the time
        peppi tries to parse. We wait (up to ~2s) for the file size to
        stabilize across two consecutive checks before declaring the
        replay ready — without this the enrichment step hits
        "I/O error: failed to fill whole buffer" and drops every
        pending episode in the match (deadly for tasks like l_cancel
        whose rewards are all post-match)."""
        replay_dir = self.cfg.replay_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "replays_online",
        )
        p = Path(replay_dir)
        if not p.exists():
            return
        slps = sorted(p.rglob("*.slp"), key=lambda f: f.stat().st_mtime)
        if not slps:
            return
        latest = slps[-1]
        # Wait for size to stabilize.
        import time as _t
        last_size = -1
        for _ in range(40):  # 40 * 50ms = 2s max
            try:
                cur = latest.stat().st_size
            except OSError:
                cur = -1
            if cur > 0 and cur == last_size:
                break
            last_size = cur
            _t.sleep(0.05)
        self._last_replay_path = latest
        log.info("match replay: %s (size=%d)", latest, last_size)

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
        per_ep = (28800 if self.cfg.whole_match_episode
                  else self.cfg.max_episode_frames)
        max_steps = n_episodes * per_ep + 60 * 60 * 60  # broad safety
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
                # The max-episode-frames cap used to abortively close
                # (dropping the entire episode silently) on long
                # combos. That cost us real reward signal — a combo
                # that runs 10+ seconds because opp gets launched far
                # is a legit combo, not garbage. Now: treat the cap
                # as if should_end returned True and score normally.
                cap_hit = (
                    not self.cfg.whole_match_episode
                    and len(self._pending) >= self.cfg.max_episode_frames
                )
                if cap_hit or self.task.should_end(
                        self._state_history, self._episode_open_idx):
                    if cap_hit:
                        log.warning(
                            "episode hit max_episode_frames=%d cap; "
                            "scoring at cap",
                            self.cfg.max_episode_frames)
                    self._score_and_close_open_episode()
                    continue

        if len(collected) < n_episodes:
            log.warning("collected %d/%d episodes before step cap (%d steps)",
                        len(collected), n_episodes, steps_this_call)
        return collected
