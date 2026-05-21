"""Parallel-actor pool for RLVR.

N DolphinActor instances run concurrently, one per thread. A
BatchCoordinator synchronizes them at the forward-pass barrier and runs
ONE batched inference per tick (per N actors), so all N share the same
GPU forward kernels. The shared model + opp_model live as single
instances; only the per-actor state (Dolphin process, controller,
policy cache) differs.

Speedup target: ~N× over a single-actor pipeline. Each Dolphin is still
hard-capped at 60 fps realtime emulation (CLAUDE.md pitfall #18 — FFW
is unfaithful), so the only path to faster training is concurrent
realtime sessions.

Lockstep batching strategy: the coordinator waits up to `max_wait_ms`
for N submissions to arrive, then fires whatever it has. Under normal
operation all N actors submit within ~1 ms of each other; the timeout
only kicks in when one actor is at the menu between matches and not
producing forward requests. Out-of-game actors never call submit, so
they never deadlock the in-game ones — the timeout fires and the
batch goes ahead with whoever's ready.
"""
from __future__ import annotations

import logging
import threading
import time
from copy import copy
from typing import Any, Dict, List, Optional

import torch

from rlvr.online.dolphin_actor import ActorConfig, DolphinActor
from rlvr.online.episode import OnlineTask
from rlvr.online.trajectory import Episode

log = logging.getLogger("rlvr.actor_pool")


class _Submission:
    """One per-actor inference request. The owning actor thread blocks
    on `event` until the coordinator fills `theta` / `opp` and sets it."""

    __slots__ = ("actor_id", "gpu_window", "opp_window",
                 "theta", "opp", "event")

    def __init__(self, actor_id: int,
                 gpu_window: Dict[str, torch.Tensor],
                 opp_window: Optional[Dict[str, torch.Tensor]]):
        self.actor_id = actor_id
        self.gpu_window = gpu_window
        self.opp_window = opp_window
        self.theta: Optional[Dict[str, torch.Tensor]] = None
        self.opp: Optional[Dict[str, torch.Tensor]] = None
        self.event = threading.Event()


class BatchCoordinator:
    """Single-threaded inference dispatcher. Each actor thread calls
    `submit_and_wait()` to enqueue its (trainee_window, opp_window)
    inputs and block until the per-actor outputs come back.

    The coordinator thread loops:
      1. Wait for at least one submission to land.
      2. Wait up to `max_wait_ms` for more (target: N total).
      3. Stack pending submissions along batch dim, run one trainee
         forward + one opp forward (opp on a side stream to overlap).
      4. Split outputs per-actor and signal each event.
    """

    def __init__(self, n: int, model, opp_model, device: str,
                 max_wait_ms: float = 25.0):
        self.n = n
        self.device = device
        # Compile the inference path. Same logic as _PolicyRunner: shares
        # parameters with the raw model, so PPO's optimizer updates
        # propagate transparently. No precision change.
        self.model = (torch.compile(model)
                      if "cuda" in str(device) else model)
        self.opp_model = (torch.compile(opp_model)
                          if "cuda" in str(device) and opp_model is not None
                          else opp_model)
        self.max_wait = max_wait_ms / 1000.0
        self._pending: List[_Submission] = []
        self._cond = threading.Condition()
        self._stop = False
        self._opp_stream: Optional[torch.cuda.Stream] = (
            torch.cuda.Stream(device=device)
            if "cuda" in str(device) and opp_model is not None
            else None
        )
        self._thread = threading.Thread(
            target=self._loop, name="batch-coord", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def submit_and_wait(self, actor_id: int,
                        gpu_window: Dict[str, torch.Tensor],
                        opp_window: Optional[Dict[str, torch.Tensor]]):
        """Called from an actor thread. Blocks until the per-actor
        outputs are ready, then returns `(theta_logits, opp_logits)`.
        `opp_logits` is None when `opp_window` is None."""
        sub = _Submission(actor_id, gpu_window, opp_window)
        with self._cond:
            self._pending.append(sub)
            self._cond.notify_all()
        sub.event.wait()
        return sub.theta, sub.opp

    def _loop(self) -> None:
        while True:
            with self._cond:
                # Wait for the first submission to arrive.
                while not self._pending and not self._stop:
                    self._cond.wait()
                if self._stop:
                    for sub in self._pending:
                        sub.event.set()
                    return

                # Wait for batch to fill (N) OR for max_wait timeout.
                t_first = time.monotonic()
                while (len(self._pending) < self.n
                       and not self._stop):
                    remaining = self.max_wait - (time.monotonic() - t_first)
                    if remaining <= 0:
                        break
                    self._cond.wait(timeout=remaining)
                if self._stop:
                    for sub in self._pending:
                        sub.event.set()
                    return

                subs = self._pending
                self._pending = []

            self._run_batch(subs)

    def _run_batch(self, subs: List[_Submission]) -> None:
        """Stack inputs, run forwards, distribute outputs back per-actor."""
        # Trainee batch is always non-empty (caller never submits None
        # gpu_window). Opp batch only includes actors with opp configured.
        trainee_inputs = [s.gpu_window for s in subs]
        trainee_batch = {
            k: torch.stack([t[k] for t in trainee_inputs], dim=0)
            for k in trainee_inputs[0]
        }
        opp_subs = [s for s in subs if s.opp_window is not None]
        opp_batch = None
        if opp_subs:
            opp_inputs = [s.opp_window for s in opp_subs]
            opp_batch = {
                k: torch.stack([t[k] for t in opp_inputs], dim=0)
                for k in opp_inputs[0]
            }

        with torch.no_grad():
            opp_out = None
            if opp_batch is not None and self._opp_stream is not None:
                self._opp_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self._opp_stream):
                    opp_out = self.opp_model(opp_batch)
            trainee_out = self.model(trainee_batch)
            if opp_batch is not None and self._opp_stream is None:
                opp_out = self.opp_model(opp_batch)
            if opp_batch is not None and self._opp_stream is not None:
                torch.cuda.current_stream().wait_stream(self._opp_stream)

        # Distribute trainee outputs per-actor.
        for i, sub in enumerate(subs):
            sub.theta = {k: v[i:i + 1] for k, v in trainee_out.items()}
        # Distribute opp outputs per-opp-actor.
        if opp_out is not None:
            for j, sub in enumerate(opp_subs):
                sub.opp = {k: v[j:j + 1] for k, v in opp_out.items()}
        for sub in subs:
            sub.event.set()


class ActorPool:
    """N parallel DolphinActor instances sharing model + opp_model + a
    BatchCoordinator. Exposes the same `collect(n_episodes)` /
    `start()` / `stop()` / `start_keepalive()` / `stop_keepalive()`
    interface as DolphinActor, so loop.py can use either transparently.
    """

    def __init__(
        self,
        n: int,
        cfg: ActorConfig,
        task: OnlineTask,
        model,
        ref_model,
        ctx: dict,
        device: str,
        opp_model,
        opp_max_seq_len: int,
        opp_ctx: dict,
        opp_n_btn: int,
        model_seq_len: int = 256,
    ):
        self.n = n
        self.coord = BatchCoordinator(n, model, opp_model, device)
        self.actors: List[DolphinActor] = []
        for i in range(n):
            # Per-actor cfg with distinct actor_id (for log prefix).
            acfg = copy(cfg)
            acfg.actor_id = i
            actor = DolphinActor(
                cfg=acfg, task=task, model=model, ref_model=ref_model,
                ctx=ctx, device=device, model_seq_len=model_seq_len,
                self_port=1,
                injected_opp_model=opp_model,
                injected_opp_max_seq_len=opp_max_seq_len,
                injected_opp_ctx=opp_ctx,
                injected_opp_n_btn=opp_n_btn,
                coordinator=self.coord,
            )
            self.actors.append(actor)

        self._episodes_lock = threading.Lock()
        self._episodes: List[Episode] = []
        self._global_quota = 0
        self._stop_collect = threading.Event()

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> None:
        self.coord.start()
        # Start Dolphins sequentially — racing N libmelee Console.run()
        # calls causes random EXI socket collisions on the same machine.
        for a in self.actors:
            a.start()

    def stop(self) -> None:
        self._stop_collect.set()
        for a in self.actors:
            try:
                a.stop()
            except Exception:
                pass
        self.coord.stop()

    def start_keepalive(self) -> None:
        for a in self.actors:
            a.start_keepalive()

    def stop_keepalive(self) -> None:
        for a in self.actors:
            a.stop_keepalive()

    # -- collect ------------------------------------------------------------
    def collect(self, n_episodes: int) -> List[Episode]:
        """Run N actor threads concurrently; each contributes episodes
        to a shared list until the GLOBAL quota is met. Returns the
        accumulated list."""
        self._global_quota = n_episodes
        self._episodes = []
        self._stop_collect.clear()
        threads: List[threading.Thread] = []
        for i, a in enumerate(self.actors):
            t = threading.Thread(
                target=self._actor_collect_loop, args=(a, i),
                name=f"actor-{i}", daemon=True,
            )
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        with self._episodes_lock:
            return list(self._episodes)

    def _actor_collect_loop(self, actor: DolphinActor, actor_id: int) -> None:
        """One thread per actor: drive _step_one_frame, detect match
        endings + episode-cap boundaries, push completed episodes into
        the shared list, exit when the global quota is met."""
        per_ep = (28800 if actor.cfg.whole_match_episode
                  else actor.cfg.max_episode_frames)
        max_steps = self._global_quota * per_ep + 60 * 60 * 60
        steps = 0
        while steps < max_steps and not self._stop_collect.is_set():
            with self._episodes_lock:
                if len(self._episodes) >= self._global_quota:
                    return
            was_in_game = actor._in_game
            steps += 1
            try:
                actor._step_one_frame()
            except Exception as exc:
                log.error("actor %d crashed in _step_one_frame: %s",
                          actor_id, exc, exc_info=True)
                self._stop_collect.set()
                return

            # Match end: finalize buffered episodes.
            if was_in_game and not actor._in_game:
                finalized = actor._finalize_match_episodes()
                if finalized:
                    with self._episodes_lock:
                        self._episodes.extend(finalized)

            # Scenario-task episode boundary (only if open).
            if actor._episode_open_idx is not None and actor._in_game:
                cap_hit = (
                    not actor.cfg.whole_match_episode
                    and len(actor._pending) >= actor.cfg.max_episode_frames
                )
                if cap_hit or actor.task.should_end(
                        actor._state_history, actor._episode_open_idx):
                    if cap_hit:
                        log.warning(
                            "actor=%d episode hit max_episode_frames=%d cap; "
                            "scoring at cap", actor_id,
                            actor.cfg.max_episode_frames)
                    actor._score_and_close_open_episode()

        if not self._stop_collect.is_set():
            log.warning("actor=%d hit max_steps=%d before global quota",
                        actor_id, max_steps)
