"""Savestate drill trainer (L-cancel RLVR drilling, piece 2 of 2).

Consumes the state library written by rlvr/online/miss_harvest.py. For
each library state per update:

  1. LOADSTATE it over the bot controller pipe (patched emulator_ss
     Dolphin), detect the rewind via a frame-counter discontinuity
     (skip_rollback_frames=False makes it observable).
  2. Restore the policy's harvested context window + prev controller
     state into the _PolicyRunner.
  3. ~10 neutral-input warmup frames (the game settles post-load).
  4. Policy control (sampling, T=1.0) for up to ~240 frames; score the
     FIRST completed aerial landing LIVE with the avoidable-lag rule.
     No landing in the window, or a damage/dead-exit landing -> drop.
  5. N rollouts per state (reload each time) = one matched-context GRPO
     group; advantage = reward - group mean; zero-variance groups are
     skipped (no signal).

Episodes from all non-degenerate groups feed ppo_update with
OnlinePPOConfig.use_metadata_advantage=True (per-episode precomputed
advantages, bypassing the global z-scoring) and KL to a frozen ref.

Rewards are computed from LIVE state only — never from .slp written
across loads (duplicate frame spans corrupt replay-based scoring).

Usage:
  python3 -m rlvr.online.drill_loop \
    --ckpt checkpoints/AVG_mastfox.pt --data-dir data/foxrank_master_v2 \
    --run-name fox-lcancel-drill-v1 --updates 3
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from rlvr.online.dolphin_actor import (
    _PolicyRunner,
    _press_controller,
    _sample_four_heads,
)
from rlvr.online.ppo import OnlinePPOConfig, ppo_update
from rlvr.online.savestate_util import (
    LandingRunTracker,
    SavestateSession,
    SessionConfig,
    neutral_prev_sent,
    press_neutral,
    restore_policy,
    send_loadstate,
)
from rlvr.online.trajectory import Episode, FrameRecord
from tools.inference_utils import build_frame, load_inference_context, load_mimic_model

log = logging.getLogger("rlvr.online.drill")

REPO = Path(__file__).resolve().parents[2]


@dataclass
class LibraryState:
    id: str
    sav_path: Path
    meta: dict
    snap: dict          # {"window": {...}, "prev_sent": {...}, "capture_frame": int}


def load_library(lib_dir: Path) -> List[LibraryState]:
    states = []
    for sidecar in sorted(lib_dir.glob("*.json")):
        sid = sidecar.stem
        sav = lib_dir / f"{sid}.sav"
        ctxpt = lib_dir / f"{sid}.ctx.pt"
        if not sav.exists() or not ctxpt.exists():
            log.warning("library state %s incomplete (missing .sav/.ctx.pt); "
                        "skipping", sid)
            continue
        meta = json.loads(sidecar.read_text())
        snap = torch.load(ctxpt, map_location="cpu", weights_only=False)
        states.append(LibraryState(id=sid, sav_path=sav, meta=meta, snap=snap))
    return states


class DrillRunner:
    """Owns the Dolphin session + policy runner and produces drill
    episodes from library states."""

    def __init__(self, session: SavestateSession, policy: _PolicyRunner,
                 ctx: dict, device: str, temperature: float = 1.0,
                 warmup_frames: int = 10, max_control_frames: int = 240,
                 load_timeout_s: float = 30.0):
        self.session = session
        self.policy = policy
        self.ctx = ctx
        self.device = device
        self.temperature = temperature
        self.warmup_frames = warmup_frames
        self.max_control_frames = max_control_frames
        self.load_timeout_s = load_timeout_s
        self.tracker = LandingRunTracker()
        # instrumentation
        self.load_latencies_s: List[float] = []
        self.load_frame_deltas: List[int] = []
        self.rollout_walls: List[float] = []

    # -- match management -----------------------------------------------------

    def ensure_in_game(self, min_frames: int = 60) -> None:
        """Step until we are in a live match (menu-navigating as needed)
        and it has been running for min_frames."""
        count = 0
        while True:
            gs = self.session.console.step()
            if gs is None:
                continue
            if self.session.in_game(gs):
                count += 1
                press_neutral(self.session.ego_ctrl)
                if count >= min_frames:
                    return
            else:
                count = 0
                self.session.menu_frame(gs)

    # -- one drill rollout ------------------------------------------------------

    def rollout(self, state: LibraryState) -> Tuple[Optional[Episode], str]:
        """LOADSTATE -> restore context -> warmup -> policy control ->
        live-score first landing. Returns (episode | None, reason)."""
        t_start = time.time()
        send_loadstate(self.session.ego_ctrl, str(state.sav_path))
        t_sent = time.time()
        capture_frame = int(state.snap.get("capture_frame",
                                           state.meta["capture_frame"]))

        phase = "wait"
        prev_frame: Optional[int] = None
        menu_steps = 0
        warmup_left = self.warmup_frames
        prev = None                     # prev_sent for warmup build_frame
        records: List[FrameRecord] = []
        landed = False
        control_steps = 0
        tail_left = 90

        while True:
            gs = self.session.console.step()
            if gs is None:
                continue
            if not self.session.in_game(gs):
                if phase == "wait":
                    # Load may still fire and yank us back in-game; don't
                    # navigate menus (that could start a NEW match under
                    # the pending host job). Just wait, bounded.
                    menu_steps += 1
                    if menu_steps > 2000:
                        return None, "menu_stuck"
                    continue
                return None, "match_ended"

            f = int(gs.frame)

            if phase == "wait":
                if time.time() - t_sent > self.load_timeout_s:
                    return None, "load_timeout"
                if prev_frame is None:
                    prev_frame = f
                    press_neutral(self.session.ego_ctrl)
                    continue
                jump = f - prev_frame
                prev_frame = f
                if 0 <= jump <= 5:
                    # normal frame advance (or a rare dropped frame)
                    press_neutral(self.session.ego_ctrl)
                    continue
                # Discontinuity = the load landed. Measure + restore.
                self.load_latencies_s.append(time.time() - t_sent)
                self.load_frame_deltas.append(f - capture_frame)
                restore_policy(self.policy, state.snap, self.device)
                self.tracker.reset()
                prev = (dict(self.policy.prev_sent)
                        if self.policy.prev_sent else neutral_prev_sent())
                phase = "warmup"
                # fall through: this frame is the first warmup frame

            if phase == "warmup":
                frame = build_frame(gs, prev, self.ctx)
                if frame is not None:
                    self.policy.push_frame(frame)
                press_neutral(self.session.ego_ctrl)
                prev = neutral_prev_sent()
                ps = gs.players.get(1)
                if ps is not None:
                    self.tracker.push(f, int(ps.action.value))
                warmup_left -= 1
                if warmup_left <= 0:
                    if self.tracker.open:
                        # A landing began under neutral control — not
                        # attributable to the policy. Drop.
                        return None, "landing_during_warmup"
                    self.policy.prev_sent = prev
                    phase = "control"
                continue

            # -- control / tail: policy drives every frame -------------------
            frame = build_frame(gs, self.policy.prev_sent, self.ctx)
            if frame is None:
                continue
            self.policy.push_frame(frame)
            with torch.no_grad():
                logits = self.policy.forward_latest()
            (m_i, s_i, c_i, b_i), lp_old = _sample_four_heads(
                logits, self.temperature)
            n_btn = int(logits["btn_logits"].shape[-1])
            self.policy.prev_sent = _press_controller(
                self.session.ego_ctrl, m_i, s_i, c_i, b_i, n_btn)

            if not landed:
                # Record until (and including) the landing-entry frame.
                obs = {k: v.clone()
                       for k, v in self.policy._cpu_window.items()}
                records.append(FrameRecord(
                    obs=obs,
                    sampled_indices=torch.tensor(
                        [m_i, s_i, c_i, b_i], dtype=torch.long),
                    logprob_old=torch.tensor(lp_old, dtype=torch.float32),
                    logprob_ref=torch.tensor(lp_old, dtype=torch.float32),
                    reward=0.0,
                    game_frame_id=f,
                ))

            ps = gs.players.get(1)
            if ps is None:
                continue
            run = self.tracker.push(f, int(ps.action.value))

            if run is not None:
                # First completed landing decides the rollout.
                if not run.scoreable:
                    return None, "opponent_interrupted"
                self.rollout_walls.append(time.time() - t_start)
                ep = Episode(
                    task_id="l_cancel_drill",
                    frames=records,
                    terminal_reward=run.reward,
                    start_game_frame=records[0].game_frame_id if records else f,
                    end_game_frame=f,
                    metadata={
                        "state_id": state.id,
                        "result": ("l_cancel_success" if run.reward > 0
                                   else "l_cancel_missed"),
                        "landing_state": run.landing_state,
                        "move": run.move,
                        "realized_lag": run.length,
                        "avoidable_lag": run.avoidable_lag,
                    },
                )
                return ep, "scored"

            if self.tracker.open and not landed:
                landed = True     # stop recording; keep stepping to score

            if landed:
                tail_left -= 1
                if tail_left <= 0:
                    return None, "landing_run_never_closed"
            else:
                control_steps += 1
                if control_steps >= self.max_control_frames:
                    return None, "no_landing"

    # -- sanity match -----------------------------------------------------------

    def sanity_match(self) -> dict:
        """Play ONE full normal match (no loads) under policy control and
        report length + live L-cancel stats. Confirms the policy still
        plays after drill updates."""
        # Finish whatever match state we're in first.
        gs = self.session.console.step()
        while gs is None or self.session.in_game(gs):
            if gs is not None:
                # neutral through the residual match is slow; drive with
                # the policy so it ends at a natural pace
                self._policy_frame(gs)
            gs = self.session.console.step()
        # Now in menus: start a fresh match and play it out.
        self.tracker.reset()
        stats = {"frames": 0, "landings": 0, "misses": 0}
        in_game = False
        while True:
            if gs is None:
                gs = self.session.console.step()
                continue
            if not self.session.in_game(gs):
                if in_game:
                    break  # the fresh match ended
                self.session.menu_frame(gs)
                gs = self.session.console.step()
                continue
            in_game = True
            stats["frames"] += 1
            run = self._policy_frame(gs)
            if run is not None and run.scoreable:
                stats["landings"] += 1
                if run.avoidable_lag > 0:
                    stats["misses"] += 1
            gs = self.session.console.step()
        return stats

    def _policy_frame(self, gs):
        frame = build_frame(gs, self.policy.prev_sent, self.ctx)
        if frame is None:
            return None
        self.policy.push_frame(frame)
        with torch.no_grad():
            logits = self.policy.forward_latest()
        idx, _ = _sample_four_heads(logits, self.temperature)
        n_btn = int(logits["btn_logits"].shape[-1])
        self.policy.prev_sent = _press_controller(
            self.session.ego_ctrl, *idx, n_btn)
        ps = gs.players.get(1)
        if ps is None:
            return None
        return self.tracker.push(int(gs.frame), int(ps.action.value))


def train(args) -> None:
    device = args.device
    library = load_library(Path(args.library))
    if not library:
        raise SystemExit(f"no library states in {args.library} — run "
                         "rlvr.online.miss_harvest first")
    log.info("library: %d states (%s)", len(library),
             Counter(s.meta["move"] for s in library))

    model, mcfg = load_mimic_model(str(args.ckpt), device)
    model.eval()   # deterministic re-forward; does not block gradients
    ref_path = args.ref_ckpt or args.ckpt
    ref_model, _ = load_mimic_model(str(ref_path), device)
    for p in ref_model.parameters():
        p.requires_grad_(False)
    ref_model.eval()
    from dataclasses import asdict
    try:
        model_cfg_snapshot = asdict(mcfg)
    except TypeError:
        model_cfg_snapshot = mcfg

    ctx = load_inference_context(str(args.data_dir))
    policy = _PolicyRunner(model, mcfg.max_seq_len, device, ctx)

    session = SavestateSession(SessionConfig(
        dolphin_path=str(args.dolphin_path), iso_path=str(args.iso_path),
        slippi_port=args.slippi_port, cpu_level=args.cpu_level,
        enable_ffw=not args.no_ffw, gfx_backend=args.gfx_backend,
    ))
    session.start()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ppo_cfg = OnlinePPOConfig(
        clip_eps=args.clip_eps, kl_beta=args.kl_beta,
        use_metadata_advantage=True,
    )

    runner = DrillRunner(
        session, policy, ctx, device, temperature=args.temperature,
        warmup_frames=args.warmup_frames,
        max_control_frames=args.max_rollout_frames,
    )
    state_cycle = itertools.cycle(library)
    ckpt_dir = Path(args.checkpoint_dir)

    t0 = time.time()
    try:
        runner.ensure_in_game()
        for update in range(1, args.updates + 1):
            t_collect = time.time()
            all_eps: List[Episode] = []
            group_lines = []
            n_states = min(args.states_per_update, len(library))
            for _ in range(n_states):
                st = next(state_cycle)
                eps: List[Episode] = []
                drops: Counter = Counter()
                for _k in range(args.rollouts_per_state):
                    ep, why = runner.rollout(st)
                    if ep is not None:
                        eps.append(ep)
                    else:
                        drops[why] += 1
                        if why in ("match_ended", "menu_stuck"):
                            runner.ensure_in_game()
                rewards = [ep.terminal_reward for ep in eps]
                mean_r = sum(rewards) / len(rewards) if rewards else float("nan")
                degenerate = (len(eps) < 2
                              or sum(rewards) in (0.0, float(len(rewards))))
                group_lines.append(
                    f"{st.id[-28:]}: n={len(eps)} mean={mean_r:.2f} "
                    f"rewards={[int(r) for r in rewards]} "
                    f"drops={dict(drops)}{' [skipped: degenerate]' if degenerate else ''}")
                if not degenerate:
                    for ep in eps:
                        ep.metadata["advantage"] = ep.terminal_reward - mean_r
                    all_eps.extend(eps)
            t_collect = time.time() - t_collect

            for line in group_lines:
                log.info("EVT_GROUP update=%d %s", update, line)
            if not all_eps:
                log.warning("update %d: every group degenerate/empty; "
                            "no gradient step", update)
                continue

            session.start_keepalive()
            try:
                t_ppo = time.time()
                for _epoch in range(max(1, args.ppo_epochs)):
                    metrics = ppo_update(model, all_eps, optimizer, ppo_cfg,
                                         device=device, ref_model=ref_model)
                t_ppo = time.time() - t_ppo
            finally:
                session.stop_keepalive()

            log.info(
                "update=%d groups_used=%d eps=%d frames=%d kl=%.4f "
                "clip_frac=%.2f loss=%.4f t_collect=%.1fs t_ppo=%.1fs",
                update, len({ep.metadata["state_id"] for ep in all_eps}),
                len(all_eps), metrics["n_frames"], metrics["kl"],
                metrics["clip_frac"], metrics["loss"], t_collect, t_ppo)

            if args.checkpoint_every > 0 and update % args.checkpoint_every == 0:
                from rlvr.online.loop import _save_ckpt
                ck = ckpt_dir / f"{args.run_name}_update{update:04d}.pt"
                _save_ckpt(ck, model, optimizer, model_cfg_snapshot, update,
                           "l_cancel_drill")
                log.info("saved %s", ck)

        # Final checkpoint.
        from rlvr.online.loop import _save_ckpt
        final = ckpt_dir / f"{args.run_name}_final.pt"
        _save_ckpt(final, model, optimizer, model_cfg_snapshot, args.updates,
                   "l_cancel_drill")
        log.info("final: %s", final)

        # Instrumentation summary.
        lat = sorted(runner.load_latencies_s)
        deltas = sorted(runner.load_frame_deltas)
        walls = sorted(runner.rollout_walls)
        def _p(xs, q):
            return xs[min(len(xs) - 1, int(len(xs) * q))] if xs else None
        log.info("load latency (s): p50=%s p95=%s n=%d | rewind frame delta "
                 "vs sidecar: p50=%s p95=%s | scored-rollout wall (s): "
                 "p50=%s p95=%s n=%d",
                 _p(lat, .5), _p(lat, .95), len(lat),
                 _p(deltas, .5), _p(deltas, .95),
                 _p(walls, .5), _p(walls, .95), len(walls))

        if args.sanity_match:
            log.info("running post-drill sanity match (policy vs CPU)...")
            stats = runner.sanity_match()
            log.info("sanity match: %s", stats)
    finally:
        session.stop()
    log.info("done in %.1fs", time.time() - t0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", default=str(REPO / "checkpoints/AVG_mastfox.pt"))
    ap.add_argument("--ref-ckpt", default=None,
                    help="Frozen KL reference (default: --ckpt).")
    ap.add_argument("--data-dir", default=str(REPO / "data/foxrank_master_v2"))
    ap.add_argument("--library", default=str(REPO / "states/lcancel_misses"))
    ap.add_argument("--dolphin-path",
                    default=str(REPO / "emulator_ss/Binaries/dolphin-emu"))
    ap.add_argument("--iso-path", default=str(REPO / "melee.iso"))
    ap.add_argument("--run-name", default="fox-lcancel-drill-v1")
    ap.add_argument("--updates", type=int, default=3)
    ap.add_argument("--states-per-update", type=int, default=4)
    ap.add_argument("--rollouts-per-state", type=int, default=8,
                    help="N reloads per state = one matched-context group.")
    # Tuned recipe from the 2026-07-14 search (loop.py docstring).
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--kl-beta", type=float, default=0.003)
    ap.add_argument("--ppo-epochs", type=int, default=4)
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--warmup-frames", type=int, default=10)
    ap.add_argument("--max-rollout-frames", type=int, default=240)
    ap.add_argument("--cpu-level", type=int, default=9)
    ap.add_argument("--checkpoint-dir", default=str(REPO / "checkpoints"))
    ap.add_argument("--checkpoint-every", type=int, default=0,
                    help="0 = only save the final checkpoint.")
    ap.add_argument("--slippi-port", type=int, default=52110,
                    help="52100-52199 only; distinct per concurrent instance.")
    ap.add_argument("--gfx-backend", default="Null")
    ap.add_argument("--no-ffw", action="store_true")
    ap.add_argument("--sanity-match", action="store_true",
                    help="After training, play one full normal match and "
                         "report length + live L-cancel stats.")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  [%(levelname)s]  %(message)s")
    train(args)


if __name__ == "__main__":
    main()
