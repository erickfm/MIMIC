"""Miss-targeted savestate harvester (L-cancel RLVR drilling, piece 1 of 2).

Runs FFW bot-vs-CPU rollouts with the policy on port 1 and detects
L-cancel misses LIVE (avoidable-lag rule, no .slp involved). While the
match runs, it maintains a rolling ring of Dolphin savestates (SAVESTATE
pipe verb every ~120 frames to N rotating slot files). When a miss is
detected, the ring slot captured comfortably BEFORE the landing
(>= --margin-frames, default 180) is copied into the state library along
with:

  <id>.sav      the Dolphin savestate
  <id>.ctx.pt   torch.save of the policy's context window + prev_sent
                at the (approximate) savestate capture moment
  <id>.json     metadata sidecar (capture/landing frames, move, lag, ...)

Save-latency compensation: the .sav lands 1-150 game frames after the
pipe send (host-job dispatch, worse under FFW). We poll for the slot
file's appearance each frame and stamp capture_frame + snapshot the
policy context at first appearance — within a few frames of the true
save point. drill_loop.py measures the residual delta (observed rewind
frame vs sidecar capture_frame) at load time.

Usage (defaults match the L-cancel drilling setup):
  python3 -m rlvr.online.miss_harvest \
    --ckpt checkpoints/AVG_mastfox.pt --data-dir data/foxrank_master_v2 \
    --max-misses 10
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import melee
import torch

from rlvr.online.dolphin_actor import (
    _PolicyRunner,
    _press_controller,
    _sample_four_heads,
)
from rlvr.online.savestate_util import (
    LandingRunTracker,
    SavestateSession,
    SessionConfig,
    send_savestate,
    snapshot_policy,
)
from tools.inference_utils import build_frame, load_inference_context, load_mimic_model

log = logging.getLogger("rlvr.online.miss_harvest")

REPO = Path(__file__).resolve().parents[2]


@dataclass
class RingEntry:
    slot: int
    path: str
    capture_frame: int          # game frame at .sav file appearance (~save frame)
    sent_frame: int             # game frame the SAVESTATE verb was sent
    snap: dict                  # snapshot_policy() at capture


class SavestateRing:
    """Rolling ring of savestate slot files inside one match.

    send() starts a save into the next slot (invalidating its old
    entry); poll() watches for the pending file to land and finalizes
    the entry with a policy-context snapshot."""

    def __init__(self, tmp_dir: Path, n_slots: int, interval: int):
        self.tmp_dir = tmp_dir
        self.n_slots = n_slots
        self.interval = interval
        self.entries: List[Optional[RingEntry]] = [None] * n_slots
        self._next_slot = 0
        self._pending: Optional[dict] = None
        self.latencies_frames: List[int] = []
        tmp_dir.mkdir(parents=True, exist_ok=True)

    def reset(self) -> None:
        """Match boundary: frame ids restart, all entries are stale."""
        self.entries = [None] * self.n_slots
        self._pending = None

    def maybe_send(self, ctrl, game_frame: int, in_game_step: int) -> None:
        if self._pending is not None:
            # Stale pending save (host job never fired?) — drop after 600f.
            if game_frame - self._pending["sent_frame"] > 600:
                log.warning("savestate to slot %d never landed (sent f%d); "
                            "dropping pending", self._pending["slot"],
                            self._pending["sent_frame"])
                self._pending = None
            return
        if in_game_step % self.interval != 0:
            return
        slot = self._next_slot
        self._next_slot = (self._next_slot + 1) % self.n_slots
        path = str(self.tmp_dir / f"ring_{slot}.sav")
        self.entries[slot] = None            # old file is about to be replaced
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        send_savestate(ctrl, path)
        self._pending = {"slot": slot, "path": path, "sent_frame": game_frame,
                         "sent_wall": time.time()}

    def poll(self, policy, game_frame: int) -> None:
        if self._pending is None:
            return
        p = self._pending
        if not os.path.exists(p["path"]):
            return
        # File appeared: the host job fired (game paused during the state
        # serialize, so the .sav corresponds to ~this frame; the disk
        # write may still be in flight but completes long before any
        # harvest copy, which happens >= margin_frames later).
        self.entries[p["slot"]] = RingEntry(
            slot=p["slot"], path=p["path"], capture_frame=game_frame,
            sent_frame=p["sent_frame"], snap=snapshot_policy(policy),
        )
        self.latencies_frames.append(game_frame - p["sent_frame"])
        self._pending = None

    def pick_for_landing(self, landing_frame: int, margin: int,
                         lookback_max: int = 700) -> Optional[RingEntry]:
        """Newest landed entry at least `margin` frames before the
        landing (and not older than lookback_max)."""
        best = None
        for e in self.entries:
            if e is None:
                continue
            age = landing_frame - e.capture_frame
            if margin <= age <= lookback_max:
                if best is None or e.capture_frame > best.capture_frame:
                    best = e
        return best


def harvest(args) -> dict:
    device = args.device
    model, mcfg = load_mimic_model(str(args.ckpt), device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    ctx = load_inference_context(str(args.data_dir))
    policy = _PolicyRunner(model, mcfg.max_seq_len, device, ctx)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_tag = time.strftime("%Y%m%d-%H%M%S")

    session = SavestateSession(SessionConfig(
        dolphin_path=str(args.dolphin_path), iso_path=str(args.iso_path),
        slippi_port=args.slippi_port, cpu_level=args.cpu_level,
        enable_ffw=not args.no_ffw, gfx_backend=args.gfx_backend,
    ))
    ring = SavestateRing(Path(args.tmp_dir) / f"ring_{args.slippi_port}",
                         n_slots=args.ring_slots, interval=args.save_interval)
    tracker = LandingRunTracker()

    stats = {"landings": 0, "misses": 0, "harvested": 0, "unscoreable": 0,
             "no_slot": 0, "matches": 0, "match_lengths": []}
    harvested_meta: List[dict] = []

    session.start()
    t0 = time.time()
    in_game = False
    in_game_step = 0
    match_start_frame = 0
    try:
        while (stats["harvested"] < args.max_misses
               and time.time() - t0 < args.max_minutes * 60):
            gs = session.console.step()
            if gs is None:
                continue
            if not session.in_game(gs):
                if in_game:
                    stats["matches"] += 1
                    mlen = in_game_step
                    stats["match_lengths"].append(mlen)
                    log.info("match %d ended: %d frames, harvested so far=%d "
                             "(landings=%d misses=%d)", stats["matches"], mlen,
                             stats["harvested"], stats["landings"],
                             stats["misses"])
                    in_game = False
                    in_game_step = 0
                    ring.reset()
                    tracker.reset()
                session.menu_frame(gs)
                continue

            if not in_game:
                in_game = True
                match_start_frame = int(gs.frame)
            in_game_step += 1
            game_frame = int(gs.frame)

            # Policy step: build -> push -> forward -> sample -> press.
            frame = build_frame(gs, policy.prev_sent, ctx)
            if frame is None:
                continue
            policy.push_frame(frame)
            with torch.no_grad():
                logits = policy.forward_latest()
            (m_i, s_i, c_i, b_i), _ = _sample_four_heads(
                logits, args.temperature)
            n_btn = int(logits["btn_logits"].shape[-1])
            policy.prev_sent = _press_controller(
                session.ego_ctrl, m_i, s_i, c_i, b_i, n_btn)

            # Ring maintenance.
            ring.poll(policy, game_frame)
            ring.maybe_send(session.ego_ctrl, game_frame, in_game_step)

            # Live miss detection.
            ps = gs.players.get(1)
            if ps is None:
                continue
            run = tracker.push(game_frame, int(ps.action.value))
            if run is None:
                continue
            if not run.scoreable:
                stats["unscoreable"] += 1
                continue
            stats["landings"] += 1
            if run.avoidable_lag == 0:
                continue
            # -- a miss --
            stats["misses"] += 1
            entry = ring.pick_for_landing(run.start_frame, args.margin_frames)
            if entry is None:
                stats["no_slot"] += 1
                log.info("miss at f%d (%s, lag %d) but no eligible ring slot "
                         "(too early in match?)", run.start_frame, run.move,
                         run.length)
                continue
            sid = (f"{run_tag}_m{stats['matches']:02d}"
                   f"_f{run.start_frame}_{run.move.lower()}")
            shutil.copyfile(entry.path, out_dir / f"{sid}.sav")
            torch.save({"window": entry.snap["window"],
                        "prev_sent": entry.snap["prev_sent"],
                        "capture_frame": entry.capture_frame},
                       out_dir / f"{sid}.ctx.pt")
            meta = {
                "id": sid,
                "capture_frame": entry.capture_frame,
                "save_sent_frame": entry.sent_frame,
                "landing_frame": run.start_frame,
                "landing_state": run.landing_state,
                "move": run.move,
                "realized_lag": run.length,
                "avoidable_lag": run.avoidable_lag,
                "exit_state": run.exit_state,
                "frames_before_landing": run.start_frame - entry.capture_frame,
                "match_idx": stats["matches"],
                "cpu_level": args.cpu_level,
                "character": session.cfg.character,
                "cpu_character": session.cfg.cpu_character,
                "stage": session.cfg.stage,
                "ckpt": str(args.ckpt),
                "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            (out_dir / f"{sid}.json").write_text(json.dumps(meta, indent=2))
            harvested_meta.append(meta)
            stats["harvested"] += 1
            log.info("HARVESTED %s: %s lag=%d avoidable=%d, state from "
                     "f%d (%d frames pre-landing) [%d/%d]",
                     sid, run.move, run.length, run.avoidable_lag,
                     entry.capture_frame, meta["frames_before_landing"],
                     stats["harvested"], args.max_misses)
    finally:
        session.stop()

    elapsed = time.time() - t0
    lat = sorted(ring.latencies_frames)
    stats["save_latency_frames_p50"] = lat[len(lat) // 2] if lat else None
    stats["save_latency_frames_p95"] = (
        lat[min(len(lat) - 1, int(len(lat) * 0.95))] if lat else None)
    stats["elapsed_s"] = round(elapsed, 1)
    stats["miss_rate"] = (round(stats["misses"] / stats["landings"], 4)
                          if stats["landings"] else None)
    log.info("harvest done: %s", json.dumps(
        {k: v for k, v in stats.items() if k != "match_lengths"}))
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", default=str(REPO / "checkpoints/AVG_mastfox.pt"))
    ap.add_argument("--data-dir", default=str(REPO / "data/foxrank_master_v2"))
    ap.add_argument("--dolphin-path",
                    default=str(REPO / "emulator_ss/Binaries/dolphin-emu"),
                    help="Patched savestate-capable Dolphin (emulator_ss).")
    ap.add_argument("--iso-path", default=str(REPO / "melee.iso"))
    ap.add_argument("--out-dir", default=str(REPO / "states/lcancel_misses"))
    ap.add_argument("--tmp-dir", default="/tmp/mimic_drill_ring")
    ap.add_argument("--max-misses", type=int, default=10,
                    help="Stop after harvesting this many miss states.")
    ap.add_argument("--max-minutes", type=float, default=90.0)
    ap.add_argument("--cpu-level", type=int, default=9)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--ring-slots", type=int, default=6)
    ap.add_argument("--save-interval", type=int, default=120,
                    help="Frames between rolling SAVESTATEs.")
    ap.add_argument("--margin-frames", type=int, default=180,
                    help="Harvested state must be at least this many frames "
                         "before the missed landing (covers the 1-150 frame "
                         "save-job latency with headroom).")
    ap.add_argument("--slippi-port", type=int, default=52100,
                    help="52100-52199 only; distinct per concurrent instance.")
    ap.add_argument("--gfx-backend", default="Null")
    ap.add_argument("--no-ffw", action="store_true")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  [%(levelname)s]  %(message)s")
    harvest(args)


if __name__ == "__main__":
    main()
