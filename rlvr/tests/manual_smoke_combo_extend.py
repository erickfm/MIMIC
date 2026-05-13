"""Manual smoke test for ComboExtendOnlineTask against shard data.

Loads ~10 games from data/fox_all_v2/train_shard_000.pt, denormalizes
the relevant PlayerState fields back to raw values, reconstructs
GameState histories per-game, streams them through the task's
should_start / should_end / compute_outcome state machine, and reports
distributions.

This is not a pytest test — it requires a 800MB shard on disk and
takes ~10 seconds. Run manually:
    python -m rlvr.tests.manual_smoke_combo_extend

Pass criteria:
  - Episodes detected per game in a reasonable range (5-50 per game
    for a 30-second-to-2-minute Melee match).
  - No NaN terminal rewards.
  - Episode-length distribution heavy on the short side (5-60 frames)
    with a long tail (combos up to a few seconds).
  - Reward distribution: lots near 0 (sub-threshold + 0-damage),
    some at intermediate values (real combos), some at 1.0 (stock
    confirms + 100%+ damage).

This validates the task's logic against realistic human-play state
trajectories before we put it in front of Dolphin / a training loop.
"""
from __future__ import annotations

import collections
import statistics
from pathlib import Path

import torch

from rlvr.online.tasks.combo_extend_online import ComboExtendOnlineTask
from rlvr.state.gamestate import ControllerInput, GameState, PlayerState


# Normalization params from data/fox_all_v2/mimic_norm.json (see
# docs/research-notes-2026-05-13.md for the gotcha — percent uses
# normalize, not standardize).
PCT_MIN, PCT_MAX = 0.0, 343.4003601074219
STOCK_MIN, STOCK_MAX = 0.0, 4.0
HITSTUN_MAX = 120.0  # log_max(120) transform


def _denorm_normalize(norm: float, lo: float, hi: float) -> float:
    """Invert the 'normalize' transform: norm = 2*(raw-lo)/(hi-lo) - 1."""
    return (norm + 1.0) * (hi - lo) / 2.0 + lo


def _denorm_log_max(norm: float, mx: float) -> float:
    """Invert the 'log_max' transform: norm = log1p(clip(raw, 0, mx))/log1p(mx)."""
    import math
    if norm <= 0:
        return 0.0
    raw = math.expm1(norm * math.log1p(mx))
    return min(raw, mx)


# Shard column indices (mimic/features.py:numeric_state full schema).
COL_PERCENT = 2
COL_STOCK = 3
COL_HITSTUN = 11


def build_player_state(port: int, action: int, percent_norm: float,
                       stock_norm: float, hitstun_norm: float) -> PlayerState:
    """Construct a minimal PlayerState with the fields the task reads."""
    percent_raw = _denorm_normalize(percent_norm, PCT_MIN, PCT_MAX)
    stock_raw = int(round(_denorm_normalize(stock_norm, STOCK_MIN, STOCK_MAX)))
    hitstun_raw = _denorm_log_max(hitstun_norm, HITSTUN_MAX)
    return PlayerState(
        character=1,  # fox; not used by task
        port=port,
        position_x=0.0, position_y=0.0,
        percent=percent_raw,
        stock=stock_raw,
        jumps_left=2,
        speed_air_x_self=0.0, speed_ground_x_self=0.0,
        speed_x_attack=0.0, speed_y_attack=0.0, speed_y_self=0.0,
        hitlag_left=0.0,
        hitstun_frames_left=hitstun_raw,
        shield_strength=60.0,
        on_ground=True, off_stage=False, facing=True,
        invulnerable=False, moonwalkwarning=False,
        action=action,
        l_cancel=0,
        controller=ControllerInput.neutral(),
    )


def reconstruct_history(shard_states, start: int, end: int) -> list:
    """Yield GameState objects, frame by frame, for game [start, end)."""
    # Self is port 1, opp is port 2 (matches sorted-by-port convention).
    self_action = shard_states["self_action"][start:end].tolist()
    opp_action = shard_states["opp_action"][start:end].tolist()
    self_num = shard_states["self_numeric"][start:end].numpy()
    opp_num = shard_states["opp_numeric"][start:end].numpy()
    history = []
    for f in range(end - start):
        self_ps = build_player_state(
            port=1,
            action=int(self_action[f]),
            percent_norm=float(self_num[f, COL_PERCENT]),
            stock_norm=float(self_num[f, COL_STOCK]),
            hitstun_norm=float(self_num[f, COL_HITSTUN]),
        )
        opp_ps = build_player_state(
            port=2,
            action=int(opp_action[f]),
            percent_norm=float(opp_num[f, COL_PERCENT]),
            stock_norm=float(opp_num[f, COL_STOCK]),
            hitstun_norm=float(opp_num[f, COL_HITSTUN]),
        )
        history.append(GameState(
            schema_version="v0.1",
            frame_idx=f,
            stage=32,  # FD as a placeholder; not used by task
            players=(self_ps, opp_ps),
        ))
    return history


def run_task_on_history(task: ComboExtendOnlineTask, history: list) -> list:
    """Stream history through the task; return list of (start, end, outcome).

    Tracks episode length via a per-episode frame counter (NOT via the
    deque index, since the deque has finite maxlen and the start-frame
    can slide off).
    """
    state_history = collections.deque(maxlen=256)
    episode_open_idx = None
    episode_start_frame = None
    episode_frame_count = 0
    episodes = []

    for i, gs in enumerate(history):
        state_history.append(gs)

        if episode_open_idx is None:
            if task.should_start(state_history):
                episode_open_idx = len(state_history) - 1
                episode_start_frame = i
                episode_frame_count = 1
        else:
            episode_frame_count += 1
            if task.should_end(state_history, episode_open_idx):
                outcome = task.compute_outcome(state_history, episode_open_idx)
                episodes.append({
                    "start_frame": episode_start_frame,
                    "end_frame": i,
                    "length": episode_frame_count,
                    "terminal_reward": outcome.terminal_reward,
                    "result": outcome.metadata.get("result", "?"),
                    "damage": outcome.metadata.get("damage", None),
                })
                episode_open_idx = None
                episode_start_frame = None
                episode_frame_count = 0
    return episodes


def main():
    shard_path = Path("data/fox_all_v2/train_shard_000.pt")
    if not shard_path.exists():
        raise SystemExit(f"shard not found: {shard_path}")
    print(f"loading {shard_path}...")
    shard = torch.load(shard_path, map_location="cpu",
                       weights_only=False, mmap=True)
    states = shard["states"]
    offsets = shard["offsets"]
    n_games = int(shard["n_games"])

    n_games_to_scan = min(10, n_games)
    print(f"scanning first {n_games_to_scan} games...")

    task = ComboExtendOnlineTask(self_port=1)
    all_episodes = []
    per_game_counts = []

    for g in range(n_games_to_scan):
        s, e = int(offsets[g]), int(offsets[g + 1])
        history = reconstruct_history(states, s, e)
        eps = run_task_on_history(task, history)
        all_episodes.extend(eps)
        per_game_counts.append(len(eps))
        print(f"  game {g}: {e-s} frames, {len(eps)} episodes detected")

    print()
    print(f"=== Summary across {n_games_to_scan} games ===")
    print(f"total episodes: {len(all_episodes)}")
    if not all_episodes:
        print("NO EPISODES — task isn't firing. Bug.")
        return
    print(f"episodes per game: min={min(per_game_counts)}  "
          f"median={statistics.median(per_game_counts)}  "
          f"max={max(per_game_counts)}")
    print()

    # Episode length distribution.
    lengths = [ep["length"] for ep in all_episodes]
    print(f"episode length frames:")
    print(f"  min={min(lengths)}  median={statistics.median(lengths)}  "
          f"max={max(lengths)}  mean={statistics.mean(lengths):.1f}")
    # Histogram by quartile.
    sorted_lens = sorted(lengths)
    q = [sorted_lens[int(len(sorted_lens) * x)]
         for x in (0.25, 0.5, 0.75, 0.9, 0.99)]
    print(f"  quartiles 25/50/75/90/99: {q}")

    # Terminal reward distribution.
    rewards = [ep["terminal_reward"] for ep in all_episodes]
    nan_count = sum(1 for r in rewards if r != r)
    print()
    print(f"terminal_reward:")
    print(f"  min={min(rewards):.3f}  max={max(rewards):.3f}  "
          f"mean={statistics.mean(rewards):.3f}")
    print(f"  NaN count: {nan_count} (must be 0)")
    # Bucket: zero / small (0.01-0.4) / large (0.4-0.99) / max (1.0).
    n_zero = sum(1 for r in rewards if r == 0.0)
    n_small = sum(1 for r in rewards if 0 < r < 0.4)
    n_large = sum(1 for r in rewards if 0.4 <= r < 0.99)
    n_max = sum(1 for r in rewards if r >= 0.99)
    print(f"  bucket counts:")
    print(f"    reward = 0     : {n_zero}  (sub-threshold / aborted)")
    print(f"    0 < r < 0.4    : {n_small}")
    print(f"    0.4 <= r < 0.99: {n_large}")
    print(f"    r >= 0.99      : {n_max}  (stock confirms + 100%+ damage)")

    # Result-type distribution.
    print()
    result_counts = collections.Counter(ep["result"] for ep in all_episodes)
    print(f"result types:")
    for k, v in result_counts.most_common():
        print(f"  {k}: {v}")

    # Sanity checks.
    print()
    print(f"=== Pass criteria ===")
    p1 = nan_count == 0
    p2 = max(lengths) < 600   # no 600-frame runaway
    p3 = min(lengths) >= 1    # no zero-length
    p4 = len(all_episodes) >= 10 * n_games_to_scan / 5  # >= 2 ep/game avg
    print(f"  no NaN rewards: {'PASS' if p1 else 'FAIL'}")
    print(f"  no runaway episodes (max len < 600): "
          f"{'PASS' if p2 else f'FAIL (max={max(lengths)})'}")
    print(f"  no zero-length episodes: "
          f"{'PASS' if p3 else f'FAIL (min={min(lengths)})'}")
    print(f"  at least ~2 ep/game on average: "
          f"{'PASS' if p4 else f'FAIL ({len(all_episodes)}/{n_games_to_scan} = {len(all_episodes)/n_games_to_scan:.1f})'}")


if __name__ == "__main__":
    main()
