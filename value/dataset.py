"""Windowed dataset for V(s) over fox_all_v2 shards.

Each yielded item is (state_dict, outcome_scalar):
  - state_dict: each key is (W, ...) — a W-frame window of state.
    After DataLoader collation: (B, W, ...).
  - outcome_scalar: scalar in {0.0, 0.5, 1.0} — the game's outcome,
    derived from stock-drop-recency (one label per window since outcome
    is a game-level invariant within a single game).

The shard schema (fox_all_v2 / WM schema) provides the full feature set
ValueEncoder consumes. The naive last-frame stock comparison
mis-classifies ~50% of games as draws because trajectories truncate
before the loser's final death — see compute_game_outcomes below for the
recency-aware tiebreak.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import IterableDataset, get_worker_info


# self_numeric column index for stock (per mimic/features.py:numeric_state)
STOCK_COL = 3
PERCENT_COL = 2


def _last_stock_drop_index(stk: torch.Tensor) -> int:
    """Return index of last frame where normalized stock decreased, or -1."""
    if stk.numel() < 2:
        return -1
    diffs = stk[1:] - stk[:-1]
    drops = (diffs < -0.1).nonzero(as_tuple=True)[0]
    if drops.numel() == 0:
        return -1
    return int(drops[-1].item())


def compute_game_outcomes(
    self_stock_col: torch.Tensor,
    opp_stock_col: torch.Tensor,
    offsets: torch.Tensor,
    n_games: int,
    self_pct_col: torch.Tensor = None,
    opp_pct_col: torch.Tensor = None,
) -> List[Tuple[int, int, float]]:
    """Per-game (start, end, outcome) where outcome ∈ {0.0, 0.5, 1.0}.

    Outcome resolution:
      1. If final-frame stocks differ → higher wins.
      2. If equal (typical for truncated trajectories) → player whose last
         stock-drop is more recent in the trajectory is the loser.
      3. If no stock drops at all → percent tiebreak (lower wins).
      4. Otherwise 0.5.
    """
    out = []
    for g in range(n_games):
        s = int(offsets[g].item())
        e = int(offsets[g + 1].item())
        last = e - 1
        ss = self_stock_col[last].item()
        os_ = opp_stock_col[last].item()
        if ss > os_:
            outcome = 1.0
        elif ss < os_:
            outcome = 0.0
        else:
            s_drop = _last_stock_drop_index(self_stock_col[s:e])
            o_drop = _last_stock_drop_index(opp_stock_col[s:e])
            if s_drop > o_drop:
                outcome = 0.0
            elif s_drop < o_drop:
                outcome = 1.0
            elif self_pct_col is not None and opp_pct_col is not None:
                spc = self_pct_col[last].item()
                opc = opp_pct_col[last].item()
                if spc < opc:
                    outcome = 1.0
                elif spc > opc:
                    outcome = 0.0
                else:
                    outcome = 0.5
            else:
                outcome = 0.5
        out.append((s, e, outcome))
    return out


class FoxValueWindowedDataset(IterableDataset):
    """Windowed value-function dataset over fox_all_v2 shards.

    Args:
        data_dir: path to fox_all_v2/ with tensor_manifest.json
        split: 'train' or 'val'
        window: number of frames per yielded sample
        windows_per_shard_visit: random windows sampled per shard before
            moving on. Larger amortizes shard load cost.
        state_keys: required keys to slice from each shard. If a shard
            lacks a key, samples from that shard are skipped (warning).
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        window: int = 60,
        windows_per_shard_visit: int = 4096,
        rank: int = 0,
        world_size: int = 1,
        distributed: bool = True,
        seed: int = 0,
        state_keys: list = None,
        min_position: float = 0.0,
        max_position: float = 1.0,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.window = window
        self.windows_per_shard_visit = windows_per_shard_visit
        self._rank = rank
        self._world_size = world_size
        self._distributed = distributed
        self._seed = seed
        self._min_position = min_position
        self._max_position = max_position
        if state_keys is None:
            raise ValueError("state_keys must be provided (which shard keys to load)")
        self._state_keys = list(state_keys)

        with open(self.data_dir / "tensor_manifest.json") as f:
            manifest = json.load(f)
        key = "val_shards" if split == "val" else "train_shards"
        self.shards = [self.data_dir / n for n in manifest[key]]
        nkey = "n_val_games" if split == "val" else "n_train_games"
        self.n_games = manifest.get(nkey, len(self.shards) * 60)
        fkey = "n_val_frames" if split == "val" else "n_train_frames"
        self.n_frames = manifest.get(fkey, self.n_games * 8000)

    def __len__(self):
        return len(self.shards) * self.windows_per_shard_visit

    def _shard_files(self):
        files = list(self.shards)
        if self._world_size > 1 and self._distributed:
            files = files[self._rank :: self._world_size]
        worker_info = get_worker_info()
        if worker_info is not None:
            files = files[worker_info.id :: worker_info.num_workers]
        return files

    def __iter__(self):
        files = self._shard_files()
        worker_info = get_worker_info()
        wid = worker_info.id if worker_info is not None else 0
        rng = random.Random(self._seed + self._rank * 1000 + wid)
        rng.shuffle(files)

        W = self.window
        for shard_path in files:
            try:
                shard = torch.load(
                    shard_path, map_location="cpu",
                    weights_only=False, mmap=True,
                )
            except Exception as exc:
                import warnings
                warnings.warn(f"skipping {shard_path.name}: {exc}", stacklevel=2)
                continue

            states = shard["states"]
            offsets = shard["offsets"]
            n_games = int(shard["n_games"])
            if n_games == 0:
                continue

            # Check that every required key exists in this shard
            missing = [k for k in self._state_keys if k not in states]
            if missing:
                import warnings
                warnings.warn(
                    f"{shard_path.name}: missing keys {missing[:3]}{'...' if len(missing)>3 else ''}, "
                    f"skipping shard", stacklevel=2)
                continue

            # Per-game outcome from stock trajectories
            self_stock = states["self_numeric"][:, STOCK_COL]
            opp_stock = states["opp_numeric"][:, STOCK_COL]
            self_pct = states["self_numeric"][:, PERCENT_COL]
            opp_pct = states["opp_numeric"][:, PERCENT_COL]
            game_info = compute_game_outcomes(
                self_stock, opp_stock, offsets, n_games,
                self_pct_col=self_pct, opp_pct_col=opp_pct,
            )

            # Build the list of (game_idx, valid_window_starts) once
            # rather than rejecting too-short games on each sample.
            game_ranges = []
            for gi, (gs, ge, outcome) in enumerate(game_info):
                max_start = ge - gs - W
                if max_start < 0:
                    continue
                game_ranges.append((gs, max_start, outcome))
            if not game_ranges:
                continue

            for _ in range(self.windows_per_shard_visit):
                gs, max_start, outcome = rng.choice(game_ranges)
                # Position filter: pick a window whose LAST frame's position
                # in the game falls in [min_position, max_position]. Window
                # spans positions [(offset+W-1) - (W-1), offset+W-1] / length.
                # Restrict the random offset so the last frame is in range.
                game_length = max_start + W  # = e - s
                lo_offset = max(0,
                                int(self._min_position * game_length) - (W - 1))
                hi_offset = min(max_start,
                                int(self._max_position * game_length) - (W - 1))
                if hi_offset < lo_offset:
                    continue
                offset = rng.randint(lo_offset, hi_offset)
                abs_start = gs + offset
                state = {}
                for k in self._state_keys:
                    v = states[k]
                    state[k] = v[abs_start: abs_start + W]
                yield state, torch.tensor(outcome, dtype=torch.float32)


def collate_windows(batch):
    """Stack list of (state_dict_W, outcome) into batched (B, W, ...)."""
    states, outcomes = zip(*batch)
    keys = states[0].keys()
    out = {}
    for k in keys:
        out[k] = torch.stack([s[k] for s in states], dim=0)
    y = torch.stack(list(outcomes), dim=0)
    return out, y


# Convenience: the full list of shard keys ValueEncoder needs.
# Use this as `state_keys=` when constructing the dataset.
VALUE_ENCODER_KEYS: List[str] = [
    # Categorical / global
    "stage",
    "self_character", "opp_character",
    "self_action", "opp_action",
    # Per-player numeric+flags + action_elapsed
    "self_numeric", "opp_numeric",
    "self_flags", "opp_flags",
    "self_action_elapsed", "opp_action_elapsed",
    # Stage geometry (top-level shard key)
    "numeric",
    # Controllers
    "self_controller", "opp_controller",
]
# Projectile slots: proj{0..7}_owner/type/subtype + {0..7}_numeric
for _i in range(8):
    VALUE_ENCODER_KEYS.extend([
        f"proj{_i}_owner", f"proj{_i}_type", f"proj{_i}_subtype",
        f"{_i}_numeric",
    ])
