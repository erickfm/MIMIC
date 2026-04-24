"""World-model dataset iterator.

Pulls windows of `W` frames from per-game shards produced by
`tools/slp_to_shards.py`. Each `__iter__` sample is a 3-tuple:

    state_dict:  per-frame inputs for MimicFlatEncoder (state[t])
    next_ctrl:   per-frame t+1 controllers used as encoder conditioning
    target_state: per-frame t+1 state for the fields we predict

All three dicts have tensors of shape (W, ...). The loader reads W+1
consecutive frames and slices [:W] for state, [1:W+1] for conditioning
and targets.

Fields:
  state_dict keys (input to encoder):
    stage, self_character, opp_character, self_action, opp_action,
    self_numeric, opp_numeric, self_flags, opp_flags, self_controller
  next_ctrl keys (encoder's next_ctrl_dim conditioning):
    next_self_controller (56), next_opp_buttons (12),
    next_opp_analog (4), next_opp_c_dir (int64)
  target_state keys (WM heads):
    self_action, opp_action (int64),
    self_numeric, opp_numeric (float, 13 dim),
    self_flags, opp_flags (float, 5 dim)
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator, Tuple

import torch
from torch.utils.data import IterableDataset, get_worker_info

# Keys we read from the shard's `states` dict. Only these are sliced.
# Static fields (stage, character) also sit in `states` but don't change
# across frames — we still read the whole W+1 window and let the encoder
# see a scalar-constant per frame.
_STATE_KEYS = (
    "stage",
    "self_character", "opp_character",
    "self_action", "opp_action",
    "self_numeric", "opp_numeric",
    "self_flags", "opp_flags",
    "self_controller",
    # Opponent raw controller (not in MimicFlatEncoder's state input,
    # but needed for next_ctrl conditioning):
    "opp_buttons", "opp_analog", "opp_c_dir",
    # Action-elapsed counters (t+1 used as regression target, never input).
    "self_action_elapsed", "opp_action_elapsed",
)


class WorldModelDataset(IterableDataset):
    """Per-game shard iterator yielding (state, next_ctrl, target) 3-tuples.

    Wraps the same per-game shard format as `StreamingMeleeDataset` but
    emits WM-shaped samples instead of (state, bc_targets).
    """

    def __init__(
        self,
        data_dir: str,
        sequence_length: int = 180,
        split: str = "train",
        rank: int = 0,
        world_size: int = 1,
        character_filter: int = None,
        distributed: bool = True,
        windows_per_game: int = 100,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.W = sequence_length
        self.split = split
        self._rank = rank
        self._world_size = world_size
        self._distributed = distributed
        self._char_filter = character_filter
        self._windows_per_game = windows_per_game

        with open(self.data_dir / "norm_stats.json") as fh:
            self.norm_stats = json.load(fh)

        manifest_path = self.data_dir / "tensor_manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as fh:
                manifest = json.load(fh)
        else:
            # No manifest — auto-split: last ~5% of train_shard_*.pt held out
            # for val.  Keeps the WM pipeline usable on shard dirs that were
            # built before the manifest format existed.
            train_shards = sorted(
                p.name for p in self.data_dir.glob("train_shard_*.pt")
            )
            if not train_shards:
                raise RuntimeError(
                    f"{manifest_path} missing and no train_shard_*.pt found in "
                    f"{self.data_dir}. WM dataset needs per-game shards "
                    f"(build with tools/slp_to_shards.py) or at minimum a "
                    f"flat directory of shard files."
                )
            n_val = max(1, int(len(train_shards) * 0.05))
            manifest = {
                "train_shards": train_shards[:-n_val],
                "val_shards": train_shards[-n_val:],
                # Games-per-shard estimate is used only for __len__ reporting.
                "n_train_games": (len(train_shards) - n_val) * 50,
                "n_val_games": n_val * 50,
            }

        key = "val_shards" if split == "val" else "train_shards"
        self.files = [self.data_dir / n for n in manifest[key]]
        nkey = "n_val_games" if split == "val" else "n_train_games"
        self.n_games = manifest.get(nkey, len(self.files) * 50)
        self._total_windows = self.n_games * windows_per_game

        # Probe the first shard to surface the numeric/flag widths and
        # `action_elapsed` availability for the head construction to match.
        _probe = torch.load(self.files[0], weights_only=True, mmap=True)
        _states = _probe["states"]
        self.n_numeric = int(_states["self_numeric"].shape[-1])
        self.n_flags = int(_states["self_flags"].shape[-1])
        self.has_action_elapsed = "self_action_elapsed" in _states

    def __len__(self) -> int:
        return self._total_windows

    def _shard_files(self, files):
        worker_info = get_worker_info()
        if self._world_size > 1 and self._distributed:
            files = files[self._rank :: self._world_size]
        if worker_info is not None:
            files = files[worker_info.id :: worker_info.num_workers]
        return files

    def __iter__(self) -> Iterator[Tuple[dict, dict, dict]]:
        files = list(self.files)
        random.Random(42).shuffle(files)
        files = self._shard_files(files)
        random.shuffle(files)

        # Need one extra frame beyond the window: state[i+W] is the last target.
        W = self.W
        need = W + 1

        for path in files:
            shard = torch.load(path, weights_only=True, mmap=True)
            offsets = shard["offsets"]
            n_games = shard["n_games"]
            states = shard["states"]

            # Build valid (game_start, max_w) list.
            game_ranges = []
            for g in range(n_games):
                start = offsets[g].item()
                end = offsets[g + 1].item()
                max_w = (end - start) - need
                if max_w < 0:
                    continue
                if self._char_filter is not None:
                    if states["self_character"][start].item() != self._char_filter:
                        continue
                game_ranges.append((start, max_w))

            # Random window sampling per game.
            windows_per_game = max(1, min(self._windows_per_game, 100))
            window_starts = []
            for g_start, max_w in game_ranges:
                for _ in range(windows_per_game):
                    window_starts.append(g_start + random.randint(0, max_w))
            random.shuffle(window_starts)

            for abs_start in window_starts:
                # Slice W+1 frames once, then split.
                end = abs_start + need
                raw = {k: states[k][abs_start:end] for k in _STATE_KEYS}

                # state[:W]
                state = {
                    "stage": raw["stage"][:W],
                    "self_character": raw["self_character"][:W],
                    "opp_character": raw["opp_character"][:W],
                    "self_action": raw["self_action"][:W],
                    "opp_action": raw["opp_action"][:W],
                    "self_numeric": raw["self_numeric"][:W],
                    "opp_numeric": raw["opp_numeric"][:W],
                    "self_flags": raw["self_flags"][:W],
                    "opp_flags": raw["opp_flags"][:W],
                    "self_controller": raw["self_controller"][:W],
                }
                # t+1 conditioning
                next_ctrl = {
                    "next_self_controller": raw["self_controller"][1:W + 1],
                    "next_opp_buttons": raw["opp_buttons"][1:W + 1],
                    "next_opp_analog": raw["opp_analog"][1:W + 1],
                    "next_opp_c_dir": raw["opp_c_dir"][1:W + 1],
                }
                # t+1 target (what we predict)
                target = {
                    "self_action": raw["self_action"][1:W + 1],
                    "opp_action": raw["opp_action"][1:W + 1],
                    "self_numeric": raw["self_numeric"][1:W + 1],
                    "opp_numeric": raw["opp_numeric"][1:W + 1],
                    "self_flags": raw["self_flags"][1:W + 1],
                    "opp_flags": raw["opp_flags"][1:W + 1],
                    "self_action_elapsed": raw["self_action_elapsed"][1:W + 1],
                    "opp_action_elapsed": raw["opp_action_elapsed"][1:W + 1],
                }
                yield state, next_ctrl, target
