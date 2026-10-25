# dataset.py
# ---------------------------------------------------------------------------
# Pretokenized tensor shard dataset for MIMIC training.
#
# StreamingMeleeDataset -- IterableDataset that reads per-game .pt shards
# (tensor_manifest.json): games concatenated with an offsets array, windows
# extracted on the fly. Produced by tools/slp_to_shards.py.
# ---------------------------------------------------------------------------

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import IterableDataset, get_worker_info

from .features import encode_controller_onehot


class StreamingMeleeDataset(IterableDataset):
    """Streams pretokenized .pt shards for training.

    Requires tensor_manifest.json (per-game shards) and norm_stats.json (for checkpoint saving).
    """

    def __init__(
        self,
        data_dir: str,
        sequence_length: int = 60,
        reaction_delay: int = 0,
        split: str = "train",
        rank: int = 0,
        world_size: int = 1,
        hal_controller_encoding: bool = False,
        controller_combo_map: dict = None,
        n_controller_combos: int = 5,
        **kwargs,
    ) -> None:
        super().__init__()
        self.data_dir        = Path(data_dir)
        self.sequence_length = sequence_length
        self.reaction_delay  = reaction_delay
        self.split           = split
        self._rank           = rank
        self._world_size     = world_size
        self._distributed    = kwargs.pop("distributed", True)
        self._char_filter    = kwargs.pop("character_filter", None)  # e.g. 1 for Fox
        self._hal_ctrl_enc   = hal_controller_encoding
        self._combo_map      = controller_combo_map
        self._n_combos       = n_controller_combos
        # Only slice keys the model uses (if known) — avoids loading 58 unused keys
        self._state_keys     = None  # set by caller to filter state dict keys
        # whole_games: yield padded batches of entire games (frame_budget frames each) instead of
        # windows, so every position has exactly the history it has live (see _iter_whole_games)
        self._whole_games    = kwargs.pop("whole_games", False)
        self._frame_budget   = kwargs.pop("frame_budget", None)

        with open(self.data_dir / "norm_stats.json") as fh:
            self.norm_stats: Dict[str, Tuple[float, float]] = json.load(fh)

        manifest_path = self.data_dir / "tensor_manifest.json"

        if manifest_path.exists():
            with open(manifest_path) as fh:
                manifest = json.load(fh)
            key = "val_shards" if split == "val" else "train_shards"
            self.files = [self.data_dir / n for n in manifest[key]]
            nkey = "n_val_games" if split == "val" else "n_train_games"
            self.n_games = manifest[nkey]
            # Random window sampling: ~100 windows per game per pass
            self._total_windows = self.n_games * 100

        else:
            raise RuntimeError(f"No tensor_manifest.json in {data_dir} (per-game v2 shards).")

    def __len__(self):
        return self._total_windows

    # ------------------------------------------------------------------
    # Sharding across DDP ranks and DataLoader workers
    # ------------------------------------------------------------------
    def _shard_files(self, files):
        worker_info = get_worker_info()
        if self._world_size > 1 and self._distributed:
            files = files[self._rank :: self._world_size]
        if worker_info is not None:
            files = files[worker_info.id :: worker_info.num_workers]
        return files

    # ------------------------------------------------------------------
    # Per-game shards (from upload_dataset.py)
    # ------------------------------------------------------------------
    def _iter_pergame(self):
        files = list(self.files)
        rng = random.Random(42)
        rng.shuffle(files)
        files = self._shard_files(files)
        random.shuffle(files)

        W, R = self.sequence_length, self.reaction_delay

        for path in files:
            shard = torch.load(path, weights_only=True, mmap=True)
            offsets = shard["offsets"]
            n_games = shard["n_games"]
            states = shard["states"]
            targets = shard["targets"]

            game_ranges: List[Tuple[int, int]] = []  # (start, max_w) per valid game
            skipped_games = 0
            for g in range(n_games):
                start = offsets[g].item()
                end = offsets[g + 1].item()
                n_frames = end - start
                max_w = n_frames - W - R
                if max_w < 0:
                    skipped_games += 1
                    continue
                # Character filter: skip games where self isn't the target character
                if self._char_filter is not None:
                    if states["self_character"][start].item() != self._char_filter:
                        skipped_games += 1
                        continue
                game_ranges.append((start, max_w))
            if skipped_games > 0 and n_games > 0:
                pct = 100 * skipped_games / n_games
                if pct > 5:
                    import warnings
                    warnings.warn(
                        f"seq_len={W}: skipped {skipped_games}/{n_games} games "
                        f"({pct:.0f}%) in {path.name} — too short for context window",
                        stacklevel=2,
                    )

            # Random window sampling: N random windows per game per shard visit.
            # HAL samples one random window per __getitem__ call; across training
            # each game is visited ~5000 times. We amortize shard loading by
            # sampling multiple random windows per game per visit.
            windows_per_game = max(1, min(100, max_w // 2)) if game_ranges else 1
            window_indices = []
            for game_start, max_w in game_ranges:
                for _ in range(windows_per_game):
                    window_indices.append(game_start + random.randint(0, max_w))
            random.shuffle(window_indices)
            for abs_start in window_indices:
                _sk = self._state_keys
                if _sk is not None:
                    state = {k: states[k][abs_start: abs_start + W] for k in _sk}
                else:
                    state = {k: v[abs_start: abs_start + W] for k, v in states.items()}
                target = {k: v[abs_start + R: abs_start + W + R] for k, v in targets.items()}
                # HAL-style: encode controller as one-hot vector (only if not pre-encoded in shard)
                if (self._hal_ctrl_enc and self._combo_map is not None
                        and "self_controller" not in state
                        and "self_buttons" in state and "self_analog" in state):
                    onehot = encode_controller_onehot(
                        state["self_buttons"].numpy(),
                        state["self_analog"].numpy(),
                        state["self_c_dir"].numpy(),
                        self._combo_map,
                        self._n_combos,
                        norm_stats=self.norm_stats,
                    )
                    state["self_controller"] = torch.from_numpy(onehot)
                    del state["self_buttons"]
                    del state["self_analog"]
                    del state["self_c_dir"]
                yield state, target

    # ------------------------------------------------------------------
    # Whole games
    # ------------------------------------------------------------------
    def _iter_whole_games(self):
        """Batches of entire games, from frame 0 to the end, so every position sees exactly the
        history a streaming model has at that frame of a live game. Per shard visit, games are
        grouped by length (rounded up to 1,024 frames) and packed frame_budget // length per batch,
        padded at the end; targets["loss_mask"] marks the real frames (causal attention: padding
        after a game never reaches it). Use with DataLoader(batch_size=None)."""
        files = list(self.files)
        rng = random.Random(42)
        rng.shuffle(files)
        files = self._shard_files(files)
        random.shuffle(files)
        R, budget, BUCKET = self.reaction_delay, self._frame_budget, 1024
        for path in files:
            shard = torch.load(path, weights_only=True, mmap=True)
            offsets, states, targets = shard["offsets"], shard["states"], shard["targets"]
            if "self_controller" not in states:
                raise RuntimeError(f"{path.name}: whole-game mode needs pre-encoded self_controller")
            buckets: Dict[int, List[Tuple[int, int]]] = {}
            for g in range(int(shard["n_games"])):
                start, end = offsets[g].item(), offsets[g + 1].item()
                n = end - start - R
                if n <= 0:
                    continue
                if self._char_filter is not None and states["self_character"][start].item() != self._char_filter:
                    continue
                buckets.setdefault(-(-n // BUCKET) * BUCKET, []).append((start, n))
            batches = []
            for Lb, games in buckets.items():
                random.shuffle(games)
                per = max(1, budget // Lb)
                batches += [(Lb, games[i:i + per]) for i in range(0, len(games), per)]
            random.shuffle(batches)
            keys = self._state_keys or list(states)
            for Lb, games in batches:
                st = {k: states[k].new_zeros((len(games), Lb) + tuple(states[k].shape[1:])) for k in keys}
                tg = {k: v.new_zeros((len(games), Lb) + tuple(v.shape[1:])) for k, v in targets.items()}
                mask = torch.zeros(len(games), Lb)
                for b, (start, n) in enumerate(games):
                    for k in keys:
                        st[k][b, :n] = states[k][start:start + n]
                    for k, v in targets.items():
                        tg[k][b, :n] = v[start + R:start + R + n]
                    mask[b, :n] = 1.0
                tg["loss_mask"] = mask
                yield st, tg

    def __iter__(self):
        if self._whole_games:
            yield from self._iter_whole_games()
        else:
            yield from self._iter_pergame()
