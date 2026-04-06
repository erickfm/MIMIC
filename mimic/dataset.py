# dataset.py
# ---------------------------------------------------------------------------
# Pretokenized tensor shard dataset for MIMIC training.
#
# StreamingMeleeDataset -- IterableDataset that reads pretokenized .pt shards.
# Two shard formats are supported:
#   1. Per-game shards (tensor_manifest.json) -- games concatenated with
#      offsets array; windows extracted on the fly.  Produced by
#      upload_dataset.py for HuggingFace distribution.
#   2. Pre-windowed shards (tensor_meta.json) -- each sample is a
#      pre-sliced window.  Produced locally by tensorize.py for maximum
#      single-machine throughput.
# ---------------------------------------------------------------------------

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import IterableDataset, get_worker_info

from .features import encode_controller_onehot


class StreamingMeleeDataset(IterableDataset):
    """Streams pretokenized .pt shards for training.

    Requires one of:
      - tensor_manifest.json  (per-game shards from upload_dataset.py)
      - tensor_meta.json      (pre-windowed shards from tensorize.py)
    Plus norm_stats.json (for checkpoint saving).
    """

    def __init__(
        self,
        data_dir: str,
        sequence_length: int = 60,
        reaction_delay: int = 1,
        split: str = "train",
        rank: int = 0,
        world_size: int = 1,
        controller_offset: bool = False,
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
        self._controller_offset = controller_offset
        self._hal_ctrl_enc   = hal_controller_encoding
        self._combo_map      = controller_combo_map
        self._n_combos       = n_controller_combos

        with open(self.data_dir / "norm_stats.json") as fh:
            self.norm_stats: Dict[str, Tuple[float, float]] = json.load(fh)

        prewindowed_path = self.data_dir / "tensor_meta.json"
        manifest_path = self.data_dir / "tensor_manifest.json"

        if prewindowed_path.exists():
            self._mode = "prewindowed"
            with open(prewindowed_path) as fh:
                tmeta = json.load(fh)
            key = "val_shards" if split == "val" else "train_shards"
            self.files = [self.data_dir / n for n in tmeta[key]]
            self.n_games = len(self.files)
            wkey = "n_val_windows" if split == "val" else "n_train_windows"
            self._total_windows = tmeta[wkey]

        elif manifest_path.exists():
            self._mode = "pergame"
            with open(manifest_path) as fh:
                manifest = json.load(fh)
            key = "val_shards" if split == "val" else "train_shards"
            self.files = [self.data_dir / n for n in manifest[key]]
            nkey = "n_val_games" if split == "val" else "n_train_games"
            self.n_games = manifest[nkey]
            # Random window sampling: ~100 windows per game per pass
            self._total_windows = self.n_games * 100

        else:
            raise RuntimeError(
                f"No tensor_meta.json or tensor_manifest.json in {data_dir}. "
                f"Run upload_dataset.py or tensorize.py first."
            )

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
    # Pre-windowed shards (from tensorize.py)
    # ------------------------------------------------------------------
    def _iter_prewindowed(self):
        files = list(self.files)
        rng = random.Random(42)
        rng.shuffle(files)
        files = self._shard_files(files)
        random.shuffle(files)

        for path in files:
            shard = torch.load(path, weights_only=True)
            n = shard["n"]
            indices = list(range(n))
            random.shuffle(indices)
            for i in indices:
                state = {k: v[i] for k, v in shard["states"].items()}
                target = {k: v[i] for k, v in shard["targets"].items()}

                # HAL-style: encode controller as one-hot vector (only if not pre-encoded)
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
            shard = torch.load(path, weights_only=True)
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
                state = {k: v[abs_start: abs_start + W] for k, v in states.items()}
                target = {k: v[abs_start + R: abs_start + W + R] for k, v in targets.items()}
                # HAL-style: shift self-controller by -1 so position i sees frame i-1's controller
                if self._controller_offset and W > 1:
                    # Shift whichever controller keys exist (raw or pre-encoded)
                    ctrl_keys = (["self_controller"] if "self_controller" in state
                                 else ["self_buttons", "self_analog", "self_c_dir"])
                    for ck in ctrl_keys:
                        if ck in state:
                            orig = state[ck]
                            shifted = torch.zeros_like(orig)
                            shifted[1:] = orig[:-1]
                            state[ck] = shifted

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
    def __iter__(self):
        if self._mode == "prewindowed":
            yield from self._iter_prewindowed()
        else:
            yield from self._iter_pergame()
