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
        **kwargs,
    ) -> None:
        super().__init__()
        self.data_dir        = Path(data_dir)
        self.sequence_length = sequence_length
        self.reaction_delay  = reaction_delay
        self.split           = split
        self._rank           = rank
        self._world_size     = world_size

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
            fkey = "n_val_frames" if split == "val" else "n_train_frames"
            W, R = sequence_length, reaction_delay
            n_frames = manifest[fkey]
            self._total_windows = max(0, n_frames - self.n_games * (W + R - 1))

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
        if self._world_size > 1:
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

            window_indices: List[Tuple[int, int]] = []
            skipped_games = 0
            for g in range(n_games):
                start = offsets[g].item()
                end = offsets[g + 1].item()
                n_frames = end - start
                max_w = n_frames - W - R
                if max_w < 0:
                    skipped_games += 1
                    continue
                for w in range(max_w + 1):
                    window_indices.append((start + w, start + w))
            if skipped_games > 0 and n_games > 0:
                pct = 100 * skipped_games / n_games
                if pct > 5:
                    import warnings
                    warnings.warn(
                        f"seq_len={W}: skipped {skipped_games}/{n_games} games "
                        f"({pct:.0f}%) in {path.name} — too short for context window",
                        stacklevel=2,
                    )

            random.shuffle(window_indices)

            for abs_start, _ in window_indices:
                state = {k: v[abs_start: abs_start + W] for k, v in states.items()}
                target = {k: v[abs_start + R: abs_start + W + R] for k, v in targets.items()}
                yield state, target

    # ------------------------------------------------------------------
    def __iter__(self):
        if self._mode == "prewindowed":
            yield from self._iter_prewindowed()
        else:
            yield from self._iter_pergame()
