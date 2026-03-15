# dataset.py
# ---------------------------------------------------------------------------
# Slippi frame data -> fixed-length windows for MIMIC training.
#
# Two dataset classes:
#   MeleeFrameDatasetWithDelay  - reads raw parquets, preprocesses at init
#                                 (slow startup, self-contained, good for debug)
#   StreamingMeleeDataset       - IterableDataset that reads raw parquets
#                                 on-the-fly using precomputed metadata
#                                 (instant startup, scales to any dataset size)
# ---------------------------------------------------------------------------

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

import features as F

# ---------------------------------------------------------------------------
# Raw-parquet dataset (preprocesses at init -- slow but self-contained)
# ---------------------------------------------------------------------------
def _load_cluster_centers(data_dir: Path):
    """Load stick_clusters.json if present, returning (stick_centers, shoulder_centers) or (None, None)."""
    path = data_dir / "stick_clusters.json"
    if not path.exists():
        return None, None
    with open(path) as fh:
        raw = json.load(fh)
    stick = np.array(raw["stick_centers"], dtype=np.float32) if "stick_centers" in raw else None
    shoulder = np.array(raw["shoulder_centers"], dtype=np.float32) if "shoulder_centers" in raw else None
    return stick, shoulder


class MeleeFrameDatasetWithDelay(Dataset):
    """Fixed-length windows over Slippi frame data with a reaction delay."""

    def __init__(
        self,
        parquet_dir: str,
        sequence_length: int = 30,
        reaction_delay: int = 1,
        split: str = "train",
        val_frac: float = 0.1,
        norm_stats: Dict[str, Tuple[float, float]] = None,
        no_opp_inputs: bool = False,
    ) -> None:
        super().__init__()
        self.parquet_dir     = Path(parquet_dir)
        self.sequence_length = sequence_length
        self.reaction_delay  = reaction_delay
        self._no_opp_inputs  = no_opp_inputs

        all_files = sorted(self.parquet_dir.glob("*.parquet"))
        if not all_files:
            raise RuntimeError(f"No .parquet files found in {parquet_dir}")

        rng = random.Random(42)
        shuffled = list(all_files)
        rng.shuffle(shuffled)
        n_val = int(len(shuffled) * val_frac)
        if split == "val" and n_val > 0:
            self.files = shuffled[:n_val]
        else:
            self.files = shuffled[n_val:] if n_val > 0 else shuffled

        raw_dfs: Dict[Path, pd.DataFrame] = {}
        for f in self.files:
            df = pd.read_parquet(f)
            df = df[df["frame"] >= 0].reset_index(drop=True)
            raw_dfs[f] = df

        self.index_map: List[Tuple[Path, int]] = []
        for f, df in raw_dfs.items():
            max_start = len(df) - (sequence_length + reaction_delay)
            if max_start >= 0:
                self.index_map.extend([(f, s) for s in range(max_start + 1)])
        if not self.index_map:
            raise RuntimeError("No valid windows across the dataset.")

        self._fg = F.build_feature_groups(no_opp_inputs=no_opp_inputs)
        self._categorical_cols = F.get_categorical_cols(self._fg)
        self._norm_cols = F.get_norm_cols(self._fg)

        dynamic_maps = F.build_categorical_mappings_streaming(
            [f for f in self.files], self._categorical_cols)

        self._df_cache: Dict[Path, pd.DataFrame] = {}
        for f, df in raw_dfs.items():
            self._df_cache[f] = F.preprocess_df(df, self._categorical_cols, dynamic_maps)

        if norm_stats is not None:
            self.norm_stats = norm_stats
        else:
            col_sum: Dict[str, float] = {}
            col_sq:  Dict[str, float] = {}
            col_n:   Dict[str, int]   = {}
            for df in self._df_cache.values():
                F.update_norm_accumulators(df, self._norm_cols, col_sum, col_sq, col_n)
            self.norm_stats = F.finalize_norm_stats(self._norm_cols, col_sum, col_sq, col_n)

        for df in self._df_cache.values():
            F.apply_normalization(df, self.norm_stats)

        used_cols = set()
        for _, meta in F.walk_groups(self._fg, return_meta=True):
            used_cols.update(meta["cols"])
        used_cols.update(["self_main_x", "self_main_y", "self_l_shldr",
                          "self_r_shldr", "self_c_dir", *F.btn_cols("self")])
        for f in self._df_cache:
            df = self._df_cache[f]
            drop = [c for c in df.columns if c not in used_cols]
            if drop:
                self._df_cache[f] = df.drop(columns=drop)

        self._stick_centers, self._shoulder_centers = _load_cluster_centers(self.parquet_dir)

        self._target_cache: Dict[Path, Dict[str, torch.Tensor]] = {}
        for f, df in self._df_cache.items():
            self._target_cache[f] = F.build_targets_batch(
                df, self.norm_stats,
                stick_centers=self._stick_centers,
                shoulder_centers=self._shoulder_centers)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx: int):
        fpath, start_idx = self.index_map[idx]
        W, R = self.sequence_length, self.reaction_delay
        df = self._df_cache[fpath]

        end_idx = start_idx + W
        slice_df = df.iloc[start_idx:end_idx]
        state_seq = F.df_to_state_tensors(slice_df, self._fg)

        target_start = start_idx + R
        target_end   = start_idx + W + R
        targets = self._target_cache[fpath]
        target = {k: v[target_start:target_end] for k, v in targets.items()}

        return state_seq, target


# ---------------------------------------------------------------------------
# Streaming dataset (instant startup, reads raw parquets on-the-fly)
# ---------------------------------------------------------------------------
SHUFFLE_BUFFER_SIZE = 8192


class StreamingMeleeDataset(IterableDataset):
    """Streams raw parquets with on-the-fly preprocessing.

    Requires precomputed metadata from preprocess.py:
      - norm_stats.json
      - cat_maps.json
      - file_index.json
    """

    def __init__(
        self,
        data_dir: str,
        sequence_length: int = 60,
        reaction_delay: int = 1,
        split: str = "train",
        val_frac: float = 0.1,
        no_opp_inputs: bool = False,
    ) -> None:
        super().__init__()
        self.data_dir        = Path(data_dir)
        self.sequence_length = sequence_length
        self.reaction_delay  = reaction_delay
        self.split           = split
        self._no_opp_inputs  = no_opp_inputs

        with open(self.data_dir / "norm_stats.json") as fh:
            self.norm_stats: Dict[str, Tuple[float, float]] = json.load(fh)
        with open(self.data_dir / "cat_maps.json") as fh:
            raw = json.load(fh)
            self.cat_maps: Dict[str, Dict[int, int]] = {
                col: {int(k): v for k, v in m.items()} for col, m in raw.items()
            }
        with open(self.data_dir / "file_index.json") as fh:
            self.file_index: Dict[str, int] = json.load(fh)

        self._stick_centers, self._shoulder_centers = _load_cluster_centers(self.data_dir)

        self._fg = F.build_feature_groups(no_opp_inputs=no_opp_inputs)
        self._categorical_cols = F.get_categorical_cols(self._fg)

        all_names = sorted(self.file_index.keys())
        rng = random.Random(42)
        rng.shuffle(all_names)
        n_val = int(len(all_names) * val_frac)

        if split == "val" and n_val > 0:
            names = all_names[:n_val]
        else:
            names = all_names[n_val:] if n_val > 0 else all_names

        self.files = [self.data_dir / n for n in names]
        self.n_games = len(self.files)

        W, R = sequence_length, reaction_delay
        self._total_windows = sum(
            max(0, self.file_index[n] - W - R + 1)
            for n in names
        )

    def __len__(self):
        return self._total_windows

    def _process_game(self, path: Path):
        """Load one parquet, preprocess, return (state_tensors, target_tensors)."""
        df = pd.read_parquet(path)
        df = df[df["frame"] >= 0].reset_index(drop=True)
        if len(df) < 2:
            return None, None, 0

        df = F.preprocess_df(df, self._categorical_cols, self.cat_maps)
        F.apply_normalization(df, self.norm_stats)

        state   = F.df_to_state_tensors(df, self._fg)
        targets = F.build_targets_batch(
            df, self.norm_stats,
            stick_centers=self._stick_centers,
            shoulder_centers=self._shoulder_centers)
        return state, targets, len(df)

    def __iter__(self):
        worker_info = get_worker_info()

        files = list(self.files)
        rng = random.Random(42)
        rng.shuffle(files)
        if worker_info is not None:
            files = files[worker_info.id :: worker_info.num_workers]
        random.shuffle(files)

        W, R = self.sequence_length, self.reaction_delay
        buf: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]] = []

        for path in files:
            state, targets, n_frames = self._process_game(path)
            if state is None:
                continue

            max_start = n_frames - W - R
            for s in range(max_start + 1):
                end = s + W
                target_start = s + R
                target_end   = s + W + R

                window_state  = {k: v[s:end].clone() for k, v in state.items()}
                window_target = {k: v[target_start:target_end].clone() for k, v in targets.items()}

                buf.append((window_state, window_target))

                if len(buf) >= SHUFFLE_BUFFER_SIZE:
                    random.shuffle(buf)
                    half = SHUFFLE_BUFFER_SIZE // 2
                    for item in buf[:half]:
                        yield item
                    buf = buf[half:]

        random.shuffle(buf)
        yield from buf
