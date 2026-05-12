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
    self_numeric, opp_numeric, self_flags, opp_flags, self_controller,
    opp_controller (56-dim, if baked — see add_opp_controller_to_shards.py)
  next_ctrl keys (encoder's next_ctrl_dim conditioning):
    next_self_controller (56), next_opp_buttons (12),
    next_opp_analog (4), next_opp_c_dir (int64)
  target_state keys (WM heads):
    self_action, opp_action (int64),
    self_numeric, opp_numeric (float, 13 dim, normalized),
    self_flags, opp_flags (float, 5 dim),
    self_action_elapsed, opp_action_elapsed (float, normalized),
  + integer-bucket targets (always emitted, cheap; loss uses them iff
    discretize_counters=True on the model):
    self_percent_int, self_stock_int, self_jumps_int,
    self_hitlag_int, self_hitstun_int, self_elapsed_int   (int64)
    (plus all opp_* counterparts)
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
    # Opp controller: baked 56-dim one-hot (same layout as self_controller).
    # Produced by tools/add_opp_controller_to_shards.py. Fed to the encoder
    # symmetrically with self via `include_opp_controller=True`.
    "opp_controller",
    # Raw opp inputs — still needed to build next_ctrl conditioning.
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

        # Transform params for de-normalizing discrete-target columns back
        # to raw integers. `hal_norm.json` (or `mimic_norm.json`) holds the
        # transform spec. For columns that aren't in that file, the shard
        # pipeline falls back to z-score via norm_stats.json.
        hal_norm_path = self.data_dir / "hal_norm.json"
        if not hal_norm_path.exists():
            hal_norm_path = self.data_dir / "mimic_norm.json"
        if hal_norm_path.exists():
            with open(hal_norm_path) as fh:
                self.feat_norm = json.load(fh).get("features", {})
        else:
            self.feat_norm = {}

        # Cap values for the discrete int targets (clamp before CE so rare
        # out-of-range frames land in the top/overflow bin). Limits are:
        # percent 0..236 (237 bins), stock 0..4 (5), jumps 0..6 (7),
        # hitlag 0..20 (21), hitstun 0..60 (61 + 1 overflow = 62),
        # elapsed 0..60 (61 + 1 overflow = 62). See compute_wm_loss.
        self.DISC_BINS = {
            "percent": 237, "stock": 5, "jumps": 7,
            "hitlag": 21, "hitstun": 62, "elapsed": 62,
        }

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

    def _denorm_to_int(
        self,
        norm_vec: torch.Tensor,   # (W,) already-normalized values
        suffix: str,              # "percent", "stock", ... (hal_norm key)
        col_prefix: str,          # "self" or "opp"
        max_bin: int,
    ) -> torch.Tensor:
        """Invert whichever normalization was applied at shard-build time
        to recover the raw integer, then clamp to [0, max_bin]. Used for
        CE targets on integer-counter columns.
        """
        params = self.feat_norm.get(suffix)
        if params is None:
            # Fall back to z-score from norm_stats. The key names follow a
            # column-specific convention — action_elapsed is stored as
            # "{side}_action_frame" in norm_stats, hitlag/hitstun as
            # "{side}_{name}_left", everything else as "{side}_{name}".
            if suffix in ("hitlag", "hitstun"):
                key = f"{col_prefix}_{suffix}_left"
            elif suffix == "action_elapsed":
                key = f"{col_prefix}_action_frame"
            else:
                key = f"{col_prefix}_{suffix}"
            stats = self.norm_stats.get(key)
            if stats is None:
                raise KeyError(f"no norm params for {suffix}/{key}")
            mean, std = stats
            raw = norm_vec * std + mean
        else:
            t = params["transform"]
            if t == "standardize":
                raw = norm_vec * params["std"] + params["mean"]
            elif t == "normalize":
                mn, mx = params["min"], params["max"]
                raw = (norm_vec + 1.0) * 0.5 * (mx - mn) + mn
            elif t == "invert_normalize":
                mn, mx = params["min"], params["max"]
                raw = mx - (norm_vec + 1.0) * 0.5 * (mx - mn)
            else:
                raw = norm_vec
        return raw.round().long().clamp_(0, max_bin)

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
                    # Opp 56-dim one-hot, symmetric with self_controller —
                    # both are t-aligned with state[t] (the controllers that
                    # produced state[t]). Encoder reads when
                    # include_opp_controller=True.
                    "opp_controller": raw["opp_controller"][:W],
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
                # Integer-bucket targets for discretize_counters — cheap to
                # always emit; loss uses them only if the model is configured
                # with discretize_counters=True. Column indices in
                # self_numeric: percent=2, stock=3, jumps_left=4,
                # hitlag_left=10, hitstun_left=11.
                bins = self.DISC_BINS
                for side in ("self", "opp"):
                    num = target[f"{side}_numeric"]
                    elapsed = target[f"{side}_action_elapsed"]
                    target[f"{side}_percent_int"] = self._denorm_to_int(
                        num[..., 2], "percent", side, bins["percent"] - 1)
                    target[f"{side}_stock_int"] = self._denorm_to_int(
                        num[..., 3], "stock", side, bins["stock"] - 1)
                    target[f"{side}_jumps_int"] = self._denorm_to_int(
                        num[..., 4], "jumps_left", side, bins["jumps"] - 1)
                    target[f"{side}_hitlag_int"] = self._denorm_to_int(
                        num[..., 10], "hitlag", side, bins["hitlag"] - 1)
                    target[f"{side}_hitstun_int"] = self._denorm_to_int(
                        num[..., 11], "hitstun", side, bins["hitstun"] - 1)
                    target[f"{side}_elapsed_int"] = self._denorm_to_int(
                        elapsed, "action_elapsed", side, bins["elapsed"] - 1)
                yield state, next_ctrl, target
