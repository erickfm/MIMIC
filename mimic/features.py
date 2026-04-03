# features.py
# ---------------------------------------------------------------------------
# Single source of truth for MIMIC feature engineering.
# Used by dataset.py, upload_dataset.py, tensorize.py, inference.py, and
# slp_to_shards.py.
# ---------------------------------------------------------------------------

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from .cat_maps import (
    STAGE_MAP,
    CHARACTER_MAP,
    ACTION_MAP,
    PROJECTILE_TYPE_MAP,
)

# ---------------------------------------------------------------------------
# Column-name generators
# ---------------------------------------------------------------------------
BTN = [
    "BUTTON_A", "BUTTON_B", "BUTTON_X", "BUTTON_Y",
    "BUTTON_Z", "BUTTON_L", "BUTTON_R", "BUTTON_START",
    "BUTTON_D_UP", "BUTTON_D_DOWN", "BUTTON_D_LEFT", "BUTTON_D_RIGHT",
]

STAGE_GEOM_COLS = [
    "blastzone_left", "blastzone_right", "blastzone_top", "blastzone_bottom",
    "stage_edge_left", "stage_edge_right",
    "left_platform_height",  "left_platform_left",  "left_platform_right",
    "right_platform_height", "right_platform_left", "right_platform_right",
    "top_platform_height",   "top_platform_left",   "top_platform_right",
    "randall_height", "randall_left", "randall_right",
]

ENUM_MAPS: Dict[str, Dict[int, int]] = {
    "stage":      STAGE_MAP,
    "_character": CHARACTER_MAP,
    "_action":    ACTION_MAP,
    "_type":      PROJECTILE_TYPE_MAP,
    "c_dir":      {i: i for i in range(5)},
}

PROJ_SLOTS = 8


def btn_cols(prefix: str) -> List[str]:
    return [f"{prefix}_btn_{b}" for b in BTN]


def analog_cols(prefix: str) -> List[str]:
    return [f"{prefix}_main_x", f"{prefix}_main_y",
            f"{prefix}_l_shldr", f"{prefix}_r_shldr"]


def numeric_state(prefix: str, hal_minimal: bool = False) -> List[str]:
    if hal_minimal:
        # HAL uses only these 9 features per player (no speeds, ECB, hitlag/hitstun)
        return [
            f"{prefix}_pos_x", f"{prefix}_pos_y",
            f"{prefix}_percent", f"{prefix}_stock",
            f"{prefix}_jumps_left",
            f"{prefix}_invuln_left", f"{prefix}_shield_strength",
        ]
    base = [
        f"{prefix}_pos_x", f"{prefix}_pos_y",
        f"{prefix}_percent", f"{prefix}_stock",
        f"{prefix}_jumps_left",
        f"{prefix}_speed_air_x_self", f"{prefix}_speed_ground_x_self",
        f"{prefix}_speed_x_attack", f"{prefix}_speed_y_attack",
        f"{prefix}_speed_y_self",
        f"{prefix}_hitlag_left", f"{prefix}_hitstun_left",
        f"{prefix}_invuln_left", f"{prefix}_shield_strength",
    ]
    ecb = [f"{prefix}_ecb_{part}_{axis}"
           for part in ("bottom", "left", "right", "top")
           for axis in ("x", "y")]
    return base + ecb


def flags(prefix: str) -> List[str]:
    return [f"{prefix}_on_ground", f"{prefix}_off_stage",
            f"{prefix}_facing", f"{prefix}_invulnerable",
            f"{prefix}_moonwalkwarning"]


def categorical_ids(prefix: str) -> List[str]:
    return [f"{prefix}_port", f"{prefix}_character",
            f"{prefix}_action", f"{prefix}_costume"]


# ---------------------------------------------------------------------------
# Feature-group spec + walking
# ---------------------------------------------------------------------------
def build_feature_groups(no_opp_inputs: bool = False,
                         no_self_inputs: bool = False,
                         hal_minimal: bool = False) -> Dict[str, Dict]:
    opp_cats = categorical_ids("opp")
    if not no_opp_inputs:
        opp_cats = opp_cats + ["opp_c_dir"]

    opp_nana_cats = ["opp_nana_character", "opp_nana_action"]
    if not no_opp_inputs:
        opp_nana_cats = opp_nana_cats + ["opp_nana_c_dir"]

    opp_group: Dict[str, Any] = {
        "categorical": opp_cats,
        "flags":   flags("opp"),
        "numeric": numeric_state("opp", hal_minimal=hal_minimal),
        "action_elapsed": ["opp_action_frame"],
    }
    if not no_opp_inputs:
        opp_group["buttons"] = btn_cols("opp")
        opp_group["analog"]  = analog_cols("opp")

    opp_nana_group: Dict[str, Any] = {
        "categorical": opp_nana_cats,
        "flags":   flags("opp_nana") + ["opp_nana_present"],
        "numeric": numeric_state("opp_nana") + [
            "opp_nana_stock", "opp_nana_jumps_left",
            "opp_nana_hitlag_left", "opp_nana_hitstun_left",
            "opp_nana_invuln_left",
        ],
        "action_elapsed": ["opp_nana_action_frame"],
    }
    if not no_opp_inputs:
        opp_nana_group["buttons"] = btn_cols("opp_nana")
        opp_nana_group["analog"]  = analog_cols("opp_nana")

    self_cats = categorical_ids("self")
    if not no_self_inputs:
        self_cats = self_cats + ["self_c_dir"]

    self_group: Dict[str, Any] = {
        "categorical": self_cats,
        "flags":   flags("self"),
        "numeric": numeric_state("self", hal_minimal=hal_minimal),
        "action_elapsed": ["self_action_frame"],
    }
    if not no_self_inputs:
        self_group["buttons"] = btn_cols("self")
        self_group["analog"]  = analog_cols("self")

    self_nana_cats = ["self_nana_character", "self_nana_action"]
    if not no_self_inputs:
        self_nana_cats = self_nana_cats + ["self_nana_c_dir"]

    self_nana_group: Dict[str, Any] = {
        "categorical": self_nana_cats,
        "flags":   flags("self_nana") + ["self_nana_present"],
        "numeric": numeric_state("self_nana") + [
            "self_nana_stock", "self_nana_jumps_left",
            "self_nana_hitlag_left", "self_nana_hitstun_left",
            "self_nana_invuln_left",
        ],
        "action_elapsed": ["self_nana_action_frame"],
    }
    if not no_self_inputs:
        self_nana_group["buttons"] = btn_cols("self_nana")
        self_nana_group["analog"]  = analog_cols("self_nana")

    return {
        "global": {
            "numeric": ["distance", "frame", *STAGE_GEOM_COLS],
            "categorical": ["stage"],
        },
        "players": {
            "self": self_group,
            "opp": opp_group,
            "self_nana": self_nana_group,
            "opp_nana": opp_nana_group,
        },
        "projectiles": {
            k: {
                "categorical": [f"proj{k}_owner", f"proj{k}_type", f"proj{k}_subtype"],
                "numeric": [f"proj{k}_pos_x", f"proj{k}_pos_y",
                            f"proj{k}_speed_x", f"proj{k}_speed_y",
                            f"proj{k}_frame"],
            }
            for k in range(PROJ_SLOTS)
        },
    }


_LEAF_KEYS = frozenset(("numeric", "categorical", "buttons", "flags", "analog", "action_elapsed"))


def walk_groups(fg: Dict, *, return_meta: bool = False):
    """Iterate over leaf feature groups in the spec."""
    stack = [((), fg)]
    while stack:
        prefix, node = stack.pop()
        if isinstance(node, dict) and all(k in _LEAF_KEYS for k in node):
            for ftype, cols in node.items():
                if cols:
                    meta = {"ftype": ftype, "cols": cols,
                            "entity": prefix[-1] if prefix else "global"}
                    yield (prefix, meta) if return_meta else cols
        else:
            for k, sub in node.items():
                stack.append(((*prefix, k), sub))


def get_categorical_cols(fg: Dict) -> List[str]:
    out: List[str] = []
    for _, meta in walk_groups(fg, return_meta=True):
        if meta["ftype"] == "categorical":
            out.extend(meta["cols"])
    return out


def get_norm_cols(fg: Dict) -> List[str]:
    out: List[str] = []
    for _, meta in walk_groups(fg, return_meta=True):
        if meta["ftype"] in ("numeric", "analog", "action_elapsed"):
            out.extend(meta["cols"])
    return sorted(set(out))


# ---------------------------------------------------------------------------
# Enum / categorical mapping
# ---------------------------------------------------------------------------
def get_enum_map(col: str, dynamic_maps: Dict[str, Dict[int, int]] | None = None) -> Dict[int, int]:
    if col == "stage":
        return ENUM_MAPS["stage"]
    if col.endswith("_c_dir"):
        return ENUM_MAPS["c_dir"]
    for suffix, m in ENUM_MAPS.items():
        if suffix != "stage" and col.endswith(suffix):
            return m
    if dynamic_maps and col in dynamic_maps:
        return dynamic_maps[col]
    return {}


def _cols_needing_dynamic_map(categorical_cols: List[str]) -> List[str]:
    return [c for c in categorical_cols
            if c not in {"stage"}
            and not c.endswith("_character")
            and not c.endswith("_action")
            and not c.endswith("_type")
            and not c.endswith("_c_dir")]


def _finalize_dynamic_maps(
    raw_unique: Dict[str, set],
) -> Dict[str, Dict[int, int]]:
    maps: Dict[str, Dict[int, int]] = {}
    for c, s in raw_unique.items():
        vals = sorted(s)
        if 0 not in vals:
            vals.insert(0, 0)
        maps[c] = {raw: idx for idx, raw in enumerate(vals)}
    return maps


# ---------------------------------------------------------------------------
# Cluster centers
# ---------------------------------------------------------------------------
_DEFAULT_CLUSTERS_PATH = Path("data/full/stick_clusters.json")


def load_cluster_centers(data_dir: Path = None, clusters_path: Path = None,
                         stick_clusters: str = None):
    """Load stick_clusters.json, returning (stick_centers, shoulder_centers) or (None, None).

    Resolution order:
      0. Built-in cluster set if *stick_clusters* is ``"hal37"``
      1. Explicit *clusters_path* if given
      2. ``data_dir / stick_clusters.json``
      3. ``data/full/stick_clusters.json`` (canonical default)
    """
    if stick_clusters == "hal37":
        print(f"  Using HAL's 37 hand-designed stick clusters", flush=True)
        # Load shoulder centers from data file (may be 4-bin from preprocessing)
        _, shoulder_from_data = load_cluster_centers(data_dir=data_dir, clusters_path=clusters_path)
        shoulder = shoulder_from_data if shoulder_from_data is not None else np.array([0.0, 0.4, 1.0], dtype=np.float32)
        return HAL_STICK_CLUSTERS_37, shoulder

    candidates = []
    if clusters_path is not None:
        candidates.append(Path(clusters_path))
    if data_dir is not None:
        candidates.append(Path(data_dir) / "stick_clusters.json")
    candidates.append(_DEFAULT_CLUSTERS_PATH)

    for path in candidates:
        if path.exists():
            with open(path) as fh:
                raw = json.load(fh)
            stick = np.array(raw["stick_centers"], dtype=np.float32) if "stick_centers" in raw else None
            shoulder = np.array(raw["shoulder_centers"], dtype=np.float32) if "shoulder_centers" in raw else None
            print(f"  Loaded clusters from {path}", flush=True)
            return stick, shoulder
    return None, None


# ---------------------------------------------------------------------------
# HAL's hand-designed stick clusters (Melee-mechanical angles)
# ---------------------------------------------------------------------------
HAL_STICK_CLUSTERS_37 = np.array([
    [0.5000, 0.5000],  # neutral
    [0.6750, 0.5000], [0.3250, 0.5000], [0.5000, 0.6750], [0.5000, 0.3250],  # partial tilts
    [0.8375, 0.5000], [0.1625, 0.5000], [0.5000, 0.8375], [0.5000, 0.1625],  # full tilts
    [1.0000, 0.5000], [0.5000, 1.0000], [0.0000, 0.5000], [0.5000, 0.0000],  # full press/dash
    [0.9750, 0.3500], [0.0250, 0.3500], [0.9750, 0.6500], [0.0250, 0.6500],  # wavedash 17deg
    [0.9250, 0.2500], [0.9250, 0.7500], [0.0750, 0.2500], [0.0750, 0.7500],  # 30deg
    [0.8500, 0.1500], [0.1500, 0.1500], [0.8500, 0.8500], [0.1500, 0.8500],  # 45deg shield drop
    [0.7500, 0.7500], [0.2500, 0.7500], [0.7500, 0.2500], [0.2500, 0.2500],  # angled f-tilts
    [0.7500, 0.9250], [0.2500, 0.9250], [0.7500, 0.0750], [0.2500, 0.0750],  # 60deg
    [0.6500, 0.0250], [0.6500, 0.9750], [0.3500, 0.0250], [0.3500, 0.9750],  # 72.5deg
], dtype=np.float32)

HAL_CSTICK_CLUSTERS_9 = np.array([
    [0.5000, 0.5000],  # neutral
    [1.0000, 0.5000], [0.0000, 0.5000], [0.5000, 0.0000], [0.5000, 1.0000],  # cardinals
    [0.1500, 0.1500], [0.8500, 0.1500], [0.8500, 0.8500], [0.1500, 0.8500],  # diagonals
], dtype=np.float32)

HAL_SHOULDER_CLUSTERS_3 = np.array([0.0, 0.4, 1.0], dtype=np.float32)

# Map 5-class c_dir → 9-cluster index (verified against HAL_CSTICK_CLUSTERS_9):
#   0(neutral)→0, 1(up)→4, 2(down)→3, 3(left)→2, 4(right)→1
CDIR_5_TO_9_MAP = np.array([0, 4, 3, 2, 1], dtype=np.int64)


# ---------------------------------------------------------------------------
# HAL-style controller encoding
# ---------------------------------------------------------------------------

def load_controller_combos(data_dir: Path):
    """Load controller_combos.json → (combo_list, combo_to_idx dict)."""
    path = Path(data_dir) / "controller_combos.json"
    with open(path) as f:
        data = json.load(f)
    combos = [tuple(c) for c in data["combos"]]
    combo_to_idx = {c: i for i, c in enumerate(combos)}
    return combos, combo_to_idx


def _nearest_2d(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Nearest cluster index for each (x,y) point. points: (N,2), centers: (C,2) → (N,)."""
    dists = np.sum((points[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=-1)
    return np.argmin(dists, axis=1)


def _nearest_1d(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Nearest cluster index for 1D values. values: (N,), centers: (C,) → (N,)."""
    dists = (values[:, np.newaxis] - centers[np.newaxis, :]) ** 2
    return np.argmin(dists, axis=1)


def _collapse_buttons_np(btns: np.ndarray) -> np.ndarray:
    """Collapse 12-dim raw buttons to 5-dim logical: [A, B, Jump, Z, Shoulder].

    Input cols: A=0, B=1, X=2, Y=3, Z=4, L=5, R=6, ...
    """
    a = btns[:, 0] > 0.5
    b = btns[:, 1] > 0.5
    jump = (btns[:, 2] > 0.5) | (btns[:, 3] > 0.5)
    z = btns[:, 4] > 0.5
    shoulder = (btns[:, 5] > 0.5) | (btns[:, 6] > 0.5)
    return np.stack([a, b, jump, z, shoulder], axis=-1).astype(np.int32)


def encode_controller_onehot(
    buttons: np.ndarray,
    analog: np.ndarray,
    c_dir: np.ndarray,
    combo_to_idx: dict,
    n_combos: int,
    norm_stats: dict = None,
) -> np.ndarray:
    """Encode controller state as HAL-style one-hot vector.

    Args:
        buttons: (T, 12) binary float — raw button states
        analog: (T, 4) float — [main_x, main_y, l_shldr, r_shldr], possibly normalized
        c_dir: (T,) int64 — 5-class c-stick direction
        combo_to_idx: dict mapping (int,...) tuples to class indices
        n_combos: total number of button combo classes
        norm_stats: if provided, denormalize analog cols {col_name: (mean, std)}

    Returns:
        (T, 37 + 9 + n_combos + 3) float32 one-hot vector
    """
    T = len(buttons)
    analog = analog.copy().astype(np.float32)

    # Denormalize analog if needed
    if norm_stats is not None:
        for i, col in enumerate(["self_main_x", "self_main_y", "self_l_shldr", "self_r_shldr"]):
            if col in norm_stats:
                mean, std = norm_stats[col]
                analog[:, i] = analog[:, i] * std + mean

    # 1. Main stick → nearest of 37 clusters → 37-dim one-hot
    main_xy = analog[:, 0:2]
    main_idx = _nearest_2d(main_xy, HAL_STICK_CLUSTERS_37)
    main_onehot = np.eye(37, dtype=np.float32)[main_idx]

    # 2. C-stick → 5-class→9-cluster mapping → 9-dim one-hot
    c_idx = CDIR_5_TO_9_MAP[c_dir.astype(np.int64)]
    c_onehot = np.eye(9, dtype=np.float32)[c_idx]

    # 3. Buttons → collapse → lookup combo → n_combos-dim one-hot
    collapsed = _collapse_buttons_np(buttons)
    btn_indices = np.zeros(T, dtype=np.int64)
    for t in range(T):
        combo = tuple(collapsed[t].tolist())
        btn_indices[t] = combo_to_idx.get(combo, 0)  # default to NONE
    btn_onehot = np.eye(n_combos, dtype=np.float32)[btn_indices]

    # 4. Shoulder → max(L,R) → nearest of [0.0, 0.4, 1.0] → 3-dim one-hot
    shldr_max = np.maximum(analog[:, 2], analog[:, 3])
    shldr_idx = _nearest_1d(shldr_max, HAL_SHOULDER_CLUSTERS_3)
    shldr_onehot = np.eye(3, dtype=np.float32)[shldr_idx]

    return np.concatenate([main_onehot, c_onehot, btn_onehot, shldr_onehot], axis=-1)


def encode_controller_onehot_single(
    main_x: float, main_y: float,
    c_x: float, c_y: float,
    l_shldr: float, r_shldr: float,
    buttons: dict,
    combo_to_idx: dict,
    n_combos: int,
) -> np.ndarray:
    """Encode a single frame's controller state (for inference).

    Args:
        main_x, main_y: main stick position [0, 1]
        c_x, c_y: c-stick position [0, 1]
        l_shldr, r_shldr: shoulder values [0, 1]
        buttons: dict {button_name: 0/1} with keys like "BUTTON_A", etc.
        combo_to_idx: combo tuple → class index mapping
        n_combos: total number of combo classes

    Returns:
        (37 + 9 + n_combos + 3,) float32 one-hot vector
    """
    # 1. Main stick
    main_xy = np.array([[main_x, main_y]], dtype=np.float32)
    main_idx = _nearest_2d(main_xy, HAL_STICK_CLUSTERS_37)[0]
    main_onehot = np.eye(37, dtype=np.float32)[main_idx]

    # 2. C-stick — use raw (c_x, c_y) at inference for better accuracy
    c_xy = np.array([[c_x, c_y]], dtype=np.float32)
    c_idx = _nearest_2d(c_xy, HAL_CSTICK_CLUSTERS_9)[0]
    c_onehot = np.eye(9, dtype=np.float32)[c_idx]

    # 3. Buttons — collapse and lookup combo
    a = int(buttons.get("BUTTON_A", 0))
    b = int(buttons.get("BUTTON_B", 0))
    jump = int(buttons.get("BUTTON_X", 0) or buttons.get("BUTTON_Y", 0))
    z = int(buttons.get("BUTTON_Z", 0))
    shoulder = int(buttons.get("BUTTON_L", 0) or buttons.get("BUTTON_R", 0))
    combo = (a, b, jump, z, shoulder)
    btn_idx = combo_to_idx.get(combo, 0)
    btn_onehot = np.eye(n_combos, dtype=np.float32)[btn_idx]

    # 4. Shoulder
    shldr_max = max(l_shldr, r_shldr)
    shldr_idx = _nearest_1d(np.array([shldr_max]), HAL_SHOULDER_CLUSTERS_3)[0]
    shldr_onehot = np.eye(3, dtype=np.float32)[shldr_idx]

    return np.concatenate([main_onehot, c_onehot, btn_onehot, shldr_onehot])


# ---------------------------------------------------------------------------
# C-stick direction encoding
# ---------------------------------------------------------------------------
def encode_cstick_dir(df: pd.DataFrame, prefix: str, dead_zone: float = 0.15) -> None:
    dx = df[f"{prefix}_c_x"].astype("float32") - 0.5
    dy = df[f"{prefix}_c_y"].astype("float32") - 0.5
    mag = np.hypot(dx, dy)
    cat = np.zeros_like(mag, dtype="int64")
    alive = mag > dead_zone
    horiz = alive & (np.abs(dx) >= np.abs(dy))
    vert  = alive & (np.abs(dy) >  np.abs(dx))
    cat[horiz & (dx > 0)] = 4
    cat[horiz & (dx < 0)] = 3
    cat[vert  & (dy > 0)] = 1
    cat[vert  & (dy < 0)] = 2
    df[f"{prefix}_c_dir"] = cat


# ---------------------------------------------------------------------------
# DataFrame preprocessing
# ---------------------------------------------------------------------------
def preprocess_df(
    df: pd.DataFrame,
    categorical_cols: List[str],
    dynamic_maps: Dict[str, Dict[int, int]] | None = None,
) -> pd.DataFrame:
    """Full preprocessing: c-stick encoding, fillna, categorical mapping, nana flags."""
    for p in ("self", "opp", "self_nana", "opp_nana"):
        encode_cstick_dir(df, p, dead_zone=0.15)

    df = df.drop(columns=["startAt"], errors="ignore")

    df["distance"] = np.hypot(
        df["self_pos_x"] - df["opp_pos_x"],
        df["self_pos_y"] - df["opp_pos_y"],
    ).astype("float32")

    # Coerce object-dtype columns that should be numeric (e.g. platform
    # geometry columns that are 100% NaN on stages without platforms).
    for c in df.columns:
        if df[c].dtype == object:
            try:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            except (ValueError, TypeError):
                pass

    num_cols  = [c for c, dt in df.dtypes.items() if dt.kind in ("i", "f")]
    bool_cols = [c for c, dt in df.dtypes.items() if dt == "bool"]
    df[num_cols]  = df[num_cols].fillna(0.0)
    df[bool_cols] = df[bool_cols].fillna(False)

    arr = df[num_cols].astype(np.float32).to_numpy()
    mask = ~np.isfinite(arr)
    if mask.any():
        arr[mask] = 0.0
        df.loc[:, num_cols] = arr

    for c in categorical_cols:
        raw = df[c].fillna(0)
        m = get_enum_map(c, dynamic_maps)
        df[c] = raw.map(lambda x, _m=m: _m.get(x, 0)).astype("int64")

    df = df.copy()

    _zero = pd.Series(0, index=df.index, dtype="int64")
    df["self_nana_present"] = (df.get("self_nana_character", _zero) > 0).astype("float32")
    df["opp_nana_present"]  = (df.get("opp_nana_character", _zero) > 0).astype("float32")

    return df


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
def apply_normalization(
    df: pd.DataFrame,
    norm_stats: Dict[str, Tuple[float, float]],
) -> None:
    """Normalize columns in-place."""
    for c, (mean, std) in norm_stats.items():
        if c in df.columns:
            df[c] = ((df[c].astype(np.float32) - mean) / std).astype(np.float32)


# ---------------------------------------------------------------------------
# DataFrame → tensors
# ---------------------------------------------------------------------------
def df_to_state_tensors(
    df: pd.DataFrame,
    fg: Dict,
) -> Dict[str, torch.Tensor]:
    """Convert a DataFrame slice into the state tensor dict the model expects."""
    state: Dict[str, torch.Tensor] = {}
    for _, meta in walk_groups(fg, return_meta=True):
        cols, ftype, entity = meta["cols"], meta["ftype"], meta["entity"]
        key = f"{entity}_{ftype}" if entity != "global" else ftype

        if ftype == "categorical":
            for col in cols:
                state[col] = torch.from_numpy(df[col].values.copy()).long()
        else:
            arrs = [torch.from_numpy(df[col].astype("float32").values.copy())
                    for col in cols]
            state[key] = torch.stack(arrs, dim=-1) if len(arrs) > 1 else arrs[0]
    return state


def assign_stick_clusters(main_x: np.ndarray, main_y: np.ndarray,
                          centers: np.ndarray) -> np.ndarray:
    """Vectorized nearest-cluster assignment for (N,) arrays.

    Returns int64 array of cluster indices.
    """
    xy = np.stack([main_x, main_y], axis=-1)          # (N, 2)
    dists = np.sum((xy[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
    return dists.argmin(axis=-1).astype(np.int64)


def assign_shoulder_bins(vals: np.ndarray,
                         centers: np.ndarray) -> np.ndarray:
    """Nearest-bin assignment for 1D shoulder values.

    Returns int64 array of bin indices.
    """
    dists = np.abs(vals[:, None] - centers[None, :])
    return dists.argmin(axis=-1).astype(np.int64)


def build_targets_batch(
    df: pd.DataFrame,
    norm_stats: Dict[str, Tuple[float, float]],
    stick_centers: np.ndarray | None = None,
    shoulder_centers: np.ndarray | None = None,
) -> Dict[str, torch.Tensor]:
    """Build target tensors for an entire DataFrame (vectorized).

    Analog targets are denormalized back to original [0,1] space so that
    the model's Sigmoid heads and MSE loss operate on natural values.

    When stick_centers / shoulder_centers are provided, also produces
    discrete cluster/bin index targets for cross-entropy training.
    """
    def _denorm_col(col: str) -> torch.Tensor:
        vals = df[col].astype("float32").values.copy()
        if col in norm_stats:
            mean, std = norm_stats[col]
            vals = vals * std + mean
        return torch.from_numpy(vals)

    targets: Dict[str, torch.Tensor] = {
        "main_x":  _denorm_col("self_main_x"),
        "main_y":  _denorm_col("self_main_y"),
        "l_shldr": _denorm_col("self_l_shldr"),
        "r_shldr": _denorm_col("self_r_shldr"),
    }

    if stick_centers is not None:
        mx = targets["main_x"].numpy()
        my = targets["main_y"].numpy()
        targets["main_cluster"] = torch.from_numpy(
            assign_stick_clusters(mx, my, stick_centers))

    if shoulder_centers is not None:
        targets["l_bin"] = torch.from_numpy(
            assign_shoulder_bins(targets["l_shldr"].numpy(), shoulder_centers))
        targets["r_bin"] = torch.from_numpy(
            assign_shoulder_bins(targets["r_shldr"].numpy(), shoulder_centers))

    c_dir = torch.from_numpy(df["self_c_dir"].values.astype("int64"))
    c_dir_onehot = torch.zeros(len(df), 5, dtype=torch.float32)
    c_dir_onehot.scatter_(1, c_dir.unsqueeze(1), 1.0)
    targets["c_dir"] = c_dir_onehot

    targets["btns"] = torch.from_numpy(
        df[btn_cols("self")].astype("float32").values.copy())

    return targets


# ---------------------------------------------------------------------------
# Numpy-native helpers (no pandas)
# ---------------------------------------------------------------------------
def encode_cstick_dir_np(
    cx: np.ndarray, cy: np.ndarray, dead_zone: float = 0.15,
) -> np.ndarray:
    """Vectorized c-stick direction from raw float32 arrays. Returns int64."""
    dx = cx.astype(np.float32) - 0.5
    dy = cy.astype(np.float32) - 0.5
    mag = np.hypot(dx, dy)
    cat = np.zeros(len(cx), dtype=np.int64)
    alive = mag > dead_zone
    horiz = alive & (np.abs(dx) >= np.abs(dy))
    vert  = alive & (np.abs(dy) >  np.abs(dx))
    cat[horiz & (dx > 0)] = 4
    cat[horiz & (dx < 0)] = 3
    cat[vert  & (dy > 0)] = 1
    cat[vert  & (dy < 0)] = 2
    return cat


def apply_categorical_map_np(
    values: np.ndarray, mapping: Dict[int, int],
) -> np.ndarray:
    """Vectorized categorical mapping using a numpy LUT.

    For maps whose raw keys are small non-negative ints (stage, character,
    action, projectile type), builds a lookup table for O(1) vectorized
    mapping.  Falls back to a Python loop for maps with negative or large keys.
    """
    if not mapping:
        return np.zeros_like(values, dtype=np.int64)
    raw_keys = list(mapping.keys())
    min_k, max_k = min(raw_keys), max(raw_keys)
    # Use LUT when range is reasonable and non-negative
    if min_k >= 0 and (max_k - min_k) < 8192:
        lut = np.zeros(max_k + 1, dtype=np.int64)
        for k, v in mapping.items():
            lut[k] = v
        ival = values.astype(np.int64)
        # Clamp out-of-range to 0 (same as dict .get(x, 0))
        mask = (ival >= 0) & (ival <= max_k)
        out = np.zeros(len(values), dtype=np.int64)
        out[mask] = lut[ival[mask]]
        return out
    # Fallback: Python loop
    get = mapping.get
    return np.array([get(int(v), 0) for v in values], dtype=np.int64)


# ---------------------------------------------------------------------------
# ColumnSchema — maps column names to pre-allocated array positions
# ---------------------------------------------------------------------------
@dataclass
class ColumnSchema:
    """Maps feature column names to positions in pre-allocated numpy arrays.

    Built from walk_groups() so the array layout exactly matches the tensor
    dict produced by df_to_state_tensors().

    Categoricals are 1D int64 arrays (one per column).
    Non-categoricals are always 2D float32 ``(max_frames, width)`` — even for
    width-1 groups like action_elapsed — so writes can always use
    ``arrays[key][frame_i, idx]``.  The conversion step squeezes width-1
    groups back to 1D to match the model's expected shapes.
    """
    # key -> (dtype, width)
    array_specs: Dict[str, Tuple[np.dtype, int]] = field(default_factory=dict)
    # col_name -> (array_key, col_idx)  col_idx=None for categoricals (1D)
    col_to_pos: Dict[str, Tuple[str, int | None]] = field(default_factory=dict)
    # ordered list of all categorical column names
    categorical_cols: List[str] = field(default_factory=list)
    # ordered (tensor_key, ftype, cols) for iterating in walk order
    tensor_layout: List[Tuple[str, str, List[str]]] = field(default_factory=list)
    # (dst_key, dst_idx, src_key, src_idx) — fill duplicate column positions
    # (nana numeric groups have 5 duplicated columns each)
    duplicate_fills: List[Tuple[str, int, str, int]] = field(default_factory=list)

    def allocate(self, max_frames: int) -> Dict[str, np.ndarray]:
        """Return zero-initialized arrays matching the schema."""
        arrays: Dict[str, np.ndarray] = {}
        for key, (dtype, width) in self.array_specs.items():
            if dtype == np.int64:          # categorical — 1D
                arrays[key] = np.zeros(max_frames, dtype=dtype)
            else:                          # grouped — always 2D
                arrays[key] = np.zeros((max_frames, width), dtype=dtype)
        return arrays

    def fill_duplicates(self, arrays: Dict[str, np.ndarray], n: int) -> None:
        """Copy values into duplicate column positions (call after writing)."""
        for dst_key, dst_idx, src_key, src_idx in self.duplicate_fills:
            arrays[dst_key][:n, dst_idx] = arrays[src_key][:n, src_idx]

    def arrays_to_state_tensors(
        self, arrays: Dict[str, np.ndarray], n: int,
    ) -> Dict[str, torch.Tensor]:
        """Convert pre-allocated numpy arrays to the state tensor dict."""
        state: Dict[str, torch.Tensor] = {}
        for key, ftype, cols in self.tensor_layout:
            if ftype == "categorical":
                for col in cols:
                    state[col] = torch.from_numpy(
                        arrays[col][:n].copy()).long()
            else:
                _, width = self.array_specs[key]
                if width == 1:
                    state[key] = torch.from_numpy(
                        arrays[key][:n, 0].astype(np.float32).copy())
                else:
                    state[key] = torch.from_numpy(
                        arrays[key][:n].astype(np.float32).copy())
        return state

    def lookup(self, col_name: str) -> Tuple[str, int | None]:
        """Return (array_key, col_idx) for a column name."""
        return self.col_to_pos[col_name]


def build_column_schema(fg: Dict) -> ColumnSchema:
    """Build a ColumnSchema from a feature group spec.

    Iterates walk_groups() in the same order as df_to_state_tensors(),
    guaranteeing identical tensor key names and column ordering.
    """
    schema = ColumnSchema()
    for _, meta in walk_groups(fg, return_meta=True):
        cols, ftype, entity = meta["cols"], meta["ftype"], meta["entity"]
        key = f"{entity}_{ftype}" if entity != "global" else ftype

        if ftype == "categorical":
            for col in cols:
                schema.array_specs[col] = (np.int64, 1)
                schema.col_to_pos[col] = (col, None)
                schema.categorical_cols.append(col)
            schema.tensor_layout.append((key, ftype, list(cols)))
        else:
            width = len(cols)
            schema.array_specs[key] = (np.float32, width)
            for i, col in enumerate(cols):
                if col in schema.col_to_pos:
                    # Duplicate column (e.g. nana numeric extras that repeat
                    # fields already in numeric_state).  Record as needing a
                    # post-extraction copy from the original position.
                    orig_key, orig_idx = schema.col_to_pos[col]
                    schema.duplicate_fills.append((key, i, orig_key, orig_idx))
                else:
                    schema.col_to_pos[col] = (key, i)
            schema.tensor_layout.append((key, ftype, list(cols)))
    return schema
