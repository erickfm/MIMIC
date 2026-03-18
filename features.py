# features.py
# ---------------------------------------------------------------------------
# Single source of truth for MIMIC feature engineering.
# Used by dataset.py, upload_dataset.py, tensorize.py, and inference.py.
# ---------------------------------------------------------------------------

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from cat_maps import (
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


def numeric_state(prefix: str) -> List[str]:
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
                         no_self_inputs: bool = False) -> Dict[str, Dict]:
    opp_cats = categorical_ids("opp")
    if not no_opp_inputs:
        opp_cats = opp_cats + ["opp_c_dir"]

    opp_nana_cats = ["opp_nana_character", "opp_nana_action"]
    if not no_opp_inputs:
        opp_nana_cats = opp_nana_cats + ["opp_nana_c_dir"]

    opp_group: Dict[str, Any] = {
        "categorical": opp_cats,
        "flags":   flags("opp"),
        "numeric": numeric_state("opp"),
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
        "numeric": numeric_state("self"),
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


def load_cluster_centers(data_dir: Path = None, clusters_path: Path = None):
    """Load stick_clusters.json, returning (stick_centers, shoulder_centers) or (None, None).

    Resolution order:
      1. Explicit *clusters_path* if given
      2. ``data_dir / stick_clusters.json``
      3. ``data/full/stick_clusters.json`` (canonical default)
    """
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
