# features.py
# ---------------------------------------------------------------------------
# Single source of truth for FRAME feature engineering.
# Used by dataset.py, preprocess.py, and inference.py.
# ---------------------------------------------------------------------------

from __future__ import annotations

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
def build_feature_groups() -> Dict[str, Dict]:
    return {
        "global": {
            "numeric": ["distance", "frame", *STAGE_GEOM_COLS],
            "categorical": ["stage"],
        },
        "players": {
            "self": {
                "categorical": categorical_ids("self") + ["self_c_dir"],
                "buttons": btn_cols("self"),
                "flags":   flags("self"),
                "analog":  analog_cols("self"),
                "numeric": numeric_state("self"),
                "action_elapsed": ["self_action_frame"],
            },
            "opp": {
                "categorical": categorical_ids("opp") + ["opp_c_dir"],
                "buttons": btn_cols("opp"),
                "flags":   flags("opp"),
                "analog":  analog_cols("opp"),
                "numeric": numeric_state("opp"),
                "action_elapsed": ["opp_action_frame"],
            },
            "self_nana": {
                "categorical": ["self_nana_character", "self_nana_action", "self_nana_c_dir"],
                "buttons": btn_cols("self_nana"),
                "flags":   flags("self_nana") + ["self_nana_present"],
                "analog":  analog_cols("self_nana"),
                "numeric": numeric_state("self_nana") + [
                    "self_nana_stock", "self_nana_jumps_left",
                    "self_nana_hitlag_left", "self_nana_hitstun_left",
                    "self_nana_invuln_left",
                ],
                "action_elapsed": ["self_nana_action_frame"],
            },
            "opp_nana": {
                "categorical": ["opp_nana_character", "opp_nana_action", "opp_nana_c_dir"],
                "buttons": btn_cols("opp_nana"),
                "flags":   flags("opp_nana") + ["opp_nana_present"],
                "analog":  analog_cols("opp_nana"),
                "numeric": numeric_state("opp_nana") + [
                    "opp_nana_stock", "opp_nana_jumps_left",
                    "opp_nana_hitlag_left", "opp_nana_hitstun_left",
                    "opp_nana_invuln_left",
                ],
                "action_elapsed": ["opp_nana_action_frame"],
            },
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


def build_categorical_mappings_streaming(
    parquet_files: List[Any],
    categorical_cols: List[str],
) -> Dict[str, Dict[int, int]]:
    """Build dense-ID maps by streaming parquet files one at a time."""
    need = _cols_needing_dynamic_map(categorical_cols)
    raw_unique: Dict[str, set] = {c: set() for c in need}
    for f in parquet_files:
        df = pd.read_parquet(f)
        df = df[df["frame"] >= 0]
        for c in need:
            if c in df.columns:
                vals = df[c].dropna().astype("int64")
                raw_unique[c].update(int(v) for v in vals)
    return _finalize_dynamic_maps(raw_unique)


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
# Normalization stats (streaming — one df at a time)
# ---------------------------------------------------------------------------
def update_norm_accumulators(
    df: pd.DataFrame,
    norm_cols: List[str],
    col_sum: Dict[str, float],
    col_sq: Dict[str, float],
    col_n: Dict[str, int],
) -> None:
    """Accumulate running sum / sum-of-squares / count for Welford-style stats."""
    for c in norm_cols:
        if c not in df.columns:
            continue
        vals = df[c].astype(np.float64).values
        valid = np.isfinite(vals)
        v = vals[valid]
        col_sum[c] = col_sum.get(c, 0.0) + v.sum()
        col_sq[c]  = col_sq.get(c, 0.0) + (v ** 2).sum()
        col_n[c]   = col_n.get(c, 0) + len(v)


def finalize_norm_stats(
    norm_cols: List[str],
    col_sum: Dict[str, float],
    col_sq: Dict[str, float],
    col_n: Dict[str, int],
) -> Dict[str, Tuple[float, float]]:
    stats: Dict[str, Tuple[float, float]] = {}
    for c in norm_cols:
        n = col_n.get(c, 0)
        if n > 1:
            mean = col_sum[c] / n
            var  = col_sq[c] / n - mean ** 2
            std  = float(np.sqrt(max(var, 0.0)))
            if std < 1e-6:
                std = 1.0
            stats[c] = (float(mean), std)
        else:
            stats[c] = (0.0, 1.0)
    return stats


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


def build_targets_batch(
    df: pd.DataFrame,
    norm_stats: Dict[str, Tuple[float, float]],
) -> Dict[str, torch.Tensor]:
    """Build target tensors for an entire DataFrame (vectorized).

    Analog targets are denormalized back to original [0,1] space so that
    the model's Sigmoid heads and MSE loss operate on natural values.
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

    c_dir = torch.from_numpy(df["self_c_dir"].values.astype("int64"))
    c_dir_onehot = torch.zeros(len(df), 5, dtype=torch.float32)
    c_dir_onehot.scatter_(1, c_dir.unsqueeze(1), 1.0)
    targets["c_dir"] = c_dir_onehot

    targets["btns"] = torch.from_numpy(
        df[btn_cols("self")].astype("float32").values.copy())

    return targets
