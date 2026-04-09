#!/usr/bin/env python3
"""Direct .slp → .pt shard pipeline.

Replaces the two-step .slp → parquet → .pt pipeline with a single pass:
  .slp → pre-allocated numpy arrays → torch tensors → .pt shards

Usage:
    # Tensorize + upload (streaming, multiprocessing)
    python tools/slp_to_shards.py --slp-dir /data/slp --meta-dir data/full \
        --repo erickfm/mimic-melee --stream --workers 64

    # Tensorize only (no upload)
    python tools/slp_to_shards.py --slp-dir /data/slp --meta-dir data/full \
        --repo erickfm/mimic-melee --no-upload

    # Upload a previously staged directory
    python tools/slp_to_shards.py --staging-dir /data/staged \
        --repo erickfm/mimic-melee --upload-only
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import os as _os
_os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import resource as _resource
_soft, _hard = _resource.getrlimit(_resource.RLIMIT_NOFILE)
_resource.setrlimit(_resource.RLIMIT_NOFILE, (min(_hard, 65536), _hard))



import argparse
import json
import logging
import multiprocessing as mp
import os
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from melee import Console, stages
from melee.enums import Button, Menu

import mimic.features as F
from mimic.features import (
    BTN,
    STAGE_GEOM_COLS,
    ColumnSchema,
    apply_categorical_map_np,
    assign_shoulder_bins,
    assign_stick_clusters,
    build_column_schema,
    build_feature_groups,
    encode_cstick_dir_np,
    get_enum_map,
    load_cluster_centers,
)

log = logging.getLogger(__name__)

# --- Constants ---------------------------------------------------------------
MAX_PROJ = 8
MAX_FRAMES = 30000          # ~8 min at 60fps
MIN_FRAMES = 1500           # ~25 sec — match HAL's minimum game length filter
BTN_ENUMS = [Button[name] for name in BTN]
METADATA_FILES = ["norm_stats.json", "cat_maps.json", "stick_clusters.json"]
_CSTICK_PREFIXES = ("self", "opp", "self_nana", "opp_nana")


# --- Stage geometry (from slippi-frame-extractor) ----------------------------

def _extract_stage_static(stage) -> Dict[str, float]:
    """Extract per-game stage geometry (constant across all frames)."""
    if stage and stage in stages.BLASTZONES:
        b0, b1, b2, b3 = stages.BLASTZONES[stage]
        edge_val = stages.EDGE_POSITION[stage]
        lp0, lp1, lp2 = stages.left_platform_position(stage)
        rp0, rp1, rp2 = stages.right_platform_position(stage)
        tp = stages.top_platform_position(stage)
        tp0, tp1, tp2 = tp if tp[0] is not None else (float("nan"),) * 3
    else:
        b0 = b1 = b2 = b3 = float("nan")
        edge_val =          float("nan")
        lp0 = lp1 = lp2 =  float("nan")
        rp0 = rp1 = rp2 =  float("nan")
        tp0 = tp1 = tp2 =  float("nan")
    return {
        "blastzone_left": b0, "blastzone_right": b1,
        "blastzone_top": b2, "blastzone_bottom": b3,
        "stage_edge_left": -edge_val, "stage_edge_right": edge_val,
        "left_platform_height": lp0, "left_platform_left": lp1,
        "left_platform_right": lp2,
        "right_platform_height": rp0, "right_platform_left": rp1,
        "right_platform_right": rp2,
        "top_platform_height": tp0, "top_platform_left": tp1,
        "top_platform_right": tp2,
    }


# --- Write-plan builder -----------------------------------------------------

def _build_write_map(schema: ColumnSchema, prefix: str, is_nana: bool = False):
    """Pre-compute (array_key, col_idx) for each field of a player prefix.

    Returns dict mapping short_field_name -> (key, idx).
    For categoricals idx is None (1D array).  For grouped idx is int (2D).
    """
    s = schema.col_to_pos
    m: Dict[str, Tuple[str, int | None]] = {}

    # Categoricals
    for f in ["character", "action"]:
        col = f"{prefix}_{f}"
        if col in s:
            m[f] = s[col]
    if not is_nana:
        for f in ["port", "costume"]:
            col = f"{prefix}_{f}"
            if col in s:
                m[f] = s[col]

    # Numeric state fields
    for f in [
        "pos_x", "pos_y", "percent", "stock", "jumps_left",
        "speed_air_x_self", "speed_ground_x_self",
        "speed_x_attack", "speed_y_attack", "speed_y_self",
        "hitlag_left", "hitstun_left", "invuln_left", "shield_strength",
        "ecb_bottom_x", "ecb_bottom_y", "ecb_left_x", "ecb_left_y",
        "ecb_right_x", "ecb_right_y", "ecb_top_x", "ecb_top_y",
    ]:
        col = f"{prefix}_{f}"
        if col in s:
            m[f] = s[col]

    # Flags
    for f in ["on_ground", "off_stage", "facing", "invulnerable",
              "moonwalkwarning"]:
        col = f"{prefix}_{f}"
        if col in s:
            m[f] = s[col]

    # Action elapsed
    col = f"{prefix}_action_frame"
    if col in s:
        m["action_frame"] = s[col]

    # Buttons
    for btn_name in BTN:
        col = f"{prefix}_btn_{btn_name}"
        if col in s:
            m[f"btn_{btn_name}"] = s[col]

    # Analog
    for f in ["main_x", "main_y", "l_shldr", "r_shldr"]:
        col = f"{prefix}_{f}"
        if col in s:
            m[f] = s[col]

    return m


def _build_proj_write_map(schema: ColumnSchema, slot: int):
    """Pre-compute (array_key, col_idx) for projectile slot fields."""
    s = schema.col_to_pos
    m: Dict[str, Tuple[str, int | None]] = {}
    pref = f"proj{slot}"
    for f in ["owner", "type", "subtype"]:
        col = f"{pref}_{f}"
        if col in s:
            m[f] = s[col]
    for f in ["pos_x", "pos_y", "speed_x", "speed_y", "frame"]:
        col = f"{pref}_{f}"
        if col in s:
            m[f] = s[col]
    return m


# --- Per-frame array writes --------------------------------------------------

def _write_player(arrays, wmap, frame_i, ps, port):
    """Write a main player's state into pre-allocated arrays at frame_i."""
    # Categoricals (1D int64 arrays, idx=None)
    # ps.port doesn't exist in file-based replays; match extract.py which
    # falls back to -1 (→ maps to 0 after categorical mapping)
    key, _ = wmap["port"]
    arrays[key][frame_i] = getattr(ps, 'port', -1)
    key, _ = wmap["character"]
    arrays[key][frame_i] = ps.character.value
    key, _ = wmap["action"]
    arrays[key][frame_i] = ps.action.value
    key, _ = wmap["costume"]
    arrays[key][frame_i] = ps.costume

    # Action elapsed (2D float32)
    key, idx = wmap["action_frame"]
    arrays[key][frame_i, idx] = ps.action_frame

    # Position
    key, idx = wmap["pos_x"]
    arrays[key][frame_i, idx] = float(ps.position.x)
    key, idx = wmap["pos_y"]
    arrays[key][frame_i, idx] = float(ps.position.y)

    # Combat state
    key, idx = wmap["percent"]
    arrays[key][frame_i, idx] = float(ps.percent)
    key, idx = wmap["stock"]
    arrays[key][frame_i, idx] = ps.stock
    key, idx = wmap["jumps_left"]
    arrays[key][frame_i, idx] = ps.jumps_left
    key, idx = wmap["shield_strength"]
    arrays[key][frame_i, idx] = float(ps.shield_strength)

    # Speeds
    key, idx = wmap["speed_air_x_self"]
    arrays[key][frame_i, idx] = float(ps.speed_air_x_self)
    key, idx = wmap["speed_ground_x_self"]
    arrays[key][frame_i, idx] = float(ps.speed_ground_x_self)
    key, idx = wmap["speed_x_attack"]
    arrays[key][frame_i, idx] = float(ps.speed_x_attack)
    key, idx = wmap["speed_y_attack"]
    arrays[key][frame_i, idx] = float(ps.speed_y_attack)
    key, idx = wmap["speed_y_self"]
    arrays[key][frame_i, idx] = float(ps.speed_y_self)

    # Hit/stun/invuln
    key, idx = wmap["hitlag_left"]
    arrays[key][frame_i, idx] = ps.hitlag_left
    key, idx = wmap["hitstun_left"]
    arrays[key][frame_i, idx] = ps.hitstun_frames_left
    key, idx = wmap["invuln_left"]
    arrays[key][frame_i, idx] = ps.invulnerability_left

    # ECBs
    key, idx = wmap["ecb_bottom_x"]
    arrays[key][frame_i, idx] = float(ps.ecb_bottom[0])
    key, idx = wmap["ecb_bottom_y"]
    arrays[key][frame_i, idx] = float(ps.ecb_bottom[1])
    key, idx = wmap["ecb_left_x"]
    arrays[key][frame_i, idx] = float(ps.ecb_left[0])
    key, idx = wmap["ecb_left_y"]
    arrays[key][frame_i, idx] = float(ps.ecb_left[1])
    key, idx = wmap["ecb_right_x"]
    arrays[key][frame_i, idx] = float(ps.ecb_right[0])
    key, idx = wmap["ecb_right_y"]
    arrays[key][frame_i, idx] = float(ps.ecb_right[1])
    key, idx = wmap["ecb_top_x"]
    arrays[key][frame_i, idx] = float(ps.ecb_top[0])
    key, idx = wmap["ecb_top_y"]
    arrays[key][frame_i, idx] = float(ps.ecb_top[1])

    # Flags (stored as float32 in the flags group)
    key, idx = wmap["on_ground"]
    arrays[key][frame_i, idx] = float(ps.on_ground)
    key, idx = wmap["off_stage"]
    arrays[key][frame_i, idx] = float(ps.off_stage)
    key, idx = wmap["facing"]
    arrays[key][frame_i, idx] = float(ps.facing)
    key, idx = wmap["invulnerable"]
    arrays[key][frame_i, idx] = float(ps.invulnerable)
    key, idx = wmap["moonwalkwarning"]
    arrays[key][frame_i, idx] = float(ps.moonwalkwarning)

    # Analog (main stick, shoulders)
    key, idx = wmap["main_x"]
    arrays[key][frame_i, idx] = ps.controller_state.main_stick[0]
    key, idx = wmap["main_y"]
    arrays[key][frame_i, idx] = ps.controller_state.main_stick[1]
    key, idx = wmap["l_shldr"]
    arrays[key][frame_i, idx] = ps.controller_state.l_shoulder
    key, idx = wmap["r_shldr"]
    arrays[key][frame_i, idx] = ps.controller_state.r_shoulder

    # Buttons
    btn_state = ps.controller_state.button
    for btn_enum in BTN_ENUMS:
        field_name = f"btn_{btn_enum.name}"
        if field_name in wmap:
            key, idx = wmap[field_name]
            arrays[key][frame_i, idx] = float(btn_state.get(btn_enum, False))


def _write_nana(arrays, wmap, frame_i, nana):
    """Write a Nana (IC partner) state into pre-allocated arrays at frame_i."""
    # Categoricals
    key, _ = wmap["character"]
    arrays[key][frame_i] = nana.character.value
    key, _ = wmap["action"]
    arrays[key][frame_i] = nana.action.value

    # Action elapsed
    key, idx = wmap["action_frame"]
    arrays[key][frame_i, idx] = nana.action_frame

    # Position
    key, idx = wmap["pos_x"]
    arrays[key][frame_i, idx] = float(nana.position.x)
    key, idx = wmap["pos_y"]
    arrays[key][frame_i, idx] = float(nana.position.y)

    # Combat state
    key, idx = wmap["percent"]
    arrays[key][frame_i, idx] = float(nana.percent)
    key, idx = wmap["stock"]
    arrays[key][frame_i, idx] = nana.stock
    key, idx = wmap["jumps_left"]
    arrays[key][frame_i, idx] = nana.jumps_left
    key, idx = wmap["shield_strength"]
    arrays[key][frame_i, idx] = float(nana.shield_strength)

    # Speeds
    key, idx = wmap["speed_air_x_self"]
    arrays[key][frame_i, idx] = float(nana.speed_air_x_self)
    key, idx = wmap["speed_ground_x_self"]
    arrays[key][frame_i, idx] = float(nana.speed_ground_x_self)
    key, idx = wmap["speed_x_attack"]
    arrays[key][frame_i, idx] = float(nana.speed_x_attack)
    key, idx = wmap["speed_y_attack"]
    arrays[key][frame_i, idx] = float(nana.speed_y_attack)
    key, idx = wmap["speed_y_self"]
    arrays[key][frame_i, idx] = float(nana.speed_y_self)

    # Hit/stun/invuln
    key, idx = wmap["hitlag_left"]
    arrays[key][frame_i, idx] = nana.hitlag_left
    key, idx = wmap["hitstun_left"]
    arrays[key][frame_i, idx] = nana.hitstun_frames_left
    key, idx = wmap["invuln_left"]
    arrays[key][frame_i, idx] = nana.invulnerability_left

    # ECBs
    key, idx = wmap["ecb_bottom_x"]
    arrays[key][frame_i, idx] = float(nana.ecb_bottom[0])
    key, idx = wmap["ecb_bottom_y"]
    arrays[key][frame_i, idx] = float(nana.ecb_bottom[1])
    key, idx = wmap["ecb_left_x"]
    arrays[key][frame_i, idx] = float(nana.ecb_left[0])
    key, idx = wmap["ecb_left_y"]
    arrays[key][frame_i, idx] = float(nana.ecb_left[1])
    key, idx = wmap["ecb_right_x"]
    arrays[key][frame_i, idx] = float(nana.ecb_right[0])
    key, idx = wmap["ecb_right_y"]
    arrays[key][frame_i, idx] = float(nana.ecb_right[1])
    key, idx = wmap["ecb_top_x"]
    arrays[key][frame_i, idx] = float(nana.ecb_top[0])
    key, idx = wmap["ecb_top_y"]
    arrays[key][frame_i, idx] = float(nana.ecb_top[1])

    # Flags
    key, idx = wmap["on_ground"]
    arrays[key][frame_i, idx] = float(nana.on_ground)
    key, idx = wmap["off_stage"]
    arrays[key][frame_i, idx] = float(nana.off_stage)
    key, idx = wmap["facing"]
    arrays[key][frame_i, idx] = float(nana.facing)
    key, idx = wmap["invulnerable"]
    arrays[key][frame_i, idx] = float(nana.invulnerable)
    key, idx = wmap["moonwalkwarning"]
    arrays[key][frame_i, idx] = float(nana.moonwalkwarning)

    # Analog
    key, idx = wmap["main_x"]
    arrays[key][frame_i, idx] = nana.controller_state.main_stick[0]
    key, idx = wmap["main_y"]
    arrays[key][frame_i, idx] = nana.controller_state.main_stick[1]
    key, idx = wmap["l_shldr"]
    arrays[key][frame_i, idx] = nana.controller_state.l_shoulder
    key, idx = wmap["r_shldr"]
    arrays[key][frame_i, idx] = nana.controller_state.r_shoulder

    # Buttons
    btn_state = nana.controller_state.button
    for btn_enum in BTN_ENUMS:
        field_name = f"btn_{btn_enum.name}"
        if field_name in wmap:
            key, idx = wmap[field_name]
            arrays[key][frame_i, idx] = float(btn_state.get(btn_enum, False))


def _write_projectiles(arrays, proj_wmaps, frame_i, projectiles):
    """Write up to MAX_PROJ projectiles into pre-allocated arrays at frame_i."""
    for j in range(MAX_PROJ):
        wmap = proj_wmaps[j]
        if j < len(projectiles):
            proj = projectiles[j]
            key, _ = wmap["owner"]
            arrays[key][frame_i] = proj.owner
            key, _ = wmap["type"]
            arrays[key][frame_i] = proj.type.value
            key, _ = wmap["subtype"]
            arrays[key][frame_i] = proj.subtype

            key, idx = wmap["pos_x"]
            arrays[key][frame_i, idx] = float(proj.position.x)
            key, idx = wmap["pos_y"]
            arrays[key][frame_i, idx] = float(proj.position.y)
            key, idx = wmap["speed_x"]
            arrays[key][frame_i, idx] = float(proj.speed.x)
            key, idx = wmap["speed_y"]
            arrays[key][frame_i, idx] = float(proj.speed.y)
            key, idx = wmap["frame"]
            arrays[key][frame_i, idx] = getattr(
                proj, 'frame', getattr(proj, 'expiration_frames', -1))
        else:
            # Absent projectiles: categoricals default to 0 (pre-allocated),
            # numerics default to 0 (pre-allocated).  We set categoricals to
            # sentinel values matching extract.py defaults.
            key, _ = wmap["owner"]
            arrays[key][frame_i] = -1
            key, _ = wmap["type"]
            arrays[key][frame_i] = -1
            key, _ = wmap["subtype"]
            arrays[key][frame_i] = -1
            key, idx = wmap["frame"]
            arrays[key][frame_i, idx] = -1


# --- Preprocessing (in-place on arrays) --------------------------------------

def preprocess_arrays(
    arrays: Dict[str, np.ndarray],
    c_tmps: Dict[str, np.ndarray],
    schema: ColumnSchema,
    norm_stats: Dict[str, Tuple[float, float]],
    cat_maps: Dict[str, Dict[int, int]],
    n: int,
    hal_norm: Dict = None,
) -> Optional[Dict]:
    """In-place preprocessing on pre-allocated arrays (replaces preprocess_df +
    apply_normalization).

    Order matters:
      1. c-stick direction encoding
      2. distance from positions
      3. NaN/inf cleanup
      4. categorical mapping
      5. nana_present flags (must be after cat mapping)
      6. normalization
    """
    # 1. C-stick direction
    for prefix in _CSTICK_PREFIXES:
        cdir_col = f"{prefix}_c_dir"
        if cdir_col not in schema.col_to_pos:
            continue
        cx = c_tmps[f"{prefix}_c_x"][:n]
        cy = c_tmps[f"{prefix}_c_y"][:n]
        cdir = encode_cstick_dir_np(cx, cy, dead_zone=0.15)
        arrays[cdir_col][:n] = cdir

    # 2. Distance (computed from positions, not gs.distance)
    self_pos_key, self_pos_x_idx = schema.col_to_pos["self_pos_x"]
    _, self_pos_y_idx = schema.col_to_pos["self_pos_y"]
    opp_pos_key, opp_pos_x_idx = schema.col_to_pos["opp_pos_x"]
    _, opp_pos_y_idx = schema.col_to_pos["opp_pos_y"]
    dist_key, dist_idx = schema.col_to_pos["distance"]
    arrays[dist_key][:n, dist_idx] = np.hypot(
        arrays[self_pos_key][:n, self_pos_x_idx]
        - arrays[opp_pos_key][:n, opp_pos_x_idx],
        arrays[self_pos_key][:n, self_pos_y_idx]
        - arrays[opp_pos_key][:n, opp_pos_y_idx],
    ).astype(np.float32)

    # 3. NaN/inf cleanup on all float32 arrays
    for key, (dtype, _) in schema.array_specs.items():
        if dtype == np.float32:
            arr = arrays[key][:n]
            mask = ~np.isfinite(arr)
            if mask.any():
                arr[mask] = 0.0

    # 4. Categorical mapping
    for col in schema.categorical_cols:
        m = get_enum_map(col, cat_maps)
        if m:
            arrays[col][:n] = apply_categorical_map_np(arrays[col][:n], m)

    # 5. Nana present flags
    for prefix in ("self_nana", "opp_nana"):
        present_col = f"{prefix}_present"
        char_col = f"{prefix}_character"
        if present_col in schema.col_to_pos and char_col in schema.col_to_pos:
            pres_key, pres_idx = schema.col_to_pos[present_col]
            arrays[pres_key][:n, pres_idx] = (
                arrays[char_col][:n] > 0
            ).astype(np.float32)

    # 5.5. Save raw analog/button/c_dir for controller encoding (before normalization)
    _raw_ctrl_data = None
    if hal_norm is not None:
        _raw_ctrl_data = {}
        for prefix in ("self", "opp"):
            btn_key = f"{prefix}_buttons"
            analog_cols = [f"{prefix}_main_x", f"{prefix}_main_y",
                          f"{prefix}_l_shldr", f"{prefix}_r_shldr"]
            cdir_col = f"{prefix}_c_dir"

            if btn_key in arrays:
                _raw_ctrl_data[f"{prefix}_buttons"] = arrays[btn_key][:n].copy()

            analog_raw = np.zeros((n, 4), dtype=np.float32)
            for i, col in enumerate(analog_cols):
                if col in schema.col_to_pos:
                    key, idx = schema.col_to_pos[col]
                    analog_raw[:, i] = arrays[key][:n, idx].copy()
            _raw_ctrl_data[f"{prefix}_analog"] = analog_raw

            if cdir_col in arrays:
                _raw_ctrl_data[f"{prefix}_c_dir"] = arrays[cdir_col][:n].copy()

    # 6. Normalization
    for col, (mean, std) in norm_stats.items():
        if col not in schema.col_to_pos:
            continue
        key, idx = schema.col_to_pos[col]

        # Check if HAL normalization applies to this column
        hal_params = None
        if hal_norm:
            # col is like "self_pos_x" or "opp_facing" — strip prefix to match hal_norm keys
            for prefix in ("self_", "opp_", "self_nana_", "opp_nana_"):
                if col.startswith(prefix):
                    suffix = col[len(prefix):]
                    if suffix in hal_norm:
                        hal_params = hal_norm[suffix]
                    break

        if hal_params is not None:
            from mimic.features import hal_normalize_array
            if idx is not None:
                arrays[key][:n, idx] = hal_normalize_array(
                    arrays[key][:n, idx].astype(np.float32), hal_params
                ).astype(np.float32)
            else:
                arrays[key][:n] = hal_normalize_array(
                    arrays[key][:n].astype(np.float32), hal_params
                ).astype(np.float32)
        else:
            if idx is not None:
                arrays[key][:n, idx] = (
                    (arrays[key][:n, idx] - mean) / std
                ).astype(np.float32)
            else:
                arrays[key][:n] = (
                    (arrays[key][:n].astype(np.float32) - mean) / std
                ).astype(np.float32)

    return _raw_ctrl_data


# --- Target building --------------------------------------------------------

def build_targets_from_arrays(
    arrays: Dict[str, np.ndarray],
    schema: ColumnSchema,
    norm_stats: Dict[str, Tuple[float, float]],
    stick_centers: Optional[np.ndarray],
    shoulder_centers: Optional[np.ndarray],
    n: int,
    c_tmps: Dict[str, np.ndarray] = None,
) -> Dict[str, torch.Tensor]:
    """Build target tensors from preprocessed arrays (replaces build_targets_batch)."""

    def _denorm(col: str) -> np.ndarray:
        key, idx = schema.col_to_pos[col]
        vals = arrays[key][:n, idx].astype(np.float32).copy()
        if col in norm_stats:
            mean, std = norm_stats[col]
            vals = vals * std + mean
        return vals

    main_x = _denorm("self_main_x")
    main_y = _denorm("self_main_y")
    l_shldr = _denorm("self_l_shldr")
    r_shldr = _denorm("self_r_shldr")

    targets: Dict[str, torch.Tensor] = {
        "main_x":  torch.from_numpy(main_x),
        "main_y":  torch.from_numpy(main_y),
        "l_shldr": torch.from_numpy(l_shldr),
        "r_shldr": torch.from_numpy(r_shldr),
    }

    if stick_centers is not None:
        targets["main_cluster"] = torch.from_numpy(
            assign_stick_clusters(main_x, main_y, stick_centers))
    if shoulder_centers is not None:
        targets["l_bin"] = torch.from_numpy(
            assign_shoulder_bins(l_shldr, shoulder_centers))
        targets["r_bin"] = torch.from_numpy(
            assign_shoulder_bins(r_shldr, shoulder_centers))

    # C-stick targets: 9-cluster from raw x/y if available, else 5-class categorical
    if c_tmps is not None and "self_c_x" in c_tmps:
        from mimic.features import HAL_CSTICK_CLUSTERS_9
        cx = c_tmps["self_c_x"][:n].astype(np.float32)
        cy = c_tmps["self_c_y"][:n].astype(np.float32)
        c_xy = np.stack([cx, cy], axis=-1)
        dists = np.sum((c_xy[:, None, :] - HAL_CSTICK_CLUSTERS_9[None, :, :]) ** 2, axis=-1)
        c_idx = torch.from_numpy(dists.argmin(axis=-1).astype(np.int64))
        c_dir_onehot = torch.zeros(n, 9, dtype=torch.float32)
        c_dir_onehot.scatter_(1, c_idx.unsqueeze(1), 1.0)
        targets["c_dir"] = c_dir_onehot
    else:
        c_dir = torch.from_numpy(arrays["self_c_dir"][:n].copy())
        c_dir_onehot = torch.zeros(n, 5, dtype=torch.float32)
        c_dir_onehot.scatter_(1, c_dir.unsqueeze(1), 1.0)
        targets["c_dir"] = c_dir_onehot

    # Buttons (store both raw multi-hot and HAL-style early_release single-label)
    btn_key = None
    for tkey, ftype, cols in schema.tensor_layout:
        if tkey == "self_buttons":
            btn_key = tkey
            break
    if btn_key is not None:
        raw_btns = arrays[btn_key][:n].astype(np.float32).copy()
        targets["btns"] = torch.from_numpy(raw_btns)

        # HAL-style early_release single-label: A=0, B=1, Jump(X|Y)=2, Z=3, NONE=4
        # Combines X/Y into Jump, resolves multi-press by keeping newest and releasing older
        a = raw_btns[:, 0] > 0.5    # btn_A
        b = raw_btns[:, 1] > 0.5    # btn_B
        jump = (raw_btns[:, 2] > 0.5) | (raw_btns[:, 3] > 0.5)  # X|Y
        z = raw_btns[:, 4] > 0.5    # btn_Z
        stacked = np.stack([a, b, jump, z], axis=-1)  # (n, 4)
        no_btn = ~(a | b | jump | z)

        # Early release: track which buttons are held, keep only newest on change
        single = np.full(n, 4, dtype=np.int64)  # default NONE
        prev_buttons = set()
        for i in range(n):
            curr_buttons = set(np.where(stacked[i])[0])
            if not curr_buttons:
                single[i] = 4  # NONE
                prev_buttons = set()
            elif curr_buttons != prev_buttons:
                new_buttons = curr_buttons - prev_buttons
                if new_buttons:
                    single[i] = min(new_buttons)  # newest press, tie-break by priority
                else:
                    # buttons were released, not pressed — match HAL: NO_BUTTON
                    single[i] = 4
                prev_buttons = curr_buttons
            else:
                # same buttons held — keep previous label
                single[i] = single[i - 1] if i > 0 else (min(curr_buttons) if curr_buttons else 4)
        targets["btns_single"] = torch.from_numpy(single)

    return targets


# --- Core extraction: .slp → (states, targets, n_frames) × 2 ---------------

def extract_replay(
    slp_path: str,
    schema: ColumnSchema,
    norm_stats: Dict[str, Tuple[float, float]],
    cat_maps: Dict[str, Dict[int, int]],
    stick_centers: Optional[np.ndarray],
    shoulder_centers: Optional[np.ndarray],
    hal_norm: Dict = None,
    combo_map: Dict = None,
    n_combos: int = 5,
) -> Optional[List[Tuple[Dict[str, torch.Tensor],
                          Dict[str, torch.Tensor], int]]]:
    """Extract a single .slp replay into two perspective tensor tuples.

    Returns a list of 2 (states, targets, n_frames) tuples, or None on failure.
    """
    try:
        return _extract_replay_inner(
            slp_path, schema, norm_stats, cat_maps,
            stick_centers, shoulder_centers, hal_norm,
            combo_map, n_combos)
    except Exception:
        return None


def _extract_replay_inner(
    slp_path, schema, norm_stats, cat_maps, stick_centers, shoulder_centers,
    hal_norm=None, combo_map=None, n_combos=5,
):
    console = Console(is_dolphin=False, path=slp_path, allow_old_version=True)
    console.connect()

    # Pre-compute write maps for both perspectives
    # P1-perspective: p1=self, p2=opp
    # P2-perspective: p1=opp, p2=self
    p1_self_wmap = _build_write_map(schema, "self")
    p1_opp_wmap  = _build_write_map(schema, "opp")
    p2_self_wmap = _build_write_map(schema, "self")
    p2_opp_wmap  = _build_write_map(schema, "opp")

    p1_self_nana_wmap = _build_write_map(schema, "self_nana", is_nana=True)
    p1_opp_nana_wmap  = _build_write_map(schema, "opp_nana", is_nana=True)
    p2_self_nana_wmap = _build_write_map(schema, "self_nana", is_nana=True)
    p2_opp_nana_wmap  = _build_write_map(schema, "opp_nana", is_nana=True)

    proj_wmaps = [_build_proj_write_map(schema, j) for j in range(MAX_PROJ)]

    # Allocate arrays for both perspectives
    p1_arrays = schema.allocate(MAX_FRAMES)
    p2_arrays = schema.allocate(MAX_FRAMES)

    # Pre-fill absent-nana numeric defaults to -1 (matching extract.py preseed).
    # These fields are in NANA_INTS; categoricals (character, action) are fine
    # at 0 since both -1→0 and 0→0 after categorical mapping.
    _NANA_INT_FIELDS = [
        "action_frame", "hitlag_left", "hitstun_left",
        "invuln_left", "jumps_left", "stock",
    ]
    for prefix in ("self_nana", "opp_nana"):
        for field in _NANA_INT_FIELDS:
            col = f"{prefix}_{field}"
            if col in schema.col_to_pos:
                key, idx = schema.col_to_pos[col]
                for arr_set in (p1_arrays, p2_arrays):
                    if idx is not None:
                        arr_set[key][:, idx] = -1
                    else:
                        arr_set[key][:] = -1

    # C-stick temp arrays (not in schema — used to derive c_dir)
    # Main players default to 0.0 (always overwritten), nana defaults to 0.5
    # (neutral) so absent nana produces c_dir=0, matching the old pipeline
    # where NaN c_stick → NaN magnitude → c_dir=0.
    p1_c_tmps = {}
    p2_c_tmps = {}
    for p in _CSTICK_PREFIXES:
        default = 0.5 if "nana" in p else 0.0
        for ax in ("c_x", "c_y"):
            p1_c_tmps[f"{p}_{ax}"] = np.full(MAX_FRAMES, default,
                                              dtype=np.float32)
            p2_c_tmps[f"{p}_{ax}"] = np.full(MAX_FRAMES, default,
                                              dtype=np.float32)

    # Global write positions
    frame_key, frame_idx = schema.col_to_pos["frame"]
    stage_col = "stage"  # categorical

    # Stage geometry column indices in the global numeric array
    stage_geom_indices = {}
    for geom_col in STAGE_GEOM_COLS[:15]:  # exclude randall (last 3)
        if geom_col in schema.col_to_pos:
            stage_geom_indices[geom_col] = schema.col_to_pos[geom_col]
    randall_indices = {}
    for geom_col in STAGE_GEOM_COLS[15:]:  # randall_height, randall_left, randall_right
        if geom_col in schema.col_to_pos:
            randall_indices[geom_col] = schema.col_to_pos[geom_col]

    stage = None
    stage_static = None
    n_frames = 0

    while True:
        gs = console.step()
        if gs is None:
            break
        if gs.menu_state != Menu.IN_GAME:
            continue
        if gs.frame < 0:
            continue

        fi = n_frames
        if fi >= MAX_FRAMES:
            break

        # Stage geometry (computed once per game)
        if stage_static is None:
            stage = gs.stage
            stage_static = _extract_stage_static(stage)
            # Write static geometry to all frames later (bulk write)

        # Frame number (global numeric)
        for arr_set in (p1_arrays, p2_arrays):
            arr_set[frame_key][fi, frame_idx] = gs.frame

        # Stage categorical
        for arr_set in (p1_arrays, p2_arrays):
            arr_set[stage_col][fi] = stage.value if stage else 0

        # Randall (dynamic, Yoshi's Story only)
        if stage and stage.name == "YOSHIS_STORY":
            r0, r1, r2 = stages.randall_position(gs.frame)
        else:
            r0 = r1 = r2 = float("nan")
        randall_vals = {"randall_height": r0, "randall_left": r1,
                        "randall_right": r2}
        for geom_col, val in randall_vals.items():
            if geom_col in randall_indices:
                key, idx = randall_indices[geom_col]
                for arr_set in (p1_arrays, p2_arrays):
                    arr_set[key][fi, idx] = val

        # Players
        players = list(gs.players.items())
        if len(players) < 2:
            continue  # skip frame; fi position will be overwritten next frame

        port1, ps1 = players[0]
        port2, ps2 = players[1]

        # Projectiles (same in both perspectives)
        for arr_set in (p1_arrays, p2_arrays):
            _write_projectiles(arr_set, proj_wmaps, fi, gs.projectiles)

        # P1-perspective: ps1=self, ps2=opp
        _write_player(p1_arrays, p1_self_wmap, fi, ps1, port1)
        _write_player(p1_arrays, p1_opp_wmap, fi, ps2, port2)
        # C-stick temps for p1-perspective
        p1_c_tmps["self_c_x"][fi] = ps1.controller_state.c_stick[0]
        p1_c_tmps["self_c_y"][fi] = ps1.controller_state.c_stick[1]
        p1_c_tmps["opp_c_x"][fi] = ps2.controller_state.c_stick[0]
        p1_c_tmps["opp_c_y"][fi] = ps2.controller_state.c_stick[1]

        # P2-perspective: ps2=self, ps1=opp
        _write_player(p2_arrays, p2_self_wmap, fi, ps2, port2)
        _write_player(p2_arrays, p2_opp_wmap, fi, ps1, port1)
        # C-stick temps for p2-perspective
        p2_c_tmps["self_c_x"][fi] = ps2.controller_state.c_stick[0]
        p2_c_tmps["self_c_y"][fi] = ps2.controller_state.c_stick[1]
        p2_c_tmps["opp_c_x"][fi] = ps1.controller_state.c_stick[0]
        p2_c_tmps["opp_c_y"][fi] = ps1.controller_state.c_stick[1]

        # Nana (p1)
        nana1 = ps1.nana
        if nana1:
            _write_nana(p1_arrays, p1_self_nana_wmap, fi, nana1)
            _write_nana(p2_arrays, p2_opp_nana_wmap, fi, nana1)
            p1_c_tmps["self_nana_c_x"][fi] = nana1.controller_state.c_stick[0]
            p1_c_tmps["self_nana_c_y"][fi] = nana1.controller_state.c_stick[1]
            p2_c_tmps["opp_nana_c_x"][fi] = nana1.controller_state.c_stick[0]
            p2_c_tmps["opp_nana_c_y"][fi] = nana1.controller_state.c_stick[1]

        # Nana (p2)
        nana2 = ps2.nana
        if nana2:
            _write_nana(p1_arrays, p1_opp_nana_wmap, fi, nana2)
            _write_nana(p2_arrays, p2_self_nana_wmap, fi, nana2)
            p1_c_tmps["opp_nana_c_x"][fi] = nana2.controller_state.c_stick[0]
            p1_c_tmps["opp_nana_c_y"][fi] = nana2.controller_state.c_stick[1]
            p2_c_tmps["self_nana_c_x"][fi] = nana2.controller_state.c_stick[0]
            p2_c_tmps["self_nana_c_y"][fi] = nana2.controller_state.c_stick[1]

        n_frames += 1

    if n_frames < MIN_FRAMES:
        return None

    # --- HAL-matching game quality filters ---
    # Read percent and stock from p1-perspective arrays (self=p1, opp=p2)
    _p1_pct_key, _p1_pct_idx = p1_self_wmap["percent"]
    _p2_pct_key, _p2_pct_idx = p1_opp_wmap["percent"]
    _p1_stk_key, _p1_stk_idx = p1_self_wmap["stock"]
    _p2_stk_key, _p2_stk_idx = p1_opp_wmap["stock"]

    p1_pct = p1_arrays[_p1_pct_key][:n_frames, _p1_pct_idx]
    p2_pct = p1_arrays[_p2_pct_key][:n_frames, _p2_pct_idx]
    p1_stk = p1_arrays[_p1_stk_key][:n_frames, _p1_stk_idx]
    p2_stk = p1_arrays[_p2_stk_key][:n_frames, _p2_stk_idx]

    # Damage check: skip if either player took 0 damage the entire game
    if np.all(p1_pct == 0) or np.all(p2_pct == 0):
        return None

    # Completion check: skip if neither player lost all stocks
    if p1_stk[-1] != 0 and p2_stk[-1] != 0:
        return None

    # Bulk-write static stage geometry (constant across all frames)
    if stage_static:
        for geom_col, val in stage_static.items():
            if geom_col in stage_geom_indices:
                key, idx = stage_geom_indices[geom_col]
                for arr_set in (p1_arrays, p2_arrays):
                    arr_set[key][:n_frames, idx] = val

    # Preprocess both perspectives (before filling duplicates, so that
    # duplicate columns get the already-normalized values)
    p1_raw_ctrl = preprocess_arrays(p1_arrays, p1_c_tmps, schema, norm_stats, cat_maps, n_frames, hal_norm)
    p2_raw_ctrl = preprocess_arrays(p2_arrays, p2_c_tmps, schema, norm_stats, cat_maps, n_frames, hal_norm)

    # Fill duplicate column positions (nana numeric extras — copies the
    # normalized values from the original positions)
    schema.fill_duplicates(p1_arrays, n_frames)
    schema.fill_duplicates(p2_arrays, n_frames)

    # Controller combo map: prefer parameter, fall back to worker state
    _combo_map_local = combo_map
    _n_combos_local = n_combos
    if _combo_map_local is None and _W:
        _combo_map_local = _W.get("combo_map")
        _n_combos_local = _W.get("n_combos", 5)

    # Convert to tensors
    results = []
    for arr_set, c_tmp, raw_ctrl in [(p1_arrays, p1_c_tmps, p1_raw_ctrl),
                                      (p2_arrays, p2_c_tmps, p2_raw_ctrl)]:
        states = schema.arrays_to_state_tensors(arr_set, n_frames)
        targets = build_targets_from_arrays(
            arr_set, schema, norm_stats, stick_centers, shoulder_centers, n_frames,
            c_tmps=c_tmp)

        # Bake controller encoding into state tensors
        if raw_ctrl is not None and _combo_map_local is not None:
            from mimic.features import encode_controller_onehot
            onehot = encode_controller_onehot(
                raw_ctrl["self_buttons"],
                raw_ctrl["self_analog"],
                raw_ctrl["self_c_dir"],
                _combo_map_local, _n_combos_local,
                norm_stats=None,  # values are already raw
            )
            states["self_controller"] = torch.from_numpy(onehot)
            for k in ("self_buttons", "self_analog", "self_c_dir"):
                states.pop(k, None)

        results.append((states, targets, n_frames))

    return results


# --- Sharding ----------------------------------------------------------------

def flush_shard(buf_states, buf_targets, buf_offsets, prefix, shard_idx,
                out_dir):
    """Concatenate buffered games and save a shard .pt file."""
    offsets = torch.tensor(buf_offsets, dtype=torch.int64)
    states = {k: torch.cat([s[k] for s in buf_states], dim=0)
              for k in buf_states[0]}
    targets = {k: torch.cat([t[k] for t in buf_targets], dim=0)
               for k in buf_targets[0]}

    fname = f"{prefix}_shard_{shard_idx:03d}.pt"
    torch.save({
        "states": states,
        "targets": targets,
        "offsets": offsets,
        "n_games": len(buf_states),
    }, out_dir / fname)

    total_frames = offsets[-1].item()
    size_bytes = (out_dir / fname).stat().st_size
    print(f"    {fname}: {len(buf_states)} games, {total_frames:,} frames "
          f"({size_bytes / 1e9:.2f} GB)", flush=True)
    return fname, len(buf_states), total_frames, size_bytes


# --- Worker pool -------------------------------------------------------------

_W: Dict = {}


def _init_worker(schema, norm_stats, cat_maps, stick_centers, shoulder_centers,
                  hal_norm=None, combo_map=None, n_combos=5):
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(hard, 65536), hard))
    _W["schema"] = schema
    _W["norm_stats"] = norm_stats
    _W["cat_maps"] = cat_maps
    _W["stick_centers"] = stick_centers
    _W["shoulder_centers"] = shoulder_centers
    _W["hal_norm"] = hal_norm
    _W["combo_map"] = combo_map
    _W["n_combos"] = n_combos


def _worker_fn(slp_path: str):
    """Process a single .slp file, return list of (states, targets, n_frames).

    Returns numpy arrays to avoid torch's /dev/shm-based tensor sharing
    which exhausts shared memory with many workers.
    """
    result = extract_replay(
        slp_path, _W["schema"], _W["norm_stats"], _W["cat_maps"],
        _W["stick_centers"], _W["shoulder_centers"], _W.get("hal_norm"),
        _W.get("combo_map"), _W.get("n_combos", 5),
    )
    if result is None:
        return None
    return [
        ({k: v.numpy() for k, v in states.items()},
         {k: v.numpy() for k, v in targets.items()},
         n_frames)
        for states, targets, n_frames in result
    ]


def _numpy_to_torch(result):
    """Convert worker's numpy arrays back to torch tensors."""
    if result is None:
        return None
    return [
        ({k: torch.from_numpy(v) for k, v in states.items()},
         {k: torch.from_numpy(v) for k, v in targets.items()},
         n_frames)
        for states, targets, n_frames in result
    ]


# --- Result iterator ---------------------------------------------------------

@contextmanager
def _make_result_iter(slp_files, schema, norm_stats, cat_maps,
                      stick_centers, shoulder_centers, n_workers,
                      hal_norm=None, combo_map=None, n_combos=5):
    """Yield an iterator of extract_replay results."""
    if n_workers > 1:
        pool = mp.Pool(
            n_workers,
            initializer=_init_worker,
            initargs=(schema, norm_stats, cat_maps,
                      stick_centers, shoulder_centers, hal_norm,
                      combo_map, n_combos),
            maxtasksperchild=500,
        )
        try:
            yield pool.imap_unordered(_worker_fn, slp_files, chunksize=1)
        finally:
            pool.close()
            pool.join()
    else:
        def _serial():
            for path in slp_files:
                yield extract_replay(
                    path, schema, norm_stats, cat_maps,
                    stick_centers, shoulder_centers, hal_norm,
                    combo_map, n_combos,
                )
        yield _serial()


# --- Split -------------------------------------------------------------------

def _get_split_files(slp_files: List[str], val_frac: float,
                     seed: int) -> Dict[str, List[str]]:
    """Split .slp files into train/val by filename."""
    rng = random.Random(seed)
    shuffled = list(slp_files)
    rng.shuffle(shuffled)
    n_val = int(len(shuffled) * val_frac)
    return {
        "train": sorted(shuffled[n_val:]),
        "val": sorted(shuffled[:n_val]) if n_val > 0 else [],
    }


# --- Manifest ----------------------------------------------------------------

def _build_manifest(results, val_frac, seed):
    train_shards = [r[0] for r in results.get("train", [])]
    val_shards = [r[0] for r in results.get("val", [])]
    return {
        "train_shards": train_shards,
        "val_shards": val_shards,
        "n_train_games": sum(r[1] for r in results.get("train", [])),
        "n_val_games": sum(r[1] for r in results.get("val", [])),
        "n_train_frames": sum(r[2] for r in results.get("train", [])),
        "n_val_frames": sum(r[2] for r in results.get("val", [])),
        "val_frac": val_frac,
        "seed": seed,
    }


def _print_summary(manifest):
    print(f"\n  Manifest: {len(manifest['train_shards'])} train shards "
          f"({manifest['n_train_games']} games, "
          f"{manifest['n_train_frames']:,} frames), "
          f"{len(manifest['val_shards'])} val shards "
          f"({manifest['n_val_games']} games, "
          f"{manifest['n_val_frames']:,} frames)")


# --- Resume state ------------------------------------------------------------

def _save_resume_state(staging_dir, split, next_shard_idx, n_files_done, infos):
    """Atomically save resume checkpoint after a confirmed shard upload."""
    state = {
        "next_shard_idx": next_shard_idx,
        "n_files_done": n_files_done,
        "infos": infos,
    }
    path = staging_dir / f"_resume_{split}.json"
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f)
    tmp.rename(path)


def _load_resume_state(staging_dir, split):
    """Load resume checkpoint, or return None if not found."""
    path = staging_dir / f"_resume_{split}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# --- Metadata staging --------------------------------------------------------

def stage_metadata(meta_dir: Path, staging_dir: Path) -> None:
    staging_dir.mkdir(parents=True, exist_ok=True)
    for name in METADATA_FILES:
        src = meta_dir / name
        if src.exists():
            shutil.copy2(src, staging_dir / name)
            print(f"  Copied {name}")


# --- Upload ------------------------------------------------------------------

def _upload_shard(api, repo_id, local_path, fname):
    t0 = time.time()
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=fname,
        repo_id=repo_id,
        repo_type="dataset",
    )
    dt = time.time() - t0
    size_gb = local_path.stat().st_size / 1e9
    local_path.unlink()
    return fname, dt, size_gb


def do_upload(staging_dir: Path, repo_id: str, clean: bool = False) -> None:
    from huggingface_hub import HfApi, upload_large_folder
    api = HfApi()
    if clean:
        try:
            api.delete_repo(repo_id=repo_id, repo_type="dataset")
            print(f"  Deleted old repo {repo_id}")
        except Exception:
            pass
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    print(f"  Uploading to https://huggingface.co/datasets/{repo_id} ...")
    upload_large_folder(
        folder_path=str(staging_dir),
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"  Upload complete.")


# --- Batch mode --------------------------------------------------------------

def create_tensor_shards(
    slp_files: List[str],
    meta_dir: Path,
    staging_dir: Path,
    shard_gb: float,
    val_frac: float,
    seed: int,
    n_workers: int = 0,
    hal_norm: Dict = None,
    character_filter: int = None,
) -> Dict:
    schema, norm_stats, cat_maps, stick_centers, shoulder_centers = (
        _load_prereqs(meta_dir))
    splits = _get_split_files(slp_files, val_frac, seed)
    staging_dir.mkdir(parents=True, exist_ok=True)
    shard_bytes = int(shard_gb * 1e9)

    # Load controller combos if using HAL normalization
    combo_map_local = None
    n_combos_local = 5
    if hal_norm is not None:
        cc_path = meta_dir / "controller_combos.json"
        if cc_path.exists():
            from mimic.features import load_controller_combos
            _, combo_map_local = load_controller_combos(meta_dir)
            n_combos_local = len(combo_map_local)
            print(f"  Controller encoding: {n_combos_local} combos from {cc_path}")

    results = {}
    for split, split_files in splits.items():
        if not split_files:
            results[split] = []
            continue
        print(f"\n  Tensorizing {len(split_files)} {split} .slp files "
              f"({n_workers} workers) ...")
        shard_infos = _tensorize_split(
            split_files, split, staging_dir, shard_bytes,
            schema, norm_stats, cat_maps, stick_centers, shoulder_centers,
            n_workers, hal_norm, combo_map_local, n_combos_local,
            character_filter=character_filter,
        )
        results[split] = shard_infos

    manifest = _build_manifest(results, val_frac, seed)
    with open(staging_dir / "tensor_manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    _print_summary(manifest)
    return manifest


def _tensorize_split(
    split_files, split, out_dir, shard_bytes,
    schema, norm_stats, cat_maps, stick_centers, shoulder_centers, n_workers,
    hal_norm=None, combo_map=None, n_combos=5, character_filter=None,
):
    buf_states: List = []
    buf_targets: List = []
    buf_offsets = [0]
    buf_bytes = 0
    shard_idx = 0
    n_processed = 0
    n_games = 0
    infos = []

    with _make_result_iter(
        split_files, schema, norm_stats, cat_maps,
        stick_centers, shoulder_centers, n_workers, hal_norm,
        combo_map, n_combos,
    ) as result_iter:
        for raw_result in result_iter:
            result = _numpy_to_torch(raw_result) if n_workers > 1 else raw_result
            n_processed += 1
            if result is None:
                continue

            # Each result is a list of 2 (states, targets, n_frames) tuples
            for states, targets, n_frames in result:
                # Character filter: only keep perspective where self matches
                if character_filter is not None:
                    if states["self_character"][0].item() != character_filter:
                        continue
                buf_states.append(states)
                buf_targets.append(targets)
                buf_offsets.append(buf_offsets[-1] + n_frames)
                n_games += 1

                frame_bytes = sum(
                    v.nelement() * v.element_size() for v in states.values())
                frame_bytes += sum(
                    v.nelement() * v.element_size() for v in targets.values())
                buf_bytes += frame_bytes

                if buf_bytes >= shard_bytes:
                    info = flush_shard(buf_states, buf_targets, buf_offsets,
                                       split, shard_idx, out_dir)
                    infos.append(info)
                    shard_idx += 1
                    buf_states, buf_targets = [], []
                    buf_offsets, buf_bytes = [0], 0

            if n_processed % 500 == 0:
                print(f"      [{n_processed}/{len(split_files)}] .slp files, "
                      f"{n_games} games ...", flush=True)

        if buf_states:
            info = flush_shard(buf_states, buf_targets, buf_offsets,
                               split, shard_idx, out_dir)
            infos.append(info)

    return infos


# --- Streaming mode ----------------------------------------------------------

def create_and_stream_upload(
    slp_files: List[str],
    meta_dir: Path,
    staging_dir: Path,
    shard_gb: float,
    val_frac: float,
    seed: int,
    repo_id: str,
    clean: bool,
    n_workers: int = 0,
    resume: bool = False,
    hal_norm: Dict = None,
) -> Dict:
    from huggingface_hub import HfApi
    api = HfApi()
    if clean and not resume:
        try:
            api.delete_repo(repo_id=repo_id, repo_type="dataset")
            print(f"  Deleted old repo {repo_id}")
        except Exception:
            pass
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

    schema, norm_stats, cat_maps, stick_centers, shoulder_centers = (
        _load_prereqs(meta_dir))
    splits = _get_split_files(slp_files, val_frac, seed)

    staging_dir.mkdir(parents=True, exist_ok=True)
    shard_bytes = int(shard_gb * 1e9)

    combo_map_local = None
    n_combos_local = 5
    if hal_norm is not None:
        cc_path = meta_dir / "controller_combos.json"
        if cc_path.exists():
            from mimic.features import load_controller_combos
            _, combo_map_local = load_controller_combos(meta_dir)
            n_combos_local = len(combo_map_local)

    if not resume:
        print(f"\n=== Uploading metadata ===")
        stage_metadata(meta_dir, staging_dir)
        for name in METADATA_FILES:
            src = staging_dir / name
            if src.exists():
                api.upload_file(
                    path_or_fileobj=str(src),
                    path_in_repo=name,
                    repo_id=repo_id,
                    repo_type="dataset",
                )
                print(f"  Uploaded {name}")

    results = {}
    for split, split_files in splits.items():
        if not split_files:
            results[split] = []
            continue

        resume_state = _load_resume_state(staging_dir, split) if resume else None
        if resume_state and resume_state["n_files_done"] >= len(split_files):
            print(f"\n=== Split '{split}' already complete "
                  f"({len(resume_state['infos'])} shards) ===")
            results[split] = [tuple(i) for i in resume_state["infos"]]
            continue

        remaining = (len(split_files) - resume_state["n_files_done"]
                     if resume_state else len(split_files))
        print(f"\n=== Tensorizing + streaming {remaining} {split} "
              f".slp files ({n_workers} workers) ===")
        shard_infos = _tensorize_split_streaming(
            split_files, split, staging_dir, shard_bytes,
            schema, norm_stats, cat_maps, stick_centers, shoulder_centers,
            api, repo_id, n_workers, resume_state=resume_state,
        )
        results[split] = shard_infos

    manifest = _build_manifest(results, val_frac, seed)
    manifest_path = staging_dir / "tensor_manifest.json"
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    api.upload_file(
        path_or_fileobj=str(manifest_path),
        path_in_repo="tensor_manifest.json",
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"\n  Uploaded tensor_manifest.json")
    _print_summary(manifest)
    return manifest


def _tensorize_split_streaming(
    split_files, split, out_dir, shard_bytes,
    schema, norm_stats, cat_maps, stick_centers, shoulder_centers,
    api, repo_id, n_workers, resume_state=None,
):
    total_split_files = len(split_files)

    # Resume from checkpoint
    if resume_state:
        skip = resume_state["n_files_done"]
        shard_idx = resume_state["next_shard_idx"]
        infos = [tuple(i) for i in resume_state["infos"]]
        n_games = sum(i[1] for i in infos)
        split_files = split_files[skip:]
        n_offset = skip
        print(f"    Resuming from shard {shard_idx}, "
              f"skipping {skip} already-processed files "
              f"({n_games} games in {len(infos)} shards)")
    else:
        shard_idx = 0
        infos = []
        n_games = 0
        n_offset = 0

    buf_states: List = []
    buf_targets: List = []
    buf_offsets = [0]
    buf_bytes = 0
    n_processed = 0

    upload_pool = ThreadPoolExecutor(max_workers=1)
    pending_upload = None
    pending_n_files = 0

    def _wait_pending():
        nonlocal pending_upload
        if pending_upload is not None:
            fname, dt, size_gb = pending_upload.result()
            print(f"      Uploaded {fname} ({size_gb:.2f} GB in {dt:.1f}s, "
                  f"{size_gb/max(dt,0.01):.2f} GB/s)", flush=True)
            _save_resume_state(
                out_dir, split, shard_idx, pending_n_files, infos)
            pending_upload = None

    def _flush_and_submit():
        nonlocal buf_states, buf_targets, buf_offsets, buf_bytes, shard_idx
        nonlocal pending_upload, pending_n_files
        _wait_pending()
        pending_n_files = n_processed + n_offset
        info = flush_shard(buf_states, buf_targets, buf_offsets,
                           split, shard_idx, out_dir)
        fname = info[0]
        local_path = out_dir / fname
        pending_upload = upload_pool.submit(
            _upload_shard, api, repo_id, local_path, fname,
        )
        infos.append(info)
        shard_idx += 1
        buf_states, buf_targets, buf_offsets, buf_bytes = [], [], [0], 0

    with _make_result_iter(
        split_files, schema, norm_stats, cat_maps,
        stick_centers, shoulder_centers, n_workers, hal_norm,
        combo_map_local, n_combos_local,
    ) as result_iter:
        t_split = time.time()
        for raw_result in result_iter:
            result = _numpy_to_torch(raw_result) if n_workers > 1 else raw_result
            n_processed += 1
            if result is None:
                continue

            for states, targets, n_frames in result:
                buf_states.append(states)
                buf_targets.append(targets)
                buf_offsets.append(buf_offsets[-1] + n_frames)
                n_games += 1

                frame_bytes = sum(
                    v.nelement() * v.element_size() for v in states.values())
                frame_bytes += sum(
                    v.nelement() * v.element_size() for v in targets.values())
                buf_bytes += frame_bytes

            # Flush at file boundaries (all games from this file buffered)
            if buf_bytes >= shard_bytes:
                _flush_and_submit()

            if n_processed % 500 == 0:
                total_done = n_processed + n_offset
                elapsed = time.time() - t_split
                rate = n_processed / elapsed
                eta = (len(split_files) - n_processed) / max(rate, 1)
                print(f"      [{total_done}/{total_split_files}] .slp files, "
                      f"{n_games} games  ({rate:.0f} files/s, "
                      f"ETA {eta/60:.0f}m)", flush=True)

        if buf_states:
            _flush_and_submit()

    _wait_pending()
    _save_resume_state(
        out_dir, split, shard_idx, n_processed + n_offset, infos)
    upload_pool.shutdown()
    return infos


# --- Helpers -----------------------------------------------------------------

def _load_prereqs(meta_dir: Path):
    with open(meta_dir / "norm_stats.json") as fh:
        norm_stats = json.load(fh)
    with open(meta_dir / "cat_maps.json") as fh:
        raw = json.load(fh)
        cat_maps = {col: {int(k): v for k, v in m.items()}
                    for col, m in raw.items()}
    stick_centers, shoulder_centers = load_cluster_centers(meta_dir)
    fg = build_feature_groups()
    schema = build_column_schema(fg)
    return schema, norm_stats, cat_maps, stick_centers, shoulder_centers


def _find_slp_files(slp_dir: str) -> List[str]:
    """Find all .slp files in a directory."""
    slp_dir = Path(slp_dir)
    files = sorted(
        str(f) for f in slp_dir.iterdir()
        if f.suffix.lower() == ".slp"
    )
    return files


# --- CLI ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Direct .slp to .pt shard pipeline")
    parser.add_argument("--slp-dir", type=str, default=None,
                        help="Directory containing .slp replay files")
    parser.add_argument("--meta-dir", type=str, default=None,
                        help="Directory with norm_stats.json, cat_maps.json, "
                             "stick_clusters.json")
    parser.add_argument("--repo", required=True,
                        help="HuggingFace repo ID (e.g. erickfm/mimic-melee)")
    parser.add_argument("--shard-gb", type=float, default=4.0,
                        help="Target shard size in GB (default: 4)")
    parser.add_argument("--staging-dir", default=None,
                        help="Staging directory for shards")
    parser.add_argument("--keep-staging", action="store_true",
                        help="Don't delete staging dir after upload")
    parser.add_argument("--val-frac", type=float, default=0.1,
                        help="Fraction of .slp files for validation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-upload", action="store_true",
                        help="Tensorize only, skip upload")
    parser.add_argument("--upload-only", action="store_true",
                        help="Upload an existing staging directory")
    parser.add_argument("--clean", action="store_true",
                        help="Delete existing HF repo before uploading")
    parser.add_argument("--stream", action="store_true",
                        help="Upload each shard immediately after creation")
    parser.add_argument("--resume", action="store_true",
                        help="Resume an interrupted --stream run")
    parser.add_argument("--workers", type=int, default=0,
                        help="Parallel workers (0=auto)")
    parser.add_argument("--hal-norm", type=str, default=None,
                        help="Path to hal_norm.json for HAL-style normalization")
    parser.add_argument("--character", type=int, default=None,
                        help="Only keep perspective where self_character matches "
                             "this index (e.g. 1 for Fox)")
    args = parser.parse_args()

    if args.resume and args.clean:
        parser.error("--resume and --clean are mutually exclusive")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    n_workers = args.workers if args.workers > 0 else min(
        os.cpu_count() or 4, 64)

    if args.upload_only:
        if not args.staging_dir:
            parser.error("--upload-only requires --staging-dir")
        staging_dir = Path(args.staging_dir)
    else:
        if not args.slp_dir:
            parser.error("--slp-dir is required unless --upload-only")
        if not args.meta_dir:
            parser.error("--meta-dir is required unless --upload-only")
        meta_dir = Path(args.meta_dir)
        staging_dir = (Path(args.staging_dir) if args.staging_dir
                       else Path(args.slp_dir).parent / "shards_upload")

    t0 = time.time()
    print(f"Workers: {n_workers}")

    if args.upload_only:
        print(f"\n=== Uploading staged directory ===")
        do_upload(staging_dir, args.repo, clean=args.clean)
    else:
        slp_files = _find_slp_files(args.slp_dir)
        print(f"Found {len(slp_files)} .slp files in {args.slp_dir}")

        hal_norm = None
        if args.hal_norm:
            with open(args.hal_norm) as f:
                hal_norm = json.load(f)["features"]
            print(f"  HAL normalization: {len(hal_norm)} features from {args.hal_norm}")

        if not slp_files:
            print("No .slp files found. Exiting.")
            return

        if args.stream and not args.no_upload:
            print(f"\n=== Streaming tensorize + upload ===")
            create_and_stream_upload(
                slp_files, meta_dir, staging_dir, args.shard_gb,
                args.val_frac, args.seed, args.repo, args.clean, n_workers,
                resume=args.resume, hal_norm=hal_norm,
            )
        else:
            print(f"\n=== Creating tensor shards ===")
            manifest = create_tensor_shards(
                slp_files, meta_dir, staging_dir, args.shard_gb,
                args.val_frac, args.seed, n_workers, hal_norm,
                character_filter=args.character,
            )
            print(f"\n=== Staging metadata ===")
            stage_metadata(meta_dir, staging_dir)

            if not args.no_upload:
                print(f"\n=== Uploading to HuggingFace ===")
                do_upload(staging_dir, args.repo, clean=args.clean)

    if not args.keep_staging and staging_dir.exists():
        print(f"\n=== Cleaning up staging directory ===")
        shutil.rmtree(staging_dir)
        print(f"  Removed {staging_dir}")

    elapsed = time.time() - t0
    h, m = divmod(int(elapsed), 3600)
    m, s = divmod(m, 60)
    print(f"\n=== Done in {h}h {m}m {s}s ===")


if __name__ == "__main__":
    main()
