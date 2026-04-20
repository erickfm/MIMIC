#!/usr/bin/env python3
"""Compute norm_stats.json and cat_maps.json from .slp replay files.

Streams through .slp files in two passes:
  Pass 1: Collect unique values for dynamic categoricals (ports, costumes,
           projectile subtypes/owners), accumulate running sum/sum-of-squares/count
           for numeric columns.
  Pass 2: Finalize mean/std (std floor = 1.0) and dense categorical mappings.

Usage:
    python tools/build_norm_stats.py --slp-dir /data/slp --out-dir data/full
    python tools/build_norm_stats.py --slp-dir /data/slp --out-dir data/full --n-files 5000
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import math
import os
import multiprocessing as mp
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
from melee import Console, stages
from melee.enums import Button, Menu

import mimic.features as F
from mimic.features import BTN, STAGE_GEOM_COLS

BTN_ENUMS = [Button[name] for name in BTN]


def _extract_frame_values(gs, players):
    """Extract raw numeric and categorical values from one frame for both perspectives."""
    port1, ps1 = players[0]
    port2, ps2 = players[1]

    stage = gs.stage
    if stage and stage in stages.BLASTZONES:
        stage_static = {
            "blastzone_left": stages.BLASTZONES[stage][0],
            "blastzone_right": stages.BLASTZONES[stage][1],
            "blastzone_top": stages.BLASTZONES[stage][2],
            "blastzone_bottom": stages.BLASTZONES[stage][3],
            "stage_edge_left": -stages.EDGE_POSITION[stage],
            "stage_edge_right": stages.EDGE_POSITION[stage],
        }
        for name, func in [("left_platform", stages.left_platform_position),
                            ("right_platform", stages.right_platform_position)]:
            h, l, r = func(stage)
            stage_static[f"{name}_height"] = h
            stage_static[f"{name}_left"] = l
            stage_static[f"{name}_right"] = r
        tp = stages.top_platform_position(stage)
        h, l, r = tp if tp[0] is not None else (float("nan"),) * 3
        stage_static["top_platform_height"] = h
        stage_static["top_platform_left"] = l
        stage_static["top_platform_right"] = r
    else:
        stage_static = {k: float("nan") for k in STAGE_GEOM_COLS[:15]}

    if stage and stage.name == "YOSHIS_STORY":
        r0, r1, r2 = stages.randall_position(gs.frame)
    else:
        r0 = r1 = r2 = float("nan")
    stage_static["randall_height"] = r0
    stage_static["randall_left"] = r1
    stage_static["randall_right"] = r2

    # Build values for both perspectives
    rows = []
    for self_ps, self_port, opp_ps, opp_port in [
        (ps1, port1, ps2, port2), (ps2, port2, ps1, port1)
    ]:
        row = {"frame": gs.frame, "stage": stage.value if stage else 0}
        row.update(stage_static)

        # Distance
        row["distance"] = math.hypot(
            float(self_ps.position.x) - float(opp_ps.position.x),
            float(self_ps.position.y) - float(opp_ps.position.y),
        )

        for prefix, ps, port in [("self", self_ps, self_port),
                                   ("opp", opp_ps, opp_port)]:
            row[f"{prefix}_port"] = port
            row[f"{prefix}_character"] = ps.character.value
            row[f"{prefix}_action"] = ps.action.value
            row[f"{prefix}_costume"] = ps.costume
            row[f"{prefix}_action_frame"] = ps.action_frame
            row[f"{prefix}_pos_x"] = float(ps.position.x)
            row[f"{prefix}_pos_y"] = float(ps.position.y)
            row[f"{prefix}_percent"] = float(ps.percent)
            row[f"{prefix}_stock"] = ps.stock
            row[f"{prefix}_jumps_left"] = ps.jumps_left
            row[f"{prefix}_shield_strength"] = float(ps.shield_strength)
            row[f"{prefix}_speed_air_x_self"] = float(ps.speed_air_x_self)
            row[f"{prefix}_speed_ground_x_self"] = float(ps.speed_ground_x_self)
            row[f"{prefix}_speed_x_attack"] = float(ps.speed_x_attack)
            row[f"{prefix}_speed_y_attack"] = float(ps.speed_y_attack)
            row[f"{prefix}_speed_y_self"] = float(ps.speed_y_self)
            row[f"{prefix}_hitlag_left"] = ps.hitlag_left
            row[f"{prefix}_hitstun_left"] = ps.hitstun_frames_left
            row[f"{prefix}_invuln_left"] = ps.invulnerability_left
            for part in ("bottom", "left", "right", "top"):
                ecb = getattr(ps, f"ecb_{part}")
                row[f"{prefix}_ecb_{part}_x"] = float(ecb[0])
                row[f"{prefix}_ecb_{part}_y"] = float(ecb[1])
            row[f"{prefix}_on_ground"] = float(ps.on_ground)
            row[f"{prefix}_off_stage"] = float(ps.off_stage)
            row[f"{prefix}_facing"] = float(ps.facing)
            row[f"{prefix}_invulnerable"] = float(ps.invulnerable)
            row[f"{prefix}_moonwalkwarning"] = float(ps.moonwalkwarning)
            row[f"{prefix}_main_x"] = ps.controller_state.main_stick[0]
            row[f"{prefix}_main_y"] = ps.controller_state.main_stick[1]
            row[f"{prefix}_l_shldr"] = ps.controller_state.l_shoulder
            row[f"{prefix}_r_shldr"] = ps.controller_state.r_shoulder
            for btn_enum in BTN_ENUMS:
                row[f"{prefix}_btn_{btn_enum.name}"] = float(
                    ps.controller_state.button.get(btn_enum, False))

            # Nana
            nana = ps.nana
            np_ = f"{prefix}_nana"
            if nana:
                row[f"{np_}_character"] = nana.character.value
                row[f"{np_}_action"] = nana.action.value
                row[f"{np_}_action_frame"] = nana.action_frame
                row[f"{np_}_pos_x"] = float(nana.position.x)
                row[f"{np_}_pos_y"] = float(nana.position.y)
                row[f"{np_}_percent"] = float(nana.percent)
                row[f"{np_}_stock"] = nana.stock
                row[f"{np_}_jumps_left"] = nana.jumps_left
                row[f"{np_}_shield_strength"] = float(nana.shield_strength)
                row[f"{np_}_speed_air_x_self"] = float(nana.speed_air_x_self)
                row[f"{np_}_speed_ground_x_self"] = float(nana.speed_ground_x_self)
                row[f"{np_}_speed_x_attack"] = float(nana.speed_x_attack)
                row[f"{np_}_speed_y_attack"] = float(nana.speed_y_attack)
                row[f"{np_}_speed_y_self"] = float(nana.speed_y_self)
                row[f"{np_}_hitlag_left"] = nana.hitlag_left
                row[f"{np_}_hitstun_left"] = nana.hitstun_frames_left
                row[f"{np_}_invuln_left"] = nana.invulnerability_left
                for part in ("bottom", "left", "right", "top"):
                    ecb = getattr(nana, f"ecb_{part}")
                    row[f"{np_}_ecb_{part}_x"] = float(ecb[0])
                    row[f"{np_}_ecb_{part}_y"] = float(ecb[1])
                row[f"{np_}_main_x"] = nana.controller_state.main_stick[0]
                row[f"{np_}_main_y"] = nana.controller_state.main_stick[1]
                row[f"{np_}_l_shldr"] = nana.controller_state.l_shoulder
                row[f"{np_}_r_shldr"] = nana.controller_state.r_shoulder

        # Projectiles
        for j in range(8):
            pp = f"proj{j}"
            if j < len(gs.projectiles):
                proj = gs.projectiles[j]
                row[f"{pp}_owner"] = proj.owner
                row[f"{pp}_type"] = proj.type.value
                row[f"{pp}_subtype"] = proj.subtype
            else:
                row[f"{pp}_owner"] = -1
                row[f"{pp}_type"] = -1
                row[f"{pp}_subtype"] = -1

        rows.append(row)

    return rows


def _process_one_file(args):
    """Worker: parse a single .slp, return partial accumulators.

    Returns (col_sum, col_sq, col_n, col_min, col_max, cat_unique,
             success: bool).
    Must be module-level + picklable for mp.Pool.
    """
    slp_path, norm_cols, dynamic_cat_cols = args
    col_sum = {c: 0.0 for c in norm_cols}
    col_sq  = {c: 0.0 for c in norm_cols}
    col_n   = {c: 0   for c in norm_cols}
    col_min = {c: float("inf")  for c in norm_cols}
    col_max = {c: float("-inf") for c in norm_cols}
    cat_unique = {c: set() for c in dynamic_cat_cols}

    try:
        console = Console(is_dolphin=False, path=str(slp_path),
                          allow_old_version=True)
        console.connect()
    except Exception:
        return (col_sum, col_sq, col_n, col_min, col_max, cat_unique, False)

    while True:
        gs = console.step()
        if gs is None:
            break
        if gs.menu_state != Menu.IN_GAME or gs.frame < 0:
            continue
        players = list(gs.players.items())
        if len(players) < 2:
            continue

        for row in _extract_frame_values(gs, players):
            for c in norm_cols:
                v = row.get(c)
                if v is not None and math.isfinite(v):
                    col_sum[c] += v
                    col_sq[c]  += v * v
                    col_n[c]   += 1
                    if v < col_min[c]:
                        col_min[c] = v
                    if v > col_max[c]:
                        col_max[c] = v
            for c in dynamic_cat_cols:
                v = row.get(c)
                if v is not None:
                    cat_unique[c].add(int(v))

    return (col_sum, col_sq, col_n, col_min, col_max, cat_unique, True)


def compute_stats(
    slp_dir: Path, n_files: int = 5000, seed: int = 42, n_workers: int = 0,
) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, Dict[int, int]]]:
    """Parallel streaming computation of norm stats and dynamic cat maps.

    n_workers=0 → auto, uses min(cpu_count(), 64). Single-threaded at
    n_workers=1 (falls back to serial path for debugging).
    """
    files = sorted(f for f in slp_dir.iterdir() if f.suffix.lower() == ".slp")
    if not files:
        raise RuntimeError(f"No .slp files in {slp_dir}")

    rng = random.Random(seed)
    sample = rng.sample(files, min(n_files, len(files)))

    # Get the columns we need to track
    fg = F.build_feature_groups()
    norm_cols = F.get_norm_cols(fg)
    cat_cols = F.get_categorical_cols(fg)
    dynamic_cat_cols = F._cols_needing_dynamic_map(cat_cols)

    # Final accumulators (reduced from workers)
    col_sum: Dict[str, float] = {c: 0.0 for c in norm_cols}
    col_sq:  Dict[str, float] = {c: 0.0 for c in norm_cols}
    col_n:   Dict[str, int]   = {c: 0 for c in norm_cols}
    col_min: Dict[str, float] = {c: float("inf") for c in norm_cols}
    col_max: Dict[str, float] = {c: float("-inf") for c in norm_cols}
    cat_unique: Dict[str, Set[int]] = {c: set() for c in dynamic_cat_cols}

    if n_workers <= 0:
        n_workers = min(os.cpu_count() or 4, 64)
    print(f"  [parallel] {n_workers} workers on {len(sample)} .slp files",
          flush=True)

    work = [(p, norm_cols, dynamic_cat_cols) for p in sample]
    n_processed = 0
    n_done = 0

    def _reduce(result):
        nonlocal n_processed
        psum, psq, pn, pmin, pmax, pcat, ok = result
        if ok:
            n_processed += 1
        for c in norm_cols:
            col_sum[c] += psum[c]
            col_sq[c]  += psq[c]
            col_n[c]   += pn[c]
            if pmin[c] < col_min[c]:
                col_min[c] = pmin[c]
            if pmax[c] > col_max[c]:
                col_max[c] = pmax[c]
        for c in dynamic_cat_cols:
            cat_unique[c].update(pcat[c])

    if n_workers == 1:
        for w in work:
            _reduce(_process_one_file(w))
            n_done += 1
            if n_done % 100 == 0:
                print(f"  [{n_done}/{len(sample)}] processed ...", flush=True)
    else:
        # chunksize = ~5 balances IPC vs load-balancing for ~0.3s/file parse
        with mp.Pool(n_workers) as pool:
            for result in pool.imap_unordered(_process_one_file, work, chunksize=5):
                _reduce(result)
                n_done += 1
                if n_done % 200 == 0:
                    print(f"  [{n_done}/{len(sample)}] processed ...", flush=True)

    print(f"  Processed {n_processed}/{len(sample)} .slp files successfully")

    # Finalize norm stats
    norm_stats = {}
    for c in norm_cols:
        n = col_n[c]
        if n > 0:
            mean = col_sum[c] / n
            var = (col_sq[c] / n) - mean * mean
            std = max(math.sqrt(max(var, 0.0)), 1.0)  # floor at 1.0
            norm_stats[c] = [float(mean), float(std)]
        else:
            norm_stats[c] = [0.0, 1.0]

    # Min/max stats (separate from norm_stats for compatibility)
    minmax = {}
    for c in norm_cols:
        n = col_n[c]
        if n > 0:
            minmax[c] = [float(col_min[c]), float(col_max[c])]
        else:
            minmax[c] = [0.0, 0.0]

    # Finalize dynamic cat maps
    cat_maps = F._finalize_dynamic_maps(cat_unique)

    return norm_stats, cat_maps, minmax


def main():
    parser = argparse.ArgumentParser(
        description="Compute norm_stats.json and cat_maps.json from .slp files")
    parser.add_argument("--slp-dir", type=str, required=True,
                        help="Directory with .slp replay files")
    parser.add_argument("--out-dir", type=str, required=True,
                        help="Output directory for JSON files")
    parser.add_argument("--n-files", type=int, default=5000,
                        help="Number of .slp files to sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0,
                        help="Parallel workers (0=auto, uses min(cpu_count, 64))")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Computing Normalization Stats & Categorical Maps ===")
    norm_stats, cat_maps, minmax = compute_stats(
        Path(args.slp_dir), n_files=args.n_files, seed=args.seed,
        n_workers=args.workers)

    ns_path = out_dir / "norm_stats.json"
    with open(ns_path, "w") as f:
        json.dump(norm_stats, f, indent=2)
    print(f"\nSaved {len(norm_stats)} norm stats to {ns_path}")

    mm_path = out_dir / "norm_minmax.json"
    with open(mm_path, "w") as f:
        json.dump(minmax, f, indent=2)
    print(f"Saved {len(minmax)} min/max stats to {mm_path}")

    cm_path = out_dir / "cat_maps.json"
    with open(cm_path, "w") as f:
        json.dump(cat_maps, f, indent=2)
    print(f"Saved {len(cat_maps)} cat maps to {cm_path}")


if __name__ == "__main__":
    main()
