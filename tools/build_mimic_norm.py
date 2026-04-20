#!/usr/bin/env python3
"""Generate mimic_norm.json from norm_stats.json and norm_minmax.json.

Usage:
    python tools/build_mimic_norm.py --norm-stats data/fox_meta/norm_stats.json \
        --minmax data/fox_meta/norm_minmax.json --out data/fox_meta/mimic_norm.json
"""
import argparse
import json

# Per-feature transform specs.
#
# Legacy group (bootstrapped from HAL's input_configs.py baseline() — see
# mimic/features.py mimic_normalize for formulas):
#   normalize        — bounded uniform-ish (percent, stock, jumps_left, binary flags)
#   standardize      — bounded bell-ish, linear semantics (positions)
#   invert_normalize — polarity-flipped bounded (shield, full→-1, empty→+1)
#
# Post-2026-04-20 additions for the MIMIC-extended features:
#   tanh_scale — heavy-tailed signed; tanh(x/scale) preserves sign, saturates extremes
#   linear_max — zero-inflated bounded nonneg; x/max (no clamp in fwd, engine bounds)
#   log_max    — heavy-tailed nonneg with meaningful magnitude; log1p(clamp(x,0,max))/log1p(max)
#
# Per-feature scale/max picks (justified inline):
MIMIC_TRANSFORMS = {
    # Legacy
    "percent":          {"transform": "normalize"},
    "stock":            {"transform": "normalize"},
    "facing":           {"transform": "normalize"},
    "invulnerable":     {"transform": "normalize"},
    "jumps_left":       {"transform": "normalize"},
    "on_ground":        {"transform": "normalize"},
    "shield_strength":  {"transform": "invert_normalize"},
    "pos_x":            {"transform": "standardize"},
    "pos_y":            {"transform": "standardize"},
    # Self-motion velocities: scale=5.0 covers ~2× Fox dash speed (2.5).
    # Walk (0.3) → tanh=0.06 (distinguishable from zero), run (2.5) → 0.46,
    # fast-fall (3) → 0.54. Extreme self-velocity (>8) saturates toward 1.
    "speed_air_x_self":     {"transform": "tanh_scale", "scale": 5.0},
    "speed_ground_x_self":  {"transform": "tanh_scale", "scale": 5.0},
    "speed_y_self":         {"transform": "tanh_scale", "scale": 5.0},
    # Knockback velocities: scale=10.0 preserves discrimination across the full
    # hit-severity range. Small knockback (5) → 0.46, medium (10) → 0.76,
    # big (15) → 0.91, kill (25) → 0.99. Scale=5 would saturate before kill-range.
    "speed_x_attack":       {"transform": "tanh_scale", "scale": 10.0},
    "speed_y_attack":       {"transform": "tanh_scale", "scale": 10.0},
    # Hitlag: Melee engine caps hitlag at ~20 frames. Linear is fine — exact
    # frame count matters less than "am I frozen" signal.
    "hitlag_left":          {"transform": "linear_max", "max": 20.0},
    # Hitstun: real values go up to ~100 frames on kill-level hits. Log
    # compresses the tail (differences 60 vs 80 vs 100 don't need fine
    # distinction — all "still combo'd") while keeping 0 vs 5 vs 15 crisp.
    "hitstun_left":         {"transform": "log_max",    "max": 120.0},
}

# Binary flags not in norm_stats (they're in the "flags" feature group).
BINARY_DEFAULTS = {
    "facing": {"transform": "normalize", "min": 0.0, "max": 1.0, "mean": 0.5, "std": 0.5},
    "invulnerable": {"transform": "normalize", "min": 0.0, "max": 1.0, "mean": 0.5, "std": 0.5},
    "on_ground": {"transform": "normalize", "min": 0.0, "max": 1.0, "mean": 0.5, "std": 0.5},
}


def main():
    parser = argparse.ArgumentParser(description="Generate mimic_norm.json")
    parser.add_argument("--norm-stats", type=str, required=True)
    parser.add_argument("--minmax", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    with open(args.norm_stats) as f:
        norm_stats = json.load(f)
    with open(args.minmax) as f:
        minmax = json.load(f)

    features = {}
    for suffix, spec in MIMIC_TRANSFORMS.items():
        transform = spec["transform"]

        # Binary flags — hardcoded defaults (not in norm_stats).
        if suffix in BINARY_DEFAULTS:
            features[suffix] = BINARY_DEFAULTS[suffix]
            continue

        # Transforms that need data-driven stats: standardize, normalize, invert_normalize.
        if transform in ("standardize", "normalize", "invert_normalize"):
            col = f"self_{suffix}"
            if col not in norm_stats:
                print(f"  WARNING: {col} not in norm_stats, skipping")
                continue
            mean, std = norm_stats[col]
            mn, mx = minmax.get(col, [mean - 4 * std, mean + 4 * std])
            features[suffix] = {
                "transform": transform,
                "min": mn, "max": mx,
                "mean": mean, "std": std,
            }

        # Transforms with fixed params (no stats needed): tanh_scale, linear_max, log_max.
        elif transform == "tanh_scale":
            features[suffix] = {"transform": transform, "scale": spec["scale"]}
        elif transform in ("linear_max", "log_max"):
            features[suffix] = {"transform": transform, "max": spec["max"]}
        else:
            print(f"  WARNING: unknown transform {transform!r} for {suffix}")

    result = {"features": features}
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved mimic_norm.json with {len(features)} features to {args.out}")
    for k, v in features.items():
        t = v["transform"]
        if t in ("standardize", "normalize", "invert_normalize"):
            print(f"  {k}: {t} min={v['min']:.3f} max={v['max']:.3f} mean={v['mean']:.3f} std={v['std']:.3f}")
        elif t == "tanh_scale":
            print(f"  {k}: {t} scale={v['scale']}")
        elif t in ("linear_max", "log_max"):
            print(f"  {k}: {t} max={v['max']}")


if __name__ == "__main__":
    main()
