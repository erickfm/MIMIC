#!/usr/bin/env python3
"""Generate hal_norm.json from norm_stats.json and norm_minmax.json.

Usage:
    python tools/build_hal_norm.py --norm-stats data/fox_meta/norm_stats.json \
        --minmax data/fox_meta/norm_minmax.json --out data/fox_meta/hal_norm.json
"""
import argparse
import json

# HAL's transform type per feature (from hal/preprocess/input_configs.py baseline())
HAL_TRANSFORMS = {
    "percent": "normalize",
    "stock": "normalize",
    "facing": "normalize",
    "invulnerable": "normalize",
    "jumps_left": "normalize",
    "on_ground": "normalize",
    "shield_strength": "invert_normalize",
    "pos_x": "standardize",
    "pos_y": "standardize",
}

# Binary flags not in norm_stats (they're in the "flags" feature group)
BINARY_DEFAULTS = {
    "facing": {"transform": "normalize", "min": 0.0, "max": 1.0, "mean": 0.5, "std": 0.5},
    "invulnerable": {"transform": "normalize", "min": 0.0, "max": 1.0, "mean": 0.5, "std": 0.5},
    "on_ground": {"transform": "normalize", "min": 0.0, "max": 1.0, "mean": 0.5, "std": 0.5},
}


def main():
    parser = argparse.ArgumentParser(description="Generate hal_norm.json")
    parser.add_argument("--norm-stats", type=str, required=True)
    parser.add_argument("--minmax", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    with open(args.norm_stats) as f:
        norm_stats = json.load(f)
    with open(args.minmax) as f:
        minmax = json.load(f)

    features = {}
    for suffix, transform in HAL_TRANSFORMS.items():
        if suffix in BINARY_DEFAULTS:
            features[suffix] = BINARY_DEFAULTS[suffix]
            continue

        col = f"self_{suffix}"
        if col not in norm_stats:
            print(f"  WARNING: {col} not in norm_stats, skipping")
            continue

        mean, std = norm_stats[col]
        mn, mx = minmax.get(col, [mean - 4 * std, mean + 4 * std])

        features[suffix] = {
            "transform": transform,
            "min": mn,
            "max": mx,
            "mean": mean,
            "std": std,
        }

    result = {"features": features}
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved hal_norm.json with {len(features)} features to {args.out}")
    for k, v in features.items():
        print(f"  {k}: {v['transform']} min={v['min']:.3f} max={v['max']:.3f} mean={v['mean']:.3f} std={v['std']:.3f}")


if __name__ == "__main__":
    main()
