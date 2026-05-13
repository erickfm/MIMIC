"""Compute per-character empirical kill percent from fox_all_v2 shards.

Method: find frames where a player enters a DYING state (action IDs 0-10 per
slippistats's DYING_START..END range). The player's percent at the frame
BEFORE entering DYING is "the percent at which they died."

Aggregate across all death events per character. The mean per-char is our
"kill percent" — what % a player typically dies at in this dataset's
condition mix (stages, opponents, skill level).

Why empirical, not tier-list values:
  * tier-list kill percents depend on stage, DI, move quality
  * our dataset is ranked play with diverse conditions
  * empirical kill percents reflect the actual death distribution
    in the data, so the char-adjusted percent bucketing matches the
    distribution the model sees.

Usage:
    python -m value.char_kill_percents --out value/char_kill_percents.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from value.derived_features import DYING_START, DYING_END


# Percent normalization (from mimic_norm.json for fox_all_v2). The transform
# is "normalize" (min/max to [-1, +1]):
#   normalized = 2 * (raw - min) / (max - min) - 1
# Inverting:
#   raw = (normalized + 1) * (max - min) / 2 + min
PCT_MIN = 0.0
PCT_MAX = 343.4003601074219


def denorm_percent(normalized: np.ndarray) -> np.ndarray:
    """Invert the 'normalize' min/max transform applied to percent in shards."""
    return (normalized + 1.0) * (PCT_MAX - PCT_MIN) / 2.0 + PCT_MIN


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/fox_all_v2")
    p.add_argument("--max-shards", type=int, default=20,
                   help="Cap shards to scan (each is ~10s)")
    p.add_argument("--out", default="value/char_kill_percents.json")
    args = p.parse_args()

    with open(Path(args.data_dir) / "tensor_manifest.json") as f:
        manifest = json.load(f)
    train_shards = manifest["train_shards"][:args.max_shards]
    print(f"scanning {len(train_shards)} shards for death events...")

    # Collect death percents per character.
    # For each frame fi, if player just entered DYING state (action[fi] ∈ DYING
    # AND action[fi-1] not in DYING), record the percent at fi-1.
    per_char_self: dict = defaultdict(list)
    per_char_opp: dict = defaultdict(list)

    for shard_name in train_shards:
        shard = torch.load(Path(args.data_dir) / shard_name,
                           map_location="cpu", weights_only=False, mmap=True)
        states = shard["states"]
        offsets = shard["offsets"]
        n_games = int(shard["n_games"])

        self_action = states["self_action"].numpy()
        opp_action = states["opp_action"].numpy()
        self_pct = states["self_numeric"][:, 2].numpy()
        opp_pct = states["opp_numeric"][:, 2].numpy()
        self_char = states["self_character"].numpy()
        opp_char = states["opp_character"].numpy()

        for g in range(n_games):
            s = int(offsets[g].item())
            e = int(offsets[g + 1].item())
            # Self death events
            sa = self_action[s:e]
            sp_raw = denorm_percent(self_pct[s:e])
            sc = self_char[s]  # constant per game
            self_dying = (sa >= DYING_START) & (sa <= DYING_END)
            # First-frame transitions: dying[i]=True AND dying[i-1]=False
            if len(self_dying) >= 2:
                transitions = self_dying[1:] & ~self_dying[:-1]
                death_idxs = np.where(transitions)[0]  # index into [1:]
                for di in death_idxs:
                    # Percent at the frame BEFORE the death animation began
                    pct = float(sp_raw[di])  # = sp_raw[di+1 - 1]
                    if pct > 0 and pct < 999:  # sanity bounds
                        per_char_self[int(sc)].append(pct)

            # Opp death events
            oa = opp_action[s:e]
            op_raw = denorm_percent(opp_pct[s:e])
            oc = opp_char[s]
            opp_dying = (oa >= DYING_START) & (oa <= DYING_END)
            if len(opp_dying) >= 2:
                transitions = opp_dying[1:] & ~opp_dying[:-1]
                death_idxs = np.where(transitions)[0]
                for di in death_idxs:
                    pct = float(op_raw[di])
                    if pct > 0 and pct < 999:
                        per_char_opp[int(oc)].append(pct)

    # Pool self + opp observations per character.
    # Self is always Fox (or rarely Falco) so this only adds Fox samples.
    # Opp gives us diverse character coverage.
    combined: dict = defaultdict(list)
    for c, vals in per_char_self.items():
        combined[c].extend(vals)
    for c, vals in per_char_opp.items():
        combined[c].extend(vals)

    # Compute robust statistics per char.
    import sys
    sys.path.insert(0, "/home/erick/projects/MIMIC")
    import melee
    from mimic.cat_maps import CHARACTER_MAP
    inv_char = {v: k for k, v in CHARACTER_MAP.items()}

    print(f"\n{'char':<22} {'n_deaths':>9} {'mean':>7} {'median':>7} "
          f"{'p25':>7} {'p75':>7}")
    out_table = {}
    for cid in sorted(combined.keys(), key=lambda c: -len(combined[c])):
        vals = np.array(combined[cid])
        if len(vals) < 20:
            continue
        try:
            name = melee.enums.Character(inv_char.get(cid, cid)).name
        except Exception:
            name = f"char_{cid}"
        mean = float(vals.mean())
        median = float(np.median(vals))
        p25, p75 = float(np.percentile(vals, 25)), float(np.percentile(vals, 75))
        print(f"  {name:<22} {len(vals):>9} {mean:>7.1f} {median:>7.1f} "
              f"{p25:>7.1f} {p75:>7.1f}")
        out_table[cid] = {
            "name": name,
            "n_deaths": int(len(vals)),
            "kill_pct_mean": mean,
            "kill_pct_median": median,
            "kill_pct_p25": p25,
            "kill_pct_p75": p75,
        }

    Path(args.out).write_text(json.dumps(out_table, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
