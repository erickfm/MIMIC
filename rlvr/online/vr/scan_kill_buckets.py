"""Step 8a — scan per-character kill-percent buckets for low_percent_kill.

Streams `data/fox_all_v2` shards, records percent-at-death per character
(stock decrement → peak opp percent in the 12 frames before), and writes
the p15 of each character's distribution to `kill_percent_buckets.json`,
which `low_percent_kill.py` loads to override its hardcoded defaults.

Run more shards than the original 40-shard scan so Luigi and the rest of
the unmeasured cast clear the MIN_DEATHS threshold.

Usage:
    python -m rlvr.online.vr.scan_kill_buckets --shards 120
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch

# percent `normalize` transform max — hf_checkpoints/fox/mimic_norm.json.
PMAX = 343.4003601074219
MIN_DEATHS = 150        # characters with fewer deaths are left to the fallback
PERCENTILE = 15         # p15 — see docs/vr-proposals/low-percent-kill.md


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/fox_all_v2")
    ap.add_argument("--shards", type=int, default=120)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(__file__), "kill_percent_buckets.json"))
    args = ap.parse_args()

    all_shards = sorted(glob.glob(
        os.path.join(args.data_dir, "train_shard_*.pt")))
    if not all_shards:
        raise SystemExit(f"no shards in {args.data_dir}")
    step = max(1, len(all_shards) // args.shards)
    sel = all_shards[::step][:args.shards]
    print(f"{len(all_shards)} shards total; scanning {len(sel)}")

    deaths: dict[int, list[float]] = {}
    for n, f in enumerate(sel):
        s = torch.load(f, map_location="cpu", weights_only=False)
        st = s["states"]
        off = s["offsets"].tolist()
        sn = st["self_numeric"].numpy()      # self is always Fox (char 1)
        on = st["opp_numeric"].numpy()
        oc = st["opp_character"].numpy()
        for g in range(len(off) - 1):
            a, b = off[g], off[g + 1]
            if b - a < 2:
                continue
            for num, ch in ((sn, 1), (on, int(oc[a]))):
                stock = np.rint((num[a:b, 3] + 1.0) * 2.0).astype(int)
                pct = (num[a:b, 2] + 1.0) * (PMAX / 2.0)
                for i in np.where(stock[1:] < stock[:-1])[0] + 1:
                    lo = max(0, i - 12)
                    kp = float(pct[lo:i].max()) if i > lo else float(pct[i])
                    deaths.setdefault(ch, []).append(kp)
        del s
        if (n + 1) % 20 == 0:
            print(f"  {n + 1}/{len(sel)} shards")

    buckets: dict[str, float] = {}
    counts: dict[str, int] = {}
    for ch, arr in deaths.items():
        nz = np.array([d for d in arr if d >= 1.0])
        if len(nz) >= MIN_DEATHS:
            buckets[str(ch)] = round(float(np.percentile(nz, PERCENTILE)), 1)
            counts[str(ch)] = len(nz)

    with open(args.out, "w") as fh:
        json.dump(buckets, fh, indent=2, sort_keys=True)
    print(f"\nwrote {len(buckets)} character buckets -> {args.out}")
    for ch in sorted(buckets, key=int):
        print(f"  char {ch:>2}: <{buckets[ch]:.0f}%  (n={counts[ch]})")


if __name__ == "__main__":
    main()
